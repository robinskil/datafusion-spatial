//! Measurement functions.
//!
//! # Planar and spherical are different questions
//!
//! `ST_Distance`, `ST_Area` and `ST_Length` are planar. They treat coordinates as Cartesian, so on
//! longitude and latitude data they return degrees, not metres. That is what PostGIS does for the
//! `geometry` type.
//!
//! `ST_DistanceSphere` and `ST_DistanceSpheroid` are the spherical answers, in metres, backed by
//! [`Haversine`] and [`Geodesic`].

use arrow_array::builder::Float64Builder;
use arrow_array::{Array, Float64Array};
use arrow_buffer::NullBuffer;
use geo::line_measures::LengthMeasurable;
use geo::{Area, Distance, Euclidean, Geodesic, Geometry, HausdorffDistance, Haversine, Point};
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::GeoArrowResult;

use crate::materialize::{all_null, GeometryReader};
use crate::predicate::{broadcast_len, broadcast_nulls, Operand};

/// A measurement over one geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryMeasure {
    /// `ST_Area`. Always positive.
    Area,
    /// `ST_Length`. The length of lineal parts. Zero for a point or a polygon, as in PostGIS.
    Length,
    /// `ST_Perimeter`. The boundary length of areal parts. Zero for a point or a line.
    Perimeter,
}

impl UnaryMeasure {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::Area => "ST_Area",
            Self::Length => "ST_Length",
            Self::Perimeter => "ST_Perimeter",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::Area => "st_area",
            Self::Length => "st_length",
            Self::Perimeter => "st_perimeter",
        }
    }

    /// Every unary measure, for registration.
    pub const ALL: [Self; 3] = [Self::Area, Self::Length, Self::Perimeter];
}

/// `ST_Area`.
pub fn st_area(array: &dyn GeoArrowArray) -> GeoArrowResult<Float64Array> {
    unary_measure(array, UnaryMeasure::Area)
}

/// `ST_Length`.
pub fn st_length(array: &dyn GeoArrowArray) -> GeoArrowResult<Float64Array> {
    unary_measure(array, UnaryMeasure::Length)
}

/// `ST_Perimeter`.
pub fn st_perimeter(array: &dyn GeoArrowArray) -> GeoArrowResult<Float64Array> {
    unary_measure(array, UnaryMeasure::Perimeter)
}

/// Any measurement over one geometry.
pub fn unary_measure(
    array: &dyn GeoArrowArray,
    measure: UnaryMeasure,
) -> GeoArrowResult<Float64Array> {
    // A point or a multi point has no area, no length and no perimeter. Answer from the schema
    // and never touch a coordinate.
    if let Some(zero) = always_zero(array, measure) {
        return Ok(zero);
    }
    if all_null(array) {
        return Ok(Float64Array::new_null(array.len()));
    }

    // One reader for the batch. It matches the coordinate buffer once and reuses one geometry.
    let mut reader = GeometryReader::new(array)?;
    let mut builder = Float64Builder::with_capacity(array.len());
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => builder.append_value(match measure {
                UnaryMeasure::Area => geom.unsigned_area(),
                UnaryMeasure::Length => planar_length(geom),
                UnaryMeasure::Perimeter => planar_perimeter(geom),
            }),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// Constant-fold the cases where the type alone settles the answer.
fn always_zero(array: &dyn GeoArrowArray, measure: UnaryMeasure) -> Option<Float64Array> {
    use geoarrow_schema::GeoArrowType::*;
    let zero = matches!(
        (measure, array.data_type()),
        (
            UnaryMeasure::Area,
            Point(_) | MultiPoint(_) | LineString(_) | MultiLineString(_)
        ) | (
            UnaryMeasure::Length,
            Point(_) | MultiPoint(_) | Polygon(_) | MultiPolygon(_)
        ) | (
            UnaryMeasure::Perimeter,
            Point(_) | MultiPoint(_) | LineString(_) | MultiLineString(_)
        )
    );
    zero.then(|| Float64Array::new(vec![0.0f64; array.len()].into(), array.logical_nulls()))
}

/// The length of the lineal parts of a geometry.
///
/// PostGIS returns zero for a polygon here. `ST_Perimeter` is the areal version.
fn planar_length(geom: &Geometry<f64>) -> f64 {
    match geom {
        Geometry::Line(line) => line.length(&Euclidean),
        Geometry::LineString(line) => line.length(&Euclidean),
        Geometry::MultiLineString(lines) => lines.iter().map(|l| l.length(&Euclidean)).sum(),
        Geometry::GeometryCollection(parts) => parts.iter().map(planar_length).sum(),
        _ => 0.0,
    }
}

/// The boundary length of the areal parts of a geometry.
fn planar_perimeter(geom: &Geometry<f64>) -> f64 {
    match geom {
        Geometry::Polygon(polygon) => polygon_perimeter(polygon),
        Geometry::MultiPolygon(polygons) => polygons.iter().map(polygon_perimeter).sum(),
        Geometry::Rect(rect) => polygon_perimeter(&rect.to_polygon()),
        Geometry::Triangle(triangle) => polygon_perimeter(&triangle.to_polygon()),
        Geometry::GeometryCollection(parts) => parts.iter().map(planar_perimeter).sum(),
        _ => 0.0,
    }
}

fn polygon_perimeter(polygon: &geo::Polygon<f64>) -> f64 {
    polygon.exterior().length(&Euclidean)
        + polygon
            .interiors()
            .iter()
            .map(|ring| ring.length(&Euclidean))
            .sum::<f64>()
}

/// A measurement between two geometries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryMeasure {
    /// `ST_Distance`. Planar shortest distance.
    Distance,
    /// `ST_MaxDistance`. Planar largest distance between any two points.
    MaxDistance,
    /// `ST_HausdorffDistance`.
    Hausdorff,
    /// `ST_FrechetDistance`. Line strings only.
    Frechet,
    /// `ST_DistanceSphere`. Metres on a sphere.
    Sphere,
    /// `ST_DistanceSpheroid`. Metres on the WGS 84 ellipsoid.
    Spheroid,
    /// `ST_Azimuth`. The geodesic bearing from one point to another, in radians.
    Azimuth,
}

impl BinaryMeasure {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::Distance => "ST_Distance",
            Self::MaxDistance => "ST_MaxDistance",
            Self::Hausdorff => "ST_HausdorffDistance",
            Self::Frechet => "ST_FrechetDistance",
            Self::Sphere => "ST_DistanceSphere",
            Self::Spheroid => "ST_DistanceSpheroid",
            Self::Azimuth => "ST_Azimuth",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::Distance => "st_distance",
            Self::MaxDistance => "st_maxdistance",
            Self::Hausdorff => "st_hausdorffdistance",
            Self::Frechet => "st_frechetdistance",
            Self::Sphere => "st_distancesphere",
            Self::Spheroid => "st_distancespheroid",
            Self::Azimuth => "st_azimuth",
        }
    }

    /// Every binary measure, for registration.
    pub const ALL: [Self; 7] = [
        Self::Distance,
        Self::MaxDistance,
        Self::Hausdorff,
        Self::Frechet,
        Self::Sphere,
        Self::Spheroid,
        Self::Azimuth,
    ];

    /// Compute this measurement for one pair.
    ///
    /// Returns `None` where PostGIS returns NULL, which is the case for `ST_FrechetDistance`
    /// against anything that is not a line string.
    #[inline]
    pub fn evaluate(self, left: &Geometry<f64>, right: &Geometry<f64>) -> Option<f64> {
        match self {
            Self::Distance => Some(Euclidean.distance(left, right)),
            Self::MaxDistance => Some(max_distance(left, right)),
            Self::Hausdorff => Some(left.hausdorff_distance(right)),
            Self::Frechet => match (left, right) {
                (Geometry::LineString(a), Geometry::LineString(b)) => Some(
                    geo::line_measures::FrechetDistance::frechet_distance(&Euclidean, a, b),
                ),
                _ => None,
            },
            Self::Sphere => {
                representative_points(left, right).map(|(a, b)| Haversine.distance(a, b))
            }
            Self::Spheroid => {
                representative_points(left, right).map(|(a, b)| Geodesic.distance(a, b))
            }
            // PostGIS reports the azimuth in radians clockwise from north; `geo` gives degrees.
            // Two identical points have no direction, which PostGIS reports as NULL.
            Self::Azimuth => representative_points(left, right).and_then(|(a, b)| {
                (a != b).then(|| {
                    let degrees = geo::Bearing::bearing(&Geodesic, a, b);
                    degrees.to_radians().rem_euclid(std::f64::consts::TAU)
                })
            }),
        }
    }
}

/// The spherical distances in `geo` are defined between points.
///
/// PostGIS accepts any geometry and measures between the nearest points. A full answer needs a
/// spherical nearest-point search. `geo` does not provide one. Two points give an exact answer.
/// Every other type returns NULL, not a wrong number.
fn representative_points(
    left: &Geometry<f64>,
    right: &Geometry<f64>,
) -> Option<(Point<f64>, Point<f64>)> {
    match (left, right) {
        (Geometry::Point(a), Geometry::Point(b)) => Some((*a, *b)),
        _ => None,
    }
}

/// The largest distance between any point of one geometry and any point of the other.
///
/// A pair of vertices always holds the maximum. The distance function is convex, and a convex
/// function over a polytope has its maximum at a vertex. So a walk over the vertex pairs
/// is exact.
///
/// The cost is the product of the vertex counts. A convex hull pass first would cut that down, and
/// is the obvious next optimization if this ever shows up in a profile.
pub fn max_distance(left: &Geometry<f64>, right: &Geometry<f64>) -> f64 {
    use geo::CoordsIter;

    let mut best = 0.0f64;
    let mut any = false;
    for a in left.coords_iter() {
        for b in right.coords_iter() {
            any = true;
            let (dx, dy) = (a.x - b.x, a.y - b.y);
            let distance = dx.hypot(dy);
            if distance > best {
                best = distance;
            }
        }
    }
    if any {
        best
    } else {
        f64::NAN
    }
}

/// Any measurement between two arrays of the same length.
pub fn binary_measure(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    measure: BinaryMeasure,
) -> GeoArrowResult<Float64Array> {
    let len = broadcast_len(measure.function_name(), left, right)?;
    let nulls = broadcast_nulls(left, right, len);
    let mut left_geom = Operand::new(left, len)?;
    let mut right_geom = Operand::new(right, len)?;

    let mut builder = Float64Builder::with_capacity(len);
    for index in 0..len {
        match (left_geom.get(index)?, right_geom.get(index)?) {
            (Some(lhs), Some(rhs)) => builder.append_option(measure.evaluate(lhs, rhs)),
            _ => builder.append_null(),
        }
    }

    let values = builder.finish();
    // Merge the input nulls back in, since a row may be null on either side.
    Ok(Float64Array::new(
        values.values().clone(),
        NullBuffer::union(values.nulls(), nulls.as_ref()),
    ))
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use geoarrow_array::builder::{
        GeometryBuilder, LineStringBuilder, PointBuilder, PolygonBuilder,
    };
    use geoarrow_schema::{Dimension, GeometryType, LineStringType, PointType, PolygonType};

    use super::*;

    fn shapes() -> geoarrow_array::array::GeometryArray {
        let mut builder = GeometryBuilder::new(GeometryType::new(Default::default()));
        for geom in [
            Geometry::<f64>::from(geo::wkt! { POINT(0.0 0.0) }),
            geo::wkt! { LINESTRING(0.0 0.0,3.0 4.0) }.into(),
            geo::wkt! { POLYGON((0.0 0.0,4.0 0.0,4.0 3.0,0.0 3.0,0.0 0.0)) }.into(),
        ] {
            builder.push_geometry(Some(&geom)).unwrap();
        }
        builder.finish()
    }

    #[test]
    fn area_length_and_perimeter() {
        let array = shapes();

        let area = st_area(&array).unwrap();
        assert_eq!(area.value(0), 0.0, "a point has no area");
        assert_eq!(area.value(1), 0.0, "nor does a line");
        assert_eq!(area.value(2), 12.0, "4 by 3");

        let length = st_length(&array).unwrap();
        assert_eq!(length.value(0), 0.0);
        assert_eq!(length.value(1), 5.0, "the 3-4-5 triangle");
        assert_eq!(length.value(2), 0.0, "a polygon has no length in PostGIS");

        let perimeter = st_perimeter(&array).unwrap();
        assert_eq!(perimeter.value(1), 0.0, "a line has no perimeter");
        assert_eq!(perimeter.value(2), 14.0, "2 times 4 plus 3");
    }

    /// A typed point column answers area from the schema, with no coordinate read.
    #[test]
    fn area_of_a_point_column_is_constant_folded() {
        let array = PointBuilder::from_nullable_points(
            [Some(&geo::point!(x: 1.0, y: 2.0)), None].into_iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let area = st_area(&array).unwrap();
        assert_eq!(area.value(0), 0.0);
        assert!(area.is_null(1), "the null row stays null");
    }

    #[test]
    fn distance_between_points() {
        let a = PointBuilder::from_points(
            [geo::point!(x: 0.0, y: 0.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let b = PointBuilder::from_points(
            [geo::point!(x: 3.0, y: 4.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let distance = binary_measure(&a, &b, BinaryMeasure::Distance).unwrap();
        assert_eq!(distance.value(0), 5.0);

        let max = binary_measure(&a, &b, BinaryMeasure::MaxDistance).unwrap();
        assert_eq!(max.value(0), 5.0, "one vertex each, so both agree");
    }

    #[test]
    fn max_distance_finds_the_far_corners() {
        let square: Vec<geo::Polygon<f64>> =
            vec![geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 1.0,0.0 0.0)) }];
        let a = PolygonBuilder::from_polygons(
            &square,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let point = PointBuilder::from_points(
            [geo::point!(x: 0.0, y: 0.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let max = binary_measure(&a, &point, BinaryMeasure::MaxDistance).unwrap();
        // The far corner of the unit square is the diagonal away.
        assert!((max.value(0) - 2.0f64.sqrt()).abs() < 1e-12);

        let min = binary_measure(&a, &point, BinaryMeasure::Distance).unwrap();
        assert_eq!(min.value(0), 0.0, "the point sits on the corner");
    }

    #[test]
    fn hausdorff_and_frechet() {
        let lines: Vec<geo::LineString<f64>> = vec![
            geo::wkt! { LINESTRING(0.0 0.0,1.0 0.0,2.0 0.0) },
            geo::wkt! { LINESTRING(0.0 1.0,1.0 1.0,2.0 1.0) },
        ];
        let a = LineStringBuilder::from_line_strings(
            &lines[..1],
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let b = LineStringBuilder::from_line_strings(
            &lines[1..],
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let hausdorff = binary_measure(&a, &b, BinaryMeasure::Hausdorff).unwrap();
        assert_eq!(hausdorff.value(0), 1.0, "the lines run one unit apart");

        let frechet = binary_measure(&a, &b, BinaryMeasure::Frechet).unwrap();
        assert_eq!(frechet.value(0), 1.0);
    }

    /// Frechet distance is defined for line strings, so anything else is null.
    #[test]
    fn frechet_of_a_point_is_null() {
        let points = PointBuilder::from_points(
            [geo::point!(x: 0.0, y: 0.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let frechet = binary_measure(&points, &points, BinaryMeasure::Frechet).unwrap();
        assert!(frechet.is_null(0));
    }

    /// London to Paris is roughly 344 kilometres.
    #[test]
    fn spherical_distances_are_in_metres() {
        let london = PointBuilder::from_points(
            [geo::point!(x: -0.1278, y: 51.5074)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let paris = PointBuilder::from_points(
            [geo::point!(x: 2.3522, y: 48.8566)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let sphere = binary_measure(&london, &paris, BinaryMeasure::Sphere).unwrap();
        let spheroid = binary_measure(&london, &paris, BinaryMeasure::Spheroid).unwrap();

        assert!(
            (330_000.0..350_000.0).contains(&sphere.value(0)),
            "sphere gave {}",
            sphere.value(0)
        );
        assert!(
            (330_000.0..350_000.0).contains(&spheroid.value(0)),
            "spheroid gave {}",
            spheroid.value(0)
        );
        // The ellipsoid is the more accurate of the two, so they must differ a little.
        assert_ne!(sphere.value(0), spheroid.value(0));
    }

    #[test]
    fn nulls_propagate() {
        let a = PointBuilder::from_nullable_points(
            [Some(&geo::point!(x: 0.0, y: 0.0)), None].into_iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let b = PointBuilder::from_points(
            [geo::point!(x: 3.0, y: 4.0), geo::point!(x: 1.0, y: 1.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let distance = binary_measure(&a, &b, BinaryMeasure::Distance).unwrap();
        assert_eq!(distance.value(0), 5.0);
        assert!(distance.is_null(1));
    }

    /// One constant side is measured against every row, converted once rather than per row.
    #[test]
    fn a_single_row_side_broadcasts() {
        let a = shapes();
        let origin = PointBuilder::from_points(
            [geo::point!(x: 0.0, y: 0.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let distance = binary_measure(&a, &origin, BinaryMeasure::Distance).unwrap();
        assert_eq!(distance.len(), 3);
        assert_eq!(distance.value(0), 0.0, "the point sits on the origin");
        assert_eq!(distance.value(1), 0.0, "the line starts at the origin");
    }

    #[test]
    fn a_genuine_length_mismatch_is_an_error() {
        let a = shapes();
        let two = PointBuilder::from_points(
            [geo::point!(x: 0.0, y: 0.0), geo::point!(x: 1.0, y: 1.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        assert!(binary_measure(&a, &two, BinaryMeasure::Distance).is_err());
    }
}
