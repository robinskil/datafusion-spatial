//! Linear reference.
//!
//! `ST_ClosestPoint`, `ST_ShortestLine`, `ST_LineLocatePoint` and `ST_LineInterpolatePoint`.
//!
//! These read a position along a geometry rather than a property of it. The two `Line...`
//! functions are defined on line strings only, and return null for anything else, as in PostGIS.

use std::sync::Arc;

use arrow_array::builder::Float64Builder;
use arrow_array::{Array, Float64Array};
use geo::{Closest, ClosestPoint, Euclidean, Geometry, InterpolateLine, LineLocatePoint, Point};
use geoarrow_array::builder::{LineStringBuilder, PointBuilder};
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::{GeoArrowError, GeoArrowResult};
use geoarrow_schema::{Dimension, GeoArrowType, LineStringType, PointType};

use crate::predicate::{broadcast_len, Operand};

/// The point type these functions produce from a given input type.
pub fn point_output_type(input: &GeoArrowType) -> PointType {
    PointType::new(Dimension::XY, Arc::clone(input.metadata()))
        .with_coord_type(geoarrow_schema::CoordType::Separated)
}

/// The line string type `ST_ShortestLine` produces.
pub fn line_output_type(input: &GeoArrowType) -> LineStringType {
    LineStringType::new(Dimension::XY, Arc::clone(input.metadata()))
        .with_coord_type(geoarrow_schema::CoordType::Separated)
}

/// The point of `geom` nearest to `target`.
///
/// `geo` implements [`ClosestPoint`] per geometry variant rather than for `Geometry`, so this
/// dispatches once per row. `Closest::Indeterminate` means no single nearest point exists, which
/// PostGIS reports as null.
fn closest_point_of(geom: &Geometry<f64>, target: &Point<f64>) -> Option<Point<f64>> {
    let closest = match geom {
        Geometry::Point(inner) => inner.closest_point(target),
        Geometry::Line(inner) => inner.closest_point(target),
        Geometry::LineString(inner) => inner.closest_point(target),
        Geometry::Polygon(inner) => inner.closest_point(target),
        Geometry::MultiPoint(inner) => inner.closest_point(target),
        Geometry::MultiLineString(inner) => inner.closest_point(target),
        Geometry::MultiPolygon(inner) => inner.closest_point(target),
        Geometry::Rect(inner) => inner.closest_point(target),
        Geometry::Triangle(inner) => inner.closest_point(target),
        // A collection has no single implementation. Take the nearest over its parts.
        Geometry::GeometryCollection(parts) => {
            let mut best: Option<(f64, Point<f64>)> = None;
            for part in parts.iter() {
                if let Some(candidate) = closest_point_of(part, target) {
                    let distance = geo::Distance::distance(&geo::Euclidean, candidate, *target);
                    if best.as_ref().is_none_or(|(current, _)| distance < *current) {
                        best = Some((distance, candidate));
                    }
                }
            }
            return best.map(|(_, point)| point);
        }
    };

    match closest {
        Closest::Intersection(point) | Closest::SinglePoint(point) => Some(point),
        Closest::Indeterminate => None,
    }
}

/// A representative point for the second argument.
///
/// PostGIS measures to the nearest point of the second geometry. `geo` needs a point, so a
/// non-point second argument is reduced to its nearest vertex. That is exact when the second
/// argument is a point, which is the case these functions are used for.
fn as_point(geom: &Geometry<f64>) -> Option<Point<f64>> {
    use geo::CoordsIter;
    match geom {
        Geometry::Point(point) => Some(*point),
        other => other.coords_iter().next().map(Point::from),
    }
}

/// `ST_ClosestPoint`. The point of the first geometry nearest to the second.
pub fn st_closest_point(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    output: PointType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let len = check_lengths("ST_ClosestPoint", left, right)?;
    let mut left_geom = Operand::new(left, len)?;
    let mut right_geom = Operand::new(right, len)?;

    let mut builder = PointBuilder::with_capacity(output, len);
    for index in 0..len {
        match (left_geom.get(index)?, right_geom.get(index)?) {
            (Some(lhs), Some(rhs)) => match as_point(rhs).and_then(|p| closest_point_of(lhs, &p)) {
                Some(point) => builder.push_point(Some(&point)),
                None => builder.push_null(),
            },
            _ => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// `ST_ShortestLine`. The two point line between the nearest points of the two geometries.
pub fn st_shortest_line(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    output: LineStringType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    const NO_LINE: Option<&geo::LineString<f64>> = None;

    let len = check_lengths("ST_ShortestLine", left, right)?;
    let mut left_geom = Operand::new(left, len)?;
    let mut right_geom = Operand::new(right, len)?;

    let mut builder = LineStringBuilder::with_capacity(output, Default::default());
    for index in 0..len {
        let (Some(lhs), Some(rhs)) = (left_geom.get(index)?, right_geom.get(index)?) else {
            builder.push_line_string(NO_LINE)?;
            continue;
        };
        // One end on each geometry: nearest point of A to B, and of B to A.
        let ends = as_point(rhs)
            .and_then(|target| closest_point_of(lhs, &target))
            .zip(as_point(lhs).and_then(|target| closest_point_of(rhs, &target)));

        match ends {
            Some((start, end)) => {
                let line = geo::LineString::new(vec![start.into(), end.into()]);
                builder.push_line_string(Some(&line))?;
            }
            None => builder.push_line_string(NO_LINE)?,
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// `ST_LineLocatePoint`. Where along a line string a point falls, from 0 to 1.
pub fn st_line_locate_point(
    line: &dyn GeoArrowArray,
    point: &dyn GeoArrowArray,
) -> GeoArrowResult<Float64Array> {
    let len = check_lengths("ST_LineLocatePoint", line, point)?;
    let mut line_geom = Operand::new(line, len)?;
    let mut point_geom = Operand::new(point, len)?;

    let mut builder = Float64Builder::with_capacity(len);
    for index in 0..len {
        match (line_geom.get(index)?, point_geom.get(index)?) {
            (Some(line), Some(other)) if matches!(line, Geometry::LineString(_)) => {
                let Geometry::LineString(line) = line else {
                    unreachable!()
                };
                match as_point(other) {
                    Some(target) => builder.append_option(line.line_locate_point(&target)),
                    None => builder.append_null(),
                }
            }
            // PostGIS restricts this one to line strings.
            _ => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// `ST_LineInterpolatePoint`. The point a given fraction along a line string.
///
/// A fraction outside `0..=1` gives null, as in PostGIS.
pub fn st_line_interpolate_point(
    line: &dyn GeoArrowArray,
    fraction: &Float64Array,
    output: PointType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let len = line.len();
    if fraction.len() != len {
        return Err(GeoArrowError::InvalidGeoArrow(format!(
            "ST_LineInterpolatePoint needs matching lengths, got {len} and {}",
            fraction.len()
        )));
    }

    let mut line_geom = Operand::new(line, len)?;
    let mut builder = PointBuilder::with_capacity(output, len);

    for index in 0..len {
        if fraction.is_null(index) {
            builder.push_null();
            continue;
        }
        let position = fraction.value(index);
        if !(0.0..=1.0).contains(&position) {
            builder.push_null();
            continue;
        }
        match line_geom.get(index)? {
            Some(Geometry::LineString(line)) => {
                // The metric-space form, not the deprecated Euclidean-only method.
                match Euclidean.point_at_ratio_from_start(line, position) {
                    Some(point) => builder.push_point(Some(&point)),
                    None => builder.push_null(),
                }
            }
            _ => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn check_lengths(
    function: &str,
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
) -> GeoArrowResult<usize> {
    broadcast_len(function, left, right)
}

/// `ST_Project`. The point a given distance and bearing away from a start point.
///
/// PostGIS takes the distance in metres and the azimuth in radians, and answers on the WGS 84
/// ellipsoid. This does the same, through [`geo::GeodesicDestination`].
pub fn st_project(
    array: &dyn GeoArrowArray,
    distance: &Float64Array,
    azimuth: &Float64Array,
    output: PointType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    use geo::{Destination, Geodesic};

    let len = array.len();
    for (name, values) in [("distance", distance), ("azimuth", azimuth)] {
        if values.len() != len && values.len() != 1 {
            return Err(GeoArrowError::InvalidGeoArrow(format!(
                "ST_Project needs one {name} per row or a single constant, got {} for {len} rows",
                values.len()
            )));
        }
    }

    let mut points = crate::predicate::Operand::new(array, len)?;
    let mut builder = PointBuilder::with_capacity(output, len);

    for index in 0..len {
        let distance_at = if distance.len() == 1 { 0 } else { index };
        let azimuth_at = if azimuth.len() == 1 { 0 } else { index };

        let (Some(geom), false, false) = (
            points.get(index)?,
            distance.is_null(distance_at),
            azimuth.is_null(azimuth_at),
        ) else {
            builder.push_null();
            continue;
        };
        let Geometry::Point(start) = geom else {
            // PostGIS restricts this one to points.
            builder.push_null();
            continue;
        };

        // `geo` takes the bearing in degrees, PostGIS gives it in radians.
        let bearing = azimuth.value(azimuth_at).to_degrees();
        let moved = Geodesic.destination(*start, bearing, distance.value(distance_at));
        builder.push_point(Some(&moved));
    }
    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use geo_traits::to_geo::ToGeoGeometry;
    use geoarrow_array::builder::{LineStringBuilder, PointBuilder};
    use geoarrow_array::cast::AsGeoArrowArray;
    use geoarrow_array::GeoArrowArrayAccessor;
    use geoarrow_schema::PolygonType;

    use super::*;

    fn line() -> geoarrow_array::array::LineStringArray {
        let lines: Vec<geo::LineString<f64>> = vec![geo::wkt! { LINESTRING(0.0 0.0,10.0 0.0) }];
        LineStringBuilder::from_line_strings(
            &lines,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish()
    }

    fn point(x: f64, y: f64) -> geoarrow_array::array::PointArray {
        PointBuilder::from_points(
            [geo::point!(x: x, y: y)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish()
    }

    fn read_point(array: &dyn GeoArrowArray, row: usize) -> Option<(f64, f64)> {
        let points = array.as_point();
        points.get(row).unwrap().map(|geom| {
            let geo::Geometry::Point(p) = geom.to_geometry() else {
                panic!("expected a point")
            };
            (p.x(), p.y())
        })
    }

    #[test]
    fn closest_point_lands_on_the_line() {
        let line = line();
        let target = point(5.0, 3.0);
        let output = point_output_type(&line.data_type());

        let closest = st_closest_point(&line, &target, output).unwrap();
        assert_eq!(
            read_point(closest.as_ref(), 0),
            Some((5.0, 0.0)),
            "the foot of the perpendicular"
        );
    }

    #[test]
    fn shortest_line_joins_both_geometries() {
        let line = line();
        let target = point(5.0, 3.0);
        let output = line_output_type(&line.data_type());

        let shortest = st_shortest_line(&line, &target, output).unwrap();
        let geo::Geometry::LineString(joined) =
            shortest.as_line_string().value(0).unwrap().to_geometry()
        else {
            panic!("expected a line string")
        };

        assert_eq!(joined.0.len(), 2);
        assert_eq!((joined.0[0].x, joined.0[0].y), (5.0, 0.0));
        assert_eq!((joined.0[1].x, joined.0[1].y), (5.0, 3.0));
    }

    #[test]
    fn line_locate_point_returns_a_fraction() {
        let line = line();
        let quarter = point(2.5, 0.0);
        let located = st_line_locate_point(&line, &quarter).unwrap();
        assert!((located.value(0) - 0.25).abs() < 1e-12);

        // Off the line, but still projected onto it.
        let off = point(7.5, 4.0);
        let located = st_line_locate_point(&line, &off).unwrap();
        assert!((located.value(0) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn line_interpolate_point_walks_the_line() {
        let line = line();
        let output = point_output_type(&line.data_type());

        let half = Float64Array::from(vec![0.5]);
        let midpoint = st_line_interpolate_point(&line, &half, output.clone()).unwrap();
        assert_eq!(read_point(midpoint.as_ref(), 0), Some((5.0, 0.0)));

        let ends = Float64Array::from(vec![0.0]);
        let start = st_line_interpolate_point(&line, &ends, output.clone()).unwrap();
        assert_eq!(read_point(start.as_ref(), 0), Some((0.0, 0.0)));

        // Outside the unit interval gives null.
        for bad in [-0.1, 1.1] {
            let fraction = Float64Array::from(vec![bad]);
            let result = st_line_interpolate_point(&line, &fraction, output.clone()).unwrap();
            assert!(
                read_point(result.as_ref(), 0).is_none(),
                "{bad} must be null"
            );
        }
    }

    #[test]
    fn line_functions_are_null_for_other_types() {
        use geoarrow_array::builder::PolygonBuilder;

        let squares: Vec<geo::Polygon<f64>> =
            vec![geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 0.0)) }];
        let polygon = PolygonBuilder::from_polygons(
            &squares,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let target = point(0.5, 0.5);

        assert!(st_line_locate_point(&polygon, &target).unwrap().is_null(0));

        let output = point_output_type(&polygon.data_type());
        let fraction = Float64Array::from(vec![0.5]);
        let result = st_line_interpolate_point(&polygon, &fraction, output).unwrap();
        assert!(read_point(result.as_ref(), 0).is_none());
    }

    #[test]
    fn nulls_propagate() {
        let lines: Vec<geo::LineString<f64>> = vec![geo::wkt! { LINESTRING(0.0 0.0,10.0 0.0) }];
        let line = LineStringBuilder::from_nullable_line_strings(
            &[Some(&lines[0]), None],
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let targets = PointBuilder::from_points(
            [geo::point!(x: 5.0, y: 1.0), geo::point!(x: 5.0, y: 1.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let located = st_line_locate_point(&line, &targets).unwrap();
        assert!(!located.is_null(0));
        assert!(located.is_null(1));
    }
}
