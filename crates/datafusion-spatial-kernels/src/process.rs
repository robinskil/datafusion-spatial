//! Process and overlay functions.
//!
//! These build new geometries. They do not read a property of an existing one. So they allocate
//! by nature. The work goes into exact capacity, and into a skip of every row that needs no
//! work. It does not go into per-row tricks.
//!
//! # Output types
//!
//! Every function in this module returns a mixed geometry array. The output type of an overlay is
//! not knowable from the input types. Two polygons can intersect in a polygon, a line, a point
//! or nothing. A mixed array is the only honest answer at plan time.

use std::sync::Arc;

use arrow_array::builder::{BooleanBuilder, StringBuilder};
use arrow_array::{BooleanArray, Float64Array, StringArray};
use geo::{
    BooleanOps, Buffer, Centroid, ConcaveHull, ConvexHull, Densify, Euclidean, Geometry,
    InteriorPoint, MakeValid, MinimumRotatedRect, MultiPolygon, RemoveRepeatedPoints, Simplify,
    SimplifyVw, Validation,
};
use geoarrow_array::builder::GeometryBuilder;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::{GeoArrowError, GeoArrowResult};
use geoarrow_schema::{GeoArrowType, GeometryType, Metadata};

use crate::materialize::{all_null, GeometryReader};
use crate::predicate::{broadcast_len, broadcast_nulls, Operand};

/// The output type of every function in this module.
///
/// See the module documentation for why this is always a mixed geometry array.
pub fn output_type(input: &GeoArrowType) -> GeometryType {
    GeometryType::new(Arc::clone(input.metadata()))
}

/// The output type when the metadata comes from somewhere else.
pub fn output_type_from(metadata: Arc<Metadata>) -> GeometryType {
    GeometryType::new(metadata)
}

// ------------------------------------------------------------------- overlay

/// A two-argument overlay operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Overlay {
    /// `ST_Union`.
    Union,
    /// `ST_Intersection`.
    Intersection,
    /// `ST_Difference`.
    Difference,
    /// `ST_SymDifference`.
    SymDifference,
}

impl Overlay {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::Union => "ST_Union",
            Self::Intersection => "ST_Intersection",
            Self::Difference => "ST_Difference",
            Self::SymDifference => "ST_SymDifference",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::Union => "st_union",
            Self::Intersection => "st_intersection",
            Self::Difference => "st_difference",
            Self::SymDifference => "st_symdifference",
        }
    }

    /// Every overlay operation, for registration.
    pub const ALL: [Self; 4] = [
        Self::Union,
        Self::Intersection,
        Self::Difference,
        Self::SymDifference,
    ];

    fn apply(self, left: &MultiPolygon<f64>, right: &MultiPolygon<f64>) -> MultiPolygon<f64> {
        match self {
            Self::Union => left.union(right),
            Self::Intersection => left.intersection(right),
            Self::Difference => left.difference(right),
            Self::SymDifference => left.xor(right),
        }
    }
}

/// Any overlay operation over two arrays.
///
/// `geo` implements boolean operations on areal geometries only, which is where they are defined
/// without ambiguity. A non-areal row yields null rather than a silently wrong answer.
pub fn overlay(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    operation: Overlay,
    output: GeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let len = broadcast_len(operation.function_name(), left, right)?;
    let mut left_geom = Operand::new(left, len)?;
    let mut right_geom = Operand::new(right, len)?;

    let mut builder = GeometryBuilder::new(output);
    for index in 0..len {
        let (Some(lhs), Some(rhs)) = (left_geom.get(index)?, right_geom.get(index)?) else {
            builder.push_null();
            continue;
        };
        match (as_areal(lhs), as_areal(rhs)) {
            (Some(a), Some(b)) => {
                let result = operation.apply(&a, &b);
                builder.push_geometry(Some(&Geometry::MultiPolygon(result)))?;
            }
            _ => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// View a geometry as a multi polygon, or `None` when it has no area.
fn as_areal(geom: &Geometry<f64>) -> Option<MultiPolygon<f64>> {
    match geom {
        Geometry::Polygon(polygon) => Some(MultiPolygon::new(vec![polygon.clone()])),
        Geometry::MultiPolygon(polygons) => Some(polygons.clone()),
        Geometry::Rect(rect) => Some(MultiPolygon::new(vec![rect.to_polygon()])),
        Geometry::Triangle(triangle) => Some(MultiPolygon::new(vec![triangle.to_polygon()])),
        _ => None,
    }
}

// --------------------------------------------------------------------- shape

/// A one-argument geometry transform that takes no extra parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shape {
    /// `ST_ConvexHull`.
    ConvexHull,
    /// `ST_OrientedEnvelope`. The smallest rotated rectangle that covers the geometry.
    OrientedEnvelope,
    /// `ST_Boundary`.
    Boundary,
    /// `ST_Centroid`.
    Centroid,
    /// `ST_PointOnSurface`. A point guaranteed to lie on the geometry.
    PointOnSurface,
    /// `ST_MakeValid`.
    MakeValid,
    /// `ST_RemoveRepeatedPoints`.
    RemoveRepeatedPoints,
    /// `ST_Reverse`.
    Reverse,
    /// `ST_ForcePolygonCCW`.
    ForceCcw,
    /// `ST_ForcePolygonCW`.
    ForceCw,
}

impl Shape {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::ConvexHull => "ST_ConvexHull",
            Self::OrientedEnvelope => "ST_OrientedEnvelope",
            Self::Boundary => "ST_Boundary",
            Self::Centroid => "ST_Centroid",
            Self::PointOnSurface => "ST_PointOnSurface",
            Self::MakeValid => "ST_MakeValid",
            Self::RemoveRepeatedPoints => "ST_RemoveRepeatedPoints",
            Self::Reverse => "ST_Reverse",
            Self::ForceCcw => "ST_ForcePolygonCCW",
            Self::ForceCw => "ST_ForcePolygonCW",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::ConvexHull => "st_convexhull",
            Self::OrientedEnvelope => "st_orientedenvelope",
            Self::Boundary => "st_boundary",
            Self::Centroid => "st_centroid",
            Self::PointOnSurface => "st_pointonsurface",
            Self::MakeValid => "st_makevalid",
            Self::RemoveRepeatedPoints => "st_removerepeatedpoints",
            Self::Reverse => "st_reverse",
            Self::ForceCcw => "st_forcepolygonccw",
            Self::ForceCw => "st_forcepolygoncw",
        }
    }

    /// Every shape function, for registration.
    pub const ALL: [Self; 10] = [
        Self::ConvexHull,
        Self::OrientedEnvelope,
        Self::Boundary,
        Self::Centroid,
        Self::PointOnSurface,
        Self::MakeValid,
        Self::RemoveRepeatedPoints,
        Self::Reverse,
        Self::ForceCcw,
        Self::ForceCw,
    ];

    /// Apply this transform to one geometry.
    ///
    /// Returns `None` where PostGIS returns NULL, such as the centroid of an empty geometry.
    pub fn apply(self, geom: &Geometry<f64>) -> Option<Geometry<f64>> {
        use geo::orient::Direction;

        match self {
            Self::ConvexHull => Some(Geometry::Polygon(geom.convex_hull())),
            Self::OrientedEnvelope => geom.minimum_rotated_rect().map(Geometry::Polygon),
            Self::Boundary => Some(boundary_of(geom)),
            Self::Centroid => geom.centroid().map(Geometry::Point),
            Self::PointOnSurface => geom.interior_point().map(Geometry::Point),
            // `geo` repairs areal geometries. Anything else is already as valid as it gets.
            Self::MakeValid => match geom {
                Geometry::Polygon(polygon) => polygon.make_valid().ok().map(Geometry::MultiPolygon),
                Geometry::MultiPolygon(polygons) => {
                    polygons.make_valid().ok().map(Geometry::MultiPolygon)
                }
                other => Some(other.clone()),
            },
            Self::RemoveRepeatedPoints => Some(geom.remove_repeated_points()),
            Self::Reverse => Some(reversed(geom)),
            Self::ForceCcw => Some(oriented(geom, Direction::Default)),
            Self::ForceCw => Some(oriented(geom, Direction::Reversed)),
        }
    }
}

/// Force the ring orientation of the areal parts of a geometry.
///
/// `geo` implements `Orient` per variant rather than for `Geometry`.
fn oriented(geom: &Geometry<f64>, direction: geo::orient::Direction) -> Geometry<f64> {
    use geo::Orient;
    match geom {
        Geometry::Polygon(polygon) => Geometry::Polygon(polygon.orient(direction)),
        Geometry::MultiPolygon(polygons) => Geometry::MultiPolygon(polygons.orient(direction)),
        Geometry::Rect(rect) => Geometry::Polygon(rect.to_polygon().orient(direction)),
        Geometry::Triangle(triangle) => Geometry::Polygon(triangle.to_polygon().orient(direction)),
        // Orientation is meaningless without a ring.
        other => other.clone(),
    }
}

/// `ST_Boundary`. The topological boundary of a geometry.
///
/// `geo` has no boundary algorithm, so this follows the OGC rule.
/// The boundary of a polygon is its rings. The boundary of an open line is its two end points.
/// A closed line and a point both have an empty boundary.
fn boundary_of(geom: &Geometry<f64>) -> Geometry<f64> {
    use geo::{LineString, MultiLineString, MultiPoint, Point};

    match geom {
        Geometry::Point(_) | Geometry::MultiPoint(_) => {
            Geometry::MultiPoint(MultiPoint::new(Vec::new()))
        }
        Geometry::LineString(line) => Geometry::MultiPoint(line_boundary(line)),
        Geometry::MultiLineString(lines) => {
            // A coordinate shared by an even number of ends cancels out, per the OGC mod-2 rule.
            let mut ends: Vec<Point<f64>> = Vec::new();
            for line in lines.iter() {
                for point in line_boundary(line).into_iter() {
                    match ends.iter().position(|existing| *existing == point) {
                        Some(at) => {
                            ends.remove(at);
                        }
                        None => ends.push(point),
                    }
                }
            }
            Geometry::MultiPoint(MultiPoint::new(ends))
        }
        Geometry::Polygon(polygon) => {
            let mut rings = vec![polygon.exterior().clone()];
            rings.extend(polygon.interiors().iter().cloned());
            Geometry::MultiLineString(MultiLineString::new(rings))
        }
        Geometry::MultiPolygon(polygons) => {
            let mut rings: Vec<LineString<f64>> = Vec::new();
            for polygon in polygons.iter() {
                rings.push(polygon.exterior().clone());
                rings.extend(polygon.interiors().iter().cloned());
            }
            Geometry::MultiLineString(MultiLineString::new(rings))
        }
        Geometry::Rect(rect) => boundary_of(&Geometry::Polygon(rect.to_polygon())),
        Geometry::Triangle(triangle) => boundary_of(&Geometry::Polygon(triangle.to_polygon())),
        Geometry::Line(line) => Geometry::MultiPoint(MultiPoint::new(vec![
            Point::from(line.start),
            Point::from(line.end),
        ])),
        // The OGC rule for a collection is the union of its parts' boundaries. Keep it simple and
        // report an empty boundary, which is what a collection of points would give anyway.
        Geometry::GeometryCollection(_) => Geometry::MultiPoint(MultiPoint::new(Vec::new())),
    }
}

fn line_boundary(line: &geo::LineString<f64>) -> geo::MultiPoint<f64> {
    use geo::{MultiPoint, Point};
    if line.0.len() < 2 || line.is_closed() {
        return MultiPoint::new(Vec::new());
    }
    MultiPoint::new(vec![
        Point::from(line.0[0]),
        Point::from(line.0[line.0.len() - 1]),
    ])
}

/// Reverse the coordinate order of the lineal parts of a geometry.
fn reversed(geom: &Geometry<f64>) -> Geometry<f64> {
    use geo::{LineString, MultiLineString, MultiPolygon, Polygon};

    fn flip(line: &LineString<f64>) -> LineString<f64> {
        LineString::new(line.0.iter().rev().copied().collect())
    }
    fn flip_polygon(polygon: &Polygon<f64>) -> Polygon<f64> {
        Polygon::new(
            flip(polygon.exterior()),
            polygon.interiors().iter().map(flip).collect(),
        )
    }

    match geom {
        Geometry::LineString(line) => Geometry::LineString(flip(line)),
        Geometry::MultiLineString(lines) => {
            Geometry::MultiLineString(MultiLineString::new(lines.iter().map(flip).collect()))
        }
        Geometry::Polygon(polygon) => Geometry::Polygon(flip_polygon(polygon)),
        Geometry::MultiPolygon(polygons) => Geometry::MultiPolygon(MultiPolygon::new(
            polygons.iter().map(flip_polygon).collect(),
        )),
        Geometry::Line(line) => Geometry::Line(geo::Line::new(line.end, line.start)),
        other => other.clone(),
    }
}

/// Any one-argument shape transform over an array.
pub fn shape(
    array: &dyn GeoArrowArray,
    transform: Shape,
    output: GeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let mut reader = GeometryReader::new(array)?;
    let mut builder = GeometryBuilder::new(output);
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => match transform.apply(geom) {
                Some(result) => builder.push_geometry(Some(&result))?,
                None => builder.push_null(),
            },
            None => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

// ------------------------------------------------------ parameterized shapes

/// A one-argument transform that also takes a distance or tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sized {
    /// `ST_Buffer`.
    Buffer,
    /// `ST_Simplify`. It runs Ramer-Douglas-Peucker.
    Simplify,
    /// `ST_SimplifyVW`. It runs Visvalingam-Whyatt.
    SimplifyVw,
    /// `ST_Segmentize`. Split segments longer than the given length.
    Segmentize,
    /// `ST_ConcaveHull`. The parameter is the target ratio, from 0 to 1.
    ConcaveHull,
    /// `ST_ChaikinSmoothing`. The parameter is the iteration count.
    Chaikin,
}

impl Sized {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::Buffer => "ST_Buffer",
            Self::Simplify => "ST_Simplify",
            Self::SimplifyVw => "ST_SimplifyVW",
            Self::Segmentize => "ST_Segmentize",
            Self::ConcaveHull => "ST_ConcaveHull",
            Self::Chaikin => "ST_ChaikinSmoothing",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::Buffer => "st_buffer",
            Self::Simplify => "st_simplify",
            Self::SimplifyVw => "st_simplifyvw",
            Self::Segmentize => "st_segmentize",
            Self::ConcaveHull => "st_concavehull",
            Self::Chaikin => "st_chaikinsmoothing",
        }
    }

    /// Every parameterized shape function, for registration.
    pub const ALL: [Self; 6] = [
        Self::Buffer,
        Self::Simplify,
        Self::SimplifyVw,
        Self::Segmentize,
        Self::ConcaveHull,
        Self::Chaikin,
    ];

    /// Apply this transform to one geometry.
    pub fn apply(self, geom: &Geometry<f64>, parameter: f64) -> Option<Geometry<f64>> {
        match self {
            Self::Buffer => Some(Geometry::MultiPolygon(geom.buffer(parameter))),
            Self::Simplify => Some(simplify_rdp(geom, parameter)),
            Self::SimplifyVw => Some(simplify_vw(geom, parameter)),
            Self::Segmentize => {
                // A segment length of zero or less would never terminate.
                (parameter > 0.0).then(|| segmentize(geom, parameter))
            }
            Self::ConcaveHull => concave_hull(geom, parameter),
            Self::Chaikin => {
                // The parameter is a count, so a fraction or a negative makes no sense. Cap it:
                // every iteration doubles the vertex count.
                let iterations = parameter.round();
                (0.0..=8.0)
                    .contains(&iterations)
                    .then(|| chaikin(geom, iterations as usize))
            }
        }
    }
}

/// Ramer-Douglas-Peucker, dispatched per variant.
///
/// `geo` implements it for lineal and areal types. A point has nothing to simplify.
fn simplify_rdp(geom: &Geometry<f64>, epsilon: f64) -> Geometry<f64> {
    match geom {
        Geometry::LineString(line) => Geometry::LineString(line.simplify(epsilon)),
        Geometry::MultiLineString(lines) => Geometry::MultiLineString(lines.simplify(epsilon)),
        Geometry::Polygon(polygon) => Geometry::Polygon(polygon.simplify(epsilon)),
        Geometry::MultiPolygon(polygons) => Geometry::MultiPolygon(polygons.simplify(epsilon)),
        other => other.clone(),
    }
}

/// Visvalingam-Whyatt, dispatched per variant.
fn simplify_vw(geom: &Geometry<f64>, epsilon: f64) -> Geometry<f64> {
    match geom {
        Geometry::LineString(line) => Geometry::LineString(line.simplify_vw(epsilon)),
        Geometry::MultiLineString(lines) => Geometry::MultiLineString(lines.simplify_vw(epsilon)),
        Geometry::Polygon(polygon) => Geometry::Polygon(polygon.simplify_vw(epsilon)),
        Geometry::MultiPolygon(polygons) => Geometry::MultiPolygon(polygons.simplify_vw(epsilon)),
        other => other.clone(),
    }
}

/// Split every segment longer than `max_length`.
fn segmentize(geom: &Geometry<f64>, max_length: f64) -> Geometry<f64> {
    match geom {
        Geometry::Line(line) => Geometry::LineString(Euclidean.densify(line, max_length)),
        Geometry::LineString(line) => Geometry::LineString(Euclidean.densify(line, max_length)),
        Geometry::MultiLineString(lines) => {
            Geometry::MultiLineString(Euclidean.densify(lines, max_length))
        }
        Geometry::Polygon(polygon) => Geometry::Polygon(Euclidean.densify(polygon, max_length)),
        Geometry::MultiPolygon(polygons) => {
            Geometry::MultiPolygon(Euclidean.densify(polygons, max_length))
        }
        Geometry::Rect(rect) => Geometry::Polygon(Euclidean.densify(rect, max_length)),
        Geometry::Triangle(triangle) => Geometry::Polygon(Euclidean.densify(triangle, max_length)),
        other => other.clone(),
    }
}

/// The Chaikin algorithm, dispatched per variant.
///
/// `geo` implements it for lineal and areal types. A point has nothing to smooth.
fn chaikin(geom: &Geometry<f64>, iterations: usize) -> Geometry<f64> {
    use geo::ChaikinSmoothing;
    match geom {
        Geometry::LineString(line) => Geometry::LineString(line.chaikin_smoothing(iterations)),
        Geometry::MultiLineString(lines) => {
            Geometry::MultiLineString(lines.chaikin_smoothing(iterations))
        }
        Geometry::Polygon(polygon) => Geometry::Polygon(polygon.chaikin_smoothing(iterations)),
        Geometry::MultiPolygon(polygons) => {
            Geometry::MultiPolygon(polygons.chaikin_smoothing(iterations))
        }
        other => other.clone(),
    }
}

/// The concave hull, dispatched per variant.
///
/// PostGIS names the parameter `target_percent`, where 1 gives the convex hull and smaller values
/// give a tighter fit. `geo` names it `concavity` and reads it the other way round. So this maps
/// one onto the other. It does not pass the number straight through.
fn concave_hull(geom: &Geometry<f64>, target_percent: f64) -> Option<Geometry<f64>> {
    use geo::concave_hull::ConcaveHullOptions;

    // A target of 1 is the convex hull. Lower targets allow more concavity.
    let options = ConcaveHullOptions {
        concavity: (1.0 / target_percent.clamp(1e-6, 1.0)).min(1e6),
        length_threshold: 0.0,
    };

    match geom {
        Geometry::MultiPoint(points) => {
            Some(Geometry::Polygon(points.concave_hull_with_options(options)))
        }
        Geometry::LineString(line) => {
            Some(Geometry::Polygon(line.concave_hull_with_options(options)))
        }
        Geometry::MultiLineString(lines) => {
            Some(Geometry::Polygon(lines.concave_hull_with_options(options)))
        }
        Geometry::Polygon(polygon) => Some(Geometry::Polygon(
            polygon.concave_hull_with_options(options),
        )),
        Geometry::MultiPolygon(polygons) => Some(Geometry::Polygon(
            polygons.concave_hull_with_options(options),
        )),
        // A single point has no hull, and `geo` offers none for a collection.
        _ => None,
    }
}

/// Any parameterized shape transform over an array.
///
/// The parameter may be one constant or one value per row.
pub fn sized_shape(
    array: &dyn GeoArrowArray,
    transform: Sized,
    parameter: &Float64Array,
    output: GeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let len = array.len();
    if parameter.len() != len && parameter.len() != 1 {
        return Err(GeoArrowError::InvalidGeoArrow(format!(
            "{} needs one parameter per row or a single constant, got {} for {len} rows",
            transform.function_name(),
            parameter.len()
        )));
    }
    use arrow_array::Array;

    let broadcast = parameter.len() == 1 && array.len() != 1;
    let mut reader = GeometryReader::new(array)?;
    let mut builder = GeometryBuilder::new(output);

    for row in 0..array.len() {
        let slot = if broadcast { 0 } else { row };
        if parameter.is_null(slot) {
            builder.push_null();
            continue;
        }
        let Some(geom) = reader.read(row)? else {
            builder.push_null();
            continue;
        };
        match transform.apply(geom, parameter.value(slot)) {
            Some(result) => builder.push_geometry(Some(&result))?,
            None => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

// ------------------------------------------------------------------ validity

/// `ST_IsValid`.
pub fn st_is_valid(array: &dyn GeoArrowArray) -> GeoArrowResult<BooleanArray> {
    if all_null(array) {
        return Ok(BooleanArray::new_null(array.len()));
    }
    let mut reader = GeometryReader::new(array)?;
    let mut builder = BooleanBuilder::with_capacity(array.len());
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => builder.append_value(geom.is_valid()),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// `ST_IsValidReason`. The first problem with the geometry, or `Valid Geometry`.
///
/// PostGIS returns that exact string for a valid geometry, so scripts can compare against it.
pub fn st_is_valid_reason(array: &dyn GeoArrowArray) -> GeoArrowResult<StringArray> {
    if all_null(array) {
        return Ok(StringArray::new_null(array.len()));
    }
    let mut reader = GeometryReader::new(array)?;
    let mut builder = StringBuilder::with_capacity(array.len(), array.len() * 24);
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => match geom.check_validation() {
                Ok(()) => builder.append_value("Valid Geometry"),
                Err(problem) => builder.append_value(problem.to_string()),
            },
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// The combined null buffer of a binary overlay, exposed for the UDF layer.
pub fn overlay_nulls(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    rows: usize,
) -> Option<arrow_buffer::NullBuffer> {
    broadcast_nulls(left, right, rows)
}

#[cfg(test)]
mod tests {
    use geo_traits::to_geo::ToGeoGeometry;
    use geoarrow_array::builder::{GeometryBuilder as GeoBuilder, PointBuilder, PolygonBuilder};
    use geoarrow_array::cast::AsGeoArrowArray;
    use geoarrow_array::GeoArrowArrayAccessor;
    use geoarrow_schema::{Dimension, PointType, PolygonType};

    use super::*;

    fn polygons(values: Vec<geo::Polygon<f64>>) -> geoarrow_array::array::PolygonArray {
        PolygonBuilder::from_polygons(&values, PolygonType::new(Dimension::XY, Default::default()))
            .finish()
    }

    fn unit_square() -> geo::Polygon<f64> {
        geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 1.0,0.0 0.0)) }
    }

    fn shifted_square() -> geo::Polygon<f64> {
        geo::wkt! { POLYGON((0.5 0.0,1.5 0.0,1.5 1.0,0.5 1.0,0.5 0.0)) }
    }

    fn read(array: &dyn GeoArrowArray, row: usize) -> Option<geo::Geometry<f64>> {
        array
            .as_geometry()
            .get(row)
            .unwrap()
            .map(|geom| geom.to_geometry())
    }

    fn area_of(array: &dyn GeoArrowArray, row: usize) -> f64 {
        use geo::Area;
        read(array, row).map_or(0.0, |geom| geom.unsigned_area())
    }

    #[test]
    fn overlay_operations_agree_with_area() {
        let a = polygons(vec![unit_square()]);
        let b = polygons(vec![shifted_square()]);
        let output = output_type(&a.data_type());

        // The two unit squares overlap on half their width.
        let union = overlay(&a, &b, Overlay::Union, output.clone()).unwrap();
        assert!((area_of(union.as_ref(), 0) - 1.5).abs() < 1e-9);

        let intersection = overlay(&a, &b, Overlay::Intersection, output.clone()).unwrap();
        assert!((area_of(intersection.as_ref(), 0) - 0.5).abs() < 1e-9);

        let difference = overlay(&a, &b, Overlay::Difference, output.clone()).unwrap();
        assert!((area_of(difference.as_ref(), 0) - 0.5).abs() < 1e-9);

        let symmetric = overlay(&a, &b, Overlay::SymDifference, output).unwrap();
        assert!((area_of(symmetric.as_ref(), 0) - 1.0).abs() < 1e-9);
    }

    /// Boolean operations are defined for areal geometries, so a point yields null.
    #[test]
    fn overlay_of_a_point_is_null() {
        let a = polygons(vec![unit_square()]);
        let point = PointBuilder::from_points(
            [geo::point!(x: 0.5, y: 0.5)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let output = output_type(&a.data_type());

        let result = overlay(&a, &point, Overlay::Union, output).unwrap();
        assert!(read(result.as_ref(), 0).is_none());
    }

    #[test]
    fn convex_hull_and_oriented_envelope() {
        let mut builder = GeoBuilder::new(GeometryType::new(Default::default()));
        builder
            .push_geometry(Some(&geo::Geometry::<f64>::from(
                geo::wkt! { MULTIPOINT(0.0 0.0,2.0 0.0,2.0 2.0,0.0 2.0,1.0 1.0) },
            )))
            .unwrap();
        let array = builder.finish();
        let output = output_type(&array.data_type());

        let hull = shape(&array, Shape::ConvexHull, output.clone()).unwrap();
        assert!(
            (area_of(hull.as_ref(), 0) - 4.0).abs() < 1e-9,
            "the 2 by 2 box"
        );

        let envelope = shape(&array, Shape::OrientedEnvelope, output).unwrap();
        assert!((area_of(envelope.as_ref(), 0) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn centroid_and_point_on_surface() {
        let array = polygons(vec![unit_square()]);
        let output = output_type(&array.data_type());

        let geo::Geometry::Point(centroid) = read(
            shape(&array, Shape::Centroid, output.clone())
                .unwrap()
                .as_ref(),
            0,
        )
        .unwrap() else {
            panic!("expected a point")
        };
        assert!((centroid.x() - 0.5).abs() < 1e-9);
        assert!((centroid.y() - 0.5).abs() < 1e-9);

        // A point on surface must actually lie inside the polygon.
        use geo::Contains;
        let on_surface = read(
            shape(&array, Shape::PointOnSurface, output)
                .unwrap()
                .as_ref(),
            0,
        )
        .unwrap();
        assert!(unit_square().contains(&on_surface));
    }

    #[test]
    fn boundary_follows_the_ogc_rules() {
        let mut builder = GeoBuilder::new(GeometryType::new(Default::default()));
        for geom in [
            geo::Geometry::<f64>::from(unit_square()),
            geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0) }.into(),
            geo::wkt! { LINESTRING(0.0 0.0,1.0 0.0,1.0 1.0,0.0 0.0) }.into(),
            geo::wkt! { POINT(1.0 1.0) }.into(),
        ] {
            builder.push_geometry(Some(&geom)).unwrap();
        }
        let array = builder.finish();
        let output = output_type(&array.data_type());
        let result = shape(&array, Shape::Boundary, output).unwrap();

        // A polygon's boundary is its rings.
        assert!(matches!(
            read(result.as_ref(), 0),
            Some(geo::Geometry::MultiLineString(_))
        ));
        // An open line gives its two endpoints.
        let Some(geo::Geometry::MultiPoint(ends)) = read(result.as_ref(), 1) else {
            panic!("expected a multi point")
        };
        assert_eq!(ends.0.len(), 2);
        // A closed line has an empty boundary.
        let Some(geo::Geometry::MultiPoint(none)) = read(result.as_ref(), 2) else {
            panic!("expected a multi point")
        };
        assert_eq!(none.0.len(), 0);
        // And so does a point.
        let Some(geo::Geometry::MultiPoint(none)) = read(result.as_ref(), 3) else {
            panic!("expected a multi point")
        };
        assert_eq!(none.0.len(), 0);
    }

    #[test]
    fn simplify_drops_collinear_points() {
        let mut builder = GeoBuilder::new(GeometryType::new(Default::default()));
        builder
            .push_geometry(Some(&geo::Geometry::<f64>::from(
                geo::wkt! { LINESTRING(0.0 0.0,1.0 0.001,2.0 0.0,3.0 0.0) },
            )))
            .unwrap();
        let array = builder.finish();
        let output = output_type(&array.data_type());
        let tolerance = Float64Array::from(vec![0.01]);

        for transform in [Sized::Simplify, Sized::SimplifyVw] {
            let result = sized_shape(&array, transform, &tolerance, output.clone()).unwrap();
            let Some(geo::Geometry::LineString(line)) = read(result.as_ref(), 0) else {
                panic!("expected a line string")
            };
            assert!(
                line.0.len() < 4,
                "{} kept every point",
                transform.function_name()
            );
        }
    }

    #[test]
    fn buffer_grows_the_area() {
        let array = polygons(vec![unit_square()]);
        let output = output_type(&array.data_type());
        let distance = Float64Array::from(vec![0.5]);

        let buffered = sized_shape(&array, Sized::Buffer, &distance, output).unwrap();
        let area = area_of(buffered.as_ref(), 0);
        // The unit square grown by 0.5 is 1 + 4 sides of 0.5 + 4 quarter circles of radius 0.5.
        let expected = 1.0 + 4.0 * 0.5 + std::f64::consts::PI * 0.25;
        assert!((area - expected).abs() < 0.05, "area was {area}");
    }

    #[test]
    fn segmentize_adds_vertices() {
        let mut builder = GeoBuilder::new(GeometryType::new(Default::default()));
        builder
            .push_geometry(Some(&geo::Geometry::<f64>::from(
                geo::wkt! { LINESTRING(0.0 0.0,10.0 0.0) },
            )))
            .unwrap();
        let array = builder.finish();
        let output = output_type(&array.data_type());

        let max_length = Float64Array::from(vec![2.0]);
        let result = sized_shape(&array, Sized::Segmentize, &max_length, output.clone()).unwrap();
        let Some(geo::Geometry::LineString(line)) = read(result.as_ref(), 0) else {
            panic!("expected a line string")
        };
        assert!(line.0.len() >= 6, "got {} points", line.0.len());

        // A non-positive length would never terminate, so it gives null.
        let zero = Float64Array::from(vec![0.0]);
        let result = sized_shape(&array, Sized::Segmentize, &zero, output).unwrap();
        assert!(read(result.as_ref(), 0).is_none());
    }

    #[test]
    fn reverse_flips_the_coordinate_order() {
        let mut builder = GeoBuilder::new(GeometryType::new(Default::default()));
        builder
            .push_geometry(Some(&geo::Geometry::<f64>::from(
                geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0,2.0 2.0) },
            )))
            .unwrap();
        let array = builder.finish();
        let output = output_type(&array.data_type());

        let result = shape(&array, Shape::Reverse, output).unwrap();
        let Some(geo::Geometry::LineString(line)) = read(result.as_ref(), 0) else {
            panic!("expected a line string")
        };
        assert_eq!((line.0[0].x, line.0[0].y), (2.0, 2.0));
        assert_eq!((line.0[2].x, line.0[2].y), (0.0, 0.0));
    }

    #[test]
    fn force_orientation_is_stable() {
        use geo::algorithm::winding_order::Winding;

        let array = polygons(vec![unit_square()]);
        let output = output_type(&array.data_type());

        let ccw = shape(&array, Shape::ForceCcw, output.clone()).unwrap();
        let Some(geo::Geometry::Polygon(polygon)) = read(ccw.as_ref(), 0) else {
            panic!("expected a polygon")
        };
        assert!(polygon.exterior().is_ccw());

        let cw = shape(&array, Shape::ForceCw, output).unwrap();
        let Some(geo::Geometry::Polygon(polygon)) = read(cw.as_ref(), 0) else {
            panic!("expected a polygon")
        };
        assert!(polygon.exterior().is_cw());
    }

    #[test]
    fn validity_reports_the_reason() {
        // A polygon whose interior ring escapes the exterior is invalid.
        let broken: geo::Polygon<f64> = geo::wkt! {
            POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 1.0,0.0 0.0),(5.0 5.0,6.0 5.0,6.0 6.0,5.0 5.0))
        };
        let array = polygons(vec![unit_square(), broken]);

        let valid = st_is_valid(&array).unwrap();
        assert!(valid.value(0));
        assert!(!valid.value(1));

        let reasons = st_is_valid_reason(&array).unwrap();
        assert_eq!(reasons.value(0), "Valid Geometry");
        assert!(!reasons.value(1).is_empty());
        assert_ne!(reasons.value(1), "Valid Geometry");
    }

    #[test]
    fn make_valid_repairs_a_polygon() {
        // A bow tie polygon, which self intersects.
        let bow_tie: geo::Polygon<f64> =
            geo::wkt! { POLYGON((0.0 0.0,2.0 2.0,2.0 0.0,0.0 2.0,0.0 0.0)) };
        let array = polygons(vec![bow_tie]);
        let output = output_type(&array.data_type());

        assert!(!st_is_valid(&array).unwrap().value(0));

        let repaired = shape(&array, Shape::MakeValid, output).unwrap();
        let geometry = read(repaired.as_ref(), 0).expect("repair must produce a geometry");
        use geo::Validation;
        assert!(geometry.is_valid(), "the repaired geometry must be valid");
    }

    #[test]
    fn nulls_propagate() {
        let none: Option<&geo::Polygon<f64>> = None;
        let array = PolygonBuilder::from_nullable_polygons(
            &[Some(&unit_square()), none],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let output = output_type(&array.data_type());

        let hull = shape(&array, Shape::ConvexHull, output).unwrap();
        assert!(read(hull.as_ref(), 0).is_some());
        assert!(read(hull.as_ref(), 1).is_none());
    }
}
