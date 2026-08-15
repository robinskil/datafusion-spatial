//! Geometry accessors.
//!
//! These read a property out of every row. Most of them are pure columnar work with no geometric
//! algorithm behind them, so they are the highest throughput functions in the crate.
//!
//! # Two fast paths
//!
//! **Zero copy.** `ST_X` on a point array with separated coordinates hands back the x buffer with
//! its reference count raised. Nothing is copied and nothing is allocated.
//!
//! **Constant fold.** A single-typed array already answers `ST_GeometryType`, `ST_Dimension` and
//! `ST_CoordDim` from its schema. Those kernels fill a constant array and never look at a row.

use crate::materialize::{all_null, GeometryReader};
use arrow_array::builder::{BooleanBuilder, Float64Builder, Int32Builder, StringBuilder};
use arrow_array::{BooleanArray, Float64Array, Int32Array, StringArray};
use arrow_buffer::NullBuffer;
use geo::sweep::Intersections;
use geo::{Line, LineString};
use geo_traits::to_geo::ToGeoGeometry;
use geo_traits::{
    CoordTrait, Dimensions, GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use geoarrow_array::array::{CoordBuffer, PointArray};
use geoarrow_array::cast::AsGeoArrowArray;
use geoarrow_array::{downcast_geoarrow_array, GeoArrowArray, GeoArrowArrayAccessor};
use geoarrow_schema::error::{GeoArrowError, GeoArrowResult};
use geoarrow_schema::{Dimension, GeoArrowType};

/// Which coordinate to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ordinate {
    /// The x coordinate. Present in every dimension.
    X,
    /// The y coordinate. Present in every dimension.
    Y,
    /// The z coordinate. Absent from XY and XYM.
    Z,
    /// The measure. Absent from XY and XYZ.
    M,
}

impl Ordinate {
    /// The PostGIS function name that reads this ordinate.
    pub const fn function_name(self) -> &'static str {
        match self {
            Ordinate::X => "ST_X",
            Ordinate::Y => "ST_Y",
            Ordinate::Z => "ST_Z",
            Ordinate::M => "ST_M",
        }
    }

    /// Position of this ordinate inside a coordinate, or `None` when the dimension lacks it.
    ///
    /// XYM stores the measure in slot 2, XYZM in slot 3. That difference is why this is a lookup
    /// and not a constant.
    pub const fn index_in(self, dim: Dimension) -> Option<usize> {
        match (self, dim) {
            (Ordinate::X, _) => Some(0),
            (Ordinate::Y, _) => Some(1),
            (Ordinate::Z, Dimension::XYZ | Dimension::XYZM) => Some(2),
            (Ordinate::Z, _) => None,
            (Ordinate::M, Dimension::XYM) => Some(2),
            (Ordinate::M, Dimension::XYZM) => Some(3),
            (Ordinate::M, _) => None,
        }
    }

    /// Same lookup against the geo-traits dimension of a single scalar.
    const fn index_in_traits(self, dim: Dimensions) -> Option<usize> {
        match (self, dim) {
            (Ordinate::X, _) => Some(0),
            (Ordinate::Y, _) => Some(1),
            (Ordinate::Z, Dimensions::Xyz | Dimensions::Xyzm) => Some(2),
            (Ordinate::Z, _) => None,
            (Ordinate::M, Dimensions::Xym) => Some(2),
            (Ordinate::M, Dimensions::Xyzm) => Some(3),
            (Ordinate::M, _) => None,
        }
    }
}

/// Returns true when [`ordinate`] accepts this type.
///
/// A single-typed array that is not a point is rejected at plan time. A mixed or serialized array
/// carries an unknown type per row, so it is accepted and non-point rows yield null.
pub fn accepts_ordinate(data_type: &GeoArrowType) -> bool {
    matches!(data_type, GeoArrowType::Point(_)) || is_untyped(data_type)
}

/// Returns true when the row type is unknown before execution.
pub fn is_untyped(data_type: &GeoArrowType) -> bool {
    matches!(
        data_type,
        GeoArrowType::Geometry(_)
            | GeoArrowType::Wkb(_)
            | GeoArrowType::LargeWkb(_)
            | GeoArrowType::WkbView(_)
            | GeoArrowType::Wkt(_)
            | GeoArrowType::LargeWkt(_)
            | GeoArrowType::WktView(_)
    )
}

/// `ST_X`.
pub fn st_x(array: &dyn GeoArrowArray) -> GeoArrowResult<Float64Array> {
    ordinate(array, Ordinate::X)
}

/// `ST_Y`.
pub fn st_y(array: &dyn GeoArrowArray) -> GeoArrowResult<Float64Array> {
    ordinate(array, Ordinate::Y)
}

/// `ST_Z`. Null for a two-dimensional geometry.
pub fn st_z(array: &dyn GeoArrowArray) -> GeoArrowResult<Float64Array> {
    ordinate(array, Ordinate::Z)
}

/// `ST_M`. Null for a geometry without a measure.
pub fn st_m(array: &dyn GeoArrowArray) -> GeoArrowResult<Float64Array> {
    ordinate(array, Ordinate::M)
}

/// Read one ordinate of every point in the array.
pub fn ordinate(array: &dyn GeoArrowArray, ord: Ordinate) -> GeoArrowResult<Float64Array> {
    match array.data_type() {
        GeoArrowType::Point(_) => point_ordinate(array.as_point(), ord),
        other if is_untyped(&other) => {
            downcast_geoarrow_array!(array, untyped_ordinate, ord)
        }
        other => Err(GeoArrowError::IncorrectGeometryType(format!(
            "{} requires a point argument, got {other:?}",
            ord.function_name()
        ))),
    }
}

/// The zero copy path.
fn point_ordinate(array: &PointArray, ord: Ordinate) -> GeoArrowResult<Float64Array> {
    let dim = array.data_type().dimension().unwrap_or(Dimension::XY);
    let nulls = array.logical_nulls();

    let Some(index) = ord.index_in(dim) else {
        // The dimension has no such ordinate. PostGIS returns NULL, not an error.
        return Ok(Float64Array::new_null(array.len()));
    };

    match array.coords() {
        // Separated coordinates already store one buffer per ordinate. Clone the buffer, which is
        // an atomic counter bump, and attach the existing null buffer. No copy, no allocation.
        CoordBuffer::Separated(coords) => Ok(Float64Array::new(
            coords.raw_buffers()[index].clone(),
            nulls,
        )),
        // Interleaved coordinates hold xyxy in one buffer. A strided read is the best we can do.
        CoordBuffer::Interleaved(coords) => {
            let stride = coords.dim().size();
            let raw = coords.coords();
            let mut values = Vec::with_capacity(array.len());
            values.extend(raw.iter().skip(index).step_by(stride).copied());
            Ok(Float64Array::new(values.into(), nulls))
        }
    }
}

/// Per-row path for arrays whose row type is unknown before execution.
fn untyped_ordinate<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
    ord: Ordinate,
) -> GeoArrowResult<Float64Array> {
    let mut builder = Float64Builder::with_capacity(array.len());

    for item in array.iter() {
        let Some(geom) = item else {
            builder.append_null();
            continue;
        };
        let geom = geom?;
        match geom.as_type() {
            GeometryType::Point(point) => match point.coord() {
                Some(coord) => match ord.index_in_traits(coord.dim()) {
                    Some(index) => builder.append_option(coord.nth(index)),
                    None => builder.append_null(),
                },
                // An empty point holds no coordinate.
                None => builder.append_null(),
            },
            // PostGIS returns NULL for a mixed column. It raises no error.
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish())
}

/// `ST_GeometryType`. Returns `ST_Point`, `ST_LineString` and so on.
///
/// A single-type array answers from its schema. So this fills a constant array and reads no row.
pub fn st_geometry_type(array: &dyn GeoArrowArray) -> GeoArrowResult<StringArray> {
    if let Some(name) = static_type_name(&array.data_type()) {
        return Ok(constant_strings(name, array.len(), array.logical_nulls()));
    }
    downcast_geoarrow_array!(array, geometry_type_impl)
}

/// The PostGIS type name of a statically typed array, or `None` when the type varies per row.
fn static_type_name(data_type: &GeoArrowType) -> Option<&'static str> {
    match data_type {
        GeoArrowType::Point(_) => Some("ST_Point"),
        GeoArrowType::LineString(_) => Some("ST_LineString"),
        GeoArrowType::Polygon(_) => Some("ST_Polygon"),
        GeoArrowType::MultiPoint(_) => Some("ST_MultiPoint"),
        GeoArrowType::MultiLineString(_) => Some("ST_MultiLineString"),
        GeoArrowType::MultiPolygon(_) => Some("ST_MultiPolygon"),
        GeoArrowType::GeometryCollection(_) => Some("ST_GeometryCollection"),
        // A box behaves as a polygon everywhere in PostGIS.
        GeoArrowType::Rect(_) => Some("ST_Polygon"),
        _ => None,
    }
}

fn geometry_type_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
) -> GeoArrowResult<StringArray> {
    let mut builder = StringBuilder::with_capacity(array.len(), array.len() * 12);
    for item in array.iter() {
        match item {
            Some(geom) => builder.append_value(type_name_of(&geom?)),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

fn type_name_of<G: GeometryTrait<T = f64>>(geom: &G) -> &'static str {
    match geom.as_type() {
        GeometryType::Point(_) => "ST_Point",
        GeometryType::LineString(_) => "ST_LineString",
        GeometryType::Polygon(_) => "ST_Polygon",
        GeometryType::MultiPoint(_) => "ST_MultiPoint",
        GeometryType::MultiLineString(_) => "ST_MultiLineString",
        GeometryType::MultiPolygon(_) => "ST_MultiPolygon",
        GeometryType::GeometryCollection(_) => "ST_GeometryCollection",
        GeometryType::Rect(_) | GeometryType::Triangle(_) => "ST_Polygon",
        GeometryType::Line(_) => "ST_LineString",
    }
}

/// `ST_Dimension`. The topological dimension: 0 for points, 1 for lines, 2 for areas.
pub fn st_dimension(array: &dyn GeoArrowArray) -> GeoArrowResult<Int32Array> {
    if let Some(value) = static_topological_dimension(&array.data_type()) {
        return Ok(constant_i32(value, array.len(), array.logical_nulls()));
    }
    downcast_geoarrow_array!(array, dimension_impl)
}

fn static_topological_dimension(data_type: &GeoArrowType) -> Option<i32> {
    match data_type {
        GeoArrowType::Point(_) | GeoArrowType::MultiPoint(_) => Some(0),
        GeoArrowType::LineString(_) | GeoArrowType::MultiLineString(_) => Some(1),
        GeoArrowType::Polygon(_) | GeoArrowType::MultiPolygon(_) | GeoArrowType::Rect(_) => Some(2),
        _ => None,
    }
}

fn dimension_impl<'a>(array: &'a impl GeoArrowArrayAccessor<'a>) -> GeoArrowResult<Int32Array> {
    let mut builder = Int32Builder::with_capacity(array.len());
    for item in array.iter() {
        match item {
            Some(geom) => builder.append_value(topological_dimension(&geom?)),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

fn topological_dimension<G: GeometryTrait<T = f64>>(geom: &G) -> i32 {
    match geom.as_type() {
        GeometryType::Point(_) | GeometryType::MultiPoint(_) => 0,
        GeometryType::LineString(_) | GeometryType::MultiLineString(_) | GeometryType::Line(_) => 1,
        GeometryType::Polygon(_)
        | GeometryType::MultiPolygon(_)
        | GeometryType::Rect(_)
        | GeometryType::Triangle(_) => 2,
        // A collection takes the highest dimension it holds. An empty collection is 0.
        GeometryType::GeometryCollection(gc) => gc
            .geometries()
            .map(|inner| topological_dimension(&inner))
            .max()
            .unwrap_or(0),
    }
}

/// `ST_CoordDim`. The number of ordinates per coordinate: 2, 3 or 4.
pub fn st_coord_dim(array: &dyn GeoArrowArray) -> GeoArrowResult<Int32Array> {
    if let Some(dim) = array.data_type().dimension() {
        let value = i32::try_from(dim.size()).unwrap_or(2);
        return Ok(constant_i32(value, array.len(), array.logical_nulls()));
    }
    downcast_geoarrow_array!(array, coord_dim_impl)
}

fn coord_dim_impl<'a>(array: &'a impl GeoArrowArrayAccessor<'a>) -> GeoArrowResult<Int32Array> {
    let mut builder = Int32Builder::with_capacity(array.len());
    for item in array.iter() {
        match item {
            Some(geom) => builder.append_value(geom?.dim().size() as i32),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// `ST_NPoints`. Every coordinate in the geometry, at any depth.
pub fn st_npoints(array: &dyn GeoArrowArray) -> GeoArrowResult<Int32Array> {
    downcast_geoarrow_array!(array, npoints_impl)
}

fn npoints_impl<'a>(array: &'a impl GeoArrowArrayAccessor<'a>) -> GeoArrowResult<Int32Array> {
    let mut builder = Int32Builder::with_capacity(array.len());
    for item in array.iter() {
        match item {
            Some(geom) => builder.append_value(count_coords(&geom?) as i32),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

fn count_coords<G: GeometryTrait<T = f64>>(geom: &G) -> usize {
    match geom.as_type() {
        GeometryType::Point(p) => usize::from(p.coord().is_some()),
        GeometryType::LineString(ls) => ls.num_coords(),
        GeometryType::Polygon(p) => count_polygon_coords(p),
        GeometryType::MultiPoint(mp) => mp.points().filter(|p| p.coord().is_some()).count(),
        GeometryType::MultiLineString(ml) => ml.line_strings().map(|ls| ls.num_coords()).sum(),
        GeometryType::MultiPolygon(mp) => mp.polygons().map(|p| count_polygon_coords(&p)).sum(),
        GeometryType::GeometryCollection(gc) => {
            gc.geometries().map(|inner| count_coords(&inner)).sum()
        }
        GeometryType::Rect(_) => 5,
        GeometryType::Triangle(_) => 4,
        GeometryType::Line(_) => 2,
    }
}

fn count_polygon_coords<P: PolygonTrait<T = f64>>(polygon: &P) -> usize {
    polygon.exterior().map_or(0, |ring| ring.num_coords())
        + polygon
            .interiors()
            .map(|ring| ring.num_coords())
            .sum::<usize>()
}

/// `ST_NumPoints`. The coordinate count of a line string, and null for anything else.
///
/// PostGIS restricts this one to line strings. `ST_NPoints` is the version that accepts every type.
pub fn st_num_points(array: &dyn GeoArrowArray) -> GeoArrowResult<Int32Array> {
    downcast_geoarrow_array!(array, num_points_impl)
}

fn num_points_impl<'a>(array: &'a impl GeoArrowArrayAccessor<'a>) -> GeoArrowResult<Int32Array> {
    let mut builder = Int32Builder::with_capacity(array.len());
    for item in array.iter() {
        match item {
            Some(geom) => match geom?.as_type() {
                GeometryType::LineString(ls) => builder.append_value(ls.num_coords() as i32),
                GeometryType::Line(_) => builder.append_value(2),
                _ => builder.append_null(),
            },
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// `ST_NumGeometries`. One for a single geometry, the part count for a collection.
pub fn st_num_geometries(array: &dyn GeoArrowArray) -> GeoArrowResult<Int32Array> {
    downcast_geoarrow_array!(array, num_geometries_impl)
}

fn num_geometries_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
) -> GeoArrowResult<Int32Array> {
    let mut builder = Int32Builder::with_capacity(array.len());
    for item in array.iter() {
        match item {
            Some(geom) => builder.append_value(match geom?.as_type() {
                GeometryType::MultiPoint(mp) => mp.num_points() as i32,
                GeometryType::MultiLineString(ml) => ml.num_line_strings() as i32,
                GeometryType::MultiPolygon(mp) => mp.num_polygons() as i32,
                GeometryType::GeometryCollection(gc) => gc.num_geometries() as i32,
                _ => 1,
            }),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// `ST_NumInteriorRings`. The hole count of a polygon, and null for anything else.
pub fn st_num_interior_rings(array: &dyn GeoArrowArray) -> GeoArrowResult<Int32Array> {
    downcast_geoarrow_array!(array, num_interior_rings_impl)
}

fn num_interior_rings_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
) -> GeoArrowResult<Int32Array> {
    let mut builder = Int32Builder::with_capacity(array.len());
    for item in array.iter() {
        match item {
            Some(geom) => match geom?.as_type() {
                GeometryType::Polygon(p) => builder.append_value(p.num_interiors() as i32),
                GeometryType::Rect(_) | GeometryType::Triangle(_) => builder.append_value(0),
                _ => builder.append_null(),
            },
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// `ST_IsEmpty`. True when the geometry holds no coordinate.
pub fn st_is_empty(array: &dyn GeoArrowArray) -> GeoArrowResult<BooleanArray> {
    downcast_geoarrow_array!(array, is_empty_impl)
}

fn is_empty_impl<'a>(array: &'a impl GeoArrowArrayAccessor<'a>) -> GeoArrowResult<BooleanArray> {
    let mut builder = BooleanBuilder::with_capacity(array.len());
    for item in array.iter() {
        match item {
            Some(geom) => builder.append_value(count_coords(&geom?) == 0),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// `ST_IsClosed`. True when a lineal geometry starts and ends at the same coordinate.
///
/// PostGIS returns true for a point and for any areal geometry.
pub fn st_is_closed(array: &dyn GeoArrowArray) -> GeoArrowResult<BooleanArray> {
    downcast_geoarrow_array!(array, is_closed_impl)
}

fn is_closed_impl<'a>(array: &'a impl GeoArrowArrayAccessor<'a>) -> GeoArrowResult<BooleanArray> {
    let mut builder = BooleanBuilder::with_capacity(array.len());
    for item in array.iter() {
        match item {
            Some(geom) => builder.append_value(is_closed(&geom?)),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

fn is_closed<G: GeometryTrait<T = f64>>(geom: &G) -> bool {
    match geom.as_type() {
        GeometryType::LineString(ls) => ring_is_closed(ls),
        GeometryType::MultiLineString(ml) => {
            let mut any = false;
            for ls in ml.line_strings() {
                any = true;
                if !ring_is_closed(&ls) {
                    return false;
                }
            }
            any
        }
        GeometryType::Line(_) => false,
        // Points and areal geometries are closed by definition in PostGIS.
        _ => true,
    }
}

fn ring_is_closed<L: LineStringTrait<T = f64>>(line: &L) -> bool {
    let count = line.num_coords();
    if count < 2 {
        return false;
    }
    match (line.coord(0), line.coord(count - 1)) {
        (Some(first), Some(last)) => first.x() == last.x() && first.y() == last.y(),
        _ => false,
    }
}

/// `ST_IsRing`. True when a line string is closed and does not cross itself.
///
/// Null for every type other than a line string, as in PostGIS.
pub fn st_is_ring(array: &dyn GeoArrowArray) -> GeoArrowResult<BooleanArray> {
    downcast_geoarrow_array!(array, is_ring_impl)
}

fn is_ring_impl<'a>(array: &'a impl GeoArrowArrayAccessor<'a>) -> GeoArrowResult<BooleanArray> {
    let mut builder = BooleanBuilder::with_capacity(array.len());
    for item in array.iter() {
        let Some(geom) = item else {
            builder.append_null();
            continue;
        };
        let geom = geom?;
        match geom.as_type() {
            GeometryType::LineString(ls) => {
                // The cheap test first. Most non-rings fail here and never reach the sweep.
                if !ring_is_closed(ls) {
                    builder.append_value(false);
                    continue;
                }
                match geom.to_geometry() {
                    geo::Geometry::LineString(line) => {
                        builder.append_value(line_string_is_simple(&line))
                    }
                    _ => builder.append_null(),
                }
            }
            _ => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// `ST_IsSimple`. True when the geometry has no anomalous point such as a self intersection.
///
/// Follows the JTS definition. Areal geometries are always simple, since self intersection there
/// is a validity question and not a simplicity one. Use `ST_IsValid` for that.
pub fn st_is_simple(array: &dyn GeoArrowArray) -> GeoArrowResult<BooleanArray> {
    if all_null(array) {
        return Ok(BooleanArray::new_null(array.len()));
    }
    let mut reader = GeometryReader::new(array)?;
    let mut builder = BooleanBuilder::with_capacity(array.len());
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => builder.append_value(is_simple(geom)),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

fn is_simple(geom: &geo::Geometry<f64>) -> bool {
    use geo::Geometry::*;
    match geom {
        Point(_) | Triangle(_) | Rect(_) | Polygon(_) | MultiPolygon(_) => true,
        Line(_) => true,
        // A multi point is simple when no coordinate repeats.
        MultiPoint(mp) => {
            let mut seen: Vec<(u64, u64)> = mp
                .iter()
                .map(|p| (p.x().to_bits(), p.y().to_bits()))
                .collect();
            seen.sort_unstable();
            let before = seen.len();
            seen.dedup();
            seen.len() == before
        }
        LineString(ls) => line_string_is_simple(ls),
        MultiLineString(ml) => {
            // Each part must be simple, and the parts must not cross one another.
            ml.iter().all(line_string_is_simple)
                && segments_are_simple(ml.iter().flat_map(|ls| ls.lines()).collect())
        }
        GeometryCollection(gc) => gc.iter().all(is_simple),
    }
}

/// A line string is simple when its segments meet only where they must.
///
/// Uses the Bentley-Ottmann sweep from `geo`, so the cost is `O(n log n)` and not `O(n squared)`.
fn line_string_is_simple(line: &LineString<f64>) -> bool {
    let closed = line.is_closed();
    let lines: Vec<Line<f64>> = line.lines().collect();
    let count = lines.len();
    if count < 2 {
        return true;
    }

    let indexed = lines.into_iter().enumerate().map(Indexed);
    for (Indexed((i, _)), Indexed((j, _)), _) in Intersections::from_iter(indexed) {
        if !allowed_touch(i, j, count, closed) {
            return false;
        }
    }
    true
}

/// Segments from different parts may not touch at all beyond shared endpoints.
fn segments_are_simple(lines: Vec<Line<f64>>) -> bool {
    if lines.len() < 2 {
        return true;
    }
    Intersections::from_iter(lines).next().is_none()
}

/// Two segments of one line string may meet when they are neighbours, or when they are the first
/// and last segment of a closed ring.
fn allowed_touch(i: usize, j: usize, count: usize, closed: bool) -> bool {
    let (low, high) = if i < j { (i, j) } else { (j, i) };
    if high - low == 1 {
        return true;
    }
    closed && low == 0 && high == count - 1
}

/// Carries the segment position through the sweep so neighbours can be told apart.
///
/// Two segments of a line string may touch when they are next to each other. The sweep reports
/// every pair that meets. The position tells an allowed touch from a cross.
#[derive(Debug, Clone, Copy)]
struct Indexed((usize, Line<f64>));

impl geo::sweep::Cross for Indexed {
    type Scalar = f64;

    fn line(&self) -> Line<f64> {
        self.0 .1
    }
}

fn constant_strings(value: &str, len: usize, nulls: Option<NullBuffer>) -> StringArray {
    let mut builder = StringBuilder::with_capacity(len, len * value.len());
    match &nulls {
        None => (0..len).for_each(|_| builder.append_value(value)),
        Some(nulls) => (0..len).for_each(|index| {
            if nulls.is_valid(index) {
                builder.append_value(value);
            } else {
                builder.append_null();
            }
        }),
    }
    builder.finish()
}

fn constant_i32(value: i32, len: usize, nulls: Option<NullBuffer>) -> Int32Array {
    Int32Array::new(vec![value; len].into(), nulls)
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use geoarrow_array::builder::{
        GeometryBuilder, LineStringBuilder, PointBuilder, PolygonBuilder,
    };
    use geoarrow_schema::{
        CoordType, GeometryType as GeoGeometryType, LineStringType, PolygonType,
    };

    use super::*;

    fn point_type(coord_type: CoordType, dim: Dimension) -> geoarrow_schema::PointType {
        geoarrow_schema::PointType::new(dim, Default::default()).with_coord_type(coord_type)
    }

    fn sample_points(coord_type: CoordType) -> PointArray {
        let p0 = geo::point!(x: 1.0, y: 2.0);
        let p1 = geo::point!(x: 3.0, y: 4.0);
        PointBuilder::from_nullable_points(
            [Some(&p0), None, Some(&p1)].into_iter(),
            point_type(coord_type, Dimension::XY),
        )
        .finish()
    }

    /// A mixed array with one geometry of each shape under test.
    fn mixed() -> geoarrow_array::array::GeometryArray {
        let mut builder = GeometryBuilder::new(GeoGeometryType::new(Default::default()));
        for geom in sample_geometries() {
            builder.push_geometry(Some(&geom)).unwrap();
        }
        builder.finish()
    }

    fn sample_geometries() -> Vec<geo::Geometry<f64>> {
        vec![
            geo::wkt! { POINT(1.0 2.0) }.into(),
            geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0,2.0 0.0) }.into(),
            geo::wkt! { POLYGON((0.0 0.0,4.0 0.0,4.0 4.0,0.0 4.0,0.0 0.0),(1.0 1.0,2.0 1.0,2.0 2.0,1.0 1.0)) }.into(),
            geo::wkt! { MULTIPOINT(0.0 0.0,5.0 5.0) }.into(),
        ]
    }

    #[test]
    fn st_x_reads_the_x_buffer() {
        for coord_type in [CoordType::Separated, CoordType::Interleaved] {
            let array = sample_points(coord_type);
            let x = st_x(&array).unwrap();
            let y = st_y(&array).unwrap();

            assert_eq!(x.value(0), 1.0);
            assert!(x.is_null(1));
            assert_eq!(x.value(2), 3.0);
            assert_eq!(y.value(0), 2.0);
        }
    }

    /// The zero copy claim, proven rather than assumed.
    #[test]
    fn st_x_is_zero_copy_on_separated_coords() {
        let array = sample_points(CoordType::Separated);
        let CoordBuffer::Separated(coords) = array.coords() else {
            panic!("expected separated coordinates");
        };

        let x = st_x(&array).unwrap();
        assert_eq!(x.values().as_ptr(), coords.raw_buffers()[0].as_ptr());
        let y = st_y(&array).unwrap();
        assert_eq!(y.values().as_ptr(), coords.raw_buffers()[1].as_ptr());
    }

    #[test]
    fn st_z_is_null_in_two_dimensions() {
        let array = sample_points(CoordType::Separated);
        let z = st_z(&array).unwrap();
        assert_eq!(z.len(), 3);
        assert!((0..3).all(|i| z.is_null(i)));
        assert!((0..3).all(|i| st_m(&array).unwrap().is_null(i)));
    }

    #[test]
    fn st_z_reads_the_third_buffer() {
        for coord_type in [CoordType::Separated, CoordType::Interleaved] {
            // XYZ points are built through the geometry builder, which keeps the third ordinate.
            let mut builder = PointBuilder::new(point_type(coord_type, Dimension::XYZ));
            builder.push_coord(Some(&XyzCoord(1.0, 2.0, 3.0)));
            builder.push_coord(Some(&XyzCoord(4.0, 5.0, 6.0)));
            let array = builder.finish();

            let z = st_z(&array).unwrap();
            assert_eq!(z.value(0), 3.0);
            assert_eq!(z.value(1), 6.0);
            // The measure is absent from XYZ.
            assert!(st_m(&array).unwrap().is_null(0));
        }
    }

    /// A minimal three-dimensional coordinate for the builder.
    struct XyzCoord(f64, f64, f64);

    impl CoordTrait for XyzCoord {
        type T = f64;

        fn dim(&self) -> Dimensions {
            Dimensions::Xyz
        }

        fn x(&self) -> f64 {
            self.0
        }

        fn y(&self) -> f64 {
            self.1
        }

        fn nth_or_panic(&self, n: usize) -> f64 {
            match n {
                0 => self.0,
                1 => self.1,
                2 => self.2,
                _ => panic!("XYZ has three ordinates"),
            }
        }
    }

    #[test]
    fn st_x_rejects_a_line_string_array() {
        let lines: Vec<geo::LineString<f64>> = vec![geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0) }];
        let array = LineStringBuilder::from_line_strings(
            &lines,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();

        assert!(st_x(&array).is_err());
        assert!(!accepts_ordinate(&array.data_type()));
    }

    #[test]
    fn geometry_type_is_constant_for_a_typed_array() {
        let array = sample_points(CoordType::Separated);
        let names = st_geometry_type(&array).unwrap();
        assert_eq!(names.value(0), "ST_Point");
        assert!(names.is_null(1), "a null row stays null");
        assert_eq!(names.value(2), "ST_Point");
    }

    #[test]
    fn geometry_type_varies_in_a_mixed_array() {
        let array = mixed();
        let names = st_geometry_type(&array).unwrap();
        assert_eq!(names.value(0), "ST_Point");
        assert_eq!(names.value(1), "ST_LineString");
        assert_eq!(names.value(2), "ST_Polygon");
        assert_eq!(names.value(3), "ST_MultiPoint");
    }

    #[test]
    fn dimension_and_coord_dim() {
        let array = mixed();
        let dims = st_dimension(&array).unwrap();
        assert_eq!(dims.value(0), 0, "point");
        assert_eq!(dims.value(1), 1, "line string");
        assert_eq!(dims.value(2), 2, "polygon");
        assert_eq!(dims.value(3), 0, "multi point");

        let coord_dims = st_coord_dim(&array).unwrap();
        assert!((0..4).all(|i| coord_dims.value(i) == 2));
    }

    #[test]
    fn counts_match_postgis() {
        let array = mixed();

        let npoints = st_npoints(&array).unwrap();
        assert_eq!(npoints.value(0), 1, "point");
        assert_eq!(npoints.value(1), 3, "line string");
        assert_eq!(npoints.value(2), 9, "polygon rings, 5 plus 4");
        assert_eq!(npoints.value(3), 2, "multi point");

        let num_points = st_num_points(&array).unwrap();
        assert!(num_points.is_null(0), "ST_NumPoints is line string only");
        assert_eq!(num_points.value(1), 3);
        assert!(num_points.is_null(2));

        let num_geoms = st_num_geometries(&array).unwrap();
        assert_eq!(num_geoms.value(0), 1);
        assert_eq!(num_geoms.value(1), 1);
        assert_eq!(num_geoms.value(3), 2, "multi point has two parts");

        let rings = st_num_interior_rings(&array).unwrap();
        assert!(rings.is_null(0), "not a polygon");
        assert_eq!(rings.value(2), 1, "one hole");
    }

    #[test]
    fn is_empty_and_is_closed() {
        let array = mixed();

        let empty = st_is_empty(&array).unwrap();
        assert!((0..4).all(|i| !empty.value(i)));

        let closed = st_is_closed(&array).unwrap();
        assert!(closed.value(0), "a point is closed");
        assert!(!closed.value(1), "an open line string is not");
        assert!(closed.value(2), "a polygon is closed");
    }

    #[test]
    fn is_ring_needs_closed_and_simple() {
        let rings: Vec<geo::LineString<f64>> = vec![
            // Closed and simple.
            geo::wkt! { LINESTRING(0.0 0.0,1.0 0.0,1.0 1.0,0.0 0.0) },
            // Closed but crosses itself, a bow tie.
            geo::wkt! { LINESTRING(0.0 0.0,2.0 2.0,2.0 0.0,0.0 2.0,0.0 0.0) },
            // Simple but open.
            geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0) },
        ];
        let array = LineStringBuilder::from_line_strings(
            &rings,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let is_ring = st_is_ring(&array).unwrap();
        assert!(is_ring.value(0), "closed and simple");
        assert!(!is_ring.value(1), "self crossing");
        assert!(!is_ring.value(2), "open");

        // ST_IsRing returns null for every type other than a line string.
        let points = sample_points(CoordType::Separated);
        assert!(st_is_ring(&points).unwrap().is_null(0));
    }

    #[test]
    fn is_simple_finds_self_crossings() {
        let lines: Vec<geo::LineString<f64>> = vec![
            geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0,2.0 0.0) },
            geo::wkt! { LINESTRING(0.0 0.0,2.0 2.0,2.0 0.0,0.0 2.0) },
            geo::wkt! { LINESTRING(0.0 0.0,1.0 0.0,1.0 1.0,0.0 0.0) },
        ];
        let array = LineStringBuilder::from_line_strings(
            &lines,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let simple = st_is_simple(&array).unwrap();
        assert!(simple.value(0), "a plain open line is simple");
        assert!(!simple.value(1), "the crossing line is not");
        assert!(simple.value(2), "a closed ring is simple");
    }

    #[test]
    fn is_simple_on_multi_point_finds_duplicates() {
        let mut builder = GeometryBuilder::new(GeoGeometryType::new(Default::default()));
        builder
            .push_geometry(Some(&geo::Geometry::<f64>::from(
                geo::wkt! { MULTIPOINT(0.0 0.0,1.0 1.0) },
            )))
            .unwrap();
        builder
            .push_geometry(Some(&geo::Geometry::<f64>::from(
                geo::wkt! { MULTIPOINT(0.0 0.0,0.0 0.0) },
            )))
            .unwrap();
        let array = builder.finish();

        let simple = st_is_simple(&array).unwrap();
        assert!(simple.value(0));
        assert!(!simple.value(1), "a repeated point is not simple");
    }

    #[test]
    fn polygons_are_always_simple() {
        let squares: Vec<geo::Polygon<f64>> =
            vec![geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 1.0,0.0 0.0)) }];
        let array = PolygonBuilder::from_polygons(
            &squares,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        assert!(st_is_simple(&array).unwrap().value(0));
    }
}
