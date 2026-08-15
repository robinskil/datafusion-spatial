//! Structure edits.
//!
//! These change what a geometry *is*, not where it sits. They promote a geometry to a multi
//! type, add or drop a vertex, round coordinates onto a grid, or split a collection into parts.
//!
//! # `ST_Dump` returns a list of WKB, not a set of geometries
//!
//! In PostGIS, `ST_Dump` returns a set. One input row becomes many output rows.
//! A DataFusion scalar function cannot return a set. So this returns a list.
//! Call `unnest` to expand the list:
//!
//! ```sql
//! SELECT ST_AsText(unnest(ST_Dump(geom))) AS part FROM shapes
//! ```
//!
//! The parts are WKB, not GeoArrow geometries. This crate has no choice here.
//! GeoArrow marks a column as spatial through the field metadata. The DataFusion `unnest` step
//! drops the metadata on the child field of the list. A list of geometries then arrives as a
//! plain struct. No spatial function accepts it.
//!
//! WKB avoids the problem. Geoarrow reads a plain `Binary` column as WKB, so the parts stay
//! usable without a cast. The cost is one encode per part.

use std::sync::Arc;

use crate::materialize::GeometryReader;
use arrow_array::builder::{ArrayBuilder, Int32Builder, ListBuilder};
use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, ListArray};
use arrow_buffer::NullBuffer;
use arrow_schema::FieldRef;
use geo::{Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon};
use geoarrow_array::builder::GeometryBuilder;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::{GeoArrowError, GeoArrowResult};
use geoarrow_schema::{GeoArrowType, GeometryType};

/// A one-argument structure edit that takes no extra parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Structure {
    /// `ST_Multi`. Promote a single geometry to its multi form.
    Multi,
    /// `ST_Points`. Every coordinate as one multi point.
    Points,
}

impl Structure {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::Multi => "ST_Multi",
            Self::Points => "ST_Points",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::Multi => "st_multi",
            Self::Points => "st_points",
        }
    }

    /// Every structure edit, for registration.
    pub const ALL: [Self; 2] = [Self::Multi, Self::Points];

    /// Apply this edit to one geometry.
    pub fn apply(self, geom: &Geometry<f64>) -> Geometry<f64> {
        use geo::CoordsIter;

        match self {
            Self::Multi => match geom {
                Geometry::Point(point) => Geometry::MultiPoint(MultiPoint::new(vec![*point])),
                Geometry::LineString(line) => {
                    Geometry::MultiLineString(MultiLineString::new(vec![line.clone()]))
                }
                Geometry::Polygon(polygon) => {
                    Geometry::MultiPolygon(MultiPolygon::new(vec![polygon.clone()]))
                }
                Geometry::Rect(rect) => {
                    Geometry::MultiPolygon(MultiPolygon::new(vec![rect.to_polygon()]))
                }
                Geometry::Triangle(triangle) => {
                    Geometry::MultiPolygon(MultiPolygon::new(vec![triangle.to_polygon()]))
                }
                Geometry::Line(line) => Geometry::MultiLineString(MultiLineString::new(vec![
                    LineString::new(vec![line.start, line.end]),
                ])),
                // Already a multi form or a collection.
                other => other.clone(),
            },
            Self::Points => Geometry::MultiPoint(MultiPoint::new(
                geom.coords_iter().map(geo::Point::from).collect(),
            )),
        }
    }
}

/// Any structure edit over an array.
pub fn structure(
    array: &dyn GeoArrowArray,
    edit: Structure,
    output: GeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let mut reader = GeometryReader::new(array)?;
    let mut builder = GeometryBuilder::new(output);
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => builder.push_geometry(Some(&edit.apply(geom)))?,
            None => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// `ST_SnapToGrid`. Round every coordinate onto a grid of the given size.
///
/// PostGIS also has an origin and per-axis form. This is the single-size version, which is the one
/// used to shrink a geometry before storage or comparison.
pub fn st_snap_to_grid(
    array: &dyn GeoArrowArray,
    size: &Float64Array,
    output: GeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    snap_to_grid_rows(array, size, output)
}

fn snap_to_grid_rows(
    array: &dyn GeoArrowArray,
    size: &Float64Array,
    output: GeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    use geo::MapCoords;

    let broadcast = size.len() == 1 && array.len() != 1;
    let mut builder = GeometryBuilder::new(output);

    let mut reader = GeometryReader::new(array)?;
    for row in 0..array.len() {
        let slot = if broadcast { 0 } else { row };
        // A grid of zero or less has no meaning, and would divide by zero.
        if slot >= size.len() || size.is_null(slot) || size.value(slot) <= 0.0 {
            builder.push_null();
            continue;
        }
        let Some(geom) = reader.read(row)? else {
            builder.push_null();
            continue;
        };
        let grid = size.value(slot);
        let snapped = geom.map_coords(|coord| {
            geo::coord! {
                x: (coord.x / grid).round() * grid,
                y: (coord.y / grid).round() * grid,
            }
        });
        builder.push_geometry(Some(&snapped))?;
    }
    Ok(Arc::new(builder.finish()))
}

/// Which vertex edit to apply to a line string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexEdit {
    /// `ST_AddPoint`. Append a point, or insert it at a position.
    Add,
    /// `ST_RemovePoint`. Drop the vertex at a position.
    Remove,
    /// `ST_SetPoint`. Replace the vertex at a position.
    Set,
}

impl VertexEdit {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::Add => "ST_AddPoint",
            Self::Remove => "ST_RemovePoint",
            Self::Set => "ST_SetPoint",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::Add => "st_addpoint",
            Self::Remove => "st_removepoint",
            Self::Set => "st_setpoint",
        }
    }

    /// Every vertex edit, for registration.
    pub const ALL: [Self; 3] = [Self::Add, Self::Remove, Self::Set];

    /// True when the edit needs a point argument.
    pub const fn needs_point(self) -> bool {
        matches!(self, Self::Add | Self::Set)
    }
}

/// `ST_AddPoint`, `ST_RemovePoint` or `ST_SetPoint`.
///
/// The position is zero-based, as PostGIS defines it for these three. A position outside the line
/// gives null. `ST_AddPoint` with a null position appends.
pub fn vertex_edit(
    array: &dyn GeoArrowArray,
    edit: VertexEdit,
    point: Option<&dyn GeoArrowArray>,
    position: Option<&Int32Array>,
    output: GeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let len = array.len();
    if let Some(point) = point {
        if point.len() != len && point.len() != 1 {
            return Err(GeoArrowError::InvalidGeoArrow(format!(
                "{} needs matching lengths, got {len} and {}",
                edit.function_name(),
                point.len()
            )));
        }
    }

    let mut points = match point {
        Some(array) => Some(crate::predicate::Operand::new(array, len)?),
        None => None,
    };
    let mut lines = crate::predicate::Operand::new(array, len)?;

    let mut builder = GeometryBuilder::new(output);
    for index in 0..len {
        let Some(geom) = lines.get(index)? else {
            builder.push_null();
            continue;
        };
        let Geometry::LineString(line) = geom else {
            // PostGIS restricts these three to line strings.
            builder.push_null();
            continue;
        };

        let at = match position {
            Some(values) => {
                let slot = if values.len() == 1 && len != 1 {
                    0
                } else {
                    index
                };
                if values.is_null(slot) {
                    None
                } else {
                    Some(values.value(slot))
                }
            }
            None => None,
        };

        let new_point = match &mut points {
            Some(operand) => match operand.get(index)? {
                Some(Geometry::Point(point)) => Some(*point),
                _ => {
                    builder.push_null();
                    continue;
                }
            },
            None => None,
        };

        match apply_vertex_edit(line, edit, new_point, at) {
            Some(result) => builder.push_geometry(Some(&Geometry::LineString(result)))?,
            None => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn apply_vertex_edit(
    line: &LineString<f64>,
    edit: VertexEdit,
    point: Option<geo::Point<f64>>,
    position: Option<i32>,
) -> Option<LineString<f64>> {
    let mut coords = line.0.clone();

    match edit {
        VertexEdit::Add => {
            let point = point?;
            let at = match position {
                // A negative position, or none at all, appends.
                None => coords.len(),
                Some(value) if value < 0 => coords.len(),
                Some(value) => {
                    let at = usize::try_from(value).ok()?;
                    // PostGIS allows one past the end, which appends.
                    if at > coords.len() {
                        return None;
                    }
                    at
                }
            };
            coords.insert(at, point.into());
        }
        VertexEdit::Remove => {
            let at = usize::try_from(position?).ok()?;
            if at >= coords.len() {
                return None;
            }
            // A line string needs at least two coordinates.
            if coords.len() <= 2 {
                return None;
            }
            coords.remove(at);
        }
        VertexEdit::Set => {
            let point = point?;
            let at = usize::try_from(position?).ok()?;
            if at >= coords.len() {
                return None;
            }
            coords[at] = point.into();
        }
    }

    Some(LineString::new(coords))
}

/// The Arrow field of the list `ST_Dump` produces.
///
/// A WKB part, not a GeoArrow geometry. See the module documentation for why.
pub fn dump_field(_input: &GeoArrowType) -> FieldRef {
    Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Binary,
        true,
    ))
}

/// `ST_Dump`. Split every geometry into its parts.
///
/// Returns a list per row. See the module documentation for why this is not a set.
pub fn st_dump(array: &dyn GeoArrowArray, part_field: FieldRef) -> GeoArrowResult<ListArray> {
    dump_rows(array, part_field)
}

fn dump_rows(array: &dyn GeoArrowArray, part_field: FieldRef) -> GeoArrowResult<ListArray> {
    use arrow_array::builder::BinaryBuilder;

    // Flatten every row's parts into one child array, and record where each row ends.
    let mut parts = BinaryBuilder::new();
    let mut offsets: Vec<i32> = Vec::with_capacity(array.len() + 1);
    let mut valid = Vec::with_capacity(array.len());
    let mut total: i32 = 0;
    offsets.push(0);

    let mut reader = GeometryReader::new(array)?;
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => {
                for part in explode(geom) {
                    parts.append_value(crate::aggregate::write_wkb(&part)?);
                    total += 1;
                }
                valid.push(true);
            }
            None => valid.push(false),
        }
        offsets.push(total);
    }

    let values: ArrayRef = Arc::new(parts.finish());
    let nulls = valid.iter().any(|ok| !ok).then(|| NullBuffer::from(valid));

    ListArray::try_new(
        part_field,
        arrow_buffer::OffsetBuffer::new(offsets.into()),
        values,
        nulls,
    )
    .map_err(GeoArrowError::Arrow)
}

/// The parts of a geometry, one level deep, as PostGIS defines the dump.
fn explode(geom: &Geometry<f64>) -> Vec<Geometry<f64>> {
    match geom {
        Geometry::MultiPoint(points) => points.iter().cloned().map(Geometry::Point).collect(),
        Geometry::MultiLineString(lines) => {
            lines.iter().cloned().map(Geometry::LineString).collect()
        }
        Geometry::MultiPolygon(polygons) => {
            polygons.iter().cloned().map(Geometry::Polygon).collect()
        }
        Geometry::GeometryCollection(parts) => parts.iter().cloned().collect(),
        single => vec![single.clone()],
    }
}

/// Silences the unused-import lint for builders only some profiles need.
#[allow(dead_code)]
fn _unused(_: ListBuilder<Int32Builder>) -> usize {
    ArrayBuilder::len(&Int32Builder::new())
}

#[cfg(test)]
mod tests {
    use geo_traits::to_geo::ToGeoGeometry;
    use geoarrow_array::builder::{GeometryBuilder as GeoBuilder, LineStringBuilder, PointBuilder};
    use geoarrow_array::cast::AsGeoArrowArray;
    use geoarrow_array::GeoArrowArrayAccessor;
    use geoarrow_schema::{Dimension, LineStringType, PointType};

    use super::*;

    fn mixed(values: Vec<Geometry<f64>>) -> geoarrow_array::array::GeometryArray {
        let mut builder = GeoBuilder::new(GeometryType::new(Default::default()));
        for geom in values {
            builder.push_geometry(Some(&geom)).unwrap();
        }
        builder.finish()
    }

    fn read(array: &dyn GeoArrowArray, row: usize) -> Option<Geometry<f64>> {
        array
            .as_geometry()
            .get(row)
            .unwrap()
            .map(|geom| geom.to_geometry())
    }

    fn output() -> GeometryType {
        GeometryType::new(Default::default())
    }

    #[test]
    fn multi_promotes_a_single_geometry() {
        let array = mixed(vec![
            geo::wkt! { POINT(1.0 2.0) }.into(),
            geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0) }.into(),
            geo::wkt! { MULTIPOINT(0.0 0.0,1.0 1.0) }.into(),
        ]);
        let result = structure(&array, Structure::Multi, output()).unwrap();

        assert!(matches!(
            read(result.as_ref(), 0),
            Some(Geometry::MultiPoint(_))
        ));
        assert!(matches!(
            read(result.as_ref(), 1),
            Some(Geometry::MultiLineString(_))
        ));
        // Already multi, so unchanged.
        assert!(matches!(
            read(result.as_ref(), 2),
            Some(Geometry::MultiPoint(_))
        ));
    }

    /// The GeoArrow mixed encoding has no one-element collection.
    ///
    /// `GeometryBuilder` unwraps a `GEOMETRYCOLLECTION` of one part back to that part, so
    /// `ST_ForceCollection` could only ever be a no-op here. It is not registered for that
    /// reason; this test pins the behaviour down so the decision is revisited if it changes.
    #[test]
    fn a_one_element_collection_is_flattened_by_the_encoding() {
        let collection = Geometry::GeometryCollection(geo::GeometryCollection::new_from(vec![
            Geometry::<f64>::from(geo::wkt! { POINT(1.0 2.0) }),
        ]));
        let array = mixed(vec![collection]);
        assert!(
            matches!(read(array_ref(&array), 0), Some(Geometry::Point(_))),
            "a one element collection reads back as the bare geometry"
        );

        // Two parts survive, so the collection type itself works.
        let two = Geometry::GeometryCollection(geo::GeometryCollection::new_from(vec![
            Geometry::<f64>::from(geo::wkt! { POINT(1.0 2.0) }),
            Geometry::<f64>::from(geo::wkt! { POINT(3.0 4.0) }),
        ]));
        let array = mixed(vec![two]);
        assert!(matches!(
            read(array_ref(&array), 0),
            Some(Geometry::GeometryCollection(_))
        ));
    }

    fn array_ref(array: &geoarrow_array::array::GeometryArray) -> &dyn GeoArrowArray {
        array
    }

    #[test]
    fn points_gathers_every_coordinate() {
        let array = mixed(vec![
            geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 0.0)) }.into(),
        ]);
        let result = structure(&array, Structure::Points, output()).unwrap();
        let Some(Geometry::MultiPoint(points)) = read(result.as_ref(), 0) else {
            panic!("expected a multi point")
        };
        assert_eq!(points.0.len(), 4);
    }

    #[test]
    fn snap_to_grid_rounds_coordinates() {
        let array = mixed(vec![geo::wkt! { POINT(1.234 5.678) }.into()]);
        let size = Float64Array::from(vec![0.5]);
        let result = st_snap_to_grid(&array, &size, output()).unwrap();

        let Some(Geometry::Point(point)) = read(result.as_ref(), 0) else {
            panic!("expected a point")
        };
        assert_eq!((point.x(), point.y()), (1.0, 5.5));
    }

    #[test]
    fn snap_to_grid_rejects_a_non_positive_size() {
        let array = mixed(vec![geo::wkt! { POINT(1.0 2.0) }.into()]);
        for bad in [0.0, -1.0] {
            let size = Float64Array::from(vec![bad]);
            let result = st_snap_to_grid(&array, &size, output()).unwrap();
            assert!(read(result.as_ref(), 0).is_none(), "{bad} must give null");
        }
    }

    fn line() -> geoarrow_array::array::LineStringArray {
        let lines: Vec<LineString<f64>> = vec![geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0,2.0 2.0) }];
        LineStringBuilder::from_line_strings(
            &lines,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish()
    }

    fn point_array(x: f64, y: f64) -> geoarrow_array::array::PointArray {
        PointBuilder::from_points(
            [geo::point!(x: x, y: y)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish()
    }

    #[test]
    fn add_point_appends_by_default() {
        let array = line();
        let point = point_array(9.0, 9.0);
        let result = vertex_edit(&array, VertexEdit::Add, Some(&point), None, output()).unwrap();

        let Some(Geometry::LineString(edited)) = read(result.as_ref(), 0) else {
            panic!("expected a line string")
        };
        assert_eq!(edited.0.len(), 4);
        assert_eq!((edited.0[3].x, edited.0[3].y), (9.0, 9.0));
    }

    #[test]
    fn add_point_inserts_at_a_position() {
        let array = line();
        let point = point_array(9.0, 9.0);
        let at = Int32Array::from(vec![1]);
        let result =
            vertex_edit(&array, VertexEdit::Add, Some(&point), Some(&at), output()).unwrap();

        let Some(Geometry::LineString(edited)) = read(result.as_ref(), 0) else {
            panic!("expected a line string")
        };
        assert_eq!((edited.0[1].x, edited.0[1].y), (9.0, 9.0));
        assert_eq!(edited.0.len(), 4);
    }

    #[test]
    fn remove_and_set_point() {
        let array = line();
        let at = Int32Array::from(vec![1]);

        let removed = vertex_edit(&array, VertexEdit::Remove, None, Some(&at), output()).unwrap();
        let Some(Geometry::LineString(edited)) = read(removed.as_ref(), 0) else {
            panic!("expected a line string")
        };
        assert_eq!(edited.0.len(), 2);
        assert_eq!((edited.0[1].x, edited.0[1].y), (2.0, 2.0));

        let point = point_array(7.0, 7.0);
        let set = vertex_edit(&array, VertexEdit::Set, Some(&point), Some(&at), output()).unwrap();
        let Some(Geometry::LineString(edited)) = read(set.as_ref(), 0) else {
            panic!("expected a line string")
        };
        assert_eq!((edited.0[1].x, edited.0[1].y), (7.0, 7.0));
    }

    #[test]
    fn a_position_outside_the_line_gives_null() {
        let array = line();
        let at = Int32Array::from(vec![99]);
        let removed = vertex_edit(&array, VertexEdit::Remove, None, Some(&at), output()).unwrap();
        assert!(read(removed.as_ref(), 0).is_none());
    }

    /// A line string needs two coordinates, so the last removal is refused.
    #[test]
    fn remove_will_not_collapse_a_line() {
        let lines: Vec<LineString<f64>> = vec![geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0) }];
        let array = LineStringBuilder::from_line_strings(
            &lines,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let at = Int32Array::from(vec![0]);
        let removed = vertex_edit(&array, VertexEdit::Remove, None, Some(&at), output()).unwrap();
        assert!(read(removed.as_ref(), 0).is_none());
    }

    #[test]
    fn dump_splits_a_collection() {
        let array = mixed(vec![
            geo::wkt! { MULTIPOINT(0.0 0.0,1.0 1.0,2.0 2.0) }.into(),
            geo::wkt! { POINT(9.0 9.0) }.into(),
        ]);
        let field = dump_field(&GeoArrowArray::data_type(&array));
        let dumped = st_dump(&array, field.clone()).unwrap();

        assert_eq!(dumped.len(), 2);
        assert_eq!(dumped.value_length(0), 3, "three parts");
        assert_eq!(dumped.value_length(1), 1, "a single geometry is one part");

        // The parts are WKB, which geoarrow reads back without any metadata.
        let parts = dumped.value(0);
        let wkb_field = arrow_schema::Field::new("item", arrow_schema::DataType::Binary, true);
        let geo_parts =
            geoarrow_array::array::from_arrow_array(parts.as_ref(), &wkb_field).unwrap();
        assert_eq!(geo_parts.len(), 3);
    }

    #[test]
    fn dump_keeps_null_rows_null() {
        let mut builder = GeoBuilder::new(GeometryType::new(Default::default()));
        builder
            .push_geometry(Some(&Geometry::<f64>::from(
                geo::wkt! { MULTIPOINT(0.0 0.0,1.0 1.0) },
            )))
            .unwrap();
        builder.push_null();
        let array = builder.finish();

        let field = dump_field(&GeoArrowArray::data_type(&array));
        let dumped = st_dump(&array, field).unwrap();
        assert!(!dumped.is_null(0));
        assert!(dumped.is_null(1));
    }
}
