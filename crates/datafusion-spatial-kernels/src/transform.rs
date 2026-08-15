//! Coordinate transforms that keep the geometry structure.
//!
//! `ST_FlipCoordinates`, `ST_Force2D` and `ST_Force3D` change coordinates. They never change the
//! ring structure of a geometry. The offsets stay valid, so the output reuses them.
//!
//! # Why these are nearly free
//!
//! A GeoArrow native array is a coordinate buffer plus offset buffers. Every array type exposes
//! both, so a transform rebuilds the array around a new coordinate buffer and clones the offsets.
//! An offset clone is an atomic counter bump.
//!
//! With separated coordinates the coordinate work often disappears too:
//!
//! | Transform | Separated coordinates | Interleaved coordinates |
//! |---|---|---|
//! | `ST_FlipCoordinates` | swap two buffer handles, no copy | copy with a stride |
//! | `ST_Force2D` from XYZ | drop a buffer handle, no copy | copy with a stride |
//! | `ST_Force3D` from XY | one new buffer of zeros | copy with a stride |

use std::sync::Arc;

use arrow_buffer::ScalarBuffer;
use geoarrow_array::array::{
    CoordBuffer, GeometryCollectionArray, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, SeparatedCoordBuffer,
};
use geoarrow_array::cast::AsGeoArrowArray;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::{GeoArrowError, GeoArrowResult};
use geoarrow_schema::{Dimension, GeoArrowType};

/// `ST_FlipCoordinates`. Swap x and y in every coordinate.
pub fn st_flip_coordinates(array: &dyn GeoArrowArray) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    map_coords("ST_FlipCoordinates", array, flip)
}

/// `ST_Force2D`. Drop the z and m ordinates.
pub fn st_force_2d(array: &dyn GeoArrowArray) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    force_dimension("ST_Force2D", array, Dimension::XY)
}

/// `ST_Force3D`. Add a z ordinate of zero where one is missing.
pub fn st_force_3d(array: &dyn GeoArrowArray) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    force_dimension("ST_Force3D", array, Dimension::XYZ)
}

/// The GeoArrow type this transform produces from a given input type.
///
/// The planner calls this so the output field carries the right extension metadata.
pub fn flipped_type(input: &GeoArrowType) -> GeoArrowType {
    input.clone()
}

/// The GeoArrow type `ST_Force2D` or `ST_Force3D` produces.
pub fn forced_type(input: &GeoArrowType, target: Dimension) -> GeoArrowResult<GeoArrowType> {
    if input.dimension().is_none() {
        return Err(unsupported(if target == Dimension::XY {
            "ST_Force2D"
        } else {
            "ST_Force3D"
        }));
    }
    Ok(input.clone().with_dimension(target))
}

fn unsupported(function: &str) -> GeoArrowError {
    GeoArrowError::IncorrectGeometryType(format!(
        "{function} needs a native GeoArrow array with a known dimension. \
         Cast a WKB, WKT or mixed geometry column first."
    ))
}

fn force_dimension(
    function: &str,
    array: &dyn GeoArrowArray,
    target: Dimension,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let Some(source) = array.data_type().dimension() else {
        return Err(unsupported(function));
    };
    if source == target {
        // Nothing to do. Hand back the input untouched.
        return Ok(array.slice(0, array.len()));
    }
    map_coords(function, array, move |coords| resize(coords, target))
}

/// Swap the x and y buffers.
fn flip(coords: &CoordBuffer) -> GeoArrowResult<CoordBuffer> {
    match coords {
        CoordBuffer::Separated(separated) => {
            let raw = separated.raw_buffers();
            // Two handle swaps. No coordinate is read and nothing is allocated.
            let swapped = [
                raw[1].clone(),
                raw[0].clone(),
                raw[2].clone(),
                raw[3].clone(),
            ];
            Ok(CoordBuffer::Separated(SeparatedCoordBuffer::from_array(
                swapped,
                separated.dim(),
            )?))
        }
        CoordBuffer::Interleaved(interleaved) => {
            // xyxy in one buffer. Every coordinate must be rewritten.
            let stride = interleaved.dim().size();
            let source = interleaved.coords();
            let mut values = Vec::with_capacity(source.len());
            for chunk in source.chunks_exact(stride) {
                values.push(chunk[1]);
                values.push(chunk[0]);
                values.extend_from_slice(&chunk[2..]);
            }
            Ok(CoordBuffer::Interleaved(
                geoarrow_array::array::InterleavedCoordBuffer::new(
                    values.into(),
                    interleaved.dim(),
                ),
            ))
        }
    }
}

/// Change the dimension of a coordinate buffer. A new ordinate takes the value zero.
fn resize(coords: &CoordBuffer, target: Dimension) -> GeoArrowResult<CoordBuffer> {
    let len = coords.len();
    match coords {
        CoordBuffer::Separated(separated) => {
            let raw = separated.raw_buffers();
            let source = separated.dim();
            let zeros = || ScalarBuffer::<f64>::from(vec![0.0f64; len]);
            let empty = || ScalarBuffer::<f64>::from(Vec::<f64>::new());

            // Reuse a source buffer when the target keeps that ordinate. x and y always survive.
            let buffers: [ScalarBuffer<f64>; 4] = match target {
                Dimension::XY => [raw[0].clone(), raw[1].clone(), empty(), empty()],
                Dimension::XYZ => [
                    raw[0].clone(),
                    raw[1].clone(),
                    match source {
                        Dimension::XYZ | Dimension::XYZM => raw[2].clone(),
                        _ => zeros(),
                    },
                    empty(),
                ],
                Dimension::XYM => [
                    raw[0].clone(),
                    raw[1].clone(),
                    match source {
                        Dimension::XYM => raw[2].clone(),
                        Dimension::XYZM => raw[3].clone(),
                        _ => zeros(),
                    },
                    empty(),
                ],
                Dimension::XYZM => [
                    raw[0].clone(),
                    raw[1].clone(),
                    match source {
                        Dimension::XYZ | Dimension::XYZM => raw[2].clone(),
                        _ => zeros(),
                    },
                    match source {
                        Dimension::XYZM => raw[3].clone(),
                        Dimension::XYM => raw[2].clone(),
                        _ => zeros(),
                    },
                ],
            };
            Ok(CoordBuffer::Separated(SeparatedCoordBuffer::from_array(
                buffers, target,
            )?))
        }
        CoordBuffer::Interleaved(interleaved) => {
            let source_stride = interleaved.dim().size();
            let target_stride = target.size();
            let source = interleaved.coords();
            let mut values = Vec::with_capacity(len * target_stride);
            for chunk in source.chunks_exact(source_stride) {
                for slot in 0..target_stride {
                    values.push(chunk.get(slot).copied().unwrap_or(0.0));
                }
            }
            Ok(CoordBuffer::Interleaved(
                geoarrow_array::array::InterleavedCoordBuffer::new(values.into(), target),
            ))
        }
    }
}

/// Rebuild an array around a new coordinate buffer. Every offset buffer stays.
///
/// The offsets describe the structure, not the position. A coordinate transform leaves them valid.
/// Each
/// clone below is an atomic counter bump on a shared buffer.
pub(crate) fn map_coords(
    function: &str,
    array: &dyn GeoArrowArray,
    transform: impl Fn(&CoordBuffer) -> GeoArrowResult<CoordBuffer>,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let nulls = array.logical_nulls();

    Ok(match array.data_type() {
        GeoArrowType::Point(typ) => {
            let source = array.as_point();
            Arc::new(PointArray::new(
                transform(source.coords())?,
                nulls,
                typ.metadata().clone(),
            ))
        }
        GeoArrowType::LineString(typ) => {
            let source = array.as_line_string();
            Arc::new(LineStringArray::new(
                transform(source.coords())?,
                source.geom_offsets().clone(),
                nulls,
                typ.metadata().clone(),
            ))
        }
        GeoArrowType::Polygon(typ) => {
            let source = array.as_polygon();
            Arc::new(PolygonArray::new(
                transform(source.coords())?,
                source.geom_offsets().clone(),
                source.ring_offsets().clone(),
                nulls,
                typ.metadata().clone(),
            ))
        }
        GeoArrowType::MultiPoint(typ) => {
            let source = array.as_multi_point();
            Arc::new(MultiPointArray::new(
                transform(source.coords())?,
                source.geom_offsets().clone(),
                nulls,
                typ.metadata().clone(),
            ))
        }
        GeoArrowType::MultiLineString(typ) => {
            let source = array.as_multi_line_string();
            Arc::new(MultiLineStringArray::new(
                transform(source.coords())?,
                source.geom_offsets().clone(),
                source.ring_offsets().clone(),
                nulls,
                typ.metadata().clone(),
            ))
        }
        GeoArrowType::MultiPolygon(typ) => {
            let source = array.as_multi_polygon();
            Arc::new(MultiPolygonArray::new(
                transform(source.coords())?,
                source.geom_offsets().clone(),
                source.polygon_offsets().clone(),
                source.ring_offsets().clone(),
                nulls,
                typ.metadata().clone(),
            ))
        }
        GeoArrowType::GeometryCollection(_) => {
            let _: &GeometryCollectionArray = array.as_geometry_collection();
            return Err(unsupported(function));
        }
        _ => return Err(unsupported(function)),
    })
}

#[cfg(test)]
mod tests {
    use geo_traits::to_geo::ToGeoGeometry;
    use geoarrow_array::builder::{LineStringBuilder, PointBuilder, PolygonBuilder};
    use geoarrow_array::GeoArrowArrayAccessor;
    use geoarrow_schema::{CoordType, LineStringType, PointType, PolygonType};

    use super::*;

    fn points(coord_type: CoordType) -> PointArray {
        let p0 = geo::point!(x: 1.0, y: 2.0);
        let p1 = geo::point!(x: 3.0, y: 4.0);
        PointBuilder::from_points(
            [p0, p1].iter(),
            PointType::new(Dimension::XY, Default::default()).with_coord_type(coord_type),
        )
        .finish()
    }

    fn lines(coord_type: CoordType) -> LineStringArray {
        let values: Vec<geo::LineString<f64>> = vec![
            geo::wkt! { LINESTRING(0.0 0.0,1.0 2.0,3.0 4.0) },
            geo::wkt! { LINESTRING(5.0 6.0,7.0 8.0) },
        ];
        LineStringBuilder::from_line_strings(
            &values,
            LineStringType::new(Dimension::XY, Default::default()).with_coord_type(coord_type),
        )
        .finish()
    }

    #[test]
    fn flip_swaps_x_and_y() {
        for coord_type in [CoordType::Separated, CoordType::Interleaved] {
            let array = points(coord_type);
            let flipped = st_flip_coordinates(&array).unwrap();
            let flipped = flipped.as_point();

            let first = flipped.value(0).unwrap().to_geometry();
            let geo::Geometry::Point(p) = first else {
                panic!("expected a point")
            };
            assert_eq!(p.x(), 2.0);
            assert_eq!(p.y(), 1.0);
        }
    }

    /// The claim from the module docs, proven on the buffers themselves.
    #[test]
    fn flip_is_zero_copy_on_separated_coords() {
        let array = points(CoordType::Separated);
        let CoordBuffer::Separated(source) = array.coords() else {
            panic!("expected separated coordinates");
        };
        let (x_ptr, y_ptr) = (
            source.raw_buffers()[0].as_ptr(),
            source.raw_buffers()[1].as_ptr(),
        );

        let flipped = st_flip_coordinates(&array).unwrap();
        let CoordBuffer::Separated(result) = flipped.as_point().coords() else {
            panic!("expected separated coordinates");
        };

        assert_eq!(
            result.raw_buffers()[0].as_ptr(),
            y_ptr,
            "the new x buffer must be the old y buffer, not a copy"
        );
        assert_eq!(result.raw_buffers()[1].as_ptr(), x_ptr);
    }

    /// A flip of a line string must not disturb the offsets.
    #[test]
    fn flip_keeps_the_offsets() {
        for coord_type in [CoordType::Separated, CoordType::Interleaved] {
            let array = lines(coord_type);
            let flipped = st_flip_coordinates(&array).unwrap();
            let flipped = flipped.as_line_string();

            assert_eq!(flipped.geom_offsets(), array.geom_offsets());
            let geo::Geometry::LineString(first) = flipped.value(0).unwrap().to_geometry() else {
                panic!("expected a line string")
            };
            assert_eq!(first.0[0].x, 0.0);
            assert_eq!(first.0[1].x, 2.0, "x and y swapped");
            assert_eq!(first.0[1].y, 1.0);
        }
    }

    #[test]
    fn force_3d_adds_a_zero_z() {
        let array = points(CoordType::Separated);
        let forced = st_force_3d(&array).unwrap();
        assert_eq!(forced.data_type().dimension(), Some(Dimension::XYZ));

        let CoordBuffer::Separated(coords) = forced.as_point().coords() else {
            panic!("expected separated coordinates");
        };
        assert_eq!(coords.raw_buffers()[2].as_ref(), &[0.0, 0.0]);
        // x and y are the original buffers, untouched.
        let CoordBuffer::Separated(source) = array.coords() else {
            unreachable!()
        };
        assert_eq!(
            coords.raw_buffers()[0].as_ptr(),
            source.raw_buffers()[0].as_ptr()
        );
    }

    #[test]
    fn force_2d_drops_z_without_copying_x_and_y() {
        let array = points(CoordType::Separated);
        let three_d = st_force_3d(&array).unwrap();
        let CoordBuffer::Separated(before) = three_d.as_point().coords() else {
            unreachable!()
        };
        let x_ptr = before.raw_buffers()[0].as_ptr();

        let two_d = st_force_2d(three_d.as_ref()).unwrap();
        assert_eq!(two_d.data_type().dimension(), Some(Dimension::XY));
        let CoordBuffer::Separated(after) = two_d.as_point().coords() else {
            unreachable!()
        };
        assert_eq!(
            after.raw_buffers()[0].as_ptr(),
            x_ptr,
            "x survives untouched"
        );
    }

    #[test]
    fn force_to_the_same_dimension_is_a_no_op() {
        let array = points(CoordType::Separated);
        let forced = st_force_2d(&array).unwrap();
        assert_eq!(forced.len(), array.len());
        assert_eq!(forced.data_type().dimension(), Some(Dimension::XY));
    }

    #[test]
    fn polygons_keep_both_offset_levels() {
        let squares: Vec<geo::Polygon<f64>> = vec![
            geo::wkt! { POLYGON((0.0 0.0,4.0 0.0,4.0 4.0,0.0 4.0,0.0 0.0),(1.0 1.0,2.0 1.0,2.0 2.0,1.0 1.0)) },
        ];
        let array = PolygonBuilder::from_polygons(
            &squares,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let flipped = st_flip_coordinates(&array).unwrap();
        let flipped = flipped.as_polygon();
        assert_eq!(flipped.geom_offsets(), array.geom_offsets());
        assert_eq!(flipped.ring_offsets(), array.ring_offsets());
    }

    #[test]
    fn untyped_input_is_rejected_with_advice() {
        let values: Vec<geo::LineString<f64>> = vec![geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0) }];
        let array = LineStringBuilder::from_line_strings(
            &values,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let wkb = geoarrow_array::cast::to_wkb::<i32>(&array).unwrap();

        let err = st_force_3d(&wkb).unwrap_err();
        assert!(err.to_string().contains("Cast a WKB"), "got: {err}");
    }
}
