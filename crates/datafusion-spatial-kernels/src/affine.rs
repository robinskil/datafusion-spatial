//! Affine transforms.
//!
//! `ST_Translate`, `ST_Scale`, `ST_Rotate` and `ST_Affine` all reduce to one matrix multiply per
//! coordinate, so they compose: two transforms applied in sequence could be one matrix. That is
//! why they share a single kernel with an [`AffineTransform`] parameter.
//!
//! # The structure survives, so the offsets do
//!
//! An affine transform moves coordinates. It never changes the ring structure.
//! [`crate::transform`] relies on the same rule. So these functions keep the input geometry type
//! and reuse its offset buffers. Only the coordinate buffer is rebuilt.

use std::sync::Arc;

use geo::AffineTransform;
use geoarrow_array::array::CoordBuffer;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::GeoArrowResult;
use geoarrow_schema::GeoArrowType;

use crate::transform::map_coords;

/// Which affine transform to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Affine {
    /// `ST_Translate(geom, dx, dy)`.
    Translate,
    /// `ST_Scale(geom, xfact, yfact)`, about the origin.
    Scale,
    /// `ST_Rotate(geom, radians)`, about the origin.
    Rotate,
    /// `ST_Affine(geom, a, b, d, e, xoff, yoff)`, the raw matrix.
    Matrix,
}

impl Affine {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::Translate => "ST_Translate",
            Self::Scale => "ST_Scale",
            Self::Rotate => "ST_Rotate",
            Self::Matrix => "ST_Affine",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::Translate => "st_translate",
            Self::Scale => "st_scale",
            Self::Rotate => "st_rotate",
            Self::Matrix => "st_affine",
        }
    }

    /// How many numbers follow the geometry argument.
    pub const fn parameter_count(self) -> usize {
        match self {
            Self::Translate | Self::Scale => 2,
            Self::Rotate => 1,
            Self::Matrix => 6,
        }
    }

    /// Every affine function, for registration.
    pub const ALL: [Self; 4] = [Self::Translate, Self::Scale, Self::Rotate, Self::Matrix];

    /// Build the matrix for this transform from its parameters.
    ///
    /// The caller has already checked the parameter count.
    pub fn matrix(self, parameters: &[f64]) -> AffineTransform<f64> {
        let origin = geo::coord! { x: 0.0, y: 0.0 };
        match self {
            Self::Translate => AffineTransform::translate(parameters[0], parameters[1]),
            Self::Scale => AffineTransform::scale(parameters[0], parameters[1], origin),
            // PostGIS takes radians here, while `geo` takes degrees.
            Self::Rotate => AffineTransform::rotate(parameters[0].to_degrees(), origin),
            Self::Matrix => AffineTransform::new(
                parameters[0],
                parameters[1],
                parameters[4],
                parameters[2],
                parameters[3],
                parameters[5],
            ),
        }
    }
}

/// The output type of an affine transform.
///
/// A native array keeps its type exactly, because only the coordinates move. A mixed or
/// serialized column has no single type to keep, so it becomes a mixed geometry array.
pub fn output_type(input: &GeoArrowType) -> GeoArrowType {
    if crate::accessor::is_untyped(input) {
        GeoArrowType::Geometry(geoarrow_schema::GeometryType::new(Arc::clone(
            input.metadata(),
        )))
    } else {
        input.clone()
    }
}

/// Apply one affine matrix to every coordinate of an array.
///
/// # Two paths
///
/// A native array keeps its geometry type and every offset buffer. Only the coordinate buffer is
/// rebuilt, so a polygon column stays a polygon column.
///
/// A mixed or serialized column holds an unknown type per row. Its coordinates do not sit in one
/// buffer. So it takes a per-row path and becomes a mixed geometry array. `ST_GeomFromText`
/// produces that shape, so it is the common case in hand written SQL.
pub fn affine(
    array: &dyn GeoArrowArray,
    matrix: &AffineTransform<f64>,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    if crate::accessor::is_untyped(&array.data_type()) {
        let output = geoarrow_schema::GeometryType::new(Arc::clone(array.data_type().metadata()));
        return affine_per_row(array, matrix, output);
    }
    map_coords("an affine transform", array, |coords| {
        Ok(apply_matrix(coords, matrix))
    })
}

fn affine_per_row(
    array: &dyn GeoArrowArray,
    matrix: &AffineTransform<f64>,
    output: geoarrow_schema::GeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    use geo::AffineOps;
    use geoarrow_array::builder::GeometryBuilder;

    let mut reader = crate::materialize::GeometryReader::new(array)?;
    let mut builder = GeometryBuilder::new(output);
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => {
                let moved = geom.affine_transform(matrix);
                builder.push_geometry(Some(&moved))?;
            }
            None => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// Rebuild a coordinate buffer through the matrix.
fn apply_matrix(coords: &CoordBuffer, matrix: &AffineTransform<f64>) -> CoordBuffer {
    use arrow_buffer::ScalarBuffer;
    use geoarrow_array::array::{InterleavedCoordBuffer, SeparatedCoordBuffer};

    match coords {
        CoordBuffer::Separated(separated) => {
            let raw = separated.raw_buffers();
            let (xs, ys) = (&raw[0], &raw[1]);

            // Two output buffers, both sized exactly once.
            let mut new_x = Vec::with_capacity(xs.len());
            let mut new_y = Vec::with_capacity(ys.len());
            for (&x, &y) in xs.iter().zip(ys.iter()) {
                let moved = matrix.apply(geo::coord! { x: x, y: y });
                new_x.push(moved.x);
                new_y.push(moved.y);
            }

            let buffers = [
                ScalarBuffer::from(new_x),
                ScalarBuffer::from(new_y),
                raw[2].clone(),
                raw[3].clone(),
            ];
            // The lengths cannot disagree: both came from the same input length.
            CoordBuffer::Separated(
                SeparatedCoordBuffer::from_array(buffers, separated.dim())
                    .expect("the output buffers share the input length"),
            )
        }
        CoordBuffer::Interleaved(interleaved) => {
            let stride = interleaved.dim().size();
            let source = interleaved.coords();
            let mut values = Vec::with_capacity(source.len());
            for chunk in source.chunks_exact(stride) {
                let moved = matrix.apply(geo::coord! { x: chunk[0], y: chunk[1] });
                values.push(moved.x);
                values.push(moved.y);
                // Higher ordinates are carried through unchanged. A 2D affine says nothing
                // about z or m.
                values.extend_from_slice(&chunk[2..]);
            }
            CoordBuffer::Interleaved(InterleavedCoordBuffer::new(
                values.into(),
                interleaved.dim(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use geo_traits::to_geo::ToGeoGeometry;
    use geoarrow_array::builder::{LineStringBuilder, PointBuilder};
    use geoarrow_array::cast::AsGeoArrowArray;
    use geoarrow_array::GeoArrowArrayAccessor;
    use geoarrow_schema::{CoordType, Dimension, LineStringType, PointType};

    use super::*;

    fn points(coord_type: CoordType) -> geoarrow_array::array::PointArray {
        PointBuilder::from_points(
            [geo::point!(x: 1.0, y: 2.0), geo::point!(x: 3.0, y: 4.0)].iter(),
            PointType::new(Dimension::XY, Default::default()).with_coord_type(coord_type),
        )
        .finish()
    }

    fn read_point(array: &dyn GeoArrowArray, row: usize) -> (f64, f64) {
        let geo::Geometry::Point(p) = array.as_point().value(row).unwrap().to_geometry() else {
            panic!("expected a point")
        };
        (p.x(), p.y())
    }

    #[test]
    fn translate_moves_every_coordinate() {
        for coord_type in [CoordType::Separated, CoordType::Interleaved] {
            let array = points(coord_type);
            let matrix = Affine::Translate.matrix(&[10.0, 20.0]);
            let moved = affine(&array, &matrix).unwrap();

            assert_eq!(read_point(moved.as_ref(), 0), (11.0, 22.0));
            assert_eq!(read_point(moved.as_ref(), 1), (13.0, 24.0));
        }
    }

    #[test]
    fn scale_multiplies_about_the_origin() {
        let array = points(CoordType::Separated);
        let matrix = Affine::Scale.matrix(&[2.0, 3.0]);
        let scaled = affine(&array, &matrix).unwrap();
        assert_eq!(read_point(scaled.as_ref(), 0), (2.0, 6.0));
    }

    /// PostGIS rotates by radians, so a quarter turn is pi over two.
    #[test]
    fn rotate_takes_radians() {
        let array = PointBuilder::from_points(
            [geo::point!(x: 1.0, y: 0.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let matrix = Affine::Rotate.matrix(&[std::f64::consts::FRAC_PI_2]);
        let turned = affine(&array, &matrix).unwrap();
        let (x, y) = read_point(turned.as_ref(), 0);
        assert!(x.abs() < 1e-12, "x was {x}");
        assert!((y - 1.0).abs() < 1e-12, "y was {y}");
    }

    #[test]
    fn matrix_form_matches_translate() {
        let array = points(CoordType::Separated);
        // Identity rotation with an offset is the same as a translation.
        let matrix = Affine::Matrix.matrix(&[1.0, 0.0, 0.0, 1.0, 10.0, 20.0]);
        let moved = affine(&array, &matrix).unwrap();
        assert_eq!(read_point(moved.as_ref(), 0), (11.0, 22.0));
    }

    /// The offsets are structure, not position, so a transform must leave them alone.
    #[test]
    fn offsets_survive_the_transform() {
        let lines: Vec<geo::LineString<f64>> = vec![
            geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0,2.0 2.0) },
            geo::wkt! { LINESTRING(5.0 5.0,6.0 6.0) },
        ];
        let array = LineStringBuilder::from_line_strings(
            &lines,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let matrix = Affine::Translate.matrix(&[1.0, 1.0]);
        let moved = affine(&array, &matrix).unwrap();
        let moved = moved.as_line_string();

        assert_eq!(moved.geom_offsets(), array.geom_offsets());
        let geo::Geometry::LineString(first) = moved.value(0).unwrap().to_geometry() else {
            panic!("expected a line string")
        };
        assert_eq!((first.0[0].x, first.0[0].y), (1.0, 1.0));
        assert_eq!(first.0.len(), 3);
    }

    /// The output type is the input type, so a point column stays a point column.
    #[test]
    fn the_geometry_type_is_preserved() {
        let array = points(CoordType::Separated);
        let matrix = Affine::Translate.matrix(&[1.0, 1.0]);
        let moved = affine(&array, &matrix).unwrap();
        assert_eq!(moved.data_type(), array.data_type());
    }
}
