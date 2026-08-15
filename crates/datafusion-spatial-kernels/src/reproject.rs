//! Reprojection through PROJ.
//!
//! This module only exists when the `proj` feature is on, because it pulls in a C++ library. See
//! the crate README for the two ways to link it.
//!
//! # One transform per batch, one FFI call per buffer
//!
//! Two costs dominate a naive reprojection, and both are avoided here.
//!
//! To build a [`Proj`], PROJ parses two CRS definitions. It then searches its database for a
//! conversion pipeline. That costs far more than one coordinate transform. So the crate builds it
//! once per call and reuses it for every row. It cannot live longer than the call: `Proj` holds a
//! raw context pointer and is neither `Send` nor `Sync`. [`PreparedGeometry`][ref] has the same
//! limit.
//!
//! The second cost is one FFI call per coordinate. [`Proj::convert_array`] transforms a whole
//! slice in one call. A batch of 8192 points then costs one FFI call, not 8192.
//!
//! # The structure survives, so the offsets do
//!
//! A reprojection moves coordinates. It never changes the ring structure. [`crate::transform`]
//! and [`crate::affine`] rely on the same rule. So the output keeps the input geometry type and
//! reuses every offset buffer. Only the coordinate buffer is rebuilt.
//!
//! [ref]: crate::predicate::PreparedLiteral

use std::sync::Arc;

use geoarrow_array::array::CoordBuffer;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::{GeoArrowError, GeoArrowResult};
use geoarrow_schema::GeoArrowType;
use proj::Proj;

use crate::crs::{srid_of, with_srid};
use crate::transform::map_coords;

/// The type `ST_Transform` produces: the input type, restamped with the target SRID.
///
/// A native column keeps its geometry type, because only the coordinates move. A mixed or
/// serialized column has no single type to keep, so it becomes a mixed geometry column.
pub fn output_type(input: &GeoArrowType, target_srid: i32) -> GeoArrowType {
    let shape = if crate::accessor::is_untyped(input) {
        GeoArrowType::Geometry(geoarrow_schema::GeometryType::new(Arc::clone(
            input.metadata(),
        )))
    } else {
        input.clone()
    };
    with_srid(&shape, target_srid)
}

/// Build the transformation between two SRIDs.
///
/// Kept public so a caller can build it once and hand it to [`st_transform_with`] for several
/// batches, which is the only way to reuse it: `Proj` is not `Send`, so it cannot be stored on a
/// UDF struct.
pub fn transformation(from_srid: i32, to_srid: i32) -> GeoArrowResult<Proj> {
    if from_srid == 0 {
        return Err(GeoArrowError::InvalidGeoArrow(
            "ST_Transform needs a source SRID on the input column. Stamp one with ST_SetSRID."
                .to_string(),
        ));
    }
    if to_srid == 0 {
        return Err(GeoArrowError::InvalidGeoArrow(
            "ST_Transform needs a target SRID other than zero".to_string(),
        ));
    }

    Proj::new_known_crs(
        &format!("EPSG:{from_srid}"),
        &format!("EPSG:{to_srid}"),
        None,
    )
    .map_err(|err| {
        GeoArrowError::External(Box::new(std::io::Error::other(format!(
            "PROJ cannot transform EPSG:{from_srid} to EPSG:{to_srid}: {err}"
        ))))
    })
}

/// `ST_Transform`. Reproject every coordinate into the target SRID.
///
/// The source SRID comes from the column metadata, so the input must carry one.
pub fn st_transform(
    array: &dyn GeoArrowArray,
    to_srid: i32,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let from_srid = srid_of(&array.data_type());
    let projection = transformation(from_srid, to_srid)?;
    st_transform_with(array, &projection, to_srid)
}

/// `ST_Transform` with a transformation the caller already built.
pub fn st_transform_with(
    array: &dyn GeoArrowArray,
    projection: &Proj,
    to_srid: i32,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    // A mixed or serialized column has no single coordinate buffer, so it takes a per-row path.
    // That is the shape `ST_GeomFromText` produces, so it is the common case in hand written SQL.
    let reprojected = if crate::accessor::is_untyped(&array.data_type()) {
        let output = geoarrow_schema::GeometryType::new(Arc::clone(array.data_type().metadata()));
        transform_per_row(array, projection, output)?
    } else {
        map_coords("ST_Transform", array, |coords| {
            reproject_buffer(coords, projection)
        })?
    };

    // Restamp the column with the target SRID. The values have moved, so the old one would lie.
    let metadata = with_srid(&reprojected.data_type(), to_srid)
        .metadata()
        .clone();
    Ok(reprojected.with_metadata(metadata))
}

fn transform_per_row(
    array: &dyn GeoArrowArray,
    projection: &Proj,
    output: geoarrow_schema::GeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    use geoarrow_array::builder::GeometryBuilder;
    use proj::Transform;

    let mut reader = crate::materialize::GeometryReader::new(array)?;
    let mut builder = GeometryBuilder::new(output);
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => {
                let moved = geom.transformed(projection).map_err(|err| {
                    GeoArrowError::External(Box::new(std::io::Error::other(format!(
                        "PROJ failed to transform a geometry: {err}"
                    ))))
                })?;
                builder.push_geometry(Some(&moved))?;
            }
            None => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// Transform a whole coordinate buffer in one FFI call.
fn reproject_buffer(coords: &CoordBuffer, projection: &Proj) -> GeoArrowResult<CoordBuffer> {
    use arrow_buffer::ScalarBuffer;
    use geoarrow_array::array::{InterleavedCoordBuffer, SeparatedCoordBuffer};

    // PROJ works on x and y pairs, so gather into that shape once.
    let mut pairs: Vec<geo::Coord<f64>> = Vec::with_capacity(coords.len());
    match coords {
        CoordBuffer::Separated(separated) => {
            let raw = separated.raw_buffers();
            pairs.extend(
                raw[0]
                    .iter()
                    .zip(raw[1].iter())
                    .map(|(&x, &y)| geo::coord! { x: x, y: y }),
            );
        }
        CoordBuffer::Interleaved(interleaved) => {
            let stride = interleaved.dim().size();
            pairs.extend(
                interleaved
                    .coords()
                    .chunks_exact(stride)
                    .map(|chunk| geo::coord! { x: chunk[0], y: chunk[1] }),
            );
        }
    }

    // One FFI call for the whole batch.
    if !pairs.is_empty() {
        projection.convert_array(&mut pairs).map_err(|err| {
            GeoArrowError::External(Box::new(std::io::Error::other(format!(
                "PROJ failed to transform the batch: {err}"
            ))))
        })?;
    }

    Ok(match coords {
        CoordBuffer::Separated(separated) => {
            let raw = separated.raw_buffers();
            let mut xs = Vec::with_capacity(pairs.len());
            let mut ys = Vec::with_capacity(pairs.len());
            for pair in &pairs {
                xs.push(pair.x);
                ys.push(pair.y);
            }
            // Higher ordinates are carried through: a horizontal transform says nothing about
            // z or m.
            let buffers = [
                ScalarBuffer::from(xs),
                ScalarBuffer::from(ys),
                raw[2].clone(),
                raw[3].clone(),
            ];
            CoordBuffer::Separated(SeparatedCoordBuffer::from_array(buffers, separated.dim())?)
        }
        CoordBuffer::Interleaved(interleaved) => {
            let stride = interleaved.dim().size();
            let source = interleaved.coords();
            let mut values = Vec::with_capacity(source.len());
            for (pair, chunk) in pairs.iter().zip(source.chunks_exact(stride)) {
                values.push(pair.x);
                values.push(pair.y);
                values.extend_from_slice(&chunk[2..]);
            }
            CoordBuffer::Interleaved(InterleavedCoordBuffer::new(
                values.into(),
                interleaved.dim(),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use geo_traits::to_geo::ToGeoGeometry;
    use geoarrow_array::builder::{LineStringBuilder, PointBuilder};
    use geoarrow_array::cast::AsGeoArrowArray;
    use geoarrow_array::GeoArrowArrayAccessor;
    use geoarrow_schema::{CoordType, Dimension, LineStringType, PointType};

    use super::*;
    use crate::crs::st_set_srid;

    /// London, in longitude and latitude.
    fn london(coord_type: CoordType) -> Arc<dyn GeoArrowArray> {
        let array = PointBuilder::from_points(
            [geo::point!(x: -0.1278, y: 51.5074)].iter(),
            PointType::new(Dimension::XY, Default::default()).with_coord_type(coord_type),
        )
        .finish();
        st_set_srid(&array, 4326).unwrap()
    }

    fn read_point(array: &dyn GeoArrowArray, row: usize) -> (f64, f64) {
        let geo::Geometry::Point(p) = array.as_point().value(row).unwrap().to_geometry() else {
            panic!("expected a point")
        };
        (p.x(), p.y())
    }

    /// Web Mercator metres for London are roughly (-14 200, 6 711 000).
    #[test]
    fn transform_to_web_mercator() {
        for coord_type in [CoordType::Separated, CoordType::Interleaved] {
            let array = london(coord_type);
            let moved = st_transform(array.as_ref(), 3857).unwrap();

            let (x, y) = read_point(moved.as_ref(), 0);
            assert!((-14_300.0..-14_100.0).contains(&x), "x was {x}");
            assert!((6_710_000.0..6_712_000.0).contains(&y), "y was {y}");
        }
    }

    /// The output column must carry the target SRID, not the source one.
    #[test]
    fn the_output_is_restamped() {
        let array = london(CoordType::Separated);
        assert_eq!(srid_of(&array.data_type()), 4326);

        let moved = st_transform(array.as_ref(), 3857).unwrap();
        assert_eq!(srid_of(&moved.data_type()), 3857);
    }

    /// There and back again must land where it started.
    #[test]
    fn a_round_trip_returns_the_input() {
        let array = london(CoordType::Separated);
        let (before_x, before_y) = read_point(array.as_ref(), 0);

        let there = st_transform(array.as_ref(), 3857).unwrap();
        let back = st_transform(there.as_ref(), 4326).unwrap();

        let (after_x, after_y) = read_point(back.as_ref(), 0);
        assert!((after_x - before_x).abs() < 1e-9, "x drifted to {after_x}");
        assert!((after_y - before_y).abs() < 1e-9, "y drifted to {after_y}");
        assert_eq!(srid_of(&back.data_type()), 4326);
    }

    /// A transform moves coordinates and keeps the ring structure, so the offsets survive.
    #[test]
    fn offsets_survive_the_transform() {
        let lines: Vec<geo::LineString<f64>> = vec![
            geo::wkt! { LINESTRING(-0.1278 51.5074,2.3522 48.8566,13.405 52.52) },
            geo::wkt! { LINESTRING(4.9041 52.3676,12.4964 41.9028) },
        ];
        let array = LineStringBuilder::from_line_strings(
            &lines,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let array = st_set_srid(&array, 4326).unwrap();

        let moved = st_transform(array.as_ref(), 3857).unwrap();
        assert_eq!(
            moved.as_line_string().geom_offsets(),
            array.as_line_string().geom_offsets(),
            "the offsets describe nesting, which a reprojection does not change"
        );
        assert_eq!(moved.len(), 2);
    }

    /// Without a source SRID there is nothing to transform from.
    #[test]
    fn an_unstamped_column_is_an_error() {
        let array = PointBuilder::from_points(
            [geo::point!(x: 0.0, y: 0.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let err = st_transform(&array, 3857).unwrap_err();
        assert!(err.to_string().contains("ST_SetSRID"), "got: {err}");
    }

    #[test]
    fn an_unknown_srid_is_an_error() {
        let array = london(CoordType::Separated);
        let err = st_transform(array.as_ref(), 999_999).unwrap_err();
        assert!(
            err.to_string().contains("PROJ cannot transform"),
            "got: {err}"
        );
    }

    /// The transformation serves every batch. That is why this function is public.
    #[test]
    fn one_transformation_serves_many_batches() {
        let projection = transformation(4326, 3857).unwrap();
        let array = london(CoordType::Separated);

        let first = st_transform_with(array.as_ref(), &projection, 3857).unwrap();
        let second = st_transform_with(array.as_ref(), &projection, 3857).unwrap();
        assert_eq!(
            read_point(first.as_ref(), 0),
            read_point(second.as_ref(), 0)
        );
    }
}
