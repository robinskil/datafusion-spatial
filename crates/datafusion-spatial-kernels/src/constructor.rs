//! Build geometries from plain columns.
//!
//! # Why these are nearly free
//!
//! A GeoArrow point array with separated coordinates *is* two `f64` buffers. `ST_MakePoint(x, y)`
//! over two `Float64` columns therefore adopts the input buffers and builds nothing.
//! `ST_MakeEnvelope` does the same with four.
//!
//! `ST_MakePolygon` is the same trick one level up: a line string array and a polygon array share
//! a coordinate buffer layout, and the line string offsets become the polygon ring offsets. Only
//! one small offset buffer is built.

use std::sync::Arc;

use arrow_array::{Array, Float64Array};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use geo_traits::to_geo::ToGeoLineString;
use geo_traits::GeometryTrait;
use geoarrow_array::array::{
    CoordBuffer, LineStringArray, PointArray, PolygonArray, RectArray, SeparatedCoordBuffer,
};
use geoarrow_array::builder::PolygonBuilder;
use geoarrow_array::cast::AsGeoArrowArray;
use geoarrow_array::{downcast_geoarrow_array, GeoArrowArray, GeoArrowArrayAccessor};

use crate::accessor::is_untyped;
use geoarrow_schema::error::{GeoArrowError, GeoArrowResult};
use geoarrow_schema::{BoxType, Dimension, GeoArrowType, LineStringType, Metadata, PointType};

/// `ST_Point` and `ST_MakePoint`. Build points from an x column and a y column.
///
/// The output adopts the input buffers. Nothing is copied.
pub fn st_make_point(
    x: &Float64Array,
    y: &Float64Array,
    z: Option<&Float64Array>,
    metadata: Arc<Metadata>,
) -> GeoArrowResult<PointArray> {
    let len = x.len();
    if y.len() != len || z.is_some_and(|z| z.len() != len) {
        return Err(GeoArrowError::InvalidGeoArrow(
            "ST_MakePoint needs columns of the same length".to_string(),
        ));
    }

    // A row is null when any ordinate is null, as in PostGIS.
    let mut nulls = NullBuffer::union(x.nulls(), y.nulls());
    if let Some(z) = z {
        nulls = NullBuffer::union(nulls.as_ref(), z.nulls());
    }

    let empty = ScalarBuffer::<f64>::from(Vec::<f64>::new());
    let (dim, buffers) = match z {
        None => (
            Dimension::XY,
            [x.values().clone(), y.values().clone(), empty.clone(), empty],
        ),
        Some(z) => (
            Dimension::XYZ,
            [
                x.values().clone(),
                y.values().clone(),
                z.values().clone(),
                empty,
            ],
        ),
    };

    let coords = SeparatedCoordBuffer::from_array(buffers, dim)?;
    Ok(PointArray::new(
        CoordBuffer::Separated(coords),
        nulls,
        metadata,
    ))
}

/// The type `ST_MakePoint` produces.
pub fn point_type(dim: Dimension, metadata: Arc<Metadata>) -> PointType {
    PointType::new(dim, metadata).with_coord_type(geoarrow_schema::CoordType::Separated)
}

/// `ST_MakeEnvelope` and `ST_MakeBox2D`. Build boxes from four ordinate columns.
///
/// The output adopts all four input buffers.
pub fn st_make_envelope(
    xmin: &Float64Array,
    ymin: &Float64Array,
    xmax: &Float64Array,
    ymax: &Float64Array,
    metadata: Arc<Metadata>,
) -> GeoArrowResult<RectArray> {
    let len = xmin.len();
    if [ymin.len(), xmax.len(), ymax.len()]
        .iter()
        .any(|other| *other != len)
    {
        return Err(GeoArrowError::InvalidGeoArrow(
            "ST_MakeEnvelope needs columns of the same length".to_string(),
        ));
    }

    let nulls = NullBuffer::union(
        NullBuffer::union(xmin.nulls(), ymin.nulls()).as_ref(),
        NullBuffer::union(xmax.nulls(), ymax.nulls()).as_ref(),
    );

    let empty = ScalarBuffer::<f64>::from(Vec::<f64>::new());
    let lower = SeparatedCoordBuffer::from_array(
        [
            xmin.values().clone(),
            ymin.values().clone(),
            empty.clone(),
            empty.clone(),
        ],
        Dimension::XY,
    )?;
    let upper = SeparatedCoordBuffer::from_array(
        [
            xmax.values().clone(),
            ymax.values().clone(),
            empty.clone(),
            empty,
        ],
        Dimension::XY,
    )?;

    Ok(RectArray::new(lower, upper, nulls, metadata))
}

/// The type `ST_MakeEnvelope` produces.
pub fn box_type(metadata: Arc<Metadata>) -> BoxType {
    BoxType::new(Dimension::XY, metadata)
}

/// `ST_MakePolygon`. Turn each closed line string into a polygon with no holes.
///
/// # Two paths
///
/// A typed line string array needs no coordinate work. The two sides share the coordinate buffer.
/// The line string offsets become the polygon ring offsets. The only new buffer is a small list
/// of `0, 1, 2, ...`. It states that every polygon has exactly one ring.
///
/// A mixed or serialized array carries an unknown type per row, so it takes a per-row path. That
/// is the shape produced by `ST_GeomFromText`, so it is the common case in hand written SQL.
pub fn st_make_polygon(
    array: &dyn GeoArrowArray,
    output: geoarrow_schema::PolygonType,
) -> GeoArrowResult<PolygonArray> {
    match array.data_type() {
        GeoArrowType::LineString(typ) => {
            let source: &LineStringArray = array.as_line_string();
            let len = source.len();

            // One ring per polygon. This is the only buffer built here.
            let geom_offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(1usize, len));

            Ok(PolygonArray::new(
                source.coords().clone(),
                geom_offsets,
                source.geom_offsets().clone(),
                source.logical_nulls(),
                typ.metadata().clone(),
            ))
        }
        other if is_untyped(&other) => {
            downcast_geoarrow_array!(array, make_polygon_per_row, output)
        }
        other => Err(GeoArrowError::IncorrectGeometryType(format!(
            "ST_MakePolygon needs a line string argument, got {other:?}"
        ))),
    }
}

fn make_polygon_per_row<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
    output: geoarrow_schema::PolygonType,
) -> GeoArrowResult<PolygonArray> {
    let mut builder = PolygonBuilder::with_capacity(output, Default::default());
    for item in array.iter() {
        let Some(geom) = item else {
            builder.push_polygon(NO_POLYGON)?;
            continue;
        };
        let geom = geom?;
        match geom.as_type() {
            geo_traits::GeometryType::LineString(ring) => {
                let polygon = geo::Polygon::new(ring.to_line_string(), Vec::new());
                builder.push_polygon(Some(&polygon))?;
            }
            // PostGIS raises here. A null keeps a mixed column usable.
            _ => builder.push_polygon(NO_POLYGON)?,
        }
    }
    Ok(builder.finish())
}

/// A typed `None` for the polygon builder.
const NO_POLYGON: Option<&geo::Polygon<f64>> = None;

/// The type `ST_MakePolygon` produces from a given line string type.
pub fn polygon_type_for(input: &GeoArrowType) -> geoarrow_schema::PolygonType {
    geoarrow_schema::PolygonType::new(
        input.dimension().unwrap_or(Dimension::XY),
        Arc::clone(input.metadata()),
    )
    .with_coord_type(input.coord_type().unwrap_or_default())
}

/// `ST_MakeLine`. Join two point columns into a two point line string.
///
/// The coordinates are interleaved row by row, so this one does copy. There is no layout in which
/// two separate point arrays already read as one line string array.
pub fn st_make_line(
    start: &dyn GeoArrowArray,
    end: &dyn GeoArrowArray,
    output: LineStringType,
) -> GeoArrowResult<LineStringArray> {
    let (GeoArrowType::Point(_), GeoArrowType::Point(_)) = (start.data_type(), end.data_type())
    else {
        return Err(GeoArrowError::IncorrectGeometryType(
            "ST_MakeLine needs two point arguments".to_string(),
        ));
    };
    let len = start.len();
    if end.len() != len {
        return Err(GeoArrowError::InvalidGeoArrow(
            "ST_MakeLine needs columns of the same length".to_string(),
        ));
    }

    let dim = output.dimension();
    let size = dim.size();
    let (start_coords, end_coords) = (start.as_point().coords(), end.as_point().coords());

    // Two coordinates per row, laid out one line string at a time.
    let mut values: Vec<Vec<f64>> = vec![Vec::with_capacity(len * 2); size];
    for row in 0..len {
        for (ordinate, column) in values.iter_mut().enumerate() {
            column.push(ordinate_at(start_coords, row, ordinate));
            column.push(ordinate_at(end_coords, row, ordinate));
        }
    }

    let empty = ScalarBuffer::<f64>::from(Vec::<f64>::new());
    let buffers: [ScalarBuffer<f64>; 4] = std::array::from_fn(|index| {
        values
            .get(index)
            .map(|column| ScalarBuffer::from(column.clone()))
            .unwrap_or_else(|| empty.clone())
    });

    let coords = SeparatedCoordBuffer::from_array(buffers, dim)?;
    let geom_offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(2usize, len));
    let nulls = NullBuffer::union(start.logical_nulls().as_ref(), end.logical_nulls().as_ref());

    Ok(LineStringArray::new(
        CoordBuffer::Separated(coords),
        geom_offsets,
        nulls,
        output.metadata().clone(),
    ))
}

/// Read one ordinate out of a coordinate buffer. This builds no scalar.
#[inline]
fn ordinate_at(coords: &CoordBuffer, row: usize, ordinate: usize) -> f64 {
    match coords {
        CoordBuffer::Separated(separated) => {
            let buffer = &separated.raw_buffers()[ordinate];
            buffer.get(row).copied().unwrap_or(0.0)
        }
        CoordBuffer::Interleaved(interleaved) => {
            let stride = interleaved.dim().size();
            interleaved
                .coords()
                .get(row * stride + ordinate)
                .copied()
                .unwrap_or(0.0)
        }
    }
}

/// The type `ST_MakeLine` produces from a point input type.
pub fn line_string_type_for(input: &GeoArrowType) -> LineStringType {
    LineStringType::new(
        input.dimension().unwrap_or(Dimension::XY),
        Arc::clone(input.metadata()),
    )
    .with_coord_type(geoarrow_schema::CoordType::Separated)
}

#[cfg(test)]
mod tests {
    use geo_traits::to_geo::ToGeoGeometry;
    use geoarrow_array::builder::{LineStringBuilder, PointBuilder};
    use geoarrow_array::GeoArrowArrayAccessor;

    use super::*;

    #[test]
    fn make_point_adopts_the_input_buffers() {
        let x = Float64Array::from(vec![1.0, 3.0, 5.0]);
        let y = Float64Array::from(vec![2.0, 4.0, 6.0]);
        let (x_ptr, y_ptr) = (x.values().as_ptr(), y.values().as_ptr());

        let array = st_make_point(&x, &y, None, Default::default()).unwrap();

        let CoordBuffer::Separated(coords) = array.coords() else {
            panic!("expected separated coordinates");
        };
        assert_eq!(
            coords.raw_buffers()[0].as_ptr(),
            x_ptr,
            "ST_MakePoint must adopt the x buffer, not copy it"
        );
        assert_eq!(coords.raw_buffers()[1].as_ptr(), y_ptr);

        let geo::Geometry::Point(p) = array.value(1).unwrap().to_geometry() else {
            panic!("expected a point")
        };
        assert_eq!((p.x(), p.y()), (3.0, 4.0));
    }

    #[test]
    fn make_point_is_null_when_any_ordinate_is_null() {
        let x = Float64Array::from(vec![Some(1.0), None, Some(5.0)]);
        let y = Float64Array::from(vec![Some(2.0), Some(4.0), None]);
        let array = st_make_point(&x, &y, None, Default::default()).unwrap();

        assert!(array.get(0).unwrap().is_some());
        assert!(array.get(1).unwrap().is_none());
        assert!(array.get(2).unwrap().is_none());
    }

    #[test]
    fn make_point_z_keeps_three_dimensions() {
        let x = Float64Array::from(vec![1.0]);
        let y = Float64Array::from(vec![2.0]);
        let z = Float64Array::from(vec![3.0]);
        let array = st_make_point(&x, &y, Some(&z), Default::default()).unwrap();
        assert_eq!(array.data_type().dimension(), Some(Dimension::XYZ));

        let CoordBuffer::Separated(coords) = array.coords() else {
            unreachable!()
        };
        assert_eq!(coords.raw_buffers()[2].as_ref(), &[3.0]);
    }

    #[test]
    fn make_point_rejects_a_length_mismatch() {
        let x = Float64Array::from(vec![1.0, 2.0]);
        let y = Float64Array::from(vec![1.0]);
        assert!(st_make_point(&x, &y, None, Default::default()).is_err());
    }

    #[test]
    fn make_envelope_adopts_all_four_buffers() {
        let xmin = Float64Array::from(vec![0.0]);
        let ymin = Float64Array::from(vec![1.0]);
        let xmax = Float64Array::from(vec![2.0]);
        let ymax = Float64Array::from(vec![3.0]);
        let ptr = xmin.values().as_ptr();

        let array = st_make_envelope(&xmin, &ymin, &xmax, &ymax, Default::default()).unwrap();
        assert_eq!(array.lower().raw_buffers()[0].as_ptr(), ptr);
        assert_eq!(array.upper().raw_buffers()[1].as_ref(), &[3.0]);
    }

    #[test]
    fn make_polygon_shares_the_coordinate_buffer() {
        let rings: Vec<geo::LineString<f64>> = vec![
            geo::wkt! { LINESTRING(0.0 0.0,1.0 0.0,1.0 1.0,0.0 0.0) },
            geo::wkt! { LINESTRING(5.0 5.0,6.0 5.0,6.0 6.0,5.0 5.0) },
        ];
        let lines = LineStringBuilder::from_line_strings(
            &rings,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let CoordBuffer::Separated(before) = lines.coords() else {
            unreachable!()
        };
        let x_ptr = before.raw_buffers()[0].as_ptr();

        let output = polygon_type_for(&lines.data_type());
        let polygons = st_make_polygon(&lines, output).unwrap();
        let CoordBuffer::Separated(after) = polygons.coords() else {
            unreachable!()
        };
        assert_eq!(
            after.raw_buffers()[0].as_ptr(),
            x_ptr,
            "the coordinate buffer must be shared, not copied"
        );
        // The line string offsets became the ring offsets, untouched.
        assert_eq!(polygons.ring_offsets(), lines.geom_offsets());

        let geo::Geometry::Polygon(p) = polygons.value(0).unwrap().to_geometry() else {
            panic!("expected a polygon")
        };
        assert_eq!(p.exterior().0.len(), 4);
        assert_eq!(p.interiors().len(), 0);
    }

    #[test]
    fn make_polygon_rejects_a_point_array() {
        let points = PointBuilder::from_points(
            [geo::point!(x: 0.0, y: 0.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let output = polygon_type_for(&points.data_type());
        assert!(st_make_polygon(&points, output).is_err());
    }

    #[test]
    fn make_line_joins_two_point_columns() {
        let a = PointBuilder::from_points(
            [geo::point!(x: 0.0, y: 0.0), geo::point!(x: 1.0, y: 1.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let b = PointBuilder::from_points(
            [geo::point!(x: 10.0, y: 10.0), geo::point!(x: 11.0, y: 11.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let output = line_string_type_for(&a.data_type());
        let lines = st_make_line(&a, &b, output).unwrap();

        let geo::Geometry::LineString(first) = lines.value(0).unwrap().to_geometry() else {
            panic!("expected a line string")
        };
        assert_eq!(first.0.len(), 2);
        assert_eq!((first.0[0].x, first.0[0].y), (0.0, 0.0));
        assert_eq!((first.0[1].x, first.0[1].y), (10.0, 10.0));
    }
}
