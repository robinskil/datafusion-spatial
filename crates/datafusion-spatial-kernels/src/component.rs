//! Pull one component out of a geometry.
//!
//! `ST_GeometryN`, `ST_PointN`, `ST_StartPoint`, `ST_EndPoint`, `ST_ExteriorRing` and
//! `ST_InteriorRingN`. PostGIS numbers components from one, and an index outside the range gives
//! null rather than an error. Both rules are followed here.

use std::sync::Arc;

use arrow_array::{Array, Int32Array};
use geo_traits::to_geo::{ToGeoGeometry, ToGeoLineString, ToGeoPoint, ToGeoPolygon};
use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PolygonTrait,
};
use geoarrow_array::builder::{GeometryBuilder, LineStringBuilder, PointBuilder};
use geoarrow_array::{downcast_geoarrow_array, GeoArrowArray, GeoArrowArrayAccessor};
use geoarrow_schema::error::GeoArrowResult;
use geoarrow_schema::{
    Dimension, GeoArrowType, GeometryType as GeoGeometryType, LineStringType, PointType,
};

/// A component index that may be the same for every row or vary by row.
///
/// A constant index is the common case. It stays one scalar. The crate does not expand it into
/// a full array of one repeated value.
#[derive(Debug, Clone, Copy)]
pub enum Index<'a> {
    /// The same index for every row.
    Constant(Option<i32>),
    /// One index per row.
    PerRow(&'a Int32Array),
}

impl Index<'_> {
    #[inline]
    fn at(&self, row: usize) -> Option<i32> {
        match self {
            Index::Constant(value) => *value,
            Index::PerRow(array) => {
                if array.is_null(row) {
                    None
                } else {
                    Some(array.value(row))
                }
            }
        }
    }

    /// Turn a PostGIS one-based index into a zero-based one, or `None` when it is out of range.
    #[inline]
    fn zero_based(&self, row: usize, count: usize) -> Option<usize> {
        let raw = self.at(row)?;
        if raw < 1 {
            return None;
        }
        let index = usize::try_from(raw - 1).ok()?;
        (index < count).then_some(index)
    }
}

/// The type `ST_PointN`, `ST_StartPoint` and `ST_EndPoint` produce.
pub fn point_output_type(input: &GeoArrowType) -> PointType {
    PointType::new(
        input.dimension().unwrap_or(Dimension::XY),
        Arc::clone(input.metadata()),
    )
}

/// The type `ST_ExteriorRing` and `ST_InteriorRingN` produce.
pub fn line_string_output_type(input: &GeoArrowType) -> LineStringType {
    LineStringType::new(
        input.dimension().unwrap_or(Dimension::XY),
        Arc::clone(input.metadata()),
    )
}

/// The type `ST_GeometryN` produces.
pub fn geometry_output_type(input: &GeoArrowType) -> GeoGeometryType {
    GeoGeometryType::new(Arc::clone(input.metadata()))
}

/// Which end of a line string to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LineEnd {
    /// The first coordinate.
    Start,
    /// The last coordinate.
    End,
}

/// `ST_StartPoint` and `ST_EndPoint`. Null for anything that is not a line string.
pub fn st_line_end(
    array: &dyn GeoArrowArray,
    end: LineEnd,
    output: PointType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    downcast_geoarrow_array!(array, line_end_impl, end, output)
}

fn line_end_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
    end: LineEnd,
    output: PointType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let mut builder = PointBuilder::with_capacity(output, array.len());
    for item in array.iter() {
        let Some(geom) = item else {
            builder.push_null();
            continue;
        };
        let geom = geom?;
        match geom.as_type() {
            GeometryType::LineString(line) => {
                let count = line.num_coords();
                let position = match end {
                    LineEnd::Start => 0,
                    LineEnd::End => count.saturating_sub(1),
                };
                match (count > 0).then(|| line.coord(position)).flatten() {
                    Some(coord) => builder.push_coord(Some(&coord)),
                    None => builder.push_null(),
                }
            }
            _ => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// `ST_PointN`. The nth coordinate of a line string, numbered from one.
pub fn st_point_n(
    array: &dyn GeoArrowArray,
    index: Index<'_>,
    output: PointType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    downcast_geoarrow_array!(array, point_n_impl, index, output)
}

fn point_n_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
    index: Index<'_>,
    output: PointType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let mut builder = PointBuilder::with_capacity(output, array.len());
    for (row, item) in array.iter().enumerate() {
        let Some(geom) = item else {
            builder.push_null();
            continue;
        };
        let geom = geom?;
        match geom.as_type() {
            GeometryType::LineString(line) => {
                match index
                    .zero_based(row, line.num_coords())
                    .and_then(|position| line.coord(position))
                {
                    Some(coord) => builder.push_coord(Some(&coord)),
                    None => builder.push_null(),
                }
            }
            _ => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// A typed `None` for the line string builder, which has no public `push_null`.
const NO_RING: Option<&geo::LineString<f64>> = None;

/// `ST_ExteriorRing`. The outer ring of a polygon, as a line string.
pub fn st_exterior_ring(
    array: &dyn GeoArrowArray,
    output: LineStringType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    downcast_geoarrow_array!(array, exterior_ring_impl, output)
}

fn exterior_ring_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
    output: LineStringType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let mut builder = LineStringBuilder::with_capacity(output, Default::default());
    for item in array.iter() {
        let Some(geom) = item else {
            builder.push_line_string(NO_RING)?;
            continue;
        };
        let geom = geom?;
        match geom.as_type() {
            GeometryType::Polygon(polygon) => match polygon.exterior() {
                Some(ring) => builder.push_line_string(Some(&ring))?,
                None => builder.push_line_string(NO_RING)?,
            },
            _ => builder.push_line_string(NO_RING)?,
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// `ST_InteriorRingN`. The nth hole of a polygon, numbered from one.
pub fn st_interior_ring_n(
    array: &dyn GeoArrowArray,
    index: Index<'_>,
    output: LineStringType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    downcast_geoarrow_array!(array, interior_ring_n_impl, index, output)
}

fn interior_ring_n_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
    index: Index<'_>,
    output: LineStringType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let mut builder = LineStringBuilder::with_capacity(output, Default::default());
    for (row, item) in array.iter().enumerate() {
        let Some(geom) = item else {
            builder.push_line_string(NO_RING)?;
            continue;
        };
        let geom = geom?;
        match geom.as_type() {
            GeometryType::Polygon(polygon) => {
                match index
                    .zero_based(row, polygon.num_interiors())
                    .and_then(|position| polygon.interior(position))
                {
                    Some(ring) => builder.push_line_string(Some(&ring))?,
                    None => builder.push_line_string(NO_RING)?,
                }
            }
            _ => builder.push_line_string(NO_RING)?,
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// `ST_GeometryN`. The nth part of a collection, numbered from one.
///
/// For a geometry that is not a collection, index one returns the geometry itself and every other
/// index returns null. That matches PostGIS.
pub fn st_geometry_n(
    array: &dyn GeoArrowArray,
    index: Index<'_>,
    output: GeoGeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    downcast_geoarrow_array!(array, geometry_n_impl, index, output)
}

fn geometry_n_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
    index: Index<'_>,
    output: GeoGeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let mut builder = GeometryBuilder::new(output);
    for (row, item) in array.iter().enumerate() {
        let Some(geom) = item else {
            builder.push_null();
            continue;
        };
        let geom = geom?;
        match part_at(&geom, index, row) {
            Some(part) => builder.push_geometry(Some(&part))?,
            None => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// The nth part of a geometry as an owned value.
///
/// Only the requested part is converted, not the whole geometry.
fn part_at<G: GeometryTrait<T = f64>>(
    geom: &G,
    index: Index<'_>,
    row: usize,
) -> Option<geo::Geometry<f64>> {
    match geom.as_type() {
        GeometryType::MultiPoint(mp) => index
            .zero_based(row, mp.num_points())
            .and_then(|n| mp.point(n))
            .map(|p| geo::Geometry::Point(p.to_point())),
        GeometryType::MultiLineString(ml) => index
            .zero_based(row, ml.num_line_strings())
            .and_then(|n| ml.line_string(n))
            .map(|ls| geo::Geometry::LineString(ls.to_line_string())),
        GeometryType::MultiPolygon(mp) => index
            .zero_based(row, mp.num_polygons())
            .and_then(|n| mp.polygon(n))
            .map(|p| geo::Geometry::Polygon(p.to_polygon())),
        GeometryType::GeometryCollection(gc) => index
            .zero_based(row, gc.num_geometries())
            .and_then(|n| gc.geometry(n))
            .map(|g| g.to_geometry()),
        // A single geometry is its own first part.
        _ => index.zero_based(row, 1).map(|_| geom.to_geometry()),
    }
}

#[cfg(test)]
mod tests {
    use geoarrow_array::builder::{GeometryBuilder, LineStringBuilder, PolygonBuilder};
    use geoarrow_array::cast::AsGeoArrowArray;
    use geoarrow_schema::{LineStringType as LsType, PolygonType};

    use super::*;

    fn line_array() -> geoarrow_array::array::LineStringArray {
        let values: Vec<geo::LineString<f64>> = vec![
            geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0,2.0 4.0) },
            geo::wkt! { LINESTRING(5.0 5.0,6.0 6.0) },
        ];
        LineStringBuilder::from_line_strings(
            &values,
            LsType::new(Dimension::XY, Default::default()),
        )
        .finish()
    }

    fn polygon_array() -> geoarrow_array::array::PolygonArray {
        let values: Vec<geo::Polygon<f64>> = vec![
            geo::wkt! { POLYGON((0.0 0.0,4.0 0.0,4.0 4.0,0.0 4.0,0.0 0.0),(1.0 1.0,2.0 1.0,2.0 2.0,1.0 1.0)) },
        ];
        PolygonBuilder::from_polygons(&values, PolygonType::new(Dimension::XY, Default::default()))
            .finish()
    }

    fn first_geometry(array: &dyn GeoArrowArray, row: usize) -> Option<geo::Geometry<f64>> {
        use geo_traits::to_geo::ToGeoGeometry;
        let points = array.as_point();
        points.get(row).unwrap().map(|g| g.to_geometry())
    }

    #[test]
    fn start_and_end_point() {
        let array = line_array();
        let output = point_output_type(&array.data_type());

        let start = st_line_end(&array, LineEnd::Start, output.clone()).unwrap();
        let end = st_line_end(&array, LineEnd::End, output).unwrap();

        let Some(geo::Geometry::Point(p)) = first_geometry(start.as_ref(), 0) else {
            panic!("expected a point")
        };
        assert_eq!((p.x(), p.y()), (0.0, 0.0));

        let Some(geo::Geometry::Point(p)) = first_geometry(end.as_ref(), 0) else {
            panic!("expected a point")
        };
        assert_eq!((p.x(), p.y()), (2.0, 4.0));
    }

    #[test]
    fn point_n_is_one_based_and_clamps_to_null() {
        let array = line_array();
        let output = point_output_type(&array.data_type());

        let second = st_point_n(&array, Index::Constant(Some(2)), output.clone()).unwrap();
        let Some(geo::Geometry::Point(p)) = first_geometry(second.as_ref(), 0) else {
            panic!("expected a point")
        };
        assert_eq!((p.x(), p.y()), (1.0, 1.0));

        // Zero, negative and past the end all give null, as in PostGIS.
        for out_of_range in [0, -1, 4] {
            let result =
                st_point_n(&array, Index::Constant(Some(out_of_range)), output.clone()).unwrap();
            assert!(
                result.as_point().get(0).unwrap().is_none(),
                "index {out_of_range} must give null"
            );
        }
    }

    #[test]
    fn point_n_accepts_a_per_row_index() {
        let array = line_array();
        let output = point_output_type(&array.data_type());
        let indices = Int32Array::from(vec![Some(3), Some(1)]);

        let result = st_point_n(&array, Index::PerRow(&indices), output).unwrap();
        let Some(geo::Geometry::Point(first)) = first_geometry(result.as_ref(), 0) else {
            panic!("expected a point")
        };
        assert_eq!((first.x(), first.y()), (2.0, 4.0));
        let Some(geo::Geometry::Point(second)) = first_geometry(result.as_ref(), 1) else {
            panic!("expected a point")
        };
        assert_eq!((second.x(), second.y()), (5.0, 5.0));
    }

    #[test]
    fn exterior_and_interior_rings() {
        use geo_traits::to_geo::ToGeoGeometry;

        let array = polygon_array();
        let output = line_string_output_type(&array.data_type());

        let shell = st_exterior_ring(&array, output.clone()).unwrap();
        let geo::Geometry::LineString(ring) =
            shell.as_line_string().value(0).unwrap().to_geometry()
        else {
            panic!("expected a line string")
        };
        assert_eq!(ring.0.len(), 5);

        let hole = st_interior_ring_n(&array, Index::Constant(Some(1)), output.clone()).unwrap();
        let geo::Geometry::LineString(ring) = hole.as_line_string().value(0).unwrap().to_geometry()
        else {
            panic!("expected a line string")
        };
        assert_eq!(ring.0.len(), 4);

        // There is only one hole.
        let missing = st_interior_ring_n(&array, Index::Constant(Some(2)), output).unwrap();
        assert!(missing.as_line_string().get(0).unwrap().is_none());
    }

    #[test]
    fn geometry_n_walks_a_collection() {
        use geo_traits::to_geo::ToGeoGeometry;

        let mut builder = GeometryBuilder::new(GeoGeometryType::new(Default::default()));
        builder
            .push_geometry(Some(&geo::Geometry::<f64>::from(
                geo::wkt! { MULTIPOINT(1.0 1.0,2.0 2.0,3.0 3.0) },
            )))
            .unwrap();
        builder
            .push_geometry(Some(&geo::Geometry::<f64>::from(
                geo::wkt! { POINT(9.0 9.0) },
            )))
            .unwrap();
        let array = builder.finish();
        let output = geometry_output_type(&array.data_type());

        let second = st_geometry_n(&array, Index::Constant(Some(2)), output.clone()).unwrap();
        let geo::Geometry::Point(p) = second.as_geometry().value(0).unwrap().to_geometry() else {
            panic!("expected a point")
        };
        assert_eq!((p.x(), p.y()), (2.0, 2.0));

        // A single geometry answers index one with itself.
        let first = st_geometry_n(&array, Index::Constant(Some(1)), output.clone()).unwrap();
        let geo::Geometry::Point(p) = first.as_geometry().value(1).unwrap().to_geometry() else {
            panic!("expected a point")
        };
        assert_eq!((p.x(), p.y()), (9.0, 9.0));

        // And null for any other index.
        let third = st_geometry_n(&array, Index::Constant(Some(2)), output).unwrap();
        assert!(third.as_geometry().get(1).unwrap().is_none());
    }

    #[test]
    fn wrong_type_gives_null_not_an_error() {
        let array = polygon_array();
        let output = point_output_type(&array.data_type());
        let result = st_point_n(&array, Index::Constant(Some(1)), output).unwrap();
        assert!(result.as_point().get(0).unwrap().is_none());
    }
}
