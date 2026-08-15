//! Spatial aggregates. `ST_Extent` for now.
//!
//! `ST_Extent` never builds a geometry. The whole state is four `f64` values, and a merge is four
//! comparisons. That keeps the aggregate cheap enough to run over a whole table.

use geoarrow_array::array::CoordBuffer;
use geoarrow_array::cast::AsGeoArrowArray;
use geoarrow_array::{downcast_geoarrow_array, GeoArrowArray, GeoArrowArrayAccessor};
use geoarrow_schema::error::GeoArrowResult;
use geoarrow_schema::GeoArrowType;

use crate::bbox::{bbox_of, Bbox};
use crate::materialize::{empty_geometry, geometry_filler};

/// The state of `ST_Extent` while it runs.
///
/// The state is [`Copy`] and 32 bytes wide. It fits in registers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extent(Bbox);

impl Extent {
    /// A new, empty extent.
    pub const fn new() -> Self {
        Self(Bbox::EMPTY)
    }

    /// Rebuild the state from four stored values.
    pub const fn from_bounds(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Self {
        Self(Bbox {
            minx,
            miny,
            maxx,
            maxy,
        })
    }

    /// The four values that make up the state.
    pub const fn bounds(&self) -> Bbox {
        self.0
    }

    /// Returns true while no geometry has been seen.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The result of the aggregate, or `None` when every input was null or empty.
    pub fn finish(&self) -> Option<Bbox> {
        if self.0.is_empty() {
            None
        } else {
            Some(self.0)
        }
    }

    /// Merge a partial state from another partition.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.0.merge(&other.0);
    }

    /// Fold a whole array into the state.
    pub fn update(&mut self, array: &dyn GeoArrowArray) -> GeoArrowResult<()> {
        // Fast path. A point column with separated coordinates needs a min and a max over two
        // plain f64 slices. It needs no scalar object. Without nulls it needs no per-row branch.
        if matches!(array.data_type(), GeoArrowType::Point(_)) {
            let points = array.as_point();
            if let CoordBuffer::Separated(coords) = points.coords() {
                let buffers = coords.raw_buffers();
                let (xs, ys) = (&buffers[0], &buffers[1]);

                match array.logical_nulls() {
                    None => {
                        for (&x, &y) in xs.iter().zip(ys.iter()) {
                            self.0.push_xy(x, y);
                        }
                    }
                    Some(nulls) => {
                        for index in nulls.valid_indices() {
                            self.0.push_xy(xs[index], ys[index]);
                        }
                    }
                }
                return Ok(());
            }
        }

        downcast_geoarrow_array!(array, extent_update_impl, &mut self.0)
    }
}

impl Default for Extent {
    fn default() -> Self {
        Self::new()
    }
}

fn extent_update_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
    bbox: &mut Bbox,
) -> GeoArrowResult<()> {
    for item in array.iter().flatten() {
        bbox.merge(&bbox_of(&item?));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use geoarrow_array::builder::{GeometryBuilder, PointBuilder};
    use geoarrow_schema::{CoordType, Dimension, GeometryType, PointType};

    use super::*;

    #[test]
    fn extent_of_points() {
        for coord_type in [CoordType::Separated, CoordType::Interleaved] {
            let p0 = geo::point!(x: 1.0, y: 5.0);
            let p1 = geo::point!(x: -3.0, y: 2.0);
            let array = PointBuilder::from_nullable_points(
                [Some(&p0), None, Some(&p1)].into_iter(),
                PointType::new(Dimension::XY, Default::default()).with_coord_type(coord_type),
            )
            .finish();

            let mut extent = Extent::new();
            extent.update(&array).unwrap();
            let bounds = extent.finish().expect("two valid points");

            assert_eq!(bounds.minx, -3.0);
            assert_eq!(bounds.miny, 2.0);
            assert_eq!(bounds.maxx, 1.0);
            assert_eq!(bounds.maxy, 5.0);
        }
    }

    #[test]
    fn extent_of_mixed_geometries() {
        let mut builder = GeometryBuilder::new(GeometryType::new(Default::default()));
        builder
            .push_geometry(Some(&geo::wkt! { POINT(0.0 0.0) }))
            .unwrap();
        builder
            .push_geometry(Some(
                &geo::wkt! { POLYGON((2.0 2.0,8.0 2.0,8.0 6.0,2.0 6.0,2.0 2.0)) },
            ))
            .unwrap();
        let array = builder.finish();

        let mut extent = Extent::new();
        extent.update(&array).unwrap();
        let bounds = extent.finish().unwrap();
        assert_eq!(bounds.minx, 0.0);
        assert_eq!(bounds.maxx, 8.0);
        assert_eq!(bounds.maxy, 6.0);
    }

    #[test]
    fn merge_matches_a_single_pass() {
        let p0 = geo::point!(x: 1.0, y: 1.0);
        let p1 = geo::point!(x: 4.0, y: 9.0);
        let point_type = PointType::new(Dimension::XY, Default::default());

        let left = PointBuilder::from_points([p0].iter(), point_type.clone()).finish();
        let right = PointBuilder::from_points([p1].iter(), point_type.clone()).finish();
        let both = PointBuilder::from_points([p0, p1].iter(), point_type).finish();

        let mut partial_a = Extent::new();
        partial_a.update(&left).unwrap();
        let mut partial_b = Extent::new();
        partial_b.update(&right).unwrap();
        partial_a.merge(&partial_b);

        let mut single = Extent::new();
        single.update(&both).unwrap();

        assert_eq!(partial_a, single);
    }

    #[test]
    fn all_null_input_yields_none() {
        let none: Option<&geo::Point<f64>> = None;
        let array = PointBuilder::from_nullable_points(
            [none, none].into_iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let mut extent = Extent::new();
        extent.update(&array).unwrap();
        assert!(extent.finish().is_none());
    }

    #[test]
    fn state_round_trips_through_four_values() {
        let extent = Extent::from_bounds(-1.0, -2.0, 3.0, 4.0);
        let bounds = extent.bounds();
        let rebuilt = Extent::from_bounds(bounds.minx, bounds.miny, bounds.maxx, bounds.maxy);
        assert_eq!(extent, rebuilt);
    }
}

// ------------------------------------------------- geometry-valued aggregates

/// The state of `ST_Collect` while it runs.
///
/// # Why the state is WKB
///
/// A DataFusion aggregate hands its partial state to another partition as a [`ScalarValue`].
/// The state must therefore survive a round trip through Arrow. WKB is compact and lossless, and
/// both sides already read it. So a partial collection crosses as one binary value.
///
/// [`ScalarValue`]: https://docs.rs/datafusion/latest/datafusion/common/enum.ScalarValue.html
#[derive(Debug, Default)]
pub struct Collect {
    parts: Vec<geo::Geometry<f64>>,
}

impl Collect {
    /// A new, empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true while nothing has been collected.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Add every geometry of one array.
    ///
    /// This aggregate keeps every geometry, so it cannot reuse one buffer. It still takes the
    /// fast fill, which reads the coordinates of a row as a plain slice instead of one
    /// `CoordBuffer` match per coordinate. Measured at 6.0 times on a 256 vertex polygon column.
    pub fn update(&mut self, array: &dyn GeoArrowArray) -> GeoArrowResult<()> {
        let filler = geometry_filler(array)?;
        for index in 0..array.len() {
            let mut geom = empty_geometry();
            if filler(index, &mut geom)? {
                self.parts.push(geom);
            }
        }
        Ok(())
    }

    /// Merge a partial state from another partition.
    pub fn merge(&mut self, other: Collect) {
        self.parts.extend(other.parts);
    }

    /// The collected geometries as one collection, or `None` when nothing was collected.
    pub fn finish(self) -> Option<geo::Geometry<f64>> {
        if self.parts.is_empty() {
            return None;
        }
        Some(geo::Geometry::GeometryCollection(
            geo::GeometryCollection::new_from(self.parts),
        ))
    }

    /// Encode the partial state for transfer between partitions.
    pub fn to_wkb(&self) -> GeoArrowResult<Vec<u8>> {
        let collection = geo::Geometry::GeometryCollection(geo::GeometryCollection::new_from(
            self.parts.clone(),
        ));
        write_wkb(&collection)
    }

    /// Decode a partial state produced by [`Self::to_wkb`].
    pub fn from_wkb(bytes: &[u8]) -> GeoArrowResult<Self> {
        let parts = match read_wkb(bytes)? {
            geo::Geometry::GeometryCollection(collection) => collection.0,
            other => vec![other],
        };
        Ok(Self { parts })
    }
}

/// The state of the `ST_Union` aggregate while it runs.
///
/// Every partial result is areal, so the state is one multi polygon and a merge is one boolean
/// union. The aggregate skips non-areal input, as the scalar `ST_Union` does.
#[derive(Debug, Default)]
pub struct UnionAll {
    shape: Option<geo::MultiPolygon<f64>>,
}

impl UnionAll {
    /// A new, empty union.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true while nothing has been unioned.
    pub fn is_empty(&self) -> bool {
        self.shape.is_none()
    }

    /// Fold every geometry of one array into the union.
    ///
    /// `areal_of` copies what it needs, so the row geometry does not outlive the iteration. One
    /// geometry therefore serves every row.
    pub fn update(&mut self, array: &dyn GeoArrowArray) -> GeoArrowResult<()> {
        let filler = geometry_filler(array)?;
        let mut row = empty_geometry();
        for index in 0..array.len() {
            if filler(index, &mut row)? {
                if let Some(areal) = areal_of(&row) {
                    self.absorb(areal);
                }
            }
        }
        Ok(())
    }

    /// Merge a partial state from another partition.
    pub fn merge(&mut self, other: UnionAll) {
        if let Some(shape) = other.shape {
            self.absorb(shape);
        }
    }

    fn absorb(&mut self, other: geo::MultiPolygon<f64>) {
        use geo::BooleanOps;
        self.shape = Some(match self.shape.take() {
            Some(current) => current.union(&other),
            None => other,
        });
    }

    /// The union, or `None` when no areal geometry was seen.
    pub fn finish(self) -> Option<geo::Geometry<f64>> {
        self.shape.map(geo::Geometry::MultiPolygon)
    }

    /// Encode the partial state for transfer between partitions.
    pub fn to_wkb(&self) -> GeoArrowResult<Vec<u8>> {
        let shape = self
            .shape
            .clone()
            .unwrap_or_else(|| geo::MultiPolygon::new(Vec::new()));
        write_wkb(&geo::Geometry::MultiPolygon(shape))
    }

    /// Decode a partial state produced by [`Self::to_wkb`].
    pub fn from_wkb(bytes: &[u8]) -> GeoArrowResult<Self> {
        let shape = match read_wkb(bytes)? {
            geo::Geometry::MultiPolygon(polygons) if polygons.0.is_empty() => None,
            geo::Geometry::MultiPolygon(polygons) => Some(polygons),
            other => areal_of(&other),
        };
        Ok(Self { shape })
    }
}

/// View a geometry as a multi polygon, or `None` when it has no area.
fn areal_of(geom: &geo::Geometry<f64>) -> Option<geo::MultiPolygon<f64>> {
    match geom {
        geo::Geometry::Polygon(polygon) => Some(geo::MultiPolygon::new(vec![polygon.clone()])),
        geo::Geometry::MultiPolygon(polygons) => Some(polygons.clone()),
        geo::Geometry::Rect(rect) => Some(geo::MultiPolygon::new(vec![rect.to_polygon()])),
        geo::Geometry::Triangle(triangle) => {
            Some(geo::MultiPolygon::new(vec![triangle.to_polygon()]))
        }
        _ => None,
    }
}

/// Encode one geometry as WKB.
pub fn write_wkb(geom: &geo::Geometry<f64>) -> GeoArrowResult<Vec<u8>> {
    let mut bytes = Vec::new();
    wkb::writer::write_geometry(&mut bytes, geom, &Default::default())
        .map_err(|err| geoarrow_schema::error::GeoArrowError::External(Box::new(err)))?;
    Ok(bytes)
}

/// Decode one geometry from WKB.
pub fn read_wkb(bytes: &[u8]) -> GeoArrowResult<geo::Geometry<f64>> {
    use geo_traits::to_geo::ToGeoGeometry;
    let parsed = wkb::reader::read_wkb(bytes)
        .map_err(|err| geoarrow_schema::error::GeoArrowError::External(Box::new(err)))?;
    Ok(parsed.to_geometry())
}

#[cfg(test)]
mod geometry_aggregate_tests {
    use geoarrow_array::builder::PolygonBuilder;
    use geoarrow_schema::{Dimension, PolygonType};

    use super::*;

    fn squares(values: Vec<geo::Polygon<f64>>) -> geoarrow_array::array::PolygonArray {
        PolygonBuilder::from_polygons(&values, PolygonType::new(Dimension::XY, Default::default()))
            .finish()
    }

    fn unit() -> geo::Polygon<f64> {
        geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 1.0,0.0 0.0)) }
    }

    fn shifted() -> geo::Polygon<f64> {
        geo::wkt! { POLYGON((0.5 0.0,1.5 0.0,1.5 1.0,0.5 1.0,0.5 0.0)) }
    }

    #[test]
    fn collect_gathers_every_row() {
        let array = squares(vec![unit(), shifted()]);
        let mut state = Collect::new();
        state.update(&array).unwrap();

        let Some(geo::Geometry::GeometryCollection(collection)) = state.finish() else {
            panic!("expected a collection")
        };
        assert_eq!(collection.0.len(), 2);
    }

    #[test]
    fn collect_of_nothing_is_none() {
        let state = Collect::new();
        assert!(state.is_empty());
        assert!(state.finish().is_none());
    }

    /// A partial state must survive the trip between partitions.
    #[test]
    fn collect_state_round_trips_through_wkb() {
        let array = squares(vec![unit(), shifted()]);
        let mut left = Collect::new();
        left.update(&array).unwrap();

        let encoded = left.to_wkb().unwrap();
        let decoded = Collect::from_wkb(&encoded).unwrap();

        let Some(geo::Geometry::GeometryCollection(collection)) = decoded.finish() else {
            panic!("expected a collection")
        };
        assert_eq!(collection.0.len(), 2);
    }

    #[test]
    fn collect_merge_matches_a_single_pass() {
        let left_array = squares(vec![unit()]);
        let right_array = squares(vec![shifted()]);
        let both = squares(vec![unit(), shifted()]);

        let mut partial = Collect::new();
        partial.update(&left_array).unwrap();
        let mut other = Collect::new();
        other.update(&right_array).unwrap();
        partial.merge(other);

        let mut single = Collect::new();
        single.update(&both).unwrap();

        assert_eq!(
            format!("{:?}", partial.finish()),
            format!("{:?}", single.finish())
        );
    }

    #[test]
    fn union_merges_overlapping_shapes() {
        use geo::Area;

        let array = squares(vec![unit(), shifted()]);
        let mut state = UnionAll::new();
        state.update(&array).unwrap();

        let union = state.finish().expect("two polygons were seen");
        // The two unit squares half overlap, so they cover 1.5.
        assert!((union.unsigned_area() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn union_merge_matches_a_single_pass() {
        use geo::Area;

        let mut partial = UnionAll::new();
        partial.update(&squares(vec![unit()])).unwrap();
        let mut other = UnionAll::new();
        other.update(&squares(vec![shifted()])).unwrap();
        partial.merge(other);

        let mut single = UnionAll::new();
        single.update(&squares(vec![unit(), shifted()])).unwrap();

        let merged = partial.finish().unwrap().unsigned_area();
        let direct = single.finish().unwrap().unsigned_area();
        assert!((merged - direct).abs() < 1e-9);
    }

    #[test]
    fn union_state_round_trips_through_wkb() {
        use geo::Area;

        let mut state = UnionAll::new();
        state.update(&squares(vec![unit(), shifted()])).unwrap();

        let encoded = state.to_wkb().unwrap();
        let decoded = UnionAll::from_wkb(&encoded).unwrap();
        assert!((decoded.finish().unwrap().unsigned_area() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn union_of_nothing_is_none() {
        let state = UnionAll::new();
        assert!(state.is_empty());
        assert!(state.finish().is_none());

        // An empty state still encodes and decodes.
        let encoded = UnionAll::new().to_wkb().unwrap();
        assert!(UnionAll::from_wkb(&encoded).unwrap().finish().is_none());
    }
}
