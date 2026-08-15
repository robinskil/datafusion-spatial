//! Two-dimensional bounding boxes.
//!
//! A bounding box test costs four comparisons. An exact topology test builds a graph. So every
//! predicate runs the box test first and calls the exact test only on the survivors.

use geo_traits::{
    CoordTrait, GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait, LineTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait, RectTrait,
    TriangleTrait,
};
use geoarrow_array::array::CoordBuffer;
use geoarrow_array::cast::AsGeoArrowArray;
use geoarrow_array::{downcast_geoarrow_array, GeoArrowArray, GeoArrowArrayAccessor};
use geoarrow_schema::error::GeoArrowResult;
use geoarrow_schema::GeoArrowType;

/// An axis-aligned bounding box in the XY plane.
///
/// Higher dimensions are ignored. PostGIS `&&` and the standard predicates are 2D operators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bbox {
    /// Lowest x value.
    pub minx: f64,
    /// Lowest y value.
    pub miny: f64,
    /// Highest x value.
    pub maxx: f64,
    /// Highest y value.
    pub maxy: f64,
}

impl Bbox {
    /// An inverted box. It intersects nothing, not even itself.
    ///
    /// Null rows and empty geometries both take this value. That makes the box test the null test
    /// as well, so the inner loop needs no extra branch.
    pub const EMPTY: Self = Self {
        minx: f64::INFINITY,
        miny: f64::INFINITY,
        maxx: f64::NEG_INFINITY,
        maxy: f64::NEG_INFINITY,
    };

    /// Returns true when this box holds no coordinate.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.minx > self.maxx
    }

    /// Returns true when the two boxes overlap or touch.
    ///
    /// This is the PostGIS `&&` operator. An empty box returns false against every input.
    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.minx <= other.maxx
            && other.minx <= self.maxx
            && self.miny <= other.maxy
            && other.miny <= self.maxy
    }

    /// Grow this box to hold the coordinate.
    ///
    /// A NaN coordinate fails every comparison, so it never changes the box.
    #[inline]
    pub fn push_xy(&mut self, x: f64, y: f64) {
        if x < self.minx {
            self.minx = x;
        }
        if x > self.maxx {
            self.maxx = x;
        }
        if y < self.miny {
            self.miny = y;
        }
        if y > self.maxy {
            self.maxy = y;
        }
    }

    /// Returns true when this box holds every point of the other box.
    ///
    /// An empty box is contained by nothing and contains nothing.
    #[inline]
    pub fn contains(&self, other: &Self) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && self.minx <= other.minx
            && self.maxx >= other.maxx
            && self.miny <= other.miny
            && self.maxy >= other.maxy
    }

    /// Grow this box by `radius` on every side.
    ///
    /// This is the prefilter for `ST_DWithin`: a geometry further than `radius` from this box
    /// cannot be within `radius` of its contents.
    #[inline]
    pub fn expand(&self, radius: f64) -> Self {
        if self.is_empty() {
            return *self;
        }
        Self {
            minx: self.minx - radius,
            miny: self.miny - radius,
            maxx: self.maxx + radius,
            maxy: self.maxy + radius,
        }
    }

    /// Grow this box to hold the other box.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        if other.minx < self.minx {
            self.minx = other.minx;
        }
        if other.maxx > self.maxx {
            self.maxx = other.maxx;
        }
        if other.miny < self.miny {
            self.miny = other.miny;
        }
        if other.maxy > self.maxy {
            self.maxy = other.maxy;
        }
    }
}

impl Default for Bbox {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Compute the bounding box of one geometry.
///
/// The walk runs over [`geo_traits`], so a GeoArrow scalar needs no cast to `geo_types`.
pub fn bbox_of<G: GeometryTrait<T = f64>>(geom: &G) -> Bbox {
    let mut bbox = Bbox::EMPTY;
    push_geometry(&mut bbox, geom);
    bbox
}

#[inline]
fn push_coord<C: CoordTrait<T = f64>>(bbox: &mut Bbox, coord: &C) {
    bbox.push_xy(coord.x(), coord.y());
}

#[inline]
fn push_line_string<L: LineStringTrait<T = f64>>(bbox: &mut Bbox, line_string: &L) {
    for coord in line_string.coords() {
        push_coord(bbox, &coord);
    }
}

/// Only the exterior ring matters. Every interior ring lies inside it.
#[inline]
fn push_polygon<P: PolygonTrait<T = f64>>(bbox: &mut Bbox, polygon: &P) {
    if let Some(ring) = polygon.exterior() {
        push_line_string(bbox, &ring);
    }
}

fn push_geometry<G: GeometryTrait<T = f64>>(bbox: &mut Bbox, geom: &G) {
    match geom.as_type() {
        GeometryType::Point(p) => {
            if let Some(coord) = p.coord() {
                push_coord(bbox, &coord);
            }
        }
        GeometryType::LineString(ls) => push_line_string(bbox, ls),
        GeometryType::Polygon(p) => push_polygon(bbox, p),
        GeometryType::MultiPoint(mp) => {
            for point in mp.points() {
                if let Some(coord) = point.coord() {
                    push_coord(bbox, &coord);
                }
            }
        }
        GeometryType::MultiLineString(ml) => {
            for line_string in ml.line_strings() {
                push_line_string(bbox, &line_string);
            }
        }
        GeometryType::MultiPolygon(mp) => {
            for polygon in mp.polygons() {
                push_polygon(bbox, &polygon);
            }
        }
        GeometryType::GeometryCollection(gc) => {
            for inner in gc.geometries() {
                push_geometry(bbox, &inner);
            }
        }
        GeometryType::Rect(r) => {
            push_coord(bbox, &r.min());
            push_coord(bbox, &r.max());
        }
        GeometryType::Triangle(t) => {
            for coord in t.coords() {
                push_coord(bbox, &coord);
            }
        }
        GeometryType::Line(l) => {
            for coord in l.coords() {
                push_coord(bbox, &coord);
            }
        }
    }
}

/// Fill `out` with one bounding box per row.
///
/// The vector is cleared first. Pass the same vector across batches to reuse its allocation.
///
/// A null row takes [`Bbox::EMPTY`].
pub fn fill_bboxes(array: &dyn GeoArrowArray, out: &mut Vec<Bbox>) -> GeoArrowResult<()> {
    out.clear();
    out.reserve(array.len());

    // Fast path. A point array with separated coordinates needs no scalar object at all. Read the
    // two f64 buffers straight through.
    if matches!(array.data_type(), GeoArrowType::Point(_)) {
        let points = array.as_point();
        if let CoordBuffer::Separated(coords) = points.coords() {
            let buffers = coords.raw_buffers();
            out.extend(
                buffers[0]
                    .iter()
                    .zip(buffers[1].iter())
                    .map(|(&x, &y)| Bbox {
                        minx: x,
                        miny: y,
                        maxx: x,
                        maxy: y,
                    }),
            );
            blank_nulls(array, out);
            return Ok(());
        }
    }

    // Fast path. Every native type stores the coordinates of a row in one contiguous range.
    // A box is then a min and a max over a plain `f64` slice. The generic walk below matches the
    // coordinate buffer again for every coordinate. That costs more than the comparison it
    // guards. Measured at 2.6 times on a 256 vertex polygon column.
    if let Some(runs) = crate::materialize::row_runs(array) {
        for index in 0..array.len() {
            let (start, end) = runs.range(index);
            out.push(runs.coords.bounds(start, end));
        }
        blank_nulls(array, out);
        return Ok(());
    }

    downcast_geoarrow_array!(array, fill_bboxes_impl, out)
}

/// Overwrite the boxes of null rows with [`Bbox::EMPTY`].
///
/// The fast path reads the coordinate buffers directly, which hold an unspecified value under a
/// null. This restores the null semantics.
fn blank_nulls(array: &dyn GeoArrowArray, out: &mut [Bbox]) {
    if array.logical_null_count() == 0 {
        return;
    }
    if let Some(nulls) = array.logical_nulls() {
        for (index, valid) in nulls.iter().enumerate() {
            if !valid {
                out[index] = Bbox::EMPTY;
            }
        }
    }
}

fn fill_bboxes_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
    out: &mut Vec<Bbox>,
) -> GeoArrowResult<()> {
    for item in array.iter() {
        match item {
            Some(geom) => out.push(bbox_of(&geom?)),
            None => out.push(Bbox::EMPTY),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use geo_traits::to_geo::ToGeoGeometry;
    use geoarrow_array::builder::PointBuilder;
    use geoarrow_schema::{CoordType, Dimension, PointType};

    use super::*;

    fn point_type(coord_type: CoordType) -> PointType {
        PointType::new(Dimension::XY, Default::default()).with_coord_type(coord_type)
    }

    #[test]
    fn empty_box_never_intersects() {
        let unit = Bbox {
            minx: 0.0,
            miny: 0.0,
            maxx: 1.0,
            maxy: 1.0,
        };
        assert!(!Bbox::EMPTY.intersects(&unit));
        assert!(!unit.intersects(&Bbox::EMPTY));
        assert!(!Bbox::EMPTY.intersects(&Bbox::EMPTY));
        assert!(unit.intersects(&unit));
    }

    #[test]
    fn touching_boxes_intersect() {
        let left = Bbox {
            minx: 0.0,
            miny: 0.0,
            maxx: 1.0,
            maxy: 1.0,
        };
        let right = Bbox {
            minx: 1.0,
            miny: 0.0,
            maxx: 2.0,
            maxy: 1.0,
        };
        assert!(left.intersects(&right));
    }

    #[test]
    fn bbox_of_polygon_uses_exterior_ring() {
        let polygon: geo::Polygon<f64> = geo::wkt! {
            POLYGON((0.0 0.0,10.0 0.0,10.0 10.0,0.0 10.0,0.0 0.0),(2.0 2.0,3.0 2.0,3.0 3.0,2.0 2.0))
        };
        let bbox = bbox_of(&geo::Geometry::Polygon(polygon));
        assert_eq!(bbox.minx, 0.0);
        assert_eq!(bbox.miny, 0.0);
        assert_eq!(bbox.maxx, 10.0);
        assert_eq!(bbox.maxy, 10.0);
    }

    #[test]
    fn fill_bboxes_marks_nulls_empty() {
        for coord_type in [CoordType::Separated, CoordType::Interleaved] {
            let p0 = geo::point!(x: 1.0, y: 2.0);
            let p1 = geo::point!(x: 5.0, y: 6.0);
            let array = PointBuilder::from_nullable_points(
                [Some(&p0), None, Some(&p1)].into_iter(),
                point_type(coord_type),
            )
            .finish();

            let mut out = Vec::new();
            fill_bboxes(&array, &mut out).unwrap();

            assert_eq!(out.len(), 3);
            assert_eq!(out[0].minx, 1.0);
            assert_eq!(out[0].maxy, 2.0);
            assert!(out[1].is_empty(), "null row must be empty: {:?}", out[1]);
            assert_eq!(out[2].minx, 5.0);
        }
    }

    #[test]
    fn fill_bboxes_reuses_the_vector() {
        let p0 = geo::point!(x: 1.0, y: 2.0);
        let array =
            PointBuilder::from_points([p0].iter(), point_type(CoordType::Separated)).finish();

        let mut out = vec![Bbox::EMPTY; 128];
        let before = out.as_ptr();
        fill_bboxes(&array, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out.as_ptr(),
            before,
            "clear plus reserve must not reallocate"
        );
    }

    #[test]
    fn bbox_matches_geo_bounding_rect() {
        use geo::BoundingRect;

        let geoms: [geo::Geometry<f64>; 4] = [
            geo::wkt! { POINT(1.0 2.0) }.into(),
            geo::wkt! { LINESTRING(0.0 0.0,3.0 4.0,-1.0 7.0) }.into(),
            geo::wkt! { MULTIPOINT(0.0 0.0,5.0 -2.0) }.into(),
            geo::wkt! { POLYGON((0.0 0.0,4.0 0.0,4.0 4.0,0.0 4.0,0.0 0.0)) }.into(),
        ];
        for geom in geoms {
            let ours = bbox_of(&geom);
            let theirs = geom.bounding_rect().unwrap();
            assert_eq!(ours.minx, theirs.min().x);
            assert_eq!(ours.miny, theirs.min().y);
            assert_eq!(ours.maxx, theirs.max().x);
            assert_eq!(ours.maxy, theirs.max().y);
            // Round trip through geo-traits keeps the same answer.
            assert_eq!(bbox_of(&geom.to_geometry()), ours);
        }
    }
}
