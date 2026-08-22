//! Spatial predicates.
//!
//! # Why a bounding box first, and why not the same box test for every predicate
//!
//! An exact test builds a topology graph. It can also walk every edge pair. A box test is four
//! comparisons over a flat `f64` slice. Real data is mostly disjoint. So the box test answers
//! most rows, and the exact test never runs.
//!
//! The box test is not the same for every predicate, and a generic "do the boxes overlap" filter
//! throws away most of its value:
//!
//! | Predicate | What the boxes can settle on their own |
//! |---|---|
//! | `ST_Intersects` | disjoint boxes mean false |
//! | `ST_Disjoint` | disjoint boxes mean **true**, answered without any geometry |
//! | `ST_Contains`, `ST_Covers` | the right box must sit inside the left one, or false |
//! | `ST_Within`, `ST_CoveredBy` | the left box must sit inside the right one, or false |
//! | `ST_Equals` | the boxes must match exactly, or false |
//! | `ST_Touches`, `ST_Crosses`, `ST_Overlaps` | disjoint boxes mean false |
//!
//! Containment and equality get a far stronger filter than plain overlap.
//!
//! # Which algorithm backs each predicate
//!
//! `geo` offers two families. [`Relate`] computes the whole DE-9IM matrix. The direct traits
//! ([`Intersects`], [`Contains`], [`Covers`]) answer one question. They build no graph.
//!
//! Measured over 8192 point probes against a 256 vertex polygon:
//!
//! | Path | Time |
//! |---|---|
//! | `relate` without the R-tree | 309 ms |
//! | `relate` with the R-tree | 9.60 ms |
//! | `Intersects` | 1.43 ms |
//!
//! So a predicate uses a direct algorithm whenever one exists, and the R-tree is built only for
//! the predicates that genuinely need the matrix: `ST_Touches`, `ST_Crosses`, `ST_Overlaps`,
//! `ST_Equals` and `ST_Relate`. For those, the cache is worth 32 times its cost.
//!
//! # Why a point against a polygon takes a third path
//!
//! A direct trait still reads every edge of the ring for every row. A 5000 vertex coastline costs
//! 20 times a 256 vertex ring. The winding number rule reads an edge only when the y interval of
//! that edge holds the y of the point, so an index over that interval drops the rest before the
//! loop starts.
//!
//! PostGIS takes the same short circuit in `liblwgeom/intervaltree.c`, under the same two
//! conditions: the outer geometry is polygonal and the inner geometry is a point.
//! [`PreparedLiteral`] builds [`PointInPolygonIndex`] on the first such row and reuses it for the
//! batch. One verdict answers every direct predicate, through [`Predicate::point_rule`].
//!
//! Measured over 8192 point probes against a 5000 vertex ring:
//!
//! | Predicate | Before | After |
//! |---|--:|--:|
//! | `ST_Within` | 19.2 ms | 364 us |
//! | `ST_Contains` | 19.5 ms | 360 us |
//! | `ST_Intersects` | 19.6 ms | 359 us |
//! | `ST_Disjoint` | 19.5 ms | 359 us |
//! | `ST_Covers` | 7.08 s | 360 us |
//! | `ST_CoveredBy` | 7.09 s | 362 us |
//!
//! `ST_Covers` starts four orders of magnitude behind the rest, and the reason is worth knowing.
//! `geo` has no direct algorithm for [`Covers`] between two [`Geometry`] values. It answers that
//! pair from the DE-9IM matrix. [`Predicate::needs_relate`] reports false for `ST_Covers`, so the
//! literal never builds the R-tree either. The pair therefore ran the full graph, unindexed, once
//! per row. The verdict removes that whole path for a point probe. It does not remove it for a
//! column of polygons, which is still open.
//!
//! # Why a repeated point row costs nothing
//!
//! A point column often repeats a coordinate on neighbouring rows. A denormalized table that
//! carries the location of a store or a sensor on every event row looks exactly like that.
//!
//! For a point the bounding box is the coordinate, and [`st_predicate_scalar`] has already read
//! the box before it builds anything. So two comparisons settle whether this row repeats the one
//! before it, and a repeat reuses that answer. It skips the geometry build as well as the exact
//! test, which is the larger half of the row.
//!
//! Over 8192 rows against a 5000 vertex ring, a column of one repeated point runs twelve times
//! faster. A column of distinct points pays between minus one and plus two per cent, which is
//! inside the run to run noise of the benchmark itself.
//!
//! `benches/caching.rs` prices the check as `row_repeats/*`.
//!
//! # Why the prepared geometry lives on the stack
//!
//! [`geo::PreparedGeometry`] holds its R-tree behind an [`Rc`][std::rc::Rc], so it is neither
//! `Send` nor `Sync`. It cannot be cached in a DataFusion UDF struct, which must be both. So
//! [`PreparedLiteral`] is built once per call and dropped at the end. One batch is 8192 rows by
//! default, so the build is amortized over the whole batch.

use std::cell::OnceCell;

use arrow_array::builder::StringBuilder;
use arrow_array::{BooleanArray, StringArray};
use arrow_buffer::{BooleanBufferBuilder, NullBuffer};
use geo::coordinate_position::CoordPos;
use geo::relate::IntersectionMatrix;
use geo::{
    Contains, ContainsProperly, CoordsIter, Covers, Distance, Euclidean, Geometry, Intersects,
    PreparedGeometry, Relate,
};
use geo_traits::to_geo::ToGeoGeometry;
use geoarrow_array::{downcast_geoarrow_array, GeoArrowArray, GeoArrowArrayAccessor};
use geoarrow_schema::error::{GeoArrowError, GeoArrowResult};
use geoarrow_schema::GeoArrowType;

use crate::bbox::{bbox_of, fill_bboxes, Bbox};
use crate::index::PointInPolygonIndex;
use crate::materialize::{empty_geometry, geometry_filler, GeometryFiller, GeometryReader};

/// Reusable buffers for a predicate over two arrays.
///
/// Hold one of these across batches to keep the box vectors allocated.
#[derive(Debug, Default)]
pub struct PredicateScratch {
    left: Vec<Bbox>,
    right: Vec<Bbox>,
}

impl PredicateScratch {
    /// Create empty scratch buffers.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Which side of a binary predicate a constant argument sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    /// The constant is the first argument.
    Left,
    /// The constant is the second argument.
    Right,
}

/// A two-argument spatial predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Predicate {
    /// `ST_Intersects`.
    Intersects,
    /// `ST_Disjoint`.
    Disjoint,
    /// `ST_Contains`.
    Contains,
    /// `ST_ContainsProperly`.
    ContainsProperly,
    /// `ST_Within`.
    Within,
    /// `ST_Covers`.
    Covers,
    /// `ST_CoveredBy`.
    CoveredBy,
    /// `ST_Touches`.
    Touches,
    /// `ST_Crosses`.
    Crosses,
    /// `ST_Overlaps`.
    Overlaps,
    /// `ST_Equals`.
    Equals,
}

/// How an indexed point-in-polygon verdict answers one predicate.
///
/// The index reports one of three positions for a point against a polygonal constant. Each direct
/// predicate reads that one verdict, so no row walks the rings. See
/// [`Predicate::point_rule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointRule {
    /// True inside only. `ST_Contains`, `ST_ContainsProperly` and `ST_Within`.
    Inside,
    /// True inside and on the boundary. `ST_Intersects`, `ST_Covers` and `ST_CoveredBy`.
    NotOutside,
    /// True outside only. `ST_Disjoint`.
    Outside,
}

impl PointRule {
    /// Read one verdict.
    #[inline]
    pub fn read(self, position: CoordPos) -> bool {
        match self {
            Self::Inside => position == CoordPos::Inside,
            Self::NotOutside => position != CoordPos::Outside,
            Self::Outside => position == CoordPos::Outside,
        }
    }
}

impl Predicate {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::Intersects => "ST_Intersects",
            Self::Disjoint => "ST_Disjoint",
            Self::Contains => "ST_Contains",
            Self::ContainsProperly => "ST_ContainsProperly",
            Self::Within => "ST_Within",
            Self::Covers => "ST_Covers",
            Self::CoveredBy => "ST_CoveredBy",
            Self::Touches => "ST_Touches",
            Self::Crosses => "ST_Crosses",
            Self::Overlaps => "ST_Overlaps",
            Self::Equals => "ST_Equals",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::Intersects => "st_intersects",
            Self::Disjoint => "st_disjoint",
            Self::Contains => "st_contains",
            Self::ContainsProperly => "st_containsproperly",
            Self::Within => "st_within",
            Self::Covers => "st_covers",
            Self::CoveredBy => "st_coveredby",
            Self::Touches => "st_touches",
            Self::Crosses => "st_crosses",
            Self::Overlaps => "st_overlaps",
            Self::Equals => "st_equals",
        }
    }

    /// Every predicate, for registration.
    pub const ALL: [Self; 11] = [
        Self::Intersects,
        Self::Disjoint,
        Self::Contains,
        Self::ContainsProperly,
        Self::Within,
        Self::Covers,
        Self::CoveredBy,
        Self::Touches,
        Self::Crosses,
        Self::Overlaps,
        Self::Equals,
    ];

    /// True when the exact test needs the full DE-9IM matrix.
    ///
    /// Only these benefit from the R-tree in [`PreparedLiteral`].
    pub const fn needs_relate(self) -> bool {
        matches!(
            self,
            Self::Touches | Self::Crosses | Self::Overlaps | Self::Equals
        )
    }

    /// How a point-in-polygon verdict answers this predicate, when the areal argument is the
    /// constant one.
    ///
    /// `areal_side` is the side the polygonal constant sits on. `None` means the index cannot
    /// answer: the constant is on the wrong side of a one-way predicate, or the predicate needs
    /// the DE-9IM matrix.
    ///
    /// PostGIS reads its own interval tree through the same table. A point has no interior of its
    /// own, so `ST_Contains` and `ST_Within` reduce to one question, and so do `ST_Covers`,
    /// `ST_CoveredBy` and `ST_Intersects`.
    pub const fn point_rule(self, areal_side: Side) -> Option<PointRule> {
        match (self, areal_side) {
            // The constant holds the point. Only the interior counts.
            (Self::Contains | Self::ContainsProperly, Side::Left) => Some(PointRule::Inside),
            (Self::Within, Side::Right) => Some(PointRule::Inside),
            // The boundary counts as well.
            (Self::Covers, Side::Left) => Some(PointRule::NotOutside),
            (Self::CoveredBy, Side::Right) => Some(PointRule::NotOutside),
            // Symmetric, so the constant may sit on either side.
            (Self::Intersects, _) => Some(PointRule::NotOutside),
            (Self::Disjoint, _) => Some(PointRule::Outside),
            // Either the constant is the inner argument, which no index of the constant can
            // answer, or the predicate needs the matrix.
            _ => None,
        }
    }

    /// True when a swap of the two arguments cannot change the answer.
    ///
    /// A symmetric predicate lets a constant argument always take the cached side, whichever way
    /// round the query wrote it.
    pub const fn is_symmetric(self) -> bool {
        matches!(
            self,
            Self::Intersects
                | Self::Disjoint
                | Self::Touches
                | Self::Crosses
                | Self::Overlaps
                | Self::Equals
        )
    }

    /// What the bounding boxes alone can settle, if anything.
    ///
    /// `Some(answer)` skips the exact test for that row.
    #[inline]
    pub fn bbox_verdict(self, left: &Bbox, right: &Bbox) -> Option<bool> {
        let overlap = left.intersects(right);
        match self {
            Self::Intersects => (!overlap).then_some(false),
            // The one predicate a disjoint pair of boxes proves outright.
            Self::Disjoint => (!overlap).then_some(true),
            // The inner side must fit inside the box of the outer side.
            Self::Contains | Self::ContainsProperly | Self::Covers => {
                (!left.contains(right)).then_some(false)
            }
            Self::Within | Self::CoveredBy => (!right.contains(left)).then_some(false),
            // Equal geometries have equal boxes.
            Self::Equals => (left != right).then_some(false),
            Self::Touches | Self::Crosses | Self::Overlaps => (!overlap).then_some(false),
        }
    }

    /// The exact test, with no cached index.
    #[inline]
    pub fn evaluate(self, left: &Geometry<f64>, right: &Geometry<f64>) -> bool {
        match self {
            Self::Intersects => left.intersects(right),
            Self::Disjoint => !left.intersects(right),
            Self::Contains => left.contains(right),
            Self::ContainsProperly => left.contains_properly(right),
            // `Within` is defined as the converse of `Contains`, so this is the same call.
            Self::Within => right.contains(left),
            Self::Covers => left.covers(right),
            Self::CoveredBy => right.covers(left),
            Self::Touches | Self::Crosses | Self::Overlaps | Self::Equals => {
                self.read_matrix(&left.relate(right))
            }
        }
    }

    /// Read this predicate out of an already computed DE-9IM matrix.
    #[inline]
    fn read_matrix(self, matrix: &IntersectionMatrix) -> bool {
        match self {
            Self::Intersects => matrix.is_intersects(),
            Self::Disjoint => matrix.is_disjoint(),
            Self::Contains => matrix.is_contains(),
            Self::ContainsProperly => matrix.is_contains_properly(),
            Self::Within => matrix.is_within(),
            Self::Covers => matrix.is_covers(),
            Self::CoveredBy => matrix.is_coveredby(),
            Self::Touches => matrix.is_touches(),
            Self::Crosses => matrix.is_crosses(),
            Self::Overlaps => matrix.is_overlaps(),
            Self::Equals => matrix.is_equal_topo(),
        }
    }
}

/// A constant geometry, held for repeated tests against every row of a batch.
///
/// The bounding box is computed once at construction. The R-tree is built on the first
/// [`relate`][Self::relate] call and never for a direct predicate. See the module documentation
/// for the measurements behind that split.
pub struct PreparedLiteral {
    geometry: Geometry<f64>,
    bbox: Bbox,
    coord_count: usize,
    /// Built on demand. `None` inside means the geometry is too small to be worth an R-tree.
    prepared: OnceCell<Option<Box<PreparedGeometry<'static, Geometry<f64>>>>>,
    /// Built on demand for a point probe. `None` inside means the literal is not polygonal, or is
    /// too small to be worth an edge index.
    point_index: OnceCell<Option<Box<PointInPolygonIndex>>>,
}

impl std::fmt::Debug for PreparedLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedLiteral")
            .field("geometry", &self.geometry)
            .field("bbox", &self.bbox)
            .field("indexed", &self.prepared.get().is_some())
            .field("point_indexed", &self.has_point_index())
            .finish()
    }
}

impl PreparedLiteral {
    /// Coordinate count at which the R-tree starts to pay for itself on a `relate` call.
    ///
    /// Confirm this number with `cargo bench --bench predicates` before you change it.
    pub const PREPARE_THRESHOLD: usize = 32;

    /// Coordinate count at which the point-in-polygon index starts to pay for itself.
    ///
    /// Below this the walk over every edge is already cheap, and the index build is loss.
    /// Measured over 8192 distinct point probes, with the build inside the timed loop:
    ///
    /// | Ring | Coordinates | `geo::Contains` | Indexed | Change |
    /// |---|--:|--:|--:|--:|
    /// | 8 vertices | 9 | 255 us | 312 us | +22% |
    /// | 12 vertices | 13 | 294 us | 309 us | +5% |
    /// | 14 vertices | 15 | 309 us | 300 us | -3% |
    /// | 16 vertices | 17 | 324 us | 306 us | -5% |
    /// | 24 vertices | 25 | 370 us | 310 us | -16% |
    /// | 64 vertices | 65 | 528 us | 306 us | -42% |
    /// | 5000 vertices | 5001 | 19.6 ms | 379 us | -98% |
    ///
    /// About 300 us of each figure reads the rows, which no predicate avoids. The indexed column
    /// is nearly flat, because the probe no longer grows with the ring.
    ///
    /// The two paths cross between 12 and 14 vertices. This constant counts coordinates, and a
    /// closed ring of `n` vertices holds `n + 1` of them, so 16 sits just above the crossing.
    /// Confirm the number with `cargo bench --bench predicates` before you change it.
    ///
    /// The figures above use `ST_Contains`. See the module documentation for the other
    /// predicates, and for why `ST_Covers` starts four orders of magnitude further back.
    pub const POINT_INDEX_THRESHOLD: usize = 16;

    /// Hold a literal geometry for repeated tests.
    ///
    /// This does no index work. Construction is cheap.
    pub fn new(geometry: Geometry<f64>) -> Self {
        let bbox = bbox_of(&geometry);
        let coord_count = geometry.coords_count();
        Self {
            geometry,
            bbox,
            coord_count,
            prepared: OnceCell::new(),
            point_index: OnceCell::new(),
        }
    }

    /// The bounding box of the literal. Computed once, tested against every row.
    pub fn bbox(&self) -> Bbox {
        self.bbox
    }

    /// The underlying geometry.
    pub fn geometry(&self) -> &Geometry<f64> {
        &self.geometry
    }

    /// Returns true when this literal can never intersect anything.
    pub fn is_empty(&self) -> bool {
        self.bbox.is_empty()
    }

    /// Exact intersection test through the direct algorithm.
    #[inline]
    pub fn intersects(&self, other: &Geometry<f64>) -> bool {
        self.geometry.intersects(other)
    }

    /// Full DE-9IM matrix against one geometry, with this literal as argument A.
    ///
    /// The first call builds the R-tree when the literal is large enough. Later calls reuse it.
    pub fn relate(&self, other: &Geometry<f64>) -> IntersectionMatrix {
        match self.index() {
            Some(prepared) => prepared.relate(other),
            None => self.geometry.relate(other),
        }
    }

    /// Evaluate a predicate against one row. The argument order of the query is kept.
    ///
    /// A DE-9IM predicate goes through the cached matrix. Everything else takes its direct
    /// algorithm, because that is faster even against the cache.
    #[inline]
    pub fn evaluate(
        &self,
        predicate: Predicate,
        other: &Geometry<f64>,
        literal_side: Side,
    ) -> bool {
        if predicate.needs_relate() {
            // Every relate-backed predicate here is symmetric, so the literal can always be
            // argument A and keep the benefit of the cached graph.
            debug_assert!(predicate.is_symmetric());
            return predicate.read_matrix(&self.relate(other));
        }
        // A point against a polygonal constant is the shape PostGIS short circuits. One indexed
        // verdict answers every direct predicate, so the row reads the edges that cross its own y
        // and never walks a ring.
        if let Geometry::Point(point) = other {
            if let Some(rule) = predicate.point_rule(literal_side) {
                if let Some(index) = self.point_index() {
                    return rule.read(index.locate(point.0));
                }
            }
        }
        match literal_side {
            Side::Left => predicate.evaluate(&self.geometry, other),
            Side::Right => predicate.evaluate(other, &self.geometry),
        }
    }

    /// Returns true once the R-tree exists. Test hook.
    pub fn is_indexed(&self) -> bool {
        matches!(self.prepared.get(), Some(Some(_)))
    }

    /// Returns true once the point-in-polygon index exists. Test hook.
    pub fn has_point_index(&self) -> bool {
        matches!(self.point_index.get(), Some(Some(_)))
    }

    /// The edge index over the literal, built on the first point probe.
    ///
    /// `None` when the literal is not polygonal, or holds too few vertices to repay the build.
    fn point_index(&self) -> Option<&PointInPolygonIndex> {
        self.point_index
            .get_or_init(|| {
                if self.coord_count < Self::POINT_INDEX_THRESHOLD {
                    return None;
                }
                PointInPolygonIndex::new(&self.geometry).map(Box::new)
            })
            .as_deref()
    }

    fn index(&self) -> Option<&PreparedGeometry<'static, Geometry<f64>>> {
        self.prepared
            .get_or_init(|| {
                if self.coord_count >= Self::PREPARE_THRESHOLD {
                    Some(Box::new(PreparedGeometry::from(self.geometry.clone())))
                } else {
                    None
                }
            })
            .as_deref()
    }
}

/// Any predicate over two arrays of the same length.
pub fn st_predicate(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    predicate: Predicate,
) -> GeoArrowResult<BooleanArray> {
    st_predicate_with(left, right, predicate, &mut PredicateScratch::new())
}

/// Any predicate over two arrays, with caller-owned scratch buffers.
pub fn st_predicate_with(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    predicate: Predicate,
    scratch: &mut PredicateScratch,
) -> GeoArrowResult<BooleanArray> {
    let len = broadcast_len(predicate.function_name(), left, right)?;
    // One null buffer for both sides. This removes a virtual call per row.
    let nulls = broadcast_nulls(left, right, len);

    fill_bboxes(left, &mut scratch.left)?;
    fill_bboxes(right, &mut scratch.right)?;
    let left_box = BoxSide::new(&scratch.left, len);
    let right_box = BoxSide::new(&scratch.right, len);

    // One downcast per side, not one per row. The operand holds the concrete array.
    let mut left_geom = Operand::new(left, len)?;
    let mut right_geom = Operand::new(right, len)?;

    let mut values = BooleanBufferBuilder::new(len);
    for index in 0..len {
        // A null row carries Bbox::EMPTY. The value written for it is masked by the null buffer.
        if let Some(answer) = predicate.bbox_verdict(&left_box.at(index), &right_box.at(index)) {
            values.append(answer);
            continue;
        }
        match (left_geom.get(index)?, right_geom.get(index)?) {
            (Some(lhs), Some(rhs)) => values.append(predicate.evaluate(lhs, rhs)),
            _ => values.append(false),
        }
    }

    Ok(BooleanArray::new(values.finish(), nulls))
}

/// Any predicate between an array and one constant geometry.
///
/// `literal_side` records which argument of the query the constant was, so a non-symmetric
/// predicate such as `ST_Contains` keeps its meaning.
pub fn st_predicate_scalar(
    array: &dyn GeoArrowArray,
    literal: &PreparedLiteral,
    predicate: Predicate,
    literal_side: Side,
    scratch: &mut PredicateScratch,
) -> GeoArrowResult<BooleanArray> {
    let len = array.len();
    let nulls = array.logical_nulls();
    let literal_bbox = literal.bbox();

    fill_bboxes(array, &mut scratch.left)?;
    let filler = geometry_filler(array)?;
    // One geometry for the whole batch. Each row that survives the box test refills it in place.
    let mut row = empty_geometry();

    // A point column often repeats a coordinate on neighbouring rows. A denormalized table that
    // carries the location of a store or a sensor on every event row looks exactly like that.
    //
    // For a point the bounding box is the coordinate, and the loop has already read the box. So
    // an equal box means an equal row, and the answer of the row before it still stands. Such a
    // row then skips the geometry build as well as the exact test.
    //
    // Two conditions gate this. Only a point column qualifies, because two different geometries
    // can share a box. And the literal must hold a box of its own: against a literal with a box,
    // every verdict above settles a row whose own box is empty, so no null row and no empty point
    // reaches the test below. That keeps the empty case out of the loop.
    let repeats = matches!(array.data_type(), GeoArrowType::Point(_)) && !literal_bbox.is_empty();
    // NaN equals nothing, so the first row of a batch can never match.
    let mut last_x = f64::NAN;
    let mut last_y = f64::NAN;
    let mut last_answer = false;

    let mut values = BooleanBufferBuilder::new(len);
    for index in 0..len {
        let row_bbox = scratch.left[index];
        let (left_bbox, right_bbox) = match literal_side {
            Side::Left => (literal_bbox, row_bbox),
            Side::Right => (row_bbox, literal_bbox),
        };
        if let Some(answer) = predicate.bbox_verdict(&left_bbox, &right_bbox) {
            values.append(answer);
            continue;
        }

        debug_assert!(
            !repeats || !row_bbox.is_empty(),
            "a row with an empty box must have been settled by its verdict"
        );
        if repeats && row_bbox.minx == last_x && row_bbox.miny == last_y {
            values.append(last_answer);
            continue;
        }

        let answer = if filler(index, &mut row)? {
            literal.evaluate(predicate, &row, literal_side)
        } else {
            false
        };
        if repeats {
            last_x = row_bbox.minx;
            last_y = row_bbox.miny;
            last_answer = answer;
        }
        values.append(answer);
    }

    Ok(BooleanArray::new(values.finish(), nulls))
}

/// `ST_Intersects`. Kept as a named entry point for the benchmarks.
pub fn st_intersects(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
) -> GeoArrowResult<BooleanArray> {
    st_predicate(left, right, Predicate::Intersects)
}

/// `ST_Intersects` over two arrays with caller-owned scratch.
pub fn st_intersects_with(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    scratch: &mut PredicateScratch,
) -> GeoArrowResult<BooleanArray> {
    st_predicate_with(left, right, Predicate::Intersects, scratch)
}

/// `ST_Intersects` against a constant.
pub fn st_intersects_scalar(
    array: &dyn GeoArrowArray,
    literal: &PreparedLiteral,
    scratch: &mut PredicateScratch,
) -> GeoArrowResult<BooleanArray> {
    st_predicate_scalar(array, literal, Predicate::Intersects, Side::Right, scratch)
}

/// `ST_DWithin`. True when the two geometries are no further apart than `radius`.
pub fn st_dwithin(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    radius: f64,
    scratch: &mut PredicateScratch,
) -> GeoArrowResult<BooleanArray> {
    distance_predicate(left, right, radius, scratch, false)
}

/// `ST_DFullyWithin`. True when every point of one is within `radius` of the other.
pub fn st_dfully_within(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    radius: f64,
    scratch: &mut PredicateScratch,
) -> GeoArrowResult<BooleanArray> {
    distance_predicate(left, right, radius, scratch, true)
}

fn distance_predicate(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    radius: f64,
    scratch: &mut PredicateScratch,
    fully: bool,
) -> GeoArrowResult<BooleanArray> {
    let len = broadcast_len("a distance predicate", left, right)?;
    let nulls = broadcast_nulls(left, right, len);

    fill_bboxes(left, &mut scratch.left)?;
    fill_bboxes(right, &mut scratch.right)?;
    let left_box = BoxSide::new(&scratch.left, len);
    let right_box = BoxSide::new(&scratch.right, len);

    let mut left_geom = Operand::new(left, len)?;
    let mut right_geom = Operand::new(right, len)?;

    let mut values = BooleanBufferBuilder::new(len);
    for index in 0..len {
        // Grow one box by the radius. A row outside it is further away than the radius. So no
        // row there needs a distance. This is the prefilter for ST_DWithin.
        if !fully
            && !left_box
                .at(index)
                .expand(radius)
                .intersects(&right_box.at(index))
        {
            values.append(false);
            continue;
        }
        match (left_geom.get(index)?, right_geom.get(index)?) {
            (Some(lhs), Some(rhs)) => {
                let answer = if fully {
                    crate::measure::max_distance(lhs, rhs) <= radius
                } else {
                    Euclidean.distance(lhs, rhs) <= radius
                };
                values.append(answer);
            }
            _ => values.append(false),
        }
    }

    Ok(BooleanArray::new(values.finish(), nulls))
}

/// `ST_Relate(a, b)`. The nine character DE-9IM matrix as text.
pub fn st_relate(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
) -> GeoArrowResult<StringArray> {
    let len = broadcast_len("ST_Relate", left, right)?;
    let mut left_geom = Operand::new(left, len)?;
    let mut right_geom = Operand::new(right, len)?;

    let mut builder = StringBuilder::with_capacity(len, len * 9);
    let mut text = String::with_capacity(9);
    for index in 0..len {
        match (left_geom.get(index)?, right_geom.get(index)?) {
            (Some(lhs), Some(rhs)) => {
                write_matrix(&lhs.relate(rhs), &mut text);
                builder.append_value(&text);
            }
            _ => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// Render a DE-9IM matrix as its nine characters.
///
/// `geo` has no `Display` for the matrix, only a `Debug` that wraps the text in a type name. The
/// order is interior, boundary, exterior on both axes, which is what the OGC specification uses.
fn write_matrix(matrix: &IntersectionMatrix, out: &mut String) {
    use geo::coordinate_position::CoordPos;

    const ORDER: [CoordPos; 3] = [CoordPos::Inside, CoordPos::OnBoundary, CoordPos::Outside];

    out.clear();
    for lhs in ORDER {
        for rhs in ORDER {
            out.push(match matrix.get(lhs, rhs) {
                geo::dimensions::Dimensions::Empty => 'F',
                geo::dimensions::Dimensions::ZeroDimensional => '0',
                geo::dimensions::Dimensions::OneDimensional => '1',
                geo::dimensions::Dimensions::TwoDimensional => '2',
            });
        }
    }
}

/// `ST_Relate(a, b, pattern)`. True when the matrix matches the nine character pattern.
pub fn st_relate_pattern(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    pattern: &str,
) -> GeoArrowResult<BooleanArray> {
    let len = broadcast_len("ST_Relate", left, right)?;
    if pattern.len() != 9 {
        return Err(GeoArrowError::InvalidGeoArrow(format!(
            "a DE-9IM pattern is exactly nine characters, got {}",
            pattern.len()
        )));
    }

    let nulls = broadcast_nulls(left, right, len);
    let mut left_geom = Operand::new(left, len)?;
    let mut right_geom = Operand::new(right, len)?;

    let mut values = BooleanBufferBuilder::new(len);
    for index in 0..len {
        match (left_geom.get(index)?, right_geom.get(index)?) {
            (Some(lhs), Some(rhs)) => {
                let matched = lhs
                    .relate(rhs)
                    .matches(pattern)
                    .map_err(|err| GeoArrowError::External(Box::new(err)))?;
                values.append(matched);
            }
            _ => values.append(false),
        }
    }
    Ok(BooleanArray::new(values.finish(), nulls))
}

/// Bounding boxes for one side. One value may stand for every row.
struct BoxSide<'a> {
    boxes: &'a [Bbox],
    broadcast: bool,
}

impl<'a> BoxSide<'a> {
    fn new(boxes: &'a [Bbox], rows: usize) -> Self {
        Self {
            boxes,
            broadcast: boxes.len() == 1 && rows != 1,
        }
    }

    #[inline]
    fn at(&self, index: usize) -> Bbox {
        if self.broadcast {
            self.boxes[0]
        } else {
            self.boxes[index]
        }
    }
}

/// Row accessor that yields an owned [`geo`] geometry.
///
/// The downcast happens once here. The closure runs only for a row that the box test cannot
/// settle. So the virtual call sits on the slow path, where an allocation dominates it anyway.
type GeometryAccessor<'a> = Box<dyn Fn(usize) -> GeoArrowResult<Option<Geometry<f64>>> + 'a>;

pub(crate) fn geometry_accessor(array: &dyn GeoArrowArray) -> GeoArrowResult<GeometryAccessor<'_>> {
    downcast_geoarrow_array!(array, make_geometry_accessor)
}

fn make_geometry_accessor<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
) -> GeoArrowResult<GeometryAccessor<'a>> {
    Ok(Box::new(move |index| {
        Ok(array.get(index)?.map(|geom| geom.to_geometry()))
    }))
}

/// Convert one GeoArrow row into an owned [`geo`] geometry.
///
/// Callers use this to lift a scalar argument out of an array of length one.
///
/// This builds a [`GeometryReader`] and throws it away, so it pays one downcast for one row. In a
/// loop, build the reader once and call [`GeometryReader::read`] instead.
pub fn geometry_at(
    array: &dyn GeoArrowArray,
    index: usize,
) -> GeoArrowResult<Option<Geometry<f64>>> {
    GeometryReader::new(array)?.get(index)
}

/// One side of a binary function, which may be a column or a single broadcast value.
///
/// A constant argument arrives as an array of length one. The crate builds it once and lends it
/// to every row. That beats a full column: a 256 vertex polygon costs one build, not 8192.
///
/// A column side reuses one `Geometry` across every row. The row is refilled in place from the
/// Arrow buffers, so the loop allocates nothing after the first row. `benches/caching.rs` prices
/// that at 3.2 times for small polygons and 6.8 times for large ones, and shows that every cache
/// tried against it loses.
pub(crate) struct Operand<'a> {
    filler: GeometryFiller<'a>,
    /// `Some` when this side is one value shared by every row.
    constant: Option<Option<Geometry<f64>>>,
    /// Reused by every row of a column side.
    scratch: Geometry<f64>,
}

impl<'a> Operand<'a> {
    pub(crate) fn new(array: &'a dyn GeoArrowArray, rows: usize) -> GeoArrowResult<Self> {
        let filler = geometry_filler(array)?;
        let constant = if array.len() == 1 && rows != 1 {
            let mut value = empty_geometry();
            Some(filler(0, &mut value)?.then_some(value))
        } else {
            None
        };
        Ok(Self {
            filler,
            constant,
            scratch: empty_geometry(),
        })
    }

    /// The geometry of one row. The reference is valid until the next call.
    #[inline]
    pub(crate) fn get(&mut self, index: usize) -> GeoArrowResult<Option<&Geometry<f64>>> {
        let Self {
            filler,
            constant,
            scratch,
        } = self;
        match constant {
            Some(value) => Ok(value.as_ref()),
            None => {
                if filler(index, scratch)? {
                    Ok(Some(scratch))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

/// The row count of a binary call. One side may hold a single value for every row.
pub(crate) fn broadcast_len(
    function: &str,
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
) -> GeoArrowResult<usize> {
    match (left.len(), right.len()) {
        (a, b) if a == b => Ok(a),
        (1, b) => Ok(b),
        (a, 1) => Ok(a),
        (a, b) => Err(GeoArrowError::InvalidGeoArrow(format!(
            "{function} needs two arrays of the same length, or one of length one, got {a} and {b}"
        ))),
    }
}

/// The combined null buffer of a binary call. One side may hold a single value for every row.
pub(crate) fn broadcast_nulls(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
    rows: usize,
) -> Option<NullBuffer> {
    let side = |array: &dyn GeoArrowArray| -> Option<NullBuffer> {
        match array.logical_nulls() {
            // A broadcast side that is null makes every row null.
            Some(nulls) if array.len() == 1 && rows != 1 => {
                nulls.is_null(0).then(|| NullBuffer::new_null(rows))
            }
            Some(nulls) if array.len() == rows => Some(nulls),
            _ => None,
        }
    };
    NullBuffer::union(side(left).as_ref(), side(right).as_ref())
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use geoarrow_array::builder::{GeometryBuilder, PointBuilder, PolygonBuilder};
    use geoarrow_schema::{CoordType, Dimension, GeometryType, PointType, PolygonType};

    use super::*;

    fn points(coord_type: CoordType) -> impl GeoArrowArray {
        let p0 = geo::point!(x: 0.5, y: 0.5);
        let p1 = geo::point!(x: 9.0, y: 9.0);
        let p2 = geo::point!(x: 0.1, y: 0.1);
        PointBuilder::from_nullable_points(
            [Some(&p0), Some(&p1), None, Some(&p2)].into_iter(),
            PointType::new(Dimension::XY, Default::default()).with_coord_type(coord_type),
        )
        .finish()
    }

    fn unit_square_polygon() -> geo::Polygon<f64> {
        geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 1.0,0.0 0.0)) }
    }

    fn unit_square() -> Geometry<f64> {
        unit_square_polygon().into()
    }

    fn regular_ring(sides: usize) -> Geometry<f64> {
        let mut coords: Vec<geo::Coord<f64>> = (0..sides)
            .map(|i| {
                let angle = (i as f64) / (sides as f64) * std::f64::consts::TAU;
                geo::coord! { x: angle.cos(), y: angle.sin() }
            })
            .collect();
        coords.push(coords[0]);
        Geometry::Polygon(geo::Polygon::new(geo::LineString::new(coords), vec![]))
    }

    #[test]
    fn scalar_intersects_matches_geo() {
        for coord_type in [CoordType::Separated, CoordType::Interleaved] {
            let array = points(coord_type);
            let literal = PreparedLiteral::new(unit_square());
            let mut scratch = PredicateScratch::new();
            let result = st_intersects_scalar(&array, &literal, &mut scratch).unwrap();

            assert!(result.value(0), "point inside the square");
            assert!(!result.value(1), "point far outside");
            assert!(result.is_null(2), "null input yields null");
            assert!(result.value(3), "second point inside");
        }
    }

    /// The box test alone must settle `ST_Disjoint` for a far away row.
    #[test]
    fn disjoint_is_the_complement_of_intersects() {
        let array = points(CoordType::Separated);
        let literal = PreparedLiteral::new(unit_square());
        let mut scratch = PredicateScratch::new();

        let hits = st_predicate_scalar(
            &array,
            &literal,
            Predicate::Intersects,
            Side::Right,
            &mut scratch,
        )
        .unwrap();
        let misses = st_predicate_scalar(
            &array,
            &literal,
            Predicate::Disjoint,
            Side::Right,
            &mut scratch,
        )
        .unwrap();

        for row in [0usize, 1, 3] {
            assert_eq!(
                hits.value(row),
                !misses.value(row),
                "row {row} disagreed between intersects and disjoint"
            );
        }
        assert!(misses.is_null(2));
    }

    /// Contains and Within are converses. A swap of the arguments must swap the answer.
    #[test]
    fn contains_and_within_are_converses() {
        let squares = vec![unit_square_polygon(); 4];
        let polygons = PolygonBuilder::from_polygons(
            &squares,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let array = points(CoordType::Separated);

        let contains = st_predicate(&polygons, &array, Predicate::Contains).unwrap();
        let within = st_predicate(&array, &polygons, Predicate::Within).unwrap();

        for row in [0usize, 1, 3] {
            assert_eq!(contains.value(row), within.value(row), "row {row}");
        }
        assert!(contains.value(0), "the square contains the inner point");
        assert!(!contains.value(1), "and not the far one");
    }

    #[test]
    fn covers_accepts_the_boundary_where_contains_does_not() {
        // A point exactly on the corner of the square.
        let corner = PointBuilder::from_points(
            [geo::point!(x: 0.0, y: 0.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let square = PolygonBuilder::from_polygons(
            &[unit_square_polygon()],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let contains = st_predicate(&square, &corner, Predicate::Contains).unwrap();
        let covers = st_predicate(&square, &corner, Predicate::Covers).unwrap();
        assert!(!contains.value(0), "a boundary point is not contained");
        assert!(covers.value(0), "but it is covered");
    }

    #[test]
    fn equals_needs_matching_boxes() {
        let squares = vec![unit_square_polygon(); 2];
        let a = PolygonBuilder::from_polygons(
            &squares,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let shifted: Vec<geo::Polygon<f64>> = vec![
            unit_square_polygon(),
            geo::wkt! { POLYGON((5.0 5.0,6.0 5.0,6.0 6.0,5.0 6.0,5.0 5.0)) },
        ];
        let b = PolygonBuilder::from_polygons(
            &shifted,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let equals = st_predicate(&a, &b, Predicate::Equals).unwrap();
        assert!(equals.value(0));
        assert!(!equals.value(1), "the box test settles this one");
    }

    #[test]
    fn touches_crosses_and_overlaps() {
        let mut left = GeometryBuilder::new(GeometryType::new(Default::default()));
        let mut right = GeometryBuilder::new(GeometryType::new(Default::default()));

        // The two shapes meet at one point.
        left.push_geometry(Some(&Geometry::<f64>::from(
            geo::wkt! { LINESTRING(0.0 0.0,1.0 0.0) },
        )))
        .unwrap();
        right
            .push_geometry(Some(&Geometry::<f64>::from(
                geo::wkt! { LINESTRING(1.0 0.0,2.0 0.0) },
            )))
            .unwrap();

        // The two lines cross in the middle.
        left.push_geometry(Some(&Geometry::<f64>::from(
            geo::wkt! { LINESTRING(0.0 0.0,2.0 2.0) },
        )))
        .unwrap();
        right
            .push_geometry(Some(&Geometry::<f64>::from(
                geo::wkt! { LINESTRING(0.0 2.0,2.0 0.0) },
            )))
            .unwrap();

        // Two squares that overlap in part.
        left.push_geometry(Some(&unit_square())).unwrap();
        right
            .push_geometry(Some(&Geometry::<f64>::from(
                geo::wkt! { POLYGON((0.5 0.5,1.5 0.5,1.5 1.5,0.5 1.5,0.5 0.5)) },
            )))
            .unwrap();

        let (left, right) = (left.finish(), right.finish());

        assert!(st_predicate(&left, &right, Predicate::Touches)
            .unwrap()
            .value(0));
        assert!(st_predicate(&left, &right, Predicate::Crosses)
            .unwrap()
            .value(1));
        assert!(st_predicate(&left, &right, Predicate::Overlaps)
            .unwrap()
            .value(2));
    }

    /// The R-tree must be built for a DE-9IM predicate and skipped for a direct one.
    #[test]
    fn only_relate_predicates_build_the_index() {
        let array = points(CoordType::Separated);
        let mut scratch = PredicateScratch::new();

        let direct = PreparedLiteral::new(regular_ring(64));
        st_predicate_scalar(
            &array,
            &direct,
            Predicate::Intersects,
            Side::Right,
            &mut scratch,
        )
        .unwrap();
        assert!(
            !direct.is_indexed(),
            "a direct predicate must not build the R-tree"
        );

        let indexed = PreparedLiteral::new(regular_ring(64));
        st_predicate_scalar(
            &array,
            &indexed,
            Predicate::Touches,
            Side::Right,
            &mut scratch,
        )
        .unwrap();
        assert!(
            indexed.is_indexed(),
            "a DE-9IM predicate must build the R-tree once"
        );
    }

    /// The constant path and the two-array path must agree, both ways round.
    #[test]
    fn scalar_and_array_paths_agree() {
        let array = points(CoordType::Separated);
        let squares = vec![unit_square_polygon(); 4];
        let polygons = PolygonBuilder::from_polygons(
            &squares,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let literal = PreparedLiteral::new(unit_square());
        let mut scratch = PredicateScratch::new();

        for predicate in Predicate::ALL {
            let two_arrays = st_predicate(&array, &polygons, predicate).unwrap();
            let constant_right =
                st_predicate_scalar(&array, &literal, predicate, Side::Right, &mut scratch)
                    .unwrap();
            assert_eq!(
                two_arrays,
                constant_right,
                "{} disagreed with the constant on the right",
                predicate.function_name()
            );

            let flipped = st_predicate(&polygons, &array, predicate).unwrap();
            let constant_left =
                st_predicate_scalar(&array, &literal, predicate, Side::Left, &mut scratch).unwrap();
            assert_eq!(
                flipped,
                constant_left,
                "{} disagreed with the constant on the left",
                predicate.function_name()
            );
        }
    }

    /// A ring, a ring with a hole, and two rings. Each is over the index threshold.
    fn indexable_literals() -> Vec<Geometry<f64>> {
        let Geometry::Polygon(shell) = regular_ring(128) else {
            unreachable!()
        };
        let scaled = |factor: f64, dx: f64| {
            geo::LineString::new(
                shell
                    .exterior()
                    .0
                    .iter()
                    .map(|coord| geo::coord! { x: coord.x * factor + dx, y: coord.y * factor })
                    .collect(),
            )
        };
        vec![
            Geometry::Polygon(shell.clone()),
            Geometry::Polygon(geo::Polygon::new(
                shell.exterior().clone(),
                vec![scaled(0.4, 0.0)],
            )),
            Geometry::MultiPolygon(geo::MultiPolygon::new(vec![
                geo::Polygon::new(scaled(0.8, -1.4), vec![]),
                geo::Polygon::new(scaled(0.8, 1.4), vec![]),
            ])),
        ]
    }

    /// Points spread over the shape, plus every vertex and every edge midpoint.
    ///
    /// The last two groups sit exactly on the boundary. That is where `ST_Contains` and
    /// `ST_Covers` part company, so a verdict that reads the boundary wrongly shows up here.
    fn boundary_and_spread_probes(geometry: &Geometry<f64>) -> Vec<geo::Point<f64>> {
        let mut state = 0x5EEDu64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut probes: Vec<geo::Point<f64>> = (0..300)
            .map(|_| geo::point! { x: (next() - 0.5) * 5.2, y: (next() - 0.5) * 2.6 })
            .collect();

        let rings: Vec<&geo::LineString<f64>> = match geometry {
            Geometry::Polygon(polygon) => std::iter::once(polygon.exterior())
                .chain(polygon.interiors())
                .collect(),
            Geometry::MultiPolygon(multi) => multi
                .iter()
                .flat_map(|polygon| std::iter::once(polygon.exterior()).chain(polygon.interiors()))
                .collect(),
            _ => Vec::new(),
        };
        for ring in rings {
            for edge in ring.0.windows(2) {
                probes.push(geo::point! { x: edge[0].x, y: edge[0].y });
                probes.push(geo::point! {
                    x: (edge[0].x + edge[1].x) / 2.0,
                    y: (edge[0].y + edge[1].y) / 2.0,
                });
            }
        }
        probes
    }

    /// The index must never change an answer, for any predicate, on either side.
    ///
    /// The reference is [`Predicate::evaluate`], which is the call the unindexed path makes.
    #[test]
    fn the_point_index_never_changes_an_answer() {
        for literal in indexable_literals() {
            let probes = boundary_and_spread_probes(&literal);
            let prepared = PreparedLiteral::new(literal.clone());

            for predicate in Predicate::ALL {
                for side in [Side::Left, Side::Right] {
                    for probe in &probes {
                        let row = Geometry::Point(*probe);
                        let want = match side {
                            Side::Left => predicate.evaluate(&literal, &row),
                            Side::Right => predicate.evaluate(&row, &literal),
                        };
                        assert_eq!(
                            prepared.evaluate(predicate, &row, side),
                            want,
                            "{} with the constant on the {side:?} disagreed at {probe:?}",
                            predicate.function_name()
                        );
                    }
                }
            }
            assert!(
                prepared.has_point_index(),
                "the fixture must reach the indexed path"
            );
        }
    }

    /// The table that decides which predicate reads the verdict. A `None` here is a lost
    /// speedup, so the pairs that must be indexed are pinned.
    #[test]
    fn the_point_rule_table_covers_every_direct_predicate() {
        use Predicate::*;

        // The constant is the areal side, so it holds the point.
        for predicate in [Contains, ContainsProperly] {
            assert_eq!(predicate.point_rule(Side::Left), Some(PointRule::Inside));
            assert_eq!(predicate.point_rule(Side::Right), None);
        }
        assert_eq!(Covers.point_rule(Side::Left), Some(PointRule::NotOutside));
        assert_eq!(Covers.point_rule(Side::Right), None);

        // The converse pair reads the other way round.
        assert_eq!(Within.point_rule(Side::Right), Some(PointRule::Inside));
        assert_eq!(Within.point_rule(Side::Left), None);
        assert_eq!(
            CoveredBy.point_rule(Side::Right),
            Some(PointRule::NotOutside)
        );
        assert_eq!(CoveredBy.point_rule(Side::Left), None);

        // Symmetric, so either side works.
        for side in [Side::Left, Side::Right] {
            assert_eq!(Intersects.point_rule(side), Some(PointRule::NotOutside));
            assert_eq!(Disjoint.point_rule(side), Some(PointRule::Outside));
        }

        // A DE-9IM predicate keeps the R-tree path.
        for predicate in Predicate::ALL.iter().filter(|p| p.needs_relate()) {
            for side in [Side::Left, Side::Right] {
                assert_eq!(
                    predicate.point_rule(side),
                    None,
                    "{} needs the matrix",
                    predicate.function_name()
                );
            }
        }
    }

    /// The constant path and the two array path must agree on a polygon over the threshold.
    ///
    /// The constant side reads the edge index. The two array side never builds one, so this
    /// compares the indexed kernel with the unindexed kernel, end to end.
    #[test]
    fn scalar_and_array_paths_agree_on_a_large_polygon() {
        let Geometry::Polygon(ring) = regular_ring(128) else {
            unreachable!()
        };
        let probes = boundary_and_spread_probes(&Geometry::Polygon(ring.clone()));
        let array = PointBuilder::from_points(
            probes.iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let polygons = PolygonBuilder::from_polygons(
            &vec![ring.clone(); probes.len()],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let literal = PreparedLiteral::new(Geometry::Polygon(ring));
        let mut scratch = PredicateScratch::new();

        for predicate in Predicate::ALL {
            let constant_right =
                st_predicate_scalar(&array, &literal, predicate, Side::Right, &mut scratch)
                    .unwrap();
            assert_eq!(
                st_predicate(&array, &polygons, predicate).unwrap(),
                constant_right,
                "{} disagreed with the constant on the right",
                predicate.function_name()
            );

            let constant_left =
                st_predicate_scalar(&array, &literal, predicate, Side::Left, &mut scratch).unwrap();
            assert_eq!(
                st_predicate(&polygons, &array, predicate).unwrap(),
                constant_left,
                "{} disagreed with the constant on the left",
                predicate.function_name()
            );
        }
        assert!(literal.has_point_index(), "the index must have been used");
    }

    /// A hole takes a point back out, and the boundary splits `ST_Contains` from `ST_Covers`.
    #[test]
    fn the_index_reads_a_hole_and_a_boundary() {
        let Geometry::Polygon(shell) = regular_ring(64) else {
            unreachable!()
        };
        let hole = geo::LineString::new(
            shell
                .exterior()
                .0
                .iter()
                .map(|coord| geo::coord! { x: coord.x * 0.4, y: coord.y * 0.4 })
                .collect(),
        );
        let on_the_shell = shell.exterior().0[0];
        let donut = geo::Polygon::new(shell.exterior().clone(), vec![hole]);

        let array = PointBuilder::from_points(
            [
                geo::point!(x: 0.0, y: 0.0),                     // in the hole
                geo::point!(x: 0.7, y: 0.0),                     // in the ring
                geo::point!(x: 0.95, y: 0.9),                    // outside
                geo::Point::new(on_the_shell.x, on_the_shell.y), // on the shell
            ]
            .iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let literal = PreparedLiteral::new(Geometry::Polygon(donut));
        let mut scratch = PredicateScratch::new();
        let run = |predicate, side, scratch: &mut PredicateScratch| {
            st_predicate_scalar(&array, &literal, predicate, side, scratch).unwrap()
        };

        // The interior only.
        for (predicate, side) in [
            (Predicate::Within, Side::Right),
            (Predicate::Contains, Side::Left),
        ] {
            let result = run(predicate, side, &mut scratch);
            let name = predicate.function_name();
            assert!(!result.value(0), "{name}: the hole is not inside");
            assert!(result.value(1), "{name}: the ring is");
            assert!(!result.value(2), "{name}: the corner is not");
            assert!(!result.value(3), "{name}: the shell is not inside");
        }

        // The interior and the boundary.
        for (predicate, side) in [
            (Predicate::Covers, Side::Left),
            (Predicate::CoveredBy, Side::Right),
            (Predicate::Intersects, Side::Right),
        ] {
            let result = run(predicate, side, &mut scratch);
            let name = predicate.function_name();
            assert!(!result.value(0), "{name}: the hole is still out");
            assert!(result.value(1), "{name}: the ring is in");
            assert!(!result.value(2), "{name}: the corner is out");
            assert!(result.value(3), "{name}: the shell counts here");
        }

        // The complement of intersects.
        let disjoint = run(Predicate::Disjoint, Side::Right, &mut scratch);
        assert!(disjoint.value(0));
        assert!(!disjoint.value(1));
        assert!(disjoint.value(2));
        assert!(!disjoint.value(3));

        assert!(literal.has_point_index());
    }

    /// A point column with runs of repeats must answer exactly as one without them.
    ///
    /// The scalar path reuses the answer of the row before when the box repeats. The two array
    /// path has no such reuse, so it is the reference. Nulls sit between the runs, because a null
    /// row carries an empty box and must never key a repeat.
    #[test]
    fn a_repeated_point_row_reuses_the_right_answer() {
        let Geometry::Polygon(ring) = regular_ring(128) else {
            unreachable!()
        };

        // Runs of a point inside, a point outside, a point exactly on a vertex, and nulls.
        let vertex = ring.exterior().0[0];
        let sources = [
            Some(geo::point! { x: 0.2, y: 0.1 }),
            Some(geo::point! { x: 0.2, y: 0.1 }),
            Some(geo::point! { x: 0.2, y: 0.1 }),
            None,
            None,
            Some(geo::Point::new(vertex.x, vertex.y)),
            Some(geo::Point::new(vertex.x, vertex.y)),
            Some(geo::point! { x: 40.0, y: 40.0 }),
            Some(geo::point! { x: 40.0, y: 40.0 }),
            Some(geo::point! { x: -0.3, y: 0.6 }),
            None,
            Some(geo::point! { x: -0.3, y: 0.6 }),
            // A negative zero compares equal to a zero. Both name the same coordinate.
            Some(geo::point! { x: 0.0, y: 0.5 }),
            Some(geo::point! { x: -0.0, y: 0.5 }),
        ];
        let array = PointBuilder::from_nullable_points(
            sources.iter().map(|point| point.as_ref()),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let polygons = PolygonBuilder::from_polygons(
            &vec![ring.clone(); sources.len()],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let literal = PreparedLiteral::new(Geometry::Polygon(ring));
        let mut scratch = PredicateScratch::new();
        for predicate in Predicate::ALL {
            for side in [Side::Left, Side::Right] {
                let reused =
                    st_predicate_scalar(&array, &literal, predicate, side, &mut scratch).unwrap();
                let reference = match side {
                    Side::Right => st_predicate(&array, &polygons, predicate).unwrap(),
                    Side::Left => st_predicate(&polygons, &array, predicate).unwrap(),
                };
                assert_eq!(
                    reused,
                    reference,
                    "{} with the constant on the {side:?} reused a wrong answer",
                    predicate.function_name()
                );
            }
        }
    }

    /// Only a point column may reuse a row. Two polygons can share a box and still differ.
    ///
    /// The square holds the probe. The triangle has the same box and does not. If the reuse ever
    /// escaped the point column, the second row would copy the answer of the first.
    #[test]
    fn a_polygon_row_never_reuses_the_row_before_it() {
        let square = geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 1.0,0.0 0.0)) };
        let triangle = geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,0.0 1.0,0.0 0.0)) };
        assert_eq!(
            bbox_of(&Geometry::Polygon(square.clone())),
            bbox_of(&Geometry::Polygon(triangle.clone())),
            "the fixture needs two shapes with one box"
        );

        let rows = PolygonBuilder::from_polygons(
            &[square, triangle],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();

        // The probe sits inside the square and outside the triangle.
        let literal = PreparedLiteral::new(Geometry::Point(geo::point! { x: 0.75, y: 0.75 }));
        let mut scratch = PredicateScratch::new();
        let contains = st_predicate_scalar(
            &rows,
            &literal,
            Predicate::Contains,
            Side::Right,
            &mut scratch,
        )
        .unwrap();

        assert!(contains.value(0), "the square holds the probe");
        assert!(!contains.value(1), "the triangle does not");
    }

    /// The edge index answers a point against a polygon. Every other call must leave it unbuilt.
    #[test]
    fn only_a_point_against_a_polygon_builds_the_edge_index() {
        let array = points(CoordType::Separated);
        let mut scratch = PredicateScratch::new();

        let small = PreparedLiteral::new(unit_square());
        st_predicate_scalar(&array, &small, Predicate::Within, Side::Right, &mut scratch).unwrap();
        assert!(
            !small.has_point_index(),
            "a 5 vertex constant is below the threshold"
        );

        let line = PreparedLiteral::new(Geometry::LineString(
            geo::wkt! { LINESTRING(-1.0 -1.0,9.0 9.0) },
        ));
        st_predicate_scalar(&array, &line, Predicate::Within, Side::Right, &mut scratch).unwrap();
        assert!(!line.has_point_index(), "a line has no inside to index");

        let touches = PreparedLiteral::new(regular_ring(64));
        st_predicate_scalar(
            &array,
            &touches,
            Predicate::Touches,
            Side::Right,
            &mut scratch,
        )
        .unwrap();
        assert!(
            !touches.has_point_index(),
            "a DE-9IM predicate keeps the R-tree path"
        );
        assert!(touches.is_indexed(), "and builds that R-tree instead");

        // A polygon column, not a point column. The index answers a point probe only.
        let squares = PolygonBuilder::from_polygons(
            &vec![unit_square_polygon(); 2],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let rows = PreparedLiteral::new(regular_ring(64));
        st_predicate_scalar(
            &squares,
            &rows,
            Predicate::Within,
            Side::Right,
            &mut scratch,
        )
        .unwrap();
        assert!(
            !rows.has_point_index(),
            "a polygon row is not a point probe"
        );

        // Each direct predicate, with the constant on the side that holds the point.
        for (predicate, side) in [
            (Predicate::Within, Side::Right),
            (Predicate::CoveredBy, Side::Right),
            (Predicate::Contains, Side::Left),
            (Predicate::ContainsProperly, Side::Left),
            (Predicate::Covers, Side::Left),
            (Predicate::Intersects, Side::Right),
            (Predicate::Disjoint, Side::Right),
        ] {
            let literal = PreparedLiteral::new(regular_ring(64));
            st_predicate_scalar(&array, &literal, predicate, side, &mut scratch).unwrap();
            assert!(
                literal.has_point_index(),
                "{} must reach the edge index",
                predicate.function_name()
            );
            assert!(
                !literal.is_indexed(),
                "{} must not also build the R-tree",
                predicate.function_name()
            );
        }
    }

    #[test]
    fn dwithin_uses_the_expanded_box() {
        let a = PointBuilder::from_points(
            [geo::point!(x: 0.0, y: 0.0), geo::point!(x: 0.0, y: 0.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let b = PointBuilder::from_points(
            [geo::point!(x: 3.0, y: 4.0), geo::point!(x: 30.0, y: 40.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let mut scratch = PredicateScratch::new();
        let within_five = st_dwithin(&a, &b, 5.0, &mut scratch).unwrap();
        assert!(within_five.value(0), "the 3-4-5 triangle is exactly 5 away");
        assert!(!within_five.value(1));

        let within_four = st_dwithin(&a, &b, 4.0, &mut scratch).unwrap();
        assert!(!within_four.value(0));
    }

    #[test]
    fn relate_returns_the_matrix() {
        let square = PolygonBuilder::from_polygons(
            &[unit_square_polygon()],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        let inner = PointBuilder::from_points(
            [geo::point!(x: 0.5, y: 0.5)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let matrix = st_relate(&square, &inner).unwrap();
        assert_eq!(matrix.value(0).len(), 9);

        // The same relationship, asked as a pattern.
        let contains = st_relate_pattern(&square, &inner, "T*****FF*").unwrap();
        assert!(contains.value(0));
    }

    #[test]
    fn a_bad_pattern_is_an_error() {
        let square = PolygonBuilder::from_polygons(
            &[unit_square_polygon()],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        assert!(st_relate_pattern(&square, &square, "too short").is_err());
    }

    /// A side of length one is broadcast over the batch. Any other mismatch is an error.
    #[test]
    fn a_single_row_side_broadcasts() {
        let left = points(CoordType::Separated);
        let single = PolygonBuilder::from_polygons(
            &[unit_square_polygon()],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let broadcast = st_predicate(&left, &single, Predicate::Intersects).unwrap();
        assert_eq!(broadcast.len(), left.len());
        assert!(broadcast.value(0), "the inner point hits the square");
        assert!(!broadcast.value(1), "the far point does not");
        assert!(broadcast.is_null(2));

        // The same answers as a hand-built constant column.
        let expanded = PolygonBuilder::from_polygons(
            &vec![unit_square_polygon(); 4],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        assert_eq!(
            broadcast,
            st_predicate(&left, &expanded, Predicate::Intersects).unwrap()
        );
    }

    #[test]
    fn a_genuine_length_mismatch_is_an_error() {
        let left = points(CoordType::Separated);
        let three = PointBuilder::from_points(
            [
                geo::point!(x: 0.0, y: 0.0),
                geo::point!(x: 1.0, y: 1.0),
                geo::point!(x: 2.0, y: 2.0),
            ]
            .iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();
        assert!(st_predicate(&left, &three, Predicate::Intersects).is_err());
    }
}
