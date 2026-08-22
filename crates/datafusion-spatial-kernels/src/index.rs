//! Spatial indexes for the kernels.
//!
//! Two structures live here. [`BboxGrid`] buckets whole rows by bounding box for a join.
//! [`PointInPolygonIndex`] buckets the edges of one polygon by y interval for a point probe.
//!
//! # A bounding box index for spatial joins
//!
//! A join on `ST_Intersects(a.geom, b.geom)` has no equality key, so DataFusion runs it as a
//! nested loop: every row of one side against every row of the other. The box prefilter makes one
//! pair cheap, but the pair count is the product of the two row counts.
//!
//! [`BboxGrid`] changes that. It buckets the build side by bounding box. It then answers a probe
//! box with the rows of the cells that overlap.
//!
//! ## Why a uniform grid and not an R-tree
//!
//! A grid builds in one pass. It needs no tree, no recursion and no rebalance step. Its weakness is
//! skewed
//! data, where one cell can hold most of the rows. The exact test still runs on those, so the
//! answer stays correct and the cost degrades toward the nested loop it replaced. It never does
//! worse than that.

use geo::coordinate_position::CoordPos;
use geo::kernels::Kernel;
use geo::{Coord, GeoNum, Geometry, LineString, Orientation, Polygon};

use crate::bbox::Bbox;

/// A uniform grid over the bounding boxes of one side of a join.
#[derive(Debug)]
pub struct BboxGrid {
    /// Extent of the whole build side.
    minx: f64,
    miny: f64,
    /// Width and height of one cell. Never zero.
    cell_w: f64,
    cell_h: f64,
    cols: usize,
    rows: usize,
    /// Row ids, grouped by cell. `starts[c]..starts[c + 1]` is the run for cell `c`.
    starts: Vec<u32>,
    ids: Vec<u32>,
    /// Rows with an empty box. They match nothing, so they are indexed nowhere.
    indexed: usize,
}

/// Candidate row ids for one probe, without duplicates.
///
/// A build row that spans several cells appears in each of them. The stamp array removes the
/// repeat. It clears no set for each probe.
#[derive(Debug)]
pub struct Candidates {
    stamp: Vec<u32>,
    epoch: u32,
    out: Vec<u32>,
}

impl Candidates {
    /// Room for `rows` build-side rows.
    pub fn new(rows: usize) -> Self {
        Self {
            stamp: vec![0; rows],
            epoch: 0,
            out: Vec::new(),
        }
    }

    /// Start a new probe. Returns the buffer the ids land in.
    #[inline]
    fn begin(&mut self) {
        // `wrapping_add` would let a stale stamp match after 4 billion probes. Clear instead.
        match self.epoch.checked_add(1) {
            Some(next) => self.epoch = next,
            None => {
                self.stamp.iter_mut().for_each(|slot| *slot = 0);
                self.epoch = 1;
            }
        }
        self.out.clear();
    }

    #[inline]
    fn push(&mut self, id: u32) {
        let slot = &mut self.stamp[id as usize];
        if *slot != self.epoch {
            *slot = self.epoch;
            self.out.push(id);
        }
    }

    /// The ids found by the last query.
    #[inline]
    pub fn ids(&self) -> &[u32] {
        &self.out
    }
}

impl BboxGrid {
    /// Build an index over `boxes`.
    ///
    /// An empty box takes no cell. Such a row intersects nothing, so no probe should reach it.
    pub fn build(boxes: &[Bbox]) -> Self {
        let mut minx = f64::INFINITY;
        let mut miny = f64::INFINITY;
        let mut maxx = f64::NEG_INFINITY;
        let mut maxy = f64::NEG_INFINITY;
        let mut indexed = 0usize;
        for bbox in boxes {
            if bbox.is_empty() {
                continue;
            }
            indexed += 1;
            if bbox.minx < minx {
                minx = bbox.minx;
            }
            if bbox.miny < miny {
                miny = bbox.miny;
            }
            if bbox.maxx > maxx {
                maxx = bbox.maxx;
            }
            if bbox.maxy > maxy {
                maxy = bbox.maxy;
            }
        }

        // One cell per row, laid out square. That keeps the average occupancy near one.
        let side = (indexed as f64).sqrt().ceil().max(1.0);
        let cols = side as usize;
        let rows = side as usize;

        // A degenerate extent, such as one point or a single vertical line, still needs a positive
        // cell size or every division becomes infinite.
        let width = if indexed == 0 { 1.0 } else { maxx - minx };
        let height = if indexed == 0 { 1.0 } else { maxy - miny };
        let cell_w = if width > 0.0 {
            width / cols as f64
        } else {
            1.0
        };
        let cell_h = if height > 0.0 {
            height / rows as f64
        } else {
            1.0
        };

        let mut grid = Self {
            minx: if indexed == 0 { 0.0 } else { minx },
            miny: if indexed == 0 { 0.0 } else { miny },
            cell_w,
            cell_h,
            cols,
            rows,
            starts: Vec::new(),
            ids: Vec::new(),
            indexed,
        };

        // Two passes: count per cell, then fill. This sizes `ids` exactly once.
        let mut counts = vec![0u32; cols * rows + 1];
        for bbox in boxes.iter().filter(|b| !b.is_empty()) {
            let (c0, r0, c1, r1) = grid.cell_span(bbox);
            for row in r0..=r1 {
                for col in c0..=c1 {
                    counts[row * cols + col + 1] += 1;
                }
            }
        }
        for cell in 1..counts.len() {
            counts[cell] += counts[cell - 1];
        }
        let total = counts[counts.len() - 1] as usize;

        let mut cursor = counts.clone();
        let mut ids = vec![0u32; total];
        for (id, bbox) in boxes.iter().enumerate() {
            if bbox.is_empty() {
                continue;
            }
            let (c0, r0, c1, r1) = grid.cell_span(bbox);
            for row in r0..=r1 {
                for col in c0..=c1 {
                    let cell = row * cols + col;
                    ids[cursor[cell] as usize] = id as u32;
                    cursor[cell] += 1;
                }
            }
        }

        grid.starts = counts;
        grid.ids = ids;
        grid
    }

    /// How many rows the index holds. An empty box is not one of them.
    pub fn len(&self) -> usize {
        self.indexed
    }

    /// True when no row was indexed.
    pub fn is_empty(&self) -> bool {
        self.indexed == 0
    }

    /// The inclusive cell range a box covers, clamped to the grid.
    #[inline]
    fn cell_span(&self, bbox: &Bbox) -> (usize, usize, usize, usize) {
        let col = |x: f64| {
            let raw = ((x - self.minx) / self.cell_w).floor();
            raw.clamp(0.0, (self.cols - 1) as f64) as usize
        };
        let row = |y: f64| {
            let raw = ((y - self.miny) / self.cell_h).floor();
            raw.clamp(0.0, (self.rows - 1) as f64) as usize
        };
        (
            col(bbox.minx),
            row(bbox.miny),
            col(bbox.maxx),
            row(bbox.maxy),
        )
    }

    /// Collect every build row whose cell overlaps `probe`.
    ///
    /// The result is a superset of the true matches. The caller still runs the exact test.
    pub fn query(&self, probe: &Bbox, into: &mut Candidates) {
        into.begin();
        if self.indexed == 0 || probe.is_empty() {
            return;
        }
        let (c0, r0, c1, r1) = self.cell_span(probe);
        for row in r0..=r1 {
            let base = row * self.cols;
            for col in c0..=c1 {
                let cell = base + col;
                let from = self.starts[cell] as usize;
                let to = self.starts[cell + 1] as usize;
                for slot in from..to {
                    into.push(self.ids[slot]);
                }
            }
        }
    }
}

/// A point-in-polygon index over the edges of one polygonal geometry.
///
/// # What it answers, and why only this shape
///
/// A direct predicate between a point and a polygon walks every edge of every ring for every row.
/// A 5000 vertex coastline therefore costs 5000 orientation tests per row, and the cost grows
/// with the vertex count.
///
/// The winding number rule reads an edge only when the y interval of that edge holds the y of the
/// point. Every other edge adds nothing. So an index over that interval drops them before the
/// loop starts. PostGIS takes the same short circuit in `liblwgeom/intervaltree.c`, and takes it
/// under the same two conditions: the outer geometry is polygonal and the inner geometry is a
/// point.
///
/// # Why a uniform bucket and not an interval tree
///
/// The same reason [`BboxGrid`] gives. The bucket array builds in one pass, needs no recursion
/// and holds no pointers. A ring with a long near horizontal edge makes one edge land in many
/// buckets, so the build halves the bucket count until the total entry count stays inside
/// `MAX_FANOUT` times the edge count. One bucket is the floor, which is the unindexed walk.
///
/// `geo` ships [`IntervalTreeMultiPolygon`][geo::indexed::IntervalTreeMultiPolygon], which
/// indexes the same intervals. Two things rule it out here. It exposes `Contains` only, so
/// `ST_Intersects` and `ST_Covers` cannot read the boundary verdict from it. And it sums one
/// winding number over the edges of every ring at once, where [`geo::Contains`] tests the shell
/// first and then each hole. Those two rules disagree on a polygon whose hole runs the same way
/// round as its shell, which is common in real data. This index follows the [`geo::Contains`]
/// rule, so the indexed answer and the unindexed answer are the same value.
///
/// # Correctness
///
/// The ring query holds a copy of the edge crossing rules of
/// [`geo::coordinate_position::coord_pos_relative_to_ring`], down to the orientation kernel. The
/// index only removes edges that those rules ignore, so the two agree on every input.
#[derive(Debug)]
pub struct PointInPolygonIndex {
    /// One entry per polygon. A `MultiPolygon` fills several.
    polygons: Vec<PolygonIndex>,
}

/// The largest average number of buckets one edge may occupy.
///
/// A ring of tiny edges lands each edge in one bucket. A ring with long near horizontal edges
/// spreads one edge over many. This cap bounds the memory of the second case.
const MAX_FANOUT: usize = 4;

impl PointInPolygonIndex {
    /// Index a polygonal geometry. Returns `None` for every other type.
    ///
    /// The build makes a few linear passes over the coordinates. The benchmark times it inside
    /// the loop, so the figures on
    /// [`POINT_INDEX_THRESHOLD`][crate::predicate::PreparedLiteral::POINT_INDEX_THRESHOLD]
    /// already carry that cost.
    pub fn new(geometry: &Geometry<f64>) -> Option<Self> {
        let polygons = match geometry {
            Geometry::Polygon(polygon) => vec![PolygonIndex::new(polygon)],
            Geometry::MultiPolygon(multi) => multi.iter().map(PolygonIndex::new).collect(),
            _ => return None,
        };
        Some(Self { polygons })
    }

    /// Where the point sits: inside, on the boundary, or outside.
    ///
    /// This one verdict answers every direct predicate. `ST_Contains` and `ST_Within` want
    /// `Inside`. `ST_Intersects` and `ST_Covers` want anything but `Outside`. `ST_Disjoint` wants
    /// `Outside`. PostGIS reads its own interval tree the same way.
    ///
    /// A `MultiPolygon` combines the same way `geo` combines it: one member that holds the point
    /// inside settles the answer, and the boundary counts only when no member holds it.
    #[inline]
    pub fn locate(&self, coord: Coord<f64>) -> CoordPos {
        let mut boundary = false;
        for polygon in &self.polygons {
            match polygon.locate(coord) {
                CoordPos::Inside => return CoordPos::Inside,
                CoordPos::OnBoundary => boundary = true,
                CoordPos::Outside => {}
            }
        }
        if boundary {
            CoordPos::OnBoundary
        } else {
            CoordPos::Outside
        }
    }

    /// True when the point lies strictly inside the geometry.
    ///
    /// This is the answer of `polygon.contains(point)`. A point on the boundary is not inside,
    /// which is what PostGIS and `geo` both say.
    #[inline]
    pub fn contains(&self, coord: Coord<f64>) -> bool {
        self.locate(coord) == CoordPos::Inside
    }
}

/// One polygon: the shell, then every hole.
#[derive(Debug)]
struct PolygonIndex {
    rings: Vec<RingIndex>,
}

impl PolygonIndex {
    fn new(polygon: &Polygon<f64>) -> Self {
        let mut rings = Vec::with_capacity(1 + polygon.interiors().len());
        rings.push(RingIndex::new(polygon.exterior()));
        rings.extend(polygon.interiors().iter().map(RingIndex::new));
        Self { rings }
    }

    /// Where the point sits, by the rule `geo` uses for a polygon.
    ///
    /// The shell decides first. A hole then takes an inside point back out.
    #[inline]
    fn locate(&self, coord: Coord<f64>) -> CoordPos {
        match self.rings[0].locate(coord) {
            CoordPos::Outside => CoordPos::Outside,
            CoordPos::OnBoundary => CoordPos::OnBoundary,
            CoordPos::Inside => {
                for hole in &self.rings[1..] {
                    match hole.locate(coord) {
                        CoordPos::Outside => {}
                        CoordPos::OnBoundary => return CoordPos::OnBoundary,
                        CoordPos::Inside => return CoordPos::Outside,
                    }
                }
                CoordPos::Inside
            }
        }
    }
}

/// One closed ring, with its edges bucketed by y interval.
#[derive(Debug)]
struct RingIndex {
    /// The ring vertices. Edge `i` runs from `coords[i]` to `coords[i + 1]`.
    coords: Vec<Coord<f64>>,
    bbox: Bbox,
    /// Height of one bucket. Never zero.
    cell_h: f64,
    buckets: usize,
    /// Edge ids, grouped by bucket. `starts[b]..starts[b + 1]` is the run for bucket `b`.
    starts: Vec<u32>,
    ids: Vec<u32>,
}

impl RingIndex {
    fn new(ring: &LineString<f64>) -> Self {
        let coords = ring.0.clone();
        let mut bbox = Bbox::EMPTY;
        for coord in &coords {
            bbox.push_xy(coord.x, coord.y);
        }
        let edges = coords.len().saturating_sub(1);

        // A flat ring, an empty ring or a single edge needs no division. One bucket is the whole
        // ring, which is the unindexed walk.
        let height = bbox.maxy - bbox.miny;
        let (buckets, cell_h) = if edges > 1 && height > 0.0 {
            Self::choose_buckets(&coords, bbox.miny, height, edges)
        } else {
            (1, 1.0)
        };

        // Two passes: count per bucket, then fill. This sizes `ids` exactly once.
        let mut counts = vec![0u32; buckets + 1];
        for edge in coords.windows(2) {
            let (lo, hi) = Self::span(edge, bbox.miny, cell_h, buckets);
            for bucket in lo..=hi {
                counts[bucket + 1] += 1;
            }
        }
        for bucket in 1..counts.len() {
            counts[bucket] += counts[bucket - 1];
        }

        let mut cursor = counts.clone();
        let mut ids = vec![0u32; counts[buckets] as usize];
        for (id, edge) in coords.windows(2).enumerate() {
            let (lo, hi) = Self::span(edge, bbox.miny, cell_h, buckets);
            for bucket in lo..=hi {
                ids[cursor[bucket] as usize] = id as u32;
                cursor[bucket] += 1;
            }
        }

        Self {
            coords,
            bbox,
            cell_h,
            buckets,
            starts: counts,
            ids,
        }
    }

    /// One bucket per edge, halved until the fanout fits.
    fn choose_buckets(coords: &[Coord<f64>], miny: f64, height: f64, edges: usize) -> (usize, f64) {
        let cap = edges.saturating_mul(MAX_FANOUT);
        let mut buckets = edges;
        loop {
            let cell_h = height / buckets as f64;
            if buckets == 1 || Self::fanout(coords, miny, cell_h, buckets) <= cap {
                return (buckets, cell_h);
            }
            buckets /= 2;
        }
    }

    /// Total bucket slots the edges would take at this bucket count.
    fn fanout(coords: &[Coord<f64>], miny: f64, cell_h: f64, buckets: usize) -> usize {
        let mut total = 0usize;
        for edge in coords.windows(2) {
            let (lo, hi) = Self::span(edge, miny, cell_h, buckets);
            total = total.saturating_add(hi - lo + 1);
        }
        total
    }

    /// The inclusive bucket range one edge covers.
    #[inline]
    fn span(edge: &[Coord<f64>], miny: f64, cell_h: f64, buckets: usize) -> (usize, usize) {
        let lo = bucket_index(edge[0].y.min(edge[1].y), miny, cell_h, buckets);
        let hi = bucket_index(edge[0].y.max(edge[1].y), miny, cell_h, buckets);
        (lo, hi)
    }

    /// Where the point sits relative to this ring.
    ///
    /// This is [`geo::coordinate_position::coord_pos_relative_to_ring`] over the edges of one
    /// bucket. Every edge outside that bucket misses the y of the point, and the crossing rules
    /// below give such an edge no weight, so the two answers are the same.
    #[inline]
    fn locate(&self, coord: Coord<f64>) -> CoordPos {
        // A ring with no edge generates no crossing. `geo` reads a single vertex as a boundary.
        if self.coords.len() < 2 {
            return match self.coords.first() {
                Some(only) if *only == coord => CoordPos::OnBoundary,
                _ => CoordPos::Outside,
            };
        }
        // A point outside the box of the ring crosses each upward edge and each downward edge in
        // equal number, so its winding number is zero. It is not on the ring either.
        if coord.x < self.bbox.minx
            || coord.x > self.bbox.maxx
            || coord.y < self.bbox.miny
            || coord.y > self.bbox.maxy
        {
            return CoordPos::Outside;
        }

        let bucket = bucket_index(coord.y, self.bbox.miny, self.cell_h, self.buckets);
        let from = self.starts[bucket] as usize;
        let to = self.starts[bucket + 1] as usize;

        // Edge crossing rules, copied from `geo`:
        //   1. an upward edge includes its starting endpoint, and excludes its final endpoint;
        //   2. a downward edge excludes its starting endpoint, and includes its final endpoint;
        //   3. horizontal edges are excluded;
        //   4. the edge-ray intersection point must be strictly right of the coord.
        let mut winding = 0i32;
        for &id in &self.ids[from..to] {
            let start = self.coords[id as usize];
            let end = self.coords[id as usize + 1];
            if start.y <= coord.y {
                if end.y >= coord.y {
                    let orientation = <f64 as GeoNum>::Ker::orient2d(start, end, coord);
                    if orientation == Orientation::CounterClockwise && end.y != coord.y {
                        winding += 1;
                    } else if orientation == Orientation::Collinear
                        && value_in_between(coord.x, start.x, end.x)
                    {
                        return CoordPos::OnBoundary;
                    }
                }
            } else if end.y <= coord.y {
                let orientation = <f64 as GeoNum>::Ker::orient2d(start, end, coord);
                if orientation == Orientation::Clockwise {
                    winding -= 1;
                } else if orientation == Orientation::Collinear
                    && value_in_between(coord.x, start.x, end.x)
                {
                    return CoordPos::OnBoundary;
                }
            }
        }

        if winding == 0 {
            CoordPos::Outside
        } else {
            CoordPos::Inside
        }
    }
}

/// The bucket a y value falls in, clamped to the grid.
#[inline]
fn bucket_index(y: f64, miny: f64, cell_h: f64, buckets: usize) -> usize {
    let offset = (y - miny) / cell_h;
    // A NaN offset fails this test and lands in bucket zero. So does a negative one.
    if offset > 0.0 {
        (offset as usize).min(buckets - 1)
    } else {
        0
    }
}

/// True when `value` lies between the two bounds, in either order.
///
/// `geo` keeps its own copy of this private to the crate. The crossing rules above need the same
/// inclusive test, so this repeats it.
#[inline]
fn value_in_between(value: f64, bound_1: f64, bound_2: f64) -> bool {
    if bound_1 < bound_2 {
        value >= bound_1 && value <= bound_2
    } else {
        value >= bound_2 && value <= bound_1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Bbox {
        Bbox {
            minx,
            miny,
            maxx,
            maxy,
        }
    }

    /// The index must never drop a true overlap. This is the only property that matters.
    fn assert_superset(boxes: &[Bbox], probes: &[Bbox]) {
        let grid = BboxGrid::build(boxes);
        let mut candidates = Candidates::new(boxes.len());
        for probe in probes {
            grid.query(probe, &mut candidates);
            let found: std::collections::HashSet<u32> = candidates.ids().iter().copied().collect();
            for (id, build) in boxes.iter().enumerate() {
                if build.intersects(probe) {
                    assert!(
                        found.contains(&(id as u32)),
                        "row {id} overlaps {probe:?} but the grid missed it"
                    );
                }
            }
        }
    }

    #[test]
    fn a_scattered_grid_finds_every_overlap() {
        let mut state = 0x5EEDu64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let boxes: Vec<Bbox> = (0..500)
            .map(|_| {
                let (x, y) = (next() * 100.0, next() * 100.0);
                bbox(x, y, x + next() * 5.0, y + next() * 5.0)
            })
            .collect();
        let probes: Vec<Bbox> = (0..100)
            .map(|_| {
                let (x, y) = (next() * 100.0, next() * 100.0);
                bbox(x, y, x + 3.0, y + 3.0)
            })
            .collect();
        assert_superset(&boxes, &probes);
    }

    /// Every box in one spot. The grid degenerates to one cell and must still be correct.
    #[test]
    fn identical_boxes_still_answer() {
        let boxes = vec![bbox(1.0, 1.0, 2.0, 2.0); 50];
        assert_superset(
            &boxes,
            &[bbox(1.5, 1.5, 1.6, 1.6), bbox(9.0, 9.0, 9.1, 9.1)],
        );
    }

    /// A single point has zero width and zero height. The cell size must not become infinite.
    #[test]
    fn a_degenerate_extent_does_not_divide_by_zero() {
        let boxes = vec![bbox(3.0, 4.0, 3.0, 4.0)];
        assert_superset(
            &boxes,
            &[bbox(3.0, 4.0, 3.0, 4.0), bbox(0.0, 0.0, 1.0, 1.0)],
        );
    }

    #[test]
    fn an_empty_box_is_never_a_candidate() {
        let boxes = vec![Bbox::EMPTY, bbox(0.0, 0.0, 1.0, 1.0), Bbox::EMPTY];
        let grid = BboxGrid::build(&boxes);
        assert_eq!(grid.len(), 1);

        let mut candidates = Candidates::new(boxes.len());
        grid.query(&bbox(-10.0, -10.0, 10.0, 10.0), &mut candidates);
        assert_eq!(candidates.ids(), &[1]);
    }

    #[test]
    fn an_empty_build_side_answers_nothing() {
        let grid = BboxGrid::build(&[]);
        assert!(grid.is_empty());
        let mut candidates = Candidates::new(0);
        grid.query(&bbox(0.0, 0.0, 1.0, 1.0), &mut candidates);
        assert!(candidates.ids().is_empty());
    }

    /// A box that covers many cells must appear once, not once per cell.
    #[test]
    fn a_wide_box_is_reported_once() {
        let mut boxes: Vec<Bbox> = (0..100)
            .map(|i| bbox(i as f64, 0.0, i as f64 + 0.1, 1.0))
            .collect();
        boxes.push(bbox(0.0, 0.0, 100.0, 1.0));
        let wide = (boxes.len() - 1) as u32;

        let grid = BboxGrid::build(&boxes);
        let mut candidates = Candidates::new(boxes.len());
        grid.query(&bbox(0.0, 0.0, 100.0, 1.0), &mut candidates);
        assert_eq!(
            candidates.ids().iter().filter(|&&id| id == wide).count(),
            1,
            "the wide row must appear once"
        );
    }
}

#[cfg(test)]
mod point_in_polygon_tests {
    use geo::{Contains, MultiPolygon};

    use super::*;

    /// The same cheap generator the benchmarks use, so a failure is repeatable.
    fn next(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn ring(points: Vec<(f64, f64)>) -> LineString<f64> {
        LineString::new(points.into_iter().map(|(x, y)| Coord { x, y }).collect())
    }

    /// A regular polygon with `sides` vertices, centred on the origin.
    fn regular(sides: usize, radius: f64) -> LineString<f64> {
        let mut coords: Vec<Coord<f64>> = (0..sides)
            .map(|i| {
                let angle = (i as f64) / (sides as f64) * std::f64::consts::TAU;
                Coord {
                    x: radius * angle.cos(),
                    y: radius * angle.sin(),
                }
            })
            .collect();
        coords.push(coords[0]);
        LineString::new(coords)
    }

    /// A star, so the ray crosses the ring more than twice on many rows.
    fn star(points: usize) -> LineString<f64> {
        let mut coords: Vec<Coord<f64>> = (0..points * 2)
            .map(|i| {
                let angle = (i as f64) / ((points * 2) as f64) * std::f64::consts::TAU;
                let radius = if i % 2 == 0 { 1.0 } else { 0.35 };
                Coord {
                    x: radius * angle.cos(),
                    y: radius * angle.sin(),
                }
            })
            .collect();
        coords.push(coords[0]);
        LineString::new(coords)
    }

    /// The one property that matters: the index answers what `geo::Contains` answers.
    fn assert_agrees(geometry: &Geometry<f64>, probes: &[Coord<f64>]) {
        let index = PointInPolygonIndex::new(geometry).expect("a polygonal geometry");
        for probe in probes {
            assert_eq!(
                index.contains(*probe),
                geometry.contains(probe),
                "{probe:?} disagreed with geo on {geometry:?}"
            );
        }
    }

    /// Probes over a square of `spread`, plus every vertex and every edge midpoint.
    ///
    /// The vertices and the midpoints sit exactly on the boundary. Those are the rows where a
    /// crossing rule that differs from `geo` shows up.
    fn probes(geometry: &Geometry<f64>, count: usize, spread: f64) -> Vec<Coord<f64>> {
        let mut state = 0x5EEDu64;
        let mut out: Vec<Coord<f64>> = (0..count)
            .map(|_| Coord {
                x: (next(&mut state) - 0.5) * spread,
                y: (next(&mut state) - 0.5) * spread,
            })
            .collect();

        let rings: Vec<&LineString<f64>> = match geometry {
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
                out.push(edge[0]);
                out.push(Coord {
                    x: (edge[0].x + edge[1].x) / 2.0,
                    y: (edge[0].y + edge[1].y) / 2.0,
                });
            }
        }
        out
    }

    #[test]
    fn a_plain_ring_agrees_with_geo() {
        for sides in [3usize, 5, 64, 256, 1000] {
            let geometry = Geometry::Polygon(Polygon::new(regular(sides, 1.0), vec![]));
            assert_agrees(&geometry, &probes(&geometry, 4000, 2.4));
        }
    }

    #[test]
    fn a_concave_ring_agrees_with_geo() {
        let geometry = Geometry::Polygon(Polygon::new(star(40), vec![]));
        assert_agrees(&geometry, &probes(&geometry, 4000, 2.4));
    }

    /// A hole wound the other way round from the shell. This is the well formed case.
    #[test]
    fn a_hole_agrees_with_geo() {
        let mut hole = regular(128, 0.4).0;
        hole.reverse();
        let geometry =
            Geometry::Polygon(Polygon::new(regular(256, 1.0), vec![LineString::new(hole)]));
        assert_agrees(&geometry, &probes(&geometry, 4000, 2.4));
    }

    /// A hole wound the same way as the shell. `geo` reads the rings one at a time, so the point
    /// in the hole is still outside. A single winding number over every ring would say inside.
    #[test]
    fn a_hole_wound_like_its_shell_agrees_with_geo() {
        let geometry = Geometry::Polygon(Polygon::new(regular(128, 1.0), vec![regular(64, 0.4)]));
        let index = PointInPolygonIndex::new(&geometry).unwrap();
        let centre = Coord { x: 0.0, y: 0.0 };
        assert!(
            !geometry.contains(&centre),
            "geo puts the centre in the hole"
        );
        assert!(!index.contains(centre), "the index must say the same");
        assert_agrees(&geometry, &probes(&geometry, 4000, 2.4));
    }

    #[test]
    fn a_multi_polygon_agrees_with_geo() {
        let shift = |ring: LineString<f64>, dx: f64| {
            LineString::new(
                ring.0
                    .into_iter()
                    .map(|c| Coord {
                        x: c.x + dx,
                        y: c.y,
                    })
                    .collect(),
            )
        };
        let geometry = Geometry::MultiPolygon(MultiPolygon::new(vec![
            Polygon::new(shift(regular(64, 0.9), -1.0), vec![]),
            Polygon::new(shift(regular(200, 0.9), 1.0), vec![]),
        ]));
        assert_agrees(&geometry, &probes(&geometry, 6000, 5.0));
    }

    /// Long horizontal edges make one edge land in many buckets. The build must cap that and
    /// still answer correctly.
    #[test]
    fn a_comb_with_long_edges_agrees_with_geo() {
        let teeth = 60;
        let mut points = vec![(0.0, 0.0)];
        for tooth in 0..teeth {
            let y = tooth as f64;
            points.push((100.0, y));
            points.push((100.0, y + 0.5));
            points.push((0.0, y + 0.5));
            points.push((0.0, y + 1.0));
        }
        points.push((0.0, 0.0));
        let geometry = Geometry::Polygon(Polygon::new(ring(points), vec![]));

        let index = PointInPolygonIndex::new(&geometry).unwrap();
        let shell = &index.polygons[0].rings[0];
        assert!(
            shell.ids.len() <= shell.coords.len().saturating_sub(1) * MAX_FANOUT,
            "the build must cap the fanout"
        );

        let mut state = 0xC0FFEEu64;
        let mut spread: Vec<Coord<f64>> = (0..8000)
            .map(|_| Coord {
                x: next(&mut state) * 110.0 - 5.0,
                y: next(&mut state) * 70.0 - 5.0,
            })
            .collect();
        spread.extend(probes(&geometry, 0, 0.0));
        assert_agrees(&geometry, &spread);
    }

    /// A ring with no area still has to answer, and must not divide by zero.
    #[test]
    fn a_degenerate_ring_answers() {
        let flat = Geometry::Polygon(Polygon::new(
            ring(vec![(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (0.0, 5.0)]),
            vec![],
        ));
        assert_agrees(
            &flat,
            &[
                Coord { x: 0.5, y: 5.0 },
                Coord { x: 0.5, y: 4.0 },
                Coord { x: 9.0, y: 5.0 },
            ],
        );

        let empty = Geometry::Polygon(Polygon::new(LineString::new(Vec::new()), vec![]));
        assert_agrees(&empty, &[Coord { x: 0.0, y: 0.0 }]);

        let dot = Geometry::Polygon(Polygon::new(ring(vec![(3.0, 4.0)]), vec![]));
        assert_agrees(&dot, &[Coord { x: 3.0, y: 4.0 }, Coord { x: 0.0, y: 0.0 }]);
    }

    /// A point with no coordinate reaches the kernel as NaN. It must be outside, not a panic.
    #[test]
    fn a_nan_probe_is_outside() {
        let geometry = Geometry::Polygon(Polygon::new(regular(128, 1.0), vec![]));
        let index = PointInPolygonIndex::new(&geometry).unwrap();
        assert!(!index.contains(Coord {
            x: f64::NAN,
            y: f64::NAN
        }));
    }

    #[test]
    fn a_non_polygonal_literal_has_no_index() {
        let line = Geometry::LineString(regular(64, 1.0));
        assert!(PointInPolygonIndex::new(&line).is_none());
        assert!(PointInPolygonIndex::new(&Geometry::Point(geo::point!(x: 0.0, y: 0.0))).is_none());
    }
}
