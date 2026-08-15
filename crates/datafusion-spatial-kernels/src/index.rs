//! A bounding box index for spatial joins.
//!
//! A join on `ST_Intersects(a.geom, b.geom)` has no equality key, so DataFusion runs it as a
//! nested loop: every row of one side against every row of the other. The box prefilter makes one
//! pair cheap, but the pair count is the product of the two row counts.
//!
//! This grid changes that. It buckets the build side by bounding box. It then answers a probe
//! box with the rows of the cells that overlap.
//!
//! # Why a uniform grid and not an R-tree
//!
//! A grid builds in one pass. It needs no tree, no recursion and no rebalance step. Its weakness is
//! skewed
//! data, where one cell can hold most of the rows. The exact test still runs on those, so the
//! answer stays correct and the cost degrades toward the nested loop it replaced. It never does
//! worse than that.

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
