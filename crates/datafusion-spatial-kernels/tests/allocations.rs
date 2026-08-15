//! Allocation budgets, asserted rather than assumed.
//!
//! A flamegraph shows a wide `malloc` frame when a kernel allocates per row. This file catches the
//! same defect without a profiler, deterministically, and it runs in CI on any machine.
//!
//! Each budget is checked at two batch sizes. A count that grows with the row count is the bug.
//!
//! # Why one test function
//!
//! [`dhat::Alloc`] is a global allocator and counts every thread in the process. The default test
//! harness runs test functions in parallel. A second test that allocates in the background would
//! land in the count of this test. One entry point keeps the measurement deterministic.

use datafusion_spatial_kernels::accessor::st_x;
use datafusion_spatial_kernels::aggregate::Extent;
use datafusion_spatial_kernels::bbox::fill_bboxes;
use datafusion_spatial_kernels::envelope::{bound, box_output_type, st_envelope, Bound};
use datafusion_spatial_kernels::predicate::{
    st_intersects_scalar, st_intersects_with, PredicateScratch, PreparedLiteral,
};
use geoarrow_array::array::{PointArray, PolygonArray};
use geoarrow_array::builder::{PointBuilder, PolygonBuilder};
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::{CoordType, Dimension, PointType, PolygonType};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Points spread over a wide square, so the bounding box prefilter rejects nearly every row.
fn points(count: usize, coord_type: CoordType) -> PointArray {
    let mut state = 0x5EEDu64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let values: Vec<geo::Point<f64>> = (0..count)
        .map(|_| geo::point! { x: (next() - 0.5) * 1000.0, y: (next() - 0.5) * 1000.0 })
        .collect();

    PointBuilder::from_points(
        values.iter(),
        PointType::new(Dimension::XY, Default::default()).with_coord_type(coord_type),
    )
    .finish()
}

/// `count` copies of one big holed square, big enough to cover every point above.
///
/// Every row then survives the box test, so the kernel must build a geometry for each one. That
/// is the path the budget below guards.
fn polygon_column(count: usize) -> PolygonArray {
    let shell = geo::LineString::new(vec![
        geo::coord! { x: -1000.0, y: -1000.0 },
        geo::coord! { x: 1000.0, y: -1000.0 },
        geo::coord! { x: 1000.0, y: 1000.0 },
        geo::coord! { x: -1000.0, y: 1000.0 },
        geo::coord! { x: -1000.0, y: -1000.0 },
    ]);
    let hole = geo::LineString::new(vec![
        geo::coord! { x: -1.0, y: -1.0 },
        geo::coord! { x: 1.0, y: -1.0 },
        geo::coord! { x: 1.0, y: 1.0 },
        geo::coord! { x: -1.0, y: -1.0 },
    ]);
    let shapes = vec![geo::Polygon::new(shell, vec![hole]); count];
    PolygonBuilder::from_polygons(&shapes, PolygonType::new(Dimension::XY, Default::default()))
        .finish()
}

fn square() -> geo::Geometry<f64> {
    geo::Geometry::Polygon(geo::Polygon::new(
        geo::LineString::new(vec![
            geo::coord! { x: -1.0, y: -1.0 },
            geo::coord! { x: 1.0, y: -1.0 },
            geo::coord! { x: 1.0, y: 1.0 },
            geo::coord! { x: -1.0, y: 1.0 },
            geo::coord! { x: -1.0, y: -1.0 },
        ]),
        vec![],
    ))
}

/// Current allocation count.
fn mark() -> u64 {
    dhat::HeapStats::get().total_blocks
}

/// Heap blocks allocated since [`mark`].
fn blocks_since(before: u64) -> u64 {
    dhat::HeapStats::get().total_blocks - before
}

/// Collects every breach so one run reports all of them.
#[derive(Default)]
struct Report(Vec<String>);

impl Report {
    fn check(&mut self, what: &str, rows: usize, blocks: u64, budget: u64) {
        if blocks > budget {
            self.0.push(format!(
                "{what} over {rows} rows allocated {blocks} blocks, budget is {budget}"
            ));
        }
    }
}

#[test]
fn allocation_budgets() {
    let _profiler = dhat::Profiler::builder().testing().build();
    let mut report = Report::default();

    for rows in [1024, 65536] {
        // ST_X on separated coordinates hands back the input buffer. There is nothing to allocate.
        let separated = points(rows, CoordType::Separated);
        let before = mark();
        let result = st_x(&separated).unwrap();
        report.check("ST_X separated", rows, blocks_since(before), 0);
        assert_eq!(result.len(), rows);

        // Interleaved coordinates need a strided read into one new buffer, and no more.
        //
        // Two blocks is the floor for a fresh Arrow buffer: one for the `Vec` of
        // values, one for the `Arc` that `Buffer::from_vec` wraps around it. The count must not
        // move when the row count grows by 64 times.
        let interleaved = points(rows, CoordType::Interleaved);
        let before = mark();
        let result = st_x(&interleaved).unwrap();
        report.check("ST_X interleaved", rows, blocks_since(before), 2);
        assert_eq!(result.len(), rows);

        // ST_Extent never builds a geometry, so it never allocates.
        let mut extent = Extent::new();
        let before = mark();
        extent.update(&separated).unwrap();
        report.check("ST_Extent", rows, blocks_since(before), 0);
        assert!(extent.finish().is_some());

        // A warm bounding box vector must be reused, not reallocated.
        let mut boxes = Vec::new();
        fill_bboxes(&separated, &mut boxes).unwrap();
        let before = mark();
        for _ in 0..8 {
            fill_bboxes(&separated, &mut boxes).unwrap();
        }
        report.check("fill_bboxes (8 warm passes)", rows, blocks_since(before), 0);

        // A box column already stores the four corner ordinates. To read one is a buffer
        // handoff, exactly like ST_X over a point column.
        let output = box_output_type(&GeoArrowArray::data_type(&separated));
        let envelope = st_envelope(&separated, output).unwrap();
        let before = mark();
        let xmin = bound(envelope.as_ref(), Bound::XMin).unwrap();
        report.check("ST_XMin over a box column", rows, blocks_since(before), 0);
        assert_eq!(xmin.len(), rows);

        // ST_Intersects against a constant allocates the output buffer and nothing per row.
        let literal = PreparedLiteral::new(square());
        let mut scratch = PredicateScratch::new();
        st_intersects_scalar(&separated, &literal, &mut scratch).unwrap();
        let before = mark();
        let result = st_intersects_scalar(&separated, &literal, &mut scratch).unwrap();
        report.check("ST_Intersects constant", rows, blocks_since(before), 4);
        assert_eq!(result.len(), rows);

        // Two columns, and every row survives the box test. The kernel reuses one geometry, so the
        // count must not move when the row count grows by 64 times. A build per row would
        // allocate one `Vec` per ring, which is two blocks per row.
        let polygons = polygon_column(rows);
        let mut pair_scratch = PredicateScratch::new();
        st_intersects_with(&separated, &polygons, &mut pair_scratch).unwrap();
        let before = mark();
        let result = st_intersects_with(&separated, &polygons, &mut pair_scratch).unwrap();
        report.check("ST_Intersects two columns", rows, blocks_since(before), 12);
        assert_eq!(result.len(), rows);
    }

    assert!(
        report.0.is_empty(),
        "allocation budgets exceeded:\n  {}\n\nAn allocation count that grows with the row count \
         means a kernel allocates per row.",
        report.0.join("\n  ")
    );
}
