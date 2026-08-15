//! `ST_Intersects` throughput.
//!
//! Four questions this benchmark answers:
//!
//! 1. What does the bounding box prefilter buy? `prefilter/on` against `prefilter/off`.
//! 2. What does the R-tree buy, and on which algorithm? The four `literal/*` cases.
//! 3. What does WKB input cost? `encoding/native` against `encoding/wkb`.
//! 4. What does a column argument cost against a constant? `shape/*`.

use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_spatial_kernels::materialize::GeometryReader;
use datafusion_spatial_kernels::predicate::{
    st_intersects_scalar, st_intersects_with, PredicateScratch, PreparedLiteral,
};
use geoarrow_array::builder::PolygonBuilder;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::{CoordType, Dimension, PolygonType};
use std::hint::black_box;

mod common;
use common::{points, points_as_wkb, regular_polygon, square, BATCH};

/// The exact test with no bounding box prefilter. Baseline for the prefilter measurement.
fn intersects_without_prefilter(array: &dyn GeoArrowArray, literal: &PreparedLiteral) -> usize {
    // Hoist the reader, exactly as the real kernel hoists its own. A reader built per row would
    // price the downcast into this baseline and overstate what the prefilter is worth.
    let mut reader = GeometryReader::new(array).unwrap();
    let mut hits = 0;
    for index in 0..array.len() {
        if let Some(geom) = reader.read(index).unwrap() {
            if literal.intersects(geom) {
                hits += 1;
            }
        }
    }
    hits
}

/// A query box of half-size 1 against points spread over 1000 units.
///
/// Roughly 1 row in 250000 falls inside, so nearly every row is answered by the box test.
const SPREAD: f64 = 1000.0;

fn bench_prefilter(c: &mut Criterion) {
    let mut group = c.benchmark_group("ST_Intersects/prefilter");
    group.throughput(criterion::Throughput::Elements(BATCH as u64));

    let array = points(BATCH, SPREAD, CoordType::Separated);
    let literal = PreparedLiteral::new(square(1.0));
    let mut scratch = PredicateScratch::new();

    group.bench_function("on", |b| {
        b.iter(|| {
            black_box(st_intersects_scalar(black_box(&array), &literal, &mut scratch).unwrap())
        })
    });

    group.bench_function("off", |b| {
        b.iter(|| black_box(intersects_without_prefilter(black_box(&array), &literal)))
    });

    // The baseline: the same DE-9IM predicate with no box verdict at all. The gap between this
    // and `st_touches` above is what the verdict is worth on selective data.
    let raw = square(1.0);
    group.bench_function("st_touches_no_verdict", |b| {
        b.iter(|| {
            let mut reader = GeometryReader::new(&array).unwrap();
            let mut hits = 0usize;
            for index in 0..array.len() {
                if let Some(geom) = reader.read(index).unwrap() {
                    if geo::Relate::relate(&raw, geom).is_touches() {
                        hits += 1;
                    }
                }
            }
            black_box(hits)
        })
    });

    group.finish();
}

fn bench_literal(c: &mut Criterion) {
    let mut group = c.benchmark_group("ST_Intersects/literal");
    group.throughput(criterion::Throughput::Elements(BATCH as u64));

    // Points packed tightly inside the query shape, so the box test passes and the exact test runs
    // on every row. That isolates the cost of the exact test itself.
    let array = points(BATCH, 2.0, CoordType::Separated);
    let mut scratch = PredicateScratch::new();

    let small = PreparedLiteral::new(square(1.0));
    group.bench_function("intersects_5_vertices", |b| {
        b.iter(|| black_box(st_intersects_scalar(black_box(&array), &small, &mut scratch).unwrap()))
    });

    let big = PreparedLiteral::new(regular_polygon(256, 1.0));
    group.bench_function("intersects_256_vertices", |b| {
        b.iter(|| black_box(st_intersects_scalar(black_box(&array), &big, &mut scratch).unwrap()))
    });
    assert!(
        !big.is_indexed(),
        "ST_Intersects must never build the R-tree"
    );

    // The DE-9IM baselines. `relate` is the algorithm the R-tree accelerates, and it is what
    // ST_Relate, ST_Touches, ST_Crosses and ST_Overlaps will need. These two numbers price the
    // cache. Against `intersects_256_vertices` they price the algorithm.
    let unprepared = geo::PreparedGeometry::from(regular_polygon(256, 1.0));
    group.bench_function("relate_indexed_256_vertices", |b| {
        b.iter(|| {
            let mut reader = GeometryReader::new(&array).unwrap();
            let mut hits = 0usize;
            for index in 0..array.len() {
                if let Some(geom) = reader.read(index).unwrap() {
                    if geo::Relate::relate(&unprepared, geom).is_intersects() {
                        hits += 1;
                    }
                }
            }
            black_box(hits)
        })
    });

    let raw = regular_polygon(256, 1.0);
    group.bench_function("relate_unindexed_256_vertices", |b| {
        b.iter(|| {
            let mut reader = GeometryReader::new(&array).unwrap();
            let mut hits = 0usize;
            for index in 0..array.len() {
                if let Some(geom) = reader.read(index).unwrap() {
                    if geo::Relate::relate(&raw, geom).is_intersects() {
                        hits += 1;
                    }
                }
            }
            black_box(hits)
        })
    });

    group.finish();
}

fn bench_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("ST_Intersects/encoding");
    group.throughput(criterion::Throughput::Elements(BATCH as u64));

    let literal = PreparedLiteral::new(square(1.0));
    let mut scratch = PredicateScratch::new();

    let native = points(BATCH, SPREAD, CoordType::Separated);
    group.bench_function("native", |b| {
        b.iter(|| {
            black_box(st_intersects_scalar(black_box(&native), &literal, &mut scratch).unwrap())
        })
    });

    let wkb = points_as_wkb(BATCH, SPREAD);
    group.bench_function("wkb", |b| {
        b.iter(|| black_box(st_intersects_scalar(black_box(&wkb), &literal, &mut scratch).unwrap()))
    });

    group.finish();
}

fn bench_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("ST_Intersects/shape");
    group.throughput(criterion::Throughput::Elements(BATCH as u64));

    let array = points(BATCH, SPREAD, CoordType::Separated);
    let literal = PreparedLiteral::new(square(1.0));
    let mut scratch = PredicateScratch::new();

    group.bench_function("constant_argument", |b| {
        b.iter(|| {
            black_box(st_intersects_scalar(black_box(&array), &literal, &mut scratch).unwrap())
        })
    });

    // The same square repeated once per row, which is what a column argument looks like.
    let geo::Geometry::Polygon(polygon) = square(1.0) else {
        unreachable!()
    };
    let squares = vec![polygon; BATCH];
    let column = PolygonBuilder::from_polygons(
        &squares,
        PolygonType::new(Dimension::XY, Default::default()),
    )
    .finish();

    group.bench_function("column_argument", |b| {
        b.iter(|| {
            black_box(
                st_intersects_with(black_box(&array), black_box(&column), &mut scratch).unwrap(),
            )
        })
    });

    // Points packed inside the polygons. The box verdict cannot settle a row here. The kernel
    // must build a geometry for every one. This is the path `materialize.rs` targets, and the
    // only one where the cost to build a row geometry is visible.
    let packed = points(BATCH, 2.0, CoordType::Separated);
    for (name, vertices) in [("column_overlap_5v", 5usize), ("column_overlap_256v", 256)] {
        let geo::Geometry::Polygon(shape) = regular_polygon(vertices, 1.0) else {
            unreachable!()
        };
        let shapes = vec![shape; BATCH];
        let polygons = PolygonBuilder::from_polygons(
            &shapes,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(
                    st_intersects_with(black_box(&packed), black_box(&polygons), &mut scratch)
                        .unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_prefilter,
    bench_literal,
    bench_encoding,
    bench_shape,
    bench_verdicts
);
criterion_main!(benches);

/// What the per-predicate box verdicts are worth.
///
/// A generic "do the boxes overlap" filter only helps `ST_Intersects`. Containment and equality
/// get a strictly stronger test, and `ST_Disjoint` is answered outright by disjoint boxes.
fn bench_verdicts(c: &mut Criterion) {
    use datafusion_spatial_kernels::predicate::{st_predicate_scalar, Predicate, Side};

    let mut group = c.benchmark_group("ST_Predicate/verdict");
    group.throughput(criterion::Throughput::Elements(BATCH as u64));

    let array = points(BATCH, SPREAD, CoordType::Separated);
    let literal = PreparedLiteral::new(square(1.0));
    let mut scratch = PredicateScratch::new();

    for predicate in [
        Predicate::Intersects,
        Predicate::Disjoint,
        Predicate::Contains,
        Predicate::Within,
        Predicate::Equals,
        Predicate::Touches,
    ] {
        group.bench_function(predicate.sql_name(), |b| {
            b.iter(|| {
                black_box(
                    st_predicate_scalar(
                        black_box(&array),
                        &literal,
                        predicate,
                        Side::Right,
                        &mut scratch,
                    )
                    .unwrap(),
                )
            })
        });
    }

    // The baseline: the same DE-9IM predicate with no box verdict at all. The gap between this
    // and `st_touches` above is what the verdict is worth on selective data.
    let raw = square(1.0);
    group.bench_function("st_touches_no_verdict", |b| {
        b.iter(|| {
            let mut reader = GeometryReader::new(&array).unwrap();
            let mut hits = 0usize;
            for index in 0..array.len() {
                if let Some(geom) = reader.read(index).unwrap() {
                    if geo::Relate::relate(&raw, geom).is_touches() {
                        hits += 1;
                    }
                }
            }
            black_box(hits)
        })
    });

    group.finish();
}
