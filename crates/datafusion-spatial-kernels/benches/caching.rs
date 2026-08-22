//! Does a per-batch cache pay for `ST_Intersects` and `ST_Within`?
//!
//! Both predicates take the direct `geo` trait, never the DE-9IM matrix, so the R-tree inside
//! `PreparedLiteral` never runs for them. On the column-vs-column path the remaining cost is one
//! `geo::Polygon` built for every row. `PROFILE.md` prices that path at 6.3 times the constant
//! case, and names it as the thing to fix.
//!
//! This benchmark prices six ways to spend that budget against a controlled duplicate rate.
//!
//! | Loop | What it does |
//! |---|---|
//! | `baseline` | Build the row geometry. This is the current kernel. |
//! | `oracle` | Read a pre-built table by index. No key cost, no probe cost. |
//! | `memo_result` | Hash both sides. Cache the boolean answer. |
//! | `memo_geometry` | Hash the right side. Cache the built geometry. |
//! | `memo_previous` | Compare the key against the previous row only. One entry, no map. |
//! | `scratch` | No cache. Refill one reused `Polygon` from the Arrow buffers. |
//!
//! `oracle` is the decisive line. It is what a cache costs when the key and the probe are free.
//! No real cache can beat it. If `oracle` does not beat `baseline` by a wide margin, then no
//! cache pays, whatever its hit rate.
//!
//! The left side is a pre-built `Vec<geo::Point>` in every loop. The point side is not the
//! variable under test. A fixed left side keeps the right side the only difference.
//!
//! # A repeat check over the row box
//!
//! `row_repeats/*` prices the check that `st_predicate_scalar` ships. A point column often
//! repeats a coordinate on neighbouring rows, and such a row can reuse the answer of the row
//! before it.
//!
//! The check sits in front of the geometry build, so a repeat skips that as well as the exact
//! test. It needs no key of its own either: the scalar loop has already read the box of the row,
//! and for a point that box is the coordinate. Two comparisons settle it.
//!
//! `no_repeats` and `repeats` are the same loop and differ only in those comparisons. `kernel` is
//! the shipped call, which carries the check and also builds the output array. Over 8192 rows
//! against a 5000 vertex ring, in microseconds:
//!
//! | Ring | Pattern | Distinct | `no_repeats` | `repeats` | Change |
//! |---|---|--:|--:|--:|--:|
//! | 5000v | blocked | 1 | 140.9 | 10.8 | -92% |
//! | 5000v | blocked | 64 | 156.9 | 14.9 | -90% |
//! | 5000v | blocked | 8192 | 364.1 | 362.4 | -0.5% |
//! | 5000v | cyclic | 4 | 143.2 | 147.2 | +2.7% |
//! | 5000v | cyclic | 8192 | 356.1 | 362.2 | +1.7% |
//! | 16v | blocked | 1 | 202.9 | 11.5 | -94% |
//! | 16v | blocked | 64 | 213.9 | 14.7 | -93% |
//! | 16v | blocked | 8192 | 336.0 | 357.9 | +6.5% |
//! | 16v | cyclic | 8192 | 347.2 | 329.5 | -5.1% |
//!
//! A run of repeats runs twelve to sixteen times faster, and the small ring gains more than the
//! large one because it has no index to fall back on. A column of distinct points shows no
//! consistent cost. The two 8192 rows at 16 vertices disagree on the sign, and a pair of hundred
//! sample runs at 5000 vertices gave minus one and plus two per cent. The `no_repeats` baseline
//! itself moves by three per cent between runs, so any cost here sits inside the noise.

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use geo::Intersects;
use geo_traits::to_geo::ToGeoPolygon;
use geo_traits::{CoordTrait, LineStringTrait, PolygonTrait};
use geoarrow_array::array::{CoordBuffer, PointArray, PolygonArray};
use geoarrow_array::builder::{PointBuilder, PolygonBuilder};
use geoarrow_array::GeoArrowArrayAccessor;
use geoarrow_schema::{CoordType, Dimension, PointType, PolygonType};

use datafusion_spatial_kernels::bbox::{fill_bboxes, Bbox};
use datafusion_spatial_kernels::materialize::GeometryReader;
use datafusion_spatial_kernels::predicate::{
    st_predicate_scalar, Predicate, PredicateScratch, PreparedLiteral, Side,
};

mod common;
use common::{regular_polygon, Lcg, BATCH};

/// FNV-1a over 64 bit words. Fast, and it needs no dependency.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x1000_0000_01b3;

#[inline]
fn mix(hash: u64, word: u64) -> u64 {
    (hash ^ word).wrapping_mul(FNV_PRIME)
}

/// The point cloud. Spread 2.0 against a polygon of radius 1.0 puts most points inside the
/// polygon box. The box verdict cannot settle the row, so the exact test runs.
fn point_cloud() -> Vec<geo::Point<f64>> {
    let mut rng = Lcg::new(0x5EED);
    (0..BATCH)
        .map(|_| geo::point! { x: (rng.next_f64() - 0.5) * 2.0, y: (rng.next_f64() - 0.5) * 2.0 })
        .collect()
}

/// A regular polygon of `vertices` sides. `nudge` shifts the radius by a tiny amount, so two
/// shapes differ in their coordinates but cost the same to test.
fn ring(vertices: usize, nudge: usize) -> geo::Polygon<f64> {
    let radius = 1.0 + (nudge as f64) * 1e-6;
    let mut coords: Vec<geo::Coord<f64>> = (0..vertices)
        .map(|i| {
            let angle = (i as f64) / (vertices as f64) * std::f64::consts::TAU;
            geo::coord! { x: radius * angle.cos(), y: radius * angle.sin() }
        })
        .collect();
    coords.push(coords[0]);
    geo::Polygon::new(geo::LineString::new(coords), vec![])
}

/// Which distinct shape each row holds.
///
/// `blocked` gives contiguous runs, which is the shape a nested loop join produces. The other
/// pattern interleaves them, which is the shape a denormalized table produces.
fn assignment(distinct: usize, blocked: bool) -> Vec<usize> {
    (0..BATCH)
        .map(|i| {
            if blocked {
                i * distinct / BATCH
            } else {
                i % distinct
            }
        })
        .collect()
}

/// A polygon column of `BATCH` rows drawn from `distinct` shapes.
fn polygon_column(
    distinct: usize,
    vertices: usize,
    blocked: bool,
) -> (PolygonArray, Vec<geo::Polygon<f64>>, Vec<usize>) {
    let shapes: Vec<geo::Polygon<f64>> = (0..distinct).map(|k| ring(vertices, k)).collect();
    let assign = assignment(distinct, blocked);
    let rows: Vec<geo::Polygon<f64>> = assign.iter().map(|&k| shapes[k].clone()).collect();
    let array = PolygonBuilder::from_polygons(
        &rows,
        PolygonType::new(Dimension::XY, Default::default()).with_coord_type(CoordType::Separated),
    )
    .finish();
    (array, shapes, assign)
}

/// Raw buffer view of a polygon column, so a key needs no geometry.
///
/// A cache must derive its key without a build of the thing it wants to avoid. This reads the
/// coordinates straight out of the Arrow buffers.
struct Raw<'a> {
    x: &'a [f64],
    y: &'a [f64],
    geom: &'a [i32],
    ring: &'a [i32],
}

impl<'a> Raw<'a> {
    fn new(array: &'a PolygonArray) -> Self {
        let CoordBuffer::Separated(sep) = array.coords() else {
            panic!("this benchmark builds separated coordinates")
        };
        let buffers = sep.raw_buffers();
        Self {
            x: &buffers[0],
            y: &buffers[1],
            geom: &array.geom_offsets()[..],
            ring: &array.ring_offsets()[..],
        }
    }

    /// The coordinate range of one row, across every ring.
    #[inline]
    fn range(&self, index: usize) -> (usize, usize) {
        let first_ring = self.geom[index] as usize;
        let last_ring = self.geom[index + 1] as usize;
        (
            self.ring[first_ring] as usize,
            self.ring[last_ring] as usize,
        )
    }

    /// A hash of every coordinate in the row. This is the honest cost of a cache key.
    #[inline]
    fn key(&self, index: usize) -> u64 {
        let (start, end) = self.range(index);
        let mut hash = FNV_OFFSET;
        for k in start..end {
            hash = mix(hash, self.x[k].to_bits());
            hash = mix(hash, self.y[k].to_bits());
        }
        hash
    }
}

#[inline]
fn point_key(point: &geo::Point<f64>) -> u64 {
    mix(mix(FNV_OFFSET, point.x().to_bits()), point.y().to_bits())
}

// ---------------------------------------------------------------------------
// The six loops. Each one returns the hit count, so they can be cross-checked.
// ---------------------------------------------------------------------------

/// What the kernel does today: build a geometry for every row.
fn baseline(left: &[geo::Point<f64>], right: &PolygonArray) -> usize {
    let mut hits = 0;
    for (index, point) in left.iter().enumerate() {
        let polygon = right.get(index).unwrap().unwrap().to_polygon();
        if point.intersects(&polygon) {
            hits += 1;
        }
    }
    hits
}

/// The upper bound for any cache: a pre-built table and a free key.
fn oracle(left: &[geo::Point<f64>], shapes: &[geo::Polygon<f64>], assign: &[usize]) -> usize {
    let mut hits = 0;
    for (index, point) in left.iter().enumerate() {
        if point.intersects(&shapes[assign[index]]) {
            hits += 1;
        }
    }
    hits
}

/// Cache the boolean, keyed on both sides.
fn memo_result(left: &[geo::Point<f64>], right: &PolygonArray, raw: &Raw) -> usize {
    let mut cache: HashMap<(u64, u64), bool> = HashMap::new();
    let mut hits = 0;
    for (index, point) in left.iter().enumerate() {
        let key = (point_key(point), raw.key(index));
        let answer = match cache.get(&key) {
            Some(&value) => value,
            None => {
                let polygon = right.get(index).unwrap().unwrap().to_polygon();
                let value = point.intersects(&polygon);
                cache.insert(key, value);
                value
            }
        };
        if answer {
            hits += 1;
        }
    }
    hits
}

/// Cache the built geometry, keyed on the right side only.
fn memo_geometry(left: &[geo::Point<f64>], right: &PolygonArray, raw: &Raw) -> usize {
    let mut cache: HashMap<u64, geo::Polygon<f64>> = HashMap::new();
    let mut hits = 0;
    for (index, point) in left.iter().enumerate() {
        let polygon = cache
            .entry(raw.key(index))
            .or_insert_with(|| right.get(index).unwrap().unwrap().to_polygon());
        if point.intersects(&*polygon) {
            hits += 1;
        }
    }
    hits
}

/// A one entry cache: compare the key against the previous row.
fn memo_previous(left: &[geo::Point<f64>], right: &PolygonArray, raw: &Raw) -> usize {
    let mut held = geo::Polygon::new(geo::LineString::new(Vec::new()), vec![]);
    let mut held_key = 0u64;
    let mut primed = false;
    let mut hits = 0;
    for (index, point) in left.iter().enumerate() {
        let key = raw.key(index);
        if !primed || key != held_key {
            held = right.get(index).unwrap().unwrap().to_polygon();
            held_key = key;
            primed = true;
        }
        if point.intersects(&held) {
            hits += 1;
        }
    }
    hits
}

/// No cache. Refill one `Polygon` from the buffers, so the allocation happens once per batch.
///
/// The prototype reads the shell only. The benchmark polygons have no holes.
fn scratch(left: &[geo::Point<f64>], raw: &Raw) -> usize {
    let mut polygon = geo::Polygon::new(geo::LineString::new(Vec::with_capacity(512)), vec![]);
    let mut hits = 0;
    for (index, point) in left.iter().enumerate() {
        let (start, end) = raw.range(index);
        polygon.exterior_mut(|shell| {
            shell.0.clear();
            shell
                .0
                .extend((start..end).map(|k| geo::coord! { x: raw.x[k], y: raw.y[k] }));
        });
        if point.intersects(&polygon) {
            hits += 1;
        }
    }
    hits
}

/// Read the raw buffers, but allocate a new `Vec` for every row.
///
/// This prices the cheap option: it needs no change to the `Operand` API, because it still
/// returns an owned geometry.
fn fresh_fast(left: &[geo::Point<f64>], raw: &Raw) -> usize {
    let mut hits = 0;
    for (index, point) in left.iter().enumerate() {
        let (start, end) = raw.range(index);
        let mut shell = Vec::with_capacity(end - start);
        shell.extend((start..end).map(|k| geo::coord! { x: raw.x[k], y: raw.y[k] }));
        let polygon = geo::Polygon::new(geo::LineString::new(shell), vec![]);
        if point.intersects(&polygon) {
            hits += 1;
        }
    }
    hits
}

/// Refill through the `geo_traits` accessor, which is what a generic kernel can reach.
///
/// This is `scratch` without the raw buffer shortcut. The gap between the two prices the
/// accessor itself.
fn scratch_traits(left: &[geo::Point<f64>], right: &PolygonArray) -> usize {
    let mut polygon = geo::Polygon::new(geo::LineString::new(Vec::with_capacity(512)), vec![]);
    let mut hits = 0;
    for (index, point) in left.iter().enumerate() {
        let source = right.get(index).unwrap().unwrap();
        polygon.exterior_mut(|shell| {
            shell.0.clear();
            if let Some(ring) = source.exterior() {
                shell
                    .0
                    .extend(ring.coords().map(|c| geo::coord! { x: c.x(), y: c.y() }));
            }
        });
        if point.intersects(&polygon) {
            hits += 1;
        }
    }
    hits
}

/// Every loop must agree, or the numbers below mean nothing.
fn cross_check(
    left: &[geo::Point<f64>],
    right: &PolygonArray,
    raw: &Raw,
    shapes: &[geo::Polygon<f64>],
    assign: &[usize],
) -> usize {
    let expected = baseline(left, right);
    for (name, got) in [
        ("oracle", oracle(left, shapes, assign)),
        ("memo_result", memo_result(left, right, raw)),
        ("memo_geometry", memo_geometry(left, right, raw)),
        ("memo_previous", memo_previous(left, right, raw)),
        ("scratch", scratch(left, raw)),
        ("fresh_fast", fresh_fast(left, raw)),
        ("scratch_traits", scratch_traits(left, right)),
    ] {
        assert_eq!(got, expected, "{name} disagrees with baseline");
    }
    expected
}

fn sweep(c: &mut Criterion, vertices: usize, blocked: bool) {
    let label = if blocked { "blocked" } else { "cyclic" };
    let mut group = c.benchmark_group(format!("cache/{vertices}v/{label}"));
    group.throughput(criterion::Throughput::Elements(BATCH as u64));
    group.sample_size(50);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(3));

    let left = point_cloud();

    for distinct in [1usize, 64, BATCH] {
        let (right, shapes, assign) = polygon_column(distinct, vertices, blocked);
        let raw = Raw::new(&right);
        cross_check(&left, &right, &raw, &shapes, &assign);

        let id = |name: &str| BenchmarkId::new(name, distinct);

        group.bench_function(id("baseline"), |b| {
            b.iter(|| black_box(baseline(black_box(&left), black_box(&right))))
        });
        group.bench_function(id("oracle"), |b| {
            b.iter(|| black_box(oracle(black_box(&left), &shapes, &assign)))
        });
        group.bench_function(id("memo_result"), |b| {
            b.iter(|| black_box(memo_result(black_box(&left), black_box(&right), &raw)))
        });
        group.bench_function(id("memo_geometry"), |b| {
            b.iter(|| black_box(memo_geometry(black_box(&left), black_box(&right), &raw)))
        });
        group.bench_function(id("memo_previous"), |b| {
            b.iter(|| black_box(memo_previous(black_box(&left), black_box(&right), &raw)))
        });
        group.bench_function(id("scratch"), |b| {
            b.iter(|| black_box(scratch(black_box(&left), &raw)))
        });
        group.bench_function(id("fresh_fast"), |b| {
            b.iter(|| black_box(fresh_fast(black_box(&left), &raw)))
        });
        group.bench_function(id("scratch_traits"), |b| {
            b.iter(|| black_box(scratch_traits(black_box(&left), black_box(&right))))
        });
    }

    group.finish();
}

/// Small polygons, contiguous runs. The join shape, and the best case for a cache.
fn bench_small_blocked(c: &mut Criterion) {
    sweep(c, 5, true);
}

/// Small polygons, interleaved. The denormalized table shape.
fn bench_small_cyclic(c: &mut Criterion) {
    sweep(c, 5, false);
}

/// Large polygons. The key cost grows with the vertex count, and so does the exact test.
fn bench_large_blocked(c: &mut Criterion) {
    sweep(c, 256, true);
}

criterion_group!(
    benches,
    bench_small_blocked,
    bench_small_cyclic,
    bench_large_blocked,
    bench_row_repeats
);
criterion_main!(benches);

// ---------------------------------------------------------------------------------------------
// A repeat check over the row box
// ---------------------------------------------------------------------------------------------

/// `BATCH` points drawn from `distinct` positions inside the ring.
fn point_column(distinct: usize, blocked: bool) -> PointArray {
    let mut rng = Lcg::new(0xB0BA);
    let pool: Vec<geo::Point<f64>> = (0..distinct)
        .map(|_| geo::point! { x: (rng.next_f64() - 0.5) * 2.0, y: (rng.next_f64() - 0.5) * 2.0 })
        .collect();
    let rows: Vec<geo::Point<f64>> = assignment(distinct, blocked)
        .iter()
        .map(|&slot| pool[slot])
        .collect();
    PointBuilder::from_points(
        rows.iter(),
        PointType::new(Dimension::XY, Default::default()).with_coord_type(CoordType::Separated),
    )
    .finish()
}

/// The scalar loop without the repeat check. This is the kernel before the change.
fn rows_without_repeats(
    array: &PointArray,
    literal: &PreparedLiteral,
    boxes: &mut Vec<Bbox>,
) -> usize {
    fill_bboxes(array, boxes).unwrap();
    let mut reader = GeometryReader::new(array).unwrap();
    let literal_bbox = literal.bbox();
    let mut hits = 0usize;
    for (index, row_bbox) in boxes.iter().enumerate() {
        let answer = match Predicate::Within.bbox_verdict(row_bbox, &literal_bbox) {
            Some(verdict) => verdict,
            None => match reader.read(index).unwrap() {
                Some(geom) => literal.evaluate(Predicate::Within, geom, Side::Right),
                None => false,
            },
        };
        if answer {
            hits += 1;
        }
    }
    hits
}

/// The same loop with the repeat check. The only difference is the two comparisons.
fn rows_with_repeats(
    array: &PointArray,
    literal: &PreparedLiteral,
    boxes: &mut Vec<Bbox>,
) -> usize {
    fill_bboxes(array, boxes).unwrap();
    let mut reader = GeometryReader::new(array).unwrap();
    let literal_bbox = literal.bbox();
    let mut last_x = f64::NAN;
    let mut last_y = f64::NAN;
    let mut last_answer = false;
    let mut hits = 0usize;
    for (index, row_bbox) in boxes.iter().enumerate() {
        let answer = match Predicate::Within.bbox_verdict(row_bbox, &literal_bbox) {
            Some(verdict) => verdict,
            None if row_bbox.minx == last_x && row_bbox.miny == last_y => last_answer,
            None => {
                let answer = match reader.read(index).unwrap() {
                    Some(geom) => literal.evaluate(Predicate::Within, geom, Side::Right),
                    None => false,
                };
                last_x = row_bbox.minx;
                last_y = row_bbox.miny;
                last_answer = answer;
                answer
            }
        };
        if answer {
            hits += 1;
        }
    }
    hits
}

/// Does a repeat check over the row box pay for a point column?
///
/// The check sits before the geometry build, not inside the predicate, so a repeat skips both.
/// `no_repeats` and `repeats` are the same loop and differ only in the check. `kernel` is the
/// shipped call, which carries the check and also builds the output array.
fn bench_row_repeats(c: &mut Criterion) {
    // Two sizes. 5000 vertices sits far above the index threshold, so the exact test dominates
    // the row. 16 sits below it, so the row is mostly the read and the build, and the repeat
    // check has less to skip.
    for (vertices, blocked) in [(5000usize, true), (5000, false), (16, true), (16, false)] {
        let ring = regular_polygon(vertices, 1.0);
        let label = if blocked { "blocked" } else { "cyclic" };
        let mut group = c.benchmark_group(format!("row_repeats/{vertices}v/{label}"));
        group.throughput(criterion::Throughput::Elements(BATCH as u64));
        group.sample_size(50);
        group.warm_up_time(std::time::Duration::from_secs(1));
        group.measurement_time(std::time::Duration::from_secs(3));

        for distinct in [1usize, 4, 64, BATCH] {
            let array = point_column(distinct, blocked);
            let literal = PreparedLiteral::new(ring.clone());
            let mut boxes = Vec::new();
            let mut scratch = PredicateScratch::new();

            let expected = rows_without_repeats(&array, &literal, &mut boxes);
            assert_eq!(
                rows_with_repeats(&array, &literal, &mut boxes),
                expected,
                "the repeat check changed an answer"
            );
            assert_eq!(
                st_predicate_scalar(
                    &array,
                    &literal,
                    Predicate::Within,
                    Side::Right,
                    &mut scratch,
                )
                .unwrap()
                .values()
                .count_set_bits(),
                expected,
                "the kernel disagrees with the hand rolled loop"
            );

            let id = |name: &str| BenchmarkId::new(name, distinct);
            group.bench_function(id("no_repeats"), |b| {
                b.iter(|| {
                    black_box(rows_without_repeats(
                        black_box(&array),
                        &literal,
                        &mut boxes,
                    ))
                })
            });
            group.bench_function(id("repeats"), |b| {
                b.iter(|| black_box(rows_with_repeats(black_box(&array), &literal, &mut boxes)))
            });
            group.bench_function(id("kernel"), |b| {
                b.iter(|| {
                    black_box(
                        st_predicate_scalar(
                            black_box(&array),
                            &literal,
                            Predicate::Within,
                            Side::Right,
                            &mut scratch,
                        )
                        .unwrap(),
                    )
                })
            });
        }
        group.finish();
    }
}
