//! Throughput of the functions that walk a column and build one `geo` geometry per row.
//!
//! `ST_Collect` keeps every geometry it builds, so it cannot reuse one buffer. It still gains
//! from the fast fill in `src/materialize.rs`. That fill reads the coordinates of a row as a
//! plain slice. It does not match `CoordBuffer` once per coordinate.
//!
//! The two cluster functions keep only a centroid, so they take the full path: one geometry
//! serves every row.
//!
//! `ST_MemUnion` takes the same change, but a benchmark of it measures `geo::BooleanOps` rather
//! than the fill, so it is not here.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_spatial_kernels::aggregate::Collect;
use datafusion_spatial_kernels::cluster::st_cluster_kmeans;
use datafusion_spatial_kernels::measure::{st_area, st_perimeter};
use datafusion_spatial_kernels::process::{sized_shape, st_is_valid, Sized as SizedShape};
use geoarrow_array::array::PolygonArray;
use geoarrow_array::builder::PolygonBuilder;
use geoarrow_schema::{CoordType, Dimension, PolygonType};

mod common;
use common::{Lcg, BATCH};

/// A regular polygon of `vertices` sides, centred on the given point.
fn ring(vertices: usize, cx: f64, cy: f64) -> geo::Polygon<f64> {
    let mut coords: Vec<geo::Coord<f64>> = (0..vertices)
        .map(|i| {
            let angle = (i as f64) / (vertices as f64) * std::f64::consts::TAU;
            geo::coord! { x: cx + angle.cos(), y: cy + angle.sin() }
        })
        .collect();
    coords.push(coords[0]);
    geo::Polygon::new(geo::LineString::new(coords), vec![])
}

/// `BATCH` polygons scattered over a wide square, so no two centroids match.
fn polygon_column(vertices: usize) -> PolygonArray {
    let mut rng = Lcg::new(0x5EED);
    let shapes: Vec<geo::Polygon<f64>> = (0..BATCH)
        .map(|_| {
            ring(
                vertices,
                (rng.next_f64() - 0.5) * 1000.0,
                (rng.next_f64() - 0.5) * 1000.0,
            )
        })
        .collect();
    PolygonBuilder::from_polygons(
        &shapes,
        PolygonType::new(Dimension::XY, Default::default()).with_coord_type(CoordType::Separated),
    )
    .finish()
}

fn bench_collect(c: &mut Criterion) {
    let mut group = c.benchmark_group("ST_Collect");
    group.throughput(criterion::Throughput::Elements(BATCH as u64));

    for vertices in [5usize, 256] {
        let array = polygon_column(vertices);
        group.bench_function(format!("{vertices}v"), |b| {
            b.iter(|| {
                let mut collect = Collect::new();
                collect.update(black_box(&array)).unwrap();
                black_box(collect.finish().is_some())
            })
        });
    }

    group.finish();
}

fn bench_cluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("ST_ClusterKMeans");
    group.throughput(criterion::Throughput::Elements(BATCH as u64));

    for vertices in [5usize, 256] {
        let array = polygon_column(vertices);
        group.bench_function(format!("{vertices}v"), |b| {
            b.iter(|| black_box(st_cluster_kmeans(black_box(&array), 8).unwrap()))
        });
    }

    group.finish();
}

/// The unary kernels, which build one geometry per row and keep none of them.
fn bench_unary(c: &mut Criterion) {
    let mut group = c.benchmark_group("unary");
    group.throughput(criterion::Throughput::Elements(BATCH as u64));

    for vertices in [5usize, 256] {
        let array = polygon_column(vertices);
        group.bench_function(format!("ST_Area/{vertices}v"), |b| {
            b.iter(|| black_box(st_area(black_box(&array)).unwrap()))
        });
        group.bench_function(format!("ST_Perimeter/{vertices}v"), |b| {
            b.iter(|| black_box(st_perimeter(black_box(&array)).unwrap()))
        });
        group.bench_function(format!("ST_IsValid/{vertices}v"), |b| {
            b.iter(|| black_box(st_is_valid(black_box(&array)).unwrap()))
        });
        let tolerance = arrow_array::Float64Array::from(vec![0.01]);
        let output = geoarrow_schema::GeometryType::new(Default::default());
        group.bench_function(format!("ST_Simplify/{vertices}v"), |b| {
            b.iter(|| {
                black_box(
                    sized_shape(
                        black_box(&array),
                        SizedShape::Simplify,
                        &tolerance,
                        output.clone(),
                    )
                    .unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_collect, bench_cluster, bench_unary);
criterion_main!(benches);
