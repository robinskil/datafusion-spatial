//! `ST_X` throughput.
//!
//! Separated coordinates take the zero copy path, so the time should stay flat as the batch grows.
//! Interleaved coordinates need a strided read, which is the honest cost of that layout.

use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_spatial_kernels::accessor::st_x;
use geoarrow_array::cast::to_wkb;
use geoarrow_schema::CoordType;
use std::hint::black_box;

mod common;
use common::{points, BATCH};

fn bench_st_x(c: &mut Criterion) {
    let mut group = c.benchmark_group("ST_X");
    group.throughput(criterion::Throughput::Elements(BATCH as u64));

    let separated = points(BATCH, 1000.0, CoordType::Separated);
    group.bench_function("separated", |b| {
        b.iter(|| black_box(st_x(black_box(&separated)).unwrap()))
    });

    let interleaved = points(BATCH, 1000.0, CoordType::Interleaved);
    group.bench_function("interleaved", |b| {
        b.iter(|| black_box(st_x(black_box(&interleaved)).unwrap()))
    });

    let wkb = to_wkb::<i32>(&separated).unwrap();
    group.bench_function("wkb", |b| {
        b.iter(|| black_box(st_x(black_box(&wkb)).unwrap()))
    });

    group.finish();
}

criterion_group!(benches, bench_st_x);
criterion_main!(benches);
