//! Wall-clock comparison of the nested loop join against the spatial join.
use datafusion_spatial::datafusion;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use geoarrow_array::builder::{PointBuilder, PolygonBuilder};
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::{Dimension, PointType, PolygonType};

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn pts(n: usize) -> RecordBatch {
    let mut r = Lcg(0x5EED);
    let v: Vec<geo::Point<f64>> = (0..n)
        .map(|_| geo::point! { x: r.f()*1000.0, y: r.f()*1000.0 })
        .collect();
    let t = PointType::new(Dimension::XY, Default::default());
    let a = PointBuilder::from_points(v.iter(), t.clone()).finish();
    let s = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        t.to_field("geom", true),
    ]));
    RecordBatch::try_new(
        s,
        vec![
            Arc::new(arrow_array::Int32Array::from(
                (0..n as i32).collect::<Vec<_>>(),
            )),
            a.to_array_ref(),
        ],
    )
    .unwrap()
}

fn polys(n: usize) -> RecordBatch {
    let mut r = Lcg(0xC0FFEE);
    let v: Vec<geo::Polygon<f64>> = (0..n)
        .map(|_| {
            let (x, y) = (r.f() * 1000.0, r.f() * 1000.0);
            geo::Polygon::new(
                geo::LineString::new(vec![
                    geo::coord! {x:x,y:y},
                    geo::coord! {x:x+5.0,y:y},
                    geo::coord! {x:x+5.0,y:y+5.0},
                    geo::coord! {x:x,y:y+5.0},
                    geo::coord! {x:x,y:y},
                ]),
                vec![],
            )
        })
        .collect();
    let t = PolygonType::new(Dimension::XY, Default::default());
    let a = PolygonBuilder::from_polygons(&v, t.clone()).finish();
    let s = Arc::new(Schema::new(vec![
        Field::new("box_id", DataType::Int32, false),
        t.to_field("shape", true),
    ]));
    RecordBatch::try_new(
        s,
        vec![
            Arc::new(arrow_array::Int32Array::from(
                (0..n as i32).collect::<Vec<_>>(),
            )),
            a.to_array_ref(),
        ],
    )
    .unwrap()
}

async fn run(spatial: bool, n: usize, m: usize) -> (u128, usize) {
    let ctx = if spatial {
        SessionContext::new_with_state(
            SessionStateBuilder::new()
                .with_default_features()
                .with_physical_optimizer_rule(datafusion_spatial::join::spatial_join_rule())
                .build(),
        )
    } else {
        SessionContext::new()
    };
    datafusion_spatial::register_all(&ctx);
    ctx.register_batch("shapes", polys(n)).unwrap();
    ctx.register_batch("pts", pts(m)).unwrap();
    let sql = "SELECT s.box_id, p.id FROM shapes s JOIN pts p ON ST_Intersects(s.shape, p.geom)";
    let start = Instant::now();
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    (start.elapsed().as_micros(), rows)
}

#[tokio::main]
async fn main() {
    println!(
        "{:>8} {:>8} {:>13} {:>13} {:>8} {:>10}",
        "build", "probe", "nested(us)", "spatial(us)", "gain", "rows"
    );
    for (n, m) in [
        (2000usize, 2000usize),
        (5000, 5000),
        (10000, 10000),
        (20000, 20000),
    ] {
        let (slow, r1) = run(false, n, m).await;
        let (fast, r2) = run(true, n, m).await;
        assert_eq!(r1, r2, "row counts must match at {n}x{m}");
        let gain = if fast == 0 {
            f64::INFINITY
        } else {
            slow as f64 / fast as f64
        };
        println!("{n:>8} {m:>8} {slow:>13} {fast:>13} {gain:>7.1}x {r1:>10}");
    }
}
