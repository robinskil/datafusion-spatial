//! The spatial join must agree with the nested loop join it replaces, row for row.
//!
//! Every test here runs the same SQL twice: once on a plain session, and once on a session with
//! `spatial_join_rule` installed. The two answers must match. A faster wrong answer is worthless.

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_spatial::datafusion;
use geoarrow_array::builder::{PointBuilder, PolygonBuilder};
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::{Dimension, PointType, PolygonType};
use std::sync::Arc;

/// A repeatable pseudo random generator, so both sessions see identical data.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn points_table(rows: usize, spread: f64) -> datafusion::error::Result<RecordBatch> {
    let mut rng = Lcg(0x5EED);
    let values: Vec<geo::Point<f64>> = (0..rows)
        .map(|_| geo::point! { x: rng.next_f64() * spread, y: rng.next_f64() * spread })
        .collect();
    let ids: Vec<i32> = (0..rows as i32).collect();

    let point_type = PointType::new(Dimension::XY, Default::default());
    let points = PointBuilder::from_points(values.iter(), point_type.clone()).finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        point_type.to_field("geom", true),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int32Array::from(ids)),
            points.to_array_ref(),
        ],
    )?)
}

fn squares_table(rows: usize, spread: f64, size: f64) -> datafusion::error::Result<RecordBatch> {
    let mut rng = Lcg(0xC0FFEE);
    let shapes: Vec<geo::Polygon<f64>> = (0..rows)
        .map(|_| {
            let (x, y) = (rng.next_f64() * spread, rng.next_f64() * spread);
            geo::Polygon::new(
                geo::LineString::new(vec![
                    geo::coord! { x: x, y: y },
                    geo::coord! { x: x + size, y: y },
                    geo::coord! { x: x + size, y: y + size },
                    geo::coord! { x: x, y: y + size },
                    geo::coord! { x: x, y: y },
                ]),
                vec![],
            )
        })
        .collect();
    let ids: Vec<i32> = (0..rows as i32).collect();

    let polygon_type = PolygonType::new(Dimension::XY, Default::default());
    let polygons = PolygonBuilder::from_polygons(&shapes, polygon_type.clone()).finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("box_id", DataType::Int32, false),
        polygon_type.to_field("shape", true),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int32Array::from(ids)),
            polygons.to_array_ref(),
        ],
    )?)
}

/// A session with the tables registered. `spatial` installs the join rule.
fn session(spatial: bool) -> datafusion::error::Result<SessionContext> {
    let ctx = if spatial {
        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_physical_optimizer_rule(datafusion_spatial::join::spatial_join_rule())
            .build();
        SessionContext::new_with_state(state)
    } else {
        SessionContext::new()
    };
    datafusion_spatial::register_all(&ctx);
    ctx.register_batch("shapes", squares_table(200, 100.0, 6.0)?)?;
    ctx.register_batch("pts", points_table(400, 100.0)?)?;
    Ok(ctx)
}

/// Every matched pair, sorted, so the two plans can be compared directly.
async fn pairs(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<Vec<(i32, i32)>> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;

    let batches = ctx.sql(sql).await?.collect().await?;
    let mut out = Vec::new();
    for batch in batches {
        let left = batch.column(0).as_primitive::<Int32Type>();
        let right = batch.column(1).as_primitive::<Int32Type>();
        for row in 0..batch.num_rows() {
            out.push((left.value(row), right.value(row)));
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// The plan text, so a test can prove the rule fired or did not.
async fn plan_of(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<String> {
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    Ok(format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(false)
    ))
}

const INTERSECTS: &str = "SELECT s.box_id, p.id FROM shapes s JOIN pts p \
                          ON ST_Intersects(s.shape, p.geom)";

#[tokio::test]
async fn the_rule_fires_on_st_intersects() -> datafusion::error::Result<()> {
    let plan = plan_of(&session(true)?, INTERSECTS).await?;
    assert!(
        plan.contains("SpatialJoinExec"),
        "the rule must fire, plan was:\n{plan}"
    );

    let plain = plan_of(&session(false)?, INTERSECTS).await?;
    assert!(
        !plain.contains("SpatialJoinExec"),
        "a plain session must not have the operator"
    );
    Ok(())
}

#[tokio::test]
async fn st_intersects_agrees_with_the_nested_loop() -> datafusion::error::Result<()> {
    let expected = pairs(&session(false)?, INTERSECTS).await?;
    let actual = pairs(&session(true)?, INTERSECTS).await?;
    assert!(!expected.is_empty(), "the fixture must produce matches");
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn st_contains_agrees_and_keeps_its_direction() -> datafusion::error::Result<()> {
    // Not symmetric. A swapped argument order would still compile and would be wrong.
    let sql = "SELECT s.box_id, p.id FROM shapes s JOIN pts p ON ST_Contains(s.shape, p.geom)";
    let expected = pairs(&session(false)?, sql).await?;
    let actual = pairs(&session(true)?, sql).await?;
    assert!(!expected.is_empty(), "the fixture must produce matches");
    assert_eq!(actual, expected);

    let plan = plan_of(&session(true)?, sql).await?;
    assert!(plan.contains("SpatialJoinExec"), "plan was:\n{plan}");
    Ok(())
}

#[tokio::test]
async fn st_within_agrees() -> datafusion::error::Result<()> {
    let sql = "SELECT s.box_id, p.id FROM shapes s JOIN pts p ON ST_Within(p.geom, s.shape)";
    let expected = pairs(&session(false)?, sql).await?;
    let actual = pairs(&session(true)?, sql).await?;
    assert_eq!(actual, expected);
    Ok(())
}

/// Two separate boxes prove `ST_Disjoint` true, so a grid would lose those pairs. The rule must
/// leave this plan alone.
#[tokio::test]
async fn st_disjoint_keeps_the_nested_loop() -> datafusion::error::Result<()> {
    let sql = "SELECT s.box_id, p.id FROM shapes s JOIN pts p ON ST_Disjoint(s.shape, p.geom)";
    let plan = plan_of(&session(true)?, sql).await?;
    assert!(
        !plan.contains("SpatialJoinExec"),
        "ST_Disjoint must not be rewritten, plan was:\n{plan}"
    );

    let expected = pairs(&session(false)?, sql).await?;
    let actual = pairs(&session(true)?, sql).await?;
    assert_eq!(actual, expected);
    Ok(())
}

/// A left join is a different row set. The rule handles inner joins only.
#[tokio::test]
async fn a_left_join_is_left_alone() -> datafusion::error::Result<()> {
    let sql = "SELECT s.box_id, p.id FROM shapes s LEFT JOIN pts p \
               ON ST_Intersects(s.shape, p.geom)";
    let plan = plan_of(&session(true)?, sql).await?;
    assert!(
        !plan.contains("SpatialJoinExec"),
        "a left join must keep its plan, plan was:\n{plan}"
    );
    Ok(())
}

/// DataFusion pushes a term that touches one side below the join. The join filter then holds the
/// predicate alone, and the rule still fires. The answer must not change.
#[tokio::test]
async fn a_pushable_extra_term_still_rewrites() -> datafusion::error::Result<()> {
    let sql = "SELECT s.box_id, p.id FROM shapes s JOIN pts p \
               ON ST_Intersects(s.shape, p.geom) AND p.id > 10";
    let plan = plan_of(&session(true)?, sql).await?;
    assert!(
        plan.contains("SpatialJoinExec"),
        "DataFusion pushes the id term down, so the rule should fire, plan was:\n{plan}"
    );

    let expected = pairs(&session(false)?, sql).await?;
    let actual = pairs(&session(true)?, sql).await?;
    assert!(!expected.is_empty(), "the fixture must produce matches");
    assert_eq!(actual, expected);
    Ok(())
}

/// A term that touches both sides cannot be pushed down. It stays in the join filter beside the
/// predicate, so the filter is an `AND` and the rule must decline it.
#[tokio::test]
async fn an_unpushable_extra_term_is_left_alone() -> datafusion::error::Result<()> {
    let sql = "SELECT s.box_id, p.id FROM shapes s JOIN pts p \
               ON ST_Intersects(s.shape, p.geom) AND s.box_id > p.id";
    let plan = plan_of(&session(true)?, sql).await?;
    assert!(
        !plan.contains("SpatialJoinExec"),
        "a compound join filter must keep its plan, plan was:\n{plan}"
    );

    let expected = pairs(&session(false)?, sql).await?;
    let actual = pairs(&session(true)?, sql).await?;
    assert!(!expected.is_empty(), "the fixture must produce matches");
    assert_eq!(actual, expected);
    Ok(())
}

/// A build side that matches nothing must still produce a valid empty result.
#[tokio::test]
async fn an_empty_match_set_is_handled() -> datafusion::error::Result<()> {
    let ctx = session(true)?;
    ctx.register_batch("far", points_table(10, 1.0)?)?;
    let sql = "SELECT s.box_id, f.id FROM shapes s JOIN far f \
               ON ST_Intersects(s.shape, f.geom) WHERE s.box_id < 0";
    let batches = ctx.sql(sql).await?.collect().await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 0);
    Ok(())
}

/// Nulls take no part in a join, in either plan.
#[tokio::test]
async fn nulls_agree() -> datafusion::error::Result<()> {
    let sql = "SELECT s.box_id, p.id FROM shapes s JOIN pts p \
               ON ST_Intersects(s.shape, ST_GeomFromText(NULL))";
    let expected = pairs(&session(false)?, sql).await?;
    let actual = pairs(&session(true)?, sql).await?;
    assert_eq!(actual, expected);
    assert!(actual.is_empty(), "a null argument matches nothing");
    Ok(())
}
