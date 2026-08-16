//! End-to-end SQL tests for `ST_Transform`.
//!
//! Only compiled with the `proj` feature. Run them with:
//!
//! ```bash
//! cargo test -p datafusion-spatial --features proj
//! ```
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use datafusion::prelude::SessionContext;
use datafusion_spatial::datafusion;
use geoarrow_array::builder::PointBuilder;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::{Dimension, GeoArrowType, Metadata, PointType};
use std::sync::Arc;

/// A session with a `cities` table stamped as EPSG:4326.
fn session() -> datafusion::error::Result<SessionContext> {
    let ctx = SessionContext::new();
    datafusion_spatial::register_all(&ctx);

    // Longitude and latitude, so the column needs a source SRID to be transformable.
    let wgs84 = Arc::new(Metadata::new(
        geoarrow_schema::Crs::from_srid("4326".to_string()),
        None,
    ));
    let point_type = PointType::new(Dimension::XY, wgs84);

    let values: Vec<geo::Point<f64>> = vec![
        geo::point!(x: -0.1278, y: 51.5074),
        geo::point!(x: 2.3522, y: 48.8566),
    ];
    let points = PointBuilder::from_points(values.iter(), point_type.clone()).finish();

    let schema = Arc::new(Schema::new(vec![point_type.to_field("geom", true)]));
    ctx.register_batch(
        "cities",
        RecordBatch::try_new(schema, vec![points.to_array_ref()])?,
    )?;
    Ok(ctx)
}

async fn collect(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<Vec<RecordBatch>> {
    ctx.sql(sql).await?.collect().await
}

async fn scalar_f64(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<f64> {
    let batches = collect(ctx, sql).await?;
    Ok(batches[0].column(0).as_primitive::<Float64Type>().value(0))
}

async fn scalar_i32(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<i32> {
    let batches = collect(ctx, sql).await?;
    Ok(batches[0].column(0).as_primitive::<Int32Type>().value(0))
}

/// London in Web Mercator is roughly (-14 200, 6 711 000) metres.
#[tokio::test]
async fn transform_to_web_mercator() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let x = scalar_f64(&ctx, "SELECT ST_X(ST_Transform(geom, 3857)) FROM cities").await?;
    let y = scalar_f64(&ctx, "SELECT ST_Y(ST_Transform(geom, 3857)) FROM cities").await?;

    assert!((-14_300.0..-14_100.0).contains(&x), "x was {x}");
    assert!((6_710_000.0..6_712_000.0).contains(&y), "y was {y}");
    Ok(())
}

/// The output column carries the target SRID, so a later `ST_SRID` reads the new one.
#[tokio::test]
async fn the_output_is_restamped() -> datafusion::error::Result<()> {
    let ctx = session()?;
    assert_eq!(
        scalar_i32(&ctx, "SELECT ST_SRID(geom) FROM cities").await?,
        4326
    );
    assert_eq!(
        scalar_i32(&ctx, "SELECT ST_SRID(ST_Transform(geom, 3857)) FROM cities").await?,
        3857
    );
    Ok(())
}

#[tokio::test]
async fn a_round_trip_returns_the_input() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let before = scalar_f64(&ctx, "SELECT ST_X(geom) FROM cities").await?;
    let after = scalar_f64(
        &ctx,
        "SELECT ST_X(ST_Transform(ST_Transform(geom, 3857), 4326)) FROM cities",
    )
    .await?;
    assert!(
        (after - before).abs() < 1e-9,
        "drifted from {before} to {after}"
    );
    Ok(())
}

/// A reprojection changes the answer of a planar measurement. That is the point of it.
#[tokio::test]
async fn distance_in_metres_after_transform() -> datafusion::error::Result<()> {
    let ctx = session()?;

    // In degrees, London to Paris is a small number.
    let degrees = scalar_f64(
        &ctx,
        "SELECT ST_Distance(a.geom, b.geom) FROM cities a, cities b \
         WHERE ST_X(a.geom) < 0 AND ST_X(b.geom) > 0",
    )
    .await?;
    assert!(degrees < 10.0, "degrees was {degrees}");

    // In Web Mercator it is metres, and Mercator inflates at this latitude.
    let metres = scalar_f64(
        &ctx,
        "SELECT ST_Distance(ST_Transform(a.geom, 3857), ST_Transform(b.geom, 3857)) \
         FROM cities a, cities b WHERE ST_X(a.geom) < 0 AND ST_X(b.geom) > 0",
    )
    .await?;
    assert!(metres > 300_000.0, "metres was {metres}");
    Ok(())
}

/// A per-row SRID cannot be represented, so it fails at plan time.
#[tokio::test]
async fn a_column_srid_is_rejected() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let err = collect(&ctx, "SELECT ST_Transform(geom, ST_SRID(geom)) FROM cities")
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("constant target SRID"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// Without a source SRID there is nothing to transform from, and that is a plan-time error.
#[tokio::test]
async fn an_unstamped_column_is_rejected_at_plan_time() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let err = collect(
        &ctx,
        "SELECT ST_Transform(ST_GeomFromText('POINT(0 0)'), 3857)",
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("ST_SetSRID"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// An SRID PROJ does not know is a plan-time error too, not a per-row failure.
#[tokio::test]
async fn an_unknown_srid_is_rejected_at_plan_time() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let err = collect(&ctx, "SELECT ST_Transform(geom, 999999) FROM cities")
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("PROJ cannot transform"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// `ST_SetSRID` then `ST_Transform` is the documented route for an unstamped column.
#[tokio::test]
async fn set_srid_then_transform() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let x = scalar_f64(
        &ctx,
        "SELECT ST_X(ST_Transform(ST_SetSRID(ST_GeomFromText('POINT(-0.1278 51.5074)'), 4326), 3857))",
    )
    .await?;
    assert!((-14_300.0..-14_100.0).contains(&x), "x was {x}");
    Ok(())
}

#[tokio::test]
async fn the_output_keeps_its_extension_type() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let df = ctx
        .sql("SELECT ST_Transform(geom, 3857) AS g FROM cities")
        .await?;
    let schema = df.schema().as_arrow().clone();
    let field = schema.field(0);
    let data_type = GeoArrowType::from_arrow_field(field).expect("must stay a GeoArrow type");
    assert!(
        matches!(data_type, GeoArrowType::Point(_)),
        "a point column must stay a point column, got {data_type:?}"
    );

    let batches = df.collect().await?;
    geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), field)
        .expect("must read back as a GeoArrow array");
    Ok(())
}

/// `ST_Transform` is registered only when the feature is on.
#[tokio::test]
async fn transform_is_registered() -> datafusion::error::Result<()> {
    let functions = datafusion_spatial::scalar_udfs();
    let names: Vec<&str> = functions.iter().map(|f| f.name()).collect();
    assert!(
        names.contains(&"st_transform"),
        "ST_Transform must be registered with the proj feature"
    );
    Ok(())
}
