//! End-to-end tests through a real `SessionContext`.
//!
//! These catch two classes of bug that unit tests over kernels cannot:
//!
//! 1. A wrong signature. It fails only when the SQL planner runs.
//! 2. Lost GeoArrow extension metadata, which turns a geometry back into a plain Arrow value.

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, RecordBatch};
use arrow_schema::Schema;
use datafusion::prelude::SessionContext;
use datafusion_spatial::datafusion;
use geoarrow_array::builder::PointBuilder;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::{CoordType, Dimension, GeoArrowType, PointType};
use std::sync::Arc;

/// A session with the spatial functions and a `cities` table of four points.
///
/// Row 2 is null, so every test also covers a null row.
fn session() -> datafusion::error::Result<SessionContext> {
    let ctx = SessionContext::new();
    datafusion_spatial::register_all(&ctx);

    let point_type =
        PointType::new(Dimension::XY, Default::default()).with_coord_type(CoordType::Separated);

    let p0 = geo::point!(x: 0.5, y: 0.5);
    let p1 = geo::point!(x: 40.0, y: 40.0);
    let p2 = geo::point!(x: 0.25, y: 0.75);
    let array = PointBuilder::from_nullable_points(
        [Some(&p0), Some(&p1), None, Some(&p2)].into_iter(),
        point_type.clone(),
    )
    .finish();

    let field = point_type.to_field("geom", true);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema, vec![array.to_array_ref()])?;
    ctx.register_batch("cities", batch)?;

    Ok(ctx)
}

async fn collect(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<Vec<RecordBatch>> {
    ctx.sql(sql).await?.collect().await
}

#[tokio::test]
async fn st_x_and_st_y_on_a_literal() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let batches = collect(
        &ctx,
        "SELECT ST_X(ST_GeomFromText('POINT(1 2)')) AS x, \
                ST_Y(ST_GeomFromText('POINT(1 2)')) AS y",
    )
    .await?;

    let batch = &batches[0];
    assert_eq!(batch.column(0).as_primitive::<Float64Type>().value(0), 1.0);
    assert_eq!(batch.column(1).as_primitive::<Float64Type>().value(0), 2.0);
    Ok(())
}

#[tokio::test]
async fn st_x_over_a_column() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let batches = collect(&ctx, "SELECT ST_X(geom) AS x FROM cities ORDER BY x").await?;

    let column = batches[0].column(0).as_primitive::<Float64Type>();
    assert_eq!(column.len(), 4);
    assert_eq!(column.value(0), 0.25);
    assert_eq!(column.value(1), 0.5);
    assert_eq!(column.value(2), 40.0);
    assert!(column.is_null(3), "the null row must stay null");
    Ok(())
}

/// The GeoArrow extension metadata must survive the trip through the plan.
///
/// This is what `return_field_from_args` exists for. A `return_type` implementation would drop it.
#[tokio::test]
async fn geometry_output_keeps_its_extension_type() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let df = ctx.sql("SELECT ST_GeomFromText('POINT(1 2)') AS g").await?;

    let schema = df.schema().as_arrow().clone();
    let field = schema.field(0);
    let data_type = GeoArrowType::from_arrow_field(field)
        .expect("the output field must still be a GeoArrow type");
    assert!(matches!(data_type, GeoArrowType::Geometry(_)));

    // The value round trips as a geometry as well.
    let batches = df.collect().await?;
    let array = geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), field)
        .expect("the output array must read back as a GeoArrow array");
    assert_eq!(array.len(), 1);
    Ok(())
}

#[tokio::test]
async fn st_x_rejects_a_non_geometry_argument() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let err = collect(&ctx, "SELECT ST_X(42)").await.unwrap_err();
    assert!(
        err.to_string().contains("not a geometry"),
        "unexpected error: {err}"
    );
    Ok(())
}

#[tokio::test]
async fn st_intersects_with_a_literal() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let batches = collect(
        &ctx,
        "SELECT ST_Intersects(geom, ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 1,0 0))')) AS hit \
         FROM cities",
    )
    .await?;

    let column = batches[0].column(0).as_boolean();
    assert!(column.value(0), "point inside the square");
    assert!(!column.value(1), "point far away");
    assert!(column.is_null(2), "null geometry yields null");
    assert!(column.value(3), "second point inside");
    Ok(())
}

#[tokio::test]
async fn st_intersects_filters_rows() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let batches = collect(
        &ctx,
        "SELECT COUNT(*) AS n FROM cities \
         WHERE ST_Intersects(geom, ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 1,0 0))'))",
    )
    .await?;

    let count = batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0);
    assert_eq!(count, 2);
    Ok(())
}

#[tokio::test]
async fn st_intersects_is_symmetric_in_sql() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let batches = collect(
        &ctx,
        "SELECT ST_Intersects(ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 1,0 0))'), geom) AS hit \
         FROM cities",
    )
    .await?;

    let column = batches[0].column(0).as_boolean();
    assert!(column.value(0));
    assert!(!column.value(1));
    assert!(column.is_null(2));
    Ok(())
}

#[tokio::test]
async fn st_extent_over_a_table() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let batches = collect(&ctx, "SELECT ST_Extent(geom) AS bbox FROM cities").await?;

    let batch = &batches[0];
    let field = batch.schema().field(0).clone();
    let data_type =
        GeoArrowType::from_arrow_field(&field).expect("ST_Extent must return a GeoArrow box type");
    assert!(matches!(data_type, GeoArrowType::Rect(_)));

    // A GeoArrow box is a flat struct of xmin, ymin, xmax, ymax.
    let rect = batch.column(0).as_struct();
    let ordinate = |index: usize| rect.column(index).as_primitive::<Float64Type>().value(0);
    assert_eq!(ordinate(0), 0.25, "xmin");
    assert_eq!(ordinate(1), 0.5, "ymin");
    assert_eq!(ordinate(2), 40.0, "xmax");
    assert_eq!(ordinate(3), 40.0, "ymax");
    Ok(())
}

/// Forces the partial-then-merge path of the accumulator.
#[tokio::test]
async fn st_extent_merges_across_groups() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let batches = collect(
        &ctx,
        "SELECT ST_Extent(geom) AS bbox FROM cities WHERE ST_X(geom) < 1.0",
    )
    .await?;

    let rect = batches[0].column(0).as_struct();
    // xmax is the third field of the flat box struct.
    assert_eq!(rect.column(2).as_primitive::<Float64Type>().value(0), 0.5);
    Ok(())
}

#[tokio::test]
async fn st_extent_of_all_nulls_is_null() -> datafusion::error::Result<()> {
    let ctx = session()?;
    let batches = collect(
        &ctx,
        "SELECT ST_Extent(geom) AS bbox FROM cities WHERE geom IS NULL",
    )
    .await?;

    assert!(
        batches[0].column(0).is_null(0),
        "an all-null input must give NULL"
    );
    Ok(())
}
