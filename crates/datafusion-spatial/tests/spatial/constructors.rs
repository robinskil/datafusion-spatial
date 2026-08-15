//! The functions that build a geometry from ordinates or from parts.

use crate::common::*;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::Array;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn constructors_build_geometries() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;

    assert_eq!(
        scalar_text(&ctx, "SELECT ST_AsText(ST_Point(1.0, 2.0))").await?,
        "POINT(1 2)"
    );
    assert_eq!(
        scalar_text(&ctx, "SELECT ST_AsText(ST_MakePoint(1.0, 2.0))").await?,
        "POINT(1 2)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_MakeLine(ST_MakePoint(0.0, 0.0), ST_MakePoint(1.0, 1.0)))"
        )
        .await?,
        "LINESTRING(0 0,1 1)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_MakePolygon(ST_GeomFromText('LINESTRING(0 0,1 0,1 1,0 0)')))"
        )
        .await?,
        "POLYGON((0 0,1 0,1 1,0 0))"
    );

    // ST_MakeEnvelope yields a box, which reads as a polygon.
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_GeometryType(ST_MakeEnvelope(0.0, 0.0, 1.0, 1.0))"
        )
        .await?,
        "ST_Polygon"
    );
    Ok(())
}

#[tokio::test]
async fn make_point_over_columns() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    // Build a geometry column from two ordinate columns, then read it back.
    let batches = collect(
        &ctx,
        "SELECT ST_X(ST_MakePoint(ST_X(geom), ST_Y(geom))) AS x FROM pts ORDER BY x",
    )
    .await?;
    let x = batches[0].column(0).as_primitive::<Float64Type>();
    assert_eq!(x.value(0), 1.0);
    assert_eq!(x.value(1), 3.0);
    assert!(x.is_null(2), "the null row stays null");
    Ok(())
}
