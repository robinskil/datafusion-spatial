//! `ST_ClosestPoint`, `ST_ShortestLine` and the two interpolation functions.

use crate::common::*;
use arrow_array::Array;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn linear_referencing() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let line = "ST_GeomFromText('LINESTRING(0 0,10 0)')";

    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_AsText(ST_ClosestPoint({line}, ST_GeomFromText('POINT(5 3)')))")
        )
        .await?,
        "POINT(5 0)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_AsText(ST_ShortestLine({line}, ST_GeomFromText('POINT(5 3)')))")
        )
        .await?,
        "LINESTRING(5 0,5 3)"
    );
    assert_eq!(
        scalar_f64(
            &ctx,
            &format!("SELECT ST_LineLocatePoint({line}, ST_GeomFromText('POINT(2.5 0)'))")
        )
        .await?,
        0.25
    );
    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_AsText(ST_LineInterpolatePoint({line}, 0.5))")
        )
        .await?,
        "POINT(5 0)"
    );
    Ok(())
}

#[tokio::test]
async fn line_functions_are_null_for_other_types() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let batches = collect(
        &ctx,
        &format!(
            "SELECT ST_LineLocatePoint({UNIT_SQUARE}, ST_GeomFromText('POINT(0.5 0.5)')) AS f"
        ),
    )
    .await?;
    assert!(batches[0].column(0).is_null(0));
    Ok(())
}
