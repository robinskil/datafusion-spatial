//! The functions that pull one part out of a geometry.

use crate::common::*;
use arrow_array::Array;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn component_extraction() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;

    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_StartPoint(ST_GeomFromText('LINESTRING(1 1,2 2,3 3)')))"
        )
        .await?,
        "POINT(1 1)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_EndPoint(ST_GeomFromText('LINESTRING(1 1,2 2,3 3)')))"
        )
        .await?,
        "POINT(3 3)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_PointN(ST_GeomFromText('LINESTRING(1 1,2 2,3 3)'), 2))"
        )
        .await?,
        "POINT(2 2)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_ExteriorRing(\
               ST_GeomFromText('POLYGON((0 0,4 0,4 4,0 4,0 0),(1 1,2 1,2 2,1 1))')))"
        )
        .await?,
        "LINESTRING(0 0,4 0,4 4,0 4,0 0)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_InteriorRingN(\
               ST_GeomFromText('POLYGON((0 0,4 0,4 4,0 4,0 0),(1 1,2 1,2 2,1 1))'), 1))"
        )
        .await?,
        "LINESTRING(1 1,2 1,2 2,1 1)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_GeometryN(ST_GeomFromText('MULTIPOINT(1 1,2 2,3 3)'), 2))"
        )
        .await?,
        "POINT(2 2)"
    );

    // An out of range index gives null, not an error.
    let batches = collect(
        &ctx,
        "SELECT ST_PointN(ST_GeomFromText('LINESTRING(1 1,2 2)'), 9) AS p",
    )
    .await?;
    assert!(batches[0].column(0).is_null(0));
    Ok(())
}
