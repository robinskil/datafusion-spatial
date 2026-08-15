//! `ST_SRID`, `ST_SetSRID` and the EWKB header.

use crate::common::*;
use arrow_array::cast::AsArray;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn srid_round_trips() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;

    assert_eq!(
        scalar_i32(&ctx, "SELECT ST_SRID(ST_MakePoint(1.0, 2.0))").await?,
        0,
        "no CRS reports zero"
    );
    assert_eq!(
        scalar_i32(
            &ctx,
            "SELECT ST_SRID(ST_SetSRID(ST_MakePoint(1.0, 2.0), 4326))"
        )
        .await?,
        4326
    );

    // The values must survive the restamp.
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_SetSRID(ST_MakePoint(1.0, 2.0), 4326))"
        )
        .await?,
        "POINT(1 2)"
    );
    Ok(())
}

/// A per-row SRID cannot be represented, so it must fail at plan time.
#[tokio::test]
async fn set_srid_rejects_a_column() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    let err = collect(
        &ctx,
        "SELECT ST_SetSRID(geom, CAST(ST_NPoints(geom) AS INT)) FROM pts",
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("constant SRID"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// `ST_AsEWKB` must carry the SRID the column was stamped with.
#[tokio::test]
async fn ewkb_header_carries_the_srid() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    let batches = collect(
        &ctx,
        "SELECT ST_AsEWKB(ST_SetSRID(ST_MakePoint(1.0, 2.0), 4326)) AS b, \
                ST_AsBinary(ST_MakePoint(1.0, 2.0)) AS p",
    )
    .await?;

    let extended = batches[0].column(0).as_binary::<i32>().value(0);
    let plain = batches[0].column(1).as_binary::<i32>().value(0);
    assert_eq!(extended.len(), plain.len() + 4, "four bytes of SRID");

    let srid = i32::from_le_bytes(extended[5..9].try_into().unwrap());
    assert_eq!(srid, 4326);
    Ok(())
}
