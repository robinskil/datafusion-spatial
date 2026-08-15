//! `ST_Azimuth` and `ST_Project`.

use crate::common::*;
use arrow_array::Array;
use datafusion_spatial::datafusion;

/// Due north is zero radians, and a quarter turn clockwise is due east.
#[tokio::test]
async fn azimuth_reads_clockwise_from_north() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let origin = "ST_GeomFromText('POINT(0 0)')";

    let north = scalar_f64(
        &ctx,
        &format!("SELECT ST_Azimuth({origin}, ST_GeomFromText('POINT(0 1)'))"),
    )
    .await?;
    assert!(north.abs() < 1e-6, "north was {north}");

    let east = scalar_f64(
        &ctx,
        &format!("SELECT ST_Azimuth({origin}, ST_GeomFromText('POINT(1 0)'))"),
    )
    .await?;
    assert!(
        (east - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
        "east was {east}"
    );

    // A point has no bearing to itself.
    let batches = collect(&ctx, &format!("SELECT ST_Azimuth({origin}, {origin}) AS a")).await?;
    assert!(batches[0].column(0).is_null(0));
    Ok(())
}

/// `ST_Project` is the inverse of `ST_Azimuth`, so a round trip returns the bearing.
#[tokio::test]
async fn project_moves_along_a_bearing() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    // 1000 metres due east of the origin.
    let moved = scalar_text(
        &ctx,
        "SELECT ST_AsText(ST_Project(ST_GeomFromText('POINT(0 0)'), 1000, 1.5707963267948966))",
    )
    .await?;
    assert!(moved.starts_with("POINT("), "got {moved}");

    // The bearing back out matches the one that went in.
    let bearing = scalar_f64(
        &ctx,
        "SELECT ST_Azimuth(ST_GeomFromText('POINT(0 0)'), \
           ST_Project(ST_GeomFromText('POINT(0 0)'), 1000, 1.5707963267948966))",
    )
    .await?;
    assert!(
        (bearing - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
        "bearing was {bearing}"
    );
    Ok(())
}
