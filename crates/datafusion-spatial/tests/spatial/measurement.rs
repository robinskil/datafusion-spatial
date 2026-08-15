//! Area, length, perimeter and the distance functions.

use crate::common::*;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::Array;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn area_length_and_perimeter() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let rect = "ST_GeomFromText('POLYGON((0 0,4 0,4 3,0 3,0 0))')";

    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_Area({rect})")).await?,
        12.0
    );
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_Perimeter({rect})")).await?,
        14.0
    );
    assert_eq!(
        scalar_f64(
            &ctx,
            "SELECT ST_Length(ST_GeomFromText('LINESTRING(0 0,3 4)'))"
        )
        .await?,
        5.0
    );
    // PostGIS reports zero length for a polygon and zero area for a line.
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_Length({rect})")).await?,
        0.0
    );
    Ok(())
}

#[tokio::test]
async fn distances() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let origin = "ST_GeomFromText('POINT(0 0)')";
    let far = "ST_GeomFromText('POINT(3 4)')";

    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_Distance({origin}, {far})")).await?,
        5.0
    );
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_MaxDistance({origin}, {far})")).await?,
        5.0
    );

    let hausdorff = scalar_f64(
        &ctx,
        "SELECT ST_HausdorffDistance(ST_GeomFromText('LINESTRING(0 0,2 0)'), \
                                     ST_GeomFromText('LINESTRING(0 1,2 1)'))",
    )
    .await?;
    assert_eq!(hausdorff, 1.0);

    let frechet = scalar_f64(
        &ctx,
        "SELECT ST_FrechetDistance(ST_GeomFromText('LINESTRING(0 0,2 0)'), \
                                   ST_GeomFromText('LINESTRING(0 1,2 1)'))",
    )
    .await?;
    assert_eq!(frechet, 1.0);
    Ok(())
}

/// The spherical functions answer in metres, unlike the planar ones.
#[tokio::test]
async fn spherical_distances() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let london = "ST_GeomFromText('POINT(-0.1278 51.5074)')";
    let paris = "ST_GeomFromText('POINT(2.3522 48.8566)')";

    let sphere = scalar_f64(
        &ctx,
        &format!("SELECT ST_DistanceSphere({london}, {paris})"),
    )
    .await?;
    let spheroid = scalar_f64(
        &ctx,
        &format!("SELECT ST_DistanceSpheroid({london}, {paris})"),
    )
    .await?;
    let planar = scalar_f64(&ctx, &format!("SELECT ST_Distance({london}, {paris})")).await?;

    assert!((330_000.0..350_000.0).contains(&sphere), "sphere: {sphere}");
    assert!(
        (330_000.0..350_000.0).contains(&spheroid),
        "spheroid: {spheroid}"
    );
    assert!(planar < 10.0, "the planar answer is in degrees: {planar}");
    Ok(())
}

#[tokio::test]
async fn measures_over_a_column() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let batches = collect(
        &ctx,
        "SELECT ST_Distance(geom, ST_GeomFromText('POINT(0 0)')) AS d FROM pts ORDER BY d",
    )
    .await?;
    let distances = batches[0].column(0).as_primitive::<Float64Type>();
    assert!(distances.value(0) < distances.value(1));
    assert!(
        distances.is_null(3),
        "the null row sorts last and stays null"
    );
    Ok(())
}
