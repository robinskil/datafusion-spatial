//! The text and binary formats: WKT, WKB, GeoJSON and geohash.

use crate::common::*;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn text_and_binary_round_trip() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;

    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 0))'))"
        )
        .await?,
        "POLYGON((0 0,1 0,1 1,0 0))"
    );

    // Binary out, binary in, text out. The value must survive the loop.
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_GeomFromWKB(ST_AsBinary(ST_MakePoint(7.0, 8.0))))"
        )
        .await?,
        "POINT(7 8)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_GeomFromEWKB(ST_AsEWKB(ST_MakePoint(7.0, 8.0))))"
        )
        .await?,
        "POINT(7 8)"
    );
    Ok(())
}

#[tokio::test]
async fn geojson_round_trip() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    let json = scalar_text(&ctx, "SELECT ST_AsGeoJSON(ST_MakePoint(1.0, 2.0))").await?;
    assert!(json.contains("\"Point\""), "got {json}");

    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_GeomFromGeoJSON('{\"type\":\"Point\",\"coordinates\":[1,2]}'))"
        )
        .await?,
        "POINT(1 2)"
    );
    Ok(())
}

#[tokio::test]
async fn geohash_round_trip() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    let hash = scalar_text(&ctx, "SELECT ST_GeoHash(ST_MakePoint(1.5, 2.5), 9)").await?;
    assert_eq!(hash.len(), 9);

    let x = scalar_f64(&ctx, &format!("SELECT ST_X(ST_PointFromGeoHash('{hash}'))")).await?;
    assert!((x - 1.5).abs() < 1e-3, "x was {x}");
    Ok(())
}
