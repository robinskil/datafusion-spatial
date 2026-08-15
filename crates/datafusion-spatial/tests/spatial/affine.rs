//! `ST_Translate`, `ST_Scale`, `ST_Rotate` and `ST_Affine`.

use crate::common::*;
use datafusion_spatial::datafusion;
use geoarrow_schema::GeoArrowType;

#[tokio::test]
async fn affine_transforms() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    let point = "ST_GeomFromText('POINT(1 2)')";

    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_AsText(ST_Translate({point}, 10, 20))")
        )
        .await?,
        "POINT(11 22)"
    );
    assert_eq!(
        scalar_text(&ctx, &format!("SELECT ST_AsText(ST_Scale({point}, 2, 3))")).await?,
        "POINT(2 6)"
    );
    // A quarter turn in radians, as PostGIS takes it.
    let turned = scalar_text(
        &ctx,
        "SELECT ST_AsText(ST_Rotate(ST_GeomFromText('POINT(1 0)'), 1.5707963267948966))",
    )
    .await?;
    assert!(turned.starts_with("POINT("), "got {turned}");

    // The raw matrix form, here an identity plus an offset.
    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_AsText(ST_Affine({point}, 1, 0, 0, 1, 10, 20))")
        )
        .await?,
        "POINT(11 22)"
    );
    Ok(())
}

/// An affine transform keeps the geometry type, unlike the process functions.
#[tokio::test]
async fn affine_preserves_the_geometry_type() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    let df = ctx
        .sql("SELECT ST_Translate(geom, 1, 1) AS g FROM shapes")
        .await?;
    let schema = df.schema().as_arrow().clone();
    let data_type = GeoArrowType::from_arrow_field(schema.field(0)).unwrap();
    assert!(
        matches!(data_type, GeoArrowType::Polygon(_)),
        "a polygon column must stay a polygon column, got {data_type:?}"
    );
    Ok(())
}

#[tokio::test]
async fn reverse_and_orientation() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_Reverse(ST_GeomFromText('LINESTRING(0 0,1 1,2 2)')))"
        )
        .await?,
        "LINESTRING(2 2,1 1,0 0)"
    );

    // A second force in the same direction changes nothing.
    let once = scalar_text(
        &ctx,
        &format!("SELECT ST_AsText(ST_ForcePolygonCCW({UNIT_SQUARE}))"),
    )
    .await?;
    let twice = scalar_text(
        &ctx,
        &format!("SELECT ST_AsText(ST_ForcePolygonCCW(ST_ForcePolygonCCW({UNIT_SQUARE})))"),
    )
    .await?;
    assert_eq!(once, twice);
    Ok(())
}
