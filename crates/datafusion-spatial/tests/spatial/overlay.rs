//! `ST_Union`, `ST_Intersection`, `ST_Difference` and `ST_SymDifference`.

use crate::common::*;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::Array;
use datafusion_spatial::datafusion;
use geoarrow_array::GeoArrowArray;

#[tokio::test]
async fn overlay_areas_add_up() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;

    // Two unit squares that overlap on half their width.
    let cases = [
        ("ST_Union", 1.5),
        ("ST_Intersection", 0.5),
        ("ST_Difference", 0.5),
        ("ST_SymDifference", 1.0),
    ];
    for (function, expected) in cases {
        let area = scalar_f64(
            &ctx,
            &format!("SELECT ST_Area({function}({UNIT_SQUARE}, {SHIFTED_SQUARE}))"),
        )
        .await?;
        assert!(
            (area - expected).abs() < 1e-9,
            "{function} gave area {area}, expected {expected}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn overlay_over_a_column() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    let batches = collect(
        &ctx,
        &format!("SELECT ST_Area(ST_Intersection(geom, {UNIT_SQUARE})) AS a FROM shapes"),
    )
    .await?;

    let areas = batches[0].column(0).as_primitive::<Float64Type>();
    assert!((areas.value(0) - 1.0).abs() < 1e-9, "unit against unit");
    assert!(
        (areas.value(1) - 1.0).abs() < 1e-9,
        "the unit sits inside the big one"
    );
    assert!(areas.is_null(2), "the null row stays null");
    Ok(())
}

/// Boolean operations are defined for areal geometries, so a point gives null.
///
/// A mixed geometry column is an Arrow union, and a union has no validity buffer of its own, so
/// `Array::is_null` always answers false for one. `GeoArrowArray::logical_nulls` is the accessor
/// that reads through to the child arrays, which is exactly why geoarrow defines it.
#[tokio::test]
async fn overlay_of_a_point_is_null() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    let df = ctx
        .sql(&format!(
            "SELECT ST_Union({UNIT_SQUARE}, ST_GeomFromText('POINT(5 5)')) AS u"
        ))
        .await?;
    let field = df.schema().as_arrow().field(0).clone();
    let batches = df.collect().await?;

    let array =
        geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), &field).unwrap();
    assert_eq!(
        array.logical_null_count(),
        1,
        "a non-areal argument must give null"
    );
    Ok(())
}
