//! `ST_Envelope`, the box accessors and the `&&` operator.

use crate::common::*;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::Array;
use datafusion_spatial::datafusion;

// ---------------------------------------------------------- bounds

#[tokio::test]
async fn bounds_of_a_polygon() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_XMin({RECTANGLE})")).await?,
        0.0
    );
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_YMin({RECTANGLE})")).await?,
        0.0
    );
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_XMax({RECTANGLE})")).await?,
        4.0
    );
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_YMax({RECTANGLE})")).await?,
        3.0
    );
    Ok(())
}

/// The box walker is two-dimensional, so a z bound is null rather than a guess.
#[tokio::test]
async fn z_bounds_are_null_in_two_dimensions() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    for function in ["ST_ZMin", "ST_ZMax"] {
        let batches = collect(&ctx, &format!("SELECT {function}({RECTANGLE}) AS z")).await?;
        assert!(batches[0].column(0).is_null(0), "{function} must be null");
    }
    Ok(())
}

#[tokio::test]
async fn bounds_over_a_column() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    let batches = collect(
        &ctx,
        "SELECT ST_XMin(geom) AS lo, ST_XMax(geom) AS hi FROM shapes",
    )
    .await?;

    let lo = batches[0].column(0).as_primitive::<Float64Type>();
    let hi = batches[0].column(1).as_primitive::<Float64Type>();
    assert_eq!((lo.value(0), hi.value(0)), (0.0, 1.0));
    assert_eq!((lo.value(1), hi.value(1)), (0.5, 1.5));
    assert!(lo.is_null(2), "the null row stays null");
    Ok(())
}

// ---------------------------------------------------------- envelope

#[tokio::test]
async fn envelope_is_a_box() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;

    // A box reads as a polygon, and its area matches the bounding rectangle.
    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_GeometryType(ST_Envelope({RECTANGLE}))")
        )
        .await?,
        "ST_Polygon"
    );
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_Area(ST_Envelope({RECTANGLE}))")).await?,
        12.0
    );

    // The bounds of the envelope match the bounds of the input.
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_XMax(ST_Envelope({RECTANGLE}))")).await?,
        scalar_f64(&ctx, &format!("SELECT ST_XMax({RECTANGLE})")).await?
    );
    Ok(())
}

/// An envelope of an envelope is the same box.
#[tokio::test]
async fn envelope_is_idempotent() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    assert_eq!(
        scalar_f64(
            &ctx,
            &format!("SELECT ST_Area(ST_Envelope(ST_Envelope({RECTANGLE})))")
        )
        .await?,
        scalar_f64(&ctx, &format!("SELECT ST_Area(ST_Envelope({RECTANGLE}))")).await?
    );
    Ok(())
}

#[tokio::test]
async fn expand_grows_the_box() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    let point = "ST_GeomFromText('POINT(5 5)')";

    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_XMin(ST_Expand({point}, 2))")).await?,
        3.0
    );
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_XMax(ST_Expand({point}, 2))")).await?,
        7.0
    );
    // A point expanded by r gives a 2r by 2r box.
    assert_eq!(
        scalar_f64(&ctx, &format!("SELECT ST_Area(ST_Expand({point}, 2))")).await?,
        16.0
    );
    Ok(())
}

// ---------------------------------------------------------- operator

#[tokio::test]
async fn bbox_intersects_is_the_box_test() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    let near = "ST_GeomFromText('POINT(2 2)')";
    let far = "ST_GeomFromText('POINT(99 99)')";

    assert!(
        scalar_bool(
            &ctx,
            &format!("SELECT ST_BBoxIntersects({RECTANGLE}, {near})")
        )
        .await?
    );
    assert!(
        !scalar_bool(
            &ctx,
            &format!("SELECT ST_BBoxIntersects({RECTANGLE}, {far})")
        )
        .await?
    );
    Ok(())
}

/// The operator is the cheap half of `ST_Intersects`, so it can only be more permissive.
///
/// A point in the box but outside a concave shape is exactly the case that separates them.
#[tokio::test]
async fn bbox_intersects_is_weaker_than_st_intersects() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    // An L shape, whose bounding box covers the missing corner.
    let l_shape = "ST_GeomFromText('POLYGON((0 0,2 0,2 1,1 1,1 2,0 2,0 0))')";
    let corner = "ST_GeomFromText('POINT(1.8 1.8)')";

    assert!(
        scalar_bool(
            &ctx,
            &format!("SELECT ST_BBoxIntersects({l_shape}, {corner})")
        )
        .await?,
        "the box covers the missing corner"
    );
    assert!(
        !scalar_bool(&ctx, &format!("SELECT ST_Intersects({l_shape}, {corner})")).await?,
        "but the shape does not"
    );
    Ok(())
}

#[tokio::test]
async fn bbox_intersects_filters_rows() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    let batches = collect(
        &ctx,
        "SELECT COUNT(*) AS n FROM shapes \
         WHERE ST_BBoxIntersects(geom, ST_GeomFromText('POINT(1.2 0.5)'))",
    )
    .await?;
    let count = batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0);
    assert_eq!(count, 1, "only the shifted square reaches that far");
    Ok(())
}
