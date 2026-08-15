//! Hulls, centroids, boundaries, simplify and buffer.

use crate::common::*;
use arrow_array::cast::AsArray;
use datafusion::prelude::SessionContext;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn hulls_and_envelopes() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    let scatter = "ST_GeomFromText('MULTIPOINT(0 0,2 0,2 2,0 2,1 1)')";

    assert!(
        (scalar_f64(&ctx, &format!("SELECT ST_Area(ST_ConvexHull({scatter}))")).await? - 4.0).abs()
            < 1e-9
    );
    assert!(
        (scalar_f64(
            &ctx,
            &format!("SELECT ST_Area(ST_OrientedEnvelope({scatter}))")
        )
        .await?
            - 4.0)
            .abs()
            < 1e-9
    );
    // The concave hull of a convex set is the convex hull.
    assert!(
        scalar_f64(
            &ctx,
            &format!("SELECT ST_Area(ST_ConcaveHull({scatter}, 1.0))")
        )
        .await?
            > 0.0
    );
    Ok(())
}

#[tokio::test]
async fn centroid_and_point_on_surface() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_AsText(ST_Centroid({UNIT_SQUARE}))")
        )
        .await?,
        "POINT(0.5 0.5)"
    );
    // A point on surface must lie inside the polygon.
    assert!(
        scalar_bool(
            &ctx,
            &format!("SELECT ST_Contains({UNIT_SQUARE}, ST_PointOnSurface({UNIT_SQUARE}))")
        )
        .await?
    );
    Ok(())
}

#[tokio::test]
async fn boundary_follows_the_ogc_rules() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;

    // A polygon's boundary is its rings.
    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_GeometryType(ST_Boundary({UNIT_SQUARE}))")
        )
        .await?,
        "ST_MultiLineString"
    );
    // An open line gives its two endpoints.
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_Boundary(ST_GeomFromText('LINESTRING(0 0,1 1)')))"
        )
        .await?,
        "MULTIPOINT((0 0),(1 1))"
    );
    // A closed ring has an empty boundary.
    assert_eq!(
        scalar_i32_npoints(
            &ctx,
            "ST_Boundary(ST_GeomFromText('LINESTRING(0 0,1 0,1 1,0 0)'))"
        )
        .await?,
        0
    );
    Ok(())
}

async fn scalar_i32_npoints(
    ctx: &SessionContext,
    expression: &str,
) -> datafusion::error::Result<i32> {
    let batches = collect(ctx, &format!("SELECT ST_NPoints({expression})")).await?;
    Ok(batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int32Type>()
        .value(0))
}

#[tokio::test]
async fn simplify_and_segmentize() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    let wobbly = "ST_GeomFromText('LINESTRING(0 0,1 0.001,2 0,3 0)')";

    for function in ["ST_Simplify", "ST_SimplifyVW"] {
        let kept = scalar_i32_npoints(&ctx, &format!("{function}({wobbly}, 0.01)")).await?;
        assert!(kept < 4, "{function} kept every point");
    }

    // Segmentize adds vertices. It removes none.
    let dense = scalar_i32_npoints(
        &ctx,
        "ST_Segmentize(ST_GeomFromText('LINESTRING(0 0,10 0)'), 2.0)",
    )
    .await?;
    assert!(dense >= 6, "got {dense} points");
    Ok(())
}

#[tokio::test]
async fn buffer_grows_the_area() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    let area = scalar_f64(
        &ctx,
        &format!("SELECT ST_Area(ST_Buffer({UNIT_SQUARE}, 0.5))"),
    )
    .await?;
    // The unit square grown by 0.5: the square, four side strips, four quarter circles.
    let expected = 1.0 + 4.0 * 0.5 + std::f64::consts::PI * 0.25;
    assert!((area - expected).abs() < 0.05, "area was {area}");

    // A negative distance shrinks a polygon.
    let shrunk = scalar_f64(
        &ctx,
        &format!("SELECT ST_Area(ST_Buffer({UNIT_SQUARE}, -0.25))"),
    )
    .await?;
    assert!(shrunk < 1.0 && shrunk > 0.0, "shrunk area was {shrunk}");
    Ok(())
}
