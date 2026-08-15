//! Delaunay, Voronoi and the Chaikin algorithm.

use crate::common::*;
use datafusion_spatial::datafusion;
use geoarrow_array::GeoArrowArray;

// ---------------------------------------------------------- tessellation

#[tokio::test]
async fn delaunay_and_voronoi() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let scatter = "ST_GeomFromText('MULTIPOINT(0 0,4 0,4 4,0 4,2 2)')";

    // The triangles cover the convex hull.
    let area = scalar_f64(
        &ctx,
        &format!("SELECT ST_Area(ST_DelaunayTriangles({scatter}))"),
    )
    .await?;
    assert!((area - 16.0).abs() < 1e-9, "area was {area}");

    // One Voronoi cell per input point.
    assert_eq!(
        scalar_i32(
            &ctx,
            &format!("SELECT ST_NumGeometries(ST_VoronoiPolygons({scatter}))")
        )
        .await?,
        5
    );

    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_GeometryType(ST_VoronoiLines({scatter}))")
        )
        .await?,
        "ST_MultiLineString"
    );
    Ok(())
}

/// Fewer than three vertices cannot be tessellated, so the answer is null.
#[tokio::test]
async fn tessellation_of_two_points_is_null() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let pair = "ST_GeomFromText('MULTIPOINT(0 0,1 1)')";
    for function in [
        "ST_DelaunayTriangles",
        "ST_VoronoiPolygons",
        "ST_VoronoiLines",
    ] {
        let df = ctx.sql(&format!("SELECT {function}({pair}) AS g")).await?;
        let field = df.schema().as_arrow().field(0).clone();
        let batches = df.collect().await?;
        let array =
            geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), &field).unwrap();
        assert_eq!(array.logical_null_count(), 1, "{function} must be null");
    }
    Ok(())
}

// ---------------------------------------------------------- smooth

#[tokio::test]
async fn chaikin_smooth_adds_vertices() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let zigzag = "ST_GeomFromText('LINESTRING(0 0,1 1,2 0,3 1)')";

    let before = scalar_i32(&ctx, &format!("SELECT ST_NPoints({zigzag})")).await?;
    let after = scalar_i32(
        &ctx,
        &format!("SELECT ST_NPoints(ST_ChaikinSmoothing({zigzag}, 2))"),
    )
    .await?;
    assert!(after > before, "{before} points became {after}");

    // An unreasonable iteration count gives null. It does not blow up the vertex count.
    let df = ctx
        .sql(&format!("SELECT ST_ChaikinSmoothing({zigzag}, 99) AS g"))
        .await?;
    let field = df.schema().as_arrow().field(0).clone();
    let batches = df.collect().await?;
    let array =
        geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), &field).unwrap();
    assert_eq!(array.logical_null_count(), 1);
    Ok(())
}
