//! `ST_X` and the other functions that read a property of a geometry.

use crate::common::*;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::Array;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn ordinates() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    assert_eq!(
        scalar_f64(&ctx, "SELECT ST_X(ST_MakePoint(1.5, 2.5))").await?,
        1.5
    );
    assert_eq!(
        scalar_f64(&ctx, "SELECT ST_Y(ST_MakePoint(1.5, 2.5))").await?,
        2.5
    );
    assert_eq!(
        scalar_f64(&ctx, "SELECT ST_Z(ST_MakePoint(1.5, 2.5, 3.5))").await?,
        3.5
    );

    // A two-dimensional point has no z and no measure.
    let batches = collect(&ctx, "SELECT ST_Z(ST_MakePoint(1.0, 2.0)) AS z").await?;
    assert!(batches[0].column(0).is_null(0));
    let batches = collect(&ctx, "SELECT ST_M(ST_MakePoint(1.0, 2.0)) AS m").await?;
    assert!(batches[0].column(0).is_null(0));
    Ok(())
}

#[tokio::test]
async fn type_and_dimension_accessors() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    let batches = collect(
        &ctx,
        "SELECT ST_GeometryType(geom) AS t, ST_Dimension(geom) AS d, ST_CoordDim(geom) AS c \
         FROM shapes",
    )
    .await?;

    let types = batches[0].column(0).as_string::<i32>();
    assert_eq!(types.value(0), "ST_Point");
    assert_eq!(types.value(1), "ST_LineString");
    assert_eq!(types.value(2), "ST_Polygon");
    assert_eq!(types.value(3), "ST_MultiPoint");

    let dims = batches[0].column(1).as_primitive::<Int32Type>();
    assert_eq!((dims.value(0), dims.value(1), dims.value(2)), (0, 1, 2));

    let coord_dims = batches[0].column(2).as_primitive::<Int32Type>();
    assert!((0..4).all(|i| coord_dims.value(i) == 2));
    Ok(())
}

#[tokio::test]
async fn count_accessors() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    let batches = collect(
        &ctx,
        "SELECT ST_NPoints(geom) AS n, ST_NumGeometries(geom) AS g, \
                ST_NumInteriorRings(geom) AS r, ST_NumPoints(geom) AS p \
         FROM shapes",
    )
    .await?;

    let npoints = batches[0].column(0).as_primitive::<Int32Type>();
    assert_eq!(npoints.value(0), 1);
    assert_eq!(npoints.value(1), 3);
    assert_eq!(npoints.value(2), 9, "5 shell plus 4 hole");

    let parts = batches[0].column(1).as_primitive::<Int32Type>();
    assert_eq!(parts.value(3), 2, "the multi point has two parts");

    let rings = batches[0].column(2).as_primitive::<Int32Type>();
    assert_eq!(rings.value(2), 1);
    assert!(rings.is_null(0), "not a polygon");

    let num_points = batches[0].column(3).as_primitive::<Int32Type>();
    assert_eq!(num_points.value(1), 3);
    assert!(num_points.is_null(2), "ST_NumPoints is line string only");
    Ok(())
}

#[tokio::test]
async fn boolean_accessors() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    assert!(!scalar_bool(&ctx, "SELECT ST_IsEmpty(ST_MakePoint(1.0, 2.0))").await?);
    assert!(
        scalar_bool(
            &ctx,
            "SELECT ST_IsClosed(ST_GeomFromText('LINESTRING(0 0,1 0,1 1,0 0)'))"
        )
        .await?
    );
    assert!(
        !scalar_bool(
            &ctx,
            "SELECT ST_IsClosed(ST_GeomFromText('LINESTRING(0 0,1 1)'))"
        )
        .await?
    );
    assert!(
        scalar_bool(
            &ctx,
            "SELECT ST_IsRing(ST_GeomFromText('LINESTRING(0 0,1 0,1 1,0 0)'))"
        )
        .await?
    );
    // A bow tie is closed but crosses itself, so it is not a ring.
    assert!(
        !scalar_bool(
            &ctx,
            "SELECT ST_IsRing(ST_GeomFromText('LINESTRING(0 0,2 2,2 0,0 2,0 0)'))"
        )
        .await?
    );
    assert!(
        !scalar_bool(
            &ctx,
            "SELECT ST_IsSimple(ST_GeomFromText('LINESTRING(0 0,2 2,2 0,0 2)'))"
        )
        .await?
    );
    Ok(())
}
