//! Structure edits, vertex edits and `ST_Dump`.

use crate::common::*;
use arrow_array::cast::AsArray;
use arrow_array::Array;
use datafusion_spatial::datafusion;

// ---------------------------------------------------------- structure

#[tokio::test]
async fn multi_and_points() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_GeometryType(ST_Multi(ST_GeomFromText('POINT(1 2)')))"
        )
        .await?,
        "ST_MultiPoint"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_GeometryType(ST_Multi(ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 0))')))"
        )
        .await?,
        "ST_MultiPolygon"
    );
    assert_eq!(
        scalar_i32(
            &ctx,
            "SELECT ST_NumGeometries(ST_Points(ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 0))')))"
        )
        .await?,
        4
    );
    Ok(())
}

#[tokio::test]
async fn snap_to_grid_rounds() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_SnapToGrid(ST_GeomFromText('POINT(1.234 5.678)'), 0.5))"
        )
        .await?,
        "POINT(1 5.5)"
    );
    Ok(())
}

#[tokio::test]
async fn vertex_edits() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let line = "ST_GeomFromText('LINESTRING(0 0,1 1,2 2)')";
    let point = "ST_GeomFromText('POINT(9 9)')";

    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_AsText(ST_AddPoint({line}, {point}))")
        )
        .await?,
        "LINESTRING(0 0,1 1,2 2,9 9)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_AsText(ST_AddPoint({line}, {point}, 1))")
        )
        .await?,
        "LINESTRING(0 0,9 9,1 1,2 2)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_AsText(ST_RemovePoint({line}, 1))")
        )
        .await?,
        "LINESTRING(0 0,2 2)"
    );
    assert_eq!(
        scalar_text(
            &ctx,
            &format!("SELECT ST_AsText(ST_SetPoint({line}, 1, {point}))")
        )
        .await?,
        "LINESTRING(0 0,9 9,2 2)"
    );
    Ok(())
}

// ---------------------------------------------------------- dump

/// `ST_Dump` returns a list, and `unnest` turns it back into rows.
#[tokio::test]
async fn dump_expands_with_unnest() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let batches = collect(
        &ctx,
        "SELECT ST_AsText(unnest(ST_Dump(\
           ST_GeomFromText('MULTIPOINT(1 1,2 2,3 3)')))) AS part",
    )
    .await?;

    let parts: Vec<String> = batches
        .iter()
        .flat_map(|batch| {
            let column = batch.column(0).as_string::<i32>();
            (0..column.len())
                .map(|i| column.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect();

    assert_eq!(parts, vec!["POINT(1 1)", "POINT(2 2)", "POINT(3 3)"]);
    Ok(())
}

#[tokio::test]
async fn dump_of_a_single_geometry_is_one_part() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let batches = collect(
        &ctx,
        "SELECT COUNT(*) AS n FROM (\
           SELECT unnest(ST_Dump(ST_GeomFromText('POINT(1 1)'))) AS part)",
    )
    .await?;
    let count = batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0);
    assert_eq!(count, 1);
    Ok(())
}
