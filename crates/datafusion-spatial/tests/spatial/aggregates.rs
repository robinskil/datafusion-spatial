//! `ST_Extent`, `ST_Collect` and `ST_MemUnion`.

use crate::common::*;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use datafusion_spatial::datafusion;
use geoarrow_array::GeoArrowArray;

#[tokio::test]
async fn extent_over_a_table() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    let batches = collect(
        &ctx,
        "SELECT ST_XMin(ST_Extent(geom)) AS lo, ST_XMax(ST_Extent(geom)) AS hi FROM shapes",
    )
    .await?;

    assert_eq!(
        batches[0].column(0).as_primitive::<Float64Type>().value(0),
        0.0
    );
    assert_eq!(
        batches[0].column(1).as_primitive::<Float64Type>().value(0),
        1.5
    );
    Ok(())
}

#[tokio::test]
async fn collect_gathers_every_row() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    assert_eq!(
        scalar_text(&ctx, "SELECT ST_GeometryType(ST_Collect(geom)) FROM shapes").await?,
        "ST_GeometryCollection"
    );
    // Two non-null rows of five points each.
    let points = collect(&ctx, "SELECT ST_NPoints(ST_Collect(geom)) AS n FROM shapes").await?;
    assert_eq!(
        points[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        10
    );
    Ok(())
}

#[tokio::test]
async fn union_aggregate_merges_the_table() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    // Two unit squares that half overlap cover 1.5.
    let area = scalar_f64(&ctx, "SELECT ST_Area(ST_MemUnion(geom)) FROM shapes").await?;
    assert!((area - 1.5).abs() < 1e-9, "area was {area}");
    Ok(())
}

/// The scalar and aggregate unions must agree on the same input.
///
/// They cannot share the `ST_Union` name. DataFusion reads the scalar registry first. It does not
/// try the aggregate registry after an argument count mismatch. So the aggregate takes the name
/// `ST_MemUnion`.
#[tokio::test]
async fn scalar_and_aggregate_unions_agree() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    let unit = "ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 1,0 0))')";
    let shifted = "ST_GeomFromText('POLYGON((0.5 0,1.5 0,1.5 1,0.5 1,0.5 0))')";

    let scalar = scalar_f64(
        &ctx,
        &format!("SELECT ST_Area(ST_Union({unit}, {shifted}))"),
    )
    .await?;
    let aggregate = scalar_f64(&ctx, "SELECT ST_Area(ST_MemUnion(geom)) FROM shapes").await?;
    assert!((scalar - aggregate).abs() < 1e-9, "the two forms disagree");
    Ok(())
}

/// A call to the aggregate under the scalar name must fail with a clear message.
#[tokio::test]
async fn one_argument_st_union_is_a_planning_error() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    let err = collect(&ctx, "SELECT ST_Union(geom) FROM shapes")
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("expected 2 arguments"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// A GROUP BY forces the accumulator to merge partial states.
#[tokio::test]
async fn aggregates_merge_across_groups() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    let batches = collect(
        &ctx,
        "SELECT ST_XMin(geom) > 0.25 AS grp, ST_Area(ST_MemUnion(geom)) AS a \
         FROM shapes WHERE geom IS NOT NULL GROUP BY grp ORDER BY grp",
    )
    .await?;

    let areas = batches[0].column(1).as_primitive::<Float64Type>();
    assert_eq!(areas.len(), 2, "two groups");
    assert!((areas.value(0) - 1.0).abs() < 1e-9);
    assert!((areas.value(1) - 1.0).abs() < 1e-9);
    Ok(())
}

#[tokio::test]
async fn aggregates_of_all_nulls_are_null() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    for function in ["ST_Extent", "ST_Collect", "ST_MemUnion"] {
        let df = ctx
            .sql(&format!(
                "SELECT {function}(geom) AS g FROM shapes WHERE geom IS NULL"
            ))
            .await?;
        let field = df.schema().as_arrow().field(0).clone();
        let batches = df.collect().await?;

        // A box column has a validity buffer. A mixed geometry column is a union and has none.
        // So read the logical nulls through geoarrow in both cases.
        let array =
            geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), &field).unwrap();
        assert_eq!(
            array.logical_null_count(),
            1,
            "{function} over no rows must be null"
        );
    }
    Ok(())
}
