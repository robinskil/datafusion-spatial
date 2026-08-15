//! `ST_ClusterKMeans` and `ST_ClusterDBSCAN`.

use crate::common::*;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use datafusion_spatial::datafusion;

/// The cluster functions are window functions, as in PostGIS.
#[tokio::test]
async fn kmeans_separates_two_groups() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let batches = collect(
        &ctx,
        "SELECT ST_ClusterKMeans(geom, 2) OVER () AS cluster FROM pts",
    )
    .await?;

    let ids = batches[0].column(0).as_primitive::<Int32Type>();
    assert_eq!(ids.len(), 6);
    assert_eq!(ids.value(0), ids.value(1));
    assert_eq!(ids.value(1), ids.value(2));
    assert_eq!(ids.value(3), ids.value(4));
    assert_ne!(ids.value(0), ids.value(3), "the two groups must differ");
    Ok(())
}

#[tokio::test]
async fn dbscan_separates_two_groups() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let batches = collect(
        &ctx,
        "SELECT ST_ClusterDBSCAN(geom, 1.0, 2) OVER () AS cluster FROM pts",
    )
    .await?;

    let ids = batches[0].column(0).as_primitive::<Int32Type>();
    assert_eq!(ids.value(0), ids.value(1));
    assert_eq!(ids.value(3), ids.value(4));
    assert_ne!(ids.value(0), ids.value(3));
    Ok(())
}

/// The same query twice must give the same assignment.
#[tokio::test]
async fn kmeans_is_reproducible() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let sql = "SELECT ST_ClusterKMeans(geom, 2) OVER () AS cluster FROM pts";
    let first = collect(&ctx, sql).await?;
    let second = collect(&ctx, sql).await?;
    assert_eq!(first[0].column(0), second[0].column(0));
    Ok(())
}

#[tokio::test]
async fn clusters_rejects_a_bad_parameter() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    for sql in [
        "SELECT ST_ClusterKMeans(geom, 0) OVER () FROM pts",
        "SELECT ST_ClusterKMeans(geom, 99) OVER () FROM pts",
        "SELECT ST_ClusterDBSCAN(geom, 0, 2) OVER () FROM pts",
    ] {
        assert!(
            collect(&ctx, sql).await.is_err(),
            "{sql} should have failed"
        );
    }
    Ok(())
}
