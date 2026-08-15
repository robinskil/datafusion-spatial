//! `ST_FlipCoordinates`, `ST_Force2D` and `ST_Force3D`.

use crate::common::*;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn flip_and_force() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;

    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_FlipCoordinates(ST_MakePoint(1.0, 2.0)))"
        )
        .await?,
        "POINT(2 1)"
    );
    assert_eq!(
        scalar_i32(
            &ctx,
            "SELECT ST_CoordDim(ST_Force3D(ST_MakePoint(1.0, 2.0)))"
        )
        .await?,
        3
    );
    assert_eq!(
        scalar_i32(
            &ctx,
            "SELECT ST_CoordDim(ST_Force2D(ST_Force3D(ST_MakePoint(1.0, 2.0))))"
        )
        .await?,
        2
    );

    // The transform must survive a column, not only a literal.
    let batches = collect(&ctx, "SELECT ST_X(ST_FlipCoordinates(geom)) AS x FROM pts").await?;
    let x = batches[0].column(0).as_primitive::<Float64Type>();
    assert_eq!(x.value(0), 2.0, "x and y swapped");
    Ok(())
}
