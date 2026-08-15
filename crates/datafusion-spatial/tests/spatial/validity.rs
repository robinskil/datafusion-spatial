//! `ST_IsValid`, `ST_IsValidReason` and `ST_MakeValid`.

use crate::common::*;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn validity_and_repair() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    let bow_tie = "ST_GeomFromText('POLYGON((0 0,2 2,2 0,0 2,0 0))')";

    assert!(scalar_bool(&ctx, &format!("SELECT ST_IsValid({UNIT_SQUARE})")).await?);
    assert!(!scalar_bool(&ctx, &format!("SELECT ST_IsValid({bow_tie})")).await?);

    assert_eq!(
        scalar_text(&ctx, &format!("SELECT ST_IsValidReason({UNIT_SQUARE})")).await?,
        "Valid Geometry"
    );
    let reason = scalar_text(&ctx, &format!("SELECT ST_IsValidReason({bow_tie})")).await?;
    assert_ne!(reason, "Valid Geometry");
    assert!(!reason.is_empty());

    // And the repaired geometry really is valid.
    assert!(scalar_bool(&ctx, &format!("SELECT ST_IsValid(ST_MakeValid({bow_tie}))")).await?);
    Ok(())
}
