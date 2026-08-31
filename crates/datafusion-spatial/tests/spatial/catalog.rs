//! `SHOW FUNCTIONS` must list the functions this crate registers.
//!
//! DataFusion builds `SHOW FUNCTIONS` from `information_schema.parameters`, and it fills that view
//! from the example argument types of a signature. A signature that names no example type gets no
//! row, and the function then leaves the view while it still runs. See
//! [`datafusion_spatial::udf::signature`] for the signatures that avoid it.

use arrow_array::cast::AsArray;
use arrow_array::Array;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_spatial::datafusion;
use std::collections::BTreeSet;

/// The functions DataFusion cannot put in the catalog.
///
/// `ST_SetSRID` stamps the coordinate reference system on its output column, so its return type
/// follows the *value* of its second argument. DataFusion builds the catalog with no argument
/// values at all, so it cannot derive that type and writes no row. `ST_Transform`, which the
/// `proj` feature adds, has the same shape. Both still run.
const WITHOUT_A_CATALOG_ROW: [&str; 2] = ["st_setsrid", "st_transform"];

/// Every function name this crate registers, in lower case.
fn registered_names() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for func in datafusion_spatial::scalar_udfs() {
        names.insert(func.name().to_ascii_lowercase());
    }
    for func in datafusion_spatial::aggregate_udfs() {
        names.insert(func.name().to_ascii_lowercase());
    }
    for func in datafusion_spatial::window_udfs() {
        names.insert(func.name().to_ascii_lowercase());
    }
    names
}

/// Every `st_` name that `SHOW FUNCTIONS` reports.
async fn listed_names() -> datafusion::error::Result<BTreeSet<String>> {
    // `SHOW FUNCTIONS` reads `information_schema`, which a session turns on.
    let config = SessionConfig::new().with_information_schema(true);
    let ctx = SessionContext::new_with_config(config);
    datafusion_spatial::register_all(&ctx);

    let batches = ctx.sql("SHOW FUNCTIONS").await?.collect().await?;
    let mut names = BTreeSet::new();
    for batch in &batches {
        let column = batch.column(0).as_string::<i32>();
        for row in 0..column.len() {
            let name = column.value(row).to_ascii_lowercase();
            if name.starts_with("st_") {
                names.insert(name);
            }
        }
    }
    Ok(names)
}

#[tokio::test]
async fn show_functions_lists_every_registered_function() -> datafusion::error::Result<()> {
    let listed = listed_names().await?;

    let mut expected = registered_names();
    for name in WITHOUT_A_CATALOG_ROW {
        expected.remove(name);
    }

    let missing: Vec<_> = expected.difference(&listed).collect();
    assert!(
        missing.is_empty(),
        "{} registered functions are absent from SHOW FUNCTIONS: {missing:?}",
        missing.len()
    );

    // The other direction guards the exception list above. A function that gains a catalog row
    // must leave that list, so the list cannot grow stale without notice.
    let extra: Vec<_> = listed.difference(&expected).collect();
    assert!(
        extra.is_empty(),
        "SHOW FUNCTIONS lists names this test does not expect: {extra:?}"
    );

    assert_eq!(listed.len(), expected.len());
    Ok(())
}

/// The five functions the report named. Each one takes a geometry, which is the shape that used to
/// drop out of the view.
#[tokio::test]
async fn show_functions_lists_the_functions_that_take_a_geometry() -> datafusion::error::Result<()>
{
    let listed = listed_names().await?;
    for name in [
        "st_distance",
        "st_intersects",
        "st_buffer",
        "st_extent",
        "st_clusterkmeans",
    ] {
        assert!(listed.contains(name), "SHOW FUNCTIONS omits {name}");
    }
    Ok(())
}
