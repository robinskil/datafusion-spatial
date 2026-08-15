#![forbid(unsafe_code)]
//! PostGIS-compatible spatial functions for Apache DataFusion, on GeoArrow.
//!
//! # Usage
//!
//! ```no_run
//! use datafusion_spatial::datafusion::prelude::SessionContext;
//!
//! # async fn run() -> datafusion_spatial::datafusion::error::Result<()> {
//! let ctx = SessionContext::new();
//! datafusion_spatial::register_all(&ctx);
//!
//! let df = ctx.sql("SELECT ST_X(ST_GeomFromText('POINT(1 2)'))").await?;
//! df.show().await?;
//! # Ok(())
//! # }
//! ```

// ---------------------------------------------------------------- DataFusion version selection
//
// One Cargo feature picks the DataFusion major. The chosen crate is renamed to `datafusion` here,
// so every `use datafusion::...` path below is version independent.
//
// The two majors this crate supports both build on arrow 58, which is the version geoarrow 0.8
// needs. That agreement is the reason both can be offered at all.

#[cfg(not(any(feature = "df53", feature = "df54")))]
compile_error!(
    "datafusion-spatial needs one DataFusion version feature. Turn on `df53` or `df54`."
);

#[cfg(all(feature = "df53", feature = "df54"))]
compile_error!(
    "datafusion-spatial accepts one DataFusion version feature, not two. Set \
     `default-features = false` and then turn on `df53` or `df54`."
);

/// The DataFusion crate this build links against.
///
/// Read DataFusion through this re-export, not through a direct dependency. It cannot drift from
/// the version these functions were compiled against.
#[cfg(all(feature = "df53", not(feature = "df54")))]
pub extern crate datafusion_53 as datafusion;

/// The DataFusion crate this build links against.
#[cfg(all(feature = "df54", not(feature = "df53")))]
pub extern crate datafusion_54 as datafusion;

pub mod join;

use std::sync::Arc;

use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::{AggregateUDF, ScalarUDF};
use datafusion::prelude::SessionContext;

pub use datafusion_spatial_kernels as kernels;
pub use datafusion_spatial_udf as udf;

/// Every scalar function this crate provides.
pub fn scalar_udfs() -> Vec<ScalarUDF> {
    let mut functions = vec![udf::st_x(), udf::st_y(), udf::st_z(), udf::st_m()];
    functions.extend(udf::simple_accessors());
    functions.extend(udf::components());
    functions.extend(udf::transforms());
    functions.extend(udf::constructors());
    functions.extend(udf::io_functions());
    functions.extend(udf::predicates());
    functions.extend(udf::measures());
    functions.extend(udf::linear_functions());
    functions.extend(udf::processing());
    functions.extend(udf::envelopes());
    functions.extend(udf::edits());
    #[cfg(feature = "proj")]
    functions.extend(udf::reprojections());
    functions
}

/// Every window function this crate provides.
///
/// A cluster function reads all the rows of a partition at once. PostGIS defines it as a window
/// function and so does this crate.
pub fn window_udfs() -> Vec<datafusion::logical_expr::WindowUDF> {
    udf::clusters()
}

/// Every aggregate function this crate provides.
pub fn aggregate_udfs() -> Vec<AggregateUDF> {
    vec![udf::st_extent(), udf::st_collect(), udf::st_memunion_agg()]
}

/// Register every spatial function on a session.
///
/// A function already registered under the same name is replaced.
pub fn register_all(ctx: &SessionContext) {
    for func in scalar_udfs() {
        ctx.register_udf(func);
    }
    for func in aggregate_udfs() {
        ctx.register_udaf(func);
    }
    for func in window_udfs() {
        ctx.register_udwf(func);
    }
}

/// Register every spatial function on any [`FunctionRegistry`].
///
/// Use this when you build a session state by hand instead of through [`SessionContext`].
pub fn register_into(registry: &mut dyn FunctionRegistry) -> datafusion::error::Result<()> {
    for func in scalar_udfs() {
        registry.register_udf(Arc::new(func))?;
    }
    for func in aggregate_udfs() {
        registry.register_udaf(Arc::new(func))?;
    }
    for func in window_udfs() {
        registry.register_udwf(Arc::new(func))?;
    }
    Ok(())
}
