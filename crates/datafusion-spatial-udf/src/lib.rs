#![forbid(unsafe_code)]
//! DataFusion bindings for [`datafusion_spatial_kernels`].
//!
//! Each UDF here is glue: a signature, a plan-time type check, and a call into a kernel. No
//! geometric work happens in this crate.
//!
//! # Why `return_field_from_args`
//!
//! Every function that returns a geometry implements
//! [`return_field_from_args`][datafusion::logical_expr::ScalarUDFImpl::return_field_from_args]
//! instead of `return_type`. A bare `DataType` cannot hold the GeoArrow extension metadata.
//! A function that used `return_type` would drop the coordinate reference system. The next
//! function in the chain could then not read the geometry. This is a correctness requirement,
//! not an optimization.
//!
//! # Which columns count as geometries
//!
//! A column with GeoArrow extension metadata is a geometry. GeoArrow also reads a plain `Utf8`
//! column as WKT, and a plain `Binary` column as WKB. So a raw text column from CSV works
//! without a cast. The cost is one surprise: `ST_AsText` on any string column returns that
//! string. Pass such a column through `ST_GeomFromText` to check the parse step.

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

#[macro_use]
pub mod macros;

pub mod accessor;
pub mod aggregate;
pub mod cluster;
pub mod component;
pub mod constructor;
pub mod edit;
pub mod envelope;
pub mod io;
pub mod linear;
pub mod measure;
pub mod predicate;
pub mod process;
/// `ST_Transform`. Needs the `proj` feature.
#[cfg(feature = "proj")]
pub mod reproject;
pub mod transform;
pub mod util;

pub use accessor::{simple_accessors, st_m, st_x, st_y, st_z};
pub use aggregate::{st_collect, st_extent, st_memunion_agg};
pub use cluster::clusters;
pub use component::components;
pub use constructor::constructors;
pub use edit::edits;
pub use envelope::envelopes;
pub use io::{io_functions, st_geomfromtext};
pub use linear::linear_functions;
pub use measure::measures;
pub use predicate::{predicates, st_intersects};
pub use process::processing;
#[cfg(feature = "proj")]
pub use reproject::reprojections;
pub use transform::transforms;
