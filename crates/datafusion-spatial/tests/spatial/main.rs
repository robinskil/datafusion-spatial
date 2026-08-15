//! End-to-end SQL tests, one module per function family.
//!
//! These modules share one test binary rather than one binary each. DataFusion is a large
//! crate, and every extra integration test target links the whole of it again.
//!
//! Run one family with a filter:
//!
//! ```bash
//! cargo test --test spatial predicates
//! ```

mod common;

mod accessors;
mod affine;
mod aggregates;
mod bearings;
mod bounding_box;
mod clusters;
mod components;
mod constructors;
mod crs;
mod edits;
mod extension_types;
mod io;
mod join;
mod linear_reference;
mod measurement;
mod overlay;
mod predicates;
mod registration;
mod shape;
mod tessellation;
mod transforms;
mod validity;

// `ST_Transform` exists only with the PROJ feature.
#[cfg(feature = "proj")]
mod reprojection;
