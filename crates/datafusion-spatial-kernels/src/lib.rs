#![forbid(unsafe_code)]
//! Array-in, array-out spatial kernels.
//!
//! This crate holds every speed-sensitive code path. It has no dependency on a query engine, so
//! each kernel can be benchmarked and profiled on its own.
//!
//! # Design rules
//!
//! Every kernel follows these rules. See `benches/PROFILE.md` for the measured effect.
//!
//! 1. Downcast one time per batch. Never match a geometry type per row.
//! 2. Slice buffers instead of a copy. A [`ScalarBuffer`][arrow_buffer::ScalarBuffer] clone is an
//!    atomic counter bump.
//! 3. Reuse the null buffer of the input.
//! 4. Test a bounding box before an exact predicate.
//! 5. Convert to `geo_types` only when a `geo` algorithm demands an owned value.

pub mod accessor;
pub mod affine;
pub mod aggregate;
pub mod bbox;
pub mod cluster;
pub mod component;
pub mod constructor;
pub mod crs;
pub mod edit;
pub mod envelope;
pub mod index;
pub mod io;
pub mod linear;
pub mod materialize;
pub mod measure;
pub mod predicate;
pub mod process;
/// Reprojection through PROJ. Needs the `proj` feature, which links a C++ library.
#[cfg(feature = "proj")]
pub mod reproject;
pub mod tessellate;
pub mod transform;

pub use bbox::Bbox;
