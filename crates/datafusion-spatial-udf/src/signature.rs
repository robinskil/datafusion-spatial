//! Signatures for the functions that take a geometry.
//!
//! # Why not `Signature::any`
//!
//! A geometry argument has many Arrow storage types: a struct of coordinate buffers, an
//! interleaved list, a union, WKB bytes or WKT text. No fixed type list covers them all, so the
//! argument test must accept any type and read the field metadata later. [`TypeSignature::Any`]
//! does that.
//!
//! [`TypeSignature::Any`] also reports no example argument type. DataFusion fills
//! `information_schema.parameters` from those example types, and `SHOW FUNCTIONS` joins that view
//! against itself. A function with no example type gets no row, so both drop it. That hid 113 of
//! the 123 functions of this crate.
//!
//! # What these signatures do instead
//!
//! Each argument gets [`TypeSignatureClass::Any`] as its desired type. That class accepts every
//! type and casts nothing, which is what [`TypeSignature::Any`] did. Each argument also names one
//! example type, which fills the catalog. A call is never restricted by that example.

use std::sync::Arc;

use datafusion::common::types::{logical_float64, logical_string, LogicalField, NativeType};
use datafusion::logical_expr::{
    Coercion, Signature, TypeSignature, TypeSignatureClass, Volatility,
};

/// What one argument of a spatial function holds.
///
/// The kind never restricts a call. It names the example type that the function catalog shows.
/// An example must be a type the function really accepts, because DataFusion asks the function
/// for its return type with exactly that example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arg {
    /// A geometry, in any GeoArrow encoding. The catalog shows WKB, which is one of them.
    Geometry,
    /// A geometry that must carry a coordinate buffer. WKB and WKT do not, so the catalog shows
    /// an interleaved GeoArrow point instead.
    Coordinates,
    /// A distance, a tolerance, an angle or another real number.
    Number,
    /// A count, an index or an SRID.
    Integer,
    /// Text, such as a DE-9IM pattern.
    Text,
}

impl Arg {
    /// The example type of this argument.
    fn example(self) -> TypeSignatureClass {
        match self {
            Self::Geometry => TypeSignatureClass::Binary,
            Self::Coordinates => TypeSignatureClass::Native(Arc::new(NativeType::FixedSizeList(
                Arc::new(LogicalField {
                    name: "xy".to_string(),
                    logical_type: logical_float64(),
                    nullable: false,
                }),
                2,
            ))),
            Self::Number => TypeSignatureClass::Float,
            Self::Integer => TypeSignatureClass::Integer,
            Self::Text => TypeSignatureClass::Native(logical_string()),
        }
    }
}

/// A signature for one argument shape.
pub fn args(kinds: &[Arg]) -> Signature {
    Signature::new(shape(kinds), Volatility::Immutable)
}

/// A signature for a function that takes `count` geometries and nothing else.
pub fn geometries(count: usize) -> Signature {
    args(&vec![Arg::Geometry; count])
}

/// A signature for a geometry followed by `count` arguments of one kind.
pub fn geometry_then(kind: Arg, count: usize) -> Signature {
    let mut kinds = vec![Arg::Geometry];
    kinds.resize(count + 1, kind);
    args(&kinds)
}

/// A signature for a function with more than one argument shape, such as an optional argument.
pub fn one_of(shapes: &[&[Arg]]) -> Signature {
    Signature::one_of(
        shapes.iter().map(|kinds| shape(kinds)).collect(),
        Volatility::Immutable,
    )
}

fn shape(kinds: &[Arg]) -> TypeSignature {
    TypeSignature::Coercible(kinds.iter().map(|kind| coercion(*kind)).collect())
}

/// Accept the argument as it stands, and name an example type for the catalog.
///
/// The desired type is [`TypeSignatureClass::Any`], which matches every argument and returns it
/// unchanged. So the implicit branch never runs: `allowed_source_types` is read for the example
/// type alone, and `default_casted_type` is never applied.
fn coercion(kind: Arg) -> Coercion {
    Coercion::new_implicit(
        TypeSignatureClass::Any,
        vec![kind.example()],
        NativeType::Binary,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;

    /// Every argument must offer at least one example type. An argument with none empties the
    /// whole list, and the function then leaves `information_schema.parameters`.
    #[test]
    fn every_argument_has_an_example_type() {
        for kind in [
            Arg::Geometry,
            Arg::Coordinates,
            Arg::Number,
            Arg::Integer,
            Arg::Text,
        ] {
            let signature = args(&[kind]);
            let examples = signature.type_signature.get_example_types();
            assert!(!examples.is_empty(), "{kind:?} names no example type");
        }
    }

    #[test]
    fn a_shape_yields_one_example_per_argument() {
        let signature = args(&[Arg::Geometry, Arg::Number]);
        assert_eq!(
            signature.type_signature.get_example_types(),
            vec![vec![DataType::Binary, DataType::Float64]]
        );
    }

    #[test]
    fn one_of_yields_an_example_for_every_shape() {
        let signature = one_of(&[&[Arg::Geometry], &[Arg::Geometry, Arg::Integer]]);
        assert_eq!(
            signature.type_signature.get_example_types(),
            vec![
                vec![DataType::Binary],
                vec![DataType::Binary, DataType::Int64]
            ]
        );
    }
}
