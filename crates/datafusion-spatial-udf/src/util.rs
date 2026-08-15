//! Glue between DataFusion values and GeoArrow arrays.
//!
//! Nothing here does geometric work. The speed sensitive code lives in
//! `datafusion-spatial-kernels`.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BinaryArray, Float64Array, Int32Array, StringArray};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{plan_err, DataFusionError, Result, ScalarValue};
use datafusion::logical_expr::{ColumnarValue, ReturnFieldArgs};
use geoarrow_array::array::from_arrow_array;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::GeoArrowError;
use geoarrow_schema::{GeoArrowType, Metadata};

/// Wrap a GeoArrow error as a DataFusion error.
pub fn to_df(err: GeoArrowError) -> DataFusionError {
    DataFusionError::External(Box::new(err))
}

/// Read the GeoArrow type of an argument field.
///
/// The field carries the GeoArrow extension metadata. A bare [`DataType`] does not, which is why
/// every UDF here reads the field and not the type.
pub fn geo_type(function: &str, argument: usize, field: &Field) -> Result<GeoArrowType> {
    GeoArrowType::from_arrow_field(field).map_err(|err| {
        DataFusionError::Plan(format!(
            "{function} argument {} is not a geometry: {err}",
            argument + 1
        ))
    })
}

/// Reject two arguments that carry different coordinate reference systems.
///
/// PostGIS raises `Operation on mixed SRID geometries`. We do the same, at plan time.
pub fn check_same_crs(
    function: &str,
    left: &GeoArrowType,
    right: &GeoArrowType,
) -> Result<Arc<Metadata>> {
    let (lhs, rhs) = (left.metadata(), right.metadata());
    if lhs.crs() != rhs.crs() {
        return plan_err!(
            "{function} cannot mix coordinate reference systems: {:?} and {:?}",
            lhs.crs(),
            rhs.crs()
        );
    }
    Ok(Arc::clone(lhs))
}

/// Turn one argument into a GeoArrow array.
///
/// A scalar argument becomes an array of length one. The caller keeps that length, so a literal
/// geometry is parsed once per call and not once per row.
pub fn geo_array(value: &ColumnarValue, field: &FieldRef) -> Result<Arc<dyn GeoArrowArray>> {
    let array = to_array(value)?;
    from_arrow_array(array.as_ref(), field).map_err(to_df)
}

/// Materialize an argument as a plain Arrow array.
pub fn to_array(value: &ColumnarValue) -> Result<ArrayRef> {
    Ok(match value {
        ColumnarValue::Array(array) => Arc::clone(array),
        ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(1)?,
    })
}

/// Materialize an argument as a plain Arrow array of a known row count.
///
/// Use this when a function mixes scalar and column arguments and every side must line up.
pub fn to_array_of_size(value: &ColumnarValue, rows: usize) -> Result<ArrayRef> {
    Ok(match value {
        ColumnarValue::Array(array) => Arc::clone(array),
        ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(rows)?,
    })
}

/// Returns true when every argument is a constant.
pub fn all_scalar(args: &[ColumnarValue]) -> bool {
    args.iter()
        .all(|arg| matches!(arg, ColumnarValue::Scalar(_)))
}

/// Wrap a result array. It becomes a scalar when every input was a scalar.
pub fn wrap_result(array: ArrayRef, scalar_input: bool) -> Result<ColumnarValue> {
    if scalar_input && array.len() == 1 {
        Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
            array.as_ref(),
            0,
        )?))
    } else {
        Ok(ColumnarValue::Array(array))
    }
}

/// Wrap a GeoArrow result. It becomes a scalar when every input was a scalar.
pub fn wrap_geo_result(array: Arc<dyn GeoArrowArray>, scalar_input: bool) -> Result<ColumnarValue> {
    wrap_result(array.to_array_ref(), scalar_input)
}

/// An output field that holds the GeoArrow extension metadata of a geometry type.
pub fn geo_field(name: &str, data_type: &GeoArrowType) -> FieldRef {
    Arc::new(data_type.to_field(name, true))
}

/// Read a constant integer argument at plan time.
///
/// Returns `Ok(None)` when the argument is a column rather than a constant.
pub fn constant_i32(args: &ReturnFieldArgs, position: usize) -> Option<Option<i32>> {
    match args.scalar_arguments.get(position)? {
        Some(ScalarValue::Int32(value)) => Some(*value),
        Some(ScalarValue::Int64(value)) => Some(value.and_then(|v| i32::try_from(v).ok())),
        Some(ScalarValue::Null) => Some(None),
        Some(_) => None,
        None => None,
    }
}

/// Read an integer argument at run time. The argument must be a constant.
pub fn require_constant_i32(
    function: &str,
    argument: usize,
    value: &ColumnarValue,
) -> Result<Option<i32>> {
    match value {
        ColumnarValue::Scalar(ScalarValue::Int32(v)) => Ok(*v),
        ColumnarValue::Scalar(ScalarValue::Int64(v)) => {
            Ok(v.and_then(|inner| i32::try_from(inner).ok()))
        }
        ColumnarValue::Scalar(ScalarValue::Null) => Ok(None),
        _ => plan_err!(
            "{function} argument {} must be a constant integer",
            argument + 1
        ),
    }
}

/// Read an argument as `Float64`. Any other number type widens to it.
///
/// A geometry argument forces `Signature::any`, which does no coercion at all, so
/// `ST_Translate(geom, 10, 20)` arrives with `Int64` constants. This step keeps those
/// queries valid. It also keeps the plan-time geometry check that `Signature::any` buys.
pub fn as_f64(function: &str, argument: usize, array: &ArrayRef) -> Result<Float64Array> {
    if matches!(array.data_type(), DataType::Float64) {
        return Ok(array
            .as_primitive::<arrow_array::types::Float64Type>()
            .clone());
    }
    if !array.data_type().is_numeric() {
        return plan_err!(
            "{function} argument {} must be a number, got {}",
            argument + 1,
            array.data_type()
        );
    }
    let widened = arrow_cast::cast(array.as_ref(), &DataType::Float64)?;
    Ok(widened
        .as_primitive::<arrow_array::types::Float64Type>()
        .clone())
}

/// Read an argument as `Int32`. Any other integer type narrows to it.
///
/// Like [`as_f64`], this exists because a geometry argument forces `Signature::any`, which does no
/// coercion, so `ST_RemovePoint(line, 1)` arrives with an `Int64` literal.
pub fn as_i32(function: &str, argument: usize, array: &ArrayRef) -> Result<Int32Array> {
    if matches!(array.data_type(), DataType::Int32) {
        return Ok(array
            .as_primitive::<arrow_array::types::Int32Type>()
            .clone());
    }
    if !array.data_type().is_numeric() {
        return plan_err!(
            "{function} argument {} must be a number, got {}",
            argument + 1,
            array.data_type()
        );
    }
    let narrowed = arrow_cast::cast(array.as_ref(), &DataType::Int32)?;
    Ok(narrowed
        .as_primitive::<arrow_array::types::Int32Type>()
        .clone())
}

/// Cast an argument to `Utf8`.
pub fn as_utf8(function: &str, argument: usize, array: &ArrayRef) -> Result<StringArray> {
    match array.data_type() {
        DataType::Utf8 => Ok(array.as_string::<i32>().clone()),
        other => plan_err!(
            "{function} argument {} must be Utf8, got {other}",
            argument + 1
        ),
    }
}

/// Cast an argument to `Binary`.
pub fn as_binary(function: &str, argument: usize, array: &ArrayRef) -> Result<BinaryArray> {
    match array.data_type() {
        DataType::Binary => Ok(array.as_binary::<i32>().clone()),
        other => plan_err!(
            "{function} argument {} must be Binary, got {other}",
            argument + 1
        ),
    }
}
