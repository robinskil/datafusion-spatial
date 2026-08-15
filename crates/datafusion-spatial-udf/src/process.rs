//! Process, overlay and affine transforms as DataFusion scalar UDFs.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;
use std::sync::Arc;

use arrow_array::{Array, Float64Array};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion_spatial_kernels::affine::{affine, Affine};
use datafusion_spatial_kernels::process::{
    self, output_type, overlay, shape, sized_shape, Overlay, Shape, Sized,
};
use geoarrow_schema::GeoArrowType;

use crate::util::{
    all_scalar, as_f64, check_same_crs, geo_array, geo_field, geo_type, to_array_of_size, to_df,
    wrap_geo_result, wrap_result,
};

/// `ST_Union`, `ST_Intersection`, `ST_Difference` or `ST_SymDifference`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct OverlayUdf {
    operation: Overlay,
    signature: Signature,
}

impl OverlayUdf {
    /// Build the UDF for one overlay operation.
    pub fn new(operation: Overlay) -> Self {
        Self {
            operation,
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for OverlayUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.operation.sql_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!(
            "{} needs the argument fields to determine its return type",
            self.operation.function_name()
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let name = self.operation.function_name();
        let left = geo_type(name, 0, &args.arg_fields[0])?;
        let right = geo_type(name, 1, &args.arg_fields[1])?;
        check_same_crs(name, &left, &right)?;
        Ok(geo_field(
            self.operation.sql_name(),
            &GeoArrowType::Geometry(output_type(&left)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let left = geo_array(&args.args[0], &args.arg_fields[0])?;
        let right = geo_array(&args.args[1], &args.arg_fields[1])?;
        let output = output_type(&left.data_type());
        let result =
            overlay(left.as_ref(), right.as_ref(), self.operation, output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// A one-argument shape transform such as `ST_ConvexHull` or `ST_Centroid`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ShapeUdf {
    transform: Shape,
    signature: Signature,
}

impl ShapeUdf {
    /// Build the UDF for one transform.
    pub fn new(transform: Shape) -> Self {
        Self {
            transform,
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for ShapeUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.transform.sql_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!(
            "{} needs the argument field to determine its return type",
            self.transform.function_name()
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type(self.transform.function_name(), 0, &args.arg_fields[0])?;
        Ok(geo_field(
            self.transform.sql_name(),
            &GeoArrowType::Geometry(output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = output_type(&array.data_type());
        let result = shape(array.as_ref(), self.transform, output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// A shape transform that also takes a distance or tolerance, such as `ST_Buffer`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SizedShapeUdf {
    transform: Sized,
    signature: Signature,
}

impl SizedShapeUdf {
    /// Build the UDF for one transform.
    pub fn new(transform: Sized) -> Self {
        Self {
            transform,
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SizedShapeUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.transform.sql_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!(
            "{} needs the argument fields to determine its return type",
            self.transform.function_name()
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type(self.transform.function_name(), 0, &args.arg_fields[0])?;
        Ok(geo_field(
            self.transform.sql_name(),
            &GeoArrowType::Geometry(output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = output_type(&array.data_type());

        // A constant argument stays one value. The crate does not expand it per row.
        let rows = if matches!(args.args[1], ColumnarValue::Scalar(_)) {
            1
        } else {
            array.len()
        };
        let raw = to_array_of_size(&args.args[1], rows)?;
        let parameter: Float64Array = as_f64(self.transform.function_name(), 1, &raw)?;

        let result =
            sized_shape(array.as_ref(), self.transform, &parameter, output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// `ST_Translate`, `ST_Scale`, `ST_Rotate` or `ST_Affine`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct AffineUdf {
    transform: Affine,
    signature: Signature,
}

impl AffineUdf {
    /// Build the UDF for one affine transform.
    pub fn new(transform: Affine) -> Self {
        Self {
            transform,
            signature: Signature::one_of(
                vec![TypeSignature::Any(1 + transform.parameter_count())],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for AffineUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.transform.sql_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!(
            "{} needs the argument field to determine its return type",
            self.transform.function_name()
        )
    }

    /// An affine transform keeps the input geometry type exactly.
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type(self.transform.function_name(), 0, &args.arg_fields[0])?;
        Ok(geo_field(self.transform.sql_name(), &input))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let name = self.transform.function_name();
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;

        // One matrix for the whole batch, so the parameters must be constant. A per-row matrix
        // would rebuild the transform for every coordinate and is not what PostGIS offers either.
        let mut parameters = Vec::with_capacity(self.transform.parameter_count());
        for position in 1..=self.transform.parameter_count() {
            let raw = to_array_of_size(&args.args[position], 1)?;
            let values = as_f64(name, position, &raw)?;
            if values.is_null(0) {
                return Ok(ColumnarValue::Array(arrow_array::new_null_array(
                    &args.return_field.data_type().clone(),
                    args.number_rows,
                )));
            }
            parameters.push(values.value(0));
        }

        let matrix = self.transform.matrix(&parameters);
        let result = affine(array.as_ref(), &matrix).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// `ST_IsValid`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StIsValid {
    reason: bool,
    signature: Signature,
}

impl StIsValid {
    /// Build the UDF. `reason` selects `ST_IsValidReason` over `ST_IsValid`.
    pub fn new(reason: bool) -> Self {
        Self {
            reason,
            signature: Signature::any(1, Volatility::Immutable),
        }
    }

    const fn names(&self) -> (&'static str, &'static str) {
        if self.reason {
            ("st_isvalidreason", "ST_IsValidReason")
        } else {
            ("st_isvalid", "ST_IsValid")
        }
    }

    fn output(&self) -> DataType {
        if self.reason {
            DataType::Utf8
        } else {
            DataType::Boolean
        }
    }
}

impl ScalarUDFImpl for StIsValid {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.names().0
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(self.output())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let (name, postgis_name) = self.names();
        geo_type(postgis_name, 0, &args.arg_fields[0])?;
        Ok(Arc::new(Field::new(name, self.output(), true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;

        if self.reason {
            let result = process::st_is_valid_reason(array.as_ref()).map_err(to_df)?;
            wrap_result(Arc::new(result), scalar_input)
        } else {
            let result = process::st_is_valid(array.as_ref()).map_err(to_df)?;
            wrap_result(Arc::new(result), scalar_input)
        }
    }
}

/// Every process, overlay and affine function.
pub fn processing() -> Vec<ScalarUDF> {
    let mut functions: Vec<ScalarUDF> = Overlay::ALL
        .into_iter()
        .map(|operation| ScalarUDF::new_from_impl(OverlayUdf::new(operation)))
        .collect();
    functions.extend(
        Shape::ALL
            .into_iter()
            .map(|transform| ScalarUDF::new_from_impl(ShapeUdf::new(transform))),
    );
    functions.extend(
        Sized::ALL
            .into_iter()
            .map(|transform| ScalarUDF::new_from_impl(SizedShapeUdf::new(transform))),
    );
    functions.extend(
        Affine::ALL
            .into_iter()
            .map(|transform| ScalarUDF::new_from_impl(AffineUdf::new(transform))),
    );
    functions.push(ScalarUDF::new_from_impl(StIsValid::new(false)));
    functions.push(ScalarUDF::new_from_impl(StIsValid::new(true)));
    functions
}
