//! Bounding box functions as DataFusion scalar UDFs.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;
use std::sync::Arc;

use arrow_array::Float64Array;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_spatial_kernels::envelope::{
    bound, box_output_type, st_bbox_intersects, st_envelope, st_expand, Bound,
};
use geoarrow_schema::GeoArrowType;

use crate::util::{
    all_scalar, as_f64, check_same_crs, geo_array, geo_field, geo_type, to_array_of_size, to_df,
    wrap_geo_result, wrap_result,
};

/// `ST_XMin`, `ST_YMin`, `ST_ZMin`, `ST_XMax`, `ST_YMax` or `ST_ZMax`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BoundUdf {
    which: Bound,
    signature: Signature,
}

impl BoundUdf {
    /// Build the UDF for one corner ordinate.
    pub fn new(which: Bound) -> Self {
        Self {
            which,
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for BoundUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.which.sql_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        geo_type(self.which.function_name(), 0, &args.arg_fields[0])?;
        Ok(Arc::new(Field::new(
            self.which.sql_name(),
            DataType::Float64,
            true,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let result = bound(array.as_ref(), self.which).map_err(to_df)?;
        wrap_result(Arc::new(result), scalar_input)
    }
}

/// `ST_Envelope`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StEnvelope {
    signature: Signature,
}

impl StEnvelope {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl Default for StEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StEnvelope {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_envelope"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_Envelope needs the argument field to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type("ST_Envelope", 0, &args.arg_fields[0])?;
        Ok(geo_field(
            "st_envelope",
            &GeoArrowType::Rect(box_output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = box_output_type(&array.data_type());
        let result = st_envelope(array.as_ref(), output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// `ST_Expand(geom, distance)`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StExpand {
    signature: Signature,
}

impl StExpand {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for StExpand {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StExpand {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_expand"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_Expand needs the argument field to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type("ST_Expand", 0, &args.arg_fields[0])?;
        Ok(geo_field(
            "st_expand",
            &GeoArrowType::Rect(box_output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = box_output_type(&array.data_type());

        // A constant distance stays one value. The crate does not expand it per row.
        let rows = if matches!(args.args[1], ColumnarValue::Scalar(_)) {
            1
        } else {
            array.len()
        };
        let raw = to_array_of_size(&args.args[1], rows)?;
        let distance: Float64Array = as_f64("ST_Expand", 1, &raw)?;

        let result = st_expand(array.as_ref(), &distance, output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// `ST_BBoxIntersects`, the PostGIS `&&` operator.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StBBoxIntersects {
    signature: Signature,
}

impl StBBoxIntersects {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for StBBoxIntersects {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StBBoxIntersects {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_bboxintersects"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let left = geo_type("ST_BBoxIntersects", 0, &args.arg_fields[0])?;
        let right = geo_type("ST_BBoxIntersects", 1, &args.arg_fields[1])?;
        check_same_crs("ST_BBoxIntersects", &left, &right)?;
        Ok(Arc::new(Field::new(
            "st_bboxintersects",
            DataType::Boolean,
            true,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let left = geo_array(&args.args[0], &args.arg_fields[0])?;
        let right = geo_array(&args.args[1], &args.arg_fields[1])?;
        let result = st_bbox_intersects(left.as_ref(), right.as_ref()).map_err(to_df)?;
        wrap_result(Arc::new(result), scalar_input)
    }
}

/// Every bounding box function.
pub fn envelopes() -> Vec<ScalarUDF> {
    let mut functions: Vec<ScalarUDF> = Bound::ALL
        .into_iter()
        .map(|which| ScalarUDF::new_from_impl(BoundUdf::new(which)))
        .collect();
    functions.push(ScalarUDF::new_from_impl(StEnvelope::new()));
    functions.push(ScalarUDF::new_from_impl(StExpand::new()));
    functions.push(ScalarUDF::new_from_impl(StBBoxIntersects::new()));
    functions
}
