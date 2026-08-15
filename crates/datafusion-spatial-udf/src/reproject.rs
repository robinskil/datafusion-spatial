//! `ST_Transform` as a DataFusion scalar UDF.
//!
//! Only present when the `proj` feature is on.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;

use arrow_schema::{DataType, FieldRef};
use datafusion::common::{plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_spatial_kernels::crs::srid_of;
use datafusion_spatial_kernels::reproject::{output_type, st_transform_with, transformation};

use crate::util::{
    all_scalar, constant_i32, geo_array, geo_field, geo_type, require_constant_i32, to_df,
    wrap_geo_result,
};

/// `ST_Transform(geom, srid)`.
///
/// The target SRID must be a constant. It changes the type of the output column, exactly as it
/// does for `ST_SetSRID`, so a per-row value cannot be represented.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StTransform {
    signature: Signature,
}

impl StTransform {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for StTransform {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StTransform {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_transform"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_Transform needs the argument fields to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type("ST_Transform", 0, &args.arg_fields[0])?;

        let Some(target) = constant_i32(&args, 1) else {
            return plan_err!(
                "ST_Transform needs a constant target SRID. GeoArrow stores the coordinate \
                 reference system once per column, so it cannot vary by row."
            );
        };
        let Some(target) = target else {
            return plan_err!("ST_Transform needs a non-null target SRID");
        };

        // Fail here rather than on the first batch: the source SRID is a property of the schema,
        // so a missing or unconvertible one is a plan-time error, not a data error.
        let source = srid_of(&input);
        transformation(source, target).map_err(to_df)?;

        Ok(geo_field("st_transform", &output_type(&input, target)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let target = require_constant_i32("ST_Transform", 1, &args.args[1])?.ok_or_else(|| {
            datafusion::common::DataFusionError::Plan(
                "ST_Transform needs a non-null target SRID".to_string(),
            )
        })?;

        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let source = srid_of(&array.data_type());

        // One transformation for the whole batch. It cannot live longer: `Proj` holds a raw
        // context pointer and is neither `Send` nor `Sync`.
        let projection = transformation(source, target).map_err(to_df)?;
        let result = st_transform_with(array.as_ref(), &projection, target).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// Every reprojection function.
pub fn reprojections() -> Vec<ScalarUDF> {
    vec![ScalarUDF::new_from_impl(StTransform::new())]
}
