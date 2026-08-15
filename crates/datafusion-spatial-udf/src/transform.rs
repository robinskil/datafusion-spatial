//! Coordinate transforms and `ST_SetSRID` as DataFusion scalar UDFs.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;

use arrow_schema::{DataType, FieldRef};
use datafusion::common::{plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion_spatial_kernels::{crs, transform};
use geoarrow_schema::error::GeoArrowResult;
use geoarrow_schema::{Dimension, GeoArrowType};

use crate::util::{
    all_scalar, constant_i32, geo_array, geo_field, geo_type, require_constant_i32, to_df,
    wrap_geo_result,
};

unary_transform_udf!(
    /// `ST_FlipCoordinates`. Swap x and y. Zero copy on separated coordinates.
    StFlipCoordinates,
    "st_flipcoordinates",
    "ST_FlipCoordinates",
    (|input| Ok(transform::flipped_type(input))) as fn(&GeoArrowType) -> GeoArrowResult<GeoArrowType>,
    transform::st_flip_coordinates
);

unary_transform_udf!(
    /// `ST_Force2D`. Drop z and m. Zero copy on separated coordinates.
    StForce2D,
    "st_force2d",
    "ST_Force2D",
    (|input| transform::forced_type(input, Dimension::XY))
        as fn(&GeoArrowType) -> GeoArrowResult<GeoArrowType>,
    transform::st_force_2d
);

unary_transform_udf!(
    /// `ST_Force3D`. Add a zero z where one is missing.
    StForce3D,
    "st_force3d",
    "ST_Force3D",
    (|input| transform::forced_type(input, Dimension::XYZ))
        as fn(&GeoArrowType) -> GeoArrowResult<GeoArrowType>,
    transform::st_force_3d
);

/// `ST_SetSRID(geometry, srid)`.
///
/// The SRID lives in the column metadata, not in each value, so the second argument must be a
/// constant. A per-row SRID cannot be represented and is rejected at plan time rather than
/// silently dropped.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StSetSrid {
    signature: Signature,
}

impl StSetSrid {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(vec![TypeSignature::Any(2)], Volatility::Immutable),
        }
    }
}

impl Default for StSetSrid {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StSetSrid {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_setsrid"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_SetSRID needs the argument fields to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type("ST_SetSRID", 0, &args.arg_fields[0])?;
        let Some(srid) = constant_i32(&args, 1) else {
            return plan_err!(
                "ST_SetSRID needs a constant SRID. GeoArrow stores the coordinate reference \
                 system once per column, so it cannot vary by row."
            );
        };
        let output = crs::with_srid(&input, srid.unwrap_or(0));
        Ok(geo_field("st_setsrid", &output))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let srid = require_constant_i32("ST_SetSRID", 1, &args.args[1])?.unwrap_or(0);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let result = crs::st_set_srid(array.as_ref(), srid).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// Every transform function.
pub fn transforms() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::new_from_impl(StFlipCoordinates::new()),
        ScalarUDF::new_from_impl(StForce2D::new()),
        ScalarUDF::new_from_impl(StForce3D::new()),
        ScalarUDF::new_from_impl(StSetSrid::new()),
    ]
}
