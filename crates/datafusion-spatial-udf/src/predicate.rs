//! Spatial predicates as DataFusion scalar UDFs.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;
use std::sync::Arc;

use arrow_array::{new_null_array, ArrayRef};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{plan_err, Result, ScalarValue};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion_spatial_kernels::predicate::{
    self, geometry_at, st_dfully_within, st_dwithin, st_predicate_scalar, st_predicate_with,
    Predicate, PredicateScratch, PreparedLiteral, Side,
};
use geoarrow_array::GeoArrowArray;

use crate::util::{all_scalar, check_same_crs, geo_array, geo_type, to_df, wrap_result};

/// Any of the eleven two-argument predicates.
///
/// One struct serves all of them. They differ only in which bounding box shortcut applies and
/// which `geo` algorithm answers the rows that shortcut cannot settle.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct PredicateUdf {
    predicate: Predicate,
    signature: Signature,
}

impl PredicateUdf {
    /// Build the UDF for one predicate.
    pub fn new(predicate: Predicate) -> Self {
        Self {
            predicate,
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for PredicateUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.predicate.sql_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    /// Check both arguments at plan time. The check covers the coordinate reference system.
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let name = self.predicate.function_name();
        let left = geo_type(name, 0, &args.arg_fields[0])?;
        let right = geo_type(name, 1, &args.arg_fields[1])?;
        check_same_crs(name, &left, &right)?;
        Ok(Arc::new(Field::new(
            self.predicate.sql_name(),
            DataType::Boolean,
            true,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let left_is_scalar = matches!(args.args[0], ColumnarValue::Scalar(_));
        let right_is_scalar = matches!(args.args[1], ColumnarValue::Scalar(_));

        let left = geo_array(&args.args[0], &args.arg_fields[0])?;
        let right = geo_array(&args.args[1], &args.arg_fields[1])?;
        let mut scratch = PredicateScratch::new();

        // A constant argument is prepared once and reused for the whole batch. `Side` keeps the
        // argument order of the query, which matters for ST_Contains and its relatives.
        let result: ArrayRef = if right_is_scalar && !left_is_scalar {
            prepared_against(
                left.as_ref(),
                right.as_ref(),
                self.predicate,
                Side::Right,
                &mut scratch,
            )?
        } else if left_is_scalar && !right_is_scalar {
            prepared_against(
                right.as_ref(),
                left.as_ref(),
                self.predicate,
                Side::Left,
                &mut scratch,
            )?
        } else {
            Arc::new(
                st_predicate_with(left.as_ref(), right.as_ref(), self.predicate, &mut scratch)
                    .map_err(to_df)?,
            )
        };

        wrap_result(result, scalar_input)
    }
}

/// Run the array against a one-row constant.
fn prepared_against(
    array: &dyn GeoArrowArray,
    constant: &dyn GeoArrowArray,
    predicate: Predicate,
    literal_side: Side,
    scratch: &mut PredicateScratch,
) -> Result<ArrayRef> {
    let Some(geometry) = geometry_at(constant, 0).map_err(to_df)? else {
        // A null constant makes every row null, as in PostGIS.
        return Ok(new_null_array(&DataType::Boolean, array.len()));
    };
    let literal = PreparedLiteral::new(geometry);
    Ok(Arc::new(
        st_predicate_scalar(array, &literal, predicate, literal_side, scratch).map_err(to_df)?,
    ))
}

/// `ST_DWithin` or `ST_DFullyWithin`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct DistancePredicateUdf {
    fully: bool,
    signature: Signature,
}

impl DistancePredicateUdf {
    /// Build the UDF. `fully` selects `ST_DFullyWithin` over `ST_DWithin`.
    pub fn new(fully: bool) -> Self {
        Self {
            fully,
            signature: Signature::any(3, Volatility::Immutable),
        }
    }

    const fn names(&self) -> (&'static str, &'static str) {
        if self.fully {
            ("st_dfullywithin", "ST_DFullyWithin")
        } else {
            ("st_dwithin", "ST_DWithin")
        }
    }
}

impl ScalarUDFImpl for DistancePredicateUdf {
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
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let (name, postgis_name) = self.names();
        let left = geo_type(postgis_name, 0, &args.arg_fields[0])?;
        let right = geo_type(postgis_name, 1, &args.arg_fields[1])?;
        check_same_crs(postgis_name, &left, &right)?;
        Ok(Arc::new(Field::new(name, DataType::Boolean, true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (_, postgis_name) = self.names();
        let scalar_input = all_scalar(&args.args);
        let rows = args.number_rows.max(1);

        let left = geo_array(&args.args[0], &args.arg_fields[0])?;
        let right = geo_array(&args.args[1], &args.arg_fields[1])?;

        // The radius grows the bounding box of one side, so it must be one value for the batch.
        let radius = match &args.args[2] {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => *value,
            ColumnarValue::Scalar(ScalarValue::Int64(Some(value))) => *value as f64,
            ColumnarValue::Scalar(ScalarValue::Float64(None) | ScalarValue::Null) => {
                return Ok(ColumnarValue::Array(new_null_array(
                    &DataType::Boolean,
                    rows,
                )))
            }
            _ => {
                return plan_err!(
                    "{postgis_name} needs a constant numeric distance as its third argument"
                )
            }
        };

        let mut scratch = PredicateScratch::new();
        let result = if self.fully {
            st_dfully_within(left.as_ref(), right.as_ref(), radius, &mut scratch)
        } else {
            st_dwithin(left.as_ref(), right.as_ref(), radius, &mut scratch)
        }
        .map_err(to_df)?;

        wrap_result(Arc::new(result), scalar_input)
    }
}

/// `ST_Relate(a, b)` returns the matrix. `ST_Relate(a, b, pattern)` returns a boolean.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StRelate {
    signature: Signature,
}

impl StRelate {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![TypeSignature::Any(2), TypeSignature::Any(3)],
                Volatility::Immutable,
            ),
        }
    }

    fn output_type(arg_count: usize) -> DataType {
        if arg_count >= 3 {
            DataType::Boolean
        } else {
            DataType::Utf8
        }
    }
}

impl Default for StRelate {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StRelate {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_relate"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(Self::output_type(arg_types.len()))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let left = geo_type("ST_Relate", 0, &args.arg_fields[0])?;
        let right = geo_type("ST_Relate", 1, &args.arg_fields[1])?;
        check_same_crs("ST_Relate", &left, &right)?;
        Ok(Arc::new(Field::new(
            "st_relate",
            Self::output_type(args.arg_fields.len()),
            true,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let left = geo_array(&args.args[0], &args.arg_fields[0])?;
        let right = geo_array(&args.args[1], &args.arg_fields[1])?;

        match args.args.get(2) {
            None => {
                let matrices =
                    predicate::st_relate(left.as_ref(), right.as_ref()).map_err(to_df)?;
                wrap_result(Arc::new(matrices), scalar_input)
            }
            Some(ColumnarValue::Scalar(ScalarValue::Utf8(Some(pattern)))) => {
                let matched = predicate::st_relate_pattern(left.as_ref(), right.as_ref(), pattern)
                    .map_err(to_df)?;
                wrap_result(Arc::new(matched), scalar_input)
            }
            Some(ColumnarValue::Scalar(ScalarValue::Utf8(None) | ScalarValue::Null)) => Ok(
                ColumnarValue::Array(new_null_array(&DataType::Boolean, args.number_rows.max(1))),
            ),
            Some(_) => plan_err!(
                "ST_Relate needs a constant nine character DE-9IM pattern as its third argument"
            ),
        }
    }
}

/// Every predicate function.
pub fn predicates() -> Vec<ScalarUDF> {
    let mut functions: Vec<ScalarUDF> = Predicate::ALL
        .into_iter()
        .map(|predicate| ScalarUDF::new_from_impl(PredicateUdf::new(predicate)))
        .collect();
    functions.push(ScalarUDF::new_from_impl(DistancePredicateUdf::new(false)));
    functions.push(ScalarUDF::new_from_impl(DistancePredicateUdf::new(true)));
    functions.push(ScalarUDF::new_from_impl(StRelate::new()));
    functions
}

/// `ST_Intersects`.
pub fn st_intersects() -> ScalarUDF {
    ScalarUDF::new_from_impl(PredicateUdf::new(Predicate::Intersects))
}
