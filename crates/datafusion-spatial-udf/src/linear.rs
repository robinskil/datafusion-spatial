//! Linear reference as DataFusion scalar UDFs.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;
use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion_spatial_kernels::linear::{
    self, line_output_type, point_output_type, st_line_interpolate_point,
};
use geoarrow_schema::GeoArrowType;

use crate::util::{
    all_scalar, as_f64, check_same_crs, geo_array, geo_field, geo_type, to_array_of_size, to_df,
    wrap_geo_result, wrap_result,
};

/// `ST_ClosestPoint` or `ST_ShortestLine`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct NearestUdf {
    line: bool,
    signature: Signature,
}

impl NearestUdf {
    /// Build the UDF. `line` selects `ST_ShortestLine` over `ST_ClosestPoint`.
    pub fn new(line: bool) -> Self {
        Self {
            line,
            signature: Signature::any(2, Volatility::Immutable),
        }
    }

    const fn names(&self) -> (&'static str, &'static str) {
        if self.line {
            ("st_shortestline", "ST_ShortestLine")
        } else {
            ("st_closestpoint", "ST_ClosestPoint")
        }
    }

    fn output_for(&self, input: &GeoArrowType) -> GeoArrowType {
        if self.line {
            GeoArrowType::LineString(line_output_type(input))
        } else {
            GeoArrowType::Point(point_output_type(input))
        }
    }
}

impl ScalarUDFImpl for NearestUdf {
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
        plan_err!(
            "{} needs the argument fields to determine its return type",
            self.names().1
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let (name, postgis_name) = self.names();
        let left = geo_type(postgis_name, 0, &args.arg_fields[0])?;
        let right = geo_type(postgis_name, 1, &args.arg_fields[1])?;
        check_same_crs(postgis_name, &left, &right)?;
        Ok(geo_field(name, &self.output_for(&left)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let left = geo_array(&args.args[0], &args.arg_fields[0])?;
        let right = geo_array(&args.args[1], &args.arg_fields[1])?;
        let input_type = left.data_type();

        let result = if self.line {
            linear::st_shortest_line(left.as_ref(), right.as_ref(), line_output_type(&input_type))
        } else {
            linear::st_closest_point(
                left.as_ref(),
                right.as_ref(),
                point_output_type(&input_type),
            )
        }
        .map_err(to_df)?;

        wrap_geo_result(result, scalar_input)
    }
}

/// `ST_LineLocatePoint(line, point) -> double`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StLineLocatePoint {
    signature: Signature,
}

impl StLineLocatePoint {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for StLineLocatePoint {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StLineLocatePoint {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_linelocatepoint"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let left = geo_type("ST_LineLocatePoint", 0, &args.arg_fields[0])?;
        let right = geo_type("ST_LineLocatePoint", 1, &args.arg_fields[1])?;
        check_same_crs("ST_LineLocatePoint", &left, &right)?;
        Ok(Arc::new(Field::new(
            "st_linelocatepoint",
            DataType::Float64,
            true,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let line = geo_array(&args.args[0], &args.arg_fields[0])?;
        let point = geo_array(&args.args[1], &args.arg_fields[1])?;
        let result = linear::st_line_locate_point(line.as_ref(), point.as_ref()).map_err(to_df)?;
        wrap_result(Arc::new(result), scalar_input)
    }
}

/// `ST_LineInterpolatePoint(line, fraction) -> point`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StLineInterpolatePoint {
    signature: Signature,
}

impl StLineInterpolatePoint {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(vec![TypeSignature::Any(2)], Volatility::Immutable),
        }
    }
}

impl Default for StLineInterpolatePoint {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StLineInterpolatePoint {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_lineinterpolatepoint"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_LineInterpolatePoint needs the argument field to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type("ST_LineInterpolatePoint", 0, &args.arg_fields[0])?;
        Ok(geo_field(
            "st_lineinterpolatepoint",
            &GeoArrowType::Point(point_output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let line = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = point_output_type(&line.data_type());

        let raw = to_array_of_size(&args.args[1], line.len())?;
        let fraction = as_f64("ST_LineInterpolatePoint", 1, &raw)?;

        let result = st_line_interpolate_point(line.as_ref(), &fraction, output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// Every linear reference function.
pub fn linear_functions() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::new_from_impl(NearestUdf::new(false)),
        ScalarUDF::new_from_impl(NearestUdf::new(true)),
        ScalarUDF::new_from_impl(StLineLocatePoint::new()),
        ScalarUDF::new_from_impl(StLineInterpolatePoint::new()),
        ScalarUDF::new_from_impl(StProject::new()),
    ]
}

/// `ST_Project(point, distance, azimuth)`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StProject {
    signature: Signature,
}

impl StProject {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(3, Volatility::Immutable),
        }
    }
}

impl Default for StProject {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StProject {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_project"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_Project needs the argument field to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type("ST_Project", 0, &args.arg_fields[0])?;
        Ok(geo_field(
            "st_project",
            &GeoArrowType::Point(point_output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = point_output_type(&array.data_type());

        let mut numbers = Vec::with_capacity(2);
        for position in 1..=2 {
            let rows = if matches!(args.args[position], ColumnarValue::Scalar(_)) {
                1
            } else {
                array.len()
            };
            let raw = to_array_of_size(&args.args[position], rows)?;
            numbers.push(as_f64("ST_Project", position, &raw)?);
        }

        let result =
            linear::st_project(array.as_ref(), &numbers[0], &numbers[1], output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}
