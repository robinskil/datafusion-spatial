//! Constructors as DataFusion scalar UDFs.
//!
//! `ST_MakePoint` and `ST_MakeEnvelope` adopt their input buffers. A geometry column built from
//! ordinate columns therefore copies nothing.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;
use std::sync::Arc;

use arrow_schema::{DataType, FieldRef};
use datafusion::common::{plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion_spatial_kernels::constructor;
use geoarrow_schema::{Dimension, GeoArrowType};

use crate::util::{
    all_scalar, as_f64, geo_array, geo_field, geo_type, to_array_of_size, to_df, wrap_geo_result,
};

/// `ST_Point`, `ST_MakePoint` and `ST_PointZ`.
///
/// Two arguments give XY, three give XYZ. Both shapes adopt the input buffers.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StMakePoint {
    name: &'static str,
    signature: Signature,
}

impl StMakePoint {
    /// Build the UDF under one of its PostGIS names.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            signature: Signature::one_of(
                vec![
                    TypeSignature::Uniform(2, vec![DataType::Float64]),
                    TypeSignature::Uniform(3, vec![DataType::Float64]),
                ],
                Volatility::Immutable,
            ),
        }
    }

    fn output_type(arg_count: usize) -> GeoArrowType {
        let dim = if arg_count >= 3 {
            Dimension::XYZ
        } else {
            Dimension::XY
        };
        GeoArrowType::Point(constructor::point_type(dim, Default::default()))
    }
}

impl ScalarUDFImpl for StMakePoint {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(Self::output_type(arg_types.len()).to_data_type())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(geo_field(
            self.name,
            &Self::output_type(args.arg_fields.len()),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let rows = args.number_rows.max(1);

        let x_raw = to_array_of_size(&args.args[0], rows)?;
        let y_raw = to_array_of_size(&args.args[1], rows)?;
        let x = as_f64("ST_MakePoint", 0, &x_raw)?;
        let y = as_f64("ST_MakePoint", 1, &y_raw)?;

        let z_raw = match args.args.get(2) {
            Some(value) => Some(to_array_of_size(value, rows)?),
            None => None,
        };
        let z = match &z_raw {
            Some(raw) => Some(as_f64("ST_MakePoint", 2, raw)?),
            None => None,
        };

        let array =
            constructor::st_make_point(&x, &y, z.as_ref(), Default::default()).map_err(to_df)?;
        wrap_geo_result(Arc::new(array), scalar_input)
    }
}

/// `ST_MakeEnvelope` and `ST_MakeBox2D`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StMakeEnvelope {
    name: &'static str,
    signature: Signature,
}

impl StMakeEnvelope {
    /// Build the UDF under one of its PostGIS names.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            signature: Signature::uniform(4, vec![DataType::Float64], Volatility::Immutable),
        }
    }

    fn output_type() -> GeoArrowType {
        GeoArrowType::Rect(constructor::box_type(Default::default()))
    }
}

impl ScalarUDFImpl for StMakeEnvelope {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(Self::output_type().to_data_type())
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(geo_field(self.name, &Self::output_type()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let rows = args.number_rows.max(1);

        let mut ordinates = Vec::with_capacity(4);
        for (position, value) in args.args.iter().enumerate().take(4) {
            let raw = to_array_of_size(value, rows)?;
            ordinates.push(as_f64("ST_MakeEnvelope", position, &raw)?);
        }

        let array = constructor::st_make_envelope(
            &ordinates[0],
            &ordinates[1],
            &ordinates[2],
            &ordinates[3],
            Default::default(),
        )
        .map_err(to_df)?;
        wrap_geo_result(Arc::new(array), scalar_input)
    }
}

/// `ST_MakePolygon`. Close a line string into a polygon with no holes.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StMakePolygon {
    signature: Signature,
}

impl StMakePolygon {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl Default for StMakePolygon {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StMakePolygon {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_makepolygon"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_MakePolygon needs the argument field to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type("ST_MakePolygon", 0, &args.arg_fields[0])?;
        // A mixed or serialized column carries an unknown type per row, so it is accepted and
        // checked row by row. Only a wrong *static* type is a plan-time error.
        let untyped = datafusion_spatial_kernels::accessor::is_untyped(&input);
        if !untyped && !matches!(input, GeoArrowType::LineString(_)) {
            return plan_err!("ST_MakePolygon needs a line string argument, got {input:?}");
        }
        Ok(geo_field(
            "st_makepolygon",
            &GeoArrowType::Polygon(constructor::polygon_type_for(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = constructor::polygon_type_for(&array.data_type());
        let result = constructor::st_make_polygon(array.as_ref(), output).map_err(to_df)?;
        wrap_geo_result(Arc::new(result), scalar_input)
    }
}

/// `ST_MakeLine`. Join two point columns into two point line strings.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StMakeLine {
    signature: Signature,
}

impl StMakeLine {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for StMakeLine {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StMakeLine {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_makeline"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_MakeLine needs the argument fields to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let start = geo_type("ST_MakeLine", 0, &args.arg_fields[0])?;
        let end = geo_type("ST_MakeLine", 1, &args.arg_fields[1])?;
        crate::util::check_same_crs("ST_MakeLine", &start, &end)?;
        if !matches!(start, GeoArrowType::Point(_)) || !matches!(end, GeoArrowType::Point(_)) {
            return plan_err!("ST_MakeLine needs two point arguments");
        }
        Ok(geo_field(
            "st_makeline",
            &GeoArrowType::LineString(constructor::line_string_type_for(&start)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let start = geo_array(&args.args[0], &args.arg_fields[0])?;
        let end = geo_array(&args.args[1], &args.arg_fields[1])?;
        let output = constructor::line_string_type_for(&start.data_type());
        let result =
            constructor::st_make_line(start.as_ref(), end.as_ref(), output).map_err(to_df)?;
        wrap_geo_result(Arc::new(result), scalar_input)
    }
}

/// Every constructor function.
///
/// `ST_Point`, `ST_MakePoint` and `ST_PointZ` are the same implementation under three names, and
/// `ST_MakeBox2D` is `ST_MakeEnvelope` under a second name. PostGIS has the same aliases.
pub fn constructors() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::new_from_impl(StMakePoint::new("st_point")),
        ScalarUDF::new_from_impl(StMakePoint::new("st_makepoint")),
        ScalarUDF::new_from_impl(StMakePoint::new("st_pointz")),
        ScalarUDF::new_from_impl(StMakeEnvelope::new("st_makeenvelope")),
        ScalarUDF::new_from_impl(StMakeEnvelope::new("st_makebox2d")),
        ScalarUDF::new_from_impl(StMakePolygon::new()),
        ScalarUDF::new_from_impl(StMakeLine::new()),
    ]
}
