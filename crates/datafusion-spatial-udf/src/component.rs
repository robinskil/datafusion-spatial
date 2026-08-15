//! Component extraction as DataFusion scalar UDFs.
//!
//! `ST_PointN`, `ST_InteriorRingN` and `ST_GeometryN` take an index. The index may be a constant
//! or a column, and a constant is kept as a scalar rather than expanded to one value per row.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;

use arrow_array::Int32Array;
use arrow_schema::{DataType, FieldRef};
use datafusion::common::{plan_err, Result, ScalarValue};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_spatial_kernels::component::{
    self, geometry_output_type, line_string_output_type, point_output_type, Index, LineEnd,
};
use geoarrow_schema::GeoArrowType;

use crate::util::{
    all_scalar, as_i32, geo_array, geo_field, geo_type, to_array_of_size, to_df, wrap_geo_result,
};

/// `ST_StartPoint` or `ST_EndPoint`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct LineEndUdf {
    end: LineEnd,
    name: &'static str,
    postgis_name: &'static str,
    signature: Signature,
}

impl LineEndUdf {
    /// Build the UDF for one end of a line string.
    pub fn new(end: LineEnd) -> Self {
        let (name, postgis_name) = match end {
            LineEnd::Start => ("st_startpoint", "ST_StartPoint"),
            LineEnd::End => ("st_endpoint", "ST_EndPoint"),
        };
        Self {
            end,
            name,
            postgis_name,
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for LineEndUdf {
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
        plan_err!(
            "{} needs the argument field to determine its return type",
            self.postgis_name
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type(self.postgis_name, 0, &args.arg_fields[0])?;
        Ok(geo_field(
            self.name,
            &GeoArrowType::Point(point_output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = point_output_type(&array.data_type());
        let result = component::st_line_end(array.as_ref(), self.end, output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// Which indexed component to pull out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexedComponent {
    /// `ST_PointN`.
    PointN,
    /// `ST_InteriorRingN`.
    InteriorRingN,
    /// `ST_GeometryN`.
    GeometryN,
}

impl IndexedComponent {
    const fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::PointN => ("st_pointn", "ST_PointN"),
            Self::InteriorRingN => ("st_interiorringn", "ST_InteriorRingN"),
            Self::GeometryN => ("st_geometryn", "ST_GeometryN"),
        }
    }

    fn output_type(self, input: &GeoArrowType) -> GeoArrowType {
        match self {
            Self::PointN => GeoArrowType::Point(point_output_type(input)),
            Self::InteriorRingN => GeoArrowType::LineString(line_string_output_type(input)),
            Self::GeometryN => GeoArrowType::Geometry(geometry_output_type(input)),
        }
    }
}

/// `ST_PointN`, `ST_InteriorRingN` or `ST_GeometryN`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct IndexedComponentUdf {
    component: IndexedComponent,
    signature: Signature,
}

impl IndexedComponentUdf {
    /// Build the UDF for one indexed component.
    pub fn new(component: IndexedComponent) -> Self {
        Self {
            component,
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for IndexedComponentUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.component.names().0
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!(
            "{} needs the argument fields to determine its return type",
            self.component.names().1
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let (name, postgis_name) = self.component.names();
        let input = geo_type(postgis_name, 0, &args.arg_fields[0])?;
        Ok(geo_field(name, &self.component.output_type(&input)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (_, postgis_name) = self.component.names();
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let input_type = array.data_type();

        // A constant index stays a scalar. Only a column index is materialized.
        let per_row: Option<Int32Array> = match &args.args[1] {
            ColumnarValue::Scalar(_) => None,
            ColumnarValue::Array(_) => {
                let raw = to_array_of_size(&args.args[1], array.len())?;
                Some(as_i32(postgis_name, 1, &raw)?)
            }
        };

        let index = match (&per_row, &args.args[1]) {
            (Some(values), _) => Index::PerRow(values),
            (None, ColumnarValue::Scalar(ScalarValue::Int32(value))) => Index::Constant(*value),
            (None, ColumnarValue::Scalar(ScalarValue::Int64(value))) => {
                Index::Constant(value.and_then(|v| i32::try_from(v).ok()))
            }
            (None, ColumnarValue::Scalar(ScalarValue::Null)) => Index::Constant(None),
            (None, other) => {
                return plan_err!("{postgis_name} index must be an integer, got {other:?}")
            }
        };

        let result = match self.component {
            IndexedComponent::PointN => {
                component::st_point_n(array.as_ref(), index, point_output_type(&input_type))
            }
            IndexedComponent::InteriorRingN => component::st_interior_ring_n(
                array.as_ref(),
                index,
                line_string_output_type(&input_type),
            ),
            IndexedComponent::GeometryN => {
                component::st_geometry_n(array.as_ref(), index, geometry_output_type(&input_type))
            }
        }
        .map_err(to_df)?;

        wrap_geo_result(result, scalar_input)
    }
}

/// `ST_ExteriorRing`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StExteriorRing {
    signature: Signature,
}

impl StExteriorRing {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl Default for StExteriorRing {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StExteriorRing {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_exteriorring"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_ExteriorRing needs the argument field to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type("ST_ExteriorRing", 0, &args.arg_fields[0])?;
        Ok(geo_field(
            "st_exteriorring",
            &GeoArrowType::LineString(line_string_output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = line_string_output_type(&array.data_type());
        let result = component::st_exterior_ring(array.as_ref(), output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// Every component function.
pub fn components() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::new_from_impl(LineEndUdf::new(LineEnd::Start)),
        ScalarUDF::new_from_impl(LineEndUdf::new(LineEnd::End)),
        ScalarUDF::new_from_impl(StExteriorRing::new()),
        ScalarUDF::new_from_impl(IndexedComponentUdf::new(IndexedComponent::PointN)),
        ScalarUDF::new_from_impl(IndexedComponentUdf::new(IndexedComponent::InteriorRingN)),
        ScalarUDF::new_from_impl(IndexedComponentUdf::new(IndexedComponent::GeometryN)),
    ]
}
