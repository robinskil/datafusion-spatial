//! Structure edits and tessellation as DataFusion scalar UDFs.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;
use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array};
use arrow_schema::{DataType, FieldRef};
use datafusion::common::{plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion_spatial_kernels::edit::{
    dump_field, st_dump, st_snap_to_grid, structure, vertex_edit, Structure, VertexEdit,
};
use datafusion_spatial_kernels::process::output_type;
use datafusion_spatial_kernels::tessellate::{tessellate, Tessellation};
use geoarrow_schema::GeoArrowType;

use crate::util::{
    all_scalar, as_f64, as_i32, geo_array, geo_field, geo_type, to_array_of_size, to_df,
    wrap_geo_result,
};

/// `ST_Multi` or `ST_Points`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StructureUdf {
    edit: Structure,
    signature: Signature,
}

impl StructureUdf {
    /// Build the UDF for one edit.
    pub fn new(edit: Structure) -> Self {
        Self {
            edit,
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for StructureUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.edit.sql_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!(
            "{} needs the argument field to determine its return type",
            self.edit.function_name()
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type(self.edit.function_name(), 0, &args.arg_fields[0])?;
        Ok(geo_field(
            self.edit.sql_name(),
            &GeoArrowType::Geometry(output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = output_type(&array.data_type());
        let result = structure(array.as_ref(), self.edit, output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// `ST_DelaunayTriangles`, `ST_VoronoiPolygons` or `ST_VoronoiLines`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TessellateUdf {
    kind: Tessellation,
    signature: Signature,
}

impl TessellateUdf {
    /// Build the UDF for one tessellation.
    pub fn new(kind: Tessellation) -> Self {
        Self {
            kind,
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for TessellateUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.kind.sql_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!(
            "{} needs the argument field to determine its return type",
            self.kind.function_name()
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type(self.kind.function_name(), 0, &args.arg_fields[0])?;
        Ok(geo_field(
            self.kind.sql_name(),
            &GeoArrowType::Geometry(output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = output_type(&array.data_type());
        let result = tessellate(array.as_ref(), self.kind, output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// `ST_SnapToGrid(geom, size)`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StSnapToGrid {
    signature: Signature,
}

impl StSnapToGrid {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for StSnapToGrid {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StSnapToGrid {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_snaptogrid"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_SnapToGrid needs the argument field to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type("ST_SnapToGrid", 0, &args.arg_fields[0])?;
        Ok(geo_field(
            "st_snaptogrid",
            &GeoArrowType::Geometry(output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = output_type(&array.data_type());

        let rows = if matches!(args.args[1], ColumnarValue::Scalar(_)) {
            1
        } else {
            array.len()
        };
        let raw = to_array_of_size(&args.args[1], rows)?;
        let size: Float64Array = as_f64("ST_SnapToGrid", 1, &raw)?;

        let result = st_snap_to_grid(array.as_ref(), &size, output).map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// `ST_AddPoint`, `ST_RemovePoint` or `ST_SetPoint`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct VertexEditUdf {
    edit: VertexEdit,
    signature: Signature,
}

impl VertexEditUdf {
    /// Build the UDF for one edit.
    pub fn new(edit: VertexEdit) -> Self {
        // ST_AddPoint takes an optional position, the other two require one.
        let signature = match edit {
            VertexEdit::Add => Signature::one_of(
                vec![TypeSignature::Any(2), TypeSignature::Any(3)],
                Volatility::Immutable,
            ),
            VertexEdit::Remove => Signature::any(2, Volatility::Immutable),
            VertexEdit::Set => Signature::any(3, Volatility::Immutable),
        };
        Self { edit, signature }
    }
}

impl ScalarUDFImpl for VertexEditUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.edit.sql_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!(
            "{} needs the argument field to determine its return type",
            self.edit.function_name()
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let name = self.edit.function_name();
        let input = geo_type(name, 0, &args.arg_fields[0])?;
        if let Some(at) = point_argument(self.edit) {
            if let Some(field) = args.arg_fields.get(at) {
                geo_type(name, at, field)?;
            }
        }
        Ok(geo_field(
            self.edit.sql_name(),
            &GeoArrowType::Geometry(output_type(&input)),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let name = self.edit.function_name();
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let output = output_type(&array.data_type());

        // The argument order follows PostGIS, and it is not the same for all three:
        //   ST_AddPoint(line, point [, position])
        //   ST_RemovePoint(line, position)
        //   ST_SetPoint(line, position, point)
        let point = match point_argument(self.edit) {
            Some(at) => match args.args.get(at) {
                Some(value) => Some(geo_array(value, &args.arg_fields[at])?),
                None => None,
            },
            None => None,
        };
        let position_at = position_argument(self.edit);

        let position: Option<Int32Array> = match args.args.get(position_at) {
            Some(value) => {
                let rows = if matches!(value, ColumnarValue::Scalar(_)) {
                    1
                } else {
                    array.len()
                };
                let raw = to_array_of_size(value, rows)?;
                Some(as_i32(name, position_at, &raw)?)
            }
            None => None,
        };

        let result = vertex_edit(
            array.as_ref(),
            self.edit,
            point.as_deref(),
            position.as_ref(),
            output,
        )
        .map_err(to_df)?;
        wrap_geo_result(result, scalar_input)
    }
}

/// Which argument holds the point, if any.
///
/// PostGIS puts it second for `ST_AddPoint` and third for `ST_SetPoint`.
const fn point_argument(edit: VertexEdit) -> Option<usize> {
    match edit {
        VertexEdit::Add => Some(1),
        VertexEdit::Set => Some(2),
        VertexEdit::Remove => None,
    }
}

/// Which argument holds the position.
const fn position_argument(edit: VertexEdit) -> usize {
    match edit {
        VertexEdit::Add => 2,
        VertexEdit::Remove | VertexEdit::Set => 1,
    }
}

/// `ST_Dump`. Split a geometry into its parts, as a list.
///
/// See [`datafusion_spatial_kernels::edit`] for why this returns a list rather than a set.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StDump {
    signature: Signature,
}

impl StDump {
    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl Default for StDump {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StDump {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_dump"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        plan_err!("ST_Dump needs the argument field to determine its return type")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = geo_type("ST_Dump", 0, &args.arg_fields[0])?;
        let part = dump_field(&input);
        Ok(Arc::new(arrow_schema::Field::new(
            "st_dump",
            DataType::List(part),
            true,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;

        // Take the part field from the return field, so it matches what the planner promised.
        let DataType::List(part) = args.return_field.data_type() else {
            return plan_err!("ST_Dump must return a list");
        };

        let result = st_dump(array.as_ref(), Arc::clone(part)).map_err(to_df)?;

        // Always an array, never a scalar. A `ScalarValue` holds no child field. A collapse here
        // would change the element type of the list.
        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

/// Every structure, tessellation and edit function.
pub fn edits() -> Vec<ScalarUDF> {
    let mut functions: Vec<ScalarUDF> = Structure::ALL
        .into_iter()
        .map(|edit| ScalarUDF::new_from_impl(StructureUdf::new(edit)))
        .collect();
    functions.extend(
        Tessellation::ALL
            .into_iter()
            .map(|kind| ScalarUDF::new_from_impl(TessellateUdf::new(kind))),
    );
    functions.extend(
        VertexEdit::ALL
            .into_iter()
            .map(|edit| ScalarUDF::new_from_impl(VertexEditUdf::new(edit))),
    );
    functions.push(ScalarUDF::new_from_impl(StSnapToGrid::new()));
    functions.push(ScalarUDF::new_from_impl(StDump::new()));
    functions
}
