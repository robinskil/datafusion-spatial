//! Accessor functions as DataFusion scalar UDFs.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;
use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{plan_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_spatial_kernels::accessor::{accepts_ordinate, ordinate, Ordinate};
use datafusion_spatial_kernels::{accessor, crs};

use crate::util::{all_scalar, geo_array, geo_type, to_df, wrap_result};

/// `ST_X`, `ST_Y`, `ST_Z` or `ST_M`. One struct serves all four.
///
/// Only the ordinate differs, and with separated coordinates each one is a buffer handoff.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct OrdinateUdf {
    ordinate: Ordinate,
    name: &'static str,
    signature: Signature,
}

impl OrdinateUdf {
    /// Build the UDF for one ordinate.
    pub fn new(ordinate: Ordinate) -> Self {
        let name = match ordinate {
            Ordinate::X => "st_x",
            Ordinate::Y => "st_y",
            Ordinate::Z => "st_z",
            Ordinate::M => "st_m",
        };
        Self {
            ordinate,
            name,
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for OrdinateUdf {
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
        Ok(DataType::Float64)
    }

    /// Validate the input at plan time.
    ///
    /// The argument field carries the GeoArrow extension metadata, so a `LineString` column is
    /// rejected before any batch is read.
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let data_type = geo_type(self.ordinate.function_name(), 0, &args.arg_fields[0])?;
        if !accepts_ordinate(&data_type) {
            return plan_err!(
                "{} requires a point argument, got {data_type:?}",
                self.ordinate.function_name()
            );
        }
        Ok(Arc::new(Field::new(self.name, DataType::Float64, true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let result = ordinate(array.as_ref(), self.ordinate).map_err(to_df)?;
        wrap_result(Arc::new(result), scalar_input)
    }
}

unary_geometry_udf!(
    /// `ST_GeometryType`. The PostGIS type name, such as `ST_Point`.
    StGeometryType,
    "st_geometrytype",
    "ST_GeometryType",
    DataType::Utf8,
    accessor::st_geometry_type
);

unary_geometry_udf!(
    /// `ST_Dimension`. The topological dimension: 0, 1 or 2.
    StDimension,
    "st_dimension",
    "ST_Dimension",
    DataType::Int32,
    accessor::st_dimension
);

unary_geometry_udf!(
    /// `ST_CoordDim`. The number of ordinates per coordinate.
    StCoordDim,
    "st_coorddim",
    "ST_CoordDim",
    DataType::Int32,
    accessor::st_coord_dim
);

unary_geometry_udf!(
    /// `ST_NPoints`. Every coordinate, at any depth.
    StNPoints,
    "st_npoints",
    "ST_NPoints",
    DataType::Int32,
    accessor::st_npoints
);

unary_geometry_udf!(
    /// `ST_NumPoints`. Line strings only, null otherwise.
    StNumPoints,
    "st_numpoints",
    "ST_NumPoints",
    DataType::Int32,
    accessor::st_num_points
);

unary_geometry_udf!(
    /// `ST_NumGeometries`. The part count of a collection.
    StNumGeometries,
    "st_numgeometries",
    "ST_NumGeometries",
    DataType::Int32,
    accessor::st_num_geometries
);

unary_geometry_udf!(
    /// `ST_NumInteriorRings`. The hole count of a polygon, null otherwise.
    StNumInteriorRings,
    "st_numinteriorrings",
    "ST_NumInteriorRings",
    DataType::Int32,
    accessor::st_num_interior_rings
);

unary_geometry_udf!(
    /// `ST_IsEmpty`. True when the geometry holds no coordinate.
    StIsEmpty,
    "st_isempty",
    "ST_IsEmpty",
    DataType::Boolean,
    accessor::st_is_empty
);

unary_geometry_udf!(
    /// `ST_IsClosed`. True when a lineal geometry starts and ends together.
    StIsClosed,
    "st_isclosed",
    "ST_IsClosed",
    DataType::Boolean,
    accessor::st_is_closed
);

unary_geometry_udf!(
    /// `ST_IsRing`. True for a closed line string that does not cross itself.
    StIsRing,
    "st_isring",
    "ST_IsRing",
    DataType::Boolean,
    accessor::st_is_ring
);

unary_geometry_udf!(
    /// `ST_IsSimple`. True when the geometry has no self intersection.
    StIsSimple,
    "st_issimple",
    "ST_IsSimple",
    DataType::Boolean,
    accessor::st_is_simple
);

unary_geometry_udf!(
    /// `ST_SRID`. Read from the column metadata, so every row shares one value.
    StSrid,
    "st_srid",
    "ST_SRID",
    DataType::Int32,
    crs::st_srid
);

/// `ST_X`.
pub fn st_x() -> ScalarUDF {
    ScalarUDF::new_from_impl(OrdinateUdf::new(Ordinate::X))
}

/// `ST_Y`.
pub fn st_y() -> ScalarUDF {
    ScalarUDF::new_from_impl(OrdinateUdf::new(Ordinate::Y))
}

/// `ST_Z`.
pub fn st_z() -> ScalarUDF {
    ScalarUDF::new_from_impl(OrdinateUdf::new(Ordinate::Z))
}

/// `ST_M`.
pub fn st_m() -> ScalarUDF {
    ScalarUDF::new_from_impl(OrdinateUdf::new(Ordinate::M))
}

/// Every accessor that needs no extra argument.
pub fn simple_accessors() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::new_from_impl(StGeometryType::new()),
        ScalarUDF::new_from_impl(StDimension::new()),
        ScalarUDF::new_from_impl(StCoordDim::new()),
        ScalarUDF::new_from_impl(StNPoints::new()),
        ScalarUDF::new_from_impl(StNumPoints::new()),
        ScalarUDF::new_from_impl(StNumGeometries::new()),
        ScalarUDF::new_from_impl(StNumInteriorRings::new()),
        ScalarUDF::new_from_impl(StIsEmpty::new()),
        ScalarUDF::new_from_impl(StIsClosed::new()),
        ScalarUDF::new_from_impl(StIsRing::new()),
        ScalarUDF::new_from_impl(StIsSimple::new()),
        ScalarUDF::new_from_impl(StSrid::new()),
    ]
}
