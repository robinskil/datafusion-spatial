//! Text and binary conversions as DataFusion scalar UDFs.

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
use datafusion_spatial_kernels::io;
use geoarrow_schema::GeoArrowType;

use crate::util::{
    all_scalar, as_binary, as_utf8, geo_array, geo_field, geo_type, require_constant_i32, to_array,
    to_df, wrap_geo_result, wrap_result,
};

unary_geometry_udf!(
    /// `ST_AsText`. Well-Known Text.
    StAsText,
    "st_astext",
    "ST_AsText",
    DataType::Utf8,
    io::st_as_text
);

unary_geometry_udf!(
    /// `ST_AsBinary`. Well-Known Binary.
    StAsBinary,
    "st_asbinary",
    "ST_AsBinary",
    DataType::Binary,
    io::st_as_binary
);

unary_geometry_udf!(
    /// `ST_AsEWKB`. Well-Known Binary with the column SRID in the header.
    StAsEwkb,
    "st_asewkb",
    "ST_AsEWKB",
    DataType::Binary,
    io::st_as_ewkb
);

unary_geometry_udf!(
    /// `ST_AsGeoJSON`. One GeoJSON geometry object per row.
    StAsGeoJson,
    "st_asgeojson",
    "ST_AsGeoJSON",
    DataType::Utf8,
    io::st_as_geojson
);

/// Which text or binary format a parser reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParseFormat {
    /// Well-Known Text.
    Wkt,
    /// Well-Known Binary. Also accepts the extended form.
    Wkb,
    /// GeoJSON.
    GeoJson,
    /// A geohash string, decoded to the centre of its cell.
    GeoHash,
}

impl ParseFormat {
    const fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::Wkt => ("st_geomfromtext", "ST_GeomFromText"),
            Self::Wkb => ("st_geomfromwkb", "ST_GeomFromWKB"),
            Self::GeoJson => ("st_geomfromgeojson", "ST_GeomFromGeoJSON"),
            Self::GeoHash => ("st_pointfromgeohash", "ST_PointFromGeoHash"),
        }
    }

    fn input_type(self) -> DataType {
        match self {
            Self::Wkb => DataType::Binary,
            _ => DataType::Utf8,
        }
    }

    fn output_type(self) -> GeoArrowType {
        match self {
            Self::GeoHash => GeoArrowType::Point(geoarrow_schema::PointType::new(
                geoarrow_schema::Dimension::XY,
                Default::default(),
            )),
            _ => io::parsed_type(Default::default()),
        }
    }
}

/// `ST_GeomFromText`, `ST_GeomFromWKB`, `ST_GeomFromGeoJSON` or `ST_PointFromGeoHash`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ParseUdf {
    format: ParseFormat,
    name: &'static str,
    signature: Signature,
}

impl ParseUdf {
    /// Build the UDF for one format.
    pub fn new(format: ParseFormat) -> Self {
        Self::with_name(format, format.names().0)
    }

    /// Build the UDF under a different SQL name, for a PostGIS alias.
    pub fn with_name(format: ParseFormat, name: &'static str) -> Self {
        Self {
            format,
            name,
            signature: Signature::uniform(1, vec![format.input_type()], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for ParseUdf {
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
        Ok(self.format.output_type().to_data_type())
    }

    /// Emit the GeoArrow extension metadata on the output field.
    ///
    /// Without this the geometry loses its type and the next function in the chain cannot read it.
    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(geo_field(self.name, &self.format.output_type()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (_, postgis_name) = self.format.names();
        let scalar_input = all_scalar(&args.args);
        let raw = to_array(&args.args[0])?;

        let parsed = match self.format {
            ParseFormat::Wkt => {
                io::st_geom_from_text(&as_utf8(postgis_name, 0, &raw)?, Default::default())
            }
            ParseFormat::Wkb => {
                io::st_geom_from_wkb(&as_binary(postgis_name, 0, &raw)?, Default::default())
            }
            ParseFormat::GeoJson => {
                io::st_geom_from_geojson(&as_utf8(postgis_name, 0, &raw)?, Default::default())
            }
            ParseFormat::GeoHash => {
                io::st_point_from_geohash(&as_utf8(postgis_name, 0, &raw)?, Default::default())
                    .map(|array| Arc::new(array) as Arc<dyn geoarrow_array::GeoArrowArray>)
            }
        }
        .map_err(to_df)?;

        wrap_geo_result(parsed, scalar_input)
    }
}

/// `ST_GeoHash(geometry [, precision])`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StGeoHash {
    signature: Signature,
}

impl StGeoHash {
    /// The default precision. PostGIS uses the same one.
    pub const DEFAULT_PRECISION: i32 = 20;

    /// Build the UDF.
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![TypeSignature::Any(1), TypeSignature::Any(2)],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for StGeoHash {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for StGeoHash {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "st_geohash"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        geo_type("ST_GeoHash", 0, &args.arg_fields[0])?;
        Ok(Arc::new(Field::new("st_geohash", DataType::Utf8, true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scalar_input = all_scalar(&args.args);
        let precision = match args.args.get(1) {
            Some(value) => {
                require_constant_i32("ST_GeoHash", 1, value)?.unwrap_or(Self::DEFAULT_PRECISION)
            }
            None => Self::DEFAULT_PRECISION,
        };
        if !(1..=20).contains(&precision) {
            return plan_err!("ST_GeoHash precision must be between 1 and 20, got {precision}");
        }

        let array = geo_array(&args.args[0], &args.arg_fields[0])?;
        let result = io::st_geohash(array.as_ref(), precision as usize).map_err(to_df)?;
        wrap_result(Arc::new(result), scalar_input)
    }
}

/// Every input and output function.
pub fn io_functions() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::new_from_impl(StAsText::new()),
        ScalarUDF::new_from_impl(StAsBinary::new()),
        ScalarUDF::new_from_impl(StAsEwkb::new()),
        ScalarUDF::new_from_impl(StAsGeoJson::new()),
        ScalarUDF::new_from_impl(ParseUdf::new(ParseFormat::Wkt)),
        ScalarUDF::new_from_impl(ParseUdf::new(ParseFormat::Wkb)),
        // PostGIS reads the extended format with the same parser.
        ScalarUDF::new_from_impl(ParseUdf::with_name(ParseFormat::Wkb, "st_geomfromewkb")),
        ScalarUDF::new_from_impl(ParseUdf::new(ParseFormat::GeoJson)),
        ScalarUDF::new_from_impl(ParseUdf::new(ParseFormat::GeoHash)),
        ScalarUDF::new_from_impl(StGeoHash::new()),
    ]
}

/// `ST_GeomFromText`.
pub fn st_geomfromtext() -> ScalarUDF {
    ScalarUDF::new_from_impl(ParseUdf::new(ParseFormat::Wkt))
}
