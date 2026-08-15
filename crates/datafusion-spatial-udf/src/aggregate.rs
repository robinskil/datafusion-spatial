//! `ST_Extent` as a DataFusion aggregate function.
//!
//! The whole accumulator state is four `f64` values. A merge is four comparisons. The aggregate
//! never builds a geometry, so it stays cheap over a whole table.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Signature, Volatility,
};
use datafusion_spatial_kernels::aggregate::Extent;
use datafusion_spatial_kernels::aggregate::{Collect, UnionAll};
use geoarrow_array::array::from_arrow_array;
use geoarrow_array::builder::GeometryBuilder;
use geoarrow_array::builder::RectBuilder;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::{BoxType, Dimension, GeoArrowType};

use crate::util::{geo_type, to_df};

const NAME: &str = "st_extent";

/// `ST_Extent(geometry) -> box2d`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StExtent {
    signature: Signature,
}

impl StExtent {
    /// Build the aggregate.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl Default for StExtent {
    fn default() -> Self {
        Self::new()
    }
}

/// The output type: a GeoArrow box, which is PostGIS `box2d`.
fn box_type(input: Option<&GeoArrowType>) -> BoxType {
    let metadata = input.map(|t| Arc::clone(t.metadata())).unwrap_or_default();
    BoxType::new(Dimension::XY, metadata)
}

impl AggregateUDFImpl for StExtent {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(GeoArrowType::Rect(box_type(None)).to_data_type())
    }

    /// Carry the coordinate reference system of the input onto the result.
    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        let input = geo_type("ST_Extent", 0, &arg_fields[0])?;
        let output = GeoArrowType::Rect(box_type(Some(&input)));
        Ok(Arc::new(output.to_field(NAME, true)))
    }

    /// Four plain `f64` values. No geometry crosses the partition boundary.
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(["xmin", "ymin", "xmax", "ymax"]
            .into_iter()
            .map(|name| Arc::new(Field::new(name, DataType::Float64, false)))
            .collect())
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let input_field = Arc::clone(&acc_args.expr_fields[0]);
        let input = geo_type("ST_Extent", 0, &input_field)?;
        Ok(Box::new(ExtentAccumulator {
            extent: Extent::new(),
            input_field,
            output: box_type(Some(&input)),
        }))
    }
}

#[derive(Debug)]
struct ExtentAccumulator {
    extent: Extent,
    /// The input field, kept so each batch can be read back as a GeoArrow array.
    input_field: FieldRef,
    output: BoxType,
}

impl Accumulator for ExtentAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = from_arrow_array(values[0].as_ref(), &self.input_field).map_err(to_df)?;
        self.extent.update(array.as_ref()).map_err(to_df)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut builder = RectBuilder::with_capacity(self.output.clone(), 1);
        match self.extent.finish() {
            Some(bounds) => builder.push_min_max(
                &geo::coord! { x: bounds.minx, y: bounds.miny },
                &geo::coord! { x: bounds.maxx, y: bounds.maxy },
            ),
            // Every input was null or empty. PostGIS returns NULL.
            None => builder.push_null(),
        }
        let array = builder.finish().to_array_ref();
        ScalarValue::try_from_array(array.as_ref(), 0)
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }

    /// The partial state. An untouched accumulator emits infinities, which merge correctly.
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let bounds = self.extent.bounds();
        Ok(vec![
            ScalarValue::Float64(Some(bounds.minx)),
            ScalarValue::Float64(Some(bounds.miny)),
            ScalarValue::Float64(Some(bounds.maxx)),
            ScalarValue::Float64(Some(bounds.maxy)),
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let xmin = states[0].as_primitive::<Float64Type>();
        let ymin = states[1].as_primitive::<Float64Type>();
        let xmax = states[2].as_primitive::<Float64Type>();
        let ymax = states[3].as_primitive::<Float64Type>();

        for index in 0..xmin.len() {
            self.extent.merge(&Extent::from_bounds(
                xmin.value(index),
                ymin.value(index),
                xmax.value(index),
                ymax.value(index),
            ));
        }
        Ok(())
    }
}

/// `ST_Extent`.
pub fn st_extent() -> AggregateUDF {
    AggregateUDF::new_from_impl(StExtent::new())
}

// ------------------------------------------------- geometry-valued aggregates

/// Which geometry-valued aggregate this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Gather {
    /// `ST_Collect`. Gather every row into one collection.
    Collect,
    /// `ST_Union`. Merge every areal row into one shape.
    Union,
}

impl Gather {
    const fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::Collect => ("st_collect", "ST_Collect"),
            Self::Union => ("st_memunion", "ST_MemUnion"),
        }
    }
}

/// `ST_Collect` or `ST_MemUnion` as an aggregate.
///
/// # Why the union aggregate is not called `ST_Union`
///
/// PostGIS overloads `ST_Union`: two arguments make it a scalar function, one argument makes it an
/// aggregate. DataFusion reads the scalar registry first. It does not try the aggregate registry
/// after an argument count mismatch. So one name cannot serve both.
///
/// The scalar `ST_Union(a, b)` keeps the name, and the aggregate is registered as `ST_MemUnion`,
/// which PostGIS also defines for exactly this operation.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct GatherUdf {
    gather: Gather,
    name: &'static str,
    signature: Signature,
}

impl GatherUdf {
    /// Build the aggregate.
    pub fn new(gather: Gather) -> Self {
        Self::with_name(gather, gather.names().0)
    }

    /// Build the aggregate under a different SQL name, for a PostGIS alias.
    pub fn with_name(gather: Gather, name: &'static str) -> Self {
        Self {
            gather,
            name,
            signature: Signature::any(1, Volatility::Immutable),
        }
    }

    fn output_type(input: Option<&GeoArrowType>) -> GeoArrowType {
        let metadata = input.map(|t| Arc::clone(t.metadata())).unwrap_or_default();
        GeoArrowType::Geometry(geoarrow_schema::GeometryType::new(metadata))
    }
}

impl AggregateUDFImpl for GatherUdf {
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
        Ok(Self::output_type(None).to_data_type())
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        let input = geo_type(self.gather.names().1, 0, &arg_fields[0])?;
        Ok(Arc::new(
            Self::output_type(Some(&input)).to_field(self.name, true),
        ))
    }

    /// One binary value. See [`Collect`] for why the state is WKB.
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![Arc::new(Field::new("wkb", DataType::Binary, true))])
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let input_field = Arc::clone(&acc_args.expr_fields[0]);
        let input = geo_type(self.gather.names().1, 0, &input_field)?;
        Ok(Box::new(GatherAccumulator {
            gather: self.gather,
            collect: Collect::new(),
            union: UnionAll::new(),
            input_field,
            output: Self::output_type(Some(&input)),
        }))
    }
}

#[derive(Debug)]
struct GatherAccumulator {
    gather: Gather,
    collect: Collect,
    union: UnionAll,
    /// The input field, kept so each batch can be read back as a GeoArrow array.
    input_field: FieldRef,
    output: GeoArrowType,
}

impl GatherAccumulator {
    /// Build a one-row geometry array that holds the finished value.
    fn finished(&mut self) -> Result<ScalarValue> {
        let value = match self.gather {
            Gather::Collect => std::mem::take(&mut self.collect).finish(),
            Gather::Union => std::mem::take(&mut self.union).finish(),
        };

        let GeoArrowType::Geometry(output) = self.output.clone() else {
            unreachable!("the output type is always a mixed geometry")
        };
        let mut builder = GeometryBuilder::new(output);
        match value {
            Some(geom) => builder.push_geometry(Some(&geom)).map_err(to_df)?,
            None => builder.push_null(),
        }
        ScalarValue::try_from_array(builder.finish().to_array_ref().as_ref(), 0)
    }
}

impl Accumulator for GatherAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = from_arrow_array(values[0].as_ref(), &self.input_field).map_err(to_df)?;
        match self.gather {
            Gather::Collect => self.collect.update(array.as_ref()).map_err(to_df),
            Gather::Union => self.union.update(array.as_ref()).map_err(to_df),
        }
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        self.finished()
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let bytes = match self.gather {
            Gather::Collect => self.collect.to_wkb(),
            Gather::Union => self.union.to_wkb(),
        }
        .map_err(to_df)?;
        Ok(vec![ScalarValue::Binary(Some(bytes))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let partials = states[0].as_binary::<i32>();
        for index in 0..partials.len() {
            if partials.is_null(index) {
                continue;
            }
            let bytes = partials.value(index);
            match self.gather {
                Gather::Collect => self.collect.merge(Collect::from_wkb(bytes).map_err(to_df)?),
                Gather::Union => self.union.merge(UnionAll::from_wkb(bytes).map_err(to_df)?),
            }
        }
        Ok(())
    }
}

/// `ST_Collect`.
pub fn st_collect() -> AggregateUDF {
    AggregateUDF::new_from_impl(GatherUdf::new(Gather::Collect))
}

/// `ST_MemUnion`, the aggregate union.
///
/// See [`GatherUdf`] for why this is not registered as `ST_Union`.
pub fn st_memunion_agg() -> AggregateUDF {
    AggregateUDF::new_from_impl(GatherUdf::with_name(Gather::Union, "st_memunion"))
}
