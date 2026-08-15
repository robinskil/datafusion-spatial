//! A spatial join.
//!
//! DataFusion has no equality key for `ST_Intersects(a.geom, b.geom)`, so it plans a
//! [`NestedLoopJoinExec`]: every row of one side against every row of the other. The box verdict
//! makes one pair cheap, but the pair count is the product of the two row counts.
//!
//! [`SpatialJoinExec`] replaces that. It buckets the build side into a bounding box grid, then
//! probes the grid with each row of the other side. Only the rows of the cells that overlap
//! reach the exact test.
//!
//! # How to install it
//!
//! The rule is not on by default, because it rewrites physical plans. Add it when you build the
//! session:
//!
//! ```no_run
//! use datafusion_spatial::datafusion::execution::session_state::SessionStateBuilder;
//! use datafusion_spatial::datafusion::prelude::SessionContext;
//!
//! let state = SessionStateBuilder::new()
//!     .with_default_features()
//!     .with_physical_optimizer_rule(datafusion_spatial::join::spatial_join_rule())
//!     .build();
//! let ctx = SessionContext::new_with_state(state);
//! datafusion_spatial::register_all(&ctx);
//! ```
//!
//! # What it rewrites, and what it leaves alone
//!
//! The rule is deliberately narrow. It rewrites a nested loop join only when every one of these
//! holds. Anything else keeps the plan DataFusion chose.
//!
//! - The join is `INNER`.
//! - The whole join filter is one spatial predicate, with no `AND` and no other term.
//! - Both arguments are plain columns, the first from the left input and the second from the
//!   right.
//! - Both columns carry a GeoArrow extension type.
//! - Disjoint boxes prove the predicate false. That rules out `ST_Disjoint`, where two separate
//!   boxes prove it *true* and a grid would drop the matches.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch, UInt32Array};
use datafusion::arrow::compute::{concat_batches, take};
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{exec_err, internal_err, JoinSide, JoinType, Result};
use datafusion::config::ConfigOptions;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::joins::NestedLoopJoinExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    collect, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, SendableRecordBatchStream,
};
use datafusion_spatial_kernels::bbox::{fill_bboxes, Bbox};
use datafusion_spatial_kernels::index::{BboxGrid, Candidates};
use datafusion_spatial_kernels::materialize::GeometryReader;
use datafusion_spatial_kernels::predicate::Predicate;
use futures::{StreamExt, TryStreamExt};
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::GeoArrowType;

/// The rule that swaps a nested loop join for a spatial join.
pub fn spatial_join_rule() -> Arc<dyn PhysicalOptimizerRule + Send + Sync> {
    Arc::new(SpatialJoinRule)
}

/// True when disjoint boxes prove this predicate false.
///
/// The grid only returns rows whose box overlaps the probe box. A predicate that can be true for
/// two separate boxes would lose matches, so it must keep the nested loop.
fn grid_is_safe(predicate: Predicate) -> bool {
    let left = Bbox {
        minx: 0.0,
        miny: 0.0,
        maxx: 1.0,
        maxy: 1.0,
    };
    let right = Bbox {
        minx: 10.0,
        miny: 10.0,
        maxx: 11.0,
        maxy: 11.0,
    };
    predicate.bbox_verdict(&left, &right) == Some(false)
}

/// Match a spatial predicate UDF by its registered name.
fn predicate_of(name: &str) -> Option<Predicate> {
    Predicate::ALL.into_iter().find(|p| p.sql_name() == name)
}

#[derive(Debug)]
struct SpatialJoinRule;

impl PhysicalOptimizerRule for SpatialJoinRule {
    fn name(&self) -> &str {
        "spatial_join"
    }

    fn schema_check(&self) -> bool {
        true
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            match rewrite(&node) {
                Some(replacement) => Ok(Transformed::yes(replacement)),
                // Not a shape we handle. Keep the plan DataFusion chose.
                None => Ok(Transformed::no(node)),
            }
        })
        .data()
    }
}

/// Downcast a plan node, whichever way this DataFusion major allows.
///
/// DataFusion 53 offers `ExecutionPlan::as_any`. DataFusion 54 dropped it and made `ExecutionPlan`
/// a subtrait of [`Any`], so the cast goes through a trait upcast instead.
#[cfg(feature = "df53")]
fn downcast_plan<T: 'static>(node: &Arc<dyn ExecutionPlan>) -> Option<&T> {
    node.as_any().downcast_ref::<T>()
}

/// Downcast a plan node, whichever way this DataFusion major allows.
#[cfg(feature = "df54")]
fn downcast_plan<T: 'static>(node: &Arc<dyn ExecutionPlan>) -> Option<&T> {
    (node.as_ref() as &dyn Any).downcast_ref::<T>()
}

/// Downcast a physical expression. `PhysicalExpr` changed the same way as `ExecutionPlan`.
#[cfg(feature = "df53")]
fn downcast_expr<T: 'static>(expr: &Arc<dyn PhysicalExpr>) -> Option<&T> {
    expr.as_any().downcast_ref::<T>()
}

/// Downcast a physical expression. `PhysicalExpr` changed the same way as `ExecutionPlan`.
#[cfg(feature = "df54")]
fn downcast_expr<T: 'static>(expr: &Arc<dyn PhysicalExpr>) -> Option<&T> {
    (expr.as_ref() as &dyn Any).downcast_ref::<T>()
}

/// Try to read a nested loop join as a spatial join.
fn rewrite(node: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    let join = downcast_plan::<NestedLoopJoinExec>(node)?;
    if *join.join_type() != JoinType::Inner {
        return None;
    }

    let filter = join.filter()?;
    let call = downcast_expr::<ScalarFunctionExpr>(filter.expression())?;
    let predicate = predicate_of(call.fun().name())?;
    if !grid_is_safe(predicate) {
        return None;
    }

    // Both arguments must be plain columns of the intermediate filter schema.
    let [first, second] = call.args() else {
        return None;
    };
    let first = downcast_expr::<Column>(first)?;
    let second = downcast_expr::<Column>(second)?;

    // Map the two arguments back onto the inputs. DataFusion may have swapped the sides, so the
    // first argument can come from either. Record which, because `ST_Contains` is not symmetric.
    let indices = filter.column_indices();
    let first = indices.get(first.index())?;
    let second = indices.get(second.index())?;
    let (left, right, first_is_left) = match (first.side, second.side) {
        (JoinSide::Left, JoinSide::Right) => (first, second, true),
        (JoinSide::Right, JoinSide::Left) => (second, first, false),
        // Both arguments from one side is not a join condition we can index.
        _ => return None,
    };

    // Both columns must really be geometry, or the exact test has nothing to read.
    let left_input = join.left();
    let right_input = join.right();
    GeoArrowType::from_arrow_field(left_input.schema().field(left.index)).ok()?;
    GeoArrowType::from_arrow_field(right_input.schema().field(right.index)).ok()?;

    Some(Arc::new(SpatialJoinExec::new(
        Arc::clone(left_input),
        Arc::clone(right_input),
        left.index,
        right.index,
        predicate,
        first_is_left,
        join.projection().as_ref().map(|p| p.to_vec()),
        join.schema(),
    )))
}

/// An inner join driven by a bounding box grid over the left input.
#[derive(Debug)]
pub struct SpatialJoinExec {
    /// The build side. Collected in full before the probe starts.
    left: Arc<dyn ExecutionPlan>,
    /// The probe side. Streamed.
    right: Arc<dyn ExecutionPlan>,
    left_column: usize,
    right_column: usize,
    predicate: Predicate,
    /// True when the first predicate argument comes from the left input.
    first_is_left: bool,
    /// Column indices into the left fields followed by the right fields, when the join it
    /// replaced carried its own projection.
    projection: Option<Vec<usize>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl SpatialJoinExec {
    /// Build the operator. `schema` must be the schema the join it replaces produced.
    // Two inputs, two columns, a predicate, an argument order, a projection and a schema. Every
    // one of them is needed to reproduce the join this replaces.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        left_column: usize,
        right_column: usize,
        predicate: Predicate,
        first_is_left: bool,
        projection: Option<Vec<usize>>,
        schema: SchemaRef,
    ) -> Self {
        // Output rows follow the probe side, so the partitioning does too. No ordering survives a
        // join, so the equivalence set starts empty.
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            right.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            left,
            right,
            left_column,
            right_column,
            predicate,
            first_is_left,
            projection,
            schema,
            properties: Arc::new(properties),
        }
    }
}

impl DisplayAs for SpatialJoinExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SpatialJoinExec: predicate={}, left_column={}, right_column={}",
            self.predicate.function_name(),
            self.left_column,
            self.right_column
        )
    }
}

impl ExecutionPlan for SpatialJoinExec {
    fn name(&self) -> &str {
        "SpatialJoinExec"
    }

    // DataFusion 54 made `ExecutionPlan` a subtrait of `Any` and dropped this method.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    /// The build side is read whole, on every partition, so it must arrive undivided.
    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![
            Distribution::SinglePartition,
            Distribution::UnspecifiedDistribution,
        ]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [left, right] = children.as_slice() else {
            return internal_err!("SpatialJoinExec needs exactly two children");
        };
        Ok(Arc::new(SpatialJoinExec::new(
            Arc::clone(left),
            Arc::clone(right),
            self.left_column,
            self.right_column,
            self.predicate,
            self.first_is_left,
            self.projection.clone(),
            Arc::clone(&self.schema),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let left = Arc::clone(&self.left);
        let right = Arc::clone(&self.right);
        let schema = Arc::clone(&self.schema);
        let left_column = self.left_column;
        let right_column = self.right_column;
        let predicate = self.predicate;
        let first_is_left = self.first_is_left;
        let projection = self.projection.clone();

        let batches = futures::stream::once(async move {
            // The build side is read once, in full. Everything after this step streams.
            let collected = collect(Arc::clone(&left), Arc::clone(&context)).await?;
            let build = BuildSide::new(&left.schema(), collected, left_column)?;
            let probe = right.execute(partition, context)?;
            let out = Arc::clone(&schema);

            Ok::<_, datafusion::error::DataFusionError>(probe.map(move |batch| {
                join_one_batch(
                    &build,
                    &batch?,
                    right_column,
                    predicate,
                    first_is_left,
                    projection.as_deref(),
                    &out,
                )
            }))
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            batches,
        )))
    }
}

/// The collected build side, with its boxes and its grid.
struct BuildSide {
    batch: RecordBatch,
    boxes: Vec<Bbox>,
    grid: BboxGrid,
    geometry: Arc<dyn GeoArrowArray>,
}

impl BuildSide {
    fn new(schema: &SchemaRef, batches: Vec<RecordBatch>, column: usize) -> Result<Self> {
        let batch = concat_batches(schema, &batches)?;
        let geometry = geometry_column(&batch, column)?;
        let mut boxes = Vec::new();
        fill_bboxes(geometry.as_ref(), &mut boxes)
            .map_err(|err| datafusion::error::DataFusionError::External(Box::new(err)))?;
        let grid = BboxGrid::build(&boxes);
        Ok(Self {
            batch,
            boxes,
            grid,
            geometry,
        })
    }
}

/// Read one column of a batch as a GeoArrow array.
fn geometry_column(batch: &RecordBatch, column: usize) -> Result<Arc<dyn GeoArrowArray>> {
    let field = batch.schema_ref().field(column).clone();
    geoarrow_array::array::from_arrow_array(batch.column(column).as_ref(), &field)
        .map_err(|err| datafusion::error::DataFusionError::External(Box::new(err)))
}

/// Probe the grid with every row of one batch and emit the matched pairs.
#[allow(clippy::too_many_arguments)]
fn join_one_batch(
    build: &BuildSide,
    probe: &RecordBatch,
    probe_column: usize,
    predicate: Predicate,
    first_is_left: bool,
    projection: Option<&[usize]>,
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    let probe_geometry = geometry_column(probe, probe_column)?;
    let to_df = |err: geoarrow_schema::error::GeoArrowError| {
        datafusion::error::DataFusionError::External(Box::new(err))
    };

    let mut probe_boxes = Vec::new();
    fill_bboxes(probe_geometry.as_ref(), &mut probe_boxes).map_err(to_df)?;

    let mut build_reader = GeometryReader::new(build.geometry.as_ref()).map_err(to_df)?;
    let mut probe_reader = GeometryReader::new(probe_geometry.as_ref()).map_err(to_df)?;
    let mut candidates = Candidates::new(build.batch.num_rows());

    let mut build_rows: Vec<u32> = Vec::new();
    let mut probe_rows: Vec<u32> = Vec::new();

    // The row number is the probe-side output index as well as a box subscript, so the range loop
    // is the honest form here.
    #[allow(clippy::needless_range_loop)]
    for row in 0..probe.num_rows() {
        let probe_box = probe_boxes[row];
        if probe_box.is_empty() {
            continue;
        }
        build.grid.query(&probe_box, &mut candidates);
        if candidates.ids().is_empty() {
            continue;
        }

        // The probe geometry is built once for the row, then shared by every candidate.
        let Some(probe_geom) = probe_reader.read(row).map_err(to_df)? else {
            continue;
        };
        for &id in candidates.ids() {
            let build_box = &build.boxes[id as usize];
            // The predicate reads its arguments in the query's order, which may be the reverse of
            // the plan's build and probe order.
            let (first_box, second_box) = if first_is_left {
                (build_box, &probe_box)
            } else {
                (&probe_box, build_box)
            };
            if let Some(answer) = predicate.bbox_verdict(first_box, second_box) {
                if answer {
                    build_rows.push(id);
                    probe_rows.push(row as u32);
                }
                continue;
            }
            let Some(build_geom) = build_reader.read(id as usize).map_err(to_df)? else {
                continue;
            };
            let (first, second) = if first_is_left {
                (build_geom, probe_geom)
            } else {
                (probe_geom, build_geom)
            };
            if predicate.evaluate(first, second) {
                build_rows.push(id);
                probe_rows.push(row as u32);
            }
        }
    }

    let build_take = UInt32Array::from(build_rows);
    let probe_take = UInt32Array::from(probe_rows);
    let mut joined: Vec<ArrayRef> =
        Vec::with_capacity(build.batch.num_columns() + probe.num_columns());
    for column in build.batch.columns() {
        joined.push(take(column.as_ref(), &build_take, None)?);
    }
    for column in probe.columns() {
        joined.push(take(column.as_ref(), &probe_take, None)?);
    }

    // The join this replaced may have carried its own projection over those columns.
    let columns: Vec<ArrayRef> = match projection {
        Some(indices) => indices
            .iter()
            .map(|&index| Arc::clone(&joined[index]))
            .collect(),
        None => joined,
    };

    if columns.len() != schema.fields().len() {
        return exec_err!(
            "SpatialJoinExec built {} columns for a {} column schema",
            columns.len(),
            schema.fields().len()
        );
    }
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

/// The output schema of an inner join: the left fields, then the right fields.
///
/// Exposed so a caller can build the operator without a nested loop join to copy from.
pub fn inner_join_schema(left: &Schema, right: &Schema) -> SchemaRef {
    let mut fields = left.fields().to_vec();
    fields.extend(right.fields().iter().cloned());
    Arc::new(Schema::new(fields))
}
