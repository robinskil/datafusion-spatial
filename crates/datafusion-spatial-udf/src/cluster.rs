//! Cluster functions as DataFusion window functions.
//!
//! # Why a window function and not a scalar one
//!
//! A cluster id depends on every row in the partition, not just its own. That is exactly what a
//! window function is for, and it is how PostGIS defines these two:
//!
//! ```sql
//! SELECT ST_ClusterKMeans(geom, 5) OVER () AS cluster FROM points
//! ```
//!
//! [`PartitionEvaluator::evaluate_all`] receives the whole partition at once, which is the shape
//! the cluster run kernels need.

// Only the df53 `as_any` methods need this.
#[cfg(feature = "df53")]
use std::any::Any;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::{exec_err, plan_err, Result, ScalarValue};
use datafusion::logical_expr::function::{PartitionEvaluatorArgs, WindowUDFFieldArgs};
use datafusion::logical_expr::{
    PartitionEvaluator, Signature, Volatility, WindowUDF, WindowUDFImpl,
};
use datafusion_spatial_kernels::cluster::{st_cluster_dbscan, st_cluster_kmeans, Cluster};
use geoarrow_array::array::from_arrow_array;

use crate::util::{geo_type, to_df};

/// `ST_ClusterKMeans` or `ST_ClusterDBSCAN`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ClusterUdf {
    cluster: Cluster,
    signature: Signature,
}

impl ClusterUdf {
    /// Build the window function for one algorithm.
    pub fn new(cluster: Cluster) -> Self {
        Self {
            cluster,
            signature: Signature::any(1 + cluster.parameter_count(), Volatility::Immutable),
        }
    }
}

impl WindowUDFImpl for ClusterUdf {
    // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
    #[cfg(feature = "df53")]
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.cluster.sql_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn field(&self, args: WindowUDFFieldArgs) -> Result<FieldRef> {
        // A noise row in DBSCAN has no cluster, so the output is always nullable.
        Ok(Arc::new(Field::new(args.name(), DataType::Int32, true)))
    }

    fn partition_evaluator(
        &self,
        args: PartitionEvaluatorArgs,
    ) -> Result<Box<dyn PartitionEvaluator>> {
        let name = self.cluster.function_name();
        let input_field = args.input_fields().first().cloned().ok_or_else(|| {
            datafusion::common::DataFusionError::Plan(format!("{name} needs a geometry argument"))
        })?;
        geo_type(name, 0, &input_field)?;

        Ok(Box::new(ClusterEvaluator {
            cluster: self.cluster,
            input_field,
        }))
    }
}

#[derive(Debug)]
struct ClusterEvaluator {
    cluster: Cluster,
    input_field: FieldRef,
}

impl PartitionEvaluator for ClusterEvaluator {
    /// The whole partition arrives at once. A cluster function needs exactly that.
    fn evaluate_all(&mut self, values: &[ArrayRef], num_rows: usize) -> Result<ArrayRef> {
        let name = self.cluster.function_name();
        let array = from_arrow_array(values[0].as_ref(), &self.input_field).map_err(to_df)?;

        // The parameters are constant over the partition, so read them from row zero.
        let ids = match self.cluster {
            Cluster::KMeans => {
                let k = constant_usize(name, "k", values.get(1))?;
                st_cluster_kmeans(array.as_ref(), k)
            }
            Cluster::Dbscan => {
                let eps = constant_f64(name, "eps", values.get(1))?;
                let min_points = constant_usize(name, "minpoints", values.get(2))?;
                st_cluster_dbscan(array.as_ref(), eps, min_points)
            }
        }
        .map_err(to_df)?;

        if ids.len() != num_rows {
            return exec_err!("{name} produced {} ids for {num_rows} rows", ids.len());
        }
        Ok(Arc::new(ids))
    }

    /// A cluster function reads the whole partition. A frame that moves has no meaning for it.
    fn uses_window_frame(&self) -> bool {
        false
    }

    fn supports_bounded_execution(&self) -> bool {
        false
    }
}

/// Read a parameter that must be the same for the whole partition.
fn constant_f64(function: &str, name: &str, values: Option<&ArrayRef>) -> Result<f64> {
    let Some(array) = values else {
        return plan_err!("{function} is missing its {name} argument");
    };
    if array.is_empty() || array.is_null(0) {
        return plan_err!("{function} needs a non-null {name}");
    }
    let widened = arrow_cast::cast(array.as_ref(), &DataType::Float64)?;
    Ok(widened
        .as_primitive::<arrow_array::types::Float64Type>()
        .value(0))
}

fn constant_usize(function: &str, name: &str, values: Option<&ArrayRef>) -> Result<usize> {
    let raw = constant_f64(function, name, values)?;
    if raw < 0.0 || raw.fract() != 0.0 {
        return plan_err!("{function} needs a whole non-negative {name}, got {raw}");
    }
    Ok(raw as usize)
}

/// Keeps the unused-import lint quiet for a helper only some builds need.
#[allow(dead_code)]
fn _scalar_hint() -> Option<ScalarValue> {
    None
}

/// Every cluster window function.
pub fn clusters() -> Vec<WindowUDF> {
    Cluster::ALL
        .into_iter()
        .map(|cluster| WindowUDF::new_from_impl(ClusterUdf::new(cluster)))
        .collect()
}
