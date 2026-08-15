//! Cluster functions.
//!
//! `ST_ClusterKMeans` and `ST_ClusterDBSCAN` assign a cluster id to every row. Each one reads
//! *all* the rows together. PostGIS makes them window functions for that reason. So does this
//! crate.
//!
//! Both read the representative point of each geometry, which is its centroid. PostGIS clusters on
//! the whole geometry for DBSCAN, so a row whose extent is much larger than `eps` may land in a
//! different cluster here.

use crate::materialize::{empty_geometry, geometry_filler};
use arrow_array::Int32Array;
use geo::{Centroid, Dbscan, KMeans, Point};
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::{GeoArrowError, GeoArrowResult};

/// Which cluster algorithm to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cluster {
    /// `ST_ClusterKMeans(geom, k)`.
    KMeans,
    /// `ST_ClusterDBSCAN(geom, eps, minpoints)`.
    Dbscan,
}

impl Cluster {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::KMeans => "ST_ClusterKMeans",
            Self::Dbscan => "ST_ClusterDBSCAN",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::KMeans => "st_clusterkmeans",
            Self::Dbscan => "st_clusterdbscan",
        }
    }

    /// How many numbers follow the geometry argument.
    pub const fn parameter_count(self) -> usize {
        match self {
            Self::KMeans => 1,
            Self::Dbscan => 2,
        }
    }

    /// Every cluster function, for registration.
    pub const ALL: [Self; 2] = [Self::KMeans, Self::Dbscan];
}

/// The points that take part, and where the point of each row sits in that list.
///
/// A row with no point is absent from the first vector and `None` in the second.
type ClusterInput = (Vec<Point<f64>>, Vec<Option<usize>>);

/// The representative point of every row, and which rows had one.
///
/// A null or empty geometry has no point. It takes no part, and its cluster id is null.
fn representative_points(array: &dyn GeoArrowArray) -> GeoArrowResult<ClusterInput> {
    let filler = geometry_filler(array)?;
    let mut points = Vec::with_capacity(array.len());
    let mut slot_of_row = Vec::with_capacity(array.len());

    // Only the centroid survives the iteration, so one geometry serves every row.
    let mut row = empty_geometry();
    for index in 0..array.len() {
        let centroid = if filler(index, &mut row)? {
            row.centroid()
        } else {
            None
        };
        match centroid {
            Some(point) => {
                slot_of_row.push(Some(points.len()));
                points.push(point);
            }
            None => slot_of_row.push(None),
        }
    }
    Ok((points, slot_of_row))
}

/// `ST_ClusterKMeans`. Partition the rows into `k` clusters.
///
/// A fixed seed is used, so the same input always gives the same assignment. PostGIS makes no such
/// promise, but a query that is not reproducible is worse than one that is.
pub fn st_cluster_kmeans(array: &dyn GeoArrowArray, k: usize) -> GeoArrowResult<Int32Array> {
    let (points, slot_of_row) = representative_points(array)?;

    if k == 0 || k > points.len() {
        return Err(GeoArrowError::InvalidGeoArrow(format!(
            "ST_ClusterKMeans needs k between 1 and the number of geometries, got k = {k} for {} \
             geometries",
            points.len()
        )));
    }

    let assignment = points
        .kmeans_with_seed(k, KMEANS_SEED)
        .map_err(|err| GeoArrowError::External(Box::new(err)))?;

    Ok(slot_of_row
        .into_iter()
        .map(|slot| slot.map(|at| assignment[at] as i32))
        .collect())
}

/// Fixed so a query is reproducible.
const KMEANS_SEED: u64 = 0x5EED;

/// `ST_ClusterDBSCAN`. Label the rows by density.
///
/// A row that belongs to no cluster is noise, and its id is null, as in PostGIS.
pub fn st_cluster_dbscan(
    array: &dyn GeoArrowArray,
    epsilon: f64,
    min_points: usize,
) -> GeoArrowResult<Int32Array> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(GeoArrowError::InvalidGeoArrow(format!(
            "ST_ClusterDBSCAN needs a positive eps, got {epsilon}"
        )));
    }

    let (points, slot_of_row) = representative_points(array)?;
    let assignment = points.dbscan(epsilon, min_points);

    Ok(slot_of_row
        .into_iter()
        .map(|slot| slot.and_then(|at| assignment[at]).map(|id| id as i32))
        .collect())
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use geoarrow_array::builder::PointBuilder;
    use geoarrow_schema::{Dimension, PointType};

    use super::*;

    /// Two tight groups, far apart.
    fn two_groups() -> geoarrow_array::array::PointArray {
        let values: Vec<geo::Point<f64>> = vec![
            geo::point!(x: 0.0, y: 0.0),
            geo::point!(x: 0.1, y: 0.1),
            geo::point!(x: 0.2, y: 0.0),
            geo::point!(x: 50.0, y: 50.0),
            geo::point!(x: 50.1, y: 50.1),
            geo::point!(x: 50.2, y: 50.0),
        ];
        PointBuilder::from_points(
            values.iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish()
    }

    #[test]
    fn kmeans_separates_two_groups() {
        let array = two_groups();
        let ids = st_cluster_kmeans(&array, 2).unwrap();

        // The three near the origin share an id, and so do the three far away.
        assert_eq!(ids.value(0), ids.value(1));
        assert_eq!(ids.value(1), ids.value(2));
        assert_eq!(ids.value(3), ids.value(4));
        assert_eq!(ids.value(4), ids.value(5));
        assert_ne!(ids.value(0), ids.value(3), "the two groups must differ");
    }

    /// The same input must always give the same answer.
    #[test]
    fn kmeans_is_reproducible() {
        let array = two_groups();
        let first = st_cluster_kmeans(&array, 2).unwrap();
        let second = st_cluster_kmeans(&array, 2).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn kmeans_rejects_a_bad_k() {
        let array = two_groups();
        assert!(st_cluster_kmeans(&array, 0).is_err());
        assert!(st_cluster_kmeans(&array, 99).is_err(), "k exceeds the rows");
    }

    #[test]
    fn dbscan_separates_two_groups() {
        let array = two_groups();
        let ids = st_cluster_dbscan(&array, 1.0, 2).unwrap();

        assert_eq!(ids.value(0), ids.value(1));
        assert_eq!(ids.value(3), ids.value(4));
        assert_ne!(ids.value(0), ids.value(3));
    }

    /// A point too far from any group is noise, which PostGIS reports as null.
    #[test]
    fn dbscan_marks_noise_as_null() {
        let values: Vec<geo::Point<f64>> = vec![
            geo::point!(x: 0.0, y: 0.0),
            geo::point!(x: 0.1, y: 0.1),
            geo::point!(x: 0.2, y: 0.0),
            geo::point!(x: 999.0, y: 999.0),
        ];
        let array = PointBuilder::from_points(
            values.iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let ids = st_cluster_dbscan(&array, 1.0, 3).unwrap();
        assert!(!ids.is_null(0));
        assert!(ids.is_null(3), "the far point is noise");
    }

    #[test]
    fn dbscan_rejects_a_non_positive_eps() {
        let array = two_groups();
        assert!(st_cluster_dbscan(&array, 0.0, 2).is_err());
        assert!(st_cluster_dbscan(&array, -1.0, 2).is_err());
    }

    /// A null geometry takes no part, and keeps a null id.
    #[test]
    fn nulls_are_skipped() {
        let p0 = geo::point!(x: 0.0, y: 0.0);
        let p1 = geo::point!(x: 0.1, y: 0.1);
        let p2 = geo::point!(x: 50.0, y: 50.0);
        let p3 = geo::point!(x: 50.1, y: 50.1);
        let array = PointBuilder::from_nullable_points(
            [Some(&p0), Some(&p1), None, Some(&p2), Some(&p3)].into_iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let ids = st_cluster_kmeans(&array, 2).unwrap();
        assert!(ids.is_null(2), "the null row has no cluster");
        assert_eq!(ids.value(0), ids.value(1));
        assert_eq!(ids.value(3), ids.value(4));
        assert_ne!(ids.value(0), ids.value(3));
    }
}
