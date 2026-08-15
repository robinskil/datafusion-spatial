//! Shared data builders for the benchmarks.
//!
//! Both bench binaries include this module and each uses a subset, so unused items are expected.
#![allow(dead_code)]

use geoarrow_array::array::{GenericWkbArray, PointArray};
use geoarrow_array::builder::PointBuilder;
use geoarrow_array::cast::to_wkb;
use geoarrow_schema::{CoordType, Dimension, PointType};

/// One batch. This is the DataFusion default.
pub const BATCH: usize = 8192;

/// A cheap, repeatable pseudo random generator.
///
/// The benchmark must not depend on a random crate, and the data must be identical between runs.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// `count` points spread over a square of the given size, centred on the origin.
///
/// A large `spread` against a small query box makes most rows disjoint, which is the case the
/// bounding box prefilter is built for.
pub fn points(count: usize, spread: f64, coord_type: CoordType) -> PointArray {
    let mut rng = Lcg::new(0x5EED);
    let values: Vec<geo::Point<f64>> = (0..count)
        .map(|_| {
            geo::point! {
                x: (rng.next_f64() - 0.5) * spread,
                y: (rng.next_f64() - 0.5) * spread,
            }
        })
        .collect();

    PointBuilder::from_points(
        values.iter(),
        PointType::new(Dimension::XY, Default::default()).with_coord_type(coord_type),
    )
    .finish()
}

/// The same points, serialized to WKB. This is the GeoParquet shape.
pub fn points_as_wkb(count: usize, spread: f64) -> GenericWkbArray<i32> {
    to_wkb(&points(count, spread, CoordType::Separated)).unwrap()
}

/// A square centred on the origin.
pub fn square(half_size: f64) -> geo::Geometry<f64> {
    geo::Geometry::Polygon(geo::Polygon::new(
        geo::LineString::new(vec![
            geo::coord! { x: -half_size, y: -half_size },
            geo::coord! { x:  half_size, y: -half_size },
            geo::coord! { x:  half_size, y:  half_size },
            geo::coord! { x: -half_size, y:  half_size },
            geo::coord! { x: -half_size, y: -half_size },
        ]),
        vec![],
    ))
}

/// A regular polygon with `sides` vertices, centred on the origin.
///
/// Use this to cross [`PreparedLiteral::PREPARE_THRESHOLD`] and exercise the R-tree path.
///
/// [`PreparedLiteral::PREPARE_THRESHOLD`]:
///     datafusion_spatial_kernels::predicate::PreparedLiteral::PREPARE_THRESHOLD
pub fn regular_polygon(sides: usize, radius: f64) -> geo::Geometry<f64> {
    let mut coords: Vec<geo::Coord<f64>> = (0..sides)
        .map(|i| {
            let angle = (i as f64) / (sides as f64) * std::f64::consts::TAU;
            geo::coord! { x: radius * angle.cos(), y: radius * angle.sin() }
        })
        .collect();
    coords.push(coords[0]);
    geo::Geometry::Polygon(geo::Polygon::new(geo::LineString::new(coords), vec![]))
}
