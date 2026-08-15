//! Tessellation and smooth functions.
//!
//! `ST_DelaunayTriangles`, `ST_VoronoiPolygons` and `ST_VoronoiLines` each turn one geometry into
//! many, so they return a collection per row rather than a geometry per row.
//!
//! # Where the vertices come from
//!
//! All three read only the vertices of the input. A polygon and the multi point of its corners
//! give the same triangulation, which is what PostGIS does too.

use std::sync::Arc;

use crate::materialize::GeometryReader;
use geo::{
    Geometry, GeometryCollection, MultiLineString, MultiPoint, TriangulateDelaunayUnconstrained,
    Voronoi,
};
use geoarrow_array::builder::GeometryBuilder;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::GeoArrowResult;
use geoarrow_schema::GeometryType;

/// Which tessellation to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tessellation {
    /// `ST_DelaunayTriangles`. A collection of triangles over the input vertices.
    Delaunay,
    /// `ST_VoronoiPolygons`. A collection of cells, one per input vertex.
    VoronoiPolygons,
    /// `ST_VoronoiLines`. The cell boundaries as one multi line string.
    VoronoiLines,
}

impl Tessellation {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::Delaunay => "ST_DelaunayTriangles",
            Self::VoronoiPolygons => "ST_VoronoiPolygons",
            Self::VoronoiLines => "ST_VoronoiLines",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::Delaunay => "st_delaunaytriangles",
            Self::VoronoiPolygons => "st_voronoipolygons",
            Self::VoronoiLines => "st_voronoilines",
        }
    }

    /// Every tessellation, for registration.
    pub const ALL: [Self; 3] = [Self::Delaunay, Self::VoronoiPolygons, Self::VoronoiLines];

    /// Build the tessellation of one geometry.
    ///
    /// Returns `None` when the input has too few distinct vertices to tessellate, which PostGIS
    /// also reports as an empty result rather than an error.
    pub fn apply(self, geom: &Geometry<f64>) -> Option<Geometry<f64>> {
        use geo::CoordsIter;

        // Every algorithm works from the vertex set.
        let points: MultiPoint<f64> =
            MultiPoint::new(geom.coords_iter().map(geo::Point::from).collect());
        if points.0.len() < 3 {
            return None;
        }

        match self {
            Self::Delaunay => {
                let triangles = points.unconstrained_triangulation().ok()?;
                Some(Geometry::GeometryCollection(GeometryCollection::new_from(
                    triangles
                        .into_iter()
                        .map(|triangle| Geometry::Polygon(triangle.to_polygon()))
                        .collect(),
                )))
            }
            Self::VoronoiPolygons => {
                let cells = points.voronoi_cells().ok()?;
                Some(Geometry::GeometryCollection(GeometryCollection::new_from(
                    cells.into_iter().map(Geometry::Polygon).collect(),
                )))
            }
            Self::VoronoiLines => {
                let edges = points.voronoi_edges().ok()?;
                Some(Geometry::MultiLineString(MultiLineString::new(
                    edges
                        .into_iter()
                        .map(|line| geo::LineString::new(vec![line.start, line.end]))
                        .collect(),
                )))
            }
        }
    }
}

/// Any tessellation over an array.
pub fn tessellate(
    array: &dyn GeoArrowArray,
    kind: Tessellation,
    output: GeometryType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let mut reader = GeometryReader::new(array)?;
    let mut builder = GeometryBuilder::new(output);
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => match kind.apply(geom) {
                Some(result) => builder.push_geometry(Some(&result))?,
                None => builder.push_null(),
            },
            None => builder.push_null(),
        }
    }
    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use geo_traits::to_geo::ToGeoGeometry;
    use geoarrow_array::builder::GeometryBuilder as GeoBuilder;
    use geoarrow_array::cast::AsGeoArrowArray;
    use geoarrow_array::GeoArrowArrayAccessor;

    use super::*;

    fn scatter() -> geoarrow_array::array::GeometryArray {
        let mut builder = GeoBuilder::new(GeometryType::new(Default::default()));
        builder
            .push_geometry(Some(&Geometry::<f64>::from(
                geo::wkt! { MULTIPOINT(0.0 0.0,4.0 0.0,4.0 4.0,0.0 4.0,2.0 2.0) },
            )))
            .unwrap();
        // Two points cannot be tessellated.
        builder
            .push_geometry(Some(&Geometry::<f64>::from(
                geo::wkt! { MULTIPOINT(0.0 0.0,1.0 1.0) },
            )))
            .unwrap();
        builder.finish()
    }

    fn read(array: &dyn GeoArrowArray, row: usize) -> Option<Geometry<f64>> {
        array
            .as_geometry()
            .get(row)
            .unwrap()
            .map(|geom| geom.to_geometry())
    }

    #[test]
    fn delaunay_covers_the_hull() {
        use geo::Area;

        let array = scatter();
        let output = GeometryType::new(Default::default());
        let result = tessellate(&array, Tessellation::Delaunay, output).unwrap();

        let Some(Geometry::GeometryCollection(triangles)) = read(result.as_ref(), 0) else {
            panic!("expected a collection")
        };
        assert!(!triangles.0.is_empty());
        // The triangles together cover the convex hull of the input, which is the 4 by 4 square.
        let total: f64 = triangles.0.iter().map(|t| t.unsigned_area()).sum();
        assert!((total - 16.0).abs() < 1e-9, "total area was {total}");
    }

    #[test]
    fn voronoi_gives_one_cell_per_vertex() {
        let array = scatter();
        let output = GeometryType::new(Default::default());
        let result = tessellate(&array, Tessellation::VoronoiPolygons, output).unwrap();

        let Some(Geometry::GeometryCollection(cells)) = read(result.as_ref(), 0) else {
            panic!("expected a collection")
        };
        assert_eq!(cells.0.len(), 5, "one cell per input point");
    }

    #[test]
    fn voronoi_lines_are_the_cell_boundaries() {
        let array = scatter();
        let output = GeometryType::new(Default::default());
        let result = tessellate(&array, Tessellation::VoronoiLines, output).unwrap();

        let Some(Geometry::MultiLineString(edges)) = read(result.as_ref(), 0) else {
            panic!("expected a multi line string")
        };
        assert!(!edges.0.is_empty());
        assert!(edges.0.iter().all(|line| line.0.len() == 2));
    }

    /// Fewer than three vertices cannot be tessellated, which is null rather than an error.
    #[test]
    fn too_few_points_gives_null() {
        let array = scatter();
        let output = GeometryType::new(Default::default());
        for kind in Tessellation::ALL {
            let result = tessellate(&array, kind, output.clone()).unwrap();
            assert!(
                read(result.as_ref(), 1).is_none(),
                "{} must give null for two points",
                kind.function_name()
            );
        }
    }

    /// A polygon and the multi point of its corners tessellate the same way.
    #[test]
    fn only_the_vertices_matter() {
        let mut builder = GeoBuilder::new(GeometryType::new(Default::default()));
        builder
            .push_geometry(Some(&Geometry::<f64>::from(
                geo::wkt! { POLYGON((0.0 0.0,4.0 0.0,4.0 4.0,0.0 4.0,0.0 0.0)) },
            )))
            .unwrap();
        let polygon = builder.finish();

        let output = GeometryType::new(Default::default());
        let from_polygon = tessellate(&polygon, Tessellation::Delaunay, output.clone()).unwrap();
        let Some(Geometry::GeometryCollection(triangles)) = read(from_polygon.as_ref(), 0) else {
            panic!("expected a collection")
        };
        // A square gives two triangles, whichever way the diagonal falls.
        assert_eq!(triangles.0.len(), 2);
    }
}
