//! Fixtures and query helpers shared by every module in this test binary.

#![allow(dead_code)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use datafusion::prelude::SessionContext;
use datafusion_spatial::datafusion;
use geoarrow_array::builder::{GeometryBuilder, PointBuilder, PolygonBuilder};
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::{CoordType, Dimension, GeometryType, PointType, PolygonType};
use std::sync::Arc;

/// The unit square, written as a SQL expression.
pub const UNIT_SQUARE: &str = "ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 1,0 0))')";

/// The unit square moved half a unit along x. It overlaps [`UNIT_SQUARE`] on half its width.
pub const SHIFTED_SQUARE: &str = "ST_GeomFromText('POLYGON((0.5 0,1.5 0,1.5 1,0.5 1,0.5 0))')";

/// A four by three rectangle with its lower corner at the origin.
pub const RECTANGLE: &str = "ST_GeomFromText('POLYGON((0 0,4 0,4 3,0 3,0 0))')";

/// A session with a point column and a mixed geometry column.
pub fn mixed_geometries() -> datafusion::error::Result<SessionContext> {
    let ctx = SessionContext::new();
    datafusion_spatial::register_all(&ctx);

    let point_type =
        PointType::new(Dimension::XY, Default::default()).with_coord_type(CoordType::Separated);
    let p0 = geo::point!(x: 1.0, y: 2.0);
    let p1 = geo::point!(x: 3.0, y: 4.0);
    let points = PointBuilder::from_nullable_points(
        [Some(&p0), None, Some(&p1)].into_iter(),
        point_type.clone(),
    )
    .finish();
    let schema = Arc::new(Schema::new(vec![point_type.to_field("geom", true)]));
    ctx.register_batch(
        "pts",
        RecordBatch::try_new(schema, vec![points.to_array_ref()])?,
    )?;

    let geometry_type = GeometryType::new(Default::default());
    let mut builder = GeometryBuilder::new(geometry_type.clone());
    for geom in [
        geo::Geometry::<f64>::from(geo::wkt! { POINT(1.0 2.0) }),
        geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0,2.0 0.0) }.into(),
        geo::wkt! { POLYGON((0.0 0.0,4.0 0.0,4.0 4.0,0.0 4.0,0.0 0.0),(1.0 1.0,2.0 1.0,2.0 2.0,1.0 1.0)) }.into(),
        geo::wkt! { MULTIPOINT(0.0 0.0,5.0 5.0) }.into(),
    ] {
        builder.push_geometry(Some(&geom)).unwrap();
    }
    let shapes = builder.finish();
    let schema = Arc::new(Schema::new(vec![geometry_type.to_field("geom", true)]));
    ctx.register_batch(
        "shapes",
        RecordBatch::try_new(schema, vec![shapes.to_array_ref()])?,
    )?;

    Ok(ctx)
}
/// A session with four points, one of them null.
pub fn scattered_points() -> datafusion::error::Result<SessionContext> {
    let ctx = SessionContext::new();
    datafusion_spatial::register_all(&ctx);

    let point_type =
        PointType::new(Dimension::XY, Default::default()).with_coord_type(CoordType::Separated);
    let p0 = geo::point!(x: 0.5, y: 0.5);
    let p1 = geo::point!(x: 40.0, y: 40.0);
    let p2 = geo::point!(x: 0.25, y: 0.75);
    let points = PointBuilder::from_nullable_points(
        [Some(&p0), Some(&p1), None, Some(&p2)].into_iter(),
        point_type.clone(),
    )
    .finish();

    let schema = Arc::new(Schema::new(vec![point_type.to_field("geom", true)]));
    ctx.register_batch(
        "pts",
        RecordBatch::try_new(schema, vec![points.to_array_ref()])?,
    )?;
    Ok(ctx)
}
/// A session with a unit square inside a larger square.
pub fn nested_polygons() -> datafusion::error::Result<SessionContext> {
    let ctx = SessionContext::new();
    datafusion_spatial::register_all(&ctx);

    let polygon_type = PolygonType::new(Dimension::XY, Default::default());
    let unit: geo::Polygon<f64> = geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 1.0,0.0 0.0)) };
    let big: geo::Polygon<f64> = geo::wkt! { POLYGON((0.0 0.0,4.0 0.0,4.0 4.0,0.0 4.0,0.0 0.0)) };
    let none: Option<&geo::Polygon<f64>> = None;

    let polygons = PolygonBuilder::from_nullable_polygons(
        &[Some(&unit), Some(&big), none],
        polygon_type.clone(),
    )
    .finish();

    let schema = Arc::new(Schema::new(vec![polygon_type.to_field("geom", true)]));
    ctx.register_batch(
        "shapes",
        RecordBatch::try_new(schema, vec![polygons.to_array_ref()])?,
    )?;
    Ok(ctx)
}
/// A session with two squares that overlap on half their width.
pub fn overlapping_polygons() -> datafusion::error::Result<SessionContext> {
    let ctx = SessionContext::new();
    datafusion_spatial::register_all(&ctx);

    let polygon_type = PolygonType::new(Dimension::XY, Default::default());
    let unit: geo::Polygon<f64> = geo::wkt! { POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 1.0,0.0 0.0)) };
    let shifted: geo::Polygon<f64> =
        geo::wkt! { POLYGON((0.5 0.0,1.5 0.0,1.5 1.0,0.5 1.0,0.5 0.0)) };
    let none: Option<&geo::Polygon<f64>> = None;

    let polygons = PolygonBuilder::from_nullable_polygons(
        &[Some(&unit), Some(&shifted), none],
        polygon_type.clone(),
    )
    .finish();

    let schema = Arc::new(Schema::new(vec![polygon_type.to_field("geom", true)]));
    ctx.register_batch(
        "shapes",
        RecordBatch::try_new(schema, vec![polygons.to_array_ref()])?,
    )?;
    Ok(ctx)
}
/// A session with two tight groups of points, far apart.
pub fn two_point_clusters() -> datafusion::error::Result<SessionContext> {
    let ctx = SessionContext::new();
    datafusion_spatial::register_all(&ctx);

    let point_type = PointType::new(Dimension::XY, Default::default());
    let values: Vec<geo::Point<f64>> = vec![
        geo::point!(x: 0.0, y: 0.0),
        geo::point!(x: 0.1, y: 0.1),
        geo::point!(x: 0.2, y: 0.0),
        geo::point!(x: 50.0, y: 50.0),
        geo::point!(x: 50.1, y: 50.1),
        geo::point!(x: 50.2, y: 50.0),
    ];
    let points = PointBuilder::from_points(values.iter(), point_type.clone()).finish();

    let schema = Arc::new(Schema::new(vec![point_type.to_field("geom", true)]));
    ctx.register_batch(
        "pts",
        RecordBatch::try_new(schema, vec![points.to_array_ref()])?,
    )?;
    Ok(ctx)
}

pub async fn collect(
    ctx: &SessionContext,
    sql: &str,
) -> datafusion::error::Result<Vec<RecordBatch>> {
    ctx.sql(sql).await?.collect().await
}
pub async fn scalar_bool(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<bool> {
    let batches = collect(ctx, sql).await?;
    Ok(batches[0].column(0).as_boolean().value(0))
}
pub async fn scalar_f64(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<f64> {
    let batches = collect(ctx, sql).await?;
    Ok(batches[0].column(0).as_primitive::<Float64Type>().value(0))
}
pub async fn scalar_i32(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<i32> {
    let batches = collect(ctx, sql).await?;
    Ok(batches[0].column(0).as_primitive::<Int32Type>().value(0))
}
pub async fn scalar_text(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<String> {
    let batches = collect(ctx, sql).await?;
    Ok(batches[0].column(0).as_string::<i32>().value(0).to_string())
}
