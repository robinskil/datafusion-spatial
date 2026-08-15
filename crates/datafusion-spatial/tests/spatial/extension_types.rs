//! Every geometry output must keep its GeoArrow extension metadata.

use crate::common::*;
use datafusion_spatial::datafusion;
use geoarrow_schema::GeoArrowType;

// -------------------------------------------------- accessors
/// Every function that returns a geometry must keep its GeoArrow extension type.
///
/// This is the invariant that `return_field_from_args` exists to hold. Without it a geometry
/// back into an anonymous Arrow struct and breaks the next call in the chain.
#[tokio::test]
async fn accessors_keep_the_extension_type() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;

    let expressions = [
        "ST_MakePoint(1.0, 2.0)",
        "ST_MakeEnvelope(0.0, 0.0, 1.0, 1.0)",
        "ST_MakeLine(ST_MakePoint(0.0, 0.0), ST_MakePoint(1.0, 1.0))",
        "ST_MakePolygon(ST_GeomFromText('LINESTRING(0 0,1 0,1 1,0 0)'))",
        "ST_GeomFromText('POINT(1 2)')",
        "ST_GeomFromWKB(ST_AsBinary(ST_MakePoint(1.0, 2.0)))",
        "ST_GeomFromGeoJSON('{\"type\":\"Point\",\"coordinates\":[1,2]}')",
        "ST_PointFromGeoHash('u4pruyd')",
        "ST_FlipCoordinates(ST_MakePoint(1.0, 2.0))",
        "ST_Force3D(ST_MakePoint(1.0, 2.0))",
        "ST_Force2D(ST_MakePoint(1.0, 2.0))",
        "ST_SetSRID(ST_MakePoint(1.0, 2.0), 4326)",
        "ST_StartPoint(ST_GeomFromText('LINESTRING(1 1,2 2)'))",
        "ST_EndPoint(ST_GeomFromText('LINESTRING(1 1,2 2)'))",
        "ST_PointN(ST_GeomFromText('LINESTRING(1 1,2 2)'), 1)",
        "ST_ExteriorRing(ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 0))'))",
        "ST_InteriorRingN(ST_GeomFromText('POLYGON((0 0,4 0,4 4,0 0),(1 1,2 1,2 2,1 1))'), 1)",
        "ST_GeometryN(ST_GeomFromText('MULTIPOINT(1 1,2 2)'), 1)",
    ];

    for expression in expressions {
        let df = ctx.sql(&format!("SELECT {expression} AS g")).await?;
        let schema = df.schema().as_arrow().clone();
        let field = schema.field(0);
        GeoArrowType::from_arrow_field(field)
            .unwrap_or_else(|err| panic!("{expression} lost its GeoArrow extension type: {err}"));

        // And the values must read back as a GeoArrow array.
        let batches = df.collect().await?;
        geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), field)
            .unwrap_or_else(|err| panic!("{expression} produced an unreadable array: {err}"));
    }
    Ok(())
}

/// A chain of calls proves the metadata survives more than one hop.
#[tokio::test]
async fn functions_chain() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    assert_eq!(
        scalar_text(
            &ctx,
            "SELECT ST_AsText(ST_Force2D(ST_Force3D(ST_FlipCoordinates(\
               ST_SetSRID(ST_MakePoint(1.0, 2.0), 4326)))))"
        )
        .await?,
        "POINT(2 1)"
    );
    Ok(())
}

/// A non-geometry argument must fail at plan time, not at execution.
#[tokio::test]
async fn wrong_argument_type_fails_at_plan_time() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    for sql in [
        "SELECT ST_NPoints(42)",
        "SELECT ST_GeometryType(1.5)",
        "SELECT ST_X(true)",
    ] {
        let err = collect(&ctx, sql).await.unwrap_err();
        assert!(
            err.to_string().contains("not a geometry"),
            "{sql} gave: {err}"
        );
    }
    Ok(())
}

/// A bare text column is read as WKT, and a bare binary column as WKB.
///
/// This is GeoArrow's own type inference. It makes a raw WKT column from CSV usable with no cast,
/// at the cost of a plain string column that holds something else.
#[tokio::test]
async fn bare_text_is_inferred_as_wkt() -> datafusion::error::Result<()> {
    let ctx = mixed_geometries()?;
    assert_eq!(
        scalar_i32(&ctx, "SELECT ST_NPoints('LINESTRING(0 0,1 1,2 2)')").await?,
        3,
        "a plain string is parsed as WKT"
    );
    Ok(())
}

// -------------------------------------------------- predicates and measures
/// Every phase B function that returns a geometry must keep its GeoArrow extension type.
#[tokio::test]
async fn predicates_and_measures_keep_the_extension_type() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let line = "ST_GeomFromText('LINESTRING(0 0,10 0)')";
    let point = "ST_GeomFromText('POINT(5 3)')";

    for expression in [
        format!("ST_ClosestPoint({line}, {point})"),
        format!("ST_ShortestLine({line}, {point})"),
        format!("ST_LineInterpolatePoint({line}, 0.5)"),
    ] {
        let df = ctx.sql(&format!("SELECT {expression} AS g")).await?;
        let schema = df.schema().as_arrow().clone();
        let field = schema.field(0);
        GeoArrowType::from_arrow_field(field)
            .unwrap_or_else(|err| panic!("{expression} lost its extension type: {err}"));

        let batches = df.collect().await?;
        geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), field)
            .unwrap_or_else(|err| panic!("{expression} produced an unreadable array: {err}"));
    }
    Ok(())
}

/// A per-row radius cannot drive the bounding box prefilter, so it is refused at plan time.
#[tokio::test]
async fn dwithin_rejects_a_column_radius() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let err = collect(
        &ctx,
        "SELECT ST_DWithin(geom, ST_GeomFromText('POINT(0 0)'), ST_X(geom)) FROM pts",
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("constant numeric distance"),
        "unexpected error: {err}"
    );
    Ok(())
}

#[tokio::test]
async fn predicates_reject_a_non_geometry_argument() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    for sql in [
        "SELECT ST_Contains(42, 43)",
        "SELECT ST_Area(1.5)",
        "SELECT ST_Distance(true, false)",
    ] {
        let err = collect(&ctx, sql).await.unwrap_err();
        assert!(
            err.to_string().contains("not a geometry"),
            "{sql} gave: {err}"
        );
    }
    Ok(())
}

// -------------------------------------------------- process functions
/// Every geometry-output phase C function must keep its GeoArrow extension type.
#[tokio::test]
async fn process_functions_keep_the_extension_type() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;

    let expressions = [
        format!("ST_Union({UNIT_SQUARE}, {SHIFTED_SQUARE})"),
        format!("ST_Intersection({UNIT_SQUARE}, {SHIFTED_SQUARE})"),
        format!("ST_Difference({UNIT_SQUARE}, {SHIFTED_SQUARE})"),
        format!("ST_SymDifference({UNIT_SQUARE}, {SHIFTED_SQUARE})"),
        format!("ST_ConvexHull({UNIT_SQUARE})"),
        format!("ST_OrientedEnvelope({UNIT_SQUARE})"),
        format!("ST_Boundary({UNIT_SQUARE})"),
        format!("ST_Centroid({UNIT_SQUARE})"),
        format!("ST_PointOnSurface({UNIT_SQUARE})"),
        format!("ST_MakeValid({UNIT_SQUARE})"),
        format!("ST_RemoveRepeatedPoints({UNIT_SQUARE})"),
        format!("ST_Reverse({UNIT_SQUARE})"),
        format!("ST_ForcePolygonCCW({UNIT_SQUARE})"),
        format!("ST_ForcePolygonCW({UNIT_SQUARE})"),
        format!("ST_Buffer({UNIT_SQUARE}, 0.5)"),
        format!("ST_Simplify({UNIT_SQUARE}, 0.01)"),
        format!("ST_SimplifyVW({UNIT_SQUARE}, 0.01)"),
        format!("ST_Segmentize({UNIT_SQUARE}, 0.5)"),
        format!("ST_ConcaveHull({UNIT_SQUARE}, 1.0)"),
        format!("ST_Translate({UNIT_SQUARE}, 1, 1)"),
        format!("ST_Scale({UNIT_SQUARE}, 2, 2)"),
        format!("ST_Rotate({UNIT_SQUARE}, 0.5)"),
        format!("ST_Affine({UNIT_SQUARE}, 1, 0, 0, 1, 1, 1)"),
    ];

    for expression in expressions {
        let df = ctx.sql(&format!("SELECT {expression} AS g")).await?;
        let schema = df.schema().as_arrow().clone();
        let field = schema.field(0);
        GeoArrowType::from_arrow_field(field)
            .unwrap_or_else(|err| panic!("{expression} lost its extension type: {err}"));

        let batches = df.collect().await?;
        geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), field)
            .unwrap_or_else(|err| panic!("{expression} produced an unreadable array: {err}"));
    }
    Ok(())
}

/// Process functions chain into each other and into the earlier phases.
#[tokio::test]
async fn functions_chain_across_families() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    let area = scalar_f64(
        &ctx,
        &format!("SELECT ST_Area(ST_Buffer(ST_Centroid(ST_Union({UNIT_SQUARE}, {SHIFTED_SQUARE})), 1.0))"),
    )
    .await?;
    // A unit circle, near enough given the buffer's segment count.
    assert!(
        (area - std::f64::consts::PI).abs() < 0.05,
        "area was {area}"
    );
    Ok(())
}

#[tokio::test]
async fn processing_rejects_a_non_geometry_argument() -> datafusion::error::Result<()> {
    let ctx = nested_polygons()?;
    for sql in [
        "SELECT ST_ConvexHull(42)",
        "SELECT ST_Buffer(1.5, 1.0)",
        "SELECT ST_Union(true, false)",
    ] {
        let err = collect(&ctx, sql).await.unwrap_err();
        assert!(
            err.to_string().contains("not a geometry"),
            "{sql} gave: {err}"
        );
    }
    Ok(())
}

// -------------------------------------------------- bounding box functions
#[tokio::test]
async fn bounding_box_functions_keep_the_extension_type() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;

    for expression in [
        format!("ST_Envelope({RECTANGLE})"),
        format!("ST_Expand({RECTANGLE}, 1)"),
    ] {
        let df = ctx.sql(&format!("SELECT {expression} AS g")).await?;
        let schema = df.schema().as_arrow().clone();
        let field = schema.field(0);
        let data_type = GeoArrowType::from_arrow_field(field)
            .unwrap_or_else(|err| panic!("{expression} lost its extension type: {err}"));
        assert!(
            matches!(data_type, GeoArrowType::Rect(_)),
            "{expression} must return a box"
        );

        let batches = df.collect().await?;
        geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), field)
            .unwrap_or_else(|err| panic!("{expression} produced an unreadable array: {err}"));
    }

    // The aggregates too.
    for expression in ["ST_Extent(geom)", "ST_Collect(geom)", "ST_MemUnion(geom)"] {
        let df = ctx
            .sql(&format!("SELECT {expression} AS g FROM shapes"))
            .await?;
        let schema = df.schema().as_arrow().clone();
        GeoArrowType::from_arrow_field(schema.field(0))
            .unwrap_or_else(|err| panic!("{expression} lost its extension type: {err}"));
    }
    Ok(())
}

#[tokio::test]
async fn bbox_functions_reject_a_non_geometry_argument() -> datafusion::error::Result<()> {
    let ctx = overlapping_polygons()?;
    for sql in [
        "SELECT ST_XMin(42)",
        "SELECT ST_Envelope(1.5)",
        "SELECT ST_BBoxIntersects(true, false)",
    ] {
        let err = collect(&ctx, sql).await.unwrap_err();
        assert!(
            err.to_string().contains("not a geometry"),
            "{sql} gave: {err}"
        );
    }
    Ok(())
}

// -------------------------------------------------- edit functions
#[tokio::test]
async fn edit_functions_keep_the_extension_type() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    let scatter = "ST_GeomFromText('MULTIPOINT(0 0,4 0,4 4,0 4,2 2)')";
    let line = "ST_GeomFromText('LINESTRING(0 0,1 1,2 2)')";

    for expression in [
        format!("ST_DelaunayTriangles({scatter})"),
        format!("ST_VoronoiPolygons({scatter})"),
        format!("ST_VoronoiLines({scatter})"),
        format!("ST_ChaikinSmoothing({line}, 1)"),
        format!("ST_Multi({line})"),
        format!("ST_Points({line})"),
        format!("ST_SnapToGrid({line}, 0.5)"),
        format!("ST_AddPoint({line}, ST_GeomFromText('POINT(9 9)'))"),
        format!("ST_RemovePoint({line}, 1)"),
        format!("ST_SetPoint({line}, 1, ST_GeomFromText('POINT(9 9)'))"),
        "ST_Project(ST_GeomFromText('POINT(0 0)'), 1000, 0.5)".to_string(),
    ] {
        let df = ctx.sql(&format!("SELECT {expression} AS g")).await?;
        let schema = df.schema().as_arrow().clone();
        let field = schema.field(0);
        GeoArrowType::from_arrow_field(field)
            .unwrap_or_else(|err| panic!("{expression} lost its extension type: {err}"));

        let batches = df.collect().await?;
        geoarrow_array::array::from_arrow_array(batches[0].column(0).as_ref(), field)
            .unwrap_or_else(|err| panic!("{expression} produced an unreadable array: {err}"));
    }
    Ok(())
}

#[tokio::test]
async fn edit_functions_reject_a_non_geometry_argument() -> datafusion::error::Result<()> {
    let ctx = two_point_clusters()?;
    for sql in [
        "SELECT ST_Multi(42)",
        "SELECT ST_DelaunayTriangles(1.5)",
        "SELECT ST_SnapToGrid(true, 1.0)",
    ] {
        let err = collect(&ctx, sql).await.unwrap_err();
        assert!(
            err.to_string().contains("not a geometry"),
            "{sql} gave: {err}"
        );
    }
    Ok(())
}

const MINIMUM_FUNCTIONS: usize = 122;

/// The registry must have no duplicate names within any one kind, and must not shrink.
#[tokio::test]
async fn registry_is_consistent() -> datafusion::error::Result<()> {
    let scalars = datafusion_spatial::scalar_udfs();
    let aggregates = datafusion_spatial::aggregate_udfs();
    let windows = datafusion_spatial::window_udfs();

    let mut duplicates = Vec::new();
    for (kind, mut names) in [
        (
            "scalar",
            scalars.iter().map(|f| f.name()).collect::<Vec<_>>(),
        ),
        (
            "aggregate",
            aggregates.iter().map(|f| f.name()).collect::<Vec<_>>(),
        ),
        (
            "window",
            windows.iter().map(|f| f.name()).collect::<Vec<_>>(),
        ),
    ] {
        names.sort_unstable();
        for pair in names.windows(2) {
            if pair[0] == pair[1] {
                duplicates.push(format!("{kind}: {}", pair[0]));
            }
        }
        for name in &names {
            assert_eq!(*name, name.to_lowercase(), "{name} is not lowercase");
        }
    }
    assert!(duplicates.is_empty(), "duplicate names: {duplicates:?}");

    let total = scalars.len() + aggregates.len() + windows.len();
    assert!(
        total >= MINIMUM_FUNCTIONS,
        "expected at least {MINIMUM_FUNCTIONS} functions, found {total}"
    );
    println!("{total} functions registered");
    Ok(())
}
