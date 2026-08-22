//! The spatial predicates, and the box verdict behind them.

use crate::common::*;
use arrow_array::cast::AsArray;
use arrow_array::Array;
use datafusion_spatial::datafusion;

#[tokio::test]
async fn every_predicate_runs() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let inside = "ST_GeomFromText('POINT(0.5 0.5)')";

    // The eleven predicates, each against a point inside the square.
    let expected = [
        ("ST_Intersects", true),
        ("ST_Disjoint", false),
        ("ST_Contains", true),
        ("ST_ContainsProperly", true),
        ("ST_Covers", true),
        ("ST_Touches", false),
        ("ST_Crosses", false),
        ("ST_Overlaps", false),
        ("ST_Equals", false),
    ];
    for (function, want) in expected {
        let got = scalar_bool(&ctx, &format!("SELECT {function}({UNIT_SQUARE}, {inside})")).await?;
        assert_eq!(got, want, "{function} gave {got}");
    }

    // The converse pair reads the other way round.
    assert!(scalar_bool(&ctx, &format!("SELECT ST_Within({inside}, {UNIT_SQUARE})")).await?);
    assert!(
        scalar_bool(
            &ctx,
            &format!("SELECT ST_CoveredBy({inside}, {UNIT_SQUARE})")
        )
        .await?
    );
    Ok(())
}

/// `ST_Contains` and `ST_Within` are converses. A swap of the arguments swaps the answer.
#[tokio::test]
async fn contains_and_within_are_converses() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let batches = collect(
        &ctx,
        &format!(
            "SELECT ST_Contains({UNIT_SQUARE}, geom) AS c, ST_Within(geom, {UNIT_SQUARE}) AS w FROM pts"
        ),
    )
    .await?;

    let contains = batches[0].column(0).as_boolean();
    let within = batches[0].column(1).as_boolean();
    for row in [0usize, 1, 3] {
        assert_eq!(contains.value(row), within.value(row), "row {row}");
    }
    assert!(contains.value(0));
    assert!(!contains.value(1));
    assert!(contains.is_null(2), "null geometry yields null");
    Ok(())
}

/// A large polygon constant sends every direct predicate through the edge index.
///
/// The row at `(1, 0)` is exactly the first vertex of the ring. That row separates the
/// predicates that count the boundary from the ones that do not.
#[tokio::test]
async fn a_large_constant_polygon_answers_every_direct_predicate() -> datafusion::error::Result<()>
{
    let ctx = scattered_points()?;

    // A 64 vertex circle of radius 1, centred on the origin. That is over the index threshold.
    let mut vertices: Vec<String> = (0..64)
        .map(|i| {
            let angle = (i as f64) / 64.0 * std::f64::consts::TAU;
            format!("{} {}", angle.cos(), angle.sin())
        })
        .collect();
    vertices.push(vertices[0].clone());
    let ring = format!("ST_GeomFromText('POLYGON(({}))')", vertices.join(","));

    // One row on the boundary, one inside, one outside. `ST_Point` over a column keeps the row
    // side a column, so the constant takes the indexed path.
    let rows = "(VALUES (1.0, 0.0), (0.5, 0.5), (40.0, 40.0)) AS t(x, y)";
    let batches = collect(
        &ctx,
        &format!(
            "SELECT ST_Contains({ring}, ST_Point(x, y)) AS contains, \
                    ST_Within(ST_Point(x, y), {ring}) AS within, \
                    ST_Covers({ring}, ST_Point(x, y)) AS covers, \
                    ST_CoveredBy(ST_Point(x, y), {ring}) AS coveredby, \
                    ST_Intersects(ST_Point(x, y), {ring}) AS intersects, \
                    ST_Disjoint(ST_Point(x, y), {ring}) AS disjoint \
             FROM {rows}"
        ),
    )
    .await?;

    let column = |index: usize| batches[0].column(index).as_boolean().clone();
    let (boundary, inside, outside) = (0usize, 1usize, 2usize);

    // The interior only.
    for index in [0usize, 1] {
        let got = column(index);
        assert!(
            !got.value(boundary),
            "column {index}: the vertex is not inside"
        );
        assert!(got.value(inside), "column {index}: the middle is");
        assert!(!got.value(outside), "column {index}: the far row is not");
    }

    // The interior and the boundary.
    for index in [2usize, 3, 4] {
        let got = column(index);
        assert!(got.value(boundary), "column {index}: the vertex counts");
        assert!(got.value(inside), "column {index}: so does the middle");
        assert!(!got.value(outside), "column {index}: the far row does not");
    }

    // The complement of intersects.
    let disjoint = column(5);
    assert!(!disjoint.value(boundary));
    assert!(!disjoint.value(inside));
    assert!(disjoint.value(outside));
    Ok(())
}

/// `ST_Disjoint` must be the exact complement of `ST_Intersects`.
#[tokio::test]
async fn disjoint_complements_intersects() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let batches = collect(
        &ctx,
        &format!(
            "SELECT ST_Intersects(geom, {UNIT_SQUARE}) AS i, ST_Disjoint(geom, {UNIT_SQUARE}) AS d FROM pts"
        ),
    )
    .await?;

    let hits = batches[0].column(0).as_boolean();
    let misses = batches[0].column(1).as_boolean();
    for row in [0usize, 1, 3] {
        assert_eq!(hits.value(row), !misses.value(row), "row {row}");
    }
    Ok(())
}

/// The boundary is the case that separates contains from covers.
#[tokio::test]
async fn covers_accepts_the_boundary() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let corner = "ST_GeomFromText('POINT(0 0)')";
    assert!(
        !scalar_bool(
            &ctx,
            &format!("SELECT ST_Contains({UNIT_SQUARE}, {corner})")
        )
        .await?
    );
    assert!(scalar_bool(&ctx, &format!("SELECT ST_Covers({UNIT_SQUARE}, {corner})")).await?);
    Ok(())
}

#[tokio::test]
async fn touches_crosses_and_overlaps() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;

    assert!(
        scalar_bool(
            &ctx,
            "SELECT ST_Touches(ST_GeomFromText('LINESTRING(0 0,1 0)'), \
                               ST_GeomFromText('LINESTRING(1 0,2 0)'))"
        )
        .await?
    );
    assert!(
        scalar_bool(
            &ctx,
            "SELECT ST_Crosses(ST_GeomFromText('LINESTRING(0 0,2 2)'), \
                               ST_GeomFromText('LINESTRING(0 2,2 0)'))"
        )
        .await?
    );
    assert!(
        scalar_bool(
            &ctx,
            &format!(
                "SELECT ST_Overlaps({UNIT_SQUARE}, \
                   ST_GeomFromText('POLYGON((0.5 0.5,1.5 0.5,1.5 1.5,0.5 1.5,0.5 0.5))'))"
            )
        )
        .await?
    );
    Ok(())
}

#[tokio::test]
async fn equals_ignores_vertex_order() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    // The same square, written from a different first corner.
    let rotated = "ST_GeomFromText('POLYGON((1 0,1 1,0 1,0 0,1 0))')";
    assert!(scalar_bool(&ctx, &format!("SELECT ST_Equals({UNIT_SQUARE}, {rotated})")).await?);
    assert!(
        !scalar_bool(
            &ctx,
            &format!("SELECT ST_Equals({UNIT_SQUARE}, ST_GeomFromText('POINT(0 0)'))")
        )
        .await?
    );
    Ok(())
}

#[tokio::test]
async fn dwithin_and_dfullywithin() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let origin = "ST_GeomFromText('POINT(0 0)')";
    let far = "ST_GeomFromText('POINT(3 4)')";

    assert!(scalar_bool(&ctx, &format!("SELECT ST_DWithin({origin}, {far}, 5.0)")).await?);
    assert!(!scalar_bool(&ctx, &format!("SELECT ST_DWithin({origin}, {far}, 4.9)")).await?);

    // Fully within needs the whole geometry inside the radius, not just the nearest point.
    assert!(
        !scalar_bool(
            &ctx,
            &format!("SELECT ST_DFullyWithin({UNIT_SQUARE}, {origin}, 1.0)")
        )
        .await?,
        "the far corner is more than one unit away"
    );
    assert!(
        scalar_bool(
            &ctx,
            &format!("SELECT ST_DFullyWithin({UNIT_SQUARE}, {origin}, 1.5)")
        )
        .await?
    );
    Ok(())
}

#[tokio::test]
async fn dwithin_filters_rows() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let batches = collect(
        &ctx,
        "SELECT COUNT(*) AS n FROM pts \
         WHERE ST_DWithin(geom, ST_GeomFromText('POINT(0 0)'), 2.0)",
    )
    .await?;
    let count = batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0);
    assert_eq!(count, 2, "the two nearby points");
    Ok(())
}

#[tokio::test]
async fn relate_returns_a_matrix_and_matches_a_pattern() -> datafusion::error::Result<()> {
    let ctx = scattered_points()?;
    let inside = "ST_GeomFromText('POINT(0.5 0.5)')";

    let matrix = scalar_text(&ctx, &format!("SELECT ST_Relate({UNIT_SQUARE}, {inside})")).await?;
    assert_eq!(matrix.len(), 9, "a DE-9IM matrix is nine characters");

    // The pattern for "contains".
    assert!(
        scalar_bool(
            &ctx,
            &format!("SELECT ST_Relate({UNIT_SQUARE}, {inside}, 'T*****FF*')")
        )
        .await?
    );

    // And the matrix agrees with the named predicate.
    let contains = scalar_bool(
        &ctx,
        &format!("SELECT ST_Contains({UNIT_SQUARE}, {inside})"),
    )
    .await?;
    assert!(contains);
    Ok(())
}
