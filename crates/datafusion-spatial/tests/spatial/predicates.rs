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
