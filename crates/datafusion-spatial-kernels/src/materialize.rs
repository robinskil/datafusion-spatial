//! Build one `geo` geometry per row without a new allocation.
//!
//! A binary predicate must hand `geo` an owned `Geometry`. The obvious route is `to_geometry()`
//! on the GeoArrow row. That route pays twice per row:
//!
//! 1. **A `CoordBuffer` match per coordinate.** `CoordBuffer` is an enum. The generic accessor
//!    matches it again for every coordinate it reads.
//! 2. **A `Vec` per ring.** Every row allocates and frees its own coordinate buffers.
//!
//! This module removes both. It matches the coordinate buffer once for the whole batch, then
//! refills one caller-owned `Geometry` from plain `f64` slices.
//!
//! Measured on a point column against a polygon column, 8192 rows (`benches/caching.rs`):
//!
//! | Polygon size | `to_geometry()` per row | This module |
//! |---|---|---|
//! | 5 vertices | 547 µs | 172 µs, 3.2 times faster |
//! | 256 vertices | 12.2 ms | 1.80 ms, 6.8 times faster |
//!
//! The same benchmark prices three caches against this approach. All three lose. See
//! `benches/PROFILE.md`.

use std::mem::replace;

use arrow_buffer::NullBuffer;
use geo::{Coord, Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon};
use geoarrow_array::array::CoordBuffer;
use geoarrow_array::cast::AsGeoArrowArray;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::GeoArrowResult;
use geoarrow_schema::GeoArrowType;

use crate::bbox::Bbox;

/// Fill `out` with row `index`. Returns `false` when the row is null.
///
/// The closure holds the concrete array, so the downcast happens once per batch.
pub(crate) type GeometryFiller<'a> =
    Box<dyn Fn(usize, &mut Geometry<f64>) -> GeoArrowResult<bool> + 'a>;

/// A geometry that owns no heap memory.
///
/// Used as a placeholder while the real value is taken apart and refilled.
#[inline]
pub(crate) fn empty_geometry() -> Geometry<f64> {
    Geometry::Point(Point::new(0.0, 0.0))
}

/// True when every row of the array is null.
///
/// A kernel that answers true here can fill its output with nulls and read no coordinate. The
/// guard costs one comparison and saves the whole loop. It can never lose.
#[inline]
pub fn all_null(array: &dyn GeoArrowArray) -> bool {
    !array.is_empty() && array.logical_null_count() == array.len()
}

/// A row reader for one array.
///
/// The downcast and the coordinate buffer match both happen once, in [`GeometryReader::new`].
/// Build one per batch and read every row from it.
///
/// [`crate::predicate::geometry_at`] builds one of these and throws it away. That is right for a
/// constant argument, which is one row of a length-one array. It is wrong in a loop.
pub struct GeometryReader<'a> {
    filler: GeometryFiller<'a>,
    scratch: Geometry<f64>,
}

impl<'a> GeometryReader<'a> {
    /// Build a reader for one array.
    pub fn new(array: &'a dyn GeoArrowArray) -> GeoArrowResult<Self> {
        Ok(Self {
            filler: geometry_filler(array)?,
            scratch: empty_geometry(),
        })
    }

    /// One row, as an owned geometry.
    ///
    /// This allocates for every row. Use [`GeometryReader::read`] in a loop.
    pub fn get(&self, index: usize) -> GeoArrowResult<Option<Geometry<f64>>> {
        let mut out = empty_geometry();
        Ok((self.filler)(index, &mut out)?.then_some(out))
    }

    /// One row, in a buffer that this reader owns and reuses.
    ///
    /// The reference is valid until the next call. The loop allocates nothing after the first row.
    pub fn read(&mut self, index: usize) -> GeoArrowResult<Option<&Geometry<f64>>> {
        let Self { filler, scratch } = self;
        if filler(index, scratch)? {
            Ok(Some(scratch))
        } else {
            Ok(None)
        }
    }
}

/// The coordinates of one array, matched once for the whole batch.
pub(crate) enum Coords<'a> {
    Separated { x: &'a [f64], y: &'a [f64] },
    Interleaved { xy: &'a [f64], stride: usize },
}

impl<'a> Coords<'a> {
    fn new(buffer: &'a CoordBuffer) -> Self {
        match buffer {
            CoordBuffer::Separated(separated) => {
                let raw = separated.raw_buffers();
                Self::Separated {
                    x: &raw[0],
                    y: &raw[1],
                }
            }
            CoordBuffer::Interleaved(interleaved) => Self::Interleaved {
                xy: &interleaved.coords()[..],
                stride: interleaved.dim().size(),
            },
        }
    }

    #[inline]
    fn at(&self, index: usize) -> Coord<f64> {
        match self {
            Self::Separated { x, y } => Coord {
                x: x[index],
                y: y[index],
            },
            Self::Interleaved { xy, stride } => {
                let base = index * stride;
                Coord {
                    x: xy[base],
                    y: xy[base + 1],
                }
            }
        }
    }

    /// Refill `out` with coordinates `start..end`. Keeps the existing capacity.
    #[inline]
    fn fill(&self, out: &mut Vec<Coord<f64>>, start: usize, end: usize) {
        out.clear();
        out.reserve(end - start);
        match self {
            Self::Separated { x, y } => {
                out.extend((start..end).map(|i| Coord { x: x[i], y: y[i] }));
            }
            Self::Interleaved { xy, stride } => {
                out.extend((start..end).map(|i| Coord {
                    x: xy[i * stride],
                    y: xy[i * stride + 1],
                }));
            }
        }
    }

    /// The bounding box of coordinates `start..end`.
    ///
    /// The comparisons match [`Bbox::push_xy`]: a NaN never replaces a bound.
    #[inline]
    pub(crate) fn bounds(&self, start: usize, end: usize) -> Bbox {
        if start >= end {
            return Bbox::EMPTY;
        }
        let mut bbox = Bbox::EMPTY;
        match self {
            Self::Separated { x, y } => {
                for &value in &x[start..end] {
                    if value < bbox.minx {
                        bbox.minx = value;
                    }
                    if value > bbox.maxx {
                        bbox.maxx = value;
                    }
                }
                for &value in &y[start..end] {
                    if value < bbox.miny {
                        bbox.miny = value;
                    }
                    if value > bbox.maxy {
                        bbox.maxy = value;
                    }
                }
            }
            Self::Interleaved { xy, stride } => {
                for i in start..end {
                    bbox.push_xy(xy[i * stride], xy[i * stride + 1]);
                }
            }
        }
        bbox
    }
}

/// The coordinate run of one row.
///
/// Every native type below stores the coordinates of a row in one contiguous range, whatever the
/// ring structure inside it. A bounding box needs only that range, so it needs no ring walk.
pub(crate) struct RowRuns<'a> {
    pub(crate) coords: Coords<'a>,
    chain: Chain<'a>,
}

/// How many offset buffers sit between a row and its coordinates.
enum Chain<'a> {
    /// A point. Row `i` is coordinate `i`.
    Direct,
    /// A line string or multi point.
    One(&'a [i32]),
    /// A polygon or multi line string.
    Two(&'a [i32], &'a [i32]),
    /// A multi polygon.
    Three(&'a [i32], &'a [i32], &'a [i32]),
}

impl RowRuns<'_> {
    /// The coordinate range of one row.
    #[inline]
    pub(crate) fn range(&self, index: usize) -> (usize, usize) {
        match self.chain {
            Chain::Direct => (index, index + 1),
            Chain::One(first) => (first[index] as usize, first[index + 1] as usize),
            Chain::Two(first, second) => (
                second[first[index] as usize] as usize,
                second[first[index + 1] as usize] as usize,
            ),
            Chain::Three(first, second, third) => (
                third[second[first[index] as usize] as usize] as usize,
                third[second[first[index + 1] as usize] as usize] as usize,
            ),
        }
    }
}

/// The coordinate runs of an array, when its type stores one run per row.
///
/// Returns `None` for a mixed, WKB, WKT, box or collection column. Those have no single run.
pub(crate) fn row_runs(array: &dyn GeoArrowArray) -> Option<RowRuns<'_>> {
    let runs = match array.data_type() {
        GeoArrowType::Point(_) => {
            let source = array.as_point();
            RowRuns {
                coords: Coords::new(source.coords()),
                chain: Chain::Direct,
            }
        }
        GeoArrowType::LineString(_) => {
            let source = array.as_line_string();
            RowRuns {
                coords: Coords::new(source.coords()),
                chain: Chain::One(&source.geom_offsets()[..]),
            }
        }
        GeoArrowType::MultiPoint(_) => {
            let source = array.as_multi_point();
            RowRuns {
                coords: Coords::new(source.coords()),
                chain: Chain::One(&source.geom_offsets()[..]),
            }
        }
        GeoArrowType::Polygon(_) => {
            let source = array.as_polygon();
            RowRuns {
                coords: Coords::new(source.coords()),
                chain: Chain::Two(&source.geom_offsets()[..], &source.ring_offsets()[..]),
            }
        }
        GeoArrowType::MultiLineString(_) => {
            let source = array.as_multi_line_string();
            RowRuns {
                coords: Coords::new(source.coords()),
                chain: Chain::Two(&source.geom_offsets()[..], &source.ring_offsets()[..]),
            }
        }
        GeoArrowType::MultiPolygon(_) => {
            let source = array.as_multi_polygon();
            RowRuns {
                coords: Coords::new(source.coords()),
                chain: Chain::Three(
                    &source.geom_offsets()[..],
                    &source.polygon_offsets()[..],
                    &source.ring_offsets()[..],
                ),
            }
        }
        _ => return None,
    };
    Some(runs)
}

/// A blank polygon. Its two `Vec` values are empty, so it allocates nothing.
#[inline]
fn blank_polygon() -> Polygon<f64> {
    Polygon::new(LineString::new(Vec::new()), Vec::new())
}

/// Resize a ring list, and keep the coordinate buffer of every ring that survives.
#[inline]
fn resize_rings(rings: &mut Vec<LineString<f64>>, count: usize) {
    rings.truncate(count);
    while rings.len() < count {
        rings.push(LineString::new(Vec::new()));
    }
}

/// Refill `dst` from rings `first..last`.
#[inline]
fn fill_rings(
    dst: &mut Vec<LineString<f64>>,
    coords: &Coords,
    ring_offsets: &[i32],
    first: usize,
    last: usize,
) {
    resize_rings(dst, last - first);
    for (slot, ring) in dst.iter_mut().zip(first..last) {
        coords.fill(
            &mut slot.0,
            ring_offsets[ring] as usize,
            ring_offsets[ring + 1] as usize,
        );
    }
}

/// Refill one polygon in place. The shell is the first ring, the holes are the rest.
#[inline]
fn fill_polygon(
    slot: &mut Polygon<f64>,
    coords: &Coords,
    ring_offsets: &[i32],
    first: usize,
    last: usize,
) {
    let (mut shell, mut holes) = replace(slot, blank_polygon()).into_inner();
    if first == last {
        shell.0.clear();
        holes.clear();
    } else {
        coords.fill(
            &mut shell.0,
            ring_offsets[first] as usize,
            ring_offsets[first + 1] as usize,
        );
        fill_rings(&mut holes, coords, ring_offsets, first + 1, last);
    }
    *slot = Polygon::new(shell, holes);
}

/// Take the value out of `out` when it already holds the wanted variant.
///
/// A single-typed column always hits, so the buffers survive from row to row. A mixed column
/// that changes variant falls back to a blank value, which is correct but allocates.
macro_rules! take_or_blank {
    ($out:expr, $variant:path, $blank:expr) => {
        match replace($out, empty_geometry()) {
            $variant(value) => value,
            _ => $blank,
        }
    };
}

#[inline]
fn null_at(nulls: &Option<NullBuffer>, index: usize) -> bool {
    matches!(nulls, Some(buffer) if buffer.is_null(index))
}

/// Build a filler for one array.
///
/// The five native types below hold their coordinates in one contiguous run per row, so a row is
/// a slice copy. Every other type falls back to the generic accessor, which stays correct.
pub(crate) fn geometry_filler(array: &dyn GeoArrowArray) -> GeoArrowResult<GeometryFiller<'_>> {
    match array.data_type() {
        GeoArrowType::LineString(_) => {
            let source = array.as_line_string();
            let coords = Coords::new(source.coords());
            let geoms = &source.geom_offsets()[..];
            let nulls = source.logical_nulls();
            Ok(Box::new(move |index, out| {
                if null_at(&nulls, index) {
                    return Ok(false);
                }
                let mut line =
                    take_or_blank!(out, Geometry::LineString, LineString::new(Vec::new()));
                coords.fill(
                    &mut line.0,
                    geoms[index] as usize,
                    geoms[index + 1] as usize,
                );
                *out = Geometry::LineString(line);
                Ok(true)
            }))
        }
        GeoArrowType::Polygon(_) => {
            let source = array.as_polygon();
            let coords = Coords::new(source.coords());
            let geoms = &source.geom_offsets()[..];
            let rings = &source.ring_offsets()[..];
            let nulls = source.logical_nulls();
            Ok(Box::new(move |index, out| {
                if null_at(&nulls, index) {
                    return Ok(false);
                }
                let mut polygon = take_or_blank!(out, Geometry::Polygon, blank_polygon());
                fill_polygon(
                    &mut polygon,
                    &coords,
                    rings,
                    geoms[index] as usize,
                    geoms[index + 1] as usize,
                );
                *out = Geometry::Polygon(polygon);
                Ok(true)
            }))
        }
        GeoArrowType::MultiPoint(_) => {
            let source = array.as_multi_point();
            let coords = Coords::new(source.coords());
            let geoms = &source.geom_offsets()[..];
            let nulls = source.logical_nulls();
            Ok(Box::new(move |index, out| {
                if null_at(&nulls, index) {
                    return Ok(false);
                }
                let mut points =
                    take_or_blank!(out, Geometry::MultiPoint, MultiPoint(Vec::new())).0;
                let (start, end) = (geoms[index] as usize, geoms[index + 1] as usize);
                points.clear();
                points.reserve(end - start);
                points.extend((start..end).map(|i| Point::from(coords.at(i))));
                *out = Geometry::MultiPoint(MultiPoint(points));
                Ok(true)
            }))
        }
        GeoArrowType::MultiLineString(_) => {
            let source = array.as_multi_line_string();
            let coords = Coords::new(source.coords());
            let geoms = &source.geom_offsets()[..];
            let rings = &source.ring_offsets()[..];
            let nulls = source.logical_nulls();
            Ok(Box::new(move |index, out| {
                if null_at(&nulls, index) {
                    return Ok(false);
                }
                let mut lines =
                    take_or_blank!(out, Geometry::MultiLineString, MultiLineString(Vec::new())).0;
                fill_rings(
                    &mut lines,
                    &coords,
                    rings,
                    geoms[index] as usize,
                    geoms[index + 1] as usize,
                );
                *out = Geometry::MultiLineString(MultiLineString(lines));
                Ok(true)
            }))
        }
        GeoArrowType::MultiPolygon(_) => {
            let source = array.as_multi_polygon();
            let coords = Coords::new(source.coords());
            let geoms = &source.geom_offsets()[..];
            let polygons = &source.polygon_offsets()[..];
            let rings = &source.ring_offsets()[..];
            let nulls = source.logical_nulls();
            Ok(Box::new(move |index, out| {
                if null_at(&nulls, index) {
                    return Ok(false);
                }
                let mut list =
                    take_or_blank!(out, Geometry::MultiPolygon, MultiPolygon(Vec::new())).0;
                let (start, end) = (geoms[index] as usize, geoms[index + 1] as usize);
                list.truncate(end - start);
                while list.len() < end - start {
                    list.push(blank_polygon());
                }
                for (slot, polygon) in list.iter_mut().zip(start..end) {
                    fill_polygon(
                        slot,
                        &coords,
                        rings,
                        polygons[polygon] as usize,
                        polygons[polygon + 1] as usize,
                    );
                }
                *out = Geometry::MultiPolygon(MultiPolygon(list));
                Ok(true)
            }))
        }
        // Point and Rect allocate nothing. A mixed, WKB or WKT column has no contiguous run to
        // copy. All of them take the generic path.
        _ => {
            let accessor = crate::predicate::geometry_accessor(array)?;
            Ok(Box::new(move |index, out| match accessor(index)? {
                Some(geometry) => {
                    *out = geometry;
                    Ok(true)
                }
                None => Ok(false),
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate::geometry_accessor;
    use geoarrow_array::builder::{
        GeometryBuilder, LineStringBuilder, MultiPolygonBuilder, PolygonBuilder,
    };
    use geoarrow_schema::{
        CoordType, Dimension, GeometryType, LineStringType, MultiPolygonType, PolygonType,
    };

    /// A polygon with one hole, so the ring loop is exercised.
    fn holed(offset: f64) -> geo::Polygon<f64> {
        let shell = geo::LineString::new(vec![
            geo::coord! { x: offset, y: 0.0 },
            geo::coord! { x: offset + 10.0, y: 0.0 },
            geo::coord! { x: offset + 10.0, y: 10.0 },
            geo::coord! { x: offset, y: 10.0 },
            geo::coord! { x: offset, y: 0.0 },
        ]);
        let hole = geo::LineString::new(vec![
            geo::coord! { x: offset + 2.0, y: 2.0 },
            geo::coord! { x: offset + 4.0, y: 2.0 },
            geo::coord! { x: offset + 4.0, y: 4.0 },
            geo::coord! { x: offset + 2.0, y: 2.0 },
        ]);
        geo::Polygon::new(shell, vec![hole])
    }

    /// The filler must agree with the generic accessor for every row, in both coordinate layouts.
    ///
    /// The reference is [`geometry_accessor`], not `geometry_at`. `geometry_at` now routes through
    /// the filler itself, so it would compare the fast path against itself and prove nothing.
    fn assert_matches(array: &dyn GeoArrowArray) {
        let reference = geometry_accessor(array).unwrap();
        let filler = geometry_filler(array).unwrap();
        let mut scratch = empty_geometry();
        for index in 0..array.len() {
            let expected = reference(index).unwrap();
            let filled = filler(index, &mut scratch).unwrap();
            assert_eq!(filled, expected.is_some(), "null disagreement at {index}");
            if let Some(expected) = expected {
                assert_eq!(scratch, expected, "value disagreement at {index}");
            }
        }
    }

    #[test]
    fn polygons_match_the_generic_accessor() {
        for coord_type in [CoordType::Separated, CoordType::Interleaved] {
            let polygons = vec![holed(0.0), holed(100.0), holed(200.0)];
            let array = PolygonBuilder::from_polygons(
                &polygons,
                PolygonType::new(Dimension::XY, Default::default()).with_coord_type(coord_type),
            )
            .finish();
            assert_matches(&array);
        }
    }

    #[test]
    fn line_strings_match_the_generic_accessor() {
        let lines = vec![
            geo::LineString::new(vec![
                geo::coord! { x: 0.0, y: 0.0 },
                geo::coord! { x: 1.0, y: 1.0 },
            ]),
            geo::LineString::new(vec![
                geo::coord! { x: 5.0, y: 5.0 },
                geo::coord! { x: 6.0, y: 7.0 },
                geo::coord! { x: 8.0, y: 9.0 },
            ]),
        ];
        let array = LineStringBuilder::from_line_strings(
            &lines,
            LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();
        assert_matches(&array);
    }

    /// Row 1 holds two polygons and row 0 holds one, so the reuse path must resize the list.
    #[test]
    fn multi_polygons_match_the_generic_accessor() {
        let values = vec![
            geo::MultiPolygon(vec![holed(0.0)]),
            geo::MultiPolygon(vec![holed(100.0), holed(200.0)]),
            geo::MultiPolygon(vec![holed(300.0)]),
        ];
        let array = MultiPolygonBuilder::from_multi_polygons(
            &values,
            MultiPolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();
        assert_matches(&array);
    }

    /// A mixed column changes variant from row to row. The filler must still be correct.
    #[test]
    fn a_mixed_column_falls_back_and_stays_correct() {
        let values: Vec<geo::Geometry<f64>> = vec![
            geo::Geometry::Point(geo::point! { x: 1.0, y: 2.0 }),
            geo::Geometry::Polygon(holed(0.0)),
            geo::Geometry::LineString(geo::LineString::new(vec![
                geo::coord! { x: 0.0, y: 0.0 },
                geo::coord! { x: 3.0, y: 4.0 },
            ])),
        ];
        let mut builder = GeometryBuilder::new(GeometryType::new(Default::default()));
        for value in &values {
            builder.push_geometry(Some(value)).unwrap();
        }
        assert_matches(&builder.finish());
    }

    /// Reuse must not leak the previous row into the next one.
    #[test]
    fn reuse_does_not_leak_between_rows() {
        let polygons = vec![holed(0.0), holed(100.0)];
        let array = PolygonBuilder::from_polygons(
            &polygons,
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let filler = geometry_filler(&array).unwrap();
        let mut scratch = empty_geometry();
        filler(0, &mut scratch).unwrap();
        filler(1, &mut scratch).unwrap();
        assert_eq!(scratch, geo::Geometry::Polygon(holed(100.0)));
    }
}
