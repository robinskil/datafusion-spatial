//! Conversions between geometries and text or binary formats.
//!
//! The output of `ST_AsText`, `ST_AsBinary`, `ST_AsEWKB` and `ST_AsGeoJSON` is plain text or plain
//! bytes, not a geometry. PostGIS does the same, so a cast back needs an explicit
//! `ST_GeomFrom...` call.
//!
//! # A known limit
//!
//! `ST_GeomFromEWKB` reads the extended format but drops the embedded SRID, because the underlying
//! `wkb` crate ignores it. Stamp the column afterwards with `ST_SetSRID` when the SRID matters.
//! `ST_AsEWKB` does write the SRID, taken from the column metadata.

use std::sync::Arc;

use crate::materialize::{all_null, GeometryReader};
use arrow_array::builder::{BinaryBuilder, StringBuilder};
use arrow_array::{Array, BinaryArray, GenericBinaryArray, GenericStringArray, StringArray};
use geo_traits::CoordTrait;
use geoarrow_array::array::{GenericWkbArray, GenericWktArray, PointArray};
use geoarrow_array::builder::PointBuilder;
use geoarrow_array::cast::{from_wkb, from_wkt, to_wkb, to_wkt};
use geoarrow_array::{GeoArrowArray, IntoArrow};
use geoarrow_schema::error::{GeoArrowError, GeoArrowResult};
use geoarrow_schema::{
    Dimension, GeoArrowType, GeometryType, Metadata, PointType, WkbType, WktType,
};

use crate::crs::srid_of;

/// The default output type of every `ST_GeomFrom...` function.
///
/// A mixed geometry array, so any input parses without a plan-time type declaration.
pub fn parsed_type(metadata: Arc<Metadata>) -> GeoArrowType {
    GeoArrowType::Geometry(GeometryType::new(metadata))
}

/// `ST_AsText`. Well-Known Text, one string per row.
pub fn st_as_text(array: &dyn GeoArrowArray) -> GeoArrowResult<StringArray> {
    let wkt: GenericWktArray<i32> = to_wkt(array)?;
    Ok(wkt.into_arrow())
}

/// `ST_AsBinary`. Well-Known Binary, one value per row.
pub fn st_as_binary(array: &dyn GeoArrowArray) -> GeoArrowResult<BinaryArray> {
    let wkb: GenericWkbArray<i32> = to_wkb(array)?;
    Ok(wkb.into_arrow())
}

/// `ST_AsEWKB`. Well-Known Binary with the column SRID written into the header.
///
/// When the column carries no SRID this is byte for byte the same as [`st_as_binary`].
pub fn st_as_ewkb(array: &dyn GeoArrowArray) -> GeoArrowResult<BinaryArray> {
    let srid = srid_of(&array.data_type());
    let plain = st_as_binary(array)?;
    if srid == 0 {
        return Ok(plain);
    }

    // EWKB sets a flag in the type word and inserts the SRID straight after it.
    let mut builder =
        BinaryBuilder::with_capacity(plain.len(), plain.value_data().len() + plain.len() * 4);
    for index in 0..plain.len() {
        if plain.is_null(index) {
            builder.append_null();
            continue;
        }
        match add_srid(plain.value(index), srid) {
            Some(bytes) => builder.append_value(&bytes),
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// Flag set in the WKB type word when an SRID follows it.
const EWKB_SRID_FLAG: u32 = 0x2000_0000;

/// Rewrite a WKB header as an EWKB header that holds `srid`.
fn add_srid(wkb: &[u8], srid: i32) -> Option<Vec<u8>> {
    if wkb.len() < 5 {
        return None;
    }
    // Byte 0 is the byte order: 1 means little endian.
    let little_endian = wkb[0] == 1;
    let raw: [u8; 4] = wkb[1..5].try_into().ok()?;
    let type_word = if little_endian {
        u32::from_le_bytes(raw)
    } else {
        u32::from_be_bytes(raw)
    } | EWKB_SRID_FLAG;

    let mut out = Vec::with_capacity(wkb.len() + 4);
    out.push(wkb[0]);
    if little_endian {
        out.extend_from_slice(&type_word.to_le_bytes());
        out.extend_from_slice(&srid.to_le_bytes());
    } else {
        out.extend_from_slice(&type_word.to_be_bytes());
        out.extend_from_slice(&srid.to_be_bytes());
    }
    out.extend_from_slice(&wkb[5..]);
    Some(out)
}

/// `ST_AsGeoJSON`. One GeoJSON geometry object per row.
pub fn st_as_geojson(array: &dyn GeoArrowArray) -> GeoArrowResult<StringArray> {
    if all_null(array) {
        return Ok(StringArray::new_null(array.len()));
    }
    let mut reader = GeometryReader::new(array)?;
    let mut builder = StringBuilder::with_capacity(array.len(), array.len() * 64);
    for index in 0..array.len() {
        match reader.read(index)? {
            Some(geom) => {
                let value = geojson::GeometryValue::from(geom);
                builder.append_value(value.to_string());
            }
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// `ST_GeomFromText`. Parse Well-Known Text.
pub fn st_geom_from_text(
    text: &StringArray,
    metadata: Arc<Metadata>,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let wkt = GenericWktArray::<i32>::new(text.clone(), Arc::clone(&metadata));
    from_wkt(&wkt, parsed_type(metadata))
}

/// `ST_GeomFromWKB` and `ST_GeomFromEWKB`.
///
/// The `wkb` reader accepts both the plain and the extended format. An SRID embedded in extended
/// input is ignored. See the module documentation.
pub fn st_geom_from_wkb(
    binary: &BinaryArray,
    metadata: Arc<Metadata>,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let wkb = GenericWkbArray::<i32>::new(binary.clone(), Arc::clone(&metadata));
    from_wkb(&wkb, parsed_type(metadata))
}

/// `ST_GeomFromGeoJSON`. Parse a GeoJSON geometry object.
pub fn st_geom_from_geojson(
    text: &StringArray,
    metadata: Arc<Metadata>,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let mut builder = geoarrow_array::builder::GeometryBuilder::new(GeometryType::new(metadata));

    for index in 0..text.len() {
        if text.is_null(index) {
            builder.push_null();
            continue;
        }
        let parsed: geojson::Geometry = text.value(index).parse().map_err(external)?;
        let geometry: geo::Geometry<f64> = geo::Geometry::try_from(parsed).map_err(external)?;
        builder.push_geometry(Some(&geometry))?;
    }

    Ok(Arc::new(builder.finish()))
}

fn external<E: std::error::Error + Send + Sync + 'static>(err: E) -> GeoArrowError {
    GeoArrowError::External(Box::new(err))
}

/// `ST_GeoHash`. Encode a point as a geohash string.
///
/// PostGIS accepts any geometry and hashes the centre of its box. This follows the same rule for
/// points, and returns null for any other type, which is the case that matters in practice.
pub fn st_geohash(array: &dyn GeoArrowArray, precision: usize) -> GeoArrowResult<StringArray> {
    if all_null(array) {
        return Ok(StringArray::new_null(array.len()));
    }
    let mut reader = GeometryReader::new(array)?;
    let mut builder = StringBuilder::with_capacity(array.len(), array.len() * precision);
    for index in 0..array.len() {
        let Some(geom) = reader.read(index)? else {
            builder.append_null();
            continue;
        };
        match geom {
            geo::Geometry::Point(point) => {
                match geohash::encode(geo::coord! { x: point.x(), y: point.y() }, precision) {
                    Ok(hash) => builder.append_value(hash),
                    // Out of range coordinates cannot be hashed. PostGIS errors, we return null.
                    Err(_) => builder.append_null(),
                }
            }
            _ => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// `ST_PointFromGeoHash`. Decode a geohash string to the centre of its cell.
pub fn st_point_from_geohash(
    text: &StringArray,
    metadata: Arc<Metadata>,
) -> GeoArrowResult<PointArray> {
    let mut builder =
        PointBuilder::with_capacity(PointType::new(Dimension::XY, metadata), text.len());

    for index in 0..text.len() {
        if text.is_null(index) {
            builder.push_null();
            continue;
        }
        match geohash::decode(text.value(index)) {
            Ok((coord, _, _)) => builder.push_coord(Some(&GeoCoord(coord.x, coord.y))),
            Err(_) => builder.push_null(),
        }
    }

    Ok(builder.finish())
}

/// A two-dimensional coordinate for the point builder.
struct GeoCoord(f64, f64);

impl CoordTrait for GeoCoord {
    type T = f64;

    fn dim(&self) -> geo_traits::Dimensions {
        geo_traits::Dimensions::Xy
    }

    fn x(&self) -> f64 {
        self.0
    }

    fn y(&self) -> f64 {
        self.1
    }

    fn nth_or_panic(&self, n: usize) -> f64 {
        match n {
            0 => self.0,
            1 => self.1,
            _ => panic!("XY has two ordinates"),
        }
    }
}

/// The GeoArrow types the text and binary writers accept back.
pub fn wkb_type(metadata: Arc<Metadata>) -> WkbType {
    WkbType::new(metadata)
}

/// The WKT type with the given metadata.
pub fn wkt_type(metadata: Arc<Metadata>) -> WktType {
    WktType::new(metadata)
}

/// Helper for callers that hold a `GenericStringArray` of another offset width.
pub fn narrow_strings<O: arrow_array::OffsetSizeTrait>(
    array: &GenericStringArray<O>,
) -> StringArray {
    array.iter().collect()
}

/// Helper for callers that hold a `GenericBinaryArray` of another offset width.
pub fn narrow_binary<O: arrow_array::OffsetSizeTrait>(
    array: &GenericBinaryArray<O>,
) -> BinaryArray {
    array.iter().collect()
}

#[cfg(test)]
mod tests {
    use geoarrow_array::builder::PointBuilder;
    use geoarrow_schema::{CoordType, PointType};

    use super::*;
    use crate::crs::st_set_srid;

    fn points() -> PointArray {
        let p0 = geo::point!(x: 1.5, y: 2.5);
        let p1 = geo::point!(x: -3.0, y: 4.0);
        PointBuilder::from_nullable_points(
            [Some(&p0), None, Some(&p1)].into_iter(),
            PointType::new(Dimension::XY, Default::default()).with_coord_type(CoordType::Separated),
        )
        .finish()
    }

    #[test]
    fn as_text_round_trips() {
        let array = points();
        let text = st_as_text(&array).unwrap();
        assert!(text.value(0).starts_with("POINT"));
        assert!(text.is_null(1));

        let parsed = st_geom_from_text(&text, Default::default()).unwrap();
        let back = st_as_text(parsed.as_ref()).unwrap();
        assert_eq!(text.value(0), back.value(0));
        assert!(back.is_null(1));
    }

    #[test]
    fn as_binary_round_trips() {
        let array = points();
        let binary = st_as_binary(&array).unwrap();
        let parsed = st_geom_from_wkb(&binary, Default::default()).unwrap();
        let back = st_as_binary(parsed.as_ref()).unwrap();

        assert_eq!(binary.value(0), back.value(0));
        assert!(back.is_null(1));
    }

    #[test]
    fn as_geojson_round_trips() {
        let array = points();
        let json = st_as_geojson(&array).unwrap();
        assert!(json.value(0).contains("\"Point\""));
        assert!(json.is_null(1));

        let parsed = st_geom_from_geojson(&json, Default::default()).unwrap();
        let back = st_as_geojson(parsed.as_ref()).unwrap();
        assert_eq!(json.value(0), back.value(0));
    }

    /// EWKB must add exactly four bytes and set the SRID flag.
    #[test]
    fn as_ewkb_carries_the_srid() {
        let array = points();
        let stamped = st_set_srid(&array, 4326).unwrap();

        let plain = st_as_binary(&array).unwrap();
        let extended = st_as_ewkb(stamped.as_ref()).unwrap();

        assert_eq!(extended.value(0).len(), plain.value(0).len() + 4);
        assert!(extended.is_null(1));

        let bytes = extended.value(0);
        assert_eq!(bytes[0], 1, "little endian");
        let type_word = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
        assert_ne!(type_word & EWKB_SRID_FLAG, 0, "the SRID flag must be set");
        let srid = i32::from_le_bytes(bytes[5..9].try_into().unwrap());
        assert_eq!(srid, 4326);
    }

    #[test]
    fn as_ewkb_without_a_srid_is_plain_wkb() {
        let array = points();
        assert_eq!(
            st_as_ewkb(&array).unwrap().value(0),
            st_as_binary(&array).unwrap().value(0)
        );
    }

    /// The reader accepts what the writer produced, SRID header and all.
    #[test]
    fn ewkb_parses_back() {
        let array = points();
        let stamped = st_set_srid(&array, 4326).unwrap();
        let extended = st_as_ewkb(stamped.as_ref()).unwrap();

        let parsed = st_geom_from_wkb(&extended, Default::default()).unwrap();
        let text = st_as_text(parsed.as_ref()).unwrap();
        assert!(text.value(0).starts_with("POINT"));
        // The embedded SRID is dropped, as the module documentation says.
        assert_eq!(crate::crs::srid_of(&parsed.data_type()), 0);
    }

    #[test]
    fn geohash_round_trips_to_the_cell_centre() {
        let array = points();
        let hashes = st_geohash(&array, 9).unwrap();
        assert_eq!(hashes.value(0).len(), 9);
        assert!(hashes.is_null(1));

        let back = st_point_from_geohash(&hashes, Default::default()).unwrap();
        use geo_traits::to_geo::ToGeoGeometry;
        use geoarrow_array::GeoArrowArrayAccessor;
        let geo::Geometry::Point(p) = back.value(0).unwrap().to_geometry() else {
            panic!("expected a point")
        };
        // Nine characters give roughly five metres of precision.
        assert!((p.x() - 1.5).abs() < 1e-3, "x was {}", p.x());
        assert!((p.y() - 2.5).abs() < 1e-3, "y was {}", p.y());
    }

    #[test]
    fn geohash_of_a_non_point_is_null() {
        let rings: Vec<geo::LineString<f64>> = vec![geo::wkt! { LINESTRING(0.0 0.0,1.0 1.0) }];
        let array = geoarrow_array::builder::LineStringBuilder::from_line_strings(
            &rings,
            geoarrow_schema::LineStringType::new(Dimension::XY, Default::default()),
        )
        .finish();
        assert!(st_geohash(&array, 6).unwrap().is_null(0));
    }

    #[test]
    fn bad_geojson_is_an_error() {
        let text = StringArray::from(vec!["not json"]);
        assert!(st_geom_from_geojson(&text, Default::default()).is_err());
    }
}
