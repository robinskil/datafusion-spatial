//! `ST_SRID` and `ST_SetSRID`.
//!
//! # A real difference from PostGIS
//!
//! PostGIS stores an SRID inside every geometry value, so one column can hold rows in different
//! coordinate reference systems. GeoArrow stores the CRS once, in the field metadata of the whole
//! column.
//!
//! Two consequences follow, and both are visible to the user:
//!
//! 1. `ST_SRID` returns the same value for every row. It reads the schema, not the data.
//! 2. `ST_SetSRID` changes the type of the column, so its second argument must be a constant. A
//!    per-row SRID cannot be represented and is rejected at plan time rather than silently
//!    dropped.
//!
//! The column layout is the better one for an analytic engine, and it is what GeoParquet writes.

use std::sync::Arc;

use arrow_array::Int32Array;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::GeoArrowResult;
use geoarrow_schema::{Crs, CrsType, GeoArrowType, Metadata};

/// The SRID of a column, or zero when it carries no coordinate reference system.
///
/// Zero is what PostGIS reports for an unknown SRID.
pub fn srid_of(data_type: &GeoArrowType) -> i32 {
    let crs = data_type.metadata().crs();
    let Some(value) = crs.crs_value() else {
        return 0;
    };

    match crs.crs_type() {
        // Stored as the bare number, which is what a database driver writes.
        Some(CrsType::Srid) => scalar_string(value)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        // Stored as `AUTHORITY:CODE`, for example `EPSG:4326`.
        Some(CrsType::AuthorityCode) => scalar_string(value)
            .and_then(|s| s.rsplit_once(':').map(|(_, code)| code.to_string()))
            .and_then(|code| code.parse().ok())
            .unwrap_or(0),
        // PROJJSON carries the code under `id.code`.
        Some(CrsType::Projjson) => value
            .get("id")
            .and_then(|id| id.get("code"))
            .and_then(|code| {
                code.as_i64()
                    .and_then(|n| i32::try_from(n).ok())
                    .or_else(|| code.as_str().and_then(|s| s.parse().ok()))
            })
            .unwrap_or(0),
        // A full WKT2 description has no cheap code to read. Report unknown.
        Some(CrsType::Wkt2_2019) | None => 0,
    }
}

fn scalar_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

/// The same type with a different SRID.
pub fn with_srid(data_type: &GeoArrowType, srid: i32) -> GeoArrowType {
    let edges = data_type.metadata().edges();
    let crs = if srid == 0 {
        // PostGIS treats zero as "no SRID". Clear the CRS. Do not store a literal zero.
        Crs::default()
    } else {
        Crs::from_srid(srid.to_string())
    };
    data_type
        .clone()
        .with_metadata(Arc::new(Metadata::new(crs, edges)))
}

/// `ST_SRID`. One value per row, all the same, read from the column metadata.
pub fn st_srid(array: &dyn GeoArrowArray) -> GeoArrowResult<Int32Array> {
    let srid = srid_of(&array.data_type());
    Ok(Int32Array::new(
        vec![srid; array.len()].into(),
        array.logical_nulls(),
    ))
}

/// `ST_SetSRID`. Restamp the column metadata. The values are untouched.
///
/// This is metadata-only work, so it is `O(1)` in the row count.
pub fn st_set_srid(array: &dyn GeoArrowArray, srid: i32) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let metadata = with_srid(&array.data_type(), srid).metadata().clone();
    Ok(array.slice(0, array.len()).with_metadata(metadata))
}

#[cfg(test)]
mod tests {
    use geoarrow_array::builder::PointBuilder;
    use geoarrow_schema::{Dimension, PointType};

    use super::*;

    fn points(metadata: Arc<Metadata>) -> geoarrow_array::array::PointArray {
        let p0 = geo::point!(x: 1.0, y: 2.0);
        PointBuilder::from_points([p0].iter(), PointType::new(Dimension::XY, metadata)).finish()
    }

    #[test]
    fn no_crs_reports_zero() {
        let array = points(Default::default());
        let srid = st_srid(&array).unwrap();
        assert_eq!(srid.value(0), 0);
    }

    #[test]
    fn srid_round_trips() {
        let array = points(Default::default());
        let stamped = st_set_srid(&array, 4326).unwrap();
        assert_eq!(st_srid(stamped.as_ref()).unwrap().value(0), 4326);
        assert_eq!(srid_of(&stamped.data_type()), 4326);
    }

    #[test]
    fn authority_code_is_parsed() {
        let metadata = Arc::new(Metadata::new(
            Crs::from_authority_code("EPSG:3857".to_string()),
            None,
        ));
        let array = points(metadata);
        assert_eq!(st_srid(&array).unwrap().value(0), 3857);
    }

    #[test]
    fn projjson_id_code_is_parsed() {
        let projjson = serde_json::json!({
            "type": "GeographicCRS",
            "name": "WGS 84",
            "id": { "authority": "EPSG", "code": 4326 }
        });
        let metadata = Arc::new(Metadata::new(Crs::from_projjson(projjson), None));
        let array = points(metadata);
        assert_eq!(st_srid(&array).unwrap().value(0), 4326);
    }

    #[test]
    fn setting_zero_clears_the_crs() {
        let array = points(Default::default());
        let stamped = st_set_srid(&array, 4326).unwrap();
        let cleared = st_set_srid(stamped.as_ref(), 0).unwrap();
        assert_eq!(st_srid(cleared.as_ref()).unwrap().value(0), 0);
    }

    #[test]
    fn set_srid_keeps_the_values() {
        use geo_traits::to_geo::ToGeoGeometry;
        use geoarrow_array::cast::AsGeoArrowArray;
        use geoarrow_array::GeoArrowArrayAccessor;

        let array = points(Default::default());
        let stamped = st_set_srid(&array, 4326).unwrap();
        let geo::Geometry::Point(p) = stamped.as_point().value(0).unwrap().to_geometry() else {
            panic!("expected a point")
        };
        assert_eq!((p.x(), p.y()), (1.0, 2.0));
    }
}
