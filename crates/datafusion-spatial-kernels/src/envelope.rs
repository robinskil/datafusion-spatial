//! Bounding box functions.
//!
//! # A box array is already four `f64` buffers
//!
//! GeoArrow stores a box column as two separated coordinate buffers. One holds the lower corner
//! and one holds the upper corner. So `ST_XMin` over a box column is the `ST_X` trick again.
//! It hands back the buffer with its reference count raised, and copies nothing.
//!
//! | Input | `ST_XMin` |
//! |---|---|
//! | box column | the lower x buffer, no copy |
//! | anything else | one pass to compute boxes, then one new buffer |
//!
//! `ST_Envelope` builds that box column, and it reuses [`fill_bboxes`], the same pass the predicate
//! prefilter uses.

use std::sync::Arc;

use arrow_array::{BooleanArray, Float64Array};
use arrow_buffer::{BooleanBufferBuilder, NullBuffer, ScalarBuffer};
use geoarrow_array::array::{RectArray, SeparatedCoordBuffer};
use geoarrow_array::cast::AsGeoArrowArray;
use geoarrow_array::GeoArrowArray;
use geoarrow_schema::error::GeoArrowResult;
use geoarrow_schema::{BoxType, Dimension, GeoArrowType};

use crate::bbox::{fill_bboxes, Bbox};
use crate::predicate::{broadcast_len, broadcast_nulls};

/// Which corner ordinate to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bound {
    /// `ST_XMin`.
    XMin,
    /// `ST_YMin`.
    YMin,
    /// `ST_ZMin`.
    ZMin,
    /// `ST_XMax`.
    XMax,
    /// `ST_YMax`.
    YMax,
    /// `ST_ZMax`.
    ZMax,
}

impl Bound {
    /// The PostGIS function name.
    pub const fn function_name(self) -> &'static str {
        match self {
            Self::XMin => "ST_XMin",
            Self::YMin => "ST_YMin",
            Self::ZMin => "ST_ZMin",
            Self::XMax => "ST_XMax",
            Self::YMax => "ST_YMax",
            Self::ZMax => "ST_ZMax",
        }
    }

    /// The lowercase SQL name.
    pub const fn sql_name(self) -> &'static str {
        match self {
            Self::XMin => "st_xmin",
            Self::YMin => "st_ymin",
            Self::ZMin => "st_zmin",
            Self::XMax => "st_xmax",
            Self::YMax => "st_ymax",
            Self::ZMax => "st_zmax",
        }
    }

    /// Every bound, for registration.
    pub const ALL: [Self; 6] = [
        Self::XMin,
        Self::YMin,
        Self::ZMin,
        Self::XMax,
        Self::YMax,
        Self::ZMax,
    ];

    /// True when this reads the lower corner.
    const fn is_lower(self) -> bool {
        matches!(self, Self::XMin | Self::YMin | Self::ZMin)
    }

    /// Which ordinate inside the corner.
    const fn ordinate(self) -> usize {
        match self {
            Self::XMin | Self::XMax => 0,
            Self::YMin | Self::YMax => 1,
            Self::ZMin | Self::ZMax => 2,
        }
    }

    /// Read this bound out of a computed box.
    #[inline]
    const fn read(self, bbox: &Bbox) -> f64 {
        match self {
            Self::XMin => bbox.minx,
            Self::YMin => bbox.miny,
            Self::XMax => bbox.maxx,
            Self::YMax => bbox.maxy,
            // The box walker is two-dimensional, so it carries no z.
            Self::ZMin | Self::ZMax => f64::NAN,
        }
    }
}

/// Read one corner ordinate of every row's bounding box.
pub fn bound(array: &dyn GeoArrowArray, which: Bound) -> GeoArrowResult<Float64Array> {
    // Fast path. A box column already stores exactly these numbers.
    if let GeoArrowType::Rect(_) = array.data_type() {
        return Ok(rect_bound(array.as_rect(), which));
    }

    // A z bound needs a three-dimensional box, which the two-dimensional walker cannot give.
    if matches!(which, Bound::ZMin | Bound::ZMax) {
        return Ok(Float64Array::new_null(array.len()));
    }

    let mut boxes = Vec::new();
    fill_bboxes(array, &mut boxes)?;

    let mut values = Vec::with_capacity(boxes.len());
    values.extend(boxes.iter().map(|bbox| which.read(bbox)));
    Ok(Float64Array::new(values.into(), array.logical_nulls()))
}

/// The zero copy path: clone one of the four buffers a box column already holds.
fn rect_bound(array: &RectArray, which: Bound) -> Float64Array {
    let corner = if which.is_lower() {
        array.lower()
    } else {
        array.upper()
    };
    let dim = corner.dim();
    let index = which.ordinate();

    // A two-dimensional box has no z.
    if index >= dim.size() {
        return Float64Array::new_null(array.len());
    }

    Float64Array::new(corner.raw_buffers()[index].clone(), array.logical_nulls())
}

/// The type `ST_Envelope` and `ST_Expand` produce.
pub fn box_output_type(input: &GeoArrowType) -> BoxType {
    BoxType::new(Dimension::XY, Arc::clone(input.metadata()))
}

/// `ST_Envelope`. The bounding box of every row, as a box column.
///
/// A box column is already its own envelope, so that case is handed straight back.
pub fn st_envelope(
    array: &dyn GeoArrowArray,
    output: BoxType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    if let GeoArrowType::Rect(_) = array.data_type() {
        return Ok(array.slice(0, array.len()));
    }
    let mut boxes = Vec::new();
    fill_bboxes(array, &mut boxes)?;
    build_rect(&boxes, array.logical_nulls(), output)
}

/// `ST_Expand`. Grow every bounding box by a distance on all sides.
pub fn st_expand(
    array: &dyn GeoArrowArray,
    distance: &Float64Array,
    output: BoxType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    use arrow_array::Array;

    let mut boxes = Vec::new();
    fill_bboxes(array, &mut boxes)?;

    let broadcast = distance.len() == 1 && boxes.len() != 1;
    let mut nulls = array.logical_nulls();

    for (row, bbox) in boxes.iter_mut().enumerate() {
        let slot = if broadcast { 0 } else { row };
        if slot >= distance.len() || distance.is_null(slot) {
            *bbox = Bbox::EMPTY;
            continue;
        }
        *bbox = bbox.expand(distance.value(slot));
    }

    // A null distance makes the row null, on top of any null geometry.
    if !broadcast || distance.null_count() > 0 {
        let distance_nulls = if broadcast {
            distance
                .is_null(0)
                .then(|| NullBuffer::new_null(boxes.len()))
        } else {
            distance.nulls().cloned()
        };
        nulls = NullBuffer::union(nulls.as_ref(), distance_nulls.as_ref());
    }

    build_rect(&boxes, nulls, output)
}

/// Turn a slice of boxes into a GeoArrow box column.
fn build_rect(
    boxes: &[Bbox],
    nulls: Option<NullBuffer>,
    output: BoxType,
) -> GeoArrowResult<Arc<dyn GeoArrowArray>> {
    let mut min_x = Vec::with_capacity(boxes.len());
    let mut min_y = Vec::with_capacity(boxes.len());
    let mut max_x = Vec::with_capacity(boxes.len());
    let mut max_y = Vec::with_capacity(boxes.len());

    for bbox in boxes {
        // An empty box carries infinities, which no consumer expects. Write zeros and let the
        // null buffer say the row has no value.
        if bbox.is_empty() {
            min_x.push(0.0);
            min_y.push(0.0);
            max_x.push(0.0);
            max_y.push(0.0);
        } else {
            min_x.push(bbox.minx);
            min_y.push(bbox.miny);
            max_x.push(bbox.maxx);
            max_y.push(bbox.maxy);
        }
    }

    // An empty box is a row with no geometry, so fold that into the null buffer.
    let empty: Option<NullBuffer> = boxes
        .iter()
        .any(|bbox| bbox.is_empty())
        .then(|| boxes.iter().map(|bbox| !bbox.is_empty()).collect());
    let nulls = NullBuffer::union(nulls.as_ref(), empty.as_ref());

    let blank = ScalarBuffer::<f64>::from(Vec::<f64>::new());
    let lower = SeparatedCoordBuffer::from_array(
        [
            ScalarBuffer::from(min_x),
            ScalarBuffer::from(min_y),
            blank.clone(),
            blank.clone(),
        ],
        Dimension::XY,
    )?;
    let upper = SeparatedCoordBuffer::from_array(
        [
            ScalarBuffer::from(max_x),
            ScalarBuffer::from(max_y),
            blank.clone(),
            blank,
        ],
        Dimension::XY,
    )?;

    Ok(Arc::new(RectArray::new(
        lower,
        upper,
        nulls,
        output.metadata().clone(),
    )))
}

/// `ST_BBoxIntersects`, the PostGIS `&&` operator.
///
/// This is the bounding box test on its own, with no exact geometry work at all. It is what every
/// predicate in [`crate::predicate`] runs first, exposed so a query can ask for just that.
pub fn st_bbox_intersects(
    left: &dyn GeoArrowArray,
    right: &dyn GeoArrowArray,
) -> GeoArrowResult<BooleanArray> {
    let len = broadcast_len("ST_BBoxIntersects", left, right)?;
    let nulls = broadcast_nulls(left, right, len);

    let mut left_boxes = Vec::new();
    let mut right_boxes = Vec::new();
    fill_bboxes(left, &mut left_boxes)?;
    fill_bboxes(right, &mut right_boxes)?;

    let left_broadcast = left_boxes.len() == 1 && len != 1;
    let right_broadcast = right_boxes.len() == 1 && len != 1;

    let mut values = BooleanBufferBuilder::new(len);
    for index in 0..len {
        let lhs = left_boxes[if left_broadcast { 0 } else { index }];
        let rhs = right_boxes[if right_broadcast { 0 } else { index }];
        values.append(lhs.intersects(&rhs));
    }
    Ok(BooleanArray::new(values.finish(), nulls))
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use geoarrow_array::builder::{PointBuilder, PolygonBuilder};
    use geoarrow_schema::{CoordType, PointType, PolygonType};

    use super::*;

    fn polygons() -> geoarrow_array::array::PolygonArray {
        let unit: geo::Polygon<f64> =
            geo::wkt! { POLYGON((0.0 0.0,4.0 0.0,4.0 3.0,0.0 3.0,0.0 0.0)) };
        let none: Option<&geo::Polygon<f64>> = None;
        PolygonBuilder::from_nullable_polygons(
            &[Some(&unit), none],
            PolygonType::new(Dimension::XY, Default::default()),
        )
        .finish()
    }

    fn points() -> geoarrow_array::array::PointArray {
        PointBuilder::from_points(
            [geo::point!(x: 1.0, y: 2.0), geo::point!(x: 5.0, y: 6.0)].iter(),
            PointType::new(Dimension::XY, Default::default()).with_coord_type(CoordType::Separated),
        )
        .finish()
    }

    #[test]
    fn envelope_of_a_polygon() {
        let array = polygons();
        let output = box_output_type(&array.data_type());
        let envelope = st_envelope(&array, output).unwrap();

        assert_eq!(bound(envelope.as_ref(), Bound::XMin).unwrap().value(0), 0.0);
        assert_eq!(bound(envelope.as_ref(), Bound::YMin).unwrap().value(0), 0.0);
        assert_eq!(bound(envelope.as_ref(), Bound::XMax).unwrap().value(0), 4.0);
        assert_eq!(bound(envelope.as_ref(), Bound::YMax).unwrap().value(0), 3.0);
        assert!(
            bound(envelope.as_ref(), Bound::XMin).unwrap().is_null(1),
            "the null row stays null"
        );
    }

    /// The claim from the module docs, proven on the buffers.
    #[test]
    fn bounds_are_zero_copy_on_a_box_column() {
        let array = polygons();
        let output = box_output_type(&array.data_type());
        let envelope = st_envelope(&array, output).unwrap();

        let rect = envelope.as_rect();
        let lower_x = rect.lower().raw_buffers()[0].as_ptr();
        let upper_y = rect.upper().raw_buffers()[1].as_ptr();

        let xmin = bound(envelope.as_ref(), Bound::XMin).unwrap();
        assert_eq!(
            xmin.values().as_ptr(),
            lower_x,
            "ST_XMin must hand back the lower x buffer, not a copy"
        );
        let ymax = bound(envelope.as_ref(), Bound::YMax).unwrap();
        assert_eq!(ymax.values().as_ptr(), upper_y);
    }

    #[test]
    fn envelope_of_a_box_is_itself() {
        let array = polygons();
        let output = box_output_type(&array.data_type());
        let once = st_envelope(&array, output.clone()).unwrap();
        let twice = st_envelope(once.as_ref(), output).unwrap();

        assert_eq!(
            bound(twice.as_ref(), Bound::XMax).unwrap().value(0),
            bound(once.as_ref(), Bound::XMax).unwrap().value(0)
        );
    }

    #[test]
    fn bounds_work_without_a_box_column() {
        let array = points();
        assert_eq!(bound(&array, Bound::XMin).unwrap().value(0), 1.0);
        assert_eq!(bound(&array, Bound::XMax).unwrap().value(1), 5.0);
        // A point's box is degenerate, so min equals max.
        assert_eq!(
            bound(&array, Bound::YMin).unwrap().value(0),
            bound(&array, Bound::YMax).unwrap().value(0)
        );
    }

    /// The box walker is two-dimensional, so a z bound is null rather than a guess.
    #[test]
    fn z_bounds_are_null_in_two_dimensions() {
        let array = points();
        assert!(bound(&array, Bound::ZMin).unwrap().is_null(0));

        let output = box_output_type(&array.data_type());
        let envelope = st_envelope(&array, output).unwrap();
        assert!(bound(envelope.as_ref(), Bound::ZMax).unwrap().is_null(0));
    }

    #[test]
    fn expand_grows_the_box() {
        let array = points();
        let output = box_output_type(&array.data_type());
        let distance = Float64Array::from(vec![2.0]);

        let expanded = st_expand(&array, &distance, output).unwrap();
        assert_eq!(
            bound(expanded.as_ref(), Bound::XMin).unwrap().value(0),
            -1.0
        );
        assert_eq!(bound(expanded.as_ref(), Bound::XMax).unwrap().value(0), 3.0);
        assert_eq!(bound(expanded.as_ref(), Bound::YMin).unwrap().value(0), 0.0);
    }

    #[test]
    fn expand_with_a_null_distance_is_null() {
        let array = points();
        let output = box_output_type(&array.data_type());
        let distance = Float64Array::from(vec![None::<f64>, Some(1.0)]);

        let expanded = st_expand(&array, &distance, output).unwrap();
        let xmin = bound(expanded.as_ref(), Bound::XMin).unwrap();
        assert!(xmin.is_null(0));
        assert_eq!(xmin.value(1), 4.0);
    }

    #[test]
    fn bbox_intersects_is_the_box_test_alone() {
        let array = points();
        let far = PointBuilder::from_points(
            [geo::point!(x: 1.0, y: 2.0), geo::point!(x: 99.0, y: 99.0)].iter(),
            PointType::new(Dimension::XY, Default::default()),
        )
        .finish();

        let hits = st_bbox_intersects(&array, &far).unwrap();
        assert!(hits.value(0), "the same point");
        assert!(!hits.value(1), "far apart");
    }

    /// A box column and a polygon column must answer the operator the same way.
    #[test]
    fn bbox_intersects_agrees_with_the_envelope() {
        let array = polygons();
        let output = box_output_type(&array.data_type());
        let envelope = st_envelope(&array, output).unwrap();

        let direct = st_bbox_intersects(&array, &array).unwrap();
        let through_boxes = st_bbox_intersects(envelope.as_ref(), envelope.as_ref()).unwrap();
        assert_eq!(direct, through_boxes);
    }
}
