# datafusion-spatial

PostGIS-compatible spatial functions for [Apache DataFusion](https://datafusion.apache.org).
The crate builds on [GeoArrow](https://geoarrow.org) and the [`geo`](https://docs.rs/geo) crate.

This project is not part of the PostGIS project. The function names follow the PostGIS API.

## Status

The crate registers 122 functions: 117 scalar, 3 aggregate and 2 window.
`ST_Transform` adds one more function. It needs the optional `proj` feature.

## Function reference

Call each function by its PostGIS name. Each table gives the signature that this crate accepts.

| | Meaning |
|---|---|
| ✅ | The name, the arguments and the behaviour match PostGIS. |
| ⚠️ | The function works. Read the note. The arguments or the behaviour differ. |
| ❌ | The crate does not have this function. |

### Accessors

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_X(geom)` | `ST_X(geom)` | ✅ | Zero copy on a point column with separated coordinates. |
| `ST_Y(geom)` | `ST_Y(geom)` | ✅ | |
| `ST_Z(geom)` | `ST_Z(geom)` | ✅ | Returns null for a 2D geometry. |
| `ST_M(geom)` | `ST_M(geom)` | ✅ | Returns null without a measure. |
| `ST_SRID(geom)` | `ST_SRID(geom)` | ⚠️ | Reads the column metadata. Every row returns the same value. |
| `ST_GeometryType(geom)` | `ST_GeometryType(geom)` | ✅ | The schema answers this for a single-type column. |
| `ST_Dimension(geom)` | `ST_Dimension(geom)` | ✅ | |
| `ST_CoordDim(geom)` | `ST_CoordDim(geom)` | ✅ | |
| `ST_NPoints(geom)` | `ST_NPoints(geom)` | ✅ | |
| `ST_NumPoints(geom)` | `ST_NumPoints(geom)` | ✅ | Accepts a line string only. Other types return null. PostGIS does the same. |
| `ST_NumGeometries(geom)` | `ST_NumGeometries(geom)` | ✅ | |
| `ST_NumInteriorRings(geom)` | `ST_NumInteriorRings(geom)` | ✅ | |
| `ST_IsEmpty(geom)` | `ST_IsEmpty(geom)` | ✅ | |
| `ST_IsClosed(geom)` | `ST_IsClosed(geom)` | ✅ | |
| `ST_IsRing(geom)` | `ST_IsRing(geom)` | ✅ | |
| `ST_IsSimple(geom)` | `ST_IsSimple(geom)` | ⚠️ | Follows the JTS rule. An areal geometry is always simple. Use `ST_IsValid` for those. |

### Components

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_StartPoint(geom)` | `ST_StartPoint(geom)` | ✅ | |
| `ST_EndPoint(geom)` | `ST_EndPoint(geom)` | ✅ | |
| `ST_PointN(geom, n)` | `ST_PointN(geom, n)` | ✅ | The index starts at 1. An index outside the line returns null. |
| `ST_ExteriorRing(geom)` | `ST_ExteriorRing(geom)` | ✅ | |
| `ST_InteriorRingN(geom, n)` | `ST_InteriorRingN(geom, n)` | ✅ | |
| `ST_GeometryN(geom, n)` | `ST_GeometryN(geom, n)` | ✅ | Index 1 returns the input itself when the input is not a collection. |

### Constructors

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Point(x, y)` | `ST_Point(x, y[, z])` | ⚠️ | The function adopts the input buffers and copies nothing. A third argument sets z, not the SRID. |
| `ST_MakePoint(x, y[, z])` | `ST_MakePoint(x, y[, z])` | ✅ | |
| `ST_PointZ(x, y, z)` | `ST_PointZ(x, y[, z])` | ⚠️ | This name calls `ST_MakePoint`. Two arguments return a 2D point. PostGIS needs three arguments. |
| `ST_MakeLine(a, b)` | `ST_MakeLine(a, b)` | ⚠️ | Joins two point columns into two-point lines. The crate has no aggregate form and no array form. |
| `ST_MakePolygon(ring)` | `ST_MakePolygon(ring)` | ⚠️ | Builds the shell only. The crate does not accept the second argument for holes. |
| `ST_MakeEnvelope(x1, y1, x2, y2)` | `ST_MakeEnvelope(x1, y1, x2, y2)` | ⚠️ | The function adopts all four input buffers. The crate does not accept the fifth `srid` argument. |
| `ST_MakeBox2D(pt1, pt2)` | `ST_MakeBox2D(x1, y1, x2, y2)` | ⚠️ | **The arguments differ.** PostGIS takes two points. This crate takes four ordinates. |

### Input and output

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_AsText(geom)` | `ST_AsText(geom)` | ⚠️ | The crate does not accept the `maxdecimaldigits` argument. |
| `ST_AsBinary(geom)` | `ST_AsBinary(geom)` | ✅ | |
| `ST_AsEWKB(geom)` | `ST_AsEWKB(geom)` | ✅ | The SRID comes from the column metadata. |
| `ST_AsGeoJSON(geom)` | `ST_AsGeoJSON(geom)` | ⚠️ | Accepts a geometry only. The crate does not accept the row form or the options argument. |
| `ST_GeomFromText(wkt)` | `ST_GeomFromText(wkt)` | ⚠️ | The crate does not accept the `srid` argument. Call `ST_SetSRID` after it. |
| `ST_GeomFromWKB(bytea)` | `ST_GeomFromWKB(bytea)` | ✅ | |
| `ST_GeomFromEWKB(bytea)` | `ST_GeomFromEWKB(bytea)` | ⚠️ | Reads the extended format. **The function loses the SRID inside the value.** Call `ST_SetSRID` after it. |
| `ST_GeomFromGeoJSON(json)` | `ST_GeomFromGeoJSON(json)` | ✅ | |
| `ST_GeoHash(geom[, prec])` | `ST_GeoHash(geom[, prec])` | ⚠️ | Accepts a point only. Other types return null. The default precision is 20. |
| `ST_PointFromGeoHash(hash)` | `ST_PointFromGeoHash(hash)` | ⚠️ | Returns the centre of the cell. The crate does not accept the precision argument. |

### Predicates

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Intersects(a, b)` | `ST_Intersects(a, b)` | ✅ | The box test runs first. A point against a constant polygon takes the edge index. |
| `ST_Disjoint(a, b)` | `ST_Disjoint(a, b)` | ✅ | Two separate boxes prove this predicate true. Otherwise as `ST_Intersects`. |
| `ST_Contains(a, b)` | `ST_Contains(a, b)` | ✅ | The box of `b` must sit inside the box of `a`. A point against a constant polygon takes the edge index. |
| `ST_ContainsProperly(a, b)` | `ST_ContainsProperly(a, b)` | ✅ | Same as `ST_Contains` for a point argument. |
| `ST_Within(a, b)` | `ST_Within(a, b)` | ✅ | The converse of `ST_Contains`, and it takes the same edge index. |
| `ST_Covers(a, b)` | `ST_Covers(a, b)` | ✅ | The boundary counts. A point against a constant polygon takes the edge index. |
| `ST_CoveredBy(a, b)` | `ST_CoveredBy(a, b)` | ✅ | The converse of `ST_Covers`, and it takes the same edge index. |
| `ST_Touches(a, b)` | `ST_Touches(a, b)` | ✅ | Uses the DE-9IM matrix. A constant argument gets the cached R-tree. |
| `ST_Crosses(a, b)` | `ST_Crosses(a, b)` | ✅ | |
| `ST_Overlaps(a, b)` | `ST_Overlaps(a, b)` | ✅ | |
| `ST_Equals(a, b)` | `ST_Equals(a, b)` | ✅ | Tests topological equality. The two boxes must match. |
| `ST_Relate(a, b)` | `ST_Relate(a, b)` | ✅ | Returns the nine character matrix. |
| `ST_Relate(a, b, pattern)` | `ST_Relate(a, b, pattern)` | ⚠️ | The pattern must be a constant. The crate does not accept the `boundaryNodeRule` form. |
| `ST_DWithin(a, b, d)` | `ST_DWithin(a, b, d)` | ⚠️ | `d` must be a constant. It grows the box that drives the prefilter. |
| `ST_DFullyWithin(a, b, d)` | `ST_DFullyWithin(a, b, d)` | ⚠️ | `d` must be a constant. |
| `geom_a && geom_b` | `ST_BBoxIntersects(a, b)` | ⚠️ | **This is a function, not an operator.** DataFusion has no user operators. |

### Measurement

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Area(geom)` | `ST_Area(geom)` | ⚠️ | Planar. Longitude and latitude data returns square degrees. The PostGIS `geometry` type does the same. |
| `ST_Length(geom)` | `ST_Length(geom)` | ⚠️ | Planar. Measures lineal parts only. A polygon returns zero, as in PostGIS. |
| `ST_Perimeter(geom)` | `ST_Perimeter(geom)` | ⚠️ | Planar. Measures areal parts only. |
| `ST_Distance(a, b)` | `ST_Distance(a, b)` | ⚠️ | Planar. |
| `ST_MaxDistance(a, b)` | `ST_MaxDistance(a, b)` | ⚠️ | Exact. The cost is the product of the two vertex counts. |
| `ST_HausdorffDistance(a, b)` | `ST_HausdorffDistance(a, b)` | ⚠️ | The crate does not accept the `densifyFrac` argument. |
| `ST_FrechetDistance(a, b)` | `ST_FrechetDistance(a, b)` | ⚠️ | Accepts a line string only. Other types return null. |
| `ST_DistanceSphere(a, b)` | `ST_DistanceSphere(a, b)` | ⚠️ | **Accepts two points only.** Other types return null. `geo` has no spherical nearest-point search. |
| `ST_DistanceSpheroid(a, b)` | `ST_DistanceSpheroid(a, b)` | ⚠️ | Accepts two points only. Uses WGS 84. The crate does not accept a spheroid argument. |

### Linear reference

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_ClosestPoint(a, b)` | `ST_ClosestPoint(a, b)` | ✅ | |
| `ST_ShortestLine(a, b)` | `ST_ShortestLine(a, b)` | ✅ | |
| `ST_LineLocatePoint(line, pt)` | `ST_LineLocatePoint(line, pt)` | ✅ | Accepts a line string only. Other types return null. |
| `ST_LineInterpolatePoint(line, f)` | `ST_LineInterpolatePoint(line, f)` | ✅ | A fraction outside 0 to 1 returns null. |

### Overlay

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Union(a, b)` | `ST_Union(a, b)` | ⚠️ | Accepts areal arguments only. Other types return null. Returns a mixed geometry column. |
| `ST_Intersection(a, b)` | `ST_Intersection(a, b)` | ⚠️ | Accepts areal arguments only. Other types return null. |
| `ST_Difference(a, b)` | `ST_Difference(a, b)` | ⚠️ | Accepts areal arguments only. Other types return null. |
| `ST_SymDifference(a, b)` | `ST_SymDifference(a, b)` | ⚠️ | Accepts areal arguments only. Other types return null. |

### Process

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Buffer(geom, d)` | `ST_Buffer(geom, d)` | ⚠️ | Uses round joins and round caps. The crate does not accept `quad_segs` or a style string. |
| `ST_ConvexHull(geom)` | `ST_ConvexHull(geom)` | ✅ | |
| `ST_ConcaveHull(geom, pct)` | `ST_ConcaveHull(geom, pct)` | ⚠️ | The crate maps `target_percent` onto the `geo` concavity, which runs the other way. No hole support. |
| `ST_OrientedEnvelope(geom)` | `ST_OrientedEnvelope(geom)` | ✅ | |
| `ST_Boundary(geom)` | `ST_Boundary(geom)` | ⚠️ | The crate implements the OGC rule, because `geo` has no boundary algorithm. A collection returns an empty boundary. |
| `ST_Centroid(geom)` | `ST_Centroid(geom)` | ✅ | |
| `ST_PointOnSurface(geom)` | `ST_PointOnSurface(geom)` | ✅ | |
| `ST_Simplify(geom, tol)` | `ST_Simplify(geom, tol)` | ⚠️ | Uses Ramer-Douglas-Peucker. The crate does not accept the `preserveCollapsed` flag. |
| `ST_SimplifyVW(geom, tol)` | `ST_SimplifyVW(geom, tol)` | ✅ | Uses Visvalingam-Whyatt. |
| `ST_SimplifyPreserveTopology(geom, tol)` | None | ❌ | `geo` has no topology-safe simplify. `ST_SimplifyVW` is close. It gives no guarantee, so this crate does not alias it. |
| `ST_Segmentize(geom, max)` | `ST_Segmentize(geom, max)` | ⚠️ | Planar. A length of zero or less returns null. |
| `ST_RemoveRepeatedPoints(geom)` | `ST_RemoveRepeatedPoints(geom)` | ⚠️ | The crate does not accept a tolerance argument. |
| `ST_Reverse(geom)` | `ST_Reverse(geom)` | ✅ | |
| `ST_ForcePolygonCCW(geom)` | `ST_ForcePolygonCCW(geom)` | ✅ | |
| `ST_ForcePolygonCW(geom)` | `ST_ForcePolygonCW(geom)` | ✅ | |
| `ST_Force2D(geom)` | `ST_Force2D(geom)` | ⚠️ | Accepts a native GeoArrow column only. Zero copy: the function drops one buffer handle. |
| `ST_Force3D(geom)` | `ST_Force3D(geom)` | ⚠️ | Accepts a native GeoArrow column only. The function adds a z of zero. |
| `ST_FlipCoordinates(geom)` | `ST_FlipCoordinates(geom)` | ✅ | Zero copy on separated coordinates. The function swaps two buffer handles. |
| `ST_SetSRID(geom, srid)` | `ST_SetSRID(geom, srid)` | ⚠️ | `srid` must be a constant, because it changes the column type. The function reads no row. |

### Validity

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_IsValid(geom)` | `ST_IsValid(geom)` | ✅ | |
| `ST_IsValidReason(geom)` | `ST_IsValidReason(geom)` | ⚠️ | A valid input returns `Valid Geometry`, as in PostGIS. The failure text comes from `geo`. |
| `ST_MakeValid(geom)` | `ST_MakeValid(geom)` | ⚠️ | Repairs an areal geometry. Other types pass through. The crate does not accept an options argument. |

### Affine

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Translate(geom, dx, dy)` | `ST_Translate(geom, dx, dy)` | ⚠️ | The offsets must be constants. No 3D form. The output keeps the input geometry type. |
| `ST_Scale(geom, xf, yf)` | `ST_Scale(geom, xf, yf)` | ⚠️ | Scales about the origin. The crate does not accept the factor-geometry form or the origin form. |
| `ST_Rotate(geom, rad)` | `ST_Rotate(geom, rad)` | ⚠️ | Takes radians. Rotates about the origin, as in PostGIS. No origin-point form. |
| `ST_Affine(geom, a,b,d,e,xoff,yoff)` | `ST_Affine(geom, a,b,d,e,xoff,yoff)` | ⚠️ | Supports the 2D form only. The crate does not accept the twelve-argument 3D form. |

### Bounding box

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Envelope(geom)` | `ST_Envelope(geom)` | ⚠️ | Returns a GeoArrow box column. It reads as `ST_Polygon`. PostGIS returns a polygon geometry. |
| `ST_Expand(geom, d)` | `ST_Expand(geom, d)` | ⚠️ | Returns a box column. The crate does not accept the per-axis form. |
| `ST_XMin(box)` | `ST_XMin(geom)` | ✅ | Zero copy on a box column. A box column holds four `f64` buffers. |
| `ST_YMin(box)` | `ST_YMin(geom)` | ✅ | |
| `ST_XMax(box)` | `ST_XMax(geom)` | ✅ | |
| `ST_YMax(box)` | `ST_YMax(geom)` | ✅ | |
| `ST_ZMin(box)` | `ST_ZMin(geom)` | ⚠️ | Always returns null. The box pass is two-dimensional. |
| `ST_ZMax(box)` | `ST_ZMax(geom)` | ⚠️ | Always returns null. |

### Aggregates

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Extent(geom)` | `ST_Extent(geom)` | ✅ | The state is four `f64`. The function builds no geometry. |
| `ST_Collect(geom)` | `ST_Collect(geom)` | ⚠️ | Always returns a `GEOMETRYCOLLECTION`. PostGIS returns a `MULTI` type for one input type. |
| `ST_Union(geom)` | `ST_MemUnion(geom)` | ⚠️ | **The name differs.** One name cannot serve both registries. Read the note below. |
| `ST_MemUnion(geom)` | `ST_MemUnion(geom)` | ✅ | |
| `ST_Collect(geom[])` | None | ❌ | The crate does not implement the array form. |
| `ST_MakeLine(geom)` | None | ❌ | The crate does not implement the aggregate form. |
| `ST_Polygonize(geom)` | None | ❌ | `geo` has no equivalent. |

### Tessellation

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_DelaunayTriangles(geom)` | `ST_DelaunayTriangles(geom)` | ⚠️ | Always returns a collection. Supports the unconstrained triangulation only. No tolerance and no flags. |
| `ST_VoronoiPolygons(geom)` | `ST_VoronoiPolygons(geom)` | ⚠️ | Clips each cell to the input extent plus 50 percent. PostGIS uses the same default. |
| `ST_VoronoiLines(geom)` | `ST_VoronoiLines(geom)` | ✅ | |
| `ST_ChaikinSmoothing(geom, n)` | `ST_ChaikinSmoothing(geom, n)` | ⚠️ | The limit is eight iterations, because each one doubles the vertex count. No `preserve_end_points` flag. |

### Bearings

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Azimuth(a, b)` | `ST_Azimuth(a, b)` | ⚠️ | Returns radians clockwise from north, on WGS 84. Accepts two points only. Two equal points return null. |
| `ST_Project(pt, dist, azim)` | `ST_Project(pt, dist, azim)` | ⚠️ | Takes metres and radians on WGS 84. Accepts a point only. No two-point form. |

### Edits

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Multi(geom)` | `ST_Multi(geom)` | ✅ | |
| `ST_Points(geom)` | `ST_Points(geom)` | ✅ | Returns every coordinate as one multi point. |
| `ST_SnapToGrid(geom, size)` | `ST_SnapToGrid(geom, size)` | ⚠️ | One grid size serves both axes. No origin form. A size of zero or less returns null. |
| `ST_AddPoint(line, pt[, pos])` | `ST_AddPoint(line, pt[, pos])` | ✅ | The index starts at 0. The function appends when you omit the position. |
| `ST_RemovePoint(line, pos)` | `ST_RemovePoint(line, pos)` | ⚠️ | The function returns null when the line would keep fewer than two vertices. |
| `ST_SetPoint(line, pos, pt)` | `ST_SetPoint(line, pos, pt)` | ✅ | Note the PostGIS order. The position comes before the point. |
| `ST_Dump(geom)` | `ST_Dump(geom)` | ⚠️ | **Returns a list of WKB, not a set.** Call `unnest` to expand it. Read the note below. |
| `ST_ForceCollection(geom)` | None | ❌ | The function would do nothing. The GeoArrow mixed encoding unwraps a one-element collection. |

### Clusters

These two are window functions, as in PostGIS. Add `OVER ()` to the call.

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_ClusterKMeans(geom, k)` | `ST_ClusterKMeans(geom, k)` | ⚠️ | Clusters the centroids. A fixed seed makes the query repeatable. No `max_radius` argument. |
| `ST_ClusterDBSCAN(geom, eps, min)` | `ST_ClusterDBSCAN(geom, eps, min)` | ⚠️ | Clusters the centroids. PostGIS uses the whole geometry, so a large row can differ. Noise returns null. |

### Reprojection

This function needs the `proj` feature. Read [PROJ](#proj) below.

| PostGIS | datafusion-spatial | | Notes |
|---|---|:--:|---|
| `ST_Transform(geom, srid)` | `ST_Transform(geom, srid)` | ⚠️ | The target SRID must be a constant. The source comes from the column metadata. No PROJ string form. |

### Not implemented

| Area | PostGIS functions | Reason |
|---|---|---|
| Geography type | every `geography` overload | The crate needs a second type and a spherical algorithm set. |
| Reprojection extras | `ST_Transform(geom, to_proj)` | The function must accept a raw PROJ string. Only the argument parse step is absent. |
| Edits | `ST_Snap`, `ST_Node`, `ST_Split`, `ST_LineMerge`, `ST_LineSubstring`, `ST_Subdivide` | `geo` has no equivalent. Each one needs new code. |
| 3D | `ST_3DDistance`, `ST_3DIntersects`, `ST_3DLength` and the rest | `geo` is two-dimensional. |
| Other output | `ST_AsGML`, `ST_AsKML`, `ST_AsSVG`, `ST_AsMVT` | The crate needs format writers. `geozero` covers several of them. |
| Topology-safe simplify | `ST_SimplifyPreserveTopology` | `geo` has no equivalent. |
| Set output | `ST_DumpPoints`, `ST_DumpRings` | These have the same shape as `ST_Dump`. `ST_Points` covers the common use. |

## The `ST_Union` name

PostGIS gives the name `ST_Union` to two functions. Two arguments make a scalar function.
One argument makes an aggregate.

DataFusion reads the scalar registry first. It does not read the aggregate registry after
an argument count mismatch. So one name cannot serve both.

The scalar function keeps the name. The aggregate takes the name `ST_MemUnion`.
PostGIS also defines `ST_MemUnion` for this operation.

## The `ST_Dump` output

In PostGIS, `ST_Dump` returns a set. One input row becomes many output rows.
A DataFusion scalar function cannot return a set. So this function returns a list.
Call `unnest` to expand the list:

```sql
SELECT ST_AsText(unnest(ST_Dump(geom))) AS part FROM shapes
```

The parts are WKB, not GeoArrow geometries. This crate has no choice here.
GeoArrow marks a column as spatial through the field metadata.
The DataFusion `unnest` step drops the metadata on the child field of the list.
A list of geometries then arrives as a plain struct. No spatial function accepts it.

WKB avoids the problem. A plain `Binary` column is always WKB under the rule below.
So the parts stay usable without a cast.

## PROJ

`ST_Transform` needs [PROJ](https://proj.org), a C++ library. The feature is off by default.
The crate has no native dependency until you turn the feature on.

Choose one of two ways to link PROJ:

```toml
# Link the PROJ on the machine. This build is fast. It needs PROJ and pkg-config.
datafusion-spatial = { version = "0.1", features = ["proj"] }

# Build PROJ from source and link it as a static library.
# This build needs a C++ toolchain, CMake, sqlite3 and libtiff.
datafusion-spatial = { version = "0.1", features = ["proj-bundled"] }
```

### A limit of the bundled build

The bundled build makes its own `libproj.a` and its own `proj.db`.
PROJ finds that database through a path from build time. The path points into `target/`.

This works while you build and test in place. It fails after you move the program.
It also fails after `cargo clean`.

To move the program, do these steps:

1. Copy the `share/proj` directory next to the program.
2. Set `PROJ_DATA` to that directory.

```bash
export PROJ_DATA=/path/to/share/proj
```

PROJ cannot read an EPSG code without the database. `ST_Transform` then fails at plan time.
It does not return wrong coordinates.

## Which DataFusion version

The crate supports more than one DataFusion major. Pick one with a feature.

| Feature | DataFusion | Arrow | GeoArrow |
|---|---|---|---|
| `df53` (default) | 53 | 58 | 0.8 |
| `df54` | 54 | 58 | 0.8 |

Take the default:

```toml
datafusion-spatial = "0.1"
```

Take DataFusion 54 instead:

```toml
datafusion-spatial = { version = "0.1", default-features = false, features = ["df54", "sql"] }
```

Turn off the default before you name another version. Two version features at once is a build
error, and so is none.

The `sql` feature adds the DataFusion SQL parser. `SessionContext::sql` needs it. The library
itself does not, so you can leave it off.

### Read DataFusion through this crate

```rust
use datafusion_spatial::datafusion;
```

That re-export is the DataFusion this build links against. It cannot drift from the version the
functions were compiled against. A direct dependency can drift to another version. The types then no longer match.

### How to add a version

A new major needs one thing above all: it must agree with GeoArrow on the arrow version. Two arrow
versions in one build compile, then fail at the type level.

1. Add a `datafusion-NN` row to `[workspace.dependencies]` in the root `Cargo.toml`.
2. Add a `dfNN` feature to `datafusion-spatial-udf` and to `datafusion-spatial`.
3. Add the `extern crate` line to both crate roots, next to the others.
4. Add a row to the `test` and `arrow-agreement` matrices in `.github/workflows/ci.yml`.
5. Guard any API that moved with `#[cfg(feature = "dfNN")]`.

Step 5 is small today. DataFusion 54 dropped `as_any` from the UDF traits and from
`ExecutionPlan`. Those are the only two differences this crate meets.

## Usage

```rust
use datafusion::prelude::SessionContext;

let ctx = SessionContext::new();
datafusion_spatial::register_all(&ctx);

let df = ctx.sql("SELECT ST_X(ST_GeomFromText('POINT(1 2)'))").await?;
df.show().await?;
```

## Crates

| Crate | Contents |
|---|---|
| `datafusion-spatial` | The front crate. It holds `register_all`. Depend on this one. |
| `datafusion-spatial-udf` | The DataFusion wrappers for each function. |
| `datafusion-spatial-kernels` | Array in, array out. No query engine dependency. |

The kernels crate holds every speed-sensitive code path.
You can benchmark and profile that crate without a query engine.

## Versions

| Crate | Version |
|---|---|
| `datafusion` | 53 |
| `arrow` | 58 |
| `geoarrow-array`, `geoarrow-schema` | 0.8 |
| `geo` | 0.33 |

Every crate sets `#![forbid(unsafe_code)]`.

## Design

**Zero copy where the layout permits it.** `ST_X` on a point column with separated coordinates
returns the x buffer. The function raises the reference count. It allocates nothing and copies
nothing. Two tests prove this claim. One test compares the buffer pointers. One test sets an
allocation budget of zero blocks.

**A box test before a topology test.** A box test is four comparisons. An exact test builds a
topology graph. A single overlap filter helps `ST_Intersects` only. So each predicate gets its own
rule. Containment needs one box inside the other. Equality needs two equal boxes. Two separate
boxes prove `ST_Disjoint`. The measured gain is 8.6 times for `ST_Intersects` and 15 times for
`ST_Touches`.

**The correct algorithm, not the clever one.** `geo` offers a direct trait and a DE-9IM matrix for
most predicates. The direct trait wins by about 7 times. So `ST_Contains` and its relatives use it.
The crate builds the R-tree cache only for the four predicates that need the matrix.

**An edge index for a point against a polygon.** A direct trait still reads every edge of the ring
for every row. PostGIS indexes the ring edges by their y interval and reads only the edges that
cross the row. Every direct predicate takes that path against a constant polygon of 16 coordinates
or more: `ST_Intersects`, `ST_Disjoint`, `ST_Contains`, `ST_ContainsProperly`, `ST_Within`,
`ST_Covers` and `ST_CoveredBy`. One indexed verdict answers all seven. The two paths cross at
about 13 vertices, which is where that threshold comes from.

Over 8192 probes against a 5000 vertex ring, one batch drops from 19.5 ms to 360 microseconds.
`ST_Covers` and `ST_CoveredBy` gain far more, from 7.1 seconds to 360 microseconds, because `geo`
answers that pair of `Geometry` values from the DE-9IM matrix and no direct algorithm existed.

The index follows the `geo` rule for rings, so no answer changes.

**A repeated point row costs nothing.** A point column often repeats a coordinate on neighbouring
rows. A denormalized table that carries the location of a store on every event row looks like
that. For a point the bounding box is the coordinate, and the loop has already read the box. So
two comparisons settle whether the row repeats the one before it, and a repeat reuses that answer.
It skips the geometry build as well as the exact test. A column of one repeated point runs twelve
times faster. A column of distinct points pays between minus one and plus two per cent, which is
inside the noise of the benchmark. A cache behind the exact test instead of in front of the build
does not pay; `benches/caching.rs` prices both.

**The extension metadata is a correctness requirement.** Every geometry function implements
`return_field_from_args`, not `return_type`. A plain `DataType` cannot hold the GeoArrow extension
metadata. `return_type` would drop the coordinate reference system. The next function in the chain
would then fail.

**One downcast for each batch.** A kernel calls `downcast_geoarrow_array!` once. It then runs one
loop. The loop has no type match for each row. The cheap path has no virtual call.

**One coordinate buffer match for each batch.** `CoordBuffer` is an enum. The generic accessor
matches it again for every coordinate it reads. `materialize.rs` matches it once, then reads plain
`f64` slices. It also reuses one `geo::Geometry` across the rows of a column, so the loop allocates
nothing after the first row. Together these make a two-column `ST_Intersects` 3.8 times faster. A
cache does not help here. `benches/caching.rs` prices three caches and all three lose. The same
rule speeds `ST_Collect` by 5.5 times and `ST_ClusterKMeans` by 2.1 times on large polygons.

**The offsets hold the structure, so a transform is cheap.** A GeoArrow array is a coordinate
buffer and some offset buffers. A coordinate transform keeps the offsets valid.
`ST_FlipCoordinates` swaps two buffer handles on separated coordinates. `ST_Force2D` drops one
handle. Both work for every geometry type, not for points only. `ST_MakePolygon` reuses the offsets
of a line string as the ring offsets of a polygon.

**The schema answers some functions.** A single-type column already answers `ST_GeometryType`,
`ST_Dimension` and `ST_CoordDim`. Those kernels fill a constant array. They read no row.

**A box column is four `f64` buffers.** So `ST_XMin` hands back one buffer, like `ST_X` on a point
column. The cost is zero heap blocks. `ST_Envelope` builds that column from the same box pass that
the predicate prefilter runs.

## The spatial join

DataFusion plans `ST_Intersects(a.geom, b.geom)` in a join as a nested loop. It visits every pair,
so the cost is the product of the two row counts.

This crate can replace that with a grid-driven join. Install the rule when you build the session:

```rust
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;

let state = SessionStateBuilder::new()
    .with_default_features()
    .with_physical_optimizer_rule(datafusion_spatial::join::spatial_join_rule())
    .build();
let ctx = SessionContext::new_with_state(state);
datafusion_spatial::register_all(&ctx);
```

The rule is off by default, because it rewrites physical plans. It rewrites a join only when all
of these hold. Anything else keeps the plan DataFusion chose.

1. The join is `INNER`.
2. The join filter is one spatial predicate, with no `AND`.
3. Both arguments are plain columns, one from each input.
4. Both columns carry GeoArrow metadata.
5. Two separate boxes prove the predicate false. This excludes `ST_Disjoint`.

## Rules behind the ⚠️ notes

The function reference marks each difference at the point of use. Four rules explain most of them.

**The CRS belongs to the column, not to the value.** GeoArrow holds it once, in the field metadata.
So `ST_SRID` reads the schema and returns one value for every row. `ST_SetSRID` needs a constant,
because it changes the column type. This layout suits an analytic engine, and GeoParquet writes it.
It is a real change from PostGIS. One PostGIS column can hold rows in different systems.

**A constant argument stays constant.** Some arguments drive a plan-time decision or a setup step
for each batch. Examples are the SRID above, the radius in `ST_DWithin`, the pattern in
`ST_Relate`, and the matrix in `ST_Affine`. Each one must be a constant. A column there gives a
plan-time error. The crate does not rebuild the setup for each row.

**A row that does not fit returns null.** PostGIS often raises an error for a wrong geometry type.
A mixed column here returns null for those rows. One bad row does not stop the query.
A wrong static type is still a plan-time error. That case deserves a loud failure.

**The crate infers a plain column.** A `Utf8` column without extension metadata reads as WKT.
A `Binary` column reads as WKB. GeoArrow defines this rule. A raw CSV column then works without a
cast. The cost is one surprise: `ST_AsText` on any string column returns that string.
Pass such a column through `ST_GeomFromText` to check the parse step.

## Nulls in a mixed column

A mixed geometry column is an Arrow union. A union holds no validity buffer.
So `Array::is_null` returns false for every row of a union. It returns false for a null row too.

Use `GeoArrowArray::logical_nulls` instead. It reads the child arrays.
This matters when you read results through the Arrow API, not through SQL.

## Testing

```bash
cargo test --workspace
cargo test --workspace --features proj
cargo bench -p datafusion-spatial-kernels
```

The end-to-end SQL tests sit in `crates/datafusion-spatial/tests/spatial/`, with one module per
function family: `predicates.rs`, `measurement.rs`, `aggregates.rs` and so on. They share one test
binary. DataFusion is a large crate, and every extra test target links the whole of it again.

Run one family with a filter:

```bash
cargo test --test spatial predicates
```

The first command runs the unit tests, the SQL tests and the allocation budgets.
The second command adds the `ST_Transform` tests.

## License

Apache-2.0
