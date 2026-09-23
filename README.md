# 3dtiles_georeference_3dgs

Standalone Python pipeline for converting 3D Gaussian Splatting reconstructions into georeferenced 3D Tiles 1.1 (SPZ-compressed) ready for Cesium ion and CesiumJS — without requiring LichtFeld Studio.

Tested on three real-world drone datasets in (ITB Ganesha Bandung, Taman Kota Cimahi, Jatinangor), West Java, Indonesia. Horizontal positioning appears consistent with GPS PPK references. Further geodetic validation with independent checkpoints is recommended.

---

## Features

- Direct COLMAP → ECEF georeferencing
- No LichtFeld Studio dependency
- SPZ-compressed Gaussian Splats
- 3D Tiles 1.1 export
- Cesium ion compatible
- Octree-based spatial tiling
- Supports WGS84 (GEOGCS) and UTM (PROJCS) Metashape exports
- Optional sparse-point verification using points3D.bin
- Optional geoid correction (EGM96 / EGM2008) to orthometric height
- Optional single-file `.3tz` packaging
- Fully reproducible command-line workflow

---

## Intended Workflow

This project is intended for users who:

- Train Gaussian Splatting scenes using COLMAP
- Export GPS-referenced cameras from Metashape
- Need georeferenced 3D Tiles for Cesium ion or CesiumJS
- Prefer a fully scriptable workflow without requiring LichtFeld Studio

---

## What Changed from the Original Plugin

This pipeline is built on top of [dozeri83/geo-register-plugin](https://github.com/dozeri83/geo-register-plugin).
The following changes were made:

**Removed LichtFeld Studio dependency** — reads COLMAP `images.bin` directly instead
of relying on the LFS camera API. LFS v0.5.2 introduced changes to coordinate-space
handling. For our workflow, obtaining a consistent transform between camera positions
and exported PLY coordinates became difficult, so the pipeline was redesigned to
operate directly from COLMAP data.

**Direct COLMAP → ECEF transform** — solves similarity directly from COLMAP camera
centres to GPS ECEF, bypassing the Metashape scene-space intermediate. PLY splats
and COLMAP cameras share the same coordinate space, so this eliminates the need for
`node.world_transform`.

**Removed `diag(1, -1, -1)` flip** — this flip was designed for LFS visualizer-world
convention and does not apply to COLMAP output space. Removing it corrects the
altitude from negative values to the correct terrain level.

**Added octree tiling** — the original plugin exports all splats as a single GLB.
For large scenes this causes Cesium to skip the tile entirely. Octree splitting with
`geometricError = bbox diagonal` per node makes Cesium render tiles progressively.

**Fixed `geometricError`** — removed scale multiplication. `geometricError` is in
local tile space; the transform matrix handles the coordinate change.

**Fixed `refine: "ADD"` → `"REPLACE"`** — `ADD` requires parent LOD splats which
do not exist in single-level exports.

**Added UTM support** — `metashape_parser.py` now detects chunk CRS automatically.
WGS84 (GEOGCS) and UTM (PROJCS) are both supported. Zone number and hemisphere are
parsed directly from the Metashape WKT string. All UTM zones (1–60, N/S) are supported.

**Added `points3D.bin` reader** — optional verification of terrain altitude before
running the full tile export.

---

## Pipeline Overview

```
COLMAP images.bin ──→ Camera Centres ──┐
                                       ├──→ Similarity Solver ──→ Transform Matrix
Metashape XML ──→ GPS ECEF ────────────┘                                │
                                                                        ▼
                                                              Gaussian Splat PLY
                                                                        │
                                                                        ▼
                                                                 Octree Tiling
                                                                        │
                                                                        ▼
                                                                  SPZ Encoding
                                                                        │
                                                                        ▼
                                                                 3D Tiles 1.1
                                                                        │
                                                                        ▼
                                                            Cesium ion / CesiumJS
```

---

## Requirements

```bash
pip install -r requirements.txt
```

| Package | Required for |
|---|---|
| `numpy` | everything (required) |
| `pygeodesy` | `--geoid-model` in `solve_transform.py` / `verify_tileset.py` (optional) |
| `pyshp`, `pyproj`, `matplotlib` | tools in [`crop/`](crop/README.md) (optional) |

`.3tz` packaging (Step 5) additionally requires [Node.js](https://nodejs.org) (LTS).

---

## Inputs

| File | Description |
|---|---|
| `splat.ply` | Binary Gaussian Splat PLY |
| `sparse/0/images.bin` | COLMAP sparse reconstruction cameras |
| `camera_export.xml` | Metashape camera export — WGS84 (EPSG:4326) or UTM (any zone) |
| `sparse/0/points3D.bin` | Optional sparse point cloud for verification |

**Metashape export:** File → Export → Export Cameras. The chunk CRS can be set to either:
- **WGS84 (EPSG:4326)** — camera references stored as lon, lat, ellipsoidal height
- **UTM (any zone)** — camera references stored as easting, northing, ellipsoidal height

Camera label stems in `images.bin` must match XML labels. CRS is detected automatically from the XML.

---

## Usage

### Step 1 — Solve Similarity Transform

```bash
python solve_transform.py \
    --images-bin sparse/0/images.bin \
    --metashape camera_export.xml \
    --points3d sparse/0/points3D.bin \
    --output similarity_transform.json
```

Example output (WGS84):

```
343 cameras read
341 GPS cameras loaded
68,502 sparse points read

Matched 341 cameras (COLMAP <-> Metashape XML)
Solver: 341/341 inliers, RMSE=0.0762 m
Camera centroid -> lat=-6.87067, lon=107.55432, alt=917.19 m (drone altitude)
Sparse points -> terrain alt: 790.6-839.2 m (mean=812.8 m)

scale       = 1.00000000
RMSE (GPS)  = 0.0762 m
inliers     = 341/341
```

Example output (UTM zone 48S):

```
1146 cameras read
1146 GPS cameras loaded

Matched 1146 cameras (COLMAP <-> Metashape XML)
CRS: UTM zone 48S — converting easting/northing to ECEF
Solver: 1146/1146 inliers, RMSE=0.0222 m
Camera centroid -> lat=-6.88964, lon=107.60994, alt=896.47 m (drone altitude)

scale       = 1.00000000
RMSE (GPS)  = 0.0222 m
inliers     = 1146/1146
```

#### Optional — Geoid undulation and orthometric height

```bash
python solve_transform.py \
    --images-bin sparse/0/images.bin \
    --metashape camera_export.xml \
    --output similarity_transform.json \
    --geoid-model egm2008-2_5 \
    --apply-geoid-correction
```

| Flag | Description |
|---|---|
| `--geoid-model` | `egm96-5`, `egm2008-5`, `egm2008-2_5`, `egm2008-1`. Computes the geoid undulation N and orthometric height at the scene centroid and prints them. Informational only unless `--apply-geoid-correction` is also given. |
| `--apply-geoid-correction` | Shifts the transform's translation so the output is placed at **orthometric** height instead of ellipsoidal. Requires `--geoid-model`. |

- Geoid grids are downloaded automatically from the GeographicLib distribution
  on first use and cached in `geoids/` (override with the `GEOID_DATA_DIR`
  environment variable). `egm96-5` and `egm2008-2_5` are already included in
  the repo.
- The correction is a single ECEF shift computed at the scene centroid and
  applied uniformly to the whole scene. This is a close approximation for
  scenes up to a few km across, since the geoid varies smoothly over much
  larger distances.
- Without `--apply-geoid-correction` the output is ellipsoidal (WGS84 / GPS
  heights), exactly as before. Only apply the correction when the rest of your
  scene (terrain, meshes, CityGML) uses orthometric heights.
- When applied, `similarity_transform.json` contains a `geoid` block with
  `"applied_to_transform": true`, the model, N, and the ECEF shift.
- `verify_tileset.py` also accepts `--geoid-model` to print N at the tileset
  centre (informational only).

### Step 2 — Export 3D Tiles

```bash
python tiles_exporter.py \
    splat.ply \
    similarity_transform.json \
    output_tiles/
```

| Parameter | Description |
|---|---|
| `--max-sh-degree` | Maximum SH degree (0–3) |
| `--max-splats-per-tile` | Override automatic tile sizing |
| `--min-tile-size` | Minimum octree cell size |
| `--fraction` | Subsample splats for testing |

### Step 3 — Verify Export

```bash
python verify_tileset.py output_tiles/
```

### Step 4 — Upload to Cesium ion

1. Open Cesium ion
2. Click Add Data
3. Select 3D Tiles
4. Upload the generated `output_tiles/` directory

Use this Sandcastle snippet to render and measure altitude offset.
Replace `ASSET_ID` and `defaultAccessToken` with your values:

```javascript
const ASSET_ID = 0; // <- replace

Cesium.Ion.defaultAccessToken = "your_token_here"; // <- replace

const viewer = new Cesium.Viewer("cesiumContainer", {
  terrain: Cesium.Terrain.fromWorldTerrain(),
});

const tileset = await Cesium.Cesium3DTileset.fromIonAssetId(ASSET_ID);
viewer.scene.primitives.add(tileset);
await viewer.zoomTo(tileset);

// Optional: raise tileset vertically in local ENU space
// Adjust the Z value (metres) to lift splats above terrain clipping
// const center = tileset.boundingSphere.center;
// const enuTransform = Cesium.Transforms.eastNorthUpToFixedFrame(center);
// const translation = Cesium.Matrix4.multiplyByPoint(
//   enuTransform,
//   new Cesium.Cartesian3(0, 0, 10),  // <- adjust height offset here
//   new Cesium.Cartesian3()
// );
// const offset = Cesium.Cartesian3.subtract(translation, center, new Cesium.Cartesian3());
// tileset.modelMatrix = Cesium.Matrix4.fromTranslation(offset);

const info = document.createElement("div");
info.style.cssText = `
  position:absolute; top:10px; left:10px; z-index:999;
  background:rgba(0,0,0,0.75); color:#fff;
  font:13px monospace; padding:12px 16px; border-radius:6px;
  max-width:360px; line-height:1.6;
`;
info.innerHTML = "Click on the splat to measure altitude offset";
document.getElementById("cesiumContainer").appendChild(info);

viewer.screenSpaceEventHandler.setInputAction(async (click) => {
  const pickedPos = viewer.scene.pickPosition(click.position);
  if (!Cesium.defined(pickedPos)) {
    info.innerHTML = "Click directly on the splat model.";
    return;
  }

  const cartographic = Cesium.Ellipsoid.WGS84.cartesianToCartographic(pickedPos);
  const lat = Cesium.Math.toDegrees(cartographic.latitude);
  const lon = Cesium.Math.toDegrees(cartographic.longitude);
  const splatAlt = cartographic.height;

  const terrainPositions = [Cesium.Cartographic.fromDegrees(lon, lat)];
  const sampledTerrain = await Cesium.sampleTerrainMostDetailed(
    viewer.terrainProvider,
    terrainPositions
  );
  const terrainAlt = sampledTerrain[0].height;
  const offset = splatAlt - terrainAlt;

  const offsetColor = Math.abs(offset) < 30 ? "#7fff7f" : "#ff7f7f";
  const offsetLabel = Math.abs(offset) < 5
    ? "Near perfect"
    : Math.abs(offset) < 25
    ? "Likely geoid offset (~21m EGM96)"
    : "Large offset — check transform";

  info.innerHTML = `
    <b>Clicked point</b><br>
    Lat : ${lat.toFixed(6)}<br>
    Lon : ${lon.toFixed(6)}<br>
    <br>
    <b>Altitudes (ellipsoidal WGS84)</b><br>
    Splat alt  : <b>${splatAlt.toFixed(2)} m</b><br>
    Terrain alt: <b>${terrainAlt.toFixed(2)} m</b><br>
    <br>
    <b>Offset (splat - terrain)</b><br>
    <span style="color:${offsetColor}; font-size:16px">
      <b>${offset >= 0 ? "+" : ""}${offset.toFixed(2)} m</b>
    </span><br>
    <span style="color:#aaa; font-size:11px">${offsetLabel}</span>
  `;

  viewer.entities.removeAll();
  viewer.entities.add({
    position: pickedPos,
    point: { pixelSize: 10, color: Cesium.Color.YELLOW },
  });

}, Cesium.ScreenSpaceEventType.LEFT_CLICK);
```

### Step 5 (optional) — Package as a single `.3tz` file

The output folder can be packaged into one `.3tz` file (a 3D Tiles package
based on ZIP) with [3d-tiles-tools](https://github.com/CesiumGS/3d-tiles-tools).
The tiles themselves are not modified.

```bash
npx 3d-tiles-tools convert -i output_tiles/ -o output_tiles.3tz
```

- The output path must end in `.3tz` and must not already exist
  (add `-f` to overwrite).
- On first run, `npx` asks to download `3d-tiles-tools` — answer `y`.
- `npm warn deprecated ...` messages come from the tool's dependencies and
  can be ignored.

| Platform | `.3tz` support |
|---|---|
| ArcGIS Pro 3.7 | Verified (ITB Ganesha): Add Data → `.3tz` in a scene loads as a Gaussian splat layer |
| Cesium ion | Not yet verified. A `.3tz` is a ZIP file; renaming it to `.zip` and uploading as 3D Tiles is reported to work for regular tilesets ([Cesium Community](https://community.cesium.com/t/how-to-upload-3tz-file-to-cesium-ion/40313)) |
| CesiumJS (direct) | Not supported — extract the `.3tz` (it is a ZIP) and serve the folder ([Cesium Community](https://community.cesium.com/t/how-to-load-3tz/37419)) |

---

## Optional Tools

### Crop & Georeferenced PLY Export — [`crop/`](crop/README.md)

- `crop_ply_by_boundary.py` — crop the local-frame PLY with a real-world UTM
  boundary polygon. The cropped PLY stays in the local frame and can go
  straight into `tiles_exporter.py` with the same transform.
- `export_georeferenced_ply.py` — export a UTM-offset georeferenced PLY
  (position, rotation and scale transformed) for 3DGS viewers.

Requires `pyshp`, `pyproj` and `matplotlib`. See [`crop/README.md`](crop/README.md).

### Experimental — [`experimental/`](experimental/README.md)

Design prototypes that are not part of the supported pipeline, with their
motivation, literature and test results. Currently: a 2-level additive LOD
(`refine: ADD`) prototype. See [`experimental/README.md`](experimental/README.md).

---

## Validation Result

Tested on two drone datasets in Bandung, West Java, Indonesia.
GPS RMSE reflects the fit of the similarity transform against GPS PPK camera positions.
Absolute geodetic accuracy has not been independently verified — further validation
with ground control points is recommended.

| Dataset | Splats | GPS Cameras | Tiles | GPS RMSE |
|---|---|---|---|---|
| Taman Kota Cimahi | 4,999,683 | 341 PPK | 463 | 0.076 m |
| ITB Ganesha, Bandung | 8,999,938 | 1,146 PPK | 698 | 0.022 m |
| Jatinangor Full | 24,937,728 | 4,245 PPK | 1,141 | 0.599 m |

**Taman Kota Cimahi**
![Taman Kota Cimahi overview](assets/taman_kota_cimahi_overview.jpg)
![Taman Kota Cimahi close-up](assets/taman_kota_cimahi_45deg.jpg)

**ITB Ganesha, Bandung**


![ITB Ganesha overview](assets/itb_overview.jpg)
![ITB Ganesha close-up](assets/itb_ganesha_45deg.jpg)

**Jatinangor Full**
<!-- ![Jatinangor Full overview](assets/jatinangor_full_overview.jpg) -->
![Jatinangor Full close-up](assets/jatinangor_full_45deg.jpg)

During visual inspection in Cesium ion, horizontal positioning appeared consistent
with known geographic features. Vertical positioning was not independently verified
against ground truth measurements.

---

## File Structure

```
3dtiles_georeference_3dgs/
├── solve_transform.py
├── tiles_exporter.py
├── verify_tileset.py
├── colmap_reader.py
├── metashape_parser.py
├── transform_solver.py
├── spz_encode.py
├── geoid_model.py
├── geoids/                     # cached EGM geoid grids
├── crop/
│   ├── README.md
│   ├── crop_ply_by_boundary.py
│   └── export_georeferenced_ply.py
├── experimental/
│   ├── README.md
│   └── prototype_pyramid_add.py
├── requirements.txt
├── README.md
├── PIPELINE_DEBUG_LOG.md
└── LICENSE
```

---

## Troubleshooting

**Only N cameras matched** — camera labels in `images.bin` must match labels in Metashape XML:

```bash
python -c "from colmap_reader import read_images_bin; print(list(read_images_bin('sparse/0/images.bin').keys())[:5])"
python -c "from metashape_parser import parse_metashape_xml; d=parse_metashape_xml('cam.xml'); print([c['name'] for c in d['cameras'][:5]])"
```

**No chunk transform found** — re-export cameras from Metashape with chunk CRS = WGS84 (EPSG:4326).

**Unsupported CRS** — only WGS84 (GEOGCS) and UTM (PROJCS) are supported. If your chunk uses
a local or projected CRS other than UTM, re-export from Metashape with WGS84 or UTM.

**UTM zone not detected** — ensure the Metashape WKT contains `UTM zone NN[NS]`
(e.g. `UTM zone 48S`). This is the standard Metashape format and should be present automatically.

**Splats clipping into terrain** — uncomment the ENU height offset block in the Sandcastle snippet above.

**Large scenes (>3km)** — use `--max-splats-per-tile` to increase tile count.
SPZ v3 uses 24-bit fixed-point with ±2048 unit range.

---

## Known Limitations & Future Work

**Large scene rendering (>20M splats)** — loading all tiles simultaneously during overview
causes `RangeError: Array buffer allocation failed` in Cesium Sandcastle. Workaround: zoom
in to a specific area before loading, or increase `maximumScreenSpaceError` to reduce the
number of tiles loaded at once:
```javascript
tileset.maximumScreenSpaceError = 32; // default is 16
```

**PLY outlier filter** — no built-in filter yet. Oblique datasets often contain outlier
splats far below terrain level that pull the bounding box down. Manual cleaning in
SuperSplat is currently required before export. A `--z-min / --z-max` flag is planned.

**Absolute geodetic accuracy** — GPS RMSE reflects the fit against GPS PPK camera
positions, not independently verified ground truth. Validation with independent GCPs
is recommended for production use.


**Geoid correction** — available via `--geoid-model` / `--apply-geoid-correction`
(Step 1). It uses a single undulation value at the scene centroid, applied
uniformly; very large scenes would need a per-point correction. Only global
models (EGM96, EGM2008) are supported. InaGeoid is not supported yet because it
is distributed as a login-gated point-query service, not a downloadable grid.

---

## Acknowledgements

This project builds upon ideas and implementations from:

- **[dozeri83/geo-register-plugin](https://github.com/dozeri83/geo-register-plugin)** —
  the original LichtFeld Studio georeferencing plugin. The following components were
  adapted from this work: SPZ encoding, similarity transform estimation, Metashape XML
  parsing, and 3D Tiles export structure. Special thanks to dozeri83 for creating and
  maintaining this plugin, which served as both the foundation and inspiration for this work.

- **[Niantic SPZ](https://github.com/nianticlabs/spz)** — SPZ v3 binary format specification (MIT License).

Additional implementation details, design decisions, and investigation notes can be found in `PIPELINE_DEBUG_LOG.md`.

---

## License

GPL-3.0

SPZ encoder implementation follows the Niantic SPZ specification (MIT License).