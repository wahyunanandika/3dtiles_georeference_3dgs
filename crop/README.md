# crop/ — Boundary Crop & Georeferenced PLY Export

Two optional tools that work on the **local-frame PLY** and the
`similarity_transform.json` produced by `solve_transform.py`. They are not
required for the 3D Tiles export.

| Script | Input | Output |
|---|---|---|
| `crop_ply_by_boundary.py` | local-frame PLY + real-world boundary polygon | cropped local-frame PLY |
| `export_georeferenced_ply.py` | local-frame PLY (usually the cropped one) | UTM-offset georeferenced PLY |

Both scripts import `geodetic_to_ecef` / `ecef_to_geodetic` from
`transform_solver.py` in the repo root, so the geodetic math is identical to
`solve_transform.py`. Run them from the repo root:

```bash
python crop/crop_ply_by_boundary.py ...
python crop/export_georeferenced_ply.py ...
```

---

## Requirements

In addition to `numpy`:

```bash
pip install pyshp pyproj matplotlib
```

| Package | Used by | Purpose |
|---|---|---|
| `pyshp` | `crop_ply_by_boundary.py` | read the boundary shapefile |
| `pyproj` | both | UTM ↔ WGS84 conversion |
| `matplotlib` | `crop_ply_by_boundary.py` | point-in-polygon test (`matplotlib.path.Path`) |

---

## Workflow

```
local-frame PLY ──┐
                  ├──→ crop_ply_by_boundary.py ──→ cropped local-frame PLY
boundary.shp ─────┘                                        │
(real-world UTM)                                           ├──→ tiles_exporter.py (3D Tiles)
                                                           │
                                                           └──→ export_georeferenced_ply.py
                                                                   │
                                                                   ▼
                                                        UTM-offset georeferenced PLY
                                                        (SuperSplat, PlayCanvas, gsplat.js, ...)
```

The cropped PLY stays in the local COLMAP frame, so it can go straight into
`tiles_exporter.py` with the **same** `similarity_transform.json`.

---

## 1. `crop_ply_by_boundary.py`

Crops splats using a boundary polygon in real-world UTM coordinates. The
boundary is converted UTM → WGS84 → ECEF → local frame using the inverse of
the similarity transform, then splats outside the polygon are removed.

```bash
python crop/crop_ply_by_boundary.py \
    --ply        splat_local.ply \
    --transform  similarity_transform.json \
    --boundary   boundary/boundary.shp \
    --utm-epsg   32748 \
    --output     splat_local_cropped.ply
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--ply` | yes | — | Local-frame PLY (same frame used to solve the transform) |
| `--transform` | yes | — | `similarity_transform.json` from `solve_transform.py` |
| `--boundary` | yes | — | Boundary polygon shapefile in real-world UTM (first shape is used) |
| `--utm-epsg` | no | `32748` | EPSG code of the boundary's UTM zone (32748 = WGS84 / UTM 48S) |
| `--output` | yes | — | Output cropped PLY |

**Notes**

- `--ply` must be in the same frame used by `solve_transform.py`. If your
  workflow rotates the PLY (e.g. −90° about X in SuperSplat) before entering
  the pipeline, use that rotated version, not the raw Y-up training output.
- `--boundary` must be real-world UTM. A boundary that has only been shifted
  to a local origin (e.g. `boundary_local.shp`) does not match the COLMAP
  local frame.
- The crop is **footprint-only (2D)**. The local "up" direction is derived
  from the transform, and splats are tested in the horizontal plane. Height
  is not filtered — everything inside the footprint is kept from bottom to top.

---

## 2. `export_georeferenced_ply.py`

Transforms every splat from the local frame to UTM, then subtracts an offset
so coordinates stay small enough for float32 viewers. Position, rotation and
scale are all transformed:

```
local  --[s·R·X + t]-->  ECEF  -->  lat/lon/h  -->  UTM  --[− offset]-->  UTM-offset
```

- **Rotation** (`rot_0..3`, wxyz): rotated by `R_ecef→enu · R`, with the ENU
  frame evaluated at the scene's mean lat/lon.
- **Scale** (`scale_0..2`, log-scale): `ln(s)` is **added**, since 3DGS stores
  scales as `ln(scale)`.
- The applied offset is written to the PLY header as `comment` lines, so the
  original UTM coordinates can be recovered.

```bash
python crop/export_georeferenced_ply.py \
    --ply        splat_local_cropped.ply \
    --transform  similarity_transform.json \
    --utm-epsg   32748 \
    --output     splat_georef_utm_offset.ply
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--ply` | yes | — | Local-frame PLY (usually the cropped output) |
| `--transform` | yes | — | `similarity_transform.json` from `solve_transform.py` |
| `--utm-epsg` | no | `32748` | EPSG code of the target UTM zone |
| `--offset-easting` | no | bbox min | Fixed easting offset, e.g. `788000` |
| `--offset-northing` | no | bbox min | Fixed northing offset, e.g. `9237000` (must be given with `--offset-easting`) |
| `--offset-height` | no | see below | Fixed height offset |
| `--output` | yes | — | Output PLY |

**Offset behaviour**

| Flags given | E/N offset | Height offset |
|---|---|---|
| none | bbox min of the splats | bbox min of the splat heights |
| `--offset-easting` + `--offset-northing` | the given values | `0.0` (unless `--offset-height` is set) |

Use fixed offsets to stay aligned with other data that shares a common
origin (e.g. a mesh or `boundary_local.shp`).

**Do not** feed this output into `tiles_exporter.py`. The 3D Tiles export
expects the local-frame PLY; applying the similarity transform to a UTM-offset
PLY places the tiles incorrectly.
