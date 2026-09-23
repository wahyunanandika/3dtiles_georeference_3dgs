# experimental/ — Prototypes

Scripts in this folder are **not part of the supported pipeline**. They are
kept to document design experiments and their outcomes. Use `tiles_exporter.py`
in the repo root for production exports.

Run scripts from the repo root. They import from the root modules
(`tiles_exporter.py`, `spz_encode.py`).

---

## `prototype_pyramid_add.py` — 2-level additive LOD (refine: ADD)

### Motivation

The default export (`tiles_exporter.py`) is a **flat** octree: all splats live
in leaf tiles and internal nodes have no content. When the whole scene is in
view, the viewer must load every visible leaf at full density. On the
Jatinangor dataset (~25M splats) this causes
`RangeError: Array buffer allocation failed` in Cesium (see *Known Limitations*
in the main README).

The prototype tests whether adding a coarse level of **original splats** at the
root, with 3D Tiles `refine: "ADD"`, is compatible with CesiumJS and ArcGIS Pro.

### Design

```
root  [L0: top-K splats by importance]      refine: ADD
 └─ octree (same algorithm as tiles_exporter)  refine: REPLACE
     └─ leaves [all remaining splats]
```

- Every splat is stored in exactly one tile: `L0 ∪ leaves` = input PLY.
  No splat is duplicated and no splat is synthesised, so every rendered splat
  comes from the original reconstruction at every level.
- Importance score, after the volume-weighted opacity used in L3GS:
  `sigmoid(opacity) × exp(scale_0 + scale_1 + scale_2)`
- `geometricError`:

  | Level | Value |
  |---|---|
  | tileset | bbox diagonal (nothing drawn) |
  | root | mean L0 spacing, `sqrt(footprint_area / K)` (only L0 drawn) |
  | internal octree nodes | same as root (refine together with the root) |
  | leaves | 0 |

- The similarity transform is applied via the root tile transform exactly as
  in `tiles_exporter.py`. Any `--apply-geoid-correction` shift already in
  `similarity_transform.json` is carried over unchanged.

### Usage

```bash
python experimental/prototype_pyramid_add.py \
    splat.ply similarity_transform.json out_pyramid/ \
    --l0-splats 300000
```

| Argument | Default | Description |
|---|---|---|
| `ply` | — | Local-frame PLY (same input as `tiles_exporter.py`) |
| `similarity_json` | — | `similarity_transform.json` from `solve_transform.py` |
| `out_dir` | — | New, empty output folder |
| `--l0-splats` | `300000` | Number of splats in the root (L0) tile |
| `--max-sh-degree` | `3` | Maximum SH degree (0–3) |
| `--max-splats-per-tile` | auto | Leaf size, same as `tiles_exporter.py` |
| `--min-tile-size` | `0.1` | Minimum octree cell size |
| `--fraction` | `1.0` | Random subsample for quick tests |
| `--seed` | `0` | Seed for `--fraction` |

`verify_tileset.py` reports `refine == REPLACE` as FAIL on this output. That
is expected, since the root uses `ADD`.

### Results

**ITB Ganesha (Metashape-trained PLY)**

```
splats total      : 7,612,828
  L0 (root, ADD)  : 300,000
  leaves          : 7,312,828 in 696 tiles
  L0 + leaves     : 7,612,828  (OK, no duplication)
geometricError    : tileset 1670.66 | root 2.055 | internal 2.055 | leaf 0
```

Observations:

- For datasets of this size, the flat export already loads fast enough; the
  pyramid gives no practical benefit.
- For per-tile output, the flat approach (each tile loads original splats
  directly) is preferred.
- The tileset diagonal (1670.66 m) is larger than expected for the campus,
  most likely due to outlier splats inflating the bounding box. Because the
  root `geometricError` is derived from the bbox footprint, outliers also
  inflate it.

**Jatinangor** — not yet recorded.

**ArcGIS Pro compatibility (refine: ADD with root content)** — not yet recorded.

### Conclusion

The flat export remains the default and recommended mode. For a single-file
deliverable, packaging the existing flat tileset (e.g. as `.3tz`) is being
evaluated instead of changing the tiling.

### Literature

- Shi et al., *LapisGS: Layered Progressive 3D Gaussian Splatting for Adaptive
  Streaming*, 3DV 2025. [arXiv:2408.14823](https://arxiv.org/abs/2408.14823) —
  cumulative layered representation; the official repo includes a top-down
  mode that partitions a single full model into importance-ranked layers.
- Tsai et al., *L3GS: Layered 3D Gaussian Splats for Efficient 3D Scene
  Delivery*, ACM MobiCom 2025. [arXiv:2504.05517](https://arxiv.org/abs/2504.05517) —
  volume-weighted opacity importance score.
- Kerbl et al., *A Hierarchical 3D Gaussian Representation for Real-Time
  Rendering of Very Large Datasets*, ACM TOG 43(4), 2024.
  [arXiv:2406.12080](https://arxiv.org/abs/2406.12080) — merge-based hierarchy
  (alternative considered; not used because merged nodes are synthetic
  Gaussians, need opacity > 1, and increase storage).
- Cesium blog, 27 April 2026, on Gaussian splat LOD in 3D Tiles
  ([link](https://cesium.com/blog/2026/04/27/3d-gaussian-splats-lod/)) —
  industry reference for hierarchical LOD of splats.
