"""PROTOTYPE — 2-level additive (refine: ADD) 3D Tiles export for compatibility testing.

Purpose
-------
Throwaway test script. It answers two questions before any change is made to
tiles_exporter.py:

  1. Does ArcGIS Pro (Local Scene) recognise a Gaussian splat tileset whose
     root tile has content and uses refine "ADD", and does it draw root + leaves
     together when zoomed in?
  2. Does CesiumJS show sorting artefacts where root splats and leaf splats
     overlap?

Structure
---------
    root  [L0: top-K splats by importance]      refine: ADD
     └─ octree (unchanged from tiles_exporter)  refine: REPLACE
         └─ leaves [all remaining splats]

Every splat appears in exactly one tile (L0 ∪ leaves = input PLY), so no
splat is duplicated and no splat is synthesised. The similarity transform
(including any --apply-geoid-correction shift baked into its translation by
solve_transform.py) is applied exactly as in tiles_exporter.py, via the root
tile transform.

Importance score (after L3GS-style volume-weighted opacity):
    importance = sigmoid(opacity_logit) * exp(scale_0 + scale_1 + scale_2)

Usage
-----
python experimental/prototype_pyramid_add.py splat.ply similarity_transform.json out_dir/ \
    [--l0-splats 300000] [--max-sh-degree 3] [--fraction 1.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from spz_encode import encode_spz_v3  # noqa: E402
from tiles_exporter import (  # noqa: E402
    _assign_tile_ids,
    _auto_max_splats,
    _build_octree,
    _collect_leaves,
    _write_spz_glb,
    build_arrays_from_ply,
    read_ply,
)


def _importance(opacity_logit: np.ndarray, scales_log: np.ndarray) -> np.ndarray:
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit.astype(np.float64)))
    volume  = np.exp(scales_log.astype(np.float64).sum(axis=1))
    return opacity * volume


def _l0_spacing(positions: np.ndarray) -> float:
    """Mean spacing of L0 splats, assuming a surface-like (2.5D) drone scene:
    sqrt(footprint area / N), footprint = two largest bbox extents."""
    ext = np.sort(positions.max(axis=0) - positions.min(axis=0))[::-1]
    return float(np.sqrt(ext[0] * ext[1] / max(len(positions), 1)))


def _box(pmin: np.ndarray, pmax: np.ndarray) -> list:
    c = ((pmin + pmax) * 0.5).tolist()
    h = ((pmax - pmin) * 0.5).tolist()
    return [c[0], c[1], c[2], h[0], 0.0, 0.0, 0.0, h[1], 0.0, 0.0, 0.0, h[2]]


def _octree_dict(node, internal_ge: float) -> dict:
    tile = {
        "boundingVolume": {"box": _box(node.pmin, node.pmax)},
        "geometricError": 0.0 if node.is_leaf else internal_ge,
        "refine": "REPLACE",
    }
    if node.is_leaf:
        tile["content"] = {"uri": node.tile_id}
    else:
        tile["children"] = [_octree_dict(ch, internal_ge) for ch in node.children]
    return tile


def _write_tile(path, idx, a) -> None:
    spz = encode_spz_v3(
        positions      = a["positions"][idx],
        rotations_xyzw = a["rot_xyzw"][idx],
        scales_log     = a["scales_log"][idx],
        opacity_logit  = a["opacity_logit"][idx],
        f_dc           = a["f_dc"][idx],
        f_rest_rgb     = a["f_rest_rgb"][idx] if a["f_rest_rgb"] is not None else None,
        sh_degree      = a["sh_degree"],
    )
    _write_spz_glb(path, spz, int(len(idx)), a["sh_degree"], a["positions"][idx])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PROTOTYPE: 2-level refine:ADD Gaussian splat 3D Tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("ply",             type=Path)
    ap.add_argument("similarity_json", type=Path)
    ap.add_argument("out_dir",         type=Path)
    ap.add_argument("--l0-splats",     type=int,   default=300_000,
                    help="number of splats in the root (L0) tile")
    ap.add_argument("--max-sh-degree", type=int,   choices=[0, 1, 2, 3], default=3)
    ap.add_argument("--max-splats-per-tile", type=int, default=0)
    ap.add_argument("--min-tile-size", type=float, default=0.1)
    ap.add_argument("--fraction",      type=float, default=1.0)
    ap.add_argument("--seed",          type=int,   default=0)
    args = ap.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        sys.exit(f"--fraction must be in (0, 1], got {args.fraction}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Read PLY (same path as tiles_exporter) ───────────────────────────────
    ply, names = read_ply(args.ply)
    if args.fraction < 1.0:
        n_keep = int(round(len(ply) * args.fraction))
        ply = ply[np.random.default_rng(args.seed).choice(len(ply), n_keep, replace=False)]
    pos, rot_wxyz, scl, opa, fdc, frest, sh = build_arrays_from_ply(ply, names, args.max_sh_degree)
    del ply
    n = len(pos)

    rot_xyzw = np.stack([rot_wxyz[:, 1], rot_wxyz[:, 2], rot_wxyz[:, 3], rot_wxyz[:, 0]], axis=-1)
    nrm = np.linalg.norm(rot_xyzw, axis=-1, keepdims=True)
    nrm[nrm == 0] = 1.0
    rot_xyzw = (rot_xyzw / nrm).astype(np.float32)

    arrays = dict(positions=pos.astype(np.float32), rot_xyzw=rot_xyzw,
                  scales_log=scl.astype(np.float32), opacity_logit=opa.astype(np.float32),
                  f_dc=fdc.astype(np.float32), f_rest_rgb=frest, sh_degree=sh)

    # ── Split L0 / rest by importance ────────────────────────────────────────
    k = min(args.l0_splats, n)
    order   = np.argsort(-_importance(opa, scl), kind="stable")
    l0_idx  = np.sort(order[:k])
    rest    = np.sort(order[k:])

    # ── Octree over the remaining splats (unchanged algorithm) ───────────────
    max_splats = args.max_splats_per_tile or _auto_max_splats(n)
    octree = _build_octree(pos, rest, max_splats, args.min_tile_size)
    leaves = _collect_leaves(octree)
    _assign_tile_ids(octree, [0])

    # ── Geometric errors (monotonically non-increasing down the tree) ────────
    pmin, pmax   = pos.min(axis=0), pos.max(axis=0)
    ge_tileset   = float(np.linalg.norm(pmax - pmin))  # nothing drawn
    ge_root      = _l0_spacing(pos[l0_idx])              # only L0 drawn
    ge_internal  = ge_root                               # empty nodes: refine together with root

    # ── Write content ────────────────────────────────────────────────────────
    _write_tile(args.out_dir / "root_l0.glb", l0_idx, arrays)
    for i, leaf in enumerate(leaves):
        _write_tile(args.out_dir / leaf.tile_id, leaf.indices, arrays)
        print(f"  {int(100 * (i + 1) / len(leaves)):3d}%", end="\r", flush=True)

    # ── tileset.json ─────────────────────────────────────────────────────────
    with open(args.similarity_json) as f:
        sim = json.load(f)
    s = float(sim.get("scale", sim.get("s")))
    R = np.array(sim.get("rotation", sim.get("R")), dtype=np.float64).reshape(3, 3)
    t = np.array(sim.get("translation", sim.get("t")), dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3]  = t

    child = _octree_dict(octree, ge_internal)
    root = {
        "transform":      M.T.flatten().tolist(),
        "boundingVolume": {"box": _box(pmin, pmax)},
        "geometricError": ge_root,
        "refine":         "ADD",
        "content":        {"uri": "root_l0.glb"},
        "children":       [child],
    }
    ext = ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"]
    tileset = {
        "asset": {"version": "1.1"},
        "extensionsUsed":     ["3DTILES_content_gltf"],
        "extensionsRequired": ["3DTILES_content_gltf"],
        "extensions": {"3DTILES_content_gltf": {"extensionsUsed": ext, "extensionsRequired": ext}},
        "geometricError": ge_tileset,
        "root": root,
    }
    with open(args.out_dir / "tileset.json", "w", encoding="utf-8") as f:
        json.dump(tileset, f, indent=2)

    # ── Report ───────────────────────────────────────────────────────────────
    n_leaf = sum(len(l.indices) for l in leaves)
    print(f"\nsplats total      : {n:,}")
    print(f"  L0 (root, ADD)  : {k:,}")
    print(f"  leaves          : {n_leaf:,} in {len(leaves)} tiles")
    print(f"  L0 + leaves     : {k + n_leaf:,}  ({'OK, no duplication' if k + n_leaf == n else 'MISMATCH'})")
    print(f"geometricError    : tileset {ge_tileset:.2f} | root {ge_root:.3f} | internal {ge_internal:.3f} | leaf 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
