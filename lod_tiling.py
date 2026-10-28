# Part of 3dtiles_georeference_3dgs (GPL-3.0)
#
# Tile-tree conventions (REPLACE refinement, geometricError = median splat
# diameter x error scale, errors never increasing towards the leaves) follow
# dozeri83/geo-register-plugin geo/ssog_tiles.py (GPL-3.0):
# https://github.com/dozeri83/geo-register-plugin
# The level construction is different: dozeri gets LOD levels from LichtFeld
# Studio (lf.io.save_ssog); here every internal octree node receives a
# decimated copy of its subtree, built bottom-up in numpy (standalone).
"""Hierarchical LOD (HLOD) for Gaussian splats on top of the flat octree.

Flat tiling puts content only in leaves, so a viewer must load every visible
leaf even from far away. HLOD gives every internal node its own content: a
reduced set of splats standing in for its whole subtree. With REPLACE
refinement the viewer draws a node from far away and swaps it for its
children when it gets close.

Per node content = (indices into the full arrays, log-scale offset).
No splat data is copied until a tile is encoded.

Decimation (v1): voxel clustering
  1. pick a voxel size so the occupied voxels ~= the tile budget
  2. keep, per voxel, the splat with the largest  alpha * projected area
  3. enlarge the kept splat so it covers what the voxel held:
       f = sqrt(sum(alpha*area in voxel) / (alpha*area of kept)),
     capped at --lod-max-inflate and at the voxel size, never shrinking.
Built from the children's content (not the raw leaves), so each level
reduces the one below it.
"""
from __future__ import annotations

import numpy as np

DEFAULT_ERROR_SCALE = 16.0
DEFAULT_MAX_INFLATE = 2.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def _voxel_keys(p: np.ndarray, pmin: np.ndarray, v: float) -> np.ndarray:
    q = np.floor((p - pmin) / v).astype(np.int64)
    d = q.max(axis=0) + 1
    return (q[:, 0] * d[1] + q[:, 1]) * d[2] + q[:, 2]


def decimate(positions, scales_log, opacity_logit, idx, dlog, target,
             max_inflate=DEFAULT_MAX_INFLATE):
    """Reduce (idx, dlog) to at most `target` splats. Returns (idx, dlog)."""
    n = len(idx)
    if n <= target:
        return idx, dlog

    p = positions[idx].astype(np.float64)
    s = scales_log[idx].astype(np.float64) + dlog[:, None]
    s_sorted = np.sort(s, axis=1)
    area = np.exp(s_sorted[:, 1] + s_sorted[:, 2])        # two largest axes
    w = _sigmoid(opacity_logit[idx]) * area

    pmin = p.min(axis=0)
    diag = float(np.linalg.norm(p.max(axis=0) - pmin))
    if diag == 0.0:
        keep = np.argsort(-w)[:target]
        return idx[keep], dlog[keep]

    # Largest voxel size (in log space) whose occupied-voxel count <= target.
    lo, hi = diag * 1e-6, diag
    keys = _voxel_keys(p, pmin, hi)
    v = hi
    for _ in range(24):
        mid = np.sqrt(lo * hi)
        k = _voxel_keys(p, pmin, mid)
        if np.unique(k).size > target:
            lo = mid
        else:
            hi, keys, v = mid, k, mid
        if hi / lo < 1.01:
            break

    uniq, inv = np.unique(keys, return_inverse=True)
    order = np.lexsort((-w, inv))                          # by voxel, heaviest first
    first = np.r_[True, inv[order][1:] != inv[order][:-1]]
    reps = order[first]                                    # one per voxel
    wsum = np.bincount(inv, weights=w, minlength=len(uniq))

    f = np.sqrt(wsum[inv[reps]] / np.maximum(w[reps], 1e-30))
    f = np.minimum(f, max_inflate)
    # do not grow a splat past the voxel it stands for
    radius = np.exp(s[reps].max(axis=1))
    f = np.minimum(f, np.maximum(1.0, 0.5 * v / radius))
    f = np.maximum(f, 1.0)

    return idx[reps], (dlog[reps] + np.log(f)).astype(np.float32)


def build_hlod(root, positions, scales_log, opacity_logit, budget,
               max_inflate=DEFAULT_MAX_INFLATE, progress_cb=None):
    """Attach `lod_idx` / `lod_dlog` to every node of the octree (bottom-up)."""
    internal = []

    def collect(node):
        for c in node.children:
            collect(c)
        if node.is_leaf:
            node.lod_idx = node.indices
            node.lod_dlog = np.zeros(len(node.indices), dtype=np.float32)
        else:
            internal.append(node)

    collect(root)                      # post-order: children before parents
    for i, node in enumerate(internal):
        idx = np.concatenate([c.lod_idx for c in node.children])
        dlog = np.concatenate([c.lod_dlog for c in node.children])
        node.lod_idx, node.lod_dlog = decimate(
            positions, scales_log, opacity_logit, idx, dlog, budget, max_inflate)
        if progress_cb:
            progress_cb((i + 1) / len(internal))
    return internal


def assign_errors(root, scales_log, error_scale=DEFAULT_ERROR_SCALE):
    """geometricError per node: median splat diameter x error_scale; leaves 0.

    In scene (tile-local) units, like the flat tiling: the root transform's scale
    is applied by the viewer (CesiumJS Cesium3DTile.updateGeometricErrorScale),
    so it is not multiplied in here (unlike dozeri's ssog_tiles.py).
    Made non-increasing towards the leaves (a parent covers at least its children).
    """
    def visit(node):
        if node.is_leaf:
            node.lod_error = 0.0
            return 0.0
        child_max = max(visit(c) for c in node.children)
        s = scales_log[node.lod_idx].astype(np.float64) + node.lod_dlog[:, None]
        diam = 2.0 * np.exp(s.max(axis=1))
        err = float(np.median(diam)) * error_scale
        node.lod_error = max(err, child_max)
        return node.lod_error

    return visit(root)


def assign_uris(root):
    """tiles/D<depth>/<n>.glb — D0 is the root (coarsest)."""
    counter = [0]
    nodes = []

    def visit(node):
        node.tile_id = f"tiles/D{node.depth}/{counter[0]:06d}.glb"
        counter[0] += 1
        nodes.append(node)
        for c in node.children:
            visit(c)

    visit(root)
    return nodes


def node_to_tile_dict(node, box_fn):
    tile = {
        "boundingVolume": {"box": box_fn(node.pmin, node.pmax)},
        "geometricError": node.lod_error,
        "refine": "REPLACE",
        "content": {"uri": node.tile_id},
    }
    if node.children:
        tile["children"] = [node_to_tile_dict(c, box_fn) for c in node.children]
    return tile
