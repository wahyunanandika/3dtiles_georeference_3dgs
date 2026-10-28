# Adapted from dozeri83/geo-register-plugin
# https://github.com/dozeri83/geo-register-plugin
# Original licensed under GPL-3.0
# Modifications by wahyunanandika — June 2026
# Hierarchical LOD option + tileset writer split — October 2026
"""Export a geo-registered 3DGS PLY to 3D Tiles 1.1 (SPZ-compressed GLB).

Usage (CLI)
-----------
python tiles_exporter.py splat.ply similarity_transform.json out_dir/ [options]

Options
-------
--tiling flat|lod           flat = octree, content in leaves only (default, unchanged)
                            lod  = hierarchical LOD: every node has content (see lod_tiling.py)
--tiles-version 1.1         3D Tiles version of tileset.json (2.0 reserved, not ratified yet)
--max-sh-degree  0|1|2|3    SH degree to include (default: 3)
--max-splats-per-tile INT   Override auto tile-size limit (also the per-tile budget in lod mode)
--min-tile-size FLOAT       Min octree cell diameter in scene units (default: 0.1)
--error-scale FLOAT         lod: geometricError multiplier (default: 16)
--lod-max-inflate FLOAT     lod: max enlargement of a kept splat per level (default: 2)
--fraction FLOAT            Sub-sample fraction for quick tests (0 < f ≤ 1)
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

import lod_tiling
from spz_encode import DIM_FOR_DEGREE, encode_spz_v3
from tileset_writer import (PLANNED_VERSIONS, SUPPORTED_VERSIONS, box_from_bounds,
                            root_transform, tileset_document)

GLB_MAGIC  = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN  = 0x004E4942


# ─── GLB writer ───────────────────────────────────────────────────────────────

def _write_spz_glb(
    path: Path,
    spz_blob: bytes,
    num_points: int,
    sh_degree: int,
    positions: np.ndarray,
) -> None:
    pmin = positions.min(axis=0).tolist()
    pmax = positions.max(axis=0).tolist()

    accessors: list = []
    attrs: dict = {}

    def add_acc(name, type_str, ct, normalized=False, with_minmax=False):
        acc = {"componentType": ct, "count": num_points, "type": type_str}
        if normalized:
            acc["normalized"] = True
        if with_minmax:
            acc["min"] = pmin
            acc["max"] = pmax
        accessors.append(acc)
        attrs[name] = len(accessors) - 1

    add_acc("POSITION",                              "VEC3", 5126, with_minmax=True)
    add_acc("COLOR_0",                               "VEC4", 5121, normalized=True)
    add_acc("KHR_gaussian_splatting:SCALE",          "VEC3", 5126)
    add_acc("KHR_gaussian_splatting:ROTATION",       "VEC4", 5126)
    for d in range(1, sh_degree + 1):
        for k in range({1: 3, 2: 5, 3: 7}[d]):
            add_acc(f"KHR_gaussian_splatting:SH_DEGREE_{d}_COEF_{k}", "VEC4", 5126)

    spz_len   = len(spz_blob)
    bin_pad   = (-spz_len) % 4
    bin_total = spz_len + bin_pad

    gltf = {
        "asset": {"version": "2.0", "generator": "gs_georef/tiles_exporter.py"},
        "extensionsUsed": [
            "KHR_gaussian_splatting",
            "KHR_gaussian_splatting_compression_spz_2",
            "KHR_materials_unlit",
        ],
        "extensionsRequired": [
            "KHR_gaussian_splatting",
            "KHR_gaussian_splatting_compression_spz_2",
        ],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "matrix": [
            1.0, 0.0,  0.0, 0.0,
            0.0, 0.0, -1.0, 0.0,
            0.0, 1.0,  0.0, 0.0,
            0.0, 0.0,  0.0, 1.0,
        ]}],
        "meshes": [{"primitives": [{
            "mode": 0,
            "material": 0,
            "attributes": attrs,
            "extensions": {
                "KHR_gaussian_splatting": {
                    "extensions": {
                        "KHR_gaussian_splatting_compression_spz_2": {"bufferView": 0}
                    }
                }
            },
        }]}],
        "materials": [{
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]},
            "extensions": {"KHR_materials_unlit": {}},
        }],
        "buffers":     [{"byteLength": bin_total}],
        "bufferViews": [{"buffer": 0, "byteLength": spz_len}],
        "accessors":   accessors,
    }

    json_blob  = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad   = (-len(json_blob)) % 4
    json_total = len(json_blob) + json_pad
    total_len  = 12 + 8 + json_total + 8 + bin_total

    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, total_len))
        f.write(struct.pack("<II",  json_total, CHUNK_JSON))
        f.write(json_blob)
        f.write(b" " * json_pad)
        f.write(struct.pack("<II",  bin_total, CHUNK_BIN))
        f.write(spz_blob)
        if bin_pad:
            f.write(b"\x00" * bin_pad)


# ─── Octree ───────────────────────────────────────────────────────────────────

class _OctreeNode:
    def __init__(self, indices, pmin, pmax, depth):
        self.indices  = indices
        self.pmin     = pmin
        self.pmax     = pmax
        self.depth    = depth
        self.children: list[_OctreeNode] = []
        self.tile_id: str | None = None

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def center(self):
        return (self.pmin + self.pmax) * 0.5

    @property
    def half_axes(self):
        return (self.pmax - self.pmin) * 0.5

    @property
    def geometric_error(self):
        return float(np.linalg.norm(self.pmax - self.pmin))


def _auto_max_splats(n_total: int) -> int:
    if n_total <    500_000: return  20_000
    if n_total <  2_000_000: return  30_000
    if n_total < 10_000_000: return  50_000
    if n_total < 20_000_000: return  75_000
    return 100_000


def _build_octree(positions, indices, max_splats, min_size, depth=0):
    pts  = positions[indices]
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    node = _OctreeNode(indices, pmin, pmax, depth)

    if len(indices) <= max_splats or np.linalg.norm(pmax - pmin) < min_size:
        return node

    mid    = (pmin + pmax) * 0.5
    bits   = (pts >= mid).astype(np.uint8)
    octant = bits[:, 0] | (bits[:, 1] << 1) | (bits[:, 2] << 2)

    children = []
    for k in range(8):
        mask = (octant == k)
        if not mask.any():
            continue
        child = _build_octree(positions, indices[mask], max_splats, min_size, depth + 1)
        children.append(child)

    if len(children) <= 1:
        return node
    node.children = children
    return node


def _collect_leaves(node):
    if node.is_leaf:
        return [node]
    return [leaf for child in node.children for leaf in _collect_leaves(child)]


def _assign_tile_ids(node, counter):
    if node.is_leaf:
        node.tile_id = f"tile_{counter[0]:04d}.glb"
        counter[0] += 1
    else:
        for child in node.children:
            _assign_tile_ids(child, counter)


def _node_to_tile_dict(node):
    tile = {
        "boundingVolume": {"box": box_from_bounds(node.pmin, node.pmax)},
        "geometricError": 0.0 if node.is_leaf else node.geometric_error,
        "refine": "REPLACE",
    }
    if node.is_leaf:
        tile["content"] = {"uri": node.tile_id}
    else:
        tile["children"] = [_node_to_tile_dict(ch) for ch in node.children]
    return tile


# ─── Tileset builder ──────────────────────────────────────────────────────────

def _build_tileset_tiled(sim: dict, root_node: _OctreeNode, version: str = "1.1") -> dict:
    """Flat octree tileset (content in leaves only)."""
    geom_err = float(np.linalg.norm(root_node.pmax - root_node.pmin))

    root_tile = _node_to_tile_dict(root_node)
    # Similarity matrix: COLMAP world space → ECEF (no Y/Z flip needed —
    # the flip was for LFS visualizer-world convention, not COLMAP space)
    root_tile["transform"]      = root_transform(sim)
    root_tile["geometricError"] = geom_err
    root_tile["boundingVolume"] = {"box": box_from_bounds(root_node.pmin, root_node.pmax)}

    return tileset_document(root_tile, geom_err, version)


def _build_tileset_lod(sim: dict, root_node: _OctreeNode, version: str = "1.1") -> dict:
    """Hierarchical LOD tileset (content in every node).

    geometricError stays in scene units, as in the flat tileset: the viewer
    applies the root transform's scale itself.
    """
    diag = float(np.linalg.norm(root_node.pmax - root_node.pmin))
    root_tile = lod_tiling.node_to_tile_dict(root_node, box_from_bounds)
    root_tile["transform"] = root_transform(sim)
    # Beyond this error the whole tileset is skipped; keep it >= the root's own error.
    return tileset_document(root_tile, max(diag, root_node.lod_error), version)


# ─── PLY reader ───────────────────────────────────────────────────────────────

def read_ply(path: Path):
    with open(path, "rb") as f:
        lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("EOF before end_header in PLY")
            lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(lines).decode("ascii", errors="replace")
        if "format binary_little_endian" not in header:
            raise ValueError("Only binary_little_endian PLY is supported.")
        m = re.search(r"element vertex (\d+)", header)
        if not m:
            raise ValueError("Could not find vertex count in PLY header.")
        n      = int(m.group(1))
        props  = re.findall(r"property\s+(\S+)\s+(\S+)", header)
        if not all(t == "float" for t, _ in props):
            raise ValueError("Only float32 PLY properties are supported.")
        names  = [name for _, name in props]
        dtype  = np.dtype([(name, "<f4") for name in names])
        data   = np.fromfile(f, dtype=dtype, count=n)
    if len(data) != n:
        raise ValueError(f"PLY: read {len(data)} of expected {n} vertices.")
    return data, names


def build_arrays_from_ply(ply, names, sh_degree: int):
    positions      = np.stack([ply["x"], ply["y"], ply["z"]], axis=-1).astype(np.float32)
    rotations_wxyz = np.stack(
        [ply["rot_0"], ply["rot_1"], ply["rot_2"], ply["rot_3"]], axis=-1
    ).astype(np.float32)
    scales_log     = np.stack(
        [ply["scale_0"], ply["scale_1"], ply["scale_2"]], axis=-1
    ).astype(np.float32)
    opacity_logit  = ply["opacity"].astype(np.float32)
    f_dc           = np.stack(
        [ply["f_dc_0"], ply["f_dc_1"], ply["f_dc_2"]], axis=-1
    ).astype(np.float32)

    f_rest_rgb  = None
    eff_sh      = sh_degree
    if sh_degree > 0:
        rest_names = sorted(
            (nm for nm in names if nm.startswith("f_rest_")),
            key=lambda s: int(s.rsplit("_", 1)[1]),
        )
        if rest_names and len(rest_names) % 3 == 0:
            K      = len(rest_names) // 3
            needed = DIM_FOR_DEGREE[sh_degree]
            if K >= needed:
                rest = np.stack([ply[nm] for nm in rest_names], axis=-1).astype(np.float32)
                rest = rest.reshape(-1, 3, K).transpose(0, 2, 1)
                f_rest_rgb = np.ascontiguousarray(rest[:, :needed, :])
            else:
                eff_sh = 0
        else:
            eff_sh = 0

    return positions, rotations_wxyz, scales_log, opacity_logit, f_dc, f_rest_rgb, eff_sh


# ─── Core exporter ────────────────────────────────────────────────────────────

def export_tiles(
    positions:      np.ndarray,
    rotations_wxyz: np.ndarray,
    scales_log:     np.ndarray,
    opacity_logit:  np.ndarray,
    f_dc:           np.ndarray,
    f_rest_rgb:     np.ndarray | None,
    sh_degree:      int,
    transform:      dict,
    out_dir:        Path,
    max_splats_per_tile: int  = 0,
    min_tile_size:       float = 0.1,
    progress_cb          = None,
    tiling:              str   = "flat",
    tiles_version:       str   = "1.1",
    error_scale:         float = lod_tiling.DEFAULT_ERROR_SCALE,
    lod_max_inflate:     float = lod_tiling.DEFAULT_MAX_INFLATE,
) -> dict:
    """Write tileset.json + tiles to out_dir.

    tiling="flat": tile_XXXX.glb, content in octree leaves only (original behaviour).
    tiling="lod" : tiles/D<depth>/NNNNNN.glb, content in every octree node.
    Returns a small summary dict.
    """
    if tiling not in ("flat", "lod"):
        raise ValueError(f"tiling must be 'flat' or 'lod', got {tiling!r}")
    # Fail before any heavy work if the requested version has no writer.
    tileset_document({}, 0.0, tiles_version)

    def prog(f):
        if progress_cb:
            progress_cb(f)

    positions = positions.astype(np.float32)

    # wxyz → xyzw for SPZ, normalise
    rotations_xyzw = np.stack([
        rotations_wxyz[:, 1], rotations_wxyz[:, 2],
        rotations_wxyz[:, 3], rotations_wxyz[:, 0],
    ], axis=-1).astype(np.float32)
    norms = np.linalg.norm(rotations_xyzw, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    rotations_xyzw /= norms

    prog(0.05)

    n_total   = len(positions)
    effective = max_splats_per_tile if max_splats_per_tile > 0 else _auto_max_splats(n_total)
    all_idx   = np.arange(n_total, dtype=np.int64)
    root_node = _build_octree(positions, all_idx, effective, min_tile_size)
    leaves    = _collect_leaves(root_node)
    _assign_tile_ids(root_node, [0])

    print(f"  octree: {len(leaves)} leaf tiles  "
          f"(depth {max(l.depth for l in leaves)}, "
          f"max {max(len(l.indices) for l in leaves):,} splats/tile, "
          f"threshold {effective:,})")

    prog(0.15)

    def write_tile(path: Path, idx: np.ndarray, dlog: np.ndarray | None = None) -> None:
        sl = scales_log[idx].astype(np.float32)
        if dlog is not None:
            sl = sl + dlog[:, None]
        spz = encode_spz_v3(
            positions      = positions[idx],
            rotations_xyzw = rotations_xyzw[idx],
            scales_log     = sl,
            opacity_logit  = opacity_logit[idx].astype(np.float32),
            f_dc           = f_dc[idx].astype(np.float32),
            f_rest_rgb     = f_rest_rgb[idx] if f_rest_rgb is not None else None,
            sh_degree      = sh_degree,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_spz_glb(path, spz, int(len(idx)), sh_degree, positions[idx])

    if tiling == "flat":
        for i, leaf in enumerate(leaves):
            write_tile(out_dir / leaf.tile_id, leaf.indices)
            prog(0.15 + 0.75 * (i + 1) / len(leaves))
        tileset = _build_tileset_tiled(transform, root_node, tiles_version)
        summary = {"tiling": "flat", "tiles": len(leaves), "splats_written": n_total}
    else:
        internal = lod_tiling.build_hlod(
            root_node, positions, scales_log, opacity_logit, effective,
            max_inflate=lod_max_inflate,
            progress_cb=lambda f: prog(0.15 + 0.15 * f))
        lod_tiling.assign_errors(root_node, scales_log, error_scale)
        nodes = lod_tiling.assign_uris(root_node)
        per_depth: dict[int, list[int]] = {}
        written = 0
        # coarse first, so a cancelled run still leaves the top of the tree
        for i, node in enumerate(sorted(nodes, key=lambda n: n.depth)):
            dlog = None if node.is_leaf else node.lod_dlog
            write_tile(out_dir / node.tile_id, node.lod_idx, dlog)
            per_depth.setdefault(node.depth, [0, 0])
            per_depth[node.depth][0] += 1
            per_depth[node.depth][1] += len(node.lod_idx)
            written += len(node.lod_idx)
            prog(0.30 + 0.65 * (i + 1) / len(nodes))
        print(f"\n  hlod: {len(internal)} internal + {len(leaves)} leaf tiles, "
              f"{written:,} splats written ({written / n_total:.2f}x input)")
        for d, (nt, ns) in sorted(per_depth.items()):
            print(f"    D{d}: {nt:5d} tiles, {ns:>11,} splats")
        tileset = _build_tileset_lod(transform, root_node, tiles_version)
        summary = {"tiling": "lod", "tiles": len(nodes), "splats_written": written,
                   "per_depth": {d: {"tiles": v[0], "splats": v[1]} for d, v in per_depth.items()}}

    with open(out_dir / "tileset.json", "w", encoding="utf-8") as f:
        json.dump(tileset, f, indent=2)
    prog(1.0)
    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse, sys

    ap = argparse.ArgumentParser(
        description="Export a georeferenced 3DGS PLY to 3D Tiles 1.1 (SPZ).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("ply",             type=Path, help="input PLY file")
    ap.add_argument("similarity_json", type=Path, help="similarity_transform.json")
    ap.add_argument("out_dir",         type=Path, help="output directory")
    ap.add_argument("--max-sh-degree",       type=int,   choices=[0,1,2,3], default=3)
    ap.add_argument("--max-splats-per-tile", type=int,   default=0)
    ap.add_argument("--min-tile-size",       type=float, default=0.1)
    ap.add_argument("--tiling",              choices=["flat", "lod"], default="flat",
                    help="flat = content in octree leaves only; lod = hierarchical LOD")
    ap.add_argument("--tiles-version",       choices=list(SUPPORTED_VERSIONS + PLANNED_VERSIONS),
                    default="1.1", help="3D Tiles version (2.0 reserved until ratified)")
    ap.add_argument("--error-scale",         type=float, default=lod_tiling.DEFAULT_ERROR_SCALE,
                    help="lod: geometricError multiplier (higher = refine sooner)")
    ap.add_argument("--lod-max-inflate",     type=float, default=lod_tiling.DEFAULT_MAX_INFLATE,
                    help="lod: max enlargement of a kept splat per level")
    ap.add_argument("--fraction",            type=float, default=1.0,
                    help="Sub-sample fraction for quick tests (0 < f ≤ 1)")
    ap.add_argument("--seed",                type=int,   default=0)
    args = ap.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        sys.exit(f"--fraction must be in (0, 1], got {args.fraction}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading PLY: {args.ply}")
    ply, names = read_ply(args.ply)
    print(f"  {len(ply):,} splats, {len(names)} properties")

    if args.fraction < 1.0:
        n_keep = int(round(len(ply) * args.fraction))
        print(f"  Sub-sampling → {n_keep:,} splats")
        ply = ply[np.random.default_rng(args.seed).choice(len(ply), n_keep, replace=False)]

    positions, rotations_wxyz, scales_log, opacity_logit, f_dc, f_rest_rgb, eff_sh = (
        build_arrays_from_ply(ply, names, args.max_sh_degree)
    )
    del ply

    print(f"  SH degree: {eff_sh}")
    print(f"  Position range:")
    print(f"    X: {positions[:,0].min():.2f} .. {positions[:,0].max():.2f}")
    print(f"    Y: {positions[:,1].min():.2f} .. {positions[:,1].max():.2f}")
    print(f"    Z: {positions[:,2].min():.2f} .. {positions[:,2].max():.2f}")

    with open(args.similarity_json) as f:
        sim = json.load(f)
    # normalise key names
    transform = {
        "scale":       sim.get("scale",       sim.get("s")),
        "rotation":    sim.get("rotation",    sim.get("R")),
        "translation": sim.get("translation", sim.get("t")),
    }
    print(f"\nTransform loaded from {args.similarity_json}")

    def log_prog(f):
        print(f"  {int(f * 100):3d}%", end="\r", flush=True)

    summary = export_tiles(
        positions          = positions,
        rotations_wxyz     = rotations_wxyz,
        scales_log         = scales_log,
        opacity_logit      = opacity_logit,
        f_dc               = f_dc,
        f_rest_rgb         = f_rest_rgb,
        sh_degree          = eff_sh,
        transform          = transform,
        out_dir            = args.out_dir,
        max_splats_per_tile= args.max_splats_per_tile,
        min_tile_size      = args.min_tile_size,
        progress_cb        = log_prog,
        tiling             = args.tiling,
        tiles_version      = args.tiles_version,
        error_scale        = args.error_scale,
        lod_max_inflate    = args.lod_max_inflate,
    )
    print(f"\nDone → {args.out_dir}/tileset.json + {summary['tiles']} tiles")


if __name__ == "__main__":
    main()