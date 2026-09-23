# Adapted from dozeri83/geo-register-plugin
# https://github.com/dozeri83/geo-register-plugin
# Original licensed under GPL-3.0
# Modifications by wahyunanandika — July 2026
"""Export a cropped local-frame 3DGS PLY to a UTM-offset georeferenced PLY.

Applies the similarity transform (from solve_transform.py) to every splat's
position, rotation, and scale — not just position — so the resulting PLY is
correctly georeferenced (true north-aligned, correct real-world scale) and
still small enough in magnitude to render without float32 jitter in 3DGS
viewers (SuperSplat, gsplat.js, PlayCanvas, etc).

Pipeline
--------
1. local COLMAP frame  --[s*R*X + t]-->  ECEF
2. ECEF                --[geodetic]-->   lat/lon/height
3. lat/lon/height       --[UTM proj]-->   UTM easting/northing/height
4. UTM (raw)            --[- offset]-->  UTM-offset (small numbers)

Rotation and scale are transformed alongside position:
  - rot_0..3 (quaternion, wxyz) is rotated by R
  - scale_0..2 (LOG-SCALE, per tiles_exporter.py's `scales_log`) has ln(s) ADDED
    (NOT multiplied — these are stored as ln(scale), matching original 3DGS
    training convention where the activation is exp())

Usage
-----
python crop/export_georeferenced_ply.py \
    --ply           splat_local_cropped.ply \
    --transform     similarity_transform.json \
    --utm-epsg      32748 \
    --output        splat_georef_utm_offset.ply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from transform_solver import geodetic_to_ecef, ecef_to_geodetic  # noqa: E402


# ─── PLY I/O (mirror tiles_exporter.py / crop_ply_by_boundary.py) ─────────────

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
        n = int(m.group(1))
        props = re.findall(r"property\s+(\S+)\s+(\S+)", header)
        if not all(t == "float" for t, _ in props):
            raise ValueError("Only float32 PLY properties are supported.")
        names = [name for _, name in props]
        dtype = np.dtype([(name, "<f4") for name in names])
        data = np.fromfile(f, dtype=dtype, count=n)
    if len(data) != n:
        raise ValueError(f"PLY: read {len(data)} of expected {n} vertices.")
    return data, names


def write_ply(path: Path, data: np.ndarray, names: list[str], comments: list[str]):
    n = len(data)
    header_lines = [b"ply\n", b"format binary_little_endian 1.0\n"]
    for c in comments:
        header_lines.append(f"comment {c}\n".encode("ascii"))
    header_lines.append(f"element vertex {n}\n".encode("ascii"))
    for name in names:
        header_lines.append(f"property float {name}\n".encode("ascii"))
    header_lines.append(b"end_header\n")
    with open(path, "wb") as f:
        f.write(b"".join(header_lines))
        data.tofile(f)


# ─── Quaternion helpers (wxyz, Hamilton convention — matches rot_0=w) ────────

def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_mul_batch(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Hamilton product, q_left is a single quat (4,), q_right is (N,4). Both wxyz."""
    w1, x1, y1, z1 = q_left
    w2, x2, y2, z2 = q_right[:, 0], q_right[:, 1], q_right[:, 2], q_right[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def enu_to_ecef_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rotation matrix whose columns are the East, North, Up unit vectors
    expressed in ECEF, at the given reference lat/lon. Standard geodesy formula."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    east = np.array([-so, co, 0.0])
    north = np.array([-sl * co, -sl * so, cl])
    up = np.array([cl * co, cl * so, sl])
    return np.stack([east, north, up], axis=1)  # columns = E, N, U


# ─── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a cropped local-frame 3DGS PLY to UTM-offset georeferenced PLY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ply", required=True, type=Path,
                     help="Input local-frame PLY (should be the CROPPED output "
                          "from crop_ply_by_boundary.py)")
    ap.add_argument("--transform", required=True, type=Path,
                     help="similarity_transform.json from solve_transform.py")
    ap.add_argument("--utm-epsg", default=32748, type=int,
                     help="EPSG code of target UTM zone (32748 = WGS84 UTM 48S)")
    ap.add_argument("--offset-easting", type=float, default=None,
                     help="Fixed easting offset (e.g. 788000). If omitted, "
                          "uses bbox-min of this PLY's splats instead — use "
                          "the fixed value to stay aligned with a mesh/boundary "
                          "that shares a common origin (e.g. boundary_local.shp).")
    ap.add_argument("--offset-northing", type=float, default=None,
                     help="Fixed northing offset (e.g. 9237000). See --offset-easting.")
    ap.add_argument("--offset-height", type=float, default=None,
                     help="Fixed height offset. Default: 0.0 (no shift) when "
                          "--offset-easting/northing are set, matching a mesh "
                          "that keeps raw ellipsoidal height; bbox-min of this "
                          "PLY's altitudes when no fixed E/N offset is given.")
    ap.add_argument("--output", required=True, type=Path, help="Output PLY path")
    args = ap.parse_args()

    if (args.offset_easting is None) != (args.offset_northing is None):
        sys.exit("--offset-easting and --offset-northing must be given together.")

    # 1. Load transform
    sim = json.loads(args.transform.read_text())
    s = float(sim["scale"])
    R = np.array(sim["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.array(sim["translation"], dtype=np.float64)
    q_R = quat_from_matrix(R)
    ln_s = float(np.log(s))

    print(f"Transform: scale={s:.6f} (ln={ln_s:.6f}), "
          f"quat_R=[{q_R[0]:.4f}, {q_R[1]:.4f}, {q_R[2]:.4f}, {q_R[3]:.4f}]")

    # 2. Load PLY
    print(f"Reading PLY: {args.ply}")
    data, names = read_ply(args.ply)
    n = len(data)
    print(f"  {n:,} splats")

    for req in ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
                "scale_0", "scale_1", "scale_2"):
        if req not in names:
            raise ValueError(f"PLY is missing required property '{req}'")

    # 3. Transform positions: local -> ECEF
    positions_local = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float64)
    positions_ecef = (s * (R @ positions_local.T)).T + t

    # 4. ECEF -> geodetic -> UTM
    print("Converting ECEF -> geodetic -> UTM (this loops per-point, may take a moment)...")
    lats = np.empty(n)
    lons = np.empty(n)
    alts = np.empty(n)
    for i in range(n):
        lat, lon, alt = ecef_to_geodetic(*positions_ecef[i])
        lats[i], lons[i], alts[i] = lat, lon, alt

    transformer = Transformer.from_crs("epsg:4326", f"epsg:{args.utm_epsg}", always_xy=True)
    eastings, northings = transformer.transform(lons, lats)
    positions_utm = np.stack([eastings, northings, alts], axis=-1)

    # 5. Compute offset and apply
    print(f"UTM range: E {eastings.min():.2f}..{eastings.max():.2f}, "
          f"N {northings.min():.2f}..{northings.max():.2f}, "
          f"H {alts.min():.2f}..{alts.max():.2f}")
    if args.offset_easting is not None:
        h_off = args.offset_height if args.offset_height is not None else 0.0
        offset = np.array([args.offset_easting, args.offset_northing, h_off])
        print(f"Offset (FIXED, UTM epsg:{args.utm_epsg}): "
              f"E={offset[0]:.4f}, N={offset[1]:.4f}, H={offset[2]:.4f}")
    else:
        h_off = args.offset_height if args.offset_height is not None else float(alts.min())
        offset = np.array([positions_utm[:, 0].min(), positions_utm[:, 1].min(), h_off])
        print(f"Offset (bbox min E/N, H={'FIXED' if args.offset_height is not None else 'bbox min'}, "
              f"UTM epsg:{args.utm_epsg}): E={offset[0]:.4f}, N={offset[1]:.4f}, H={offset[2]:.4f}")
    positions_offset = positions_utm - offset

    # 6. Transform rotations.
    # R alone maps local -> ECEF orientation, but our output positions are in
    # UTM (locally == East-North-Up), not ECEF. So we need one more rotation,
    # ECEF -> ENU, evaluated at a single reference point (scene is only
    # ~250m across, so a single reference lat/lon is accurate enough — same
    # "locally flat" assumption used in the GeoRefGS paper).
    lat_ref, lon_ref = float(np.mean(lats)), float(np.mean(lons))
    R_enu2ecef = enu_to_ecef_matrix(lat_ref, lon_ref)
    R_ecef2enu = R_enu2ecef.T
    R_combined = R_ecef2enu @ R
    q_combined = quat_from_matrix(R_combined)
    print(f"  ENU reference: lat={lat_ref:.6f}, lon={lon_ref:.6f}, "
          f"q_combined=[{q_combined[0]:.4f}, {q_combined[1]:.4f}, "
          f"{q_combined[2]:.4f}, {q_combined[3]:.4f}]")

    quat_local = np.stack(
        [data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=-1
    ).astype(np.float64)
    quat_global = quat_mul_batch(q_combined, quat_local)
    norms = np.linalg.norm(quat_global, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    quat_global /= norms

    # 7. Transform scales: LOG-SCALE, add ln(s)
    scale_local = np.stack(
        [data["scale_0"], data["scale_1"], data["scale_2"]], axis=-1
    ).astype(np.float64)
    scale_global = scale_local + ln_s

    # 8. Write output — copy all fields, overwrite transformed ones
    out = data.copy()
    out["x"], out["y"], out["z"] = (
        positions_offset[:, 0].astype(np.float32),
        positions_offset[:, 1].astype(np.float32),
        positions_offset[:, 2].astype(np.float32),
    )
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = (
        quat_global[:, 0].astype(np.float32), quat_global[:, 1].astype(np.float32),
        quat_global[:, 2].astype(np.float32), quat_global[:, 3].astype(np.float32),
    )
    out["scale_0"], out["scale_1"], out["scale_2"] = (
        scale_global[:, 0].astype(np.float32),
        scale_global[:, 1].astype(np.float32),
        scale_global[:, 2].astype(np.float32),
    )

    comments = [
        f"georeferenced UTM epsg:{args.utm_epsg}, offset applied (subtract to recover UTM)",
        f"offset_easting {offset[0]:.6f}",
        f"offset_northing {offset[1]:.6f}",
        f"offset_height_ellipsoidal_m {offset[2]:.6f}",
        "height is WGS84 ellipsoidal (not orthometric)",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.output, out, names, comments)
    print(f"Saved -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
