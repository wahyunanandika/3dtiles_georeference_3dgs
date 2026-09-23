"""Crop a local-frame 3DGS PLY using a real-world boundary polygon.

Pakai transform yang sama dari solve_transform.py (similarity_transform.json)
untuk invert boundary polygon (UTM, real-world) -> local COLMAP/PLY frame,
lalu filter splat yang jatuh di luar polygon.

File ini ada di folder crop/ dan meng-import fungsi geodetic<->ecef dari
transform_solver.py di root repo, biar matematikanya identik dengan yang
dipakai solve_transform.py. Jalankan dari root repo.

Usage
-----
python crop/crop_ply_by_boundary.py \
    --ply           splat_local.ply \
    --transform     similarity_transform.json \
    --boundary      01.Boundary/boundary.shp \
    --utm-epsg      32748 \
    --output        splat_local_cropped.ply

Catatan penting
----------------
- --ply HARUS PLY dalam frame yang SAMA dengan yang dipakai waktu solve
  transform (yaitu PLY yang sudah di-orient sesuai COLMAP world space --
  kalau workflow lo biasanya rotate -90 X di SuperSplat dulu sebelum masuk
  pipeline, pakai PLY versi itu, BUKAN PLY mentah hasil training Y-up).
- boundary.shp harus real-world UTM (bukan boundary_local.shp yang cuma
  di-offset origin -- itu TIDAK match local frame PLY/COLMAP).
- Crop ini footprint-only (2D, mengikuti bidang horizontal boundary).
  Tinggi (up-axis) splat tidak difilter -- semua splat di dalam footprint
  ikut kepotong penuh dari bawah sampai atas.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import shapefile  # pyshp
from pyproj import Transformer
from matplotlib.path import Path as MplPath

# import matematika ECEF<->geodetic yang identik dengan solve_transform.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from transform_solver import geodetic_to_ecef, ecef_to_geodetic  # noqa: E402


# ─── PLY I/O (mirror tiles_exporter.py: binary_little_endian, all-float props) ──

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
    return data, names, header


def write_ply(path: Path, data: np.ndarray, names: list[str]):
    n = len(data)
    header_lines = [
        b"ply\n",
        b"format binary_little_endian 1.0\n",
        f"element vertex {n}\n".encode("ascii"),
    ]
    for name in names:
        header_lines.append(f"property float {name}\n".encode("ascii"))
    header_lines.append(b"end_header\n")
    with open(path, "wb") as f:
        f.write(b"".join(header_lines))
        data.tofile(f)


# ─── Boundary loading ────────────────────────────────────────────────────────

def load_boundary_polygon_utm(shp_path: Path) -> np.ndarray:
    """Return (N,2) array of (easting, northing) for the first polygon shape."""
    sf = shapefile.Reader(str(shp_path))
    shapes = sf.shapes()
    if not shapes:
        raise ValueError(f"No shapes found in {shp_path}")
    pts = np.array(shapes[0].points, dtype=np.float64)
    # drop closing duplicate vertex if present
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Crop a local-frame 3DGS PLY using a real-world boundary polygon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ply", required=True, type=Path, help="Input local-frame PLY")
    ap.add_argument("--transform", required=True, type=Path,
                     help="similarity_transform.json from solve_transform.py")
    ap.add_argument("--boundary", required=True, type=Path,
                     help="boundary.shp in real-world UTM (not boundary_local.shp)")
    ap.add_argument("--utm-epsg", default=32748, type=int,
                     help="EPSG code of the boundary shapefile's UTM zone "
                          "(32748 = WGS84 UTM 48S)")
    ap.add_argument("--output", required=True, type=Path, help="Output cropped PLY")
    args = ap.parse_args()

    # 1. Load transform
    sim = json.loads(args.transform.read_text())
    s = float(sim["scale"])
    R = np.array(sim["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.array(sim["translation"], dtype=np.float64)

    def ecef_to_local(p_ecef: np.ndarray) -> np.ndarray:
        return R.T @ ((p_ecef - t) / s)

    def ecef_dir_to_local(d_ecef: np.ndarray) -> np.ndarray:
        # direction vector, no translation
        return (R.T @ d_ecef) / s

    # 2. Load boundary polygon (UTM -> lat/lon)
    utm_pts = load_boundary_polygon_utm(args.boundary)
    transformer = Transformer.from_crs(f"epsg:{args.utm_epsg}", "epsg:4326",
                                        always_xy=True)
    lons, lats = transformer.transform(utm_pts[:, 0], utm_pts[:, 1])
    print(f"Boundary: {len(utm_pts)} vertices, "
          f"lat {lats.min():.6f}..{lats.max():.6f}, "
          f"lon {lons.min():.6f}..{lons.max():.6f}")

    # 3. Altitude guess: ECEF position of the local origin (translation vector)
    _, _, alt_guess = ecef_to_geodetic(*t)
    print(f"Using altitude guess {alt_guess:.1f} m (ellipsoidal, from transform origin)")

    # 4. Boundary vertices -> ECEF -> local frame
    ecef_boundary = np.array([geodetic_to_ecef(lat, lon, alt_guess)
                               for lat, lon in zip(lats, lons)])
    local_boundary = np.array([ecef_to_local(e) for e in ecef_boundary])

    # 5. Determine local "up" direction (so the footprint test works regardless
    #    of which raw local axis happens to be vertical)
    lat_c, lon_c = float(np.mean(lats)), float(np.mean(lons))
    ecef_a = geodetic_to_ecef(lat_c, lon_c, alt_guess)
    ecef_b = geodetic_to_ecef(lat_c, lon_c, alt_guess + 100.0)
    up_local = ecef_dir_to_local(ecef_b - ecef_a)
    up_local /= np.linalg.norm(up_local)

    # orthonormal basis for the horizontal plane
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, up_local)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up_local, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up_local, e1)

    def project2d(pts3d: np.ndarray) -> np.ndarray:
        return np.stack([pts3d @ e1, pts3d @ e2], axis=-1)

    boundary_2d = project2d(local_boundary)
    poly_path = MplPath(boundary_2d)

    # 6. Load PLY, project positions, test containment
    print(f"Reading PLY: {args.ply}")
    data, names, _ = read_ply(args.ply)
    n_total = len(data)
    positions = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float64)
    print(f"  {n_total:,} splats")

    positions_2d = project2d(positions)
    mask = poly_path.contains_points(positions_2d)
    n_kept = int(mask.sum())
    print(f"Inside boundary: {n_kept:,} / {n_total:,} "
          f"({100.0 * n_kept / n_total:.2f}%)")

    cropped = data[mask]

    # 7. Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.output, cropped, names)
    print(f"Saved -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
