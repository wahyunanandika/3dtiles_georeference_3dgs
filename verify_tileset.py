"""Quick sanity-check for a finished tileset.

Usage
-----
python verify_tileset.py path/to/out_dir/

Optional: python verify_tileset.py path/to/out_dir/ --geoid-model egm2008-2_5
to also print orthometric height at the bbox center (informational only,
grid downloaded automatically on first use).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path


def ecef_to_lla(x, y, z):
    a = 6_378_137.0
    e2 = 0.00669437999014
    lon = math.degrees(math.atan2(y, x))
    p   = math.sqrt(x*x + y*y)
    lat = math.degrees(math.atan2(z, p * (1.0 - e2)))
    for _ in range(10):
        lr  = math.radians(lat)
        N   = a / math.sqrt(1.0 - e2 * math.sin(lr)**2)
        lat = math.degrees(math.atan2(z + e2 * N * math.sin(lr), p))
    lr  = math.radians(lat)
    N   = a / math.sqrt(1.0 - e2 * math.sin(lr)**2)
    alt = p / math.cos(lr) - N
    return lat, lon, alt


def lla_to_utm(lat_deg, lon_deg):
    """Convert WGS84 lat/lon to UTM easting, northing, zone number, hemisphere."""
    a   = 6_378_137.0
    f   = 1.0 / 298.257223563
    b   = a * (1.0 - f)
    e2  = 1.0 - (b / a) ** 2
    ep2 = e2 / (1.0 - e2)
    k0  = 0.9996

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    zone = int((lon_deg + 180) / 6) + 1
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    N   = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    T   = math.tan(lat) ** 2
    C   = ep2 * math.cos(lat) ** 2
    A   = math.cos(lat) * (lon - lon0)
    e2_ = e2
    M   = a * (
        (1 - e2_/4 - 3*e2_**2/64 - 5*e2_**3/256) * lat
        - (3*e2_/8 + 3*e2_**2/32 + 45*e2_**3/1024) * math.sin(2*lat)
        + (15*e2_**2/256 + 45*e2_**3/1024) * math.sin(4*lat)
        - (35*e2_**3/3072) * math.sin(6*lat)
    )

    easting = k0 * N * (
        A + (1 - T + C) * A**3/6
        + (5 - 18*T + T**2 + 72*C - 58*ep2) * A**5/120
    ) + 500_000.0

    northing = k0 * (M + N * math.tan(lat) * (
        A**2/2
        + (5 - T + 9*C + 4*C**2) * A**4/24
        + (61 - 58*T + T**2 + 600*C - 330*ep2) * A**6/720
    ))

    is_south = lat_deg < 0
    if is_south:
        northing += 10_000_000.0

    hemi = "S" if is_south else "N"
    return easting, northing, zone, hemi


def check(label, ok, detail=""):
    status = "OK " if ok else "FAIL"
    print(f"  {status}  {label}" + (f"  ({detail})" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Sanity-check a finished tileset.")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--geoid-model", default=None,
                    choices=["egm96-5", "egm2008-5", "egm2008-2_5", "egm2008-1"],
                    help="(optional) also print orthometric height at bbox "
                         "center using this geoid model (informational only)")
    args = ap.parse_args()

    out_dir = args.out_dir
    passed = 0
    total  = 0

    # ── tileset.json ──────────────────────────────────────────────────────────
    ts_path = out_dir / "tileset.json"
    if not ts_path.exists():
        print(f"ERROR: {ts_path} not found.")
        sys.exit(1)

    with open(ts_path) as f:
        ts = json.load(f)

    print("\n── tileset.json ──────────────────────────────")

    ge_root    = ts.get("geometricError", 0)
    box        = ts["root"]["boundingVolume"]["box"]
    cx, cy, cz = box[0], box[1], box[2]
    hx, hy, hz = box[3], box[7], box[11]
    diag       = math.sqrt((hx*2)**2 + (hy*2)**2 + (hz*2)**2)
    refine     = ts["root"].get("refine", "?")
    n_children = len(ts["root"].get("children", []))

    import numpy as np
    T = ts["root"].get("transform", [])
    if T:
        M        = np.array(T).reshape(4, 4).T
        scale_m  = float(np.linalg.norm(M[:3, 0]))
        p_box    = np.array([cx, cy, cz, 1.0])
        ecef_box = M @ p_box
        lat, lon, alt = ecef_to_lla(*ecef_box[:3])
        easting, northing, zone, hemi = lla_to_utm(lat, lon)
    else:
        scale_m = 1.0
        lat = lon = alt = easting = northing = 0.0
        zone = 0; hemi = "?"

    def c(label, ok, detail=""):
        nonlocal passed, total
        total += 1
        if ok:
            passed += 1
        return check(label, ok, detail)

    c("geometricError == bbox diagonal",
      abs(ge_root - diag) < 0.1,
      f"{ge_root:.2f} vs {diag:.2f}")
    c("refine == REPLACE",
      refine == "REPLACE",
      refine)
    c("has children (tiled)",
      n_children > 0,
      f"{n_children} children")
    c("transform scale ~ 1.0",
      abs(scale_m - 1.0) < 0.01,
      f"{scale_m:.6f}")
    c("latitude plausible",
      -90 < lat < 90,
      f"{lat:.5f} deg")
    c("longitude plausible",
      -180 < lon < 180,
      f"{lon:.5f} deg")
    c("altitude 0-5000 m",
      0 < alt < 5000,
      f"{alt:.1f} m")

    print(f"\n  Bbox center (WGS84) : lat={lat:.6f}, lon={lon:.6f}, alt={alt:.1f} m")
    print(f"  Bbox center (UTM)   : {easting:.2f} E, {northing:.2f} N, zone {zone}{hemi}")

    if args.geoid_model and T:
        from geoid_model import get_geoid_undulation, GeoidModelError
        try:
            N = get_geoid_undulation(lat, lon, model=args.geoid_model)
            print(f"  Geoid ({args.geoid_model}) : N={N:+.3f} m, "
                  f"orthometric alt={alt - N:.1f} m (ellipsoidal {alt:.1f} m)")
        except GeoidModelError as exc:
            print(f"  Geoid ({args.geoid_model}) : WARNING could not compute — {exc}")

    # ── GLB tiles ─────────────────────────────────────────────────────────────
    tiles = sorted(out_dir.glob("tile_*.glb"))
    print(f"\n── GLB tiles ({len(tiles)}) ────────────────────────────")

    if tiles:
        with open(tiles[0], "rb") as f:
            magic  = struct.unpack("<I", f.read(4))[0]
            ver    = struct.unpack("<I", f.read(4))[0]
            _total = struct.unpack("<I", f.read(4))[0]
            clen   = struct.unpack("<I", f.read(4))[0]
            _ctype = f.read(4)
            gltf   = json.loads(f.read(clen))

        c("GLB magic correct",
          magic == 0x46546C67,
          hex(magic))
        c("glTF version 2",
          ver == 2,
          str(ver))
        ext = gltf["meshes"][0]["primitives"][0].get("extensions", {})
        c("KHR_gaussian_splatting present",
          "KHR_gaussian_splatting" in ext)
        node_matrix = gltf["nodes"][0].get("matrix", [])
        c("Y-up node matrix",
          len(node_matrix) == 16 and abs(node_matrix[6] - (-1.0)) < 1e-6,
          str(node_matrix[6] if len(node_matrix) >= 7 else "?"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Summary ───────────────────────────────────")
    print(f"  {passed}/{total} checks passed")
    if passed == total:
        print("  READY TO UPLOAD TO CESIUM ION")
    else:
        print("  Fix the failing checks before uploading.")
    print()


if __name__ == "__main__":
    main()
