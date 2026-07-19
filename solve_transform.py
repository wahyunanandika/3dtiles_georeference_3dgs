"""Solve the PLY → ECEF similarity transform.

Usage
-----
python solve_transform.py \
    --images-bin  path/to/sparse/0/images.bin \
    --metashape   path/to/camera_export.xml \
    --output      similarity_transform.json

Optional: add --points3d path/to/sparse/0/points3D.bin to verify
terrain altitude in the output (printed only, not used in solve).

Optional: add --geoid-model {egm96-5,egm2008-5,egm2008-2_5,egm2008-1} to
also compute geoid undulation (N) and orthometric height at the scene
centroid. The grid file is downloaded automatically on first use. This is
informational only — the similarity transform itself always operates on
ellipsoidal (WGS84/GPS) heights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from colmap_reader import read_images_bin, read_points3d_bin
from metashape_parser import parse_metashape_xml, MetashapeXMLError
from transform_solver import solve_ply_to_ecef, ecef_to_geodetic
from geoid_model import get_geoid_undulation, GeoidModelError, GEOID_MODELS


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Solve PLY→ECEF similarity transform from COLMAP + Metashape XML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--images-bin",  required=True, type=Path,
                    help="COLMAP sparse/0/images.bin")
    ap.add_argument("--metashape",   required=True, type=Path,
                    help="Metashape camera XML export (GEOGCS/WGS84 CRS)")
    ap.add_argument("--points3d",    default=None,  type=Path,
                    help="(optional) COLMAP sparse/0/points3D.bin — used only to "
                         "verify terrain altitude in the output")
    ap.add_argument("--output",      default="similarity_transform.json", type=Path,
                    help="Output JSON path")
    ap.add_argument("--ransac-thr",  default=5.0, type=float,
                    help="RANSAC inlier threshold in metres")
    ap.add_argument("--geoid-model", default=None, choices=list(GEOID_MODELS),
                    help="(optional) also compute geoid undulation/orthometric "
                         "height at the scene centroid using this model")
    args = ap.parse_args()

    # ── 1. Read COLMAP cameras ────────────────────────────────────────────────
    print(f"Reading COLMAP images.bin: {args.images_bin}")
    try:
        colmap_cameras = read_images_bin(args.images_bin)
    except Exception as exc:
        print(f"ERROR reading images.bin: {exc}", file=sys.stderr)
        return 1
    print(f"  {len(colmap_cameras)} cameras read.")

    # ── 2. Read Metashape XML ─────────────────────────────────────────────────
    print(f"Reading Metashape XML: {args.metashape}")
    try:
        metashape_data = parse_metashape_xml(args.metashape)
    except MetashapeXMLError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    xml_cameras = metashape_data["cameras"]
    print(f"  {len(xml_cameras)} GPS cameras loaded.")

    # ── 3. Optional sparse points ─────────────────────────────────────────────
    points3d = None
    if args.points3d:
        print(f"Reading points3D.bin: {args.points3d}")
        try:
            points3d = read_points3d_bin(args.points3d)
            print(f"  {len(points3d):,} sparse points read.")
        except Exception as exc:
            print(f"  WARNING: could not read points3D.bin: {exc}")

    # ── 4. Solve ──────────────────────────────────────────────────────────────
    print("Solving PLY → ECEF similarity transform …")
    try:
        result = solve_ply_to_ecef(
            colmap_cameras,
            metashape_data,
            ransac_inlier_thr_m=args.ransac_thr,
            points3d=points3d,
        )
    except Exception as exc:
        print(f"ERROR solving transform: {exc}", file=sys.stderr)
        return 1

    # ── 5. Optional geoid undulation at scene centroid ───────────────────────
    geoid_info = None
    if args.geoid_model:
        lat_c = result["centroid_lat"]
        lon_c = result["centroid_lon"]
        alt_c = result["centroid_alt_ellipsoidal_m"]
        print(f"Computing geoid undulation ({args.geoid_model}) at scene centroid …")
        try:
            N = get_geoid_undulation(lat_c, lon_c, model=args.geoid_model)
            geoid_info = {
                "model": args.geoid_model,
                "undulation_m": N,
                "ellipsoidal_height_m": alt_c,
                "orthometric_height_m": alt_c - N,
            }
            print(f"  N (undulation)     = {N:+.3f} m")
            print(f"  orthometric height = {alt_c - N:.2f} m (ellipsoidal {alt_c:.2f} m)")
        except GeoidModelError as exc:
            print(f"  WARNING: could not compute geoid undulation: {exc}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    payload = {
        "scale":       result["scale"],
        "rotation":    result["rotation"],
        "translation": result["translation"],
        "rmse_m":      result["rmse_m"],
        "n_inliers":   result["n_inliers"],
        "n_total":     result["n_total"],
    }
    if geoid_info:
        payload["geoid"] = geoid_info
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved → {args.output}")
    print(f"  scale       = {result['scale']:.8f}")
    print(f"  RMSE (GPS)  = {result['rmse_m']:.4f} m")
    print(f"  inliers     = {result['n_inliers']}/{result['n_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
