# Adapted from dozeri83/geo-register-plugin
# https://github.com/dozeri83/geo-register-plugin
# Original licensed under GPL-3.0
# Modifications by wahyunanandika — June 2026
"""Solve the PLY → ECEF similarity transform.

Approach
--------
Solve directly:  C_colmap  →  ECEF

using matched (camera_center_colmap, GPS_ECEF) pairs from:
  - COLMAP images.bin          → C_colmap = -R.T @ T
  - Metashape camera XML       → GPS lat/lon/alt per image

The Metashape chunk-level <transform> (scene → ECEF) is read from the XML
for reference / verification but is NOT used in the primary solve — solving
directly in one step avoids compounding errors from the two-step approach.

Why this works for typical drone COLMAP datasets
-------------------------------------------------
In a nadir or oblique drone survey processed through COLMAP + Metashape, the
COLMAP world space aligns its axes roughly with the local ENU frame, so the
camera centres C_colmap carry meaningful X, Y, Z variation that Umeyama can
map to ECEF with a single similarity transform (scale ≈ 1, RMSE ≈ GPS accuracy).

The resulting transform also applies correctly to PLY splat positions because
PLY and cameras share the same COLMAP world coordinate system.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


# ─── WGS-84 constants ─────────────────────────────────────────────────────────

_A  = 6_378_137.0
_E2 = 0.00669437999014


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lr = np.radians(lat_deg)
    lo = np.radians(lon_deg)
    N  = _A / np.sqrt(1.0 - _E2 * np.sin(lr) ** 2)
    return np.array([
        (N + alt_m) * np.cos(lr) * np.cos(lo),
        (N + alt_m) * np.cos(lr) * np.sin(lo),
        (N * (1.0 - _E2) + alt_m) * np.sin(lr),
    ])


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    lon = np.degrees(np.arctan2(y, x))
    p   = np.sqrt(x * x + y * y)
    lat = np.degrees(np.arctan2(z, p * (1.0 - _E2)))
    for _ in range(10):
        lr  = np.radians(lat)
        N   = _A / np.sqrt(1.0 - _E2 * np.sin(lr) ** 2)
        lat = np.degrees(np.arctan2(z + _E2 * N * np.sin(lr), p))
    lr  = np.radians(lat)
    N   = _A / np.sqrt(1.0 - _E2 * np.sin(lr) ** 2)
    cos_lat = np.cos(lr)
    alt = (p / cos_lat - N) if abs(cos_lat) > 1e-10 else (abs(z) / np.sin(lr) - N * (1.0 - _E2))
    return float(lat), float(lon), float(alt)


# ─── Umeyama similarity solver ────────────────────────────────────────────────

def umeyama(
    src: list | np.ndarray,
    dst: list | np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Closed-form similarity (scale, R, t) minimising ||dst - (s*R@src + t)||².
    Returns (scale, R, t, rmse_m).
    """
    P = np.asarray(src, dtype=np.float64)
    Q = np.asarray(dst, dtype=np.float64)
    mu_p = P.mean(0); mu_q = Q.mean(0)
    Pc = P - mu_p;    Qc = Q - mu_q
    var_p = float(np.mean((Pc ** 2).sum(1)))
    H     = Qc.T @ Pc / len(P)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.dot(D, np.diag(S)) / var_p)
    t = mu_q - s * (R @ mu_p)
    pred = (s * (R @ P.T)).T + t
    rmse = float(np.sqrt(np.mean(np.sum((Q - pred) ** 2, axis=1))))
    return s, R, t, rmse


def ransac_umeyama(
    src: list | np.ndarray,
    dst: list | np.ndarray,
    inlier_thr_m: float = 5.0,
    confidence: float   = 0.99,
    max_iter: int       = 2000,
    min_samples: int    = 4,
    seed: int           = 0,
) -> tuple[float, np.ndarray, np.ndarray, float, np.ndarray]:
    """RANSAC wrapper around Umeyama. Returns (s, R, t, rmse, inlier_mask)."""
    rng = np.random.default_rng(seed)
    P   = np.asarray(src, dtype=np.float64)
    Q   = np.asarray(dst, dtype=np.float64)
    n   = len(P)
    assert n >= min_samples

    best_mask    = np.zeros(n, dtype=bool)
    best_inliers = 0

    for it in range(max_iter):
        idx  = rng.choice(n, min_samples, replace=False)
        s, R, t, _ = umeyama(P[idx], Q[idx])
        pred = (s * (R @ P.T)).T + t
        errs = np.linalg.norm(Q - pred, axis=1)
        mask = errs < inlier_thr_m
        if mask.sum() > best_inliers:
            best_inliers = mask.sum()
            best_mask    = mask
            ratio = best_inliers / n
            if ratio > 0.0:
                prob = ratio ** min_samples
                if prob >= 1.0:
                    break
                denom = np.log(max(1e-12, 1.0 - prob))
                needed = int(np.ceil(np.log(1.0 - confidence) / denom))
                if it >= needed:
                    break

    if best_mask.sum() >= min_samples:
        s, R, t, rmse = umeyama(P[best_mask], Q[best_mask])
        pred  = (s * (R @ P.T)).T + t
        errs  = np.linalg.norm(Q - pred, axis=1)
        best_mask = errs < inlier_thr_m
        s, R, t, rmse = umeyama(P[best_mask], Q[best_mask])
    else:
        s, R, t, rmse = umeyama(P, Q)
        best_mask = np.ones(n, dtype=bool)

    return s, R, t, rmse, best_mask


# ─── Main solver ──────────────────────────────────────────────────────────────

def solve_ply_to_ecef(
    colmap_cameras: dict[str, np.ndarray],
    metashape_data: dict,
    ransac_inlier_thr_m: float = 5.0,
    points3d: np.ndarray | None = None,
) -> dict:
    """Compute the similarity transform that maps PLY positions to ECEF.

    Solves directly: C_colmap → GPS_ECEF (single step, no scene-space intermediate).

    Parameters
    ----------
    colmap_cameras  : {stem: C_colmap}  from colmap_reader.read_images_bin
    metashape_data  : result of metashape_parser.parse_metashape_xml
    ransac_inlier_thr_m : RANSAC inlier threshold in metres (ECEF space)
    points3d        : optional (N,3) sparse points from read_points3d_bin,
                      used only for altitude verification printout

    Returns a dict suitable for writing as similarity_transform.json.
    """
    xml_cams = metashape_data["cameras"]

    # Build (C_colmap, GPS_ECEF) pairs
    src_colmap: list[np.ndarray] = []
    dst_ecef:   list[np.ndarray] = []
    xml_by_name = {c["name"]: c for c in xml_cams}

    for stem, C_col in colmap_cameras.items():
        cam = xml_by_name.get(stem)
        if cam is None:
            continue
        src_colmap.append(C_col)
        # Use pre-computed ECEF if available (UTM input), else convert from geodetic
        if "ecef" in cam:
            dst_ecef.append(np.array(cam["ecef"]))
        else:
            dst_ecef.append(geodetic_to_ecef(cam["lat"], cam["lon"], cam["alt"]))

    n_matched = len(src_colmap)
    if n_matched < 4:
        raise ValueError(
            f"Only {n_matched} matched cameras between COLMAP images.bin "
            f"and Metashape XML (need ≥ 4). "
            "Check that image filenames match camera labels."
        )

    print(f"  Matched {n_matched} cameras (COLMAP ↔ Metashape XML)")

    # Solve
    s, R, t, rmse, inlier_mask = ransac_umeyama(
        src_colmap, dst_ecef,
        inlier_thr_m=ransac_inlier_thr_m,
    )
    n_inliers = int(inlier_mask.sum())
    print(f"  Solver: {n_inliers}/{n_matched} inliers, RMSE={rmse:.4f} m")

    # Report scene origin LLA (centroid of camera positions in ECEF)
    cam_center_colmap = np.array(src_colmap).mean(axis=0)
    ecef_center = s * (R @ cam_center_colmap) + t
    lat_c, lon_c, alt_c = ecef_to_geodetic(*ecef_center)
    print(f"  Camera centroid → lat={lat_c:.5f}, lon={lon_c:.5f}, alt={alt_c:.2f} m (drone altitude)")

    # Optional: verify sparse 3D points altitude
    if points3d is not None and len(points3d) > 0:
        sample = points3d[::max(1, len(points3d)//500)]  # ~500 pts
        ecef_pts = (s * (R @ sample.T).T) + t
        alts = [ecef_to_geodetic(*e)[2] for e in ecef_pts]
        print(f"  Sparse points → terrain alt: {min(alts):.1f}–{max(alts):.1f} m "
              f"(mean={float(np.mean(alts)):.1f} m)")

    return {
        "scale":       float(s),
        "rotation":    R.tolist(),
        "translation": t.tolist(),
        "rmse_m":      float(rmse),
        "n_inliers":   n_inliers,
        "n_total":     n_matched,
        "centroid_lat": lat_c,
        "centroid_lon": lon_c,
        "centroid_alt_ellipsoidal_m": alt_c,
    }
