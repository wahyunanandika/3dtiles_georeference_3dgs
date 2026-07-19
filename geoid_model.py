"""Geoid undulation lookup (EGM96 / EGM2008) with automatic grid download.

Converts WGS84 ellipsoidal height (what GPS/PPK and this pipeline use
natively) to orthometric height (height above the geoid / roughly MSL),
and back.

    orthometric_height = ellipsoidal_height - N(lat, lon)

Grid files are the standard GeographicLib EGM96/EGM2008 ``.pgm`` datasets.
They are downloaded once (on first use) from the official GeographicLib
SourceForge distribution and cached locally — no manual setup required.

Usage
-----
    from geoid_model import get_geoid_undulation, ellipsoidal_to_orthometric

    N = get_geoid_undulation(-6.87067, 107.55432, model="egm2008-2_5")
    ortho = ellipsoidal_to_orthometric(917.19, -6.87067, 107.55432,
                                        model="egm2008-2_5")

CLI
---
    python geoid_model.py --lat -6.87067 --lon 107.55432 --model egm2008-2_5
    python geoid_model.py --lat -6.87067 --lon 107.55432 --alt 917.19 --model egm96-5

Supported models: egm96-5, egm2008-5, egm2008-2_5, egm2008-1
(InaGeoid support is not implemented yet — BIG distributes it as a
login-gated point-query web service, not a downloadable grid, so it needs
a separate integration path. See ``_query_inageoid`` below.)
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

# ─── Grid catalogue ────────────────────────────────────────────────────────
# Official GeographicLib geoid distribution (same files used by the
# `geographiclib-get-geoids` helper script). See:
# https://geographiclib.sourceforge.io/C++/doc/geoid.html

_BASE_URL = "https://sourceforge.net/projects/geographiclib/files/geoids-distrib"

GEOID_MODELS = {
    "egm96-5":      {"pgm": "egm96-5.pgm",      "size_mb": 19,  "res": "5 arcmin"},
    "egm2008-5":    {"pgm": "egm2008-5.pgm",    "size_mb": 19,  "res": "5 arcmin"},
    "egm2008-2_5":  {"pgm": "egm2008-2_5.pgm",  "size_mb": 75,  "res": "2.5 arcmin"},
    "egm2008-1":    {"pgm": "egm2008-1.pgm",    "size_mb": 470, "res": "1 arcmin"},
}

DEFAULT_MODEL = "egm2008-2_5"

# Where downloaded grids are cached. Override with GEOID_DATA_DIR env var
# if you want to point at a shared/pre-populated directory instead.
import os
GEOID_DIR = Path(os.environ.get("GEOID_DATA_DIR", Path(__file__).parent / "geoids"))

_geoid_instance_cache: dict[str, object] = {}


class GeoidModelError(Exception):
    pass


def _download_and_extract(model: str) -> Path:
    if model not in GEOID_MODELS:
        raise GeoidModelError(
            f"Unknown geoid model '{model}'. Choices: {', '.join(GEOID_MODELS)}"
        )
    info = GEOID_MODELS[model]
    GEOID_DIR.mkdir(parents=True, exist_ok=True)
    pgm_path = GEOID_DIR / info["pgm"]
    if pgm_path.exists():
        return pgm_path

    url = f"{_BASE_URL}/{model}.tar.bz2/download"
    archive_path = GEOID_DIR / f"{model}.tar.bz2"
    print(f"[geoid_model] Downloading {model} ({info['size_mb']} MB, {info['res']}) …")
    print(f"[geoid_model]   {url}")
    try:
        # SourceForge returns 403 Forbidden for requests without a browser-like
        # User-Agent (blocks the default urllib/Python UA as anti-bot measure).
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(archive_path, "wb") as out:
            out.write(resp.read())
    except Exception as exc:
        raise GeoidModelError(
            f"Failed to download {model} grid from {url}: {exc}\n"
            f"If your network blocks SourceForge, download manually and place "
            f"'{info['pgm']}' in: {GEOID_DIR}"
        ) from exc

    print(f"[geoid_model] Extracting {archive_path.name} …")
    try:
        with tarfile.open(archive_path, "r:bz2") as tf:
            for member in tf.getmembers():
                if member.name.endswith(".pgm"):
                    member.name = Path(member.name).name  # flatten path
                    tf.extract(member, GEOID_DIR)
    finally:
        archive_path.unlink(missing_ok=True)

    if not pgm_path.exists():
        raise GeoidModelError(
            f"Extraction of {model} did not produce expected file {pgm_path}"
        )
    print(f"[geoid_model] Cached at {pgm_path}")
    return pgm_path


def _get_interpolator(model: str):
    if model in _geoid_instance_cache:
        return _geoid_instance_cache[model]

    try:
        from pygeodesy.geoids import GeoidKarney
    except ImportError as exc:
        raise GeoidModelError(
            "pygeodesy is required for geoid lookups. Install with:\n"
            "  pip install pygeodesy"
        ) from exc

    pgm_path = _download_and_extract(model)
    interpolator = GeoidKarney(str(pgm_path))
    _geoid_instance_cache[model] = interpolator
    return interpolator


def get_geoid_undulation(lat: float, lon: float, model: str = DEFAULT_MODEL) -> float:
    """Return geoid undulation N (metres) at (lat, lon) for the given model.

    N is the height of the geoid above the WGS84 ellipsoid, so:
        orthometric_height = ellipsoidal_height - N
    """
    interpolator = _get_interpolator(model)
    return float(interpolator.height(lat, lon))


def ellipsoidal_to_orthometric(
    ellipsoidal_height: float, lat: float, lon: float, model: str = DEFAULT_MODEL
) -> float:
    """Ellipsoidal (WGS84/GPS) height → orthometric (~MSL) height."""
    return ellipsoidal_height - get_geoid_undulation(lat, lon, model=model)


def orthometric_to_ellipsoidal(
    orthometric_height: float, lat: float, lon: float, model: str = DEFAULT_MODEL
) -> float:
    """Orthometric (~MSL) height → ellipsoidal (WGS84/GPS) height."""
    return orthometric_height + get_geoid_undulation(lat, lon, model=model)


def _query_inageoid(lat: float, lon: float):
    """Placeholder for InaGeoid (BIG) support.

    InaGeoid is not distributed as a downloadable grid — BIG only provides
    it through a login-gated point-query service at
    https://srgi.big.go.id/map/geoid-active (Single/Multiple Coordinates,
    or Select By Area), and their terms of use prohibit obtaining the data
    any other way. To add support: export undulation values for your
    area of interest from that page (Multiple Coordinates / Select By Area),
    then wire the exported file into an interpolator here (e.g.
    scipy.interpolate.griddata over the exported points).
    """
    raise NotImplementedError(
        "InaGeoid is not wired up yet — see _query_inageoid docstring."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Look up geoid undulation / convert ellipsoidal <-> orthometric height.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--lat", required=True, type=float)
    ap.add_argument("--lon", required=True, type=float)
    ap.add_argument("--alt", default=None, type=float,
                     help="Ellipsoidal height (m) to also convert to orthometric")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(GEOID_MODELS),
                     help="Geoid model to use")
    args = ap.parse_args()

    try:
        N = get_geoid_undulation(args.lat, args.lon, model=args.model)
    except GeoidModelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"model               : {args.model}")
    print(f"lat, lon            : {args.lat:.6f}, {args.lon:.6f}")
    print(f"undulation N        : {N:+.3f} m")
    if args.alt is not None:
        ortho = args.alt - N
        print(f"ellipsoidal height  : {args.alt:.3f} m")
        print(f"orthometric height  : {ortho:.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
