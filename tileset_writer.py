# Part of 3dtiles_georeference_3dgs (GPL-3.0)
"""tileset.json writer — the only layer that knows about 3D Tiles versions.

Tree builders (flat octree, hierarchical LOD) produce a root tile dict whose
bounding volumes live in the scene (similarity input) frame. This module adds
the root transform (scene -> ECEF) and the version-specific document header.

Supported versions
------------------
1.1   3DTILES_content_gltf + KHR_gaussian_splatting(+_compression_spz_2).
2.0   Reserved. The OGC 3D Tiles 2.0 community standard is not ratified yet
      (Cesium, Sept 2026: "will be part of 3D Tiles 2.0 when ratified later
      this year"). Add the writer here once the spec text is final.
"""
from __future__ import annotations

import numpy as np

SUPPORTED_VERSIONS = ("1.1",)
PLANNED_VERSIONS = ("2.0",)

SPLAT_GLTF_EXTENSIONS = [
    "KHR_gaussian_splatting",
    "KHR_gaussian_splatting_compression_spz_2",
]


def similarity_matrix(sim: dict) -> np.ndarray:
    """4x4 scene -> ECEF matrix from {scale, rotation, translation}.

    No Y/Z flip: the similarity maps the COLMAP/Metashape scene frame directly
    to ECEF (the flip in dozeri's plugin is for the LichtFeld viewer frame).
    """
    s = float(sim["scale"])
    R = np.array(sim["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.array(sim["translation"], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def root_transform(sim: dict) -> list:
    """Column-major root tile transform (3D Tiles convention)."""
    return similarity_matrix(sim).T.flatten().tolist()


def box_from_bounds(pmin, pmax) -> list:
    """Axis-aligned 3D Tiles bounding box [center, 3 half-axes]."""
    c = ((np.asarray(pmin) + np.asarray(pmax)) * 0.5).tolist()
    h = ((np.asarray(pmax) - np.asarray(pmin)) * 0.5).tolist()
    return [
        c[0], c[1], c[2],
        h[0], 0.0,  0.0,
        0.0,  h[1], 0.0,
        0.0,  0.0,  h[2],
    ]


def tileset_document(root_tile: dict, geometric_error: float, version: str = "1.1") -> dict:
    """Wrap a root tile (transform already set) into a full tileset.json dict."""
    if version == "1.1":
        return {
            "asset": {"version": "1.1"},
            "extensionsUsed":     ["3DTILES_content_gltf"],
            "extensionsRequired": ["3DTILES_content_gltf"],
            "extensions": {"3DTILES_content_gltf": {
                "extensionsUsed":     list(SPLAT_GLTF_EXTENSIONS),
                "extensionsRequired": list(SPLAT_GLTF_EXTENSIONS),
            }},
            "geometricError": geometric_error,
            "root": root_tile,
        }
    if version in PLANNED_VERSIONS:
        raise NotImplementedError(
            f"3D Tiles {version} output is not implemented yet: the spec is not ratified. "
            f"Use one of {SUPPORTED_VERSIONS}."
        )
    raise ValueError(f"Unknown 3D Tiles version {version!r}; supported: {SUPPORTED_VERSIONS}")
