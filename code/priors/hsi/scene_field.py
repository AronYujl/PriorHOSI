"""Scene geometry for one LINGO scene: a mesh-derived signed distance field.

Why this module exists, and why the occupancy grid is *not* its primary source
====================================================================
`docs/plan/PHASE_1C_HSI.md`, section 5 of the 2026-08-12 entry, originally made
`Scene/<scene>.npy` the scoring geometry for human-scene penetration.  The
same-day revision (section A) reverses that, and this module implements the
correction:

* scene ``004``'s occupancy grid is **0.5119** occupied, and its most occupied
  height slice is the *ceiling* at y ~ 1.98 m (0.807).  Solid furniture cannot
  look like that;
* the LINGO release describes the file as "occupied by scene objects **or
  unreachable**".

`Scene/<scene>.npy` is therefore a *reachability / free-space* volume, not scene
geometry.  Scoring penetration against it gives GT a non-zero reference (~7.1 %
of GT joints sit in "occupied" cells) and rewards a model that floats far away
from every surface.

So:

===========================================  ==========================================
quantity                                     source
===========================================  ==========================================
penetration / contact (PRIMARY)              ``Scene_mesh/<scene>/mesh_low.obj``
model scene conditioning (unchanged)         ``Scene/<scene>.npy`` occupancy
reachability violation (SECONDARY diagnostic) ``Scene/<scene>.npy`` occupancy
===========================================  ==========================================

:meth:`SceneGeometry.signed_distance` is the primary quantity.
:meth:`SceneGeometry.reachability_violation` is the secondary one and is
explicitly **not** penetration.

Sign convention
===============
``signed_distance`` is **negative inside scene geometry** and positive in free
space, in metres, and its magnitude is the Euclidean distance to the mesh
surface.

Deriving "inside scene geometry" from a LINGO room scan needs one measured fact.
`mesh_low.obj` is a closed scan of the *room*: a single shell whose interior is
the walkable air and whose normals point into that air (measured on 004:
``trimesh`` signed volume ``-47.52`` m^3 ~ the room's air volume, and 97.3 % of
the faces within 5 cm of y = 0 have ``n_y > 0``, i.e. the floor points up).  The
region enclosed by the shell is therefore *free space*, and scene geometry is
its complement: floor slab, walls, furniture, and everything outside the scanned
room.

Because that polarity is a property of the data rather than of the format, it is
re-derived per scene instead of assumed: the slab strictly below the world floor
plane y = 0 must come out solid (LINGO's floor is exactly y = 0, and no scene
mesh has geometry below it).  :attr:`SceneGeometry.build_info` records the
measured fractions, and a scene that does not separate cleanly raises
:class:`SceneGeometryError` rather than silently emitting an inverted field.

How the field is built
======================
A 2 cm isotropic grid over the mesh bounding box padded by 20 cm - the DeSeG
pitch cited in the plan - then trilinear sampling.  Two stages:

1. **Exact narrow band.**  Every triangle scatters exact point-triangle
   distances into the cells of its bounding box dilated by one voxel, reduced
   with ``amin``.  Any cell whose result is <= 2 cm is exact, which covers the
   -3 cm penetration threshold and most of the +5 cm contact band.
2. **Closest-point transform for the far field.**  Each band cell keeps the
   exact closest surface point of its winning triangle; ``scipy``'s EDT feature
   transform gives every remaining cell its nearest band cell, and the distance
   is recomputed against that cell's stored surface point.  This is exact for a
   locally planar surface and degrades smoothly; ``tests/hsi/test_scene_field.py``
   bounds the residual against closed-form box and sphere fields.

The sign comes from an axis-aligned crossing (winding) count, which is exact for
a closed, consistently-wound mesh.  Non-watertight scenes fall back to a
majority vote over six axis rays - a discrete stand-in for the generalized
winding number, honest about being one - and are recorded in
``build_info["winding_fallback"]`` so no scene ever gets a silently wrong sign.

Out-of-bounds
=============
A query outside the SDF grid is *not* clamped into it and is *not* scored as
penetration: :meth:`signed_distance` returns a strictly positive distance and
:meth:`out_of_bounds` flags it, so an evaluation reports the escaped fraction
instead of absorbing it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "SceneGeometry",
    "SceneGeometryError",
    "OCC_GRID_MIN",
    "OCC_GRID_MAX",
    "OCC_GRID_SHAPE",
    "OCC_VOXEL_SIZE",
    "SDF_VOXEL_SIZE",
    "SDF_PAD",
    "SDF_CACHE_PROTOCOL_ID",
    "default_cache_dir",
    "sdf_cache_protocol_identity",
]

# ---------------------------------------------------------------------------
# Published LINGO occupancy-grid geometry (`lingo/code/datasets/lingo.py:97`,
# training branch).  Asserted against the array at load time, never trusted.
OCC_GRID_MIN: Tuple[float, float, float] = (-3.0, 0.0, -4.0)
OCC_GRID_MAX: Tuple[float, float, float] = (3.0, 2.0, 4.0)
OCC_GRID_SHAPE: Tuple[int, int, int] = (300, 100, 400)
OCC_VOXEL_SIZE: float = 0.02

# SDF grid.  2 cm is the DeSeG-on-LINGO pitch named in the plan; the pad is what
# lets a joint that has left the room still report a finite penetration depth
# before it becomes out-of-bounds.
SDF_VOXEL_SIZE: float = 0.02
SDF_PAD: float = 0.20
# Cells beyond the triangle's own bounding box that receive an exact distance.
# 1 voxel => every result <= 2 cm is exact; the cost is ~27 cell-triangle pairs
# per triangle, and raising it to 2 quadruples the build.
SDF_EXACT_BAND: int = 1
# Bumped whenever the numerics change, so a stale cache is never read back.
BUILD_VERSION: int = 1

# This identity is part of the P16-GQ treatment contract.  The filename carries
# the short form, while the JSON metadata carries the complete form so a cache
# copied under a convincing name cannot silently change the field protocol.
SDF_CACHE_PROTOCOL_ID = "mesh_low_sdf_v1"


def sdf_cache_protocol_identity() -> Dict[str, object]:
    """Return the immutable on-disk mesh/SDF cache protocol identity."""
    return {
        "id": SDF_CACHE_PROTOCOL_ID,
        "mesh_filename": "mesh_low.obj",
        "voxel_size_m": SDF_VOXEL_SIZE,
        "padding_m": SDF_PAD,
        "exact_band_voxels": SDF_EXACT_BAND,
        "build_version": BUILD_VERSION,
        "field_dtype": "<f4",
        "origin_dtype": "<f8",
        "filename_binding": (
            "scene__mesh_sha256_prefix16__h{voxel_mm}mm__p{pad_mm}mm__"
            "b{band}__v{build_version}.npz"
        ),
        "metadata_binding": "full_mesh_sha256_and_field_shape",
    }

# One 004-sized field is ~40 MB of float32.  This four-entry LRU bounds
# dataset-only accessors; the sealed evaluator separately retains one strong
# ``geometries`` entry for each of its roughly 19--20 scenes, so this constant
# must never be read as the formal evaluation residency bound.
DEFAULT_CACHE_SIZE: int = 4

_TRI_BITS = 24
_TRI_MASK = (1 << _TRI_BITS) - 1
_DIST_SCALE = 1.0e6  # 1 um key resolution
_DIST_CAP = 8.0  # metres; beyond this the CPT stage owns the cell anyway
_KEY_EMPTY = np.int64((1 << 62))
# Triangle-cell pairs materialised at once.  ~20 float32 temporaries of this
# length live at the peak of the point-triangle kernel, i.e. ~0.6 GB.
_PAIR_CHUNK = 2_000_000


def _cache_metadata(
    field: np.ndarray,
    origin: np.ndarray,
    mesh_sha256: str,
    *,
    scene_name: Optional[str] = None,
    voxel_size: float,
    pad: float,
    band: int,
) -> Dict[str, object]:
    """Return the protocol fields persisted beside a generated SDF field."""
    protocol = sdf_cache_protocol_identity()
    return {
        "cache_protocol_id": SDF_CACHE_PROTOCOL_ID,
        "mesh_filename": "mesh_low.obj",
        "mesh_sha256": str(mesh_sha256),
        "mesh_sha256_prefix": str(mesh_sha256)[:16],
        "scene_name": None if scene_name is None else str(scene_name),
        "voxel_size_m": float(voxel_size),
        "padding_m": float(pad),
        "exact_band_voxels": int(band),
        "build_version": int(BUILD_VERSION),
        "field_dtype": "<f4",
        "origin_dtype": "<f8",
        "field_shape": [int(value) for value in field.shape],
        "origin_shape": [int(value) for value in origin.shape],
        # Persist the protocol's filename and metadata bindings as well as the
        # concrete hash/shape values.  A valid-looking NPZ with an older
        # binding convention is not the P16-GQ cache protocol.
        "filename_binding": str(protocol["filename_binding"]),
        "metadata_binding": str(protocol["metadata_binding"]),
    }


def _cache_metadata_matches(
    field: np.ndarray,
    origin: np.ndarray,
    metadata: object,
    mesh_sha256: Optional[str],
    *,
    scene_name: Optional[str] = None,
    voxel_size: float,
    pad: float,
    band: int,
) -> bool:
    if field.ndim != 3 or field.dtype != np.dtype("<f4"):
        return False
    if not np.isfinite(field).all():
        return False
    if origin.shape != (3,) or origin.dtype != np.dtype("<f8"):
        return False
    if not np.isfinite(origin).all() or not isinstance(metadata, dict):
        return False
    expected = _cache_metadata(
        field,
        origin,
        "" if mesh_sha256 is None else mesh_sha256,
        scene_name=scene_name,
        voxel_size=voxel_size,
        pad=pad,
        band=band,
    )
    for key in (
        "cache_protocol_id",
        "mesh_filename",
        "voxel_size_m",
        "padding_m",
        "exact_band_voxels",
        "build_version",
        "field_dtype",
        "origin_dtype",
        "field_shape",
        "origin_shape",
        "filename_binding",
        "metadata_binding",
        "scene_name",
    ):
        if metadata.get(key) != expected[key]:
            return False
    if mesh_sha256 is not None and metadata.get("mesh_sha256") != mesh_sha256:
        return False
    if mesh_sha256 is not None and metadata.get("mesh_sha256_prefix") != mesh_sha256[:16]:
        return False
    return True


def _chunk_starts(cumulative: np.ndarray, n_items: int, budget: int) -> Sequence[int]:
    """Split ``[0, n_items)`` so each slice materialises at most ``budget`` pairs."""
    starts = [0]
    while starts[-1] < n_items:
        consumed = int(cumulative[starts[-1] - 1]) if starts[-1] else 0
        nxt = int(np.searchsorted(cumulative, consumed + budget, side="right"))
        starts.append(min(max(nxt, starts[-1] + 1), n_items))
    return starts


class SceneGeometryError(ValueError):
    """Scene data on disk violates the documented LINGO geometry contract."""


def default_cache_dir() -> Path:
    """Where prebuilt SDF grids live.  Git-ignored; override with a path or env.

    ``$INFBAGEL_SDF_CACHE`` wins; otherwise ``<repo>/.cache/hsi_sdf``, which the
    repository's ``*.npz`` rule already ignores, so a populated cache never makes
    ``tools/experiment.py`` see a dirty worktree.
    """
    env = os.environ.get("INFBAGEL_SDF_CACHE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / ".cache" / "hsi_sdf"


# ---------------------------------------------------------------------------
# Occupancy grid (secondary diagnostic)
# ---------------------------------------------------------------------------


def _validate_occupancy(scene_name: str, occupancy: np.ndarray) -> None:
    """Re-derive the documented occupancy geometry from the array itself.

    Deliberately not a bare ``assert``: evaluation may run under ``python -O``,
    and a silently mis-oriented grid would corrupt every reachability number.
    """
    if occupancy.dtype != np.bool_:
        raise SceneGeometryError(
            f"scene {scene_name!r}: occupancy dtype is {occupancy.dtype}, expected bool"
        )
    if occupancy.ndim != 3:
        raise SceneGeometryError(
            f"scene {scene_name!r}: occupancy has {occupancy.ndim} axes, expected 3"
        )
    shape = tuple(int(n) for n in occupancy.shape)
    if shape != OCC_GRID_SHAPE:
        raise SceneGeometryError(
            f"scene {scene_name!r}: occupancy shape {shape}, expected {OCC_GRID_SHAPE}"
        )
    # Axis/extent mapping.  The three world spans (6, 2, 8 m) are pairwise
    # distinct, so span/shape is isotropic for exactly one assignment of the axis
    # lengths to (x, y, z); any permutation makes at least two ratios differ.
    spans = tuple(hi - lo for hi, lo in zip(OCC_GRID_MAX, OCC_GRID_MIN))
    steps = tuple(span / n for span, n in zip(spans, shape))
    if not all(abs(step - OCC_VOXEL_SIZE) < 1e-9 for step in steps):
        raise SceneGeometryError(
            f"scene {scene_name!r}: derived voxel sizes {steps} are not isotropic "
            f"{OCC_VOXEL_SIZE}; the axis order is not (x, y, z)"
        )
    # y-up: the bbox is anchored at y = 0, so axis-1 index 0 is the floor slab.
    # It is the *reachability* floor, so it is near-solid in every published grid.
    floor = float(occupancy[:, 0, :].mean())
    if floor < 0.5:
        raise SceneGeometryError(
            f"scene {scene_name!r}: y=0 slab is only {floor:.3f} occupied; the grid "
            "does not look y-up with a floor at index 0"
        )


# ---------------------------------------------------------------------------
# Mesh numerics
# ---------------------------------------------------------------------------


def _closest_point_on_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Closest point of each triangle ``(a, b, c)`` to each query ``p``.

    Ericson's Voronoi-region test, vectorised: the seven regions are disjoint, so
    applying them as masked overwrites reproduces the early-return original.
    All inputs ``(N, 3)``; returns ``(N, 3)`` in the input dtype.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    inv = np.reciprocal(np.where(denom == 0, 1, denom))
    q = a + ab * (vb * inv)[:, None] + ac * (vc * inv)[:, None]

    def _apply(mask: np.ndarray, value: np.ndarray) -> None:
        np.copyto(q, value, where=mask[:, None])

    den = d1 - d3
    t = np.clip(d1 * np.reciprocal(np.where(den == 0, 1, den)), 0.0, 1.0)[:, None]
    _apply((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * t)

    den = d2 - d6
    t = np.clip(d2 * np.reciprocal(np.where(den == 0, 1, den)), 0.0, 1.0)[:, None]
    _apply((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * t)

    n1 = d4 - d3
    n2 = d5 - d6
    den = n1 + n2
    t = np.clip(n1 * np.reciprocal(np.where(den == 0, 1, den)), 0.0, 1.0)[:, None]
    _apply((va <= 0) & (n1 >= 0) & (n2 >= 0), b + (c - b) * t)

    _apply((d1 <= 0) & (d2 <= 0), a)
    _apply((d3 >= 0) & (d4 <= d3), b)
    _apply((d6 >= 0) & (d5 <= d6), c)
    return q


def _cell_centres(
    flat_index: np.ndarray, origin: np.ndarray, shape: Tuple[int, int, int], h: float
) -> np.ndarray:
    """World centres of the given flat (C-order) cell indices, ``(N, 3)`` float32."""
    ny, nz = shape[1], shape[2]
    iz = flat_index % nz
    rest = flat_index // nz
    iy = rest % ny
    ix = rest // ny
    idx = np.stack((ix, iy, iz), axis=-1).astype(np.float32)
    return idx * np.float32(h) + (origin + 0.5 * h).astype(np.float32)


def _unsigned_distance_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    origin: np.ndarray,
    shape: Tuple[int, int, int],
    h: float,
    band: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Unsigned distance from every cell centre to the mesh surface, in metres.

    Stage 1 is exact for every cell within ``band * h`` of the surface; stage 2
    is a closest-point transform seeded by stage 1's exact surface points.
    Returns the ``shape`` float32 grid plus build statistics.
    """
    from scipy import ndimage

    nx, ny, nz = shape
    ncell = nx * ny * nz
    tri = vertices[faces]  # (F, 3, 3)
    n_faces = int(tri.shape[0])
    if n_faces == 0:
        raise SceneGeometryError("mesh has no faces")
    if n_faces > _TRI_MASK:
        raise SceneGeometryError(
            f"mesh has {n_faces} faces, above the {_TRI_MASK} the distance key packs"
        )

    dims_arr = np.asarray(shape, dtype=np.int64)
    lo = np.floor((tri.min(axis=1) - origin) / h - 0.5).astype(np.int64) - band
    hi = np.ceil((tri.max(axis=1) - origin) / h - 0.5).astype(np.int64) + band
    lo = np.clip(lo, 0, dims_arr - 1)
    hi = np.clip(hi, 0, dims_arr - 1)
    box = hi - lo + 1
    counts = box.prod(axis=1)
    total_pairs = int(counts.sum())

    best = torch.full((ncell,), int(_KEY_EMPTY), dtype=torch.int64)
    cumulative = np.cumsum(counts)
    starts = _chunk_starts(cumulative, n_faces, _PAIR_CHUNK)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*scatter_reduce.*")
        for s, e in zip(starts[:-1], starts[1:]):
            cnt = counts[s:e]
            n_pair = int(cnt.sum())
            if n_pair == 0:
                continue
            trep = np.repeat(np.arange(s, e, dtype=np.int64), cnt)
            block = cumulative[s:e] - cnt
            offset = np.arange(n_pair, dtype=np.int64) - np.repeat(block - block[0], cnt)
            bz = np.repeat(box[s:e, 2], cnt)
            by = np.repeat(box[s:e, 1], cnt)
            oz = offset % bz
            rest = offset // bz
            oy = rest % by
            ox = rest // by
            ix = np.repeat(lo[s:e, 0], cnt) + ox
            iy = np.repeat(lo[s:e, 1], cnt) + oy
            iz = np.repeat(lo[s:e, 2], cnt) + oz
            flat = (ix * ny + iy) * nz + iz
            centres = (
                np.stack((ix, iy, iz), axis=-1).astype(np.float32) * np.float32(h)
                + (origin + 0.5 * h).astype(np.float32)
            )
            chunk = tri[trep]
            q = _closest_point_on_triangle(centres, chunk[:, 0], chunk[:, 1], chunk[:, 2])
            dist = np.linalg.norm(centres - q, axis=-1)
            key = np.minimum(dist, _DIST_CAP).astype(np.float64) * _DIST_SCALE
            key = key.astype(np.int64) * (1 << _TRI_BITS) + trep
            best.scatter_reduce_(
                0,
                torch.from_numpy(flat),
                torch.from_numpy(key),
                reduce="amin",
                include_self=True,
            )
            del trep, offset, ix, iy, iz, flat, centres, chunk, q, dist, key

    keys = best.numpy()
    del best
    seeded = keys < int(_KEY_EMPTY)
    n_band = int(seeded.sum())
    if n_band == 0:
        raise SceneGeometryError("no grid cell is adjacent to the mesh; bbox mismatch")

    band_idx = np.flatnonzero(seeded)
    band_tri = (keys[band_idx] & _TRI_MASK).astype(np.int64)
    band_p = _cell_centres(band_idx, origin, shape, h)
    band_chunk = tri[band_tri]
    band_q = _closest_point_on_triangle(
        band_p, band_chunk[:, 0], band_chunk[:, 1], band_chunk[:, 2]
    )
    del keys, band_tri, band_chunk

    # Closest-point transform: nearest seeded cell for every unseeded cell, then
    # measure against that cell's exact surface point rather than its centre.
    feature = ndimage.distance_transform_edt(
        ~seeded.reshape(shape), return_distances=False, return_indices=True
    )
    flat_feature = ((feature[0].astype(np.int64) * ny) + feature[1]) * nz + feature[2]
    del feature
    surface = np.zeros((ncell, 3), dtype=np.float32)
    surface[band_idx] = band_q
    del band_idx, band_p, band_q

    out = np.empty(ncell, dtype=np.float32)
    flat_feature = flat_feature.reshape(-1)
    step = max(1, ncell // 16)
    for beg in range(0, ncell, step):
        end = min(beg + step, ncell)
        idx = np.arange(beg, end, dtype=np.int64)
        centres = _cell_centres(idx, origin, shape, h)
        out[beg:end] = np.linalg.norm(centres - surface[flat_feature[beg:end]], axis=-1)
    stats = {
        "n_faces": float(n_faces),
        "n_cells": float(ncell),
        "n_band_cells": float(n_band),
        "n_pairs": float(total_pairs),
        "exact_radius_m": float(band * h),
    }
    return out.reshape(shape), stats


def _axis_crossing_counts(
    vertices: np.ndarray,
    faces: np.ndarray,
    origin: np.ndarray,
    shape: Tuple[int, int, int],
    h: float,
    axis: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Directed surface-crossing counts along +axis and -axis rays per cell.

    ``up[c]`` sums ``sign(n_axis)`` over crossings strictly above cell ``c``;
    ``down[c]`` sums ``-sign(n_axis)`` over crossings strictly below it.  For a
    closed, consistently wound mesh both equal the winding number of ``c``: +1
    inside for outward normals, -1 for inward ones, 0 outside.  They diverge
    exactly where the surface leaks, which is what the fallback vote exploits.
    """
    u, v = (axis + 1) % 3, (axis + 2) % 3  # right-handed, so the 2-D cross is n_axis
    nu, nv, na = shape[u], shape[v], shape[axis]
    ou, ov, oa = origin[u], origin[v], origin[axis]

    tri = vertices[faces]
    A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]
    e1 = B - A
    e2 = C - A
    cross = e1[:, u] * e2[:, v] - e1[:, v] * e2[:, u]
    live = np.flatnonzero(cross != 0)  # triangles parallel to the ray cannot cross it
    if live.size == 0:
        zero = np.zeros(shape, dtype=np.int32)
        return zero, zero.copy()

    A, e1, e2, cross = A[live], e1[live], e2[live], cross[live]
    pu_min = np.minimum(np.minimum(A[:, u], A[:, u] + e1[:, u]), A[:, u] + e2[:, u])
    pu_max = np.maximum(np.maximum(A[:, u], A[:, u] + e1[:, u]), A[:, u] + e2[:, u])
    pv_min = np.minimum(np.minimum(A[:, v], A[:, v] + e1[:, v]), A[:, v] + e2[:, v])
    pv_max = np.maximum(np.maximum(A[:, v], A[:, v] + e1[:, v]), A[:, v] + e2[:, v])

    iu0 = np.clip(np.ceil((pu_min - ou) / h - 0.5).astype(np.int64), 0, nu)
    iu1 = np.clip(np.floor((pu_max - ou) / h - 0.5).astype(np.int64), -1, nu - 1)
    iv0 = np.clip(np.ceil((pv_min - ov) / h - 0.5).astype(np.int64), 0, nv)
    iv1 = np.clip(np.floor((pv_max - ov) / h - 0.5).astype(np.int64), -1, nv - 1)
    bu = np.maximum(iu1 - iu0 + 1, 0)
    bv = np.maximum(iv1 - iv0 + 1, 0)
    counts = bu * bv
    keep = counts > 0
    if not keep.any():
        zero = np.zeros(shape, dtype=np.int32)
        return zero, zero.copy()
    A, e1, e2, cross = A[keep], e1[keep], e2[keep], cross[keep]
    iu0, iv0, bu, bv, counts = iu0[keep], iv0[keep], bu[keep], bv[keep], counts[keep]

    hist_cols: list = []
    hist_bins: list = []
    hist_sgns: list = []
    total_cols: list = []
    total_sgns: list = []
    cum = np.cumsum(counts)
    n_tri = int(counts.shape[0])
    starts = _chunk_starts(cum, n_tri, _PAIR_CHUNK)
    for s, e in zip(starts[:-1], starts[1:]):
        cnt = counts[s:e]
        n_pair = int(cnt.sum())
        if n_pair == 0:
            continue
        rep = np.repeat(np.arange(s, e, dtype=np.int64), cnt)
        block = cum[s:e] - cnt
        offset = np.arange(n_pair, dtype=np.int64) - np.repeat(block - block[0], cnt)
        bvr = np.repeat(bv[s:e], cnt)
        jv = offset % bvr
        ju = offset // bvr
        iu = np.repeat(iu0[s:e], cnt) + ju
        iv = np.repeat(iv0[s:e], cnt) + jv
        wu = (iu.astype(np.float64) + 0.5) * h + ou - A[rep, u]
        wv = (iv.astype(np.float64) + 0.5) * h + ov - A[rep, v]
        inv = np.reciprocal(cross[rep].astype(np.float64))
        l2 = (wu * e2[rep, v] - wv * e2[rep, u]) * inv
        l3 = (e1[rep, u] * wv - e1[rep, v] * wu) * inv
        inside = (l2 >= 0) & (l3 >= 0) & (l2 + l3 <= 1)
        if not inside.any():
            continue
        rep, iu, iv, l2, l3 = rep[inside], iu[inside], iv[inside], l2[inside], l3[inside]
        pos = A[rep, axis] + e1[rep, axis] * l2 + e2[rep, axis] * l3
        sgn = np.where(cross[rep] > 0, 1, -1).astype(np.int64)
        col = iu * nv + iv
        total_cols.append(col)
        total_sgns.append(sgn)
        # cnt_above[j] = sum of sgn over crossings with floor(t) >= j, where
        # t = (pos - origin)/h - 0.5 is the continuous cell-centre coordinate.
        bin_a = np.floor((pos - oa) / h - 0.5).astype(np.int64)
        ok = bin_a >= 0
        if ok.any():
            hist_cols.append(col[ok])
            hist_bins.append(np.minimum(bin_a[ok], na - 1))
            hist_sgns.append(sgn[ok])

    total = np.zeros(nu * nv, dtype=np.int64)
    if total_cols:
        total = np.bincount(
            np.concatenate(total_cols),
            weights=np.concatenate(total_sgns).astype(np.float64),
            minlength=nu * nv,
        ).astype(np.int64)
    hist = np.zeros(nu * nv * na, dtype=np.int64)
    if hist_cols:
        hist = np.bincount(
            np.concatenate(hist_cols) * na + np.concatenate(hist_bins),
            weights=np.concatenate(hist_sgns).astype(np.float64),
            minlength=nu * nv * na,
        ).astype(np.int64)

    hist = hist.reshape(nu, nv, na)
    up = np.cumsum(hist[:, :, ::-1], axis=2)[:, :, ::-1]
    down = up - total.reshape(nu, nv)[:, :, None]
    order = np.argsort([u, v, axis])
    up = np.ascontiguousarray(np.transpose(up, order)).astype(np.int32)
    down = np.ascontiguousarray(np.transpose(down, order)).astype(np.int32)
    return up, down


def _interior_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    origin: np.ndarray,
    shape: Tuple[int, int, int],
    h: float,
    *,
    watertight: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Cells enclosed by the mesh surface, plus the sign-quality statistics.

    Watertight scenes use one exact vertical crossing count.  Everything else
    votes over six axis rays: an approximation of the generalized winding number,
    not the solid-angle sum, and named that way on purpose.
    """
    up_y, down_y = _axis_crossing_counts(vertices, faces, origin, shape, h, 1)
    if watertight:
        disagreement = float(np.mean((up_y != 0) != (down_y != 0)))
        return (up_y != 0), {
            "winding_fallback": 0.0,
            "winding_ray_disagreement": disagreement,
        }
    votes = (up_y != 0).astype(np.uint8) + (down_y != 0).astype(np.uint8)
    del up_y, down_y
    for axis in (0, 2):
        up_a, down_a = _axis_crossing_counts(vertices, faces, origin, shape, h, axis)
        votes += (up_a != 0).astype(np.uint8) + (down_a != 0).astype(np.uint8)
        del up_a, down_a
    inside = votes >= 4
    ambiguous = float(np.mean((votes > 0) & (votes < 6)))
    return inside, {"winding_fallback": 1.0, "winding_ray_disagreement": ambiguous}


def _resolve_polarity(
    inside: np.ndarray, origin: np.ndarray, shape: Tuple[int, int, int], h: float
) -> Tuple[bool, Dict[str, float]]:
    """Decide whether the mesh-enclosed region is free space or scene geometry.

    The discriminator is LINGO's exact y = 0 floor: no scene mesh has geometry
    below it, so the sub-floor slab is unambiguously *scene geometry* and must
    come out solid whichever way round the shell is wound.  Returns
    ``(interior_is_free_space, measurements)``.
    """
    ny = shape[1]
    centres_y = origin[1] + (np.arange(ny) + 0.5) * h
    sub = np.flatnonzero((centres_y < -0.02) & (centres_y > origin[1]))
    if sub.size == 0:
        raise SceneGeometryError(
            "SDF grid does not extend below the world floor plane y = 0; cannot "
            "resolve the inside/outside polarity from data"
        )
    frac_sub = float(inside[:, sub, :].mean())
    stand = np.flatnonzero((centres_y >= 1.4) & (centres_y <= 1.8))
    frac_stand = float(inside[:, stand, :].mean()) if stand.size else float("nan")
    if frac_sub < 0.05:
        interior_is_free = True
    elif frac_sub > 0.95:
        interior_is_free = False
    else:
        raise SceneGeometryError(
            f"sub-floor slab is {frac_sub:.3f} enclosed by the mesh; neither "
            "polarity makes the region below y = 0 solid, so the sign would be a "
            "guess.  Pass interior_is_free_space explicitly after inspecting it."
        )
    return interior_is_free, {
        "subfloor_enclosed_fraction": frac_sub,
        "standing_slab_enclosed_fraction": frac_stand,
    }


def _mesh_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Read an OBJ and report its topology without letting a loader repair it.

    ``process=False`` keeps the published indexing, so watertightness is measured
    on the file as shipped rather than on a silently welded copy.
    """
    import trimesh

    mesh = trimesh.load_mesh(path, process=False)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    watertight = bool(mesh.is_watertight)
    winding_consistent = bool(mesh.is_winding_consistent)
    volume = float(mesh.volume)
    del mesh

    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, multiplicity = np.unique(edges, axis=0, return_counts=True)
    info = {
        "n_vertices": float(len(vertices)),
        "n_faces": float(len(faces)),
        "watertight": float(watertight),
        "winding_consistent": float(winding_consistent),
        "signed_volume": volume,
        "n_boundary_edges": float((multiplicity == 1).sum()),
        "n_nonmanifold_edges": float((multiplicity > 2).sum()),
    }
    tri = vertices[faces]
    area2 = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1
    )
    keep = area2 > 0
    info["n_degenerate_faces"] = float((~keep).sum())
    if not keep.all():
        faces = np.ascontiguousarray(faces[keep])
    return vertices, faces, info


def _build_field(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    voxel_size: float,
    pad: float,
    band: int,
    interior_is_free_space: Optional[bool],
    watertight: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Signed distance grid over the padded mesh bbox.  Negative inside geometry."""
    h = float(voxel_size)
    lo = vertices.min(axis=0).astype(np.float64) - pad
    hi = vertices.max(axis=0).astype(np.float64) + pad
    origin = np.floor(lo / h) * h
    shape = tuple(int(n) for n in np.maximum(np.ceil((hi - origin) / h), 1).astype(np.int64))

    unsigned, stats = _unsigned_distance_grid(vertices, faces, origin, shape, h, band)
    inside, sign_stats = _interior_mask(
        vertices, faces, origin, shape, h, watertight=watertight
    )
    stats.update(sign_stats)
    if interior_is_free_space is None:
        interior_is_free_space, polarity_stats = _resolve_polarity(inside, origin, shape, h)
        stats.update(polarity_stats)
    stats["interior_is_free_space"] = float(bool(interior_is_free_space))
    solid = ~inside if interior_is_free_space else inside
    stats["solid_fraction"] = float(solid.mean())
    field = np.where(solid, -unsigned, unsigned).astype(np.float32)
    return field, origin, stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SceneGeometry:
    """Scene geometry for one LINGO scene.

    Mesh-derived signed distance is the primary quantity; occupancy reachability
    is a separate, secondary diagnostic.  Instances are immutable and shareable
    between threads.
    """

    _cache: "OrderedDict[tuple, SceneGeometry]" = OrderedDict()
    _cache_lock = threading.Lock()
    _cache_size = DEFAULT_CACHE_SIZE
    _cache_hits = 0
    _cache_misses = 0

    def __init__(
        self,
        scene_name: str,
        field: np.ndarray,
        origin: np.ndarray,
        voxel_size: float,
        *,
        is_watertight: bool,
        occupancy: Optional[np.ndarray] = None,
        build_info: Optional[Dict[str, float]] = None,
    ) -> None:
        field_t = torch.as_tensor(np.ascontiguousarray(field, dtype=np.float32))
        if field_t.ndim != 3:
            raise ValueError(f"field must be 3-D, got shape {tuple(field_t.shape)}")
        self._scene_name = str(scene_name)
        self._field = field_t
        self._voxel_size = float(voxel_size)
        self._min = torch.tensor(np.asarray(origin, dtype=np.float64), dtype=torch.float32)
        self._grid_shape = torch.tensor(self._field.shape, dtype=torch.float32)
        self._max = self._min + self._grid_shape * self._voxel_size
        self._is_watertight = bool(is_watertight)
        self._build_info = dict(build_info or {})
        self._occupancy = (
            torch.as_tensor(np.ascontiguousarray(occupancy)) if occupancy is not None else None
        )
        self._occ_min = torch.tensor(OCC_GRID_MIN, dtype=torch.float32)
        self._occ_shape = torch.tensor(OCC_GRID_SHAPE, dtype=torch.long)
        # Per-(device, dtype) constant views so the hot path never rebuilds them
        # and never calls .item().
        self._views: Dict[Tuple[torch.device, torch.dtype], Dict[str, torch.Tensor]] = {}
        self._views_lock = threading.Lock()

    # -- construction -------------------------------------------------------

    @classmethod
    def from_scene(
        cls,
        scene_name: str,
        *,
        dataset_root: Path,
        mesh_root: Path,
        cache_dir: Optional[Path] = None,
        voxel_size: float = SDF_VOXEL_SIZE,
        pad: float = SDF_PAD,
        band: int = SDF_EXACT_BAND,
        use_cache: bool = True,
    ) -> "SceneGeometry":
        """Load one LINGO scene: SDF from the mesh, occupancy from the dataset.

        ``dataset_root`` holds ``Scene/<scene>.npy``; ``mesh_root`` holds
        ``<scene>/mesh_low.obj``.  A built field is memoised in a bounded LRU and,
        unless ``use_cache`` is off, persisted under :func:`default_cache_dir`
        keyed by the mesh SHA-256 and the resolution, so the ~1 min build is paid
        once per scene per machine.
        """
        dataset_root = Path(dataset_root)
        mesh_root = Path(mesh_root)
        cache_root = (
            Path(cache_dir) if cache_dir is not None else default_cache_dir()
        ).resolve()
        key = (
            str(mesh_root.resolve()),
            str(dataset_root.resolve()),
            str(cache_root),
            str(scene_name),
            float(voxel_size),
            float(pad),
            int(band),
        )
        with cls._cache_lock:
            hit = cls._cache.get(key)
            if hit is not None:
                cls._cache.move_to_end(key)
                cls._cache_hits += 1
                return hit
            cls._cache_misses += 1

        mesh_path = mesh_root / str(scene_name) / "mesh_low.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"no mesh for scene {scene_name!r} at {mesh_path}.  Mirrored scenes "
                "have no mesh by design and must not be scored."
            )
        occ_path = dataset_root / "Scene" / f"{scene_name}.npy"
        if not occ_path.exists():
            raise FileNotFoundError(f"no occupancy grid for scene {scene_name!r} at {occ_path}")
        occupancy = np.load(occ_path)
        _validate_occupancy(str(scene_name), occupancy)

        digest = _mesh_sha256(mesh_path)
        cache_path = cache_root / (
            f"{scene_name}__{digest[:16]}__h{round(voxel_size * 1000)}mm"
            f"__p{round(pad * 1000)}mm__b{band}__v{BUILD_VERSION}.npz"
        )
        payload = (
            cls._read_cache(
                cache_path,
                mesh_sha256=digest,
                scene_name=str(scene_name),
                voxel_size=voxel_size,
                pad=pad,
                band=band,
            )
            if use_cache
            else None
        )
        if payload is None:
            vertices, faces, mesh_info = _load_mesh(mesh_path)
            field, origin, stats = _build_field(
                vertices,
                faces,
                voxel_size=voxel_size,
                pad=pad,
                band=band,
                interior_is_free_space=None,
                watertight=bool(mesh_info["watertight"]),
            )
            stats.update(mesh_info)
            stats["mesh_sha256_prefix"] = digest[:16]
            stats.update(
                _cache_metadata(
                    field,
                    origin,
                    digest,
                    scene_name=str(scene_name),
                    voxel_size=voxel_size,
                    pad=pad,
                    band=band,
                )
            )
            payload = (field, origin, stats)
            if use_cache:
                cls._write_cache(cache_path, *payload)
        field, origin, stats = payload

        geometry = cls(
            str(scene_name),
            field,
            origin,
            voxel_size,
            is_watertight=bool(stats.get("watertight", 0.0)),
            occupancy=occupancy,
            build_info=stats,
        )
        with cls._cache_lock:
            existing = cls._cache.get(key)
            if existing is not None:
                cls._cache.move_to_end(key)
                return existing
            cls._cache[key] = geometry
            while len(cls._cache) > cls._cache_size:
                cls._cache.popitem(last=False)
        return geometry

    @classmethod
    def from_mesh(
        cls,
        vertices,
        faces,
        *,
        scene_name: str = "<synthetic>",
        voxel_size: float = SDF_VOXEL_SIZE,
        pad: float = SDF_PAD,
        band: int = SDF_EXACT_BAND,
        interior_is_free_space: Optional[bool] = None,
        watertight: Optional[bool] = None,
        occupancy: Optional[np.ndarray] = None,
    ) -> "SceneGeometry":
        """Build from an in-memory mesh.  Not cached; the entry point for tests.

        ``interior_is_free_space=False`` is the convention for an ordinary solid
        object (a box, a sphere); leaving it ``None`` re-derives LINGO's
        room-shell polarity from the sub-floor slab.
        """
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.int64)
        if watertight is None:
            edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
            _, multiplicity = np.unique(edges, axis=0, return_counts=True)
            watertight = bool((multiplicity == 2).all())
        field, origin, stats = _build_field(
            vertices,
            faces,
            voxel_size=voxel_size,
            pad=pad,
            band=band,
            interior_is_free_space=interior_is_free_space,
            watertight=bool(watertight),
        )
        stats["watertight"] = float(bool(watertight))
        return cls(
            scene_name,
            field,
            origin,
            voxel_size,
            is_watertight=bool(watertight),
            occupancy=occupancy,
            build_info=stats,
        )

    # -- disk cache ---------------------------------------------------------

    @staticmethod
    def _read_cache(
        path: Path,
        *,
        mesh_sha256: Optional[str] = None,
        scene_name: Optional[str] = None,
        voxel_size: float = SDF_VOXEL_SIZE,
        pad: float = SDF_PAD,
        band: int = SDF_EXACT_BAND,
    ):
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                field = np.asarray(data["field"])
                origin = np.asarray(data["origin"])
                stats = json.loads(str(data["meta"]))
        except Exception:  # a truncated or foreign file must not be fatal
            return None
        if not _cache_metadata_matches(
            field,
            origin,
            stats,
            mesh_sha256,
            scene_name=scene_name,
            voxel_size=voxel_size,
            pad=pad,
            band=band,
        ):
            return None
        return field, origin, stats

    @staticmethod
    def _write_cache(path: Path, field: np.ndarray, origin: np.ndarray, stats: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(
            tmp,
            field=field,
            origin=np.asarray(origin, dtype=np.float64),
            meta=np.array(json.dumps(stats)),
        )
        os.replace(tmp, path)

    # -- LRU ---------------------------------------------------------------

    @classmethod
    def configure_cache(cls, maxsize: int) -> None:
        """Resize the bounded in-memory LRU, in scenes.  Evicts immediately."""
        if maxsize < 1:
            raise ValueError("cache size must be at least 1")
        with cls._cache_lock:
            cls._cache_size = int(maxsize)
            while len(cls._cache) > cls._cache_size:
                cls._cache.popitem(last=False)

    @classmethod
    def cache_clear(cls) -> None:
        with cls._cache_lock:
            cls._cache.clear()
            cls._cache_hits = 0
            cls._cache_misses = 0

    @classmethod
    def cache_info(cls) -> Dict[str, int]:
        with cls._cache_lock:
            return {
                "hits": cls._cache_hits,
                "misses": cls._cache_misses,
                "size": len(cls._cache),
                "maxsize": cls._cache_size,
            }

    # -- properties ---------------------------------------------------------

    @property
    def voxel_size(self) -> float:
        return self._voxel_size

    @property
    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(min_xyz, max_xyz)`` of the SDF grid's world bbox, metres."""
        return self._min.clone(), self._max.clone()

    @property
    def is_watertight(self) -> bool:
        """Was the source mesh closed?  ``False`` means the sign used the
        six-ray fallback vote recorded in :attr:`build_info`."""
        return self._is_watertight

    @property
    def scene_name(self) -> str:
        return self._scene_name

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return tuple(int(n) for n in self._field.shape)

    @property
    def build_info(self) -> Dict[str, float]:
        """Provenance of the field: mesh topology, band size, polarity evidence."""
        return dict(self._build_info)

    @property
    def field(self) -> torch.Tensor:
        """Raw voxel-centre signed distances, metres, ``(nx, ny, nz)``."""
        return self._field

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"SceneGeometry(scene_name={self._scene_name!r}, grid={self.grid_shape}, "
            f"h={self._voxel_size}, watertight={self._is_watertight})"
        )

    # -- queries ------------------------------------------------------------

    def _view(self, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        key = (device, dtype)
        view = self._views.get(key)
        if view is not None:
            return view
        with self._views_lock:
            view = self._views.get(key)
            if view is None:
                view = {
                    "field": self._field.to(device=device, dtype=dtype),
                    "min": self._min.to(device=device, dtype=dtype),
                    "max": self._max.to(device=device, dtype=dtype),
                    "last": (self._grid_shape - 1).to(device=device, dtype=dtype),
                }
                self._views[key] = view
        return view

    @staticmethod
    def _query_dtype(points: torch.Tensor) -> torch.dtype:
        if points.dtype in (torch.float32, torch.float64):
            return points.dtype
        return torch.float32

    def signed_distance(self, points_world: torch.Tensor) -> torch.Tensor:
        """``[..., 3]`` metres, y-up world frame ``-> [...]`` signed distance, metres.

        NEGATIVE = inside scene geometry (penetrating the floor slab, a wall, or a
        piece of furniture); positive = free space.  Derived from
        ``mesh_low.obj``: trilinear sampling of a precomputed 2 cm grid, so the
        result is continuous and differentiable in ``points_world`` and usable as
        a Phase 4 guidance field.

        Never NaN for finite input.  A point outside the grid bbox is **not**
        clamped back in and **not** scored as penetration: it gets
        ``distance_to_bbox + max(nearest_in_bbox_value, 0)``, which is strictly
        positive, and :meth:`out_of_bounds` flags it so the escaped fraction is
        reported rather than absorbed.  Non-finite input propagates as NaN
        instead of raising, so a diverged rollout stays visible.
        """
        if points_world.shape[-1] != 3:
            raise ValueError(
                f"points_world must have trailing dim 3, got {tuple(points_world.shape)}"
            )
        dtype = self._query_dtype(points_world)
        view = self._view(points_world.device, dtype)
        field, lo, hi, last = view["field"], view["min"], view["max"], view["last"]

        leading = points_world.shape[:-1]
        p = points_world.to(dtype).reshape(-1, 3)

        # Continuous voxel-centre coordinates: the centre of cell i is at
        # lo + (i + 0.5) * h, so g = (p - lo)/h - 0.5 and g == i at that centre.
        g = (p - lo) / self._voxel_size - 0.5
        g = torch.clamp(g, torch.zeros_like(last), last)

        # floor() has zero gradient and the residual carries d/dp, so autograd
        # flows through the weights.  nan_to_num guards only the *index* tensors;
        # the weights keep any NaN so it reaches the output.
        base = torch.nan_to_num(g, nan=0.0).floor()
        i0 = base.to(torch.long)
        i1 = torch.minimum(i0 + 1, last.to(torch.long))
        w = g - base
        wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]

        x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
        x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
        c000 = field[x0, y0, z0]
        c001 = field[x0, y0, z1]
        c010 = field[x0, y1, z0]
        c011 = field[x0, y1, z1]
        c100 = field[x1, y0, z0]
        c101 = field[x1, y0, z1]
        c110 = field[x1, y1, z0]
        c111 = field[x1, y1, z1]

        c00 = c000 + (c100 - c000) * wx
        c01 = c001 + (c101 - c001) * wx
        c10 = c010 + (c110 - c010) * wx
        c11 = c011 + (c111 - c011) * wx
        c0 = c00 + (c10 - c00) * wy
        c1 = c01 + (c11 - c01) * wy
        interpolated = c0 + (c1 - c0) * wz

        # Exterior distance to the bbox: exactly zero in bounds, strictly positive
        # outside, so the documented positivity guarantee holds.
        overshoot = torch.clamp(lo - p, min=0.0) + torch.clamp(p - hi, min=0.0)
        outside = torch.linalg.vector_norm(overshoot, dim=-1)
        result = torch.where(
            outside > 0, outside + torch.clamp(interpolated, min=0.0), interpolated
        )
        return result.reshape(leading)

    def out_of_bounds(self, points_world: torch.Tensor) -> torch.Tensor:
        """``[...] `` bool - the point lies outside the SDF grid bbox.

        Report this fraction alongside every penetration number.  Out-of-bounds
        is *missing geometry*, not evidence of non-penetration: the LINGO scans
        do not cover the full occupancy bbox, and the released occupancy grid
        already places a measurable share of GT joints outside its own bbox.
        """
        if points_world.shape[-1] != 3:
            raise ValueError(
                f"points_world must have trailing dim 3, got {tuple(points_world.shape)}"
            )
        dtype = self._query_dtype(points_world)
        view = self._view(points_world.device, dtype)
        p = points_world.to(dtype)
        inside = ((p >= view["min"]) & (p <= view["max"])).all(dim=-1)
        return ~inside

    def reachability_violation(self, points_world: torch.Tensor) -> torch.Tensor:
        """``[...] `` bool - the point falls in an occupied cell of ``Scene/<scene>.npy``.

        **SECONDARY DIAGNOSTIC ONLY.  THIS IS NOT PENETRATION.**  The published
        grid marks a cell ``True`` when it is "occupied by scene objects *or
        unreachable*", so it is a reachability / free-space volume: scene 004 is
        0.5119 "occupied" and its most occupied height slice is the y ~ 1.98 m
        ceiling at 0.807.  About 7.1 % of ground-truth joints land in "occupied"
        cells, so the GT reference for this quantity is ~0.07 rather than 0, and
        a model that floats in the middle of the room scores better than GT.

        Use :meth:`signed_distance` for penetration.  This method exists because
        the same grid is the model's scene conditioning input, so a rollout that
        violates it is violating what it was actually shown - a different and
        weaker claim, and one that must be labelled as such in any table.

        Matches the released loader (`lingo/code/datasets/lingo.py:217`),
        including its rule that a point outside the occupancy bbox counts as
        occupied.
        """
        if self._occupancy is None:
            raise SceneGeometryError(
                f"scene {self._scene_name!r} was built without an occupancy grid"
            )
        if points_world.shape[-1] != 3:
            raise ValueError(
                f"points_world must have trailing dim 3, got {tuple(points_world.shape)}"
            )
        occ = self._occupancy
        if occ.device != points_world.device:
            occ = occ.to(points_world.device)
            self._occupancy = occ
        leading = points_world.shape[:-1]
        p = points_world.to(torch.float32).reshape(-1, 3)
        lo = self._occ_min.to(p.device)
        shape = self._occ_shape.to(p.device)
        voxel = torch.div(p - lo, OCC_VOXEL_SIZE).to(torch.long)
        in_bound = ((voxel >= 0) & (voxel < shape)).all(dim=-1)
        safe = torch.where(in_bound.unsqueeze(-1), voxel, torch.zeros_like(voxel))
        hit = occ[safe[:, 0], safe[:, 1], safe[:, 2]]
        hit = torch.where(in_bound, hit, torch.ones_like(hit))
        return hit.reshape(leading)
