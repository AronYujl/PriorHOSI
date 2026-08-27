"""Exact SMPL-X body-proxy points for mesh-SDF guidance.

The formal LINGO path uses the frozen, male, zero-beta body contract.  The
small derived asset contains every table needed by the 512-point proxy, so the
runtime never opens the 104 MiB canonical SMPL-X NPZ.  Its provenance is pinned
to the canonical source hash in both the asset metadata and this module.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


NUM_BODY_JOINTS = 22
AREA512_INDEX_SHA256 = (
    "92b3f40e60837da06414f685c798764650fafdb0baec40134015bdd35968c468"
)
AREA512_INDEX_RAW_INT64_SHA256 = (
    "862ba310b98ab3b2aa6e12a7f5cd84025dcf61b22b8ee54d8bab4ebc916a09fd"
)
AREA512_COUNT = 512

BODY_PROXY_ASSET_SHA256 = (
    "b5065f93dc1c37acb2d02c1607ee02b4743b443ddcad13476a6eb36d98fcf3b5"
)
BODY_PROXY_ASSET_SIZE_BYTES = 1_352_698
SMPLX_SOURCE_SHA256 = (
    "ab318e3f37d2bfaae26abf4e6fab445c2a610e1d63714794d60379cc263bc2a5"
)
SMPLX_SOURCE_SIZE_BYTES = 108_753_445


@dataclass(frozen=True)
class ProxyTables:
    """CPU numpy tables used to construct a body proxy."""

    weights: np.ndarray
    offsets: np.ndarray
    posedirs: np.ndarray


_NUMPY_CACHE: Dict[str, ProxyTables] = {}
# The key has only one proxy, one supported gender, and the finite set of
# devices/dtypes used by a process.  Keeping the weighted offsets and identity
# here avoids repeating their construction on every guided step.
_TORCH_CACHE_MAXSIZE = 8
_TORCH_CACHE: Dict[
    Tuple[str, torch.device, torch.dtype],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _asset_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "body_proxy_area512.npz"


def _load_index() -> np.ndarray:
    path = Path(__file__).resolve().parent / "assets" / "idx_area512.npy"
    if not path.is_file():
        raise FileNotFoundError("P16-GQ area512 index is missing: %s" % path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != AREA512_INDEX_SHA256:
        raise ValueError("P16-GQ area512 index file hash does not match the frozen asset")
    index = np.load(path, allow_pickle=False)
    if index.shape != (AREA512_COUNT,) or index.dtype != np.dtype("<i8"):
        raise ValueError("P16-GQ area512 index must be a sorted int64 vector of length 512")
    if not np.all(index[:-1] < index[1:]):
        raise ValueError("P16-GQ area512 index must be sorted and unique")
    raw_hash = hashlib.sha256(np.asarray(index, dtype="<i8").tobytes()).hexdigest()
    if raw_hash != AREA512_INDEX_RAW_INT64_SHA256:
        raise ValueError("P16-GQ area512 index raw hash does not match the frozen asset")
    return index


def _metadata_from_npz(data) -> Dict:
    try:
        value = data["metadata"].item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        metadata = json.loads(str(value))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("P16-GQ body-proxy metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("P16-GQ body-proxy metadata must be an object")
    return metadata


def _validate_derived_asset(data, metadata: Dict, index: np.ndarray) -> ProxyTables:
    required = ("vertex_indices", "weights", "offsets", "posedirs")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("P16-GQ body-proxy asset is missing %s" % missing)

    vertex_indices = np.asarray(data["vertex_indices"])
    weights = np.asarray(data["weights"])
    offsets = np.asarray(data["offsets"])
    posedirs = np.asarray(data["posedirs"])
    if vertex_indices.shape != (AREA512_COUNT,) or vertex_indices.dtype != np.dtype("<i8"):
        raise ValueError("P16-GQ body-proxy vertex_indices shape/dtype is invalid")
    if not np.array_equal(vertex_indices, index):
        raise ValueError("P16-GQ body-proxy vertex_indices disagree with idx_area512.npy")
    if weights.shape != (AREA512_COUNT, NUM_BODY_JOINTS):
        raise ValueError("P16-GQ body-proxy weights shape is invalid")
    if offsets.shape != (AREA512_COUNT, NUM_BODY_JOINTS, 3):
        raise ValueError("P16-GQ body-proxy offsets shape is invalid")
    if posedirs.shape != (AREA512_COUNT, 3, 189):
        raise ValueError("P16-GQ body-proxy posedirs shape is invalid")
    for name, value in (("weights", weights), ("offsets", offsets), ("posedirs", posedirs)):
        if value.dtype != np.dtype("<f4"):
            raise ValueError("P16-GQ body-proxy %s must be little-endian float32" % name)
        if not np.isfinite(value).all():
            raise ValueError("P16-GQ body-proxy %s contains non-finite values" % name)

    expected_metadata = {
        "asset_name": "body_proxy_area512",
        "asset_schema": 1,
        "source_asset": "SMPLX_MALE.npz",
        "gender": "male",
        "num_source_joints": 55,
        "num_vertices": 10475,
        "num_body_joints": NUM_BODY_JOINTS,
        "num_pose_blend_coefficients": 189,
        "num_proxy_points": AREA512_COUNT,
        "source_sha256": SMPLX_SOURCE_SHA256,
        "source_size_bytes": SMPLX_SOURCE_SIZE_BYTES,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                "P16-GQ body-proxy metadata %s=%r, expected %r"
                % (key, metadata.get(key), expected)
            )
    if metadata.get("index_sha256") != AREA512_INDEX_SHA256:
        raise ValueError("P16-GQ body-proxy metadata index hash is invalid")
    if not str(metadata.get("derivation", "")).startswith(
        "Read source arrays as float64; reduce each source joint weight"
    ):
        raise ValueError("P16-GQ body-proxy derivation provenance is missing")
    provenance = metadata.get("generation_provenance")
    if not isinstance(provenance, dict) or provenance.get("source_path_policy") != (
        "authoritative source read-only; path intentionally not embedded"
    ):
        raise ValueError("P16-GQ body-proxy generation provenance is invalid")
    return ProxyTables(
        weights=np.array(weights, dtype="<f4", copy=True),
        offsets=np.array(offsets, dtype="<f4", copy=True),
        posedirs=np.array(posedirs, dtype="<f4", copy=True),
    )


def load_proxy_tables(proxy: str = "area512", gender: str = "male") -> ProxyTables:
    """Load and validate the tracked, frozen area512 proxy tables.

    ``gender`` remains an explicit argument to make accidental use of a
    non-frozen body fail loudly.  The old runtime derivation from a full male or
    female SMPL-X file is intentionally gone.
    """

    if proxy != "area512":
        raise ValueError("P16-GQ has only the frozen area512 proxy, got %r" % proxy)
    gender = str(gender).lower()
    if gender != "male":
        raise ValueError("P16-GQ frozen body proxy is male-only, got %r" % gender)
    cached = _NUMPY_CACHE.get(proxy)
    if cached is not None:
        return cached

    path = _asset_path()
    if not path.is_file():
        raise FileNotFoundError("P16-GQ body-proxy asset is missing: %s" % path)
    if path.stat().st_size != BODY_PROXY_ASSET_SIZE_BYTES:
        raise ValueError("P16-GQ body-proxy asset size does not match the frozen asset")
    if hashlib.sha256(path.read_bytes()).hexdigest() != BODY_PROXY_ASSET_SHA256:
        raise ValueError("P16-GQ body-proxy asset hash does not match the frozen asset")

    index = _load_index()
    with np.load(path, allow_pickle=False) as data:
        metadata = _metadata_from_npz(data)
        tables = _validate_derived_asset(data, metadata, index)
    _NUMPY_CACHE[proxy] = tables
    return tables


def _identity(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 1, 3, 3)


def _match_table(value: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if value.device == device and value.dtype == dtype:
        return value
    return value.to(device=device, dtype=dtype)


def _torch_tables(
    proxy: str, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (proxy, torch.device(device), dtype)
    cached = _TORCH_CACHE.get(key)
    if cached is not None:
        return cached
    tables = load_proxy_tables(proxy)
    weights = torch.as_tensor(tables.weights, device=device, dtype=dtype)
    offsets = torch.as_tensor(tables.offsets, device=device, dtype=dtype)
    posedirs = torch.as_tensor(tables.posedirs, device=device, dtype=dtype)
    weighted_offsets = weights[:, :, None] * offsets
    result = (weights, posedirs, weighted_offsets, _identity(device, dtype))
    if len(_TORCH_CACHE) >= _TORCH_CACHE_MAXSIZE:
        _TORCH_CACHE.pop(next(iter(_TORCH_CACHE)))
    _TORCH_CACHE[key] = result
    return result


def _proxy_points_prepared(
    joint_positions: torch.Tensor,
    global_rotations: torch.Tensor,
    local_rotations: Optional[torch.Tensor],
    weights: torch.Tensor,
    posedirs: torch.Tensor,
    weighted_offsets: torch.Tensor,
    eye: torch.Tensor,
) -> torch.Tensor:
    batch, steps = joint_positions.shape[:2]
    points = joint_positions[..., :NUM_BODY_JOINTS, :]
    rotations = _match_table(global_rotations, points.device, points.dtype)

    base = torch.einsum("nk,btkj->btnj", weights, points)
    base = base + torch.einsum("btkij,nkj->btni", rotations, weighted_offsets)
    if local_rotations is None:
        return base

    if local_rotations.ndim == 4:
        if local_rotations.shape[0] != batch * steps:
            raise ValueError("flattened local rotations do not match [B,T]")
        local = local_rotations.reshape(batch, steps, NUM_BODY_JOINTS, 3, 3)
    elif local_rotations.ndim == 5:
        local = local_rotations
    else:
        raise ValueError("local_rotations must have shape [B*T,22,3,3] or [B,T,22,3,3]")
    if local.shape != (batch, steps, NUM_BODY_JOINTS, 3, 3):
        raise ValueError("local rotations must have shape [B,T,22,3,3]")
    local = _match_table(local, points.device, points.dtype)
    theta = (local[:, :, 1:22] - eye).reshape(batch, steps, 189)
    blend_delta = torch.einsum("nij,btj->btni", posedirs, theta)

    # Algebraically apply each joint rotation to the [B,T,N,3] blend delta,
    # then contract the joint axis with W.  This avoids the old
    # [B,T,N,22,3,3] intermediate while retaining the same sums and gradients.
    rotated_blend = torch.einsum("btkij,btnj->btkni", rotations, blend_delta)
    return base + torch.einsum("nk,btkni->btni", weights, rotated_blend)


def proxy_points_from_tables(
    joint_positions: torch.Tensor,
    global_rotations: torch.Tensor,
    local_rotations: Optional[torch.Tensor],
    weights: torch.Tensor,
    offsets: torch.Tensor,
    posedirs: torch.Tensor,
) -> torch.Tensor:
    """Apply exact reduced SMPL-X LBS and pose blend shapes.

    ``joint_positions`` is ``[B,T,22 or 24,3]`` and rotations are
    ``[B,T,22,3,3]``.  The returned tensor is ``[B,T,N,3]``.  The table argument
    form is public so arithmetic can be tested without a machine-local source
    SMPL-X file.
    """

    if joint_positions.ndim != 4 or joint_positions.shape[-1] != 3:
        raise ValueError("joint_positions must have shape [B,T,J,3]")
    if global_rotations.ndim != 5 or global_rotations.shape[-2:] != (3, 3):
        raise ValueError("global_rotations must have shape [B,T,22,3,3]")
    batch, steps = joint_positions.shape[:2]
    if global_rotations.shape[:2] != (batch, steps) or global_rotations.shape[2] != NUM_BODY_JOINTS:
        raise ValueError("joint and global-rotation batch shapes do not agree")
    if joint_positions.shape[2] < NUM_BODY_JOINTS:
        raise ValueError("joint_positions must contain the 22 body joints")
    if weights.ndim != 2 or weights.shape[1] != NUM_BODY_JOINTS:
        raise ValueError("weights must have shape [N,22]")
    if offsets.shape != (weights.shape[0], NUM_BODY_JOINTS, 3):
        raise ValueError("offsets must have shape [N,22,3]")
    if posedirs.shape != (weights.shape[0], 3, 189):
        raise ValueError("posedirs must have shape [N,3,189]")

    dtype = joint_positions.dtype
    device = joint_positions.device
    weights = _match_table(weights, device, dtype)
    offsets = _match_table(offsets, device, dtype)
    posedirs = _match_table(posedirs, device, dtype)
    weighted_offsets = weights[:, :, None] * offsets
    return _proxy_points_prepared(
        joint_positions,
        global_rotations,
        local_rotations,
        weights,
        posedirs,
        weighted_offsets,
        _identity(device, dtype),
    )


def proxy_points(
    joint_positions: torch.Tensor,
    global_rotations: torch.Tensor,
    local_rotations: torch.Tensor,
    proxy: str = "area512",
) -> torch.Tensor:
    """Build the frozen area512 proxy points for a sampler batch."""

    weights, posedirs, weighted_offsets, eye = _torch_tables(
        proxy, joint_positions.device, joint_positions.dtype
    )
    return _proxy_points_prepared(
        joint_positions,
        global_rotations,
        local_rotations,
        weights,
        posedirs,
        weighted_offsets,
        eye,
    )


__all__ = [
    "AREA512_COUNT",
    "AREA512_INDEX_RAW_INT64_SHA256",
    "AREA512_INDEX_SHA256",
    "BODY_PROXY_ASSET_SHA256",
    "BODY_PROXY_ASSET_SIZE_BYTES",
    "ProxyTables",
    "SMPLX_SOURCE_SHA256",
    "SMPLX_SOURCE_SIZE_BYTES",
    "load_proxy_tables",
    "proxy_points",
    "proxy_points_from_tables",
]
