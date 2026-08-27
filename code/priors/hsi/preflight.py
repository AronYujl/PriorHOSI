"""CPU-only, checkout-local gates for the formal P16-GQ sampler path."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from priors.hsi.body_proxy import (
    BODY_PROXY_ASSET_SHA256,
    BODY_PROXY_ASSET_SIZE_BYTES,
    SMPLX_SOURCE_SHA256,
    load_proxy_tables,
    proxy_points,
)
from priors.hsi.scene_field import (
    BUILD_VERSION,
    SDF_EXACT_BAND,
    SDF_PAD,
    SDF_VOXEL_SIZE,
    SDF_CACHE_PROTOCOL_ID,
    SceneGeometry,
    sdf_cache_protocol_identity,
)


SEALED_CHECKPOINT_SHA256 = (
    "5daaf813ca82878868602840760f35df43b642d73f73cb37e24bb5a4dbf62b4c"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_under(path: Path, root: Path, label: str) -> Path:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            "%s must be checkout-local under %s, got %s" % (label, root, path)
        ) from error
    if str(relative) in ("", "."):
        raise RuntimeError("%s must not be the checkout root: %s" % (label, path))
    return path


def _reject_parent_checkout(path: Path, repo_root: Path, label: str) -> Path:
    """Permit immutable external data, but never the sibling parent checkout."""
    path = Path(path).resolve()
    parent_checkout = Path(repo_root).resolve().parent / "InfBaGel-hsi"
    try:
        path.relative_to(parent_checkout)
    except ValueError:
        return path
    raise RuntimeError(
        "%s must not resolve into the parent checkout %s: %s"
        % (label, parent_checkout, path)
    )


def _checkpoint_sha256(checkpoint_path: Path, expected: str) -> str:
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("sealed P16-GQ checkpoint is missing: %s" % checkpoint_path)
    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise RuntimeError(
            "sealed P16-GQ checkpoint SHA-256 mismatch: expected %s, got %s"
            % (expected, actual)
        )
    return actual


def sealed_checkpoint_sha256(
    checkpoint_path: Path, expected: str = SEALED_CHECKPOINT_SHA256
) -> str:
    """Hash and enforce the immutable checkpoint identity before sampling."""
    expected = str(expected).lower()
    if expected != SEALED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "formal P16-GQ checkpoint expectation is not the sealed SHA-256"
        )
    return _checkpoint_sha256(checkpoint_path, expected)


def sdf_cache_path(
    cache_root: Path,
    scene_name: str,
    mesh_sha256: str,
    *,
    voxel_size: float = SDF_VOXEL_SIZE,
    pad: float = SDF_PAD,
    band: int = SDF_EXACT_BAND,
) -> Path:
    """Return the exact cache filename used by ``SceneGeometry.from_scene``."""
    return Path(cache_root) / (
        f"{scene_name}__{mesh_sha256[:16]}__h{round(voxel_size * 1000)}mm"
        f"__p{round(pad * 1000)}mm__b{int(band)}__v{BUILD_VERSION}.npz"
    )


def _read_validated_cache(
    path: Path, mesh_sha256: str, scene_name: Optional[str] = None
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("required SDF cache is missing: %s" % path)
    try:
        with np.load(path, allow_pickle=False) as data:
            field = np.asarray(data["field"])
            origin = np.asarray(data["origin"])
            metadata = json.loads(str(data["meta"]))
    except Exception as error:
        raise RuntimeError("required SDF cache is unreadable: %s" % path) from error
    if field.ndim != 3 or field.dtype != np.dtype("<f4"):
        raise RuntimeError(
            "SDF cache field must be finite little-endian float32 3-D: %s" % path
        )
    if not np.isfinite(field).all():
        raise RuntimeError("SDF cache field contains non-finite values: %s" % path)
    if origin.shape != (3,) or origin.dtype != np.dtype("<f8") or not np.isfinite(origin).all():
        raise RuntimeError("SDF cache origin is invalid: %s" % path)
    if not isinstance(metadata, dict):
        raise RuntimeError("SDF cache metadata is not an object: %s" % path)
    expected = sdf_cache_protocol_identity()
    checks = {
        "cache_protocol_id": SDF_CACHE_PROTOCOL_ID,
        "mesh_filename": expected["mesh_filename"],
        "mesh_sha256": mesh_sha256,
        "mesh_sha256_prefix": mesh_sha256[:16],
        "voxel_size_m": expected["voxel_size_m"],
        "padding_m": expected["padding_m"],
        "exact_band_voxels": expected["exact_band_voxels"],
        "build_version": expected["build_version"],
        "field_dtype": expected["field_dtype"],
        "origin_dtype": expected["origin_dtype"],
        "field_shape": [int(value) for value in field.shape],
        "origin_shape": [3],
        "filename_binding": expected["filename_binding"],
        "metadata_binding": expected["metadata_binding"],
    }
    for key, value in checks.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                "SDF cache %s mismatch: expected %r, got %r: %s"
                % (key, value, metadata.get(key), path)
            )
    if scene_name is not None and metadata.get("scene_name") != str(scene_name):
        raise RuntimeError(
            "SDF cache scene_name mismatch: expected %r, got %r: %s"
            % (str(scene_name), metadata.get("scene_name"), path)
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "field_shape": list(field.shape),
        "origin": origin.tolist(),
        "mesh_sha256": mesh_sha256,
        "metadata": metadata,
        "protocol_id": SDF_CACHE_PROTOCOL_ID,
    }


def run_formal_preflight(
    *,
    repo_root: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    mesh_root: Path,
    scene_names: Iterable[str],
    expected_checkpoint_sha256: str = SEALED_CHECKPOINT_SHA256,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Exercise real proxy/SDF dependencies and require complete local caches.

    This function does no model sampling and never builds a missing SDF.  The
    expected checkpoint value is checked against the sealed module constant
    before the file is hashed.
    """
    repo_root = Path(repo_root).resolve()
    checkpoint_path = _require_under(Path(checkpoint_path), repo_root, "checkpoint")
    dataset_root = _reject_parent_checkout(Path(dataset_root), repo_root, "dataset root")
    mesh_root = _reject_parent_checkout(Path(mesh_root), repo_root, "mesh root")
    env_cache = os.environ.get("INFBAGEL_SDF_CACHE")
    if not env_cache:
        raise RuntimeError("formal P16-GQ requires checkout-local INFBAGEL_SDF_CACHE")
    cache_root = _require_under(
        Path(env_cache if cache_dir is None else cache_dir), repo_root, "SDF cache"
    )
    if cache_dir is not None and cache_root != Path(env_cache).resolve():
        raise RuntimeError(
            "explicit SDF cache differs from INFBAGEL_SDF_CACHE: %s != %s"
            % (cache_root, Path(env_cache).resolve())
        )
    if not cache_root.is_dir():
        raise FileNotFoundError("formal SDF cache directory is missing: %s" % cache_root)

    checkpoint_digest = sealed_checkpoint_sha256(
        checkpoint_path, expected=str(expected_checkpoint_sha256).lower()
    )

    tables = load_proxy_tables("area512")
    identity = torch.eye(3, dtype=torch.float32).reshape(1, 1, 1, 3, 3)
    proxy = proxy_points(
        torch.zeros(1, 1, 22, 3, dtype=torch.float32),
        identity.repeat(1, 1, 22, 1, 1),
        identity.repeat(1, 1, 22, 1, 1).reshape(1, 22, 3, 3),
        proxy="area512",
    )
    if tuple(proxy.shape) != (1, 1, 512, 3) or not bool(torch.isfinite(proxy).all()):
        raise RuntimeError("real area512 proxy preflight returned an invalid tensor")

    names = tuple(sorted({str(name) for name in scene_names}))
    if not names:
        raise ValueError("formal P16-GQ preflight received no selected scenes")
    SceneGeometry.cache_clear()
    scene_records = []
    for scene_name in names:
        if not scene_name or Path(scene_name).name != scene_name:
            raise RuntimeError("formal scene name must be a single path component: %s" % scene_name)
        mesh_path = (mesh_root / scene_name / "mesh_low.obj").resolve()
        try:
            mesh_path.relative_to(mesh_root)
        except ValueError as error:
            raise RuntimeError(
                "formal scene mesh escaped the configured mesh root: %s" % mesh_path
            ) from error
        _reject_parent_checkout(mesh_path, repo_root, "scene mesh")
        if not mesh_path.is_file():
            raise FileNotFoundError("formal scene mesh is missing: %s" % mesh_path)
        mesh_digest = sha256_file(mesh_path)
        cache_path = sdf_cache_path(cache_root, scene_name, mesh_digest)
        cache_record = _read_validated_cache(cache_path, mesh_digest, scene_name)
        geometry = SceneGeometry.from_scene(
            scene_name,
            dataset_root=dataset_root,
            mesh_root=mesh_root,
            cache_dir=cache_root,
            use_cache=True,
        )
        lower, upper = geometry.bounds
        probe = ((lower + upper) / 2.0).reshape(1, 3)
        distance = geometry.signed_distance(probe)
        if not bool(torch.isfinite(distance).all()):
            raise RuntimeError("real SceneGeometry returned a non-finite probe: %s" % scene_name)
        cache_record["scene_name"] = scene_name
        cache_record["mesh_path"] = str(mesh_path)
        scene_records.append(cache_record)

    return {
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_digest,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "proxy": {
            "asset_sha256": BODY_PROXY_ASSET_SHA256,
            "asset_size_bytes": BODY_PROXY_ASSET_SIZE_BYTES,
            "source_sha256": SMPLX_SOURCE_SHA256,
            "weights_shape": list(tables.weights.shape),
            "offsets_shape": list(tables.offsets.shape),
            "posedirs_shape": list(tables.posedirs.shape),
        },
        "sdf_cache": {
            "root": str(cache_root),
            "protocol": sdf_cache_protocol_identity(),
            "build_version": BUILD_VERSION,
            "voxel_size": SDF_VOXEL_SIZE,
            "pad": SDF_PAD,
            "band": SDF_EXACT_BAND,
            "scenes": scene_records,
        },
    }


__all__ = [
    "SEALED_CHECKPOINT_SHA256",
    "SDF_CACHE_PROTOCOL_ID",
    "run_formal_preflight",
    "sdf_cache_path",
    "sealed_checkpoint_sha256",
    "sha256_file",
]
