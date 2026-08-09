#!/usr/bin/env python3
"""Authority CPU hard-gate diagnostics for the fixed D2-AE0 mechanism."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

from datasets.utils import zup_to_yup  # noqa: E402
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import HOIPriorSampler  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    OBJECT_NAMES,
    ROLE_JOINTS,
    ROLE_SWAP_PERMUTATION,
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
    SparseCurrentStateRelationField,
    TEMPORAL_ANCHORS,
    TEMPORAL_SOURCE_PERMUTATION,
    build_sparse_relation_geometry,
)
from priors.window_codec import project_to_so3  # noqa: E402
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_BASE,
    HOI_ARCHITECTURE_D2AC,
    HOI_ARCHITECTURE_D2AD,
    HOI_ARCHITECTURE_D2AE,
    assert_parameter_independence,
    build_expert,
    load_trained_hoi_prior,
)
from train_hoi_prior import (  # noqa: E402
    _d2ae_gradient_audit,
    _locked_loss_weights,
    _optimization_contract,
    _validate_d2ae_contract,
)


RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ae-cpu-contract"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
FAILURE_CLASSIFICATION = "sparse-relation-field-contract-failure-stop"

EXPECTED_BASE_PARAMETERS = 29_673_448
EXPECTED_INCREMENT_PARAMETERS = 413_953
EXPECTED_TOTAL_PARAMETERS = 30_087_401
EXPECTED_PARAMETER_INCREASE_LIMIT = 0.015
EXPECTED_MAPPING_SHA256 = (
    "1af35119c1dd54e2ad44c99f3cb91b62c1b88f62ca80cddcc96f4b201ffe0f5b"
)
EXPECTED_MANIFEST_SHA256 = (
    "e88d74a7ee434f3e6320c95d1ebb74efdc8fe4740b70ff596e502666a096f7a7"
)
EXPECTED_STACKED_TENSOR_SHA256 = (
    "793dad6a805d0a908087b273590bf171e7bce4c026297cf94d40f8c651fe4cab"
)
EXPECTED_SPLIT_SHA256 = (
    "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e"
)
EXPECTED_AUTHORITY_PYTHON = Path(
    "/data/yujinlun/anaconda3/envs/infbagel/bin/python"
)

AUTHORITY_VERIFICATION_COMMANDS = (
    '"$INFBAGEL_PYTHON" -m unittest tests.test_hoi_d2ae -v',
    '"$INFBAGEL_PYTHON" -m unittest discover -s tests -v',
    '"$INFBAGEL_PYTHON" tools/experiment.py validate',
    "git diff --check",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def yaw(
    angle_degrees: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        (
            (cosine, 0.0, sine),
            (0.0, 1.0, 0.0),
            (-sine, 0.0, cosine),
        ),
        dtype=dtype,
        device=device,
    )


def synthetic_inputs(
    batch: int = 2,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    seed: int = 123,
) -> Dict[str, torch.Tensor]:
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    current = torch.randn(
        batch, 16, 232, generator=generator, dtype=dtype, device=device,
    )
    identity = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 9)
    current[..., 219:228] = identity
    return {
        "current": current,
        "timesteps": (
            torch.arange(batch, dtype=torch.long, device=device) * 249 % 500
        ),
        "text": torch.randn(
            batch, 768, generator=generator, dtype=dtype, device=device,
        ),
        "global_bps": torch.randn(
            batch, 1024, 3, generator=generator, dtype=dtype, device=device,
        ),
        "goals": torch.randn(
            batch, 9, generator=generator, dtype=dtype, device=device,
        ),
        "progress": torch.randn(
            batch, 3, generator=generator, dtype=dtype, device=device,
        ),
        "rest_object_points": torch.randn(
            batch, 100, 3, generator=generator, dtype=dtype, device=device,
        ),
        "world_to_local_rotation": torch.stack(
            [yaw(-17 + 13 * index, dtype=dtype, device=device) for index in range(batch)]
        ),
        "object_rotation_reference": torch.stack(
            [yaw(11 - 7 * index, dtype=dtype, device=device) for index in range(batch)]
        ),
        "position_minimum": torch.tensor(
            (-2.0, -1.0, -3.0), dtype=dtype, device=device,
        ),
        "position_maximum": torch.tensor(
            (2.0, 3.0, 4.0), dtype=dtype, device=device,
        ),
        "object_minimum": torch.tensor(
            (-1.5, -0.5, -2.0), dtype=dtype, device=device,
        ),
        "object_maximum": torch.tensor(
            (1.5, 2.5, 2.0), dtype=dtype, device=device,
        ),
    }


def forward_model(
    model: torch.nn.Module,
    values: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return model(
        values["current"],
        values["timesteps"],
        values["text"],
        values["global_bps"],
        values["goals"],
        values["progress"],
        rest_object_points=values["rest_object_points"],
        world_to_local_rotation=values["world_to_local_rotation"],
        object_rotation_reference=values["object_rotation_reference"],
        position_minimum=values["position_minimum"],
        position_maximum=values["position_maximum"],
        object_minimum=values["object_minimum"],
        object_maximum=values["object_maximum"],
    )


def capture_forward(
    model: torch.nn.Module,
    values: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, object]]:
    """Capture bounded runtime summaries plus an immediate pure-builder probe.

    The model snapshot intentionally never retains per-batch geometry.  CPU
    contracts reconstruct the exact ephemeral tensors through the shared pure
    PyTorch builder and the same field modules after the forward has completed.
    """
    network = model.network
    network.set_sparse_relation_capture(True)
    try:
        output = forward_model(model, values)
        snapshot = network.sparse_relation_snapshot()
        field = network.sparse_relation_field
        geometry = build_sparse_relation_geometry(
            values["current"],
            values["rest_object_points"],
            values["world_to_local_rotation"],
            values["object_rotation_reference"],
            values["position_minimum"],
            values["position_maximum"],
            values["object_minimum"],
            values["object_maximum"],
        )
        encoded = field.point_encoder(geometry["features"])
        pooled = torch.cat(
            (encoded.mean(dim=-2), encoded.amax(dim=-2)), dim=-1,
        )
        if field._diagnostic_variant == "temporal_correspondence_permuted":
            relation_input = pooled[:, TEMPORAL_SOURCE_PERMUTATION]
        elif field._diagnostic_variant == "left_right_role_swapped":
            relation_input = pooled[:, :, ROLE_SWAP_PERMUTATION]
        else:
            relation_input = pooled
        relation = field._relation_vectors(relation_input)
        routed = relation.index_select(1, field.routing_slots)
    finally:
        network.set_sparse_relation_capture(False)
    if not isinstance(snapshot, dict):
        raise TypeError("D2-AE sparse relation snapshot must be a dictionary")
    return output, {
        **snapshot,
        "surface": geometry["surface"],
        "role_point_features": geometry["features"],
        "pooled_blocks": pooled,
        "relation_vectors": relation,
        "routed_relation": routed,
    }


def merged_config(repo: Path, formal_run_id: str):
    base = OmegaConf.load(repo / "code/config/config_train_hoi_prior.yaml")
    d2ae = OmegaConf.load(repo / "code/config/config_train_hoi_prior_d2ae.yaml")
    cfg = OmegaConf.merge(base, d2ae)
    cfg.repo_root = str(repo)
    cfg.split_manifest = str(
        repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    cfg.run_id = formal_run_id
    cfg.output_dir = str(repo / "results/experiments" / formal_run_id)
    cfg.checkpoint_dir = str(Path(cfg.output_dir) / "checkpoints")
    cfg.metrics_path = str(Path(cfg.output_dir) / "metrics.json")
    cfg.state_path = str(Path(cfg.output_dir) / "training_state.json")
    OmegaConf.resolve(cfg)
    return cfg


def identity_contract(repo: Path, *, require_clean: bool) -> Dict[str, object]:
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=repo, text=True,
        ).strip()
    ).resolve()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True,
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        text=True,
    ).splitlines()
    if root != repo.resolve() or branch != "phase/01b-hoi":
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: authority identity mismatch")
    if require_clean and status:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: authority worktree is dirty: {status[:20]}"
        )
    baseline = "b9a158f75ab0740c91c9cfc8863a65fa381b014c"
    baseline_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, "HEAD"],
        cwd=repo,
        check=False,
    ).returncode == 0
    forbidden = "860ec8ca10cb5d6bed9d901560d3eb3d811a8143"
    forbidden_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", forbidden, "HEAD"],
        cwd=repo,
        check=False,
    ).returncode == 0
    if not baseline_check or forbidden_check:
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: locked Git provenance failed")
    return {
        "path": str(root),
        "branch": branch,
        "git_commit": commit,
        "clean": not status,
        "status_porcelain": status,
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "integration_baseline_is_ancestor": baseline_check,
        "forbidden_feature_is_ancestor": forbidden_check,
    }


def sparse_asset_contract(repo: Path) -> Dict[str, object]:
    mesh_root = repo / "data/object/rest_object_geo"
    names = tuple(sorted(path.stem for path in mesh_root.glob("*.ply")))
    if names != tuple(OBJECT_NAMES):
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: object mapping mismatch")
    mapping_payload = {
        "algorithm": "sequence-name-second-underscore-field-v1",
        "names": list(names),
    }
    mapping_hash = canonical_sha256(mapping_payload)
    records = []
    selected_points = []
    dataset = PriorWindowDataset(
        str(repo),
        "hoi",
        partition="internal_validation",
        limit=1,
        split_manifest=str(
            repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        ),
    )
    for name in names:
        path = mesh_root / f"{name}.ply"
        mesh = trimesh.load_mesh(path, process=False)
        vertices = zup_to_yup(
            np.asarray(mesh.vertices, dtype=np.float32).copy()
        )
        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.linspace(
            0, len(vertices) - 1, min(100, len(vertices))
        ).round().astype(np.int64)
        points = np.ascontiguousarray(vertices[indices], dtype=np.float32)
        dataset_points = np.asarray(dataset._rest_object_points(name), dtype=np.float32)
        if not np.array_equal(points, dataset_points):
            raise RuntimeError(
                f"{FAILURE_CLASSIFICATION}: real dataset sparse points differ for {name}"
            )
        selected_points.append(torch.from_numpy(points.copy()))
        records.append({
            "name": name,
            "source_mesh": f"{name}.ply",
            "source_mesh_sha256": sha256_file(path),
            "source_vertex_count": int(len(vertices)),
            "point_count": int(len(points)),
            "selection": (
                "numpy.linspace(0,n-1,min(100,n)).round().astype(int64)"
            ),
            "indices_sha256": hashlib.sha256(
                np.ascontiguousarray(indices, dtype=np.int64).tobytes()
            ).hexdigest(),
            "points_float32_yup_sha256": hashlib.sha256(
                points.tobytes()
            ).hexdigest(),
        })
    manifest_payload = {
        "algorithm": "d2x-rest-object-points-100-yup-linspace-vertex-v1",
        "objects": records,
    }
    manifest_hash = canonical_sha256(manifest_payload)
    stacked = torch.stack(selected_points).contiguous()
    stacked_hash = tensor_sha256(stacked)
    checks = {
        "mapping": mapping_hash == EXPECTED_MAPPING_SHA256,
        "manifest": manifest_hash == EXPECTED_MANIFEST_SHA256,
        "module_manifest": (
            SPARSE_POINT_MANIFEST_SHA256
            == EXPECTED_MANIFEST_SHA256
        ),
        "module_mapping": SPARSE_POINT_MAPPING_SHA256 == EXPECTED_MAPPING_SHA256,
        "module_stacked_tensor": (
            SPARSE_POINT_TENSOR_SHA256 == EXPECTED_STACKED_TENSOR_SHA256
        ),
        "stacked_tensor": stacked_hash == EXPECTED_STACKED_TENSOR_SHA256,
        "shape": tuple(stacked.shape) == (13, 100, 3),
        "dtype": stacked.dtype == torch.float32,
        "finite": bool(torch.isfinite(stacked).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: sparse asset contract failed: {checks}"
        )
    return {
        "mapping_payload": mapping_payload,
        "mapping_sha256": mapping_hash,
        "manifest_payload": manifest_payload,
        "manifest_sha256": manifest_hash,
        "stacked_shape": list(stacked.shape),
        "stacked_tensor_sha256": stacked_hash,
        "dataset_byte_exact": True,
        "checks": checks,
    }


def manual_surface(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    current = values["current"][:, list(TEMPORAL_ANCHORS)]
    object_scale = (
        values["object_maximum"] - values["object_minimum"]
    ).reshape(1, 1, 3)
    object_base = values["object_minimum"].reshape(1, 1, 3)
    translation = (
        (current[..., 216:219] + 1.0) * object_scale / 2.0 + object_base
    )
    relative = project_to_so3(
        current[..., 219:228].reshape(*current.shape[:2], 3, 3)
    )
    reference = values["object_rotation_reference"][:, None]
    global_rotation = project_to_so3(relative @ reference)
    local_rotation = values["world_to_local_rotation"][:, None] @ global_rotation
    surface = torch.einsum(
        "bpc,btdc->btpd",
        values["rest_object_points"].to(current),
        local_rotation,
    )
    return surface + translation[:, :, None]


def geometry_contract() -> Dict[str, object]:
    torch.manual_seed(42)
    model = build_expert(
        "hoi",
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AE,
    ).eval()
    model.network.set_sparse_relation_gate_override(0.1)
    values = synthetic_inputs(batch=2)
    with torch.no_grad():
        full_output, full = capture_forward(model, values)
    expected_surface = manual_surface(values)
    surface_parity = float(
        (full["surface"] - expected_surface).abs().max()
    )
    if surface_parity > 1.0e-6:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: object-surface parity {surface_parity}"
        )

    common = yaw(53).expand(values["current"].shape[0], -1, -1)
    rotated = dict(values)
    rotated["world_to_local_rotation"] = (
        values["world_to_local_rotation"] @ common.transpose(-1, -2)
    )
    rotated["object_rotation_reference"] = (
        common @ values["object_rotation_reference"]
    )
    with torch.no_grad():
        yaw_output, yaw_snapshot = capture_forward(model, rotated)
    yaw_surface_max_abs = float(
        (full["surface"] - yaw_snapshot["surface"]).abs().max()
    )
    yaw_relation_max_abs = float(
        (full["relation_vectors"] - yaw_snapshot["relation_vectors"]).abs().max()
    )

    translated = dict(values)
    translated["current"] = values["current"].clone()
    translated["current"][:, list(TEMPORAL_ANCHORS), 216] += 0.25
    with torch.no_grad():
        _, translated_snapshot = capture_forward(model, translated)
    translation_effect = float(
        (full["relation_vectors"] - translated_snapshot["relation_vectors"])
        .abs().max()
    )

    relative_rotated = dict(values)
    relative_rotated["current"] = values["current"].clone()
    relative_rotated["current"][:, list(TEMPORAL_ANCHORS), 219:228] = (
        yaw(37).reshape(1, 1, 9)
    )
    with torch.no_grad():
        _, rotation_snapshot = capture_forward(model, relative_rotated)
    rotation_effect = float(
        (full["relation_vectors"] - rotation_snapshot["relation_vectors"])
        .abs().max()
    )

    swapped = dict(values)
    swapped["current"] = values["current"].clone()
    joints = swapped["current"][..., :84].reshape(2, 16, 28, 3)
    left = joints[..., 24, :].clone()
    joints[..., 24, :] = joints[..., 26, :]
    joints[..., 26, :] = left
    with torch.no_grad():
        _, swapped_snapshot = capture_forward(model, swapped)
    pooled = full["pooled_blocks"]
    swapped_pooled = swapped_snapshot["pooled_blocks"]
    left_right_exchange = max(
        float((pooled[:, :, 0] - swapped_pooled[:, :, 1]).abs().max()),
        float((pooled[:, :, 1] - swapped_pooled[:, :, 0]).abs().max()),
        float((pooled[:, :, 2] - swapped_pooled[:, :, 2]).abs().max()),
    )

    reordered = dict(values)
    reordered["rest_object_points"] = torch.flip(
        values["rest_object_points"], dims=(1,),
    ).contiguous()
    with torch.no_grad():
        reordered_output, reordered_snapshot = capture_forward(model, reordered)
    point_permutation_output_max_abs = float(
        (full_output - reordered_output).abs().max()
    )
    point_permutation_relation_max_abs = float(
        (full["relation_vectors"] - reordered_snapshot["relation_vectors"])
        .abs().max()
    )

    model.network.set_sparse_relation_diagnostic_variant(
        "temporal_correspondence_permuted"
    )
    with torch.no_grad():
        temporal_output, temporal_snapshot = capture_forward(model, values)
    model.network.set_sparse_relation_diagnostic_variant("full")
    temporal_effect = float((full_output - temporal_output).abs().max())

    checks = {
        "common_global_yaw_surface": yaw_surface_max_abs <= 1.0e-6,
        "common_global_yaw_relation": yaw_relation_max_abs <= 1.0e-6,
        "common_global_yaw_output": (
            float((full_output - yaw_output).abs().max()) <= 1.0e-6
        ),
        "relative_translation_sensitive": translation_effect > 1.0e-8,
        "relative_rotation_sensitive": rotation_effect > 1.0e-8,
        "left_right_exact_exchange": left_right_exchange <= 1.0e-6,
        "point_order_output_invariant": point_permutation_output_max_abs <= 1.0e-6,
        "point_order_relation_invariant": point_permutation_relation_max_abs <= 1.0e-6,
        "temporal_permutation_sensitive": temporal_effect > 1.0e-8,
        "temporal_geometry_changed": bool(
            torch.any(
                full["relation_vectors"]
                != temporal_snapshot["relation_vectors"]
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: geometry invariance/sensitivity failed: {checks}"
        )
    model.network.set_sparse_relation_gate_override(None)
    return {
        "surface_shape": list(full["surface"].shape),
        "role_point_features_shape": list(full["role_point_features"].shape),
        "pooled_blocks_shape": list(full["pooled_blocks"].shape),
        "relation_vectors_shape": list(full["relation_vectors"].shape),
        "routed_relation_shape": list(full["routed_relation"].shape),
        "surface_loss_transform_parity_max_abs": surface_parity,
        "common_global_yaw_surface_max_abs": yaw_surface_max_abs,
        "common_global_yaw_relation_max_abs": yaw_relation_max_abs,
        "relative_translation_relation_max_abs": translation_effect,
        "relative_rotation_relation_max_abs": rotation_effect,
        "left_right_pooled_exchange_max_abs": left_right_exchange,
        "point_permutation_output_max_abs": point_permutation_output_max_abs,
        "point_permutation_relation_max_abs": point_permutation_relation_max_abs,
        "temporal_permutation_output_max_abs": temporal_effect,
        "checks": checks,
    }


def _gradient_group(
    module: torch.nn.Module, prefixes: tuple[str, ...]
) -> Dict[str, object]:
    parameters = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    records = []
    for name, parameter in parameters:
        gradient = parameter.grad
        records.append({
            "name": name,
            "present": gradient is not None,
            "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
            "nonzero": gradient is not None and bool(torch.any(gradient != 0)),
        })
    return {
        "parameters": records,
        "finite": bool(records) and all(record["finite"] for record in records),
        "nonzero": bool(records) and any(record["nonzero"] for record in records),
    }


def native_sampler_metadata_contract(repo: Path) -> Dict[str, object]:
    """Intercept the official sampler metadata path for one real HOI window."""
    dataset = PriorWindowDataset(
        str(repo),
        "hoi",
        partition="train",
        split_manifest=str(
            repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        ),
    )
    item = dataset[0]
    global_index = int(dataset.indices[0])
    sequence_index = int(dataset.sequence_ids[global_index])
    sequence_name = str(dataset.scene_names[sequence_index])

    class EvaluatorDatasetContract:
        load_scene = False
        min_torch = torch.from_numpy(dataset.minimum.copy()).float()
        max_torch = torch.from_numpy(dataset.maximum.copy()).float()
        obj_min_torch = torch.from_numpy(dataset.object_minimum.copy()).float()
        obj_max_torch = torch.from_numpy(dataset.object_maximum.copy()).float()

        def normalize_torch(self, value, *, is_object=False):
            minimum = self.obj_min_torch if is_object else self.min_torch
            maximum = self.obj_max_torch if is_object else self.max_torch
            return (value - minimum.to(value)) * 2.0 / (
                maximum.to(value) - minimum.to(value)
            ) - 1.0

    capture: Dict[str, object] = {}

    class CaptureDiffusion:
        def sample(
            self,
            model,
            fixed_history,
            text_embedding,
            object_bps,
            goals,
            progress,
            **kwargs,
        ):
            del model
            capture.update({
                "fixed_history": fixed_history.detach().clone(),
                "text_embedding": text_embedding.detach().clone(),
                "object_bps": object_bps.detach().clone(),
                "goals": goals.detach().clone(),
                "progress": progress.detach().clone(),
                "relation": {
                    key: value.detach().clone()
                    for key, value in kwargs.items()
                    if torch.is_tensor(value)
                    and key not in {"generator"}
                },
            })
            output = fixed_history.new_zeros(
                fixed_history.shape[0], 16, 232,
            )
            output[:, :2] = fixed_history
            return output

    rest_vertices = {}
    for name in OBJECT_NAMES:
        mesh = trimesh.load_mesh(
            repo / "data/object/rest_object_geo" / f"{name}.ply",
            process=False,
        )
        vertices = zup_to_yup(
            np.asarray(mesh.vertices, dtype=np.float32).copy()
        )
        rest_vertices[name] = torch.from_numpy(
            np.ascontiguousarray(vertices, dtype=np.float32)
        )

    model = build_expert(
        "hoi",
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AE,
    ).eval()
    sampler = HOIPriorSampler("cpu")
    sampler.set_dataset_and_model(EvaluatorDatasetContract(), model)
    sampler.diffusion = CaptureDiffusion()
    transform = torch.eye(4).reshape(1, 4, 4)
    transform[:, :3, :3] = item["world_to_local_rotation"].transpose(-1, -2)
    pi = int(dataset.language["pi"][global_index])
    sequence_length = int(
        dataset.seq_ends[sequence_index] - dataset.seq_starts[sequence_index]
    )
    sampler.p_sample_loop(
        item["x"][:2].unsqueeze(0),
        transform,
        None,
        item["text_embedding"].unsqueeze(0),
        item["goals"][:3].unsqueeze(0),
        None,
        torch.zeros(1, 3),
        None,
        None,
        torch.tensor([pi]),
        torch.tensor([min(pi + 48, sequence_length)]),
        torch.tensor([sequence_length]),
        None,
        None,
        torch.ones(1, dtype=torch.bool),
        item["object_bps"].unsqueeze(0),
        None,
        item["object_rotation_reference"].reshape(1, 1, 3, 3),
        rest_vertices,
        {0: sequence_name},
        object_only=True,
    )
    actual = capture.get("relation")
    if not isinstance(actual, Mapping):
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: native sampler capture failed")
    expected = {
        "rest_object_points": item["rest_object_points"].unsqueeze(0),
        "world_to_local_rotation": item["world_to_local_rotation"].unsqueeze(0),
        "object_rotation_reference": item["object_rotation_reference"].unsqueeze(0),
        "position_minimum": torch.from_numpy(dataset.minimum.copy()).float(),
        "position_maximum": torch.from_numpy(dataset.maximum.copy()).float(),
        "object_minimum": torch.from_numpy(dataset.object_minimum.copy()).float(),
        "object_maximum": torch.from_numpy(dataset.object_maximum.copy()).float(),
    }
    key_parity = {
        key: float((actual[key] - expected[key]).abs().max())
        for key in expected
    }
    current = item["x"].unsqueeze(0)
    native_geometry = build_sparse_relation_geometry(current, **actual)
    train_geometry = build_sparse_relation_geometry(current, **expected)
    surface_parity = float(
        (native_geometry["surface"] - train_geometry["surface"]).abs().max()
    )
    feature_parity = float(
        (native_geometry["features"] - train_geometry["features"]).abs().max()
    )
    audit = sampler.audit_dict()
    if (
        set(actual) != set(expected)
        or max(key_parity.values()) > 1.0e-6
        or max(surface_parity, feature_parity) > 1.0e-6
        or audit.get("sparse_relation_metadata_calls") != 1
        or audit.get("sparse_relation_asset_contract", {}).get(
            "stacked_tensor_sha256"
        ) != SPARSE_POINT_TENSOR_SHA256
    ):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: native sampler metadata parity failed"
        )
    return {
        "sequence": sequence_name,
        "relation_keys": sorted(actual),
        "metadata_max_abs_by_key": key_parity,
        "surface_max_abs": surface_parity,
        "feature_max_abs": feature_parity,
        "sampler_audit": audit,
    }


def model_contract() -> Dict[str, object]:
    torch.manual_seed(42)
    base = build_expert(
        "hoi", dim_model=512, num_heads=16, num_layers=8,
    )
    torch.manual_seed(99)
    model = build_expert(
        "hoi",
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AE,
    )
    initial_model_hash = state_dict_sha256(model.state_dict())
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    shared_keys = sorted(base.state_dict())
    sparse_keys = sorted(set(model.state_dict()) - set(base.state_dict()))
    shared_state_exact = all(
        torch.equal(base.state_dict()[key], model.state_dict()[key])
        for key in shared_keys
    )
    if (
        sorted(missing) != sparse_keys
        or unexpected
        or len(shared_keys) != 119
        or len(sparse_keys) != 10
        or not all(key.startswith("network.sparse_relation_field.") for key in sparse_keys)
        or not shared_state_exact
        or float(model.network.sparse_relation_field.alpha.detach()) != 0.0
    ):
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: shared trunk load failed")
    base.eval()
    model.eval()
    values = synthetic_inputs(batch=1)
    with torch.no_grad():
        expected = base(
            values["current"], values["timesteps"], values["text"],
            values["global_bps"], values["goals"], values["progress"],
        )
        actual = forward_model(model, values)
    base_parity = float((expected - actual).abs().max())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    relation_parameters = sum(
        parameter.numel()
        for parameter in model.network.sparse_relation_field.parameters()
    )
    increase_fraction = relation_parameters / EXPECTED_BASE_PARAMETERS
    if (
        total_parameters != EXPECTED_TOTAL_PARAMETERS
        or relation_parameters != EXPECTED_INCREMENT_PARAMETERS
        or relation_parameters != SPARSE_RELATION_PARAMETER_COUNT
        or increase_fraction > EXPECTED_PARAMETER_INCREASE_LIMIT
        or base_parity > 1.0e-6
        or tuple(actual.shape) != (1, 16, 232)
    ):
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: model/count/parity contract failed")

    model.train()
    values = synthetic_inputs(batch=2)
    prediction = forward_model(model, values)
    (prediction - values["current"]).square().mean().backward()
    initial_audit = _d2ae_gradient_audit(
        model, require_relation_paths=False,
    )
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.network.sparse_relation_field.alpha.copy_(
            torch.atanh(torch.tensor(0.1))
        )
    prediction = forward_model(model, values)
    (prediction - values["current"]).square().mean().backward()
    activated_audit = _d2ae_gradient_audit(
        model, require_relation_paths=True,
    )
    activated_groups = {
        "point_encoder": _gradient_group(
            model.network.sparse_relation_field, ("point_encoder",)
        ),
        "projection": _gradient_group(
            model.network.sparse_relation_field,
            ("projection", "relation_projection", "writeback"),
        ),
        "temporal_embeddings": _gradient_group(
            model.network.sparse_relation_field,
            ("temporal_embedding", "temporal_embeddings"),
        ),
        "motion_input": _gradient_group(model.network, ("motion_input",)),
        "transformer": _gradient_group(model.network, ("transformer",)),
    }
    if not all(
        value["finite"] and value["nonzero"]
        for value in activated_groups.values()
    ):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: activated gradient groups failed"
        )

    finite_cases = {}
    so3_cases = {}
    for label, fill in (("zero", 0.0), ("constant", 3.0), ("extreme", 1.0e4)):
        case = synthetic_inputs(batch=2)
        case["current"].fill_(fill)
        with torch.no_grad():
            case_output, snapshot = capture_forward(model.eval(), case)
            case_geometry = build_sparse_relation_geometry(
                case["current"],
                case["rest_object_points"],
                case["world_to_local_rotation"],
                case["object_rotation_reference"],
                case["position_minimum"],
                case["position_maximum"],
                case["object_minimum"],
                case["object_maximum"],
            )
        rotations = case_geometry["relative_rotation"]
        identity = torch.eye(3).to(rotations)
        orthogonality = float(
            (rotations.transpose(-1, -2) @ rotations - identity).abs().max()
        )
        determinant = torch.linalg.det(rotations)
        finite_cases[label] = bool(
            torch.isfinite(case_output).all()
            and torch.isfinite(snapshot["surface"]).all()
            and torch.isfinite(snapshot["relation_vectors"]).all()
        )
        so3_cases[label] = {
            "finite": bool(torch.isfinite(rotations).all()),
            "orthogonality_max_abs": orthogonality,
            "determinant_minimum": float(determinant.amin()),
            "determinant_maximum": float(determinant.amax()),
        }
    double_field = SparseCurrentStateRelationField(512).double().eval()
    double_values = synthetic_inputs(batch=3, dtype=torch.float64)
    double_motion = torch.randn(
        3,
        16,
        512,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(456),
    )
    with torch.no_grad():
        double_output = double_field(
            double_motion,
            double_values["current"],
            double_values["rest_object_points"],
            double_values["world_to_local_rotation"],
            double_values["object_rotation_reference"],
            double_values["position_minimum"],
            double_values["position_maximum"],
            double_values["object_minimum"],
            double_values["object_maximum"],
        )
        double_geometry = build_sparse_relation_geometry(
            double_values["current"],
            double_values["rest_object_points"],
            double_values["world_to_local_rotation"],
            double_values["object_rotation_reference"],
            double_values["position_minimum"],
            double_values["position_maximum"],
            double_values["object_minimum"],
            double_values["object_maximum"],
        )
    dtype_batch = (
        tuple(double_output.shape) == (3, 16, 512)
        and double_output.dtype == torch.float64
        and double_geometry["surface"].dtype == torch.float64
        and bool(torch.isfinite(double_output).all())
    )
    if (
        not all(finite_cases.values())
        or not dtype_batch
        or not all(
            value["finite"]
            and value["orthogonality_max_abs"] <= 1.0e-5
            and abs(value["determinant_minimum"] - 1.0) <= 1.0e-5
            and abs(value["determinant_maximum"] - 1.0) <= 1.0e-5
            for value in so3_cases.values()
        )
    ):
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: finiteness/dtype contract failed")

    sampler_parity = native_sampler_metadata_contract(REPO)

    hsi = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
    assert_parameter_independence(model, hsi)
    if "rest_object_points" in inspect.signature(hsi.forward).parameters:
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: HSIPrior API changed")
    return {
        "initial_model_state_sha256": initial_model_hash,
        "base_parameters": EXPECTED_BASE_PARAMETERS,
        "relation_parameters": relation_parameters,
        "total_parameters": total_parameters,
        "parameter_increase_fraction": increase_fraction,
        "parameter_limit": EXPECTED_PARAMETER_INCREASE_LIMIT,
        "output_shape": list(actual.shape),
        "base_parity_max_abs": base_parity,
        "shared_state_key_count": len(shared_keys),
        "sparse_state_key_count": len(sparse_keys),
        "shared_state_exact": shared_state_exact,
        "sparse_state_keys": sparse_keys,
        "initial_alpha_gradient": initial_audit,
        "activated_relation_gradients": activated_audit,
        "activated_gradient_groups": activated_groups,
        "test_only_gate": 0.1,
        "test_only_probe_saved": False,
        "test_only_probe_optimizer_updates": 0,
        "zero_constant_extreme_finite": finite_cases,
        "so3_projection_finite": all(
            value["finite"] for value in so3_cases.values()
        ),
        "so3_projection_cases": so3_cases,
        "dtype_device_batch_propagation": dtype_batch,
        "train_sampler_surface_parity_max_abs": sampler_parity["surface_max_abs"],
        "train_sampler_feature_parity_max_abs": sampler_parity["feature_max_abs"],
        "native_sampler_metadata_parity": sampler_parity,
        "hsiprior_parameter_storage_independent": True,
        "mixer_clean_output_contract": [3, 16, 232],
    }


def checkpoint_rejection_contract() -> Dict[str, object]:
    from priors.d2ad import (  # local import keeps the D2-AE relation module pure
        BPS_YUP_TENSOR_SHA256,
        DEFAULT_QUERY_WORKERS,
        OBJECT_MAPPING_SHA256,
        REST_MESH_MANIFEST_SHA256,
    )
    from priors.interaction_adapter import (
        ADAPTER_PARAMETER_COUNT,
        ASSIGNMENT_SHA256,
        BPS_SHA256,
        LOCAL_BASIS_COORDINATE_SYSTEM,
    )

    common = {
        "checkpoint_type": "hoi_prior_phase1b",
        "expert": "hoi",
        "initialization": "random",
        "seed": 42,
    }
    d2ac_contract = {
        "bps_sha256": BPS_SHA256,
        "assignment_sha256": ASSIGNMENT_SHA256,
        "adapter_parameters": ADAPTER_PARAMETER_COUNT,
    }
    d2ad_contract = {
        **d2ac_contract,
        "basis_coordinate_system": LOCAL_BASIS_COORDINATE_SYSTEM,
        "basis_yup_tensor_sha256": BPS_YUP_TENSOR_SHA256,
        "rest_mesh_manifest_sha256": REST_MESH_MANIFEST_SHA256,
        "object_mapping_sha256": OBJECT_MAPPING_SHA256,
        "query_backend": "scipy.spatial.cKDTree.query",
        "query_parameters": {"k": 1, "eps": 0.0, "p": 2},
        "query_workers": DEFAULT_QUERY_WORKERS,
        "full_rest_mesh": True,
        "mesh_subsample": False,
        "stored_per_window_local_bps": False,
    }

    variants = {
        "released": {"model": {}},
        "d2x_base": {
            **common,
            "run_id": "p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723",
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_BASE,
            },
        },
        "d2ac": {
            **common,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_D2AC,
            },
            "architecture_variant": HOI_ARCHITECTURE_D2AC,
            "interaction_adapter_contract": d2ac_contract,
        },
        "d2ad": {
            **common,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_D2AD,
            },
            "architecture_variant": HOI_ARCHITECTURE_D2AD,
            "interaction_adapter_contract": d2ad_contract,
        },
    }
    rejected = {}
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for label, checkpoint in variants.items():
            path = temporary / f"{label}.pth"
            torch.save(checkpoint, path)
            try:
                load_trained_hoi_prior(
                    str(path),
                    torch.device("cpu"),
                    use_ema=False,
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AE,
                )
            except (ValueError, RuntimeError) as error:
                rejected[label] = {
                    "rejected": True,
                    "error": str(error),
                }
            else:
                rejected[label] = {"rejected": False, "error": None}
    if not all(value["rejected"] for value in rejected.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: checkpoint rejection failed: {rejected}"
        )
    return {
        "variants": rejected,
        "scientific_checkpoint_loads": 0,
        "synthetic_checkpoint_attempts": len(rejected),
    }


def static_contract(repo: Path) -> Dict[str, object]:
    sparse_path = repo / "code/priors/sparse_relation.py"
    sparse_source = sparse_path.read_text(encoding="utf-8")
    tree = ast.parse(sparse_source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    forbidden_imports = sorted(
        imported_roots & {"numpy", "scipy", "trimesh", "sklearn"}
    )
    builder_source = inspect.getsource(build_sparse_relation_geometry)
    lower_builder = builder_source.lower()
    forbidden_builder_tokens = (
        "x_start",
        "clean_target",
        "future_gt",
        "contact_label",
        "scene",
        "ckdtree",
        "cdist",
        "full_mesh",
        "stored_relation",
    )
    forbidden_builder_hits = [
        token for token in forbidden_builder_tokens if token in lower_builder
    ]
    model_source = (repo / "code/priors/models.py").read_text(encoding="utf-8")
    diffusion_source = (repo / "code/priors/diffusion.py").read_text(encoding="utf-8")
    data_source = (repo / "code/priors/data.py").read_text(encoding="utf-8")
    official_hashes = {
        # P5-P7 (5e89644) added the default-off inference contact guidance entry
        # points to the official evaluator.  Guidance is off unless a config
        # enables it, so every sealed native evaluation reproduces; the
        # null-mask and sub-term parity are locked by
        # tests/test_hoi_guidance_gt_mask.py and
        # tests/test_hoi_guidance_subterms.py.
        "code/test_infbagel_hoi.py":
            "ca274e5fe358ebaec3b1d08e4480327e94ee39ede3a1a675afad79859fd6e783",
        "code/eval_metrics.py":
            "445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547",
        "code/config/config_eval_hoi_prior.yaml":
            "89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73",
    }
    official_actual = {
        path: sha256_file(repo / path) for path in official_hashes
    }
    checks = {
        "pure_torch_imports": not forbidden_imports,
        "forbidden_relation_sources_absent": not forbidden_builder_hits,
        "no_d2ae_collator_dynamic_geometry": "D2AE" not in data_source,
        "model_accepts_current_metadata": all(
            token in model_source for token in (
                "rest_object_points",
                "world_to_local_rotation",
                "object_rotation_reference",
            )
        ),
        "sampler_forwards_current_metadata": all(
            token in diffusion_source for token in (
                "rest_object_points",
                "world_to_local_rotation",
                "object_rotation_reference",
            )
        ),
        "official_evaluator_hashes_unchanged": official_actual == official_hashes,
        "no_clean_target_forward_argument": "clean_target" not in inspect.signature(
            build_expert(
                "hoi",
                dim_model=32,
                num_heads=4,
                num_layers=1,
            ).forward
        ).parameters,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: static contract failed: {checks}"
        )
    return {
        "checks": checks,
        "forbidden_imports": forbidden_imports,
        "forbidden_builder_hits": forbidden_builder_hits,
        "official_evaluator_hashes": official_actual,
        "relation_builder_source_sha256": hashlib.sha256(
            builder_source.encode("utf-8")
        ).hexdigest(),
    }


def training_and_registry_contract(
    repo: Path,
    formal_run_id: str,
) -> Dict[str, object]:
    cfg = merged_config(repo, formal_run_id)
    _validate_d2ae_contract(cfg, 4, require_performance_gate=False)
    split_hash = sha256_file(Path(str(cfg.split_manifest)))
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: split hash mismatch")
    expected_weights = {
        "fk": 0.3569973401779424,
        "object_surface": 0.4772322188400037,
        "velocity": 0.1,
        "terminal_goal": 1.0,
    }
    weights = _locked_loss_weights(cfg)
    optimization = _optimization_contract(cfg)
    if weights != expected_weights:
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: loss weights changed")
    if optimization["optimizer"] != "Adam" or optimization["scheduler"] != "none":
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: optimizer contract changed")
    validation = subprocess.run(
        [sys.executable, "tools/experiment.py", "validate"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if validation.returncode != 0:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: registry validation failed: {validation.stdout}"
        )
    return {
        "split_sha256": split_hash,
        "loss_weights": weights,
        "optimization": optimization,
        "registry_validation_returncode": validation.returncode,
        "registry_validation_output": validation.stdout.strip(),
        "full_authority_suite_commands": list(AUTHORITY_VERIFICATION_COMMANDS),
        "resolved_config_has_unresolved_interpolation": "${" in OmegaConf.to_yaml(cfg),
    }


def run_contract(
    repo: Path,
    run_id: str,
    *,
    require_clean: bool = True,
) -> Dict[str, object]:
    repo = repo.resolve()
    run_id_match = RUN_ID_RE.fullmatch(str(run_id))
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if run_id_match is None or run_id_match.group("date") != actual_date:
        raise ValueError(
            "D2-AE CPU contract run id must use the locked stem and actual date"
        )
    formal_run_id = (
        "p1-hoi-d2ae-sparse-relation-field-s42-"
        f"{run_id_match.group('date')}"
    )
    if Path(sys.executable).resolve() != EXPECTED_AUTHORITY_PYTHON.resolve():
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: unexpected authority Python {sys.executable}"
        )
    identity = identity_contract(repo, require_clean=require_clean)
    assets = sparse_asset_contract(repo)
    geometry = geometry_contract()
    model = model_contract()
    checkpoint = checkpoint_rejection_contract()
    static = static_contract(repo)
    training = training_and_registry_contract(repo, formal_run_id)
    if training["resolved_config_has_unresolved_interpolation"]:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: unresolved Hydra interpolation"
        )
    return {
        "schema_version": 1,
        "status": "completed",
        "classification": "cpu-contract-passed",
        "run_id": run_id,
        "subphase": "1B-D2-AE0-cpu-contract",
        "seed": 42,
        "identity": identity,
        "sparse_assets": assets,
        "geometry": geometry,
        "model": model,
        "checkpoint_provenance": checkpoint,
        "static_contract": static,
        "training_and_registry": training,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "scientific_checkpoint_loads": 0,
        "checkpoint_writes": 0,
        "official_test_used": False,
        "checkpoint_selection": False,
        "consistency_started": False,
        "hsiprior_started": False,
        "mixer_started": False,
    }


def atomic_json(path: Path, value: MutableMapping[str, object]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_contract(
            args.repo_root,
            args.run_id,
            require_clean=True,
        )
    except Exception as error:
        print(f"{FAILURE_CLASSIFICATION}: {error}", flush=True)
        raise
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
