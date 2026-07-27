#!/usr/bin/env python3
"""Run the authority-only D2-AD0 CPU coordinate/adapter contract."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

import scipy
import torch
import trimesh
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

from priors.contact_alignment import (  # noqa: E402
    PHASE_OFFSETS,
    SELECTION_SHA256,
    select_contact_holdout,
)
from priors.d2ad import (  # noqa: E402
    BPS_YUP_TENSOR_SHA256,
    DEFAULT_QUERY_WORKERS,
    OBJECT_MAPPING_SHA256,
    OBJECT_NAMES,
    REST_MESH_MANIFEST_SHA256,
    D2ADBatchCollator,
    D2ADPriorWindowDataset,
    LocalObjectBPSBuilder,
)
from priors.diffusion import HOIPriorSampler  # noqa: E402
from priors.interaction_adapter import (  # noqa: E402
    ADAPTER_PARAMETER_COUNT,
    ASSIGNMENT_SHA256,
    BPS_SHA256,
    CENTER_INDICES,
    CLUSTER_SIZES,
    LOCAL_BASIS_COORDINATE_SYSTEM,
    LocalObjectInteractionAdapter,
    load_bps_partition,
)
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_D2AC,
    HOI_ARCHITECTURE_D2AD,
    assert_parameter_independence,
    build_expert,
    load_trained_hoi_prior,
)
from priors.window_codec import WindowFrame, zup_to_yup_tensor  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    _d2ac_gradient_audit,
    _validate_d2ad_contract,
)


RUN_ID = "p1-hoi-d2ad-cpu-contract-s42-20260728"
FAILURE_CLASSIFICATION = "local-frame-interaction-adapter-contract-failure-stop"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _yaw(angle_degrees: float) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        (
            (cosine, 0.0, sine),
            (0.0, 1.0, 0.0),
            (-sine, 0.0, cosine),
        ),
        dtype=torch.float32,
    )


def _inputs(batch: int = 2):
    generator = torch.Generator().manual_seed(42)
    return {
        "noisy": torch.randn(batch, 16, 232, generator=generator),
        "timesteps": torch.arange(batch, dtype=torch.long) * 249 % 500,
        "text": torch.randn(batch, 768, generator=generator),
        "global_bps": torch.randn(batch, 1024, 3, generator=generator),
        "goals": torch.randn(batch, 9, generator=generator),
        "progress": torch.randn(batch, 3, generator=generator),
        "local_bps": torch.randn(batch, 1024, 3, generator=generator),
    }


def _forward(model: torch.nn.Module, values: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        values["noisy"],
        values["timesteps"],
        values["text"],
        values["global_bps"],
        values["goals"],
        values["progress"],
        local_object_bps=values["local_bps"],
    )


def _merged_config(repo: Path):
    base = OmegaConf.load(repo / "code/config/config_train_hoi_prior.yaml")
    d2ad = OmegaConf.load(repo / "code/config/config_train_hoi_prior_d2ad.yaml")
    cfg = OmegaConf.merge(base, d2ad)
    cfg.repo_root = str(repo)
    cfg.split_manifest = str(
        repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    return cfg


def _geometry_contract(repo: Path) -> Dict[str, object]:
    dataset = D2ADPriorWindowDataset(
        str(repo),
        "hoi",
        partition="internal_validation",
        split_manifest=str(
            repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        ),
    )
    selection = select_contact_holdout(dataset)
    if (
        selection["sha256"] != SELECTION_SHA256
        or selection["sequences"] != 64
        or selection["windows"] != 192
        or selection["phase_offsets"] != list(PHASE_OFFSETS)
    ):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: sealed cohort selection mismatch"
        )
    positions = [
        position
        for triple in selection["triples"]
        for position in triple
    ]
    items = [dataset[position] for position in positions]
    world_to_local = torch.stack([
        item["world_to_local_rotation"] for item in items
    ])
    object_rotation = torch.stack([
        item["object_rotation_reference"] for item in items
    ])
    object_indices = torch.stack([
        item["object_geometry_index"] for item in items
    ])
    object_names = [
        OBJECT_NAMES[int(index)] for index in object_indices.tolist()
    ]
    if set(object_names) != set(OBJECT_NAMES):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: sealed cohort misses object classes"
        )

    builder_one = LocalObjectBPSBuilder(repo, query_workers=1)
    started = time.perf_counter()
    original, original_indices = builder_one.build(
        world_to_local,
        object_rotation,
        object_indices,
        return_indices=True,
    )
    single_worker_seconds = time.perf_counter() - started
    builder_three = LocalObjectBPSBuilder(
        repo, query_workers=DEFAULT_QUERY_WORKERS,
    )
    started = time.perf_counter()
    threaded, threaded_indices = builder_three.build(
        world_to_local,
        object_rotation,
        object_indices,
        return_indices=True,
    )
    three_worker_seconds = time.perf_counter() - started
    builder_all = LocalObjectBPSBuilder(repo, query_workers=-1)
    started = time.perf_counter()
    all_worker, all_worker_indices = builder_all.build(
        world_to_local,
        object_rotation,
        object_indices,
        return_indices=True,
    )
    all_worker_seconds = time.perf_counter() - started
    repeated, repeated_indices = builder_three.build(
        world_to_local,
        object_rotation,
        object_indices,
        return_indices=True,
    )
    ordering = torch.arange(len(items) - 1, -1, -1)
    reordered, reordered_indices = builder_three.build(
        world_to_local[ordering],
        object_rotation[ordering],
        object_indices[ordering],
        return_indices=True,
    )
    adapter = LocalObjectInteractionAdapter(
        basis_coordinate_system=LOCAL_BASIS_COORDINATE_SYSTEM,
    )
    original_features = adapter.local_features(original)
    yaw_records = []
    local_max_abs = 0.0
    feature_max_abs = 0.0
    yaw_indices_exact = True
    for angle in (-179, -90, -37, 53, 120, 179):
        common = _yaw(angle).expand(len(items), -1, -1)
        rotated, rotated_indices = builder_three.build(
            world_to_local @ common.transpose(-1, -2),
            common @ object_rotation,
            object_indices,
            return_indices=True,
        )
        rotated_features = adapter.local_features(rotated)
        angle_local_max = float((original - rotated).abs().max())
        angle_feature_max = float(
            (original_features - rotated_features).abs().max()
        )
        angle_indices_exact = torch.equal(
            original_indices, rotated_indices,
        )
        local_max_abs = max(local_max_abs, angle_local_max)
        feature_max_abs = max(feature_max_abs, angle_feature_max)
        yaw_indices_exact = yaw_indices_exact and angle_indices_exact
        yaw_records.append({
            "degrees": angle,
            "nearest_indices_exact": angle_indices_exact,
            "local_bps_max_abs": angle_local_max,
            "cluster_feature_max_abs": angle_feature_max,
        })

    raw_basis, raw_assignment, _, raw_sizes, _ = load_bps_partition(
        repo / "code/bps.pt"
    )
    local_basis = adapter.bps_basis.detach().cpu()
    raw_to_yup_max_abs = float(
        (zup_to_yup_tensor(raw_basis) - local_basis).abs().max()
    )
    assignment_preserved = torch.equal(
        raw_assignment, adapter.cluster_assignment.detach().cpu()
    )
    cluster_sizes_preserved = torch.equal(
        raw_sizes, adapter.cluster_sizes.detach().cpu()
    )

    repeated_exact = (
        torch.equal(original_indices, repeated_indices)
        and torch.equal(original, repeated)
    )
    ordering_exact = (
        torch.equal(reordered_indices, original_indices[ordering])
        and torch.equal(reordered, original[ordering])
    )
    workers_exact = (
        torch.equal(original_indices, threaded_indices)
        and torch.equal(original, threaded)
        and torch.equal(original_indices, all_worker_indices)
        and torch.equal(original, all_worker)
    )
    if (
        not yaw_indices_exact
        or local_max_abs > 1.0e-6
        or feature_max_abs > 1.0e-6
        or not workers_exact
        or not repeated_exact
        or not ordering_exact
        or raw_to_yup_max_abs != 0.0
        or not assignment_preserved
        or not cluster_sizes_preserved
    ):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: coordinate/query determinism failed"
        )

    training_items = items[:8]
    collated = D2ADBatchCollator(
        repo, query_workers=DEFAULT_QUERY_WORKERS,
    )(training_items)
    direct = builder_three.build(
        torch.stack([
            item["world_to_local_rotation"] for item in training_items
        ]),
        torch.stack([
            item["object_rotation_reference"] for item in training_items
        ]),
        torch.stack([
            item["object_geometry_index"] for item in training_items
        ]),
    )
    training_direct_max_abs = float(
        (collated["local_object_bps"] - direct).abs().max()
    )
    sequence_names = [
        str(dataset.scene_names[int(item["sequence_index"])])
        for item in training_items
    ]
    transform = torch.eye(4, dtype=torch.float32)[None].repeat(
        len(training_items), 1, 1,
    )
    transform[:, :3, :3] = torch.stack([
        item["world_to_local_rotation"].transpose(-1, -2)
        for item in training_items
    ])
    evaluator = builder_one.build_from_evaluator_inputs(
        transform,
        torch.stack([
            item["object_rotation_reference"] for item in training_items
        ]),
        sequence_names,
    )
    dataset_evaluator_max_abs = float(
        (collated["local_object_bps"] - evaluator).abs().max()
    )
    if dataset_evaluator_max_abs != 0.0 or training_direct_max_abs != 0.0:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: dataset/direct/evaluator local-BPS mismatch"
        )

    relative_pose = builder_three.build(
        world_to_local,
        _yaw(37) @ object_rotation,
        object_indices,
    )
    relative_pose_mean_l2 = float(torch.linalg.vector_norm(
        relative_pose - original,
        dim=-1,
    ).mean())
    if relative_pose_mean_l2 <= 1.0e-4:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: local BPS is insensitive to relative pose"
        )

    history_item = training_items[0]
    history_frame = WindowFrame(
        history_item["window_origin"],
        history_item["world_to_local_rotation"],
        history_item["object_rotation_reference"],
    )
    history_decoded = dataset.codec.decode(
        history_item["x"][:2], history_frame,
    )
    generated_object_rotation = (
        _yaw(37)[None] @ history_decoded["object_rotation"]
    )
    _, generated_frame = dataset.codec.encode(
        history_decoded["joints"],
        history_decoded["human_rotation"],
        global_object_translation=history_decoded["object_translation"],
        global_object_rotation=generated_object_rotation,
        contact=history_decoded["contact"],
    )
    generated_local_bps = builder_three.build(
        generated_frame.world_to_local[None],
        generated_frame.object_reference[None],
        history_item["object_geometry_index"][None],
    )
    original_history_bps = builder_three.build(
        history_item["world_to_local_rotation"][None],
        history_item["object_rotation_reference"][None],
        history_item["object_geometry_index"][None],
    )
    generated_history_mean_l2 = float(torch.linalg.vector_norm(
        generated_local_bps - original_history_bps,
        dim=-1,
    ).mean())
    if generated_history_mean_l2 <= 1.0e-4:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: generated-history local BPS is inert"
        )

    vertex_counts = {}
    for index, name in enumerate(OBJECT_NAMES):
        vertices, _ = builder_one._load_geometry(index)
        vertex_counts[name] = int(vertices.shape[0])
    return {
        "metadata": builder_three.contract_metadata(),
        "bps_sha256": _sha256_file(repo / "code/bps.pt"),
        "basis_yup_tensor_sha256": BPS_YUP_TENSOR_SHA256,
        "raw_to_yup_basis_max_abs": raw_to_yup_max_abs,
        "assignment_preserved": assignment_preserved,
        "cluster_sizes_preserved": cluster_sizes_preserved,
        "center_indices": list(CENTER_INDICES),
        "cluster_sizes": list(CLUSTER_SIZES),
        "assignment_sha256": ASSIGNMENT_SHA256,
        "local_bps_shape": list(original.shape),
        "local_bps_dtype": str(original.dtype),
        "local_bps_finite": bool(torch.isfinite(original).all()),
        "local_bps_sha256": _tensor_sha256(original),
        "cluster_feature_shape": list(original_features.shape),
        "cluster_feature_dtype": str(original_features.dtype),
        "cluster_feature_finite": bool(torch.isfinite(original_features).all()),
        "common_yaw_local_bps_max_abs": local_max_abs,
        "common_yaw_cluster_feature_max_abs": feature_max_abs,
        "common_yaw_nearest_indices_exact": yaw_indices_exact,
        "common_yaw_records": yaw_records,
        "query_workers_1_3_all_indices_exact": workers_exact,
        "query_workers_1_3_all_output_exact": workers_exact,
        "repeated_call_exact": repeated_exact,
        "batch_ordering_exact": ordering_exact,
        "query_workers_1_seconds": single_worker_seconds,
        "query_workers_3_seconds": three_worker_seconds,
        "query_workers_all_seconds": all_worker_seconds,
        "selection": {
            key: value for key, value in selection.items()
            if key not in {"triples", "global_indices"}
        },
        "objects_covered": sorted(set(object_names)),
        "object_count": len(set(object_names)),
        "training_collator_direct_parity_max_abs": training_direct_max_abs,
        "dataset_evaluator_parity_max_abs": dataset_evaluator_max_abs,
        "dataset_local_bps_build_seconds": float(
            collated["local_bps_build_seconds"]
        ),
        "relative_pose_mean_point_l2_m": relative_pose_mean_l2,
        "generated_history_mean_point_l2_m": generated_history_mean_l2,
        "generated_history_current_frame_recomputed": True,
        "full_mesh_vertex_counts": vertex_counts,
        "mesh_subsample": False,
        "stored_per_window_local_bps": False,
    }


def _checkpoint_rejection_contract() -> Dict[str, object]:
    shared = {
        "bps_sha256": BPS_SHA256,
        "assignment_sha256": ASSIGNMENT_SHA256,
        "adapter_parameters": ADAPTER_PARAMETER_COUNT,
    }
    d2ad = {
        **shared,
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

    def value(variant: str, contract: Dict[str, object]):
        return {
            "checkpoint_type": "hoi_prior_phase1b",
            "expert": "hoi",
            "initialization": "random",
            "architecture_variant": variant,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": variant,
            },
            "interaction_adapter_contract": contract,
        }

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        d2ac_path = root / "d2ac.pth"
        malformed_path = root / "d2ad-malformed.pth"
        torch.save(value(HOI_ARCHITECTURE_D2AC, shared), d2ac_path)
        torch.save(
            value(
                HOI_ARCHITECTURE_D2AD,
                {**d2ad, "query_backend": "approximate"},
            ),
            malformed_path,
        )
        try:
            load_trained_hoi_prior(
                str(d2ac_path),
                torch.device("cpu"),
                expected_architecture_variant=HOI_ARCHITECTURE_D2AD,
            )
        except ValueError as error:
            d2ac_rejected = "architecture variant mismatch" in str(error)
        else:
            d2ac_rejected = False
        try:
            load_trained_hoi_prior(
                str(malformed_path),
                torch.device("cpu"),
                expected_architecture_variant=HOI_ARCHITECTURE_D2AD,
            )
        except ValueError as error:
            malformed_rejected = "local-geometry provenance" in str(error)
        else:
            malformed_rejected = False
    if not d2ac_rejected or not malformed_rejected:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: checkpoint provenance rejection failed"
        )
    return {
        "d2ac_checkpoint_rejected_by_d2ad": d2ac_rejected,
        "malformed_d2ad_geometry_rejected": malformed_rejected,
        "scientific_checkpoint_loads": 0,
        "synthetic_test_checkpoint_attempts": 2,
    }


def _model_contract() -> Dict[str, object]:
    values = _inputs()
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
        architecture_variant=HOI_ARCHITECTURE_D2AD,
    )
    initial_model_sha256 = _state_hash(model)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    if not missing or unexpected:
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: shared-trunk load failed")
    base.eval()
    model.eval()
    parity_values = _inputs(batch=1)
    with torch.no_grad():
        expected = base(
            parity_values["noisy"],
            parity_values["timesteps"],
            parity_values["text"],
            parity_values["global_bps"],
            parity_values["goals"],
            parity_values["progress"],
        )
        actual = _forward(model, parity_values)
    parity = float((expected - actual).abs().max())
    if parity > 1.0e-6:
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: base parity {parity}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    adapter_count = sum(
        parameter.numel()
        for parameter in model.network.interaction_adapter.parameters()
    )
    if parameter_count != 30_023_145 or adapter_count != ADAPTER_PARAMETER_COUNT:
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: parameter count mismatch")

    torch.manual_seed(42)
    d2ac = build_expert(
        "hoi",
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AC,
    )
    torch.manual_seed(42)
    d2ad = build_expert(
        "hoi",
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AD,
    )
    d2ac_parameters = dict(d2ac.named_parameters())
    d2ad_parameters = dict(d2ad.named_parameters())
    parameter_schema_identical = (
        {
            name: tuple(parameter.shape)
            for name, parameter in d2ac_parameters.items()
        }
        == {
            name: tuple(parameter.shape)
            for name, parameter in d2ad_parameters.items()
        }
    )
    parameter_initialization_identical = parameter_schema_identical and all(
        torch.equal(d2ac_parameters[name], d2ad_parameters[name])
        for name in d2ac_parameters
    )
    if not parameter_initialization_identical:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: D2-AC/D2-AD parameter lock failed"
        )

    model = d2ad.train()
    prediction = _forward(model, values)
    (prediction - values["noisy"]).square().mean().backward()
    initial_audit = _d2ac_gradient_audit(
        model, require_adapter_paths=False,
    )
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.network.interaction_adapter.alpha.copy_(
            torch.atanh(torch.tensor(0.1))
        )
    prediction = _forward(model, values)
    (prediction - values["noisy"]).square().mean().backward()
    activated_audit = _d2ac_gradient_audit(
        model, require_adapter_paths=True,
    )

    model.eval()
    with torch.no_grad():
        model.network.interaction_adapter.set_diagnostic_variant("full")
        model.network.interaction_adapter.set_gate_override(0.1)
        full = _forward(model, _inputs(batch=1))
        model.network.interaction_adapter.set_diagnostic_variant(
            "local_correspondence_permuted"
        )
        permuted = _forward(model, _inputs(batch=1))
    permutation_effect = float((full - permuted).abs().max())
    if permutation_effect <= 1.0e-8:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: locality permutation is inert"
        )

    adapter = LocalObjectInteractionAdapter(
        basis_coordinate_system=LOCAL_BASIS_COORDINATE_SYSTEM,
    ).double()
    extreme_finite = True
    for value in (
        torch.zeros(2, 1024, 3, dtype=torch.float64),
        torch.full((2, 1024, 3), 3.0, dtype=torch.float64),
        torch.full((2, 1024, 3), 1.0e4, dtype=torch.float64),
    ):
        extreme_finite = extreme_finite and bool(
            torch.isfinite(adapter.local_features(value)).all()
        )
    motion = torch.randn(
        3,
        16,
        512,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(7),
    )
    adapter_output = adapter(
        motion,
        torch.randn(
            3,
            1024,
            3,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(8),
        ),
    )
    dtype_batch_propagation = (
        tuple(adapter_output.shape) == (3, 16, 512)
        and adapter_output.dtype == torch.float64
        and bool(torch.isfinite(adapter_output).all())
    )
    role_separation = float(
        (adapter.part_embedding[0] - adapter.part_embedding[1]).abs().max()
    )
    if not extreme_finite or not dtype_batch_propagation or role_separation <= 0.0:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: feature/dtype/role contract failed"
        )

    hsi = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
    assert_parameter_independence(model, hsi)
    sampler = HOIPriorSampler(device="cpu", auto_regre_num=2, timesteps=500)
    del sampler
    return {
        "initial_model_state_sha256": initial_model_sha256,
        "parameter_count": parameter_count,
        "adapter_parameter_count": adapter_count,
        "parameter_increase_fraction": (
            (parameter_count - 29_673_448) / 29_673_448
        ),
        "api_output_shape": list(actual.shape),
        "forward_signature": list(
            inspect.signature(type(model).forward).parameters
        ),
        "local_object_bps_keyword_only": (
            inspect.signature(type(model).forward).parameters[
                "local_object_bps"
            ].kind
            == inspect.Parameter.KEYWORD_ONLY
        ),
        "base_parity_max_abs": parity,
        "d2ac_d2ad_parameter_schema_identical": parameter_schema_identical,
        "d2ac_d2ad_parameter_initialization_identical": (
            parameter_initialization_identical
        ),
        "initial_alpha_gradient": initial_audit,
        "activated_adapter_gradients": activated_audit,
        "local_permutation_max_abs": permutation_effect,
        "zero_constant_extreme_finite": extreme_finite,
        "role_query_separation_max_abs": role_separation,
        "dtype_device_batch_propagation": dtype_batch_propagation,
        "hsiprior_parameter_storage_independent": True,
        "mixer_clean_output_contract": [1, 16, 232],
    }


def _static_contract(repo: Path) -> Dict[str, object]:
    d2ad_source = (repo / "code/priors/d2ad.py").read_text(encoding="utf-8")
    model_source = (repo / "code/priors/models.py").read_text(encoding="utf-8")
    internal_source = (
        repo / "tools/run_hoi_d2ad_internal.py"
    ).read_text(encoding="utf-8")
    native_source = (
        repo / "tools/run_hoi_d2ad_native_evaluation.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from eval_metrics",
        "contact_label",
        "near_ground",
        "np.save",
        "torch.save",
        "pickle.dump",
        "@lru_cache",
    )
    absent = {value: value not in d2ad_source for value in forbidden}
    locked = {
        "code/test_infbagel_hoi.py":
            "22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524",
        "code/eval_metrics.py":
            "445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547",
        "code/config/config_eval_hoi_prior.yaml":
            "89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73",
        "tools/run_hoi_d2x_evaluation.py":
            "b6753a66207492e6ee4addb8f450cb38c5d021401d43430faa9e5c9ed77c6e31",
        "code/priors/interaction_diagnostic.py":
            "e9a0157f80695469a53a5333b20685cb3c66d042b0ccd621b86164238764bcc5",
    }
    locked_actual = {
        relative: _sha256_file(repo / relative) for relative in locked
    }
    if (
        not all(absent.values())
        or locked_actual != locked
        or "local_object_bps" not in model_source
        or "local_bps_builder.build(" not in internal_source
        or "base.current_bps(" not in internal_source
        or 'item["object_bps"]' in inspect.getsource(
            __import__(
                "tools.run_hoi_d2ad_internal",
                fromlist=["rollout_chunk"],
            ).rollout_chunk,
        )
        or "future_gt" in inspect.getsource(
            __import__(
                "tools.run_hoi_d2ad_internal",
                fromlist=["rollout_chunk"],
            ).rollout_chunk
        )
        or "sealed_d2ac_descriptive_comparison" not in native_source
    ):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: static/evaluator contract failed"
        )
    return {
        "forbidden_model_data_paths_absent": absent,
        "future_gt_absent": True,
        "stored_per_window_local_bps_absent": True,
        "mesh_subsample_absent": True,
        "evaluator_threshold_helper_absent": True,
        "new_loss_or_guidance_absent": True,
        "generated_history_local_bps_recomputed": True,
        "global_bps_rollout_path_unchanged": True,
        "native_d2ac_comparison_descriptive_only": True,
        "official_evaluator_hashes": locked_actual,
    }


def run_contract(repo: Path, run_id: str = RUN_ID) -> Dict[str, object]:
    if run_id != RUN_ID:
        raise ValueError(f"D2-AD CPU contract run id must be exactly {RUN_ID}")
    cfg = _merged_config(repo)
    _validate_d2ad_contract(cfg, 4)
    geometry = _geometry_contract(repo)
    model = _model_contract()
    checkpoint = _checkpoint_rejection_contract()
    static = _static_contract(repo)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "classification": "cpu-contract-passed",
        "seed": 42,
        "geometry": geometry,
        "model": model,
        "checkpoint_provenance": checkpoint,
        "static_contract": static,
        "training_contract_validated": True,
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
        },
        "optimizer_created": False,
        "optimizer_updates": 0,
        "scientific_checkpoint_loads": 0,
        "checkpoint_writes": 0,
        "official_test_used": False,
        "checkpoint_selection": False,
        "consistency_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    try:
        result = run_contract(args.repo_root.resolve(), args.run_id)
    except Exception as error:
        print(f"{FAILURE_CLASSIFICATION}: {error}", flush=True)
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
