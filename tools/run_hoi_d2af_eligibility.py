#!/usr/bin/env python3
"""Run the no-checkpoint D2-AF0 clean-signal premise eligibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion  # noqa: E402
from priors.diffusion_schedule import (  # noqa: E402
    SQRT_ALPHA_BAR_SHA256,
    diffusion_schedule_contract_metadata,
)
from priors.sparse_relation import (  # noqa: E402
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    build_sparse_relation_geometry,
)
from tools.diagnose_hoi_d2af import (  # noqa: E402
    EXPECTED_AUTHORITY_PYTHON,
    EXPECTED_INITIAL_STATE_SHA256,
    authority_identity,
    exclusive_json,
    exclusive_text,
    sha256_file,
)
from train_hoi_prior import _d2af_formal_source_contract  # noqa: E402


RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-clean-signal-eligibility"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
CONTRACT_FAILURE = "clean-signal-contract-failure-stop"
PREMISE_FAILURE = "clean-signal-premise-negative-stop"
PASS_CLASSIFICATION = "clean-signal-premise-passed"
PARTITION = "internal_validation"
EXPECTED_SEQUENCES = 216
EXPECTED_WINDOWS = 29_382
EXPECTED_GLOBAL_INDICES_SHA256 = (
    "eab0bde2dc2ddad7ce2cc1817973ca46b9adaf24b1c906307f865930aeb11eb9"
)
EXPECTED_SEQUENCE_NAMES_SHA256 = (
    "472768c85c6d6c5b682a31a4d40a879d7a1e3d0b16085923c153db1045223fd8"
)
EXPECTED_SPLIT_SHA256 = (
    "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e"
)
BATCH_SIZE = 128
NUM_WORKERS = 0
TIMESTEPS = (0, 249, 499)
SEED = 42
NOISE_SEED_STRIDE = 1_000_003
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
ANCHOR0_TOLERANCE = 1.0e-6


def validate_actual_run_id(run_id: str) -> str:
    match = RUN_ID_RE.fullmatch(str(run_id))
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if match is None or match.group("date") != actual_date:
        raise ValueError(
            "D2-AF eligibility run id must use the locked stem and actual date"
        )
    return match.group("date")


def newline_sha256(values: Sequence[object]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def paired_bootstrap(
    difference: Sequence[float],
    *,
    sampled_indices: np.ndarray | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, object]:
    values = np.asarray(difference, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires finite one-dimensional values")
    if sampled_indices is None:
        rng = np.random.default_rng(seed)
        sampled_indices = rng.integers(
            0,
            len(values),
            size=(replicates, len(values)),
            dtype=np.int64,
        )
    if sampled_indices.shape != (replicates, len(values)):
        raise ValueError("paired bootstrap sampled-index shape mismatch")
    means = values[sampled_indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "sequence_count": int(len(values)),
        "paired_mean": float(values.mean()),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "ci_lower_gt_zero": bool(lower > 0.0),
    }


def mutable_anchor_corruption(
    noisy_features: torch.Tensor,
    clean_features: torch.Tensor,
) -> torch.Tensor:
    expected_tail = (4, 3, 100, 4)
    if (
        noisy_features.shape != clean_features.shape
        or noisy_features.ndim != 5
        or tuple(noisy_features.shape[1:]) != expected_tail
    ):
        raise ValueError(
            "D2-AF eligibility expects relation features [B,4,3,100,4]"
        )
    difference = noisy_features[:, 1:] - clean_features[:, 1:]
    corruption = difference.square().flatten(1).mean(dim=1).sqrt()
    if not bool(torch.isfinite(corruption).all()):
        raise FloatingPointError("non-finite D2-AF relation corruption")
    return corruption


def selection_contract(dataset: PriorWindowDataset) -> Dict[str, object]:
    indices = np.asarray(dataset.indices, dtype=np.int64)
    if len(indices) != EXPECTED_WINDOWS:
        raise ValueError(
            f"internal-validation windows differ: {len(indices)}"
        )
    if not bool(np.all(indices[1:] > indices[:-1])):
        raise ValueError("internal-validation global indices are not canonical")
    sequence_ids = np.asarray(
        dataset.sequence_ids[indices],
        dtype=np.int64,
    )
    unique_sequence_ids = sorted(set(sequence_ids.tolist()))
    sequence_pairs = sorted(
        (
            str(dataset.scene_names[sequence_id]),
            int(sequence_id),
        )
        for sequence_id in unique_sequence_ids
    )
    sequence_names = [name for name, _ in sequence_pairs]
    if (
        len(sequence_pairs) != EXPECTED_SEQUENCES
        or len(set(sequence_names)) != EXPECTED_SEQUENCES
    ):
        raise ValueError("internal-validation sequence set differs")
    global_hash = newline_sha256(indices.tolist())
    sequence_hash = newline_sha256(sequence_names)
    if (
        global_hash != EXPECTED_GLOBAL_INDICES_SHA256
        or sequence_hash != EXPECTED_SEQUENCE_NAMES_SHA256
    ):
        raise ValueError(
            "internal-validation canonical selection hash mismatch"
        )
    return {
        "partition": PARTITION,
        "sequences": EXPECTED_SEQUENCES,
        "windows": EXPECTED_WINDOWS,
        "global_indices_sha256": global_hash,
        "global_indices_hash_algorithm":
            "newline-terminated-base10-global-index-v1",
        "sequence_names_sha256": sequence_hash,
        "sequence_names_hash_algorithm":
            "sorted-newline-terminated-utf8-sequence-name-v1",
        "canonical_shuffle": False,
        "drop_last": False,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "sequence_pairs": sequence_pairs,
    }


def _normalization(dataset: PriorWindowDataset) -> Dict[str, torch.Tensor]:
    return {
        "position_minimum": torch.as_tensor(
            dataset.minimum,
            dtype=torch.float32,
            device="cpu",
        ),
        "position_maximum": torch.as_tensor(
            dataset.maximum,
            dtype=torch.float32,
            device="cpu",
        ),
        "object_minimum": torch.as_tensor(
            dataset.object_minimum,
            dtype=torch.float32,
            device="cpu",
        ),
        "object_maximum": torch.as_tensor(
            dataset.object_maximum,
            dtype=torch.float32,
            device="cpu",
        ),
    }


def _geometry(
    state: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    normalization: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return build_sparse_relation_geometry(
        state,
        batch["rest_object_points"],
        batch["world_to_local_rotation"],
        batch["object_rotation_reference"],
        normalization["position_minimum"],
        normalization["position_maximum"],
        normalization["object_minimum"],
        normalization["object_maximum"],
    )


def _validated_prerequisite(
    *,
    repo: Path,
    label: str,
    path: Path,
    expected_sha256: str,
    expected_classification: str,
    expected_status: str,
    current_commit: str,
    formal_source_contract: Mapping[str, object],
) -> Dict[str, object]:
    path = path.resolve()
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} path must be an existing absolute file")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError(f"{label} expected SHA-256 is malformed")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    identity = value.get("identity")
    source = value.get("formal_source_contract")
    resolved_path_value = value.get("resolved_config_path")
    resolved_path = (
        Path(resolved_path_value).resolve()
        if isinstance(resolved_path_value, str)
        and Path(resolved_path_value).is_absolute()
        else None
    )
    resolved_sha256 = str(value.get("resolved_config_sha256", ""))
    prerequisite_commit = (
        identity.get("git_commit")
        if isinstance(identity, Mapping) else None
    )
    commit_is_ancestor = bool(
        isinstance(prerequisite_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", prerequisite_commit)
        and subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                prerequisite_commit,
                current_commit,
            ],
            cwd=repo,
            check=False,
        ).returncode == 0
    )
    checks = {
        "schema_version": value.get("schema_version") == 1,
        "status": value.get("status") == expected_status,
        "classification": (
            value.get("classification") == expected_classification
        ),
        "identity": (
            isinstance(identity, Mapping)
            and commit_is_ancestor
            and identity.get("worktree_clean") is True
        ),
        "formal_source_contract": source == formal_source_contract,
        "resolved_config": (
            resolved_path is not None
            and resolved_path.is_file()
            and re.fullmatch(r"[0-9a-f]{64}", resolved_sha256) is not None
            and sha256_file(resolved_path) == resolved_sha256
        ),
        "checkpoint_loads": value.get(
            "scientific_checkpoint_loads",
            value.get("checkpoint_loads"),
        ) == 0,
        "optimizer_created": value.get("optimizer_created") is False,
        "optimizer_updates": value.get("optimizer_updates") == 0,
        "checkpoint_writes": value.get("checkpoint_writes") == 0,
    }
    if label == "functional_smoke":
        schedule = value.get("schedule")
        checks.update({
            "optimizer_updates": value.get("optimizer_updates") == 0,
            "initialization": value.get("initialization") == "random",
            "initial_model_hash": (
                value.get("initial_model_state_sha256")
                == EXPECTED_INITIAL_STATE_SHA256
            ),
            "schedule": (
                isinstance(schedule, Mapping)
                and schedule.get("sqrt_alpha_bar_sha256")
                == SQRT_ALPHA_BAR_SHA256
            ),
        })
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"{label} prerequisite contract mismatch: {failed}"
        )
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "run_id": str(value.get("run_id")),
        "status": str(value.get("status")),
        "classification": str(value.get("classification")),
        "git_commit": str(prerequisite_commit),
        "git_commit_is_current_ancestor": commit_is_ancestor,
        "formal_source_contract": dict(source),
        "resolved_config_path": str(resolved_path),
        "resolved_config_sha256": resolved_sha256,
        "checks": checks,
    }


def resolved_workload_config(
    *,
    repo: Path,
    run_id: str,
    output: Path,
    resolved_config_output: Path,
    identity: Mapping[str, object],
    formal_source_contract: Mapping[str, object],
    cpu_path: Path,
    cpu_sha256: str,
    smoke_path: Path,
    smoke_sha256: str,
) -> str:
    split_path = (
        repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    ).resolve()
    value = {
        "schema_version": 1,
        "lifecycle": "d2af_no_checkpoint_clean_signal_eligibility",
        "run_id": run_id,
        "repo_root": str(repo),
        "git_commit": identity["git_commit"],
        "python": str(Path(sys.executable).resolve()),
        "output": str(output.resolve()),
        "resolved_config_output": str(resolved_config_output.resolve()),
        "formal_source_contract": dict(formal_source_contract),
        "prerequisites": {
            "authority_cpu_contract_path": str(cpu_path.resolve()),
            "authority_cpu_contract_sha256": cpu_sha256,
            "functional_smoke_path": str(smoke_path.resolve()),
            "functional_smoke_sha256": smoke_sha256,
        },
        "data": {
            "partition": PARTITION,
            "split_manifest": str(split_path),
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "sequences": EXPECTED_SEQUENCES,
            "windows": EXPECTED_WINDOWS,
            "global_indices_sha256": EXPECTED_GLOBAL_INDICES_SHA256,
            "sequence_names_sha256": EXPECTED_SEQUENCE_NAMES_SHA256,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "shuffle": False,
            "drop_last": False,
        },
        "diagnostic": {
            "timesteps": list(TIMESTEPS),
            "noise": {
                str(timestep): {
                    "device": "cpu",
                    "dtype": "torch.float32",
                    "generator": "torch.Generator(device='cpu')",
                    "seed": SEED + NOISE_SEED_STRIDE * timestep,
                }
                for timestep in TIMESTEPS
            },
            "feature_source":
                "pure-PyTorch pre-encoder sparse relation geometry",
            "mutable_anchors": [5, 10, 15],
            "immutable_history_anchor": 0,
            "corruption":
                "sqrt(mean((feature_d-feature_clean)^2)) per window",
            "sequence_reduction": "mean",
            "bootstrap_unit": "sequence",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "anchor0_tolerance": ANCHOR0_TOLERANCE,
        },
        "provenance": {
            "checkpoint_loads": 0,
            "model_created": False,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "official_test_used": False,
            "downstream_metric_selection": False,
        },
    }
    resolved = OmegaConf.to_yaml(OmegaConf.create(value), resolve=True)
    if "${" in resolved:
        raise RuntimeError("D2-AF eligibility resolved config is incomplete")
    return resolved


def archive_or_validate_resolved_config(
    path: Path,
    resolved: str,
    *,
    resolve_only: bool,
) -> str:
    path = path.resolve()
    if resolve_only:
        exclusive_text(path, resolved)
    else:
        if not path.is_file():
            raise FileNotFoundError(
                "D2-AF eligibility requires a pre-archived resolved config"
            )
        if path.read_text(encoding="utf-8") != resolved:
            raise RuntimeError(
                "D2-AF eligibility workload differs from archived config"
            )
    return sha256_file(path)


def run_eligibility(
    repo: Path,
    run_id: str,
    *,
    cpu_contract_path: Path,
    cpu_contract_sha256: str,
    functional_smoke_path: Path,
    functional_smoke_sha256: str,
    require_clean: bool = True,
) -> Dict[str, object]:
    started = time.perf_counter()
    repo = repo.resolve()
    validate_actual_run_id(run_id)
    identity = authority_identity(repo, require_clean=require_clean)
    formal_source = _d2af_formal_source_contract(repo)
    current_commit = str(identity["git_commit"])
    cpu_binding = _validated_prerequisite(
        repo=repo,
        label="authority_cpu_contract",
        path=cpu_contract_path,
        expected_sha256=cpu_contract_sha256,
        expected_classification="cpu-contract-passed",
        expected_status="passed",
        current_commit=current_commit,
        formal_source_contract=formal_source,
    )
    smoke_binding = _validated_prerequisite(
        repo=repo,
        label="functional_smoke",
        path=functional_smoke_path,
        expected_sha256=functional_smoke_sha256,
        expected_classification="functional-smoke-passed",
        expected_status="stable",
        current_commit=current_commit,
        formal_source_contract=formal_source,
    )
    prerequisite_source_match = (
        cpu_binding["formal_source_contract"]
        == smoke_binding["formal_source_contract"]
        == formal_source
    )
    if not prerequisite_source_match:
        raise ValueError("D2-AF prerequisite source contracts differ")

    split_path = (
        repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    ).resolve()
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise ValueError("D2-AF eligibility split SHA-256 mismatch")
    dataset = PriorWindowDataset(
        str(repo),
        "hoi",
        partition=PARTITION,
        split_manifest=str(split_path),
    )
    selection = selection_contract(dataset)
    sequence_pairs = selection.pop("sequence_pairs")
    sequence_names = [name for name, _ in sequence_pairs]
    sequence_position = {
        int(sequence_id): position
        for position, (_, sequence_id) in enumerate(sequence_pairs)
    }
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    normalization = _normalization(dataset)
    diffusion = GaussianDiffusion().cpu().eval()
    generators = {
        timestep: torch.Generator(device="cpu").manual_seed(
            SEED + NOISE_SEED_STRIDE * timestep
        )
        for timestep in TIMESTEPS
    }
    noise_digests = {
        timestep: hashlib.sha256() for timestep in TIMESTEPS
    }
    sequence_sums = {
        timestep: np.zeros(EXPECTED_SEQUENCES, dtype=np.float64)
        for timestep in TIMESTEPS
    }
    sequence_counts = np.zeros(EXPECTED_SEQUENCES, dtype=np.int64)
    anchor0_clean_max_abs = {timestep: 0.0 for timestep in TIMESTEPS}
    anchor0_cross_timestep_max_abs = 0.0
    history_exact = {timestep: True for timestep in TIMESTEPS}
    windows_processed = 0
    relation_feature_shape = None

    with torch.no_grad():
        for batch in loader:
            clean = batch["x"]
            if (
                clean.device.type != "cpu"
                or clean.dtype != torch.float32
                or tuple(clean.shape[1:]) != (16, 232)
            ):
                raise ValueError("D2-AF eligibility clean-state contract failed")
            if "local_object_bps" in batch:
                raise ValueError(
                    "D2-AF eligibility received CPU dynamic geometry"
                )
            clean_geometry = _geometry(clean, batch, normalization)
            clean_features = clean_geometry["features"]
            relation_feature_shape = list(clean_features.shape[1:])
            sequence_ids = batch["sequence_index"].numpy().astype(
                np.int64,
                copy=False,
            )
            try:
                positions = np.asarray(
                    [sequence_position[int(value)] for value in sequence_ids],
                    dtype=np.int64,
                )
            except KeyError as error:
                raise ValueError(
                    "D2-AF eligibility batch contains an unregistered sequence"
                ) from error
            np.add.at(sequence_counts, positions, 1)

            anchor0_by_timestep = {}
            for timestep in TIMESTEPS:
                noise = torch.randn(
                    clean.shape,
                    generator=generators[timestep],
                    dtype=torch.float32,
                    device="cpu",
                )
                noise_digests[timestep].update(
                    noise.contiguous().numpy().tobytes()
                )
                timesteps = torch.full(
                    (clean.shape[0],),
                    timestep,
                    dtype=torch.long,
                    device="cpu",
                )
                noisy = diffusion.q_sample(clean, timesteps, noise)
                history_exact[timestep] = bool(
                    history_exact[timestep]
                    and torch.equal(noisy[:, :2], clean[:, :2])
                )
                noisy_features = _geometry(
                    noisy,
                    batch,
                    normalization,
                )["features"]
                corruption = mutable_anchor_corruption(
                    noisy_features,
                    clean_features,
                )
                np.add.at(
                    sequence_sums[timestep],
                    positions,
                    corruption.numpy().astype(np.float64, copy=False),
                )
                anchor0 = noisy_features[:, 0]
                anchor0_by_timestep[timestep] = anchor0
                anchor0_clean_max_abs[timestep] = max(
                    anchor0_clean_max_abs[timestep],
                    float((anchor0 - clean_features[:, 0]).abs().max()),
                )
            for left_index, left in enumerate(TIMESTEPS):
                for right in TIMESTEPS[left_index + 1:]:
                    anchor0_cross_timestep_max_abs = max(
                        anchor0_cross_timestep_max_abs,
                        float(
                            (
                                anchor0_by_timestep[left]
                                - anchor0_by_timestep[right]
                            ).abs().max()
                        ),
                    )
            windows_processed += int(clean.shape[0])

    if (
        windows_processed != EXPECTED_WINDOWS
        or int(sequence_counts.sum()) != EXPECTED_WINDOWS
        or bool((sequence_counts <= 0).any())
        or not all(history_exact.values())
    ):
        raise ValueError("D2-AF eligibility full-data traversal failed")
    sequence_means = {
        timestep: sequence_sums[timestep] / sequence_counts
        for timestep in TIMESTEPS
    }
    if not all(
        np.isfinite(values).all()
        for values in sequence_means.values()
    ):
        raise FloatingPointError(
            "D2-AF eligibility sequence corruption is non-finite"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_indices = rng.integers(
        0,
        EXPECTED_SEQUENCES,
        size=(BOOTSTRAP_REPLICATES, EXPECTED_SEQUENCES),
        dtype=np.int64,
    )
    bootstrap_indices_sha256 = hashlib.sha256(
        sampled_indices.tobytes()
    ).hexdigest()
    difference_249_0 = sequence_means[249] - sequence_means[0]
    difference_499_249 = sequence_means[499] - sequence_means[249]
    comparison_249_0 = paired_bootstrap(
        difference_249_0,
        sampled_indices=sampled_indices,
    )
    comparison_499_249 = paired_bootstrap(
        difference_499_249,
        sampled_indices=sampled_indices,
    )
    gates = {
        "c249_minus_c0_ci_lower_gt_zero":
            comparison_249_0["ci_lower_gt_zero"],
        "c499_minus_c249_ci_lower_gt_zero":
            comparison_499_249["ci_lower_gt_zero"],
        "anchor0_prescaling_max_abs_le_1e_minus_6": (
            anchor0_cross_timestep_max_abs <= ANCHOR0_TOLERANCE
            and max(anchor0_clean_max_abs.values()) <= ANCHOR0_TOLERANCE
        ),
    }
    premise_passed = all(gates.values())
    per_sequence = [
        {
            "sequence": sequence_names[position],
            "sequence_index": int(sequence_pairs[position][1]),
            "windows": int(sequence_counts[position]),
            "corruption": {
                str(timestep): float(sequence_means[timestep][position])
                for timestep in TIMESTEPS
            },
            "differences": {
                "c249_minus_c0": float(difference_249_0[position]),
                "c499_minus_c249": float(difference_499_249[position]),
            },
        }
        for position in range(EXPECTED_SEQUENCES)
    ]
    return {
        "schema_version": 1,
        "status": "passed" if premise_passed else "failed",
        "classification": (
            PASS_CLASSIFICATION if premise_passed else PREMISE_FAILURE
        ),
        "run_id": run_id,
        "subphase": "1B-D2-AF0-clean-signal-eligibility",
        "seed": SEED,
        "runtime_seconds": time.perf_counter() - started,
        "identity": identity,
        "formal_source_contract": formal_source,
        "authority_cpu_contract": cpu_binding,
        "functional_smoke": smoke_binding,
        "prerequisite_source_contract_match": prerequisite_source_match,
        "selection": selection,
        "schedule": diffusion_schedule_contract_metadata(),
        "noise_streams": {
            str(timestep): {
                "seed": SEED + NOISE_SEED_STRIDE * timestep,
                "device": "cpu",
                "dtype": "torch.float32",
                "shape_per_window": [16, 232],
                "values": EXPECTED_WINDOWS * 16 * 232,
                "sha256": noise_digests[timestep].hexdigest(),
                "hash_algorithm":
                    "canonical-batch128-concatenated-raw-float32-bytes-v1",
            }
            for timestep in TIMESTEPS
        },
        "corruption": {
            "feature_shape_per_window": relation_feature_shape,
            "pre_encoder": True,
            "roles": ["left_hand", "right_hand", "pelvis"],
            "points": 100,
            "components": [
                "delta_x",
                "delta_y",
                "delta_z",
                "distance",
            ],
            "mutable_anchor_slots": [1, 2, 3],
            "mutable_anchor_frames": [5, 10, 15],
            "definition":
                "sqrt(mean((feature_d-feature_clean_x0)^2)) per window",
            "sequence_reduction": "mean",
            "sequence_means_by_timestep": {
                str(timestep): float(sequence_means[timestep].mean())
                for timestep in TIMESTEPS
            },
            "comparisons": {
                "c249_minus_c0": comparison_249_0,
                "c499_minus_c249": comparison_499_249,
            },
            "bootstrap_sample_indices_sha256":
                bootstrap_indices_sha256,
            "per_sequence": per_sequence,
        },
        "anchor0_prescaling": {
            "frame": 0,
            "immutable_history": True,
            "history_exact_by_timestep": {
                str(key): value for key, value in history_exact.items()
            },
            "max_abs_noisy_minus_clean_by_timestep": {
                str(key): value
                for key, value in anchor0_clean_max_abs.items()
            },
            "cross_timestep_max_abs": anchor0_cross_timestep_max_abs,
            "maximum_allowed": ANCHOR0_TOLERANCE,
        },
        "gates": gates,
        "formal_training_authorized": premise_passed,
        "performance_benchmark_authorized": premise_passed,
        "checkpoint_loads": 0,
        "model_created": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "official_test_used": False,
        "downstream_metric_selection": False,
        "old_checkpoint_selection": False,
        "sparse_assets": {
            "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
            "manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
            "stacked_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
        },
    }


def failure_record(
    *,
    run_id: str,
    started: float,
    error: Exception,
    repo: Path,
) -> Dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
        ).strip()
    except Exception:
        commit = None
    return {
        "schema_version": 1,
        "status": "failed",
        "classification": CONTRACT_FAILURE,
        "run_id": run_id,
        "subphase": "1B-D2-AF0-clean-signal-eligibility",
        "seed": SEED,
        "git_commit": commit,
        "runtime_seconds": time.perf_counter() - started,
        "failure_type": type(error).__name__,
        "failure": str(error),
        "formal_training_authorized": False,
        "performance_benchmark_authorized": False,
        "checkpoint_loads": 0,
        "model_created": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "official_test_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-config-output", type=Path, required=True)
    parser.add_argument(
        "--authority-cpu-contract-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--authority-cpu-contract-sha256",
        required=True,
    )
    parser.add_argument(
        "--functional-smoke-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--functional-smoke-sha256",
        required=True,
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help=(
            "archive and validate the exact eligibility config without "
            "loading data"
        ),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    repo = args.repo_root.resolve()
    try:
        validate_actual_run_id(args.run_id)
        identity = authority_identity(repo, require_clean=True)
        formal_source = _d2af_formal_source_contract(repo)
        resolved = resolved_workload_config(
            repo=repo,
            run_id=args.run_id,
            output=args.output,
            resolved_config_output=args.resolved_config_output,
            identity=identity,
            formal_source_contract=formal_source,
            cpu_path=args.authority_cpu_contract_path,
            cpu_sha256=args.authority_cpu_contract_sha256,
            smoke_path=args.functional_smoke_path,
            smoke_sha256=args.functional_smoke_sha256,
        )
        resolved_sha256 = archive_or_validate_resolved_config(
            args.resolved_config_output,
            resolved,
            resolve_only=args.resolve_only,
        )
        if args.resolve_only:
            value = {
                "schema_version": 1,
                "status": "resolved-config-archived",
                "run_id": args.run_id,
                "resolved_config_path": str(
                    args.resolved_config_output.resolve()
                ),
                "resolved_config_sha256": resolved_sha256,
                "eligibility_workload_started": False,
                "checkpoint_loads": 0,
                "model_created": False,
                "optimizer_created": False,
            }
            print(json.dumps(value, indent=2, sort_keys=True), flush=True)
            return 0
        result = run_eligibility(
            repo,
            args.run_id,
            cpu_contract_path=args.authority_cpu_contract_path,
            cpu_contract_sha256=args.authority_cpu_contract_sha256,
            functional_smoke_path=args.functional_smoke_path,
            functional_smoke_sha256=args.functional_smoke_sha256,
            require_clean=True,
        )
        result["resolved_config_path"] = str(
            args.resolved_config_output.resolve()
        )
        result["resolved_config_sha256"] = resolved_sha256
        result["resolved_config_has_unresolved_interpolation"] = False
        exclusive_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        failure = failure_record(
            run_id=args.run_id,
            started=started,
            error=error,
            repo=repo,
        )
        failure_path = args.output.resolve().parent / "failure.json"
        if not failure_path.exists():
            exclusive_json(failure_path, failure)
        if not args.output.resolve().exists():
            exclusive_json(args.output, failure)
        print(f"{CONTRACT_FAILURE}: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
