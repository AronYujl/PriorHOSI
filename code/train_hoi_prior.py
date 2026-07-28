"""Phase 1B scene-free HOIPrior DDP training, validation, checkpoint and resume."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.utils import get_smpl_parents
from priors.data import PriorWindowDataset
from priors.d2ab import D2ABPriorWindowDataset, d2ab_hoi_training_losses
from priors.d2ad import (
    BPS_YUP_TENSOR_SHA256,
    DEFAULT_QUERY_WORKERS,
    OBJECT_MAPPING_SHA256,
    REST_MESH_MANIFEST_SHA256,
    D2ADBatchCollator,
    D2ADPriorWindowDataset,
    LocalObjectBPSBuilder,
)
from priors.d2z import D2ZPriorWindowDataset, d2z_hoi_training_losses
from priors.diffusion import GaussianDiffusion, normalize_progress
from priors.interaction_adapter import (
    ADAPTER_PARAMETER_COUNT,
    ASSIGNMENT_SHA256,
    BPS_SHA256 as D2AC_BPS_SHA256,
    LOCAL_BASIS_COORDINATE_SYSTEM,
)
from priors.losses import hoi_training_losses
from priors.models import (
    HOI_ARCHITECTURE_BASE,
    HOI_ARCHITECTURE_D2AC,
    HOI_ARCHITECTURE_D2AD,
    HOI_ARCHITECTURE_D2AE,
    build_expert,
)
from priors.representation import REPRESENTATION
from priors.sparse_relation import (
    BASE_PARAMETER_COUNT as D2AE_BASE_PARAMETER_COUNT,
    PARAMETER_INCREASE_FRACTION as D2AE_PARAMETER_INCREASE_FRACTION,
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
    TOTAL_PARAMETER_COUNT as D2AE_TOTAL_PARAMETER_COUNT,
    validate_sparse_relation_contract,
)
from priors.window_codec import BPS_SHA256


LOSS_KEYS = (
    "total", "reconstruction", "joint_position", "joint_rotation", "object_translation",
    "object_rotation", "contact", "fk", "object_surface", "velocity", "object_goal", "contact_accuracy",
)

D2AE_FORMAL_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ae-sparse-relation-field"
    r"(?P<retry>-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
D2AE_PERFORMANCE_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ae-performance-benchmark"
    r"(?P<retry>-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
D2AE_MINIMUM_THROUGHPUT = 2756.580356467847
D2AE_MAXIMUM_ETA_HOURS = 6.20
D2AE_FORMAL_SOURCE_SCOPES = (
    "code",
    "tools/benchmark_hoi_d2ae.py",
    "tools/smoke_hoi_d2ae.py",
    "tools/smoke_hoi_d2ac.py",
    "tools/capture_hoi_worker_preflight.py",
    "tools/experiment.py",
)


def _validate_d2ae_formal_run_id(
    run_id: str,
    *,
    require_actual_date: bool = True,
) -> Dict[str, object]:
    match = D2AE_FORMAL_RUN_ID_RE.fullmatch(str(run_id))
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if match is None or (
        require_actual_date and match.group("date") != actual_date
    ):
        raise ValueError("D2-AE formal run id must use the locked stem and actual date")
    return {
        "run_id": str(run_id),
        "date": match.group("date"),
        "date_is_actual": match.group("date") == actual_date,
        "retry": match.group("retry") is not None,
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as value:
        value.bind(("", 0))
        return int(value.getsockname()[1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_weight_initialization(
    cfg: DictConfig,
    model: torch.nn.Module,
) -> Dict[str, object]:
    """Load only the hash-locked online model weights allowed by D2-M0."""
    path_value = cfg.weight_init_checkpoint
    if path_value in (None, "", False):
        return {
            "mode": "random",
            "source_checkpoint": None,
            "source_checkpoint_sha256": None,
            "source_model_state_sha256": None,
            "initial_model_state_sha256": _state_dict_sha256(model.state_dict()),
            "restored_components": [],
            "old_optimizer_states_loaded": 0,
            "old_ema_models_loaded": 0,
            "old_scheduler_states_loaded": 0,
            "old_scaler_states_loaded": 0,
            "old_rng_states_loaded": 0,
        }
    from priors.optimizer_reset import (
        CANDIDATES,
        SOURCE_CHECKPOINT_SHA256,
        SOURCE_RUN_ID,
    )

    if str(cfg.d2m_candidate) not in CANDIDATES:
        raise ValueError("weight-only initialization is restricted to a registered D2-M candidate")
    if str(cfg.weight_init_variant) != "online":
        raise ValueError("D2-M weight-only initialization accepts online model weights only")
    if str(cfg.weight_init_sha256) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("D2-M source checkpoint configured SHA-256 mismatch")
    path = Path(str(path_value)).resolve()
    actual_sha256 = _sha256(path)
    if actual_sha256 != SOURCE_CHECKPOINT_SHA256:
        raise ValueError(f"D2-M source checkpoint file hash mismatch: {actual_sha256}")
    checkpoint = torch.load(path, map_location="cpu")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_type") != "hoi_prior_phase1b"
        or checkpoint.get("expert") != "hoi"
        or checkpoint.get("initialization") != "random"
    ):
        raise ValueError(
            "D2-M weight initialization requires a self-trained random-origin Phase 1B checkpoint; "
            "released checkpoint initialization is forbidden"
        )
    if checkpoint.get("run_id") != SOURCE_RUN_ID:
        raise ValueError(f"D2-M source run mismatch: {checkpoint.get('run_id')}")
    if checkpoint.get("model_config") != _model_config(cfg):
        raise ValueError("D2-M source model configuration mismatch")
    if checkpoint.get("data_contract_sha256") != str(cfg.data_contract_sha256):
        raise ValueError("D2-M source data contract mismatch")
    split_sha256 = _sha256(Path(str(cfg.split_manifest)).resolve())
    if checkpoint.get("split_sha256") != split_sha256:
        raise ValueError("D2-M source split mismatch")
    source_model = checkpoint.get("model")
    if not isinstance(source_model, dict):
        raise ValueError("D2-M source checkpoint is missing online model weights")
    source_model_sha256 = _state_dict_sha256(source_model)
    model.load_state_dict(source_model, strict=True)
    initial_model_sha256 = _state_dict_sha256(model.state_dict())
    if initial_model_sha256 != source_model_sha256:
        raise ValueError("D2-M source online model did not load exactly")
    return {
        "mode": "phase1b_online_weight_only",
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": actual_sha256,
        "source_run_id": checkpoint.get("run_id"),
        "source_git_commit": checkpoint.get("git_commit"),
        "source_processed_windows": checkpoint.get("processed_windows"),
        "source_optimizer_updates": checkpoint.get("optimizer_updates"),
        "source_model_state_sha256": source_model_sha256,
        "initial_model_state_sha256": initial_model_sha256,
        "restored_components": ["model"],
        "old_optimizer_states_loaded": 0,
        "old_ema_models_loaded": 0,
        "old_scheduler_states_loaded": 0,
        "old_scaler_states_loaded": 0,
        "old_rng_states_loaded": 0,
    }


def _atomic_json(path: Path, value: Dict[str, object], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _d2ae_formal_source_contract(repo: Path) -> Dict[str, object]:
    """Hash the tracked runtime tree shared by the benchmark and formal run."""
    repo = repo.resolve()
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *D2AE_FORMAL_SOURCE_SCOPES],
        cwd=repo,
    )
    relative_paths = sorted(
        item.decode("utf-8") for item in output.split(b"\0") if item
    )
    if not relative_paths:
        raise ValueError("D2-AE formal source contract resolved no tracked files")
    records = [
        {"path": relative, "sha256": _sha256(repo / relative)}
        for relative in relative_paths
    ]
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": "git-ls-files-path-content-sha256-v1",
        "scopes": list(D2AE_FORMAL_SOURCE_SCOPES),
        "tracked_file_count": len(records),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validate_d2ae_performance_gate(cfg: DictConfig) -> Dict[str, object]:
    """Require an immutable passing benchmark before formal D2-AE training."""
    path_value = cfg.get("d2ae_performance_benchmark_path")
    configured_sha256 = cfg.get("d2ae_performance_benchmark_sha256")
    if path_value in (None, "", False) or configured_sha256 in (None, "", False):
        raise ValueError("D2-AE formal training requires a sealed performance benchmark")
    path = Path(str(path_value))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("D2-AE performance benchmark path must be an existing absolute file")
    configured_sha256 = str(configured_sha256)
    actual_sha256 = _sha256(path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", configured_sha256) is None
        or actual_sha256 != configured_sha256
    ):
        raise ValueError("D2-AE performance benchmark SHA-256 mismatch")
    try:
        benchmark = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("D2-AE performance benchmark is not valid JSON") from error
    if not isinstance(benchmark, Mapping):
        raise ValueError("D2-AE performance benchmark must be a JSON object")

    repo = Path(str(cfg.repo_root)).resolve()
    identity = benchmark.get("identity")
    benchmark_run_id = str(benchmark.get("run_id", ""))
    benchmark_run_match = D2AE_PERFORMANCE_RUN_ID_RE.fullmatch(
        benchmark_run_id
    )
    formal_run_id = str(benchmark.get("formal_run_id", ""))
    configured_formal_run_id = str(cfg.run_id)
    configured_formal_match = D2AE_FORMAL_RUN_ID_RE.fullmatch(
        configured_formal_run_id
    )
    benchmark_commit = (
        identity.get("git_commit") if isinstance(identity, Mapping) else None
    )
    commit_valid = isinstance(benchmark_commit, str) and re.fullmatch(
        r"[0-9a-f]{40}", benchmark_commit,
    ) is not None
    ancestor = False
    if commit_valid:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", benchmark_commit, _git_commit(repo)],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    throughput = benchmark.get("throughput_windows_per_second")
    eta_hours = benchmark.get("full_budget_eta_hours")
    headroom = benchmark.get("memory_headroom_min_bytes")
    required_headroom = benchmark.get("memory_headroom_required_bytes")
    numeric_metrics = (
        isinstance(throughput, (int, float))
        and not isinstance(throughput, bool)
        and math.isfinite(float(throughput))
        and isinstance(eta_hours, (int, float))
        and not isinstance(eta_hours, bool)
        and math.isfinite(float(eta_hours))
    )
    memory_metrics = (
        isinstance(headroom, int)
        and not isinstance(headroom, bool)
        and isinstance(required_headroom, int)
        and not isinstance(required_headroom, bool)
        and headroom >= required_headroom
        and required_headroom >= 2 * 1024**3
    )
    checks = {
        "schema_version": benchmark.get("schema_version") == 1,
        "status": benchmark.get("status") == "passed",
        "classification": benchmark.get("classification") == "performance-gate-passed",
        "run_id": benchmark_run_match is not None,
        "formal_run_id": configured_formal_match is not None
        and benchmark_run_match is not None
        and formal_run_id == configured_formal_run_id
        and configured_formal_match.group("date")
        == benchmark_run_match.group("date"),
        "formal_authorized": benchmark.get("formal_training_authorized") is True,
        "seed": benchmark.get("seed") == 42,
        "world_size": benchmark.get("world_size") == 4,
        "micro_batch": benchmark.get("micro_batch_per_gpu") == 512,
        "effective_batch": benchmark.get("effective_batch_size") == 2048,
        "warmup_updates": benchmark.get("warmup_updates") == 64,
        "measured_updates": benchmark.get("measured_updates") == 256,
        "total_updates": benchmark.get("total_updates") == 320,
        "measured_windows": benchmark.get("measured_windows") == 524288,
        "numeric_metrics": numeric_metrics,
        "throughput": numeric_metrics
        and float(throughput) >= D2AE_MINIMUM_THROUGHPUT,
        "eta": numeric_metrics and float(eta_hours) <= D2AE_MAXIMUM_ETA_HOURS,
        "throughput_threshold": benchmark.get(
            "minimum_throughput_windows_per_second"
        ) == D2AE_MINIMUM_THROUGHPUT,
        "eta_threshold": benchmark.get("maximum_full_budget_eta_hours")
        == D2AE_MAXIMUM_ETA_HOURS,
        "memory": benchmark.get("memory_headroom_pass") is True
        and memory_metrics,
        "losses": benchmark.get("losses_finite") is True,
        "gradients": benchmark.get("gradients_finite") is True,
        "relation_gpu_only": benchmark.get("relation_gpu_only") is True,
        "all_rank_contract": benchmark.get("all_rank_contract_pass") is True,
        "contention": benchmark.get("contention_pass") is True,
        "cpu_dynamic_geometry": benchmark.get("cpu_dynamic_geometry") is False,
        "relation_build_device": benchmark.get("relation_build_device") == "cuda",
        "cuda_timing": benchmark.get("cuda_timing_synchronized") is True,
        "optimizer": benchmark.get("optimizer") == "FP32 Adam",
        "optimizer_updates": benchmark.get("optimizer_updates") == 320,
        "checkpoint_loads": benchmark.get("checkpoint_loads") == 0,
        "checkpoint_writes": benchmark.get("checkpoint_writes") == 0,
        "benchmark_weights_reusable": benchmark.get("benchmark_weights_reusable") is False,
        "sweep_on_failure": benchmark.get("sweep_authorized_on_failure") is False,
        "identity_mapping": isinstance(identity, Mapping),
        "identity_clean": isinstance(identity, Mapping)
        and identity.get("worktree_clean") is True,
        "benchmark_commit": commit_valid,
        "benchmark_ancestor": ancestor,
        "source_contract": benchmark.get("formal_source_contract")
        == _d2ae_formal_source_contract(repo),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "D2-AE performance benchmark contract mismatch: " + ", ".join(failed)
        )
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "run_id": str(benchmark["run_id"]),
        "formal_run_id": formal_run_id,
        "git_commit": benchmark_commit,
        "throughput_windows_per_second": float(throughput),
        "full_budget_eta_hours": float(eta_hours),
        "memory_headroom_min_bytes": int(headroom),
        "formal_source_contract": dict(benchmark["formal_source_contract"]),
        "checks": checks,
    }


_RESUME_TRANSITION_ALLOWED_PATHS = frozenset(
    {
        "code/config/config_train_hoi_prior.yaml",
        "code/train_hoi_prior.py",
        "docs/EXPERIMENT_PLAN.md",
        "experiments/registry.jsonl",
        "tests/test_hoi_d2ab.py",
    }
)


def _resume_commit_provenance(
    cfg: DictConfig,
    checkpoint_commit: str,
    current_commit: str,
    repo: Path,
) -> Dict[str, object]:
    """Fail-closed provenance check for an explicitly bound governance transition.

    Exact-commit resumes remain the default.  The only exception is a continuation
    config that binds the checkpoint source commit, current target commit, and the
    byte-level Git diff hash.  This lets a paused run survive a documented
    governance/test commit without permitting an arbitrary code transition.
    """
    if checkpoint_commit == current_commit:
        return {
            "mode": "exact_commit",
            "checkpoint_git_commit": checkpoint_commit,
            "current_git_commit": current_commit,
            "changed_paths": [],
            "diff_sha256": None,
        }
    authorized = bool(cfg.get("resume_commit_transition_authorized", False))
    source = cfg.get("resume_source_commit")
    target = cfg.get("resume_target_commit")
    expected_diff = cfg.get("resume_transition_diff_sha256")
    if not authorized:
        raise ValueError(
            "resume checkpoint Git commit mismatch and no explicit transition authorization: "
            f"{checkpoint_commit} != {current_commit}"
        )
    source = None if source in (None, "", False) else str(source)
    target = None if target in (None, "", False) else str(target)
    expected_diff = None if expected_diff in (None, "", False) else str(expected_diff)
    commit_pattern = re.compile(r"^[0-9a-f]{40}$")
    if (
        source is None
        or target is None
        or expected_diff is None
        or not commit_pattern.fullmatch(source)
        or not commit_pattern.fullmatch(target)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_diff)
    ):
        raise ValueError("resume commit transition requires valid source/target/diff hashes")
    if source != checkpoint_commit or target != current_commit:
        raise ValueError(
            "resume commit transition binding does not match checkpoint/current commits: "
            f"source={source}, checkpoint={checkpoint_commit}, "
            f"target={target}, current={current_commit}"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source, target],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("resume commit transition target is not a descendant of checkpoint commit")
    changed_paths = tuple(
        line
        for line in subprocess.check_output(
            ["git", "diff", "--name-only", source, target],
            cwd=str(repo),
            text=True,
        ).splitlines()
        if line
    )
    unexpected = sorted(set(changed_paths) - _RESUME_TRANSITION_ALLOWED_PATHS)
    if unexpected:
        raise ValueError(
            "resume commit transition changes non-governance/source-guard files: "
            + ", ".join(unexpected)
        )
    diff_bytes = subprocess.check_output(
        ["git", "diff", "--binary", source, target],
        cwd=str(repo),
    )
    actual_diff = hashlib.sha256(diff_bytes).hexdigest()
    if actual_diff != expected_diff:
        raise ValueError(
            "resume commit transition diff hash mismatch: "
            f"{actual_diff} != {expected_diff}"
        )
    return {
        "mode": "explicit_bound_transition",
        "checkpoint_git_commit": checkpoint_commit,
        "current_git_commit": current_commit,
        "changed_paths": list(changed_paths),
        "diff_sha256": actual_diff,
    }


def _model_config(cfg: DictConfig) -> Dict[str, object]:
    value: Dict[str, object] = {
        "dim_model": int(cfg.dim_model),
        "num_heads": int(cfg.num_heads),
        "num_layers": int(cfg.num_layers),
    }
    if (
        bool(cfg.get("d2ac_interaction_adapter", False))
        or bool(cfg.get("d2ad_local_frame_interaction_adapter", False))
        or bool(cfg.get("d2ae_sparse_relation_field", False))
    ):
        value["architecture_variant"] = str(cfg.get("hoi_architecture_variant"))
    return value


def _resume_contract(cfg: DictConfig) -> Dict[str, object]:
    """Critical immutable fields for an exact same-run resume."""
    split = Path(str(cfg.split_manifest)).resolve()
    contract = {
        "model_config": _model_config(cfg),
        "batch_size": int(cfg.batch_size),
        "num_gpus": int(cfg.num_gpus),
        "effective_batch_size": int(cfg.effective_batch_size),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "max_processed_windows": int(cfg.max_processed_windows),
        "validation_windows": int(cfg.validation_windows),
        "validation_interval_windows": int(cfg.validation_interval_windows),
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows),
        "learning_rate": float(cfg.learning_rate),
        "warmup_windows": int(cfg.warmup_windows),
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio),
        "weight_decay": float(cfg.weight_decay),
        "betas": [float(cfg.beta1), float(cfg.beta2)],
        "gradient_clip_norm": (
            None if cfg.gradient_clip_norm in (None, "", False) else float(cfg.gradient_clip_norm)
        ),
        "gradient_clipping": bool(cfg.get("gradient_clipping", True)),
        "optimizer_name": str(cfg.get("optimizer_name", "AdamW")),
        "scheduler_name": str(cfg.get("scheduler_name", "cosine")),
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "ema_0.9999")),
        "d2t_author_update_rule": bool(cfg.get("d2t_author_update_rule", False)),
        "d2u_balanced_author_update": bool(cfg.get("d2u_balanced_author_update", False)),
        "d2v_balanced_long_budget": bool(cfg.get("d2v_balanced_long_budget", False)),
        "d2x_fk_foot_temporal_routing": bool(
            cfg.get("d2x_fk_foot_temporal_routing", False)
        ),
        "d2ab_predicted_support_no_slip": bool(
            cfg.get("d2ab_predicted_support_no_slip", False)
        ),
        "d2ad_local_frame_interaction_adapter": bool(
            cfg.get("d2ad_local_frame_interaction_adapter", False)
        ),
        "d2ae_sparse_relation_field": bool(
            cfg.get("d2ae_sparse_relation_field", False)
        ),
        "fk_foot_temporal_routing": bool(cfg.get("fk_foot_temporal_routing", False)),
        "ema_decays": [float(value) for value in cfg.ema_decays],
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows),
        "fk_weight": float(cfg.fk_weight),
        "object_surface_weight": float(cfg.object_surface_weight),
        "velocity_weight": float(cfg.velocity_weight),
        "goal_weight": float(cfg.goal_weight),
        "weight_init_sha256": (
            None if cfg.weight_init_sha256 in (None, "", False) else str(cfg.weight_init_sha256)
        ),
        "weight_init_variant": (
            None if cfg.weight_init_variant in (None, "", False) else str(cfg.weight_init_variant)
        ),
        "d2m_candidate": (
            None if cfg.d2m_candidate in (None, "", False) else str(cfg.d2m_candidate)
        ),
        "d2m_rng_audit": bool(cfg.d2m_rng_audit),
        "amp": bool(cfg.amp),
        "data_contract_sha256": str(cfg.data_contract_sha256),
        "split_sha256": _sha256(split),
    }
    if _is_d2y(cfg):
        contract["d2y_routed_foot_amplification"] = True
        contract["routed_foot_residual_multiplier"] = float(
            cfg.get("routed_foot_residual_multiplier", 1.0)
        )
    if _is_d2z(cfg):
        contract["d2z_immutable_gt_near_ground_gating"] = True
        contract["routed_foot_residual_multiplier"] = float(
            cfg.get("routed_foot_residual_multiplier", 1.0)
        )
        contract["d2z_gate_audit_sha256"] = str(cfg.get("d2z_gate_audit_sha256"))
    if _is_d2ab(cfg):
        contract["d2ab_predicted_support_no_slip"] = True
        contract["d2ab_support_metadata_sha256"] = str(
            cfg.get("d2ab_support_metadata_sha256")
        )
    if _is_d2ac(cfg):
        contract["d2ac_interaction_adapter"] = True
        contract["architecture_variant"] = str(cfg.get("hoi_architecture_variant"))
        contract["d2ac_bps_sha256"] = D2AC_BPS_SHA256
        contract["d2ac_assignment_sha256"] = ASSIGNMENT_SHA256
    if _is_d2ad(cfg):
        contract["d2ad_local_frame_interaction_adapter"] = True
        contract["architecture_variant"] = str(cfg.get("hoi_architecture_variant"))
        contract["d2ad_bps_sha256"] = D2AC_BPS_SHA256
        contract["d2ad_basis_yup_tensor_sha256"] = BPS_YUP_TENSOR_SHA256
        contract["d2ad_rest_mesh_manifest_sha256"] = REST_MESH_MANIFEST_SHA256
        contract["d2ad_object_mapping_sha256"] = OBJECT_MAPPING_SHA256
        contract["d2ad_assignment_sha256"] = ASSIGNMENT_SHA256
        contract["local_bps_query_workers"] = int(cfg.local_bps_query_workers)
    if _is_d2ae(cfg):
        contract["d2ae_sparse_relation_field"] = True
        contract["architecture_variant"] = str(cfg.get("hoi_architecture_variant"))
        contract["d2ae_sparse_relation_parameters"] = SPARSE_RELATION_PARAMETER_COUNT
        contract["d2ae_sparse_point_mapping_sha256"] = SPARSE_POINT_MAPPING_SHA256
        contract["d2ae_sparse_point_manifest_sha256"] = SPARSE_POINT_MANIFEST_SHA256
        contract["d2ae_sparse_point_tensor_sha256"] = SPARSE_POINT_TENSOR_SHA256
        contract["d2ae_performance_benchmark_path"] = str(
            Path(str(cfg.d2ae_performance_benchmark_path)).resolve()
        )
        contract["d2ae_performance_benchmark_sha256"] = str(
            cfg.d2ae_performance_benchmark_sha256
        )
    return contract


def _lr_lambda(update: int, total_updates: int, warmup_updates: int, minimum_ratio: float) -> float:
    if warmup_updates and update < warmup_updates:
        return max((update + 1) / warmup_updates, 1.0 / warmup_updates)
    remaining = max(total_updates - warmup_updates, 1)
    progress = min(max((update - warmup_updates) / remaining, 0.0), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _is_d2t(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2t_author_update_rule", False))


def _is_d2u(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2u_balanced_author_update", False))


def _is_d2v(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2v_balanced_long_budget", False))


def _is_d2x(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2x_fk_foot_temporal_routing", False))


def _is_d2y(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2y_routed_foot_amplification", False))


def _is_d2z(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2z_immutable_gt_near_ground_gating", False))


def _is_d2ab(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2ab_predicted_support_no_slip", False))


def _is_d2ac(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2ac_interaction_adapter", False))


def _is_d2ad(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2ad_local_frame_interaction_adapter", False))


def _is_d2ae(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2ae_sparse_relation_field", False))


def _uses_author_update_rule(cfg: DictConfig) -> bool:
    return (
        _is_d2t(cfg) or _is_d2u(cfg) or _is_d2v(cfg)
        or _is_d2x(cfg) or _is_d2y(cfg) or _is_d2z(cfg) or _is_d2ab(cfg)
        or _is_d2ac(cfg) or _is_d2ad(cfg) or _is_d2ae(cfg)
    )


def _validate_fk_foot_temporal_routing_mode(cfg: DictConfig) -> None:
    routing = bool(cfg.get("fk_foot_temporal_routing", False))
    multiplier = float(cfg.get("routed_foot_residual_multiplier", 1.0))
    gating = bool(cfg.get("immutable_gt_near_ground_gating", False))
    if routing and not (
        _is_d2x(cfg) or _is_d2y(cfg) or _is_d2z(cfg) or _is_d2ab(cfg)
        or _is_d2ac(cfg) or _is_d2ad(cfg) or _is_d2ae(cfg)
    ):
        raise ValueError(
            "FK-foot temporal routing is restricted to registered D2-X/D2-Y/D2-Z/D2-AB/D2-AC/D2-AD/D2-AE modes"
        )
    if multiplier != 1.0 and not (_is_d2y(cfg) or _is_d2z(cfg)):
        raise ValueError("routed-foot residual amplification is restricted to registered D2-Y/D2-Z")
    if _is_d2y(cfg) and not routing:
        raise ValueError("D2-Y routed-foot residual amplification requires FK-foot routing")
    if gating and not _is_d2z(cfg):
        raise ValueError("immutable-GT near-ground gating is restricted to registered D2-Z")
    if _is_d2z(cfg) and (not routing or not gating):
        raise ValueError("D2-Z requires FK-foot routing and immutable-GT near-ground gating")
    if _is_d2ab(cfg) and not routing:
        raise ValueError("D2-AB predicted-support no-slip requires FK-foot routing")
    if _is_d2ac(cfg) and not routing:
        raise ValueError("D2-AC interaction adapter requires D2-X FK-foot routing")
    if _is_d2ad(cfg) and not routing:
        raise ValueError("D2-AD interaction adapter requires D2-X FK-foot routing")
    if _is_d2ae(cfg) and not routing:
        raise ValueError("D2-AE sparse relation field requires D2-X FK-foot routing")
    if _is_d2ab(cfg) and (
        bool(cfg.get("immutable_gt_near_ground_gating", False))
        or cfg.get("d2z_gate_audit_path") not in (None, "", False)
        or cfg.get("d2z_gate_audit_sha256") not in (None, "", False)
    ):
        raise ValueError("D2-AB cannot use D2-Z gate inputs")
    if not _is_d2z(cfg) and (
        cfg.get("d2z_gate_audit_path") not in (None, "", False)
        or cfg.get("d2z_gate_audit_sha256") not in (None, "", False)
    ):
        raise ValueError("D2-Z gate audit inputs are forbidden outside registered D2-Z")
    if not _is_d2ab(cfg) and (
        cfg.get("d2ab_support_metadata_path") not in (None, "", False)
        or cfg.get("d2ab_support_metadata_sha256") not in (None, "", False)
    ):
        raise ValueError("D2-AB support metadata is forbidden outside registered D2-AB")
    architecture_variant = str(
        cfg.get("hoi_architecture_variant", HOI_ARCHITECTURE_BASE)
    )
    if sum(int(value) for value in (_is_d2ac(cfg), _is_d2ad(cfg), _is_d2ae(cfg))) > 1:
        raise ValueError("D2-AC, D2-AD and D2-AE modes are mutually exclusive")
    if _is_d2ac(cfg):
        if architecture_variant != HOI_ARCHITECTURE_D2AC:
            raise ValueError("D2-AC requires the registered interaction-adapter architecture")
    elif _is_d2ad(cfg):
        if architecture_variant != HOI_ARCHITECTURE_D2AD:
            raise ValueError("D2-AD requires the registered local-frame adapter architecture")
    elif _is_d2ae(cfg):
        if architecture_variant != HOI_ARCHITECTURE_D2AE:
            raise ValueError("D2-AE requires the registered sparse-relation architecture")
    elif architecture_variant != HOI_ARCHITECTURE_BASE:
        raise ValueError(
            "HOIPrior architecture variants are forbidden outside registered D2-AC/D2-AD/D2-AE"
        )


def _locked_loss_weights(cfg: DictConfig) -> Dict[str, float]:
    if (
        _is_d2u(cfg) or _is_d2v(cfg) or _is_d2x(cfg)
        or _is_d2y(cfg) or _is_d2z(cfg) or _is_d2ab(cfg)
        or _is_d2ac(cfg) or _is_d2ad(cfg) or _is_d2ae(cfg)
    ):
        return {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        }
    return {
        "fk": 50.0,
        "object_surface": 50.0,
        "velocity": 0.1,
        "terminal_goal": 1.0,
    }


def _optimization_contract(cfg: DictConfig) -> Dict[str, object]:
    author_update = _uses_author_update_rule(cfg)
    return {
        "optimizer": "Adam" if author_update else "AdamW",
        "betas": [float(cfg.beta1), float(cfg.beta2)],
        "weight_decay": float(cfg.weight_decay),
        "learning_rate": float(cfg.learning_rate),
        "scheduler": "none" if author_update else "cosine",
        "warmup_windows": int(cfg.warmup_windows),
        "gradient_clipping": bool(cfg.get("gradient_clipping", not author_update)),
        "gradient_clip_norm": (
            None if cfg.gradient_clip_norm in (None, "", False) else float(cfg.gradient_clip_norm)
        ),
        "amp": bool(cfg.amp),
        "ema_decays": [float(value) for value in cfg.ema_decays],
        "primary_weight_variant": str(
            cfg.get("primary_weight_variant", "online" if author_update else "ema_0.9999")
        ),
    }


def _loss_routing_contract(cfg: DictConfig) -> Dict[str, object]:
    contract: Dict[str, object] = {
        "fk_foot_temporal_routing": bool(cfg.get("fk_foot_temporal_routing", False)),
        "foot_joint_indices": [7, 8, 10, 11],
        "routed_components": ["x", "z"],
        "velocity_weight": float(cfg.velocity_weight),
        "velocity_reduction": "mean_square",
    }
    if _is_d2y(cfg) or _is_d2z(cfg):
        contract.update({
            "routed_foot_residual_multiplier": float(
                cfg.get("routed_foot_residual_multiplier", 1.0)
            ),
            "nonrouted_residual_multiplier": 1.0,
            "weighted_slots": 8,
            "total_velocity_slots": 87,
        })
    if _is_d2z(cfg):
        contract.update({
            "amplification_support": "immutable_gt_previous_sampled_frame_near_ground",
            "gate_dtype": "bool",
            "gate_shape_per_window": [14, 4],
            "gate_stop_gradient": True,
            "floor_source": "complete_immutable_aligned_gt_sequence_30hz",
            "floor_algorithm": "code/eval_metrics.py::determine_floor_height_and_contacts",
            "foot_height_thresholds_m": {"7": 0.08, "8": 0.08, "10": 0.04, "11": 0.04},
            "active_multiplier": 1024.0,
            "inactive_multiplier": 1.0,
            "gate_audit_path": str(Path(str(cfg.d2z_gate_audit_path)).resolve()),
            "gate_audit_sha256": str(cfg.d2z_gate_audit_sha256),
        })
    if _is_d2ab(cfg):
        contract.update({
            "d2ab_predicted_support_no_slip": True,
            "support_metadata_path": str(
                Path(str(cfg.d2ab_support_metadata_path)).resolve()
            ),
            "support_metadata_sha256": str(cfg.d2ab_support_metadata_sha256),
            "support_floor_source": "raw_immutable_train_sequence_toe_y_5th_linear_quantile",
            "support_pair_definition": "left_7_10_right_8_11_logmeanexp_soft_min",
            "support_scale_m": 0.03925712490454316,
            "sample_interval_s": 0.1,
            "physical_velocity": "horizontal_position_delta_over_0.1s",
            "velocity_scale_s_per_m": 0.029363068377844033,
            "first_future_previous": "immutable_gt_history",
            "later_previous": "predicted_fk_previous_frame",
            "gt_and_floor_stop_gradient": True,
            "zero_slip_target_when_supported": True,
            "weighted_slots": 8,
            "total_velocity_slots": 87,
        })
    if _is_d2ac(cfg):
        contract.update({
            "d2ac_interaction_adapter": True,
            "architecture_variant": HOI_ARCHITECTURE_D2AC,
            "placement": "after_transformer_layer_4_before_layers_5_to_8",
            "global_bps_token_preserved": True,
            "bps_sha256": D2AC_BPS_SHA256,
            "assignment_sha256": ASSIGNMENT_SHA256,
            "adapter_parameters": ADAPTER_PARAMETER_COUNT,
            "d2ab_predicted_support_no_slip": False,
        })
    if _is_d2ad(cfg):
        contract.update({
            "d2ad_local_frame_interaction_adapter": True,
            "architecture_variant": HOI_ARCHITECTURE_D2AD,
            "placement": "after_transformer_layer_4_before_layers_5_to_8",
            "global_bps_token_preserved": True,
            "adapter_bps_coordinate_system": LOCAL_BASIS_COORDINATE_SYSTEM,
            "bps_sha256": D2AC_BPS_SHA256,
            "basis_yup_tensor_sha256": BPS_YUP_TENSOR_SHA256,
            "rest_mesh_manifest_sha256": REST_MESH_MANIFEST_SHA256,
            "object_mapping_sha256": OBJECT_MAPPING_SHA256,
            "assignment_sha256": ASSIGNMENT_SHA256,
            "adapter_parameters": ADAPTER_PARAMETER_COUNT,
            "full_rest_mesh": True,
            "mesh_subsample": False,
            "stored_per_window_local_bps": False,
            "d2ab_predicted_support_no_slip": False,
        })
    if _is_d2ae(cfg):
        contract.update({
            "d2ae_sparse_relation_field": True,
            "architecture_variant": HOI_ARCHITECTURE_D2AE,
            "placement": "after_motion_input_before_condition_concat_position_and_full_trunk",
            "global_bps_token_preserved": True,
            "temporal_anchors": [0, 5, 10, 15],
            "roles": ["left_hand", "right_hand", "pelvis"],
            "role_joints": [24, 26, 0],
            "rest_object_points": [100, 3],
            "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
            "manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
            "stacked_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
            "sparse_relation_parameters": SPARSE_RELATION_PARAMETER_COUNT,
            "current_state_only": True,
            "clean_target_used": False,
            "future_gt_used": False,
            "scene_used": False,
            "stored_relation_used": False,
            "d2ab_predicted_support_no_slip": False,
        })
    return contract


def _validate_d2t_contract(cfg: DictConfig, world_size: int) -> None:
    if not _is_d2t(cfg):
        return
    expected_run_id = "p1-hoi-d2t-author-update-rule-s42-20260721"
    exact = {
        "mode": str(cfg.mode) == "d2t-author-update-rule",
        "d2u_mode_off": not _is_d2u(cfg),
        "d2v_mode_off": not _is_d2v(cfg),
        "d2x_mode_off": not _is_d2x(cfg),
        "d2y_mode_off": not _is_d2y(cfg),
        "d2z_mode_off": not _is_d2z(cfg),
        "d2ac_mode_off": not _is_d2ac(cfg),
        "d2ab_mode_off": not _is_d2ab(cfg),
        "subphase": str(cfg.subphase) == "1B-D2-T0",
        "run_id": str(cfg.run_id) == expected_run_id,
        "seed": int(cfg.seed) == 42,
        "world_size": world_size == 4,
        "batch_size": int(cfg.batch_size) == 512,
        "effective_batch_size": int(cfg.effective_batch_size) == 2048,
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps) == 1,
        "max_processed_windows": int(cfg.max_processed_windows) == 6144000,
        "optimizer_updates": int(cfg.max_processed_windows) // int(cfg.effective_batch_size) == 3000,
        "validation_windows": int(cfg.validation_windows) == 32768,
        "validation_interval_windows": int(cfg.validation_interval_windows) == 3072000,
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows) == 3072000,
        "learning_rate": float(cfg.learning_rate) == 0.0001,
        "warmup_windows": int(cfg.warmup_windows) == 0,
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio) == 1.0,
        "weight_decay": float(cfg.weight_decay) == 0.0,
        "betas": [float(cfg.beta1), float(cfg.beta2)] == [0.9, 0.999],
        "optimizer_name": str(cfg.get("optimizer_name", "")) == "Adam",
        "scheduler_name": str(cfg.get("scheduler_name", "")) == "none",
        "gradient_clipping": not bool(cfg.get("gradient_clipping", True)),
        "gradient_clip_norm": cfg.gradient_clip_norm in (None, "", False),
        "amp": not bool(cfg.amp),
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows) == 0,
        "ema_decays": list(cfg.ema_decays) == [],
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "")) == "online",
        "random_initialization": all(
            value in (None, "", False)
            for value in (
                cfg.init_checkpoint, cfg.resume_checkpoint, cfg.weight_init_checkpoint,
                cfg.weight_init_sha256, cfg.weight_init_variant, cfg.d2m_candidate,
            )
        ),
        "rng_audit_off": not bool(cfg.d2m_rng_audit),
        "fk_foot_temporal_routing_off": not bool(cfg.fk_foot_temporal_routing),
        "routed_foot_multiplier_unit": (
            float(cfg.get("routed_foot_residual_multiplier", 1.0)) == 1.0
        ),
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-T author update-rule contract mismatch: {failed}")


def _validate_d2u_contract(cfg: DictConfig, world_size: int) -> None:
    if not _is_d2u(cfg):
        return
    exact = {
        "mode": str(cfg.mode) == "d2u-balanced-author-update",
        "d2t_mode_off": not _is_d2t(cfg),
        "d2v_mode_off": not _is_d2v(cfg),
        "d2x_mode_off": not _is_d2x(cfg),
        "d2y_mode_off": not _is_d2y(cfg),
        "d2z_mode_off": not _is_d2z(cfg),
        "d2ab_mode_off": not _is_d2ab(cfg),
        "d2ac_mode_off": not _is_d2ac(cfg),
        "subphase": str(cfg.subphase) == "1B-D2-U0",
        "run_id": str(cfg.run_id) == "p1-hoi-d2u-balanced-author-update-s42-20260721",
        "seed": int(cfg.seed) == 42,
        "world_size": world_size == 4,
        "batch_size": int(cfg.batch_size) == 512,
        "effective_batch_size": int(cfg.effective_batch_size) == 2048,
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps) == 1,
        "max_processed_windows": int(cfg.max_processed_windows) == 6144000,
        "optimizer_updates": int(cfg.max_processed_windows) // int(cfg.effective_batch_size) == 3000,
        "validation_windows": int(cfg.validation_windows) == 32768,
        "validation_interval_windows": int(cfg.validation_interval_windows) == 3072000,
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows) == 3072000,
        "learning_rate": float(cfg.learning_rate) == 0.0001,
        "warmup_windows": int(cfg.warmup_windows) == 0,
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio) == 1.0,
        "weight_decay": float(cfg.weight_decay) == 0.0,
        "betas": [float(cfg.beta1), float(cfg.beta2)] == [0.9, 0.999],
        "optimizer_name": str(cfg.get("optimizer_name", "")) == "Adam",
        "scheduler_name": str(cfg.get("scheduler_name", "")) == "none",
        "gradient_clipping": not bool(cfg.get("gradient_clipping", True)),
        "gradient_clip_norm": cfg.gradient_clip_norm in (None, "", False),
        "amp": not bool(cfg.amp),
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows) == 0,
        "ema_decays": list(cfg.ema_decays) == [],
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "")) == "online",
        "balanced_weights": {
            "fk": float(cfg.fk_weight),
            "object_surface": float(cfg.object_surface_weight),
            "velocity": float(cfg.velocity_weight),
            "terminal_goal": float(cfg.goal_weight),
        } == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
        "random_initialization": all(
            value in (None, "", False)
            for value in (
                cfg.init_checkpoint, cfg.resume_checkpoint, cfg.weight_init_checkpoint,
                cfg.weight_init_sha256, cfg.weight_init_variant, cfg.d2m_candidate,
            )
        ),
        "rng_audit_off": not bool(cfg.d2m_rng_audit),
        "fk_foot_temporal_routing_off": not bool(cfg.fk_foot_temporal_routing),
        "routed_foot_multiplier_unit": (
            float(cfg.get("routed_foot_residual_multiplier", 1.0)) == 1.0
        ),
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-U balanced author-update contract mismatch: {failed}")


def _validate_d2v_contract(cfg: DictConfig, world_size: int) -> None:
    if not _is_d2v(cfg):
        return
    exact = {
        "mode": str(cfg.mode) == "d2v-balanced-long-budget",
        "d2t_mode_off": not _is_d2t(cfg),
        "d2u_mode_off": not _is_d2u(cfg),
        "d2x_mode_off": not _is_d2x(cfg),
        "d2y_mode_off": not _is_d2y(cfg),
        "d2z_mode_off": not _is_d2z(cfg),
        "d2ab_mode_off": not _is_d2ab(cfg),
        "d2ac_mode_off": not _is_d2ac(cfg),
        "subphase": str(cfg.subphase) == "1B-D2-V0",
        "run_id": str(cfg.run_id) == "p1-hoi-d2v-balanced-long-budget-s42-20260722",
        "seed": int(cfg.seed) == 42,
        "world_size": world_size == 4,
        "batch_size": int(cfg.batch_size) == 512,
        "effective_batch_size": int(cfg.effective_batch_size) == 2048,
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps) == 1,
        "max_processed_windows": int(cfg.max_processed_windows) == 61440000,
        "optimizer_updates": int(cfg.max_processed_windows) // int(cfg.effective_batch_size) == 30000,
        "validation_windows": int(cfg.validation_windows) == 32768,
        "validation_interval_windows": int(cfg.validation_interval_windows) == 3072000,
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows) == 3072000,
        "learning_rate": float(cfg.learning_rate) == 0.0001,
        "warmup_windows": int(cfg.warmup_windows) == 0,
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio) == 1.0,
        "weight_decay": float(cfg.weight_decay) == 0.0,
        "betas": [float(cfg.beta1), float(cfg.beta2)] == [0.9, 0.999],
        "optimizer_name": str(cfg.get("optimizer_name", "")) == "Adam",
        "scheduler_name": str(cfg.get("scheduler_name", "")) == "none",
        "gradient_clipping": not bool(cfg.get("gradient_clipping", True)),
        "gradient_clip_norm": cfg.gradient_clip_norm in (None, "", False),
        "amp": not bool(cfg.amp),
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows) == 0,
        "ema_decays": list(cfg.ema_decays) == [],
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "")) == "online",
        "balanced_weights": {
            "fk": float(cfg.fk_weight),
            "object_surface": float(cfg.object_surface_weight),
            "velocity": float(cfg.velocity_weight),
            "terminal_goal": float(cfg.goal_weight),
        } == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
        "random_initialization": all(
            value in (None, "", False)
            for value in (
                cfg.init_checkpoint, cfg.resume_checkpoint, cfg.weight_init_checkpoint,
                cfg.weight_init_sha256, cfg.weight_init_variant, cfg.d2m_candidate,
            )
        ),
        "rng_audit_off": not bool(cfg.d2m_rng_audit),
        "fk_foot_temporal_routing_off": not bool(cfg.fk_foot_temporal_routing),
        "routed_foot_multiplier_unit": (
            float(cfg.get("routed_foot_residual_multiplier", 1.0)) == 1.0
        ),
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-V balanced long-budget contract mismatch: {failed}")


def _validate_d2x_contract(cfg: DictConfig, world_size: int) -> None:
    if not _is_d2x(cfg):
        return
    exact = {
        "mode": str(cfg.mode) == "d2x-fk-foot-temporal-routing",
        "d2t_mode_off": not _is_d2t(cfg),
        "d2u_mode_off": not _is_d2u(cfg),
        "d2v_mode_off": not _is_d2v(cfg),
        "d2y_mode_off": not _is_d2y(cfg),
        "d2z_mode_off": not _is_d2z(cfg),
        "d2ab_mode_off": not _is_d2ab(cfg),
        "d2ac_mode_off": not _is_d2ac(cfg),
        "subphase": str(cfg.subphase) == "1B-D2-X0-r1",
        "run_id": str(cfg.run_id) == (
            "p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723"
        ),
        "seed": int(cfg.seed) == 42,
        "world_size": world_size == 4,
        "batch_size": int(cfg.batch_size) == 512,
        "effective_batch_size": int(cfg.effective_batch_size) == 2048,
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps) == 1,
        "max_processed_windows": int(cfg.max_processed_windows) == 61440000,
        "optimizer_updates": (
            int(cfg.max_processed_windows) // int(cfg.effective_batch_size) == 30000
        ),
        "validation_windows": int(cfg.validation_windows) == 32768,
        "validation_interval_windows": int(cfg.validation_interval_windows) == 3072000,
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows) == 3072000,
        "learning_rate": float(cfg.learning_rate) == 0.0001,
        "warmup_windows": int(cfg.warmup_windows) == 0,
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio) == 1.0,
        "weight_decay": float(cfg.weight_decay) == 0.0,
        "betas": [float(cfg.beta1), float(cfg.beta2)] == [0.9, 0.999],
        "optimizer_name": str(cfg.get("optimizer_name", "")) == "Adam",
        "scheduler_name": str(cfg.get("scheduler_name", "")) == "none",
        "gradient_clipping": not bool(cfg.get("gradient_clipping", True)),
        "gradient_clip_norm": cfg.gradient_clip_norm in (None, "", False),
        "amp": not bool(cfg.amp),
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows) == 0,
        "ema_decays": list(cfg.ema_decays) == [],
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "")) == "online",
        "balanced_weights": {
            "fk": float(cfg.fk_weight),
            "object_surface": float(cfg.object_surface_weight),
            "velocity": float(cfg.velocity_weight),
            "terminal_goal": float(cfg.goal_weight),
        } == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
        "fk_foot_temporal_routing_on": bool(cfg.fk_foot_temporal_routing),
        "routed_foot_multiplier_unit": (
            float(cfg.get("routed_foot_residual_multiplier", 1.0)) == 1.0
        ),
        "random_initialization": all(
            value in (None, "", False)
            for value in (
                cfg.init_checkpoint, cfg.resume_checkpoint, cfg.weight_init_checkpoint,
                cfg.weight_init_sha256, cfg.weight_init_variant, cfg.d2m_candidate,
            )
        ),
        "rng_audit_off": not bool(cfg.d2m_rng_audit),
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-X FK-foot temporal routing contract mismatch: {failed}")


def _validate_d2y_contract(cfg: DictConfig, world_size: int) -> None:
    if not _is_d2y(cfg):
        return
    exact = {
        "mode": str(cfg.mode) == "d2y-routed-foot-amplification",
        "d2t_mode_off": not _is_d2t(cfg),
        "d2u_mode_off": not _is_d2u(cfg),
        "d2v_mode_off": not _is_d2v(cfg),
        "d2x_mode_off": not _is_d2x(cfg),
        "d2z_mode_off": not _is_d2z(cfg),
        "d2ab_mode_off": not _is_d2ab(cfg),
        "d2ac_mode_off": not _is_d2ac(cfg),
        "subphase": str(cfg.subphase) == "1B-D2-Y0",
        "run_id": str(cfg.run_id) == (
            "p1-hoi-d2y-routed-foot-amplification-s42-20260723"
        ),
        "seed": int(cfg.seed) == 42,
        "world_size": world_size == 4,
        "batch_size": int(cfg.batch_size) == 512,
        "effective_batch_size": int(cfg.effective_batch_size) == 2048,
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps) == 1,
        "max_processed_windows": int(cfg.max_processed_windows) == 61440000,
        "optimizer_updates": (
            int(cfg.max_processed_windows) // int(cfg.effective_batch_size) == 30000
        ),
        "validation_windows": int(cfg.validation_windows) == 32768,
        "validation_interval_windows": int(cfg.validation_interval_windows) == 3072000,
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows) == 3072000,
        "learning_rate": float(cfg.learning_rate) == 0.0001,
        "warmup_windows": int(cfg.warmup_windows) == 0,
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio) == 1.0,
        "weight_decay": float(cfg.weight_decay) == 0.0,
        "betas": [float(cfg.beta1), float(cfg.beta2)] == [0.9, 0.999],
        "optimizer_name": str(cfg.get("optimizer_name", "")) == "Adam",
        "scheduler_name": str(cfg.get("scheduler_name", "")) == "none",
        "gradient_clipping": not bool(cfg.get("gradient_clipping", True)),
        "gradient_clip_norm": cfg.gradient_clip_norm in (None, "", False),
        "amp": not bool(cfg.amp),
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows) == 0,
        "ema_decays": list(cfg.ema_decays) == [],
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "")) == "online",
        "balanced_weights": {
            "fk": float(cfg.fk_weight),
            "object_surface": float(cfg.object_surface_weight),
            "velocity": float(cfg.velocity_weight),
            "terminal_goal": float(cfg.goal_weight),
        } == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
        "fk_foot_temporal_routing_on": bool(cfg.fk_foot_temporal_routing),
        "routed_foot_multiplier": (
            float(cfg.get("routed_foot_residual_multiplier", 1.0)) == 1024.0
        ),
        "random_initialization": all(
            value in (None, "", False)
            for value in (
                cfg.init_checkpoint, cfg.resume_checkpoint, cfg.weight_init_checkpoint,
                cfg.weight_init_sha256, cfg.weight_init_variant, cfg.d2m_candidate,
            )
        ),
        "rng_audit_off": not bool(cfg.d2m_rng_audit),
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-Y routed-foot amplification contract mismatch: {failed}")


def _validate_d2z_contract(cfg: DictConfig, world_size: int) -> None:
    if not _is_d2z(cfg):
        return
    audit_path_value = cfg.get("d2z_gate_audit_path")
    audit_sha256 = str(cfg.get("d2z_gate_audit_sha256"))
    audit_path = (
        None if audit_path_value in (None, "", False)
        else Path(str(audit_path_value)).resolve()
    )
    actual_audit_sha256 = (
        _sha256(audit_path) if audit_path is not None and audit_path.is_file() else None
    )
    exact = {
        "mode": str(cfg.mode) == "d2z-immutable-gt-near-ground-gating",
        "d2t_mode_off": not _is_d2t(cfg),
        "d2u_mode_off": not _is_d2u(cfg),
        "d2v_mode_off": not _is_d2v(cfg),
        "d2x_mode_off": not _is_d2x(cfg),
        "d2y_mode_off": not _is_d2y(cfg),
        "d2ab_mode_off": not _is_d2ab(cfg),
        "d2ac_mode_off": not _is_d2ac(cfg),
        "subphase": str(cfg.subphase) == "1B-D2-Z0",
        "run_id": str(cfg.run_id) == (
            "p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724"
        ),
        "seed": int(cfg.seed) == 42,
        "world_size": world_size == 4,
        "batch_size": int(cfg.batch_size) == 512,
        "effective_batch_size": int(cfg.effective_batch_size) == 2048,
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps) == 1,
        "max_processed_windows": int(cfg.max_processed_windows) == 61440000,
        "optimizer_updates": (
            int(cfg.max_processed_windows) // int(cfg.effective_batch_size) == 30000
        ),
        "validation_windows": int(cfg.validation_windows) == 32768,
        "validation_interval_windows": int(cfg.validation_interval_windows) == 3072000,
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows) == 3072000,
        "learning_rate": float(cfg.learning_rate) == 0.0001,
        "warmup_windows": int(cfg.warmup_windows) == 0,
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio) == 1.0,
        "weight_decay": float(cfg.weight_decay) == 0.0,
        "betas": [float(cfg.beta1), float(cfg.beta2)] == [0.9, 0.999],
        "optimizer_name": str(cfg.get("optimizer_name", "")) == "Adam",
        "scheduler_name": str(cfg.get("scheduler_name", "")) == "none",
        "gradient_clipping": not bool(cfg.get("gradient_clipping", True)),
        "gradient_clip_norm": cfg.gradient_clip_norm in (None, "", False),
        "amp": not bool(cfg.amp),
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows) == 0,
        "ema_decays": list(cfg.ema_decays) == [],
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "")) == "online",
        "balanced_weights": {
            "fk": float(cfg.fk_weight),
            "object_surface": float(cfg.object_surface_weight),
            "velocity": float(cfg.velocity_weight),
            "terminal_goal": float(cfg.goal_weight),
        } == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
        "fk_foot_temporal_routing_on": bool(cfg.fk_foot_temporal_routing),
        "immutable_gt_near_ground_gating_on": bool(
            cfg.get("immutable_gt_near_ground_gating", False)
        ),
        "routed_foot_multiplier": (
            float(cfg.get("routed_foot_residual_multiplier", 1.0)) == 1024.0
        ),
        "gate_audit_file": audit_path is not None and audit_path.is_file(),
        "gate_audit_sha256_format": (
            len(audit_sha256) == 64
            and all(character in "0123456789abcdef" for character in audit_sha256)
        ),
        "gate_audit_sha256": actual_audit_sha256 == audit_sha256,
        "random_initialization": all(
            value in (None, "", False)
            for value in (
                cfg.init_checkpoint, cfg.resume_checkpoint, cfg.weight_init_checkpoint,
                cfg.weight_init_sha256, cfg.weight_init_variant, cfg.d2m_candidate,
            )
        ),
        "rng_audit_off": not bool(cfg.d2m_rng_audit),
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-Z immutable-GT near-ground gating contract mismatch: {failed}")


def _validate_d2ab_contract(cfg: DictConfig, world_size: int) -> None:
    if not _is_d2ab(cfg):
        return
    split_path = Path(str(cfg.split_manifest)).resolve()
    metadata_path = Path(str(cfg.d2ab_support_metadata_path)).resolve()
    try:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        train_sequences = split_payload["train"]["sequence_indices"]
        from priors.d2ab import validate_metadata

        validate_metadata(
            metadata_path,
            str(cfg.d2ab_support_metadata_sha256),
            split_path=split_path,
            expected_train_sequence_indices=train_sequences,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"D2-AB support metadata validation failed: {error}") from error
    resume_value = cfg.resume_checkpoint
    resume_allowed = (
        resume_value in (None, "", False)
        or (
            Path(str(resume_value)).name.startswith(
                "p1-hoi-d2ab-predicted-support-no-slip-s42-20260725_windows"
            )
            and Path(str(resume_value)).suffix == ".pth"
        )
    )
    exact = {
        "mode": str(cfg.mode) == "d2ab-predicted-support-no-slip",
        "d2t_mode_off": not _is_d2t(cfg),
        "d2u_mode_off": not _is_d2u(cfg),
        "d2v_mode_off": not _is_d2v(cfg),
        "d2x_mode_off": not _is_d2x(cfg),
        "d2y_mode_off": not _is_d2y(cfg),
        "d2z_mode_off": not _is_d2z(cfg),
        "subphase": str(cfg.subphase) == "1B-D2-AB0",
        "run_id": str(cfg.run_id) == (
            "p1-hoi-d2ab-predicted-support-no-slip-s42-20260725"
        ),
        "seed": int(cfg.seed) == 42,
        "world_size": world_size == 4,
        "batch_size": int(cfg.batch_size) == 512,
        "effective_batch_size": int(cfg.effective_batch_size) == 2048,
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps) == 1,
        "max_processed_windows": int(cfg.max_processed_windows) == 61440000,
        "optimizer_updates": (
            int(cfg.max_processed_windows) // int(cfg.effective_batch_size) == 30000
        ),
        "validation_windows": int(cfg.validation_windows) == 32768,
        "validation_interval_windows": int(cfg.validation_interval_windows) == 3072000,
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows) == 3072000,
        "learning_rate": float(cfg.learning_rate) == 0.0001,
        "warmup_windows": int(cfg.warmup_windows) == 0,
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio) == 1.0,
        "weight_decay": float(cfg.weight_decay) == 0.0,
        "betas": [float(cfg.beta1), float(cfg.beta2)] == [0.9, 0.999],
        "optimizer_name": str(cfg.get("optimizer_name", "")) == "Adam",
        "scheduler_name": str(cfg.get("scheduler_name", "")) == "none",
        "gradient_clipping": not bool(cfg.get("gradient_clipping", True)),
        "gradient_clip_norm": cfg.gradient_clip_norm in (None, "", False),
        "amp": not bool(cfg.amp),
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows) == 0,
        "ema_decays": list(cfg.ema_decays) == [],
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "")) == "online",
        "balanced_weights": {
            "fk": float(cfg.fk_weight),
            "object_surface": float(cfg.object_surface_weight),
            "velocity": float(cfg.velocity_weight),
            "terminal_goal": float(cfg.goal_weight),
        } == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
        "fk_foot_temporal_routing_on": bool(cfg.fk_foot_temporal_routing),
        "routed_foot_multiplier_unit": (
            float(cfg.get("routed_foot_residual_multiplier", 1.0)) == 1.0
        ),
        "immutable_gt_gate_off": not bool(
            cfg.get("immutable_gt_near_ground_gating", False)
        ),
        "metadata_sha256": str(cfg.d2ab_support_metadata_sha256) == (
            "807978580221910ad00260c2dff4f33ddacbb1bf72bad7443bf21ac48f31f079"
        ),
        "random_initialization": all(
            value in (None, "", False)
            for value in (
                cfg.init_checkpoint, cfg.weight_init_checkpoint,
                cfg.weight_init_sha256, cfg.weight_init_variant, cfg.d2m_candidate,
            )
        ),
        "resume_same_run_only": resume_allowed,
        "rng_audit_off": not bool(cfg.d2m_rng_audit),
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-AB predicted-support no-slip contract mismatch: {failed}")


def _validate_d2ac_contract(cfg: DictConfig, world_size: int) -> None:
    if not _is_d2ac(cfg):
        return
    resume_value = cfg.resume_checkpoint
    resume_allowed = (
        resume_value in (None, "", False)
        or (
            Path(str(resume_value)).name.startswith(
                "p1-hoi-d2ac-interaction-adapter-s42-20260726_windows"
            )
            and Path(str(resume_value)).suffix == ".pth"
        )
    )
    split_path = Path(str(cfg.split_manifest)).resolve()
    bps_path = Path(str(cfg.repo_root)).resolve() / "code/bps.pt"
    exact = {
        "mode": str(cfg.mode) == "d2ac-interaction-adapter",
        "d2t_mode_off": not _is_d2t(cfg),
        "d2u_mode_off": not _is_d2u(cfg),
        "d2v_mode_off": not _is_d2v(cfg),
        "d2x_mode_off": not _is_d2x(cfg),
        "d2y_mode_off": not _is_d2y(cfg),
        "d2z_mode_off": not _is_d2z(cfg),
        "d2ab_mode_off": not _is_d2ab(cfg),
        "d2ad_mode_off": not _is_d2ad(cfg),
        "subphase": str(cfg.subphase) == "1B-D2-AC0",
        "run_id": str(cfg.run_id) == (
            "p1-hoi-d2ac-interaction-adapter-s42-20260726"
        ),
        "seed": int(cfg.seed) == 42,
        "architecture_variant": (
            str(cfg.get("hoi_architecture_variant")) == HOI_ARCHITECTURE_D2AC
        ),
        "model_config": _model_config(cfg) == {
            "dim_model": 512,
            "num_heads": 16,
            "num_layers": 8,
            "architecture_variant": HOI_ARCHITECTURE_D2AC,
        },
        "world_size": world_size == 4,
        "batch_size": int(cfg.batch_size) == 512,
        "effective_batch_size": int(cfg.effective_batch_size) == 2048,
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps) == 1,
        "dataset_limit": int(cfg.dataset_limit) == 0,
        "max_processed_windows": int(cfg.max_processed_windows) == 61440000,
        "processed_frames": int(cfg.max_processed_windows) * 16 == 983040000,
        "optimizer_updates": (
            int(cfg.max_processed_windows) // int(cfg.effective_batch_size) == 30000
        ),
        "validation_windows": int(cfg.validation_windows) == 32768,
        "validation_interval_windows": int(cfg.validation_interval_windows) == 3072000,
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows) == 3072000,
        "no_artificial_pause": cfg.pause_after_windows in (None, "", False),
        "learning_rate": float(cfg.learning_rate) == 0.0001,
        "warmup_windows": int(cfg.warmup_windows) == 0,
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio) == 1.0,
        "weight_decay": float(cfg.weight_decay) == 0.0,
        "betas": [float(cfg.beta1), float(cfg.beta2)] == [0.9, 0.999],
        "optimizer_name": str(cfg.get("optimizer_name", "")) == "Adam",
        "scheduler_name": str(cfg.get("scheduler_name", "")) == "none",
        "gradient_clipping": not bool(cfg.get("gradient_clipping", True)),
        "gradient_clip_norm": cfg.gradient_clip_norm in (None, "", False),
        "amp": not bool(cfg.amp),
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows) == 0,
        "ema_decays": list(cfg.ema_decays) == [],
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "")) == "online",
        "balanced_weights": {
            "fk": float(cfg.fk_weight),
            "object_surface": float(cfg.object_surface_weight),
            "velocity": float(cfg.velocity_weight),
            "terminal_goal": float(cfg.goal_weight),
        } == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
        "d2x_fk_foot_temporal_routing_flag_off": not _is_d2x(cfg),
        "fk_foot_temporal_routing_on": bool(cfg.fk_foot_temporal_routing),
        "routed_foot_multiplier_unit": (
            float(cfg.get("routed_foot_residual_multiplier", 1.0)) == 1.0
        ),
        "immutable_gt_gate_off": not bool(
            cfg.get("immutable_gt_near_ground_gating", False)
        ),
        "d2ab_objective_off": not bool(
            cfg.get("d2ab_predicted_support_no_slip", False)
        ),
        "d2ab_metadata_absent": (
            cfg.get("d2ab_support_metadata_path") in (None, "", False)
            and cfg.get("d2ab_support_metadata_sha256") in (None, "", False)
        ),
        "d2z_inputs_absent": (
            cfg.get("d2z_gate_audit_path") in (None, "", False)
            and cfg.get("d2z_gate_audit_sha256") in (None, "", False)
        ),
        "random_initialization": all(
            value in (None, "", False)
            for value in (
                cfg.init_checkpoint, cfg.weight_init_checkpoint,
                cfg.weight_init_sha256, cfg.weight_init_variant, cfg.d2m_candidate,
            )
        ),
        "resume_same_run_only": resume_allowed,
        "rng_audit_off": not bool(cfg.d2m_rng_audit),
        "split_sha256": split_path.is_file() and _sha256(split_path) == (
            "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e"
        ),
        "bps_sha256": bps_path.is_file() and _sha256(bps_path) == D2AC_BPS_SHA256,
        "assignment_sha256_well_formed": (
            len(ASSIGNMENT_SHA256) == 64
            and all(character in "0123456789abcdef" for character in ASSIGNMENT_SHA256)
        ),
        "adapter_parameter_count": ADAPTER_PARAMETER_COUNT == 349697,
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-AC interaction-adapter contract mismatch: {failed}")


def _validate_d2ad_contract(cfg: DictConfig, world_size: int) -> None:
    if not _is_d2ad(cfg):
        return
    resume_value = cfg.resume_checkpoint
    resume_allowed = (
        resume_value in (None, "", False)
        or (
            Path(str(resume_value)).name.startswith(
                "p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260728_windows"
            )
            and Path(str(resume_value)).suffix == ".pth"
        )
    )
    split_path = Path(str(cfg.split_manifest)).resolve()
    repo = Path(str(cfg.repo_root)).resolve()
    bps_path = repo / "code/bps.pt"
    mesh_root = repo / "data/object/rest_object_geo"
    mesh_names = tuple(sorted(path.stem for path in mesh_root.glob("*.ply")))
    geometry_contract = LocalObjectBPSBuilder(
        repo,
        query_workers=int(cfg.get("local_bps_query_workers", 0)),
    ).contract_metadata()
    exact = {
        "mode": str(cfg.mode) == "d2ad-local-frame-interaction-adapter",
        "d2t_mode_off": not _is_d2t(cfg),
        "d2u_mode_off": not _is_d2u(cfg),
        "d2v_mode_off": not _is_d2v(cfg),
        "d2x_mode_off": not _is_d2x(cfg),
        "d2y_mode_off": not _is_d2y(cfg),
        "d2z_mode_off": not _is_d2z(cfg),
        "d2ab_mode_off": not _is_d2ab(cfg),
        "d2ac_mode_off": not _is_d2ac(cfg),
        "subphase": str(cfg.subphase) == "1B-D2-AD0",
        "run_id": str(cfg.run_id) == (
            "p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260728"
        ),
        "seed": int(cfg.seed) == 42,
        "architecture_variant": (
            str(cfg.get("hoi_architecture_variant")) == HOI_ARCHITECTURE_D2AD
        ),
        "model_config": _model_config(cfg) == {
            "dim_model": 512,
            "num_heads": 16,
            "num_layers": 8,
            "architecture_variant": HOI_ARCHITECTURE_D2AD,
        },
        "world_size": world_size == 4,
        "batch_size": int(cfg.batch_size) == 512,
        "effective_batch_size": int(cfg.effective_batch_size) == 2048,
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps) == 1,
        "num_workers": int(cfg.num_workers) == 4,
        "local_bps_query_workers": int(
            cfg.get("local_bps_query_workers", 0)
        ) == DEFAULT_QUERY_WORKERS,
        "dataset_limit": int(cfg.dataset_limit) == 0,
        "max_processed_windows": int(cfg.max_processed_windows) == 61440000,
        "processed_frames": int(cfg.max_processed_windows) * 16 == 983040000,
        "optimizer_updates": (
            int(cfg.max_processed_windows) // int(cfg.effective_batch_size) == 30000
        ),
        "validation_windows": int(cfg.validation_windows) == 32768,
        "validation_interval_windows": int(cfg.validation_interval_windows) == 3072000,
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows) == 3072000,
        "no_artificial_pause": cfg.pause_after_windows in (None, "", False),
        "learning_rate": float(cfg.learning_rate) == 0.0001,
        "warmup_windows": int(cfg.warmup_windows) == 0,
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio) == 1.0,
        "weight_decay": float(cfg.weight_decay) == 0.0,
        "betas": [float(cfg.beta1), float(cfg.beta2)] == [0.9, 0.999],
        "optimizer_name": str(cfg.get("optimizer_name", "")) == "Adam",
        "scheduler_name": str(cfg.get("scheduler_name", "")) == "none",
        "gradient_clipping": not bool(cfg.get("gradient_clipping", True)),
        "gradient_clip_norm": cfg.gradient_clip_norm in (None, "", False),
        "amp": not bool(cfg.amp),
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows) == 0,
        "ema_decays": list(cfg.ema_decays) == [],
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "")) == "online",
        "balanced_weights": {
            "fk": float(cfg.fk_weight),
            "object_surface": float(cfg.object_surface_weight),
            "velocity": float(cfg.velocity_weight),
            "terminal_goal": float(cfg.goal_weight),
        } == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
        "fk_foot_temporal_routing_on": bool(cfg.fk_foot_temporal_routing),
        "routed_foot_multiplier_unit": (
            float(cfg.get("routed_foot_residual_multiplier", 1.0)) == 1.0
        ),
        "immutable_gt_gate_off": not bool(
            cfg.get("immutable_gt_near_ground_gating", False)
        ),
        "d2ab_objective_off": not bool(
            cfg.get("d2ab_predicted_support_no_slip", False)
        ),
        "d2ab_metadata_absent": (
            cfg.get("d2ab_support_metadata_path") in (None, "", False)
            and cfg.get("d2ab_support_metadata_sha256") in (None, "", False)
        ),
        "d2z_inputs_absent": (
            cfg.get("d2z_gate_audit_path") in (None, "", False)
            and cfg.get("d2z_gate_audit_sha256") in (None, "", False)
        ),
        "random_initialization": all(
            value in (None, "", False)
            for value in (
                cfg.init_checkpoint,
                cfg.weight_init_checkpoint,
                cfg.weight_init_sha256,
                cfg.weight_init_variant,
                cfg.d2m_candidate,
            )
        ),
        "resume_same_run_only": resume_allowed,
        "rng_audit_off": not bool(cfg.d2m_rng_audit),
        "split_sha256": split_path.is_file() and _sha256(split_path) == (
            "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e"
        ),
        "bps_sha256": bps_path.is_file() and _sha256(bps_path) == D2AC_BPS_SHA256,
        "basis_yup_tensor_sha256": (
            geometry_contract["basis_yup_tensor_sha256"] == BPS_YUP_TENSOR_SHA256
        ),
        "rest_mesh_mapping": mesh_names == tuple(
            (
                "clothesstand", "floorlamp", "largebox", "largetable", "monitor",
                "plasticbox", "smallbox", "smalltable", "suitcase", "trashcan",
                "tripod", "whitechair", "woodchair",
            )
        ),
        "rest_mesh_manifest_sha256": (
            geometry_contract["rest_mesh_manifest_sha256"]
            == REST_MESH_MANIFEST_SHA256
        ),
        "object_mapping_sha256": (
            geometry_contract["object_mapping_sha256"] == OBJECT_MAPPING_SHA256
        ),
        "assignment_sha256_well_formed": (
            len(ASSIGNMENT_SHA256) == 64
            and all(character in "0123456789abcdef" for character in ASSIGNMENT_SHA256)
        ),
        "adapter_parameter_count": ADAPTER_PARAMETER_COUNT == 349697,
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-AD local-frame interaction-adapter contract mismatch: {failed}")


def _validate_d2ae_contract(
    cfg: DictConfig,
    world_size: int,
    *,
    require_performance_gate: bool = True,
) -> Optional[Dict[str, object]]:
    if not _is_d2ae(cfg):
        return None
    resume_value = cfg.resume_checkpoint
    is_resume = resume_value not in (None, "", False)
    run_id_contract = _validate_d2ae_formal_run_id(
        str(cfg.run_id),
        require_actual_date=not is_resume,
    )
    resume_allowed = (
        resume_value in (None, "", False)
        or (
            Path(str(resume_value)).name.startswith(
                f"{cfg.run_id}_windows"
            )
            and Path(str(resume_value)).suffix == ".pth"
        )
    )
    split_path = Path(str(cfg.split_manifest)).resolve()
    exact = {
        "mode": str(cfg.mode) == "d2ae-sparse-relation-field",
        "d2t_mode_off": not _is_d2t(cfg),
        "d2u_mode_off": not _is_d2u(cfg),
        "d2v_mode_off": not _is_d2v(cfg),
        "d2x_mode_off": not _is_d2x(cfg),
        "d2y_mode_off": not _is_d2y(cfg),
        "d2z_mode_off": not _is_d2z(cfg),
        "d2ab_mode_off": not _is_d2ab(cfg),
        "d2ac_mode_off": not _is_d2ac(cfg),
        "d2ad_mode_off": not _is_d2ad(cfg),
        "subphase": str(cfg.subphase) == "1B-D2-AE0",
        "run_id": run_id_contract["run_id"] == str(cfg.run_id),
        "run_id_date": is_resume or run_id_contract["date_is_actual"],
        "seed": int(cfg.seed) == 42,
        "architecture_variant": (
            str(cfg.get("hoi_architecture_variant")) == HOI_ARCHITECTURE_D2AE
        ),
        "model_config": _model_config(cfg) == {
            "dim_model": 512,
            "num_heads": 16,
            "num_layers": 8,
            "architecture_variant": HOI_ARCHITECTURE_D2AE,
        },
        "world_size": world_size == 4,
        "batch_size": int(cfg.batch_size) == 512,
        "effective_batch_size": int(cfg.effective_batch_size) == 2048,
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps) == 1,
        "num_workers": int(cfg.num_workers) == 4,
        "local_bps_query_workers_absent": (
            cfg.get("local_bps_query_workers") in (None, "", False)
        ),
        "dataset_limit": int(cfg.dataset_limit) == 0,
        "max_processed_windows": int(cfg.max_processed_windows) == 61440000,
        "processed_frames": int(cfg.max_processed_windows) * 16 == 983040000,
        "optimizer_updates": (
            int(cfg.max_processed_windows) // int(cfg.effective_batch_size) == 30000
        ),
        "validation_windows": int(cfg.validation_windows) == 32768,
        "validation_interval_windows": int(cfg.validation_interval_windows) == 3072000,
        "checkpoint_interval_windows": int(cfg.checkpoint_interval_windows) == 3072000,
        "no_artificial_pause": cfg.pause_after_windows in (None, "", False),
        "learning_rate": float(cfg.learning_rate) == 0.0001,
        "warmup_windows": int(cfg.warmup_windows) == 0,
        "minimum_lr_ratio": float(cfg.minimum_lr_ratio) == 1.0,
        "weight_decay": float(cfg.weight_decay) == 0.0,
        "betas": [float(cfg.beta1), float(cfg.beta2)] == [0.9, 0.999],
        "optimizer_name": str(cfg.get("optimizer_name", "")) == "Adam",
        "scheduler_name": str(cfg.get("scheduler_name", "")) == "none",
        "gradient_clipping": not bool(cfg.get("gradient_clipping", True)),
        "gradient_clip_norm": cfg.gradient_clip_norm in (None, "", False),
        "amp": not bool(cfg.amp),
        "max_consecutive_amp_overflows": int(cfg.max_consecutive_amp_overflows) == 0,
        "ema_decays": list(cfg.ema_decays) == [],
        "primary_weight_variant": str(cfg.get("primary_weight_variant", "")) == "online",
        "balanced_weights": {
            "fk": float(cfg.fk_weight),
            "object_surface": float(cfg.object_surface_weight),
            "velocity": float(cfg.velocity_weight),
            "terminal_goal": float(cfg.goal_weight),
        } == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
        "fk_foot_temporal_routing_on": bool(cfg.fk_foot_temporal_routing),
        "routed_foot_multiplier_unit": (
            float(cfg.get("routed_foot_residual_multiplier", 1.0)) == 1.0
        ),
        "immutable_gt_gate_off": not bool(
            cfg.get("immutable_gt_near_ground_gating", False)
        ),
        "d2ab_objective_off": not bool(
            cfg.get("d2ab_predicted_support_no_slip", False)
        ),
        "d2ab_metadata_absent": (
            cfg.get("d2ab_support_metadata_path") in (None, "", False)
            and cfg.get("d2ab_support_metadata_sha256") in (None, "", False)
        ),
        "d2z_inputs_absent": (
            cfg.get("d2z_gate_audit_path") in (None, "", False)
            and cfg.get("d2z_gate_audit_sha256") in (None, "", False)
        ),
        "random_initialization": all(
            value in (None, "", False)
            for value in (
                cfg.init_checkpoint,
                cfg.weight_init_checkpoint,
                cfg.weight_init_sha256,
                cfg.weight_init_variant,
                cfg.d2m_candidate,
            )
        ),
        "resume_same_run_only": resume_allowed,
        "rng_audit_off": not bool(cfg.d2m_rng_audit),
        "split_sha256": split_path.is_file() and _sha256(split_path) == (
            "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e"
        ),
        "base_parameter_count": D2AE_BASE_PARAMETER_COUNT == 29673448,
        "sparse_relation_parameter_count": SPARSE_RELATION_PARAMETER_COUNT == 413953,
        "total_parameter_count": D2AE_TOTAL_PARAMETER_COUNT == 30087401,
        "parameter_increase_below_limit": D2AE_PARAMETER_INCREASE_FRACTION <= 0.015,
        "mapping_sha256": len(SPARSE_POINT_MAPPING_SHA256) == 64,
        "manifest_sha256": len(SPARSE_POINT_MANIFEST_SHA256) == 64,
        "stacked_tensor_sha256": len(SPARSE_POINT_TENSOR_SHA256) == 64,
        "performance_gate_binding": (
            require_performance_gate
            or (
                cfg.get("d2ae_performance_benchmark_path") in (None, "", False)
                and cfg.get("d2ae_performance_benchmark_sha256")
                in (None, "", False)
            )
        ),
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-AE sparse relation field contract mismatch: {failed}")
    performance_gate = (
        _validate_d2ae_performance_gate(cfg)
        if require_performance_gate else None
    )
    return {
        "run_id": run_id_contract,
        "performance_gate_required": require_performance_gate,
        "performance_gate": performance_gate,
    }


def _validate_author_update_execution_host(cfg: DictConfig) -> None:
    if not _uses_author_update_rule(cfg):
        return
    modes = (
        (_is_d2ae(cfg), "D2-AE"),
        (_is_d2ad(cfg), "D2-AD"),
        (_is_d2ac(cfg), "D2-AC"),
        (_is_d2ab(cfg), "D2-AB"),
        (_is_d2z(cfg), "D2-Z"),
        (_is_d2y(cfg), "D2-Y"),
        (_is_d2x(cfg), "D2-X"),
        (_is_d2v(cfg), "D2-V"),
        (_is_d2u(cfg), "D2-U"),
        (_is_d2t(cfg), "D2-T"),
    )
    label = next(name for enabled, name in modes if enabled)
    if socket.gethostname() != "node01":
        raise RuntimeError(f"{label} HOIPrior CUDA workload is restricted to infbagel-4gpu/node01")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError(f"{label} requires INFBAGEL_WORKER_EXPERT=hoi")
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    if not configured_python or not Path(configured_python).is_absolute():
        raise RuntimeError(f"{label} requires an absolute verified INFBAGEL_PYTHON")
    if Path(sys.executable).resolve() != Path(configured_python).resolve():
        raise RuntimeError(
            f"{label} must execute with the verified INFBAGEL_PYTHON: "
            f"{configured_python}"
        )


def _validate_d2t_execution_host(cfg: DictConfig) -> None:
    if not _is_d2t(cfg):
        return
    _validate_author_update_execution_host(cfg)


def _validate_d2u_execution_host(cfg: DictConfig) -> None:
    if not _is_d2u(cfg):
        return
    _validate_author_update_execution_host(cfg)


def _validate_d2v_execution_host(cfg: DictConfig) -> None:
    if not _is_d2v(cfg):
        return
    _validate_author_update_execution_host(cfg)


def _validate_d2x_execution_host(cfg: DictConfig) -> None:
    if not _is_d2x(cfg):
        return
    _validate_author_update_execution_host(cfg)


def _validate_d2y_execution_host(cfg: DictConfig) -> None:
    if not _is_d2y(cfg):
        return
    _validate_author_update_execution_host(cfg)


def _validate_d2z_execution_host(cfg: DictConfig) -> None:
    if not _is_d2z(cfg):
        return
    _validate_author_update_execution_host(cfg)


def _validate_d2ac_execution_host(cfg: DictConfig) -> None:
    if not _is_d2ac(cfg):
        return
    _validate_author_update_execution_host(cfg)


def _build_optimizer(
    cfg: DictConfig,
    parameters: Iterable[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    optimizer_class = Adam if _uses_author_update_rule(cfg) else AdamW
    return optimizer_class(
        parameters, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay),
        betas=(float(cfg.beta1), float(cfg.beta2)),
    )


def _build_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    warmup_updates: int,
) -> Optional[LambdaLR]:
    if _uses_author_update_rule(cfg):
        return None
    return LambdaLR(
        optimizer,
        lambda update: _lr_lambda(update, total_updates, warmup_updates, float(cfg.minimum_lr_ratio)),
    )


def _gradient_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    norms = [
        parameter.grad.detach().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return torch.tensor(float("nan"))
    return torch.stack(norms).norm(2)


def _d2ac_gradient_audit(model: torch.nn.Module, *, require_adapter_paths: bool) -> Dict[str, object]:
    adapter = model.network.interaction_adapter
    if adapter is None:
        raise ValueError("D2-AC gradient audit requires the interaction adapter")

    def record(name: str, value: Optional[torch.Tensor]) -> Dict[str, object]:
        finite = value is not None and bool(torch.isfinite(value).all())
        nonzero = value is not None and bool(torch.any(value != 0))
        norm = (
            float(value.detach().float().norm().item())
            if finite and value is not None
            else None
        )
        return {"name": name, "finite": finite, "nonzero": nonzero, "l2_norm": norm}

    alpha = record("alpha", adapter.alpha.grad)
    result: Dict[str, object] = {
        "alpha": alpha,
        "alpha_value": float(adapter.alpha.detach().item()),
        "gate_value": float(torch.tanh(adapter.alpha.detach()).item()),
    }
    if not alpha["finite"] or not alpha["nonzero"]:
        raise FloatingPointError("D2-AC alpha gradient must be finite and nonzero")
    if not require_adapter_paths:
        return result

    in_projection = adapter.cross_attention.in_proj_weight.grad
    qkv = [None, None, None]
    if in_projection is not None:
        qkv = list(in_projection.split(128, dim=0))
    groups = {
        "object_encoder": [
            parameter.grad for parameter in adapter.object_encoder.parameters()
        ],
        "object_identity": [adapter.object_identity.grad],
        "part_embedding": [adapter.part_embedding.grad],
        "query_projection": [
            parameter.grad for parameter in adapter.query_projection.parameters()
        ],
        "attention_q_projection": [qkv[0]],
        "attention_k_projection": [qkv[1]],
        "attention_v_projection": [qkv[2]],
        "attention_out_projection": [
            parameter.grad for parameter in adapter.cross_attention.out_proj.parameters()
        ],
        "writeback": [parameter.grad for parameter in adapter.writeback.parameters()],
    }
    group_records: Dict[str, object] = {}
    for name, gradients in groups.items():
        records = [record(f"{name}[{index}]", gradient) for index, gradient in enumerate(gradients)]
        group_records[name] = {
            "parameters": records,
            "finite": all(item["finite"] for item in records),
            "nonzero": all(item["nonzero"] for item in records),
        }
    result["adapter_groups"] = group_records
    failed = sorted(
        name for name, value in group_records.items()
        if not value["finite"] or not value["nonzero"]
    )
    if failed:
        raise FloatingPointError(
            f"D2-AC activated adapter gradients must be finite/nonzero: {failed}"
        )
    if result["gate_value"] == 0.0:
        raise FloatingPointError("D2-AC gate did not activate after the initial alpha update")
    return result


def _d2ae_gradient_audit(
    model: torch.nn.Module,
    *,
    require_relation_paths: bool,
) -> Dict[str, object]:
    field = model.network.sparse_relation_field
    if field is None:
        raise ValueError("D2-AE gradient audit requires the sparse relation field")

    def record(name: str, value: Optional[torch.Tensor]) -> Dict[str, object]:
        finite = value is not None and bool(torch.isfinite(value).all())
        nonzero = value is not None and bool(torch.any(value != 0))
        norm = (
            float(value.detach().float().norm().item())
            if finite and value is not None
            else None
        )
        return {"name": name, "finite": finite, "nonzero": nonzero, "l2_norm": norm}

    alpha = record("alpha", field.alpha.grad)
    result: Dict[str, object] = {
        "alpha": alpha,
        "alpha_value": float(field.alpha.detach().item()),
        "gate_value": float(torch.tanh(field.alpha.detach()).item()),
    }
    if not alpha["finite"] or not alpha["nonzero"]:
        raise FloatingPointError("D2-AE alpha gradient must be finite and nonzero")
    if not require_relation_paths:
        return result
    groups = {
        "point_encoder": [
            parameter.grad for parameter in field.point_encoder.parameters()
        ],
        "projection": [parameter.grad for parameter in field.projection.parameters()],
        "temporal_embeddings": [field.temporal_embeddings.grad],
        "relation_norm": [parameter.grad for parameter in field.relation_norm.parameters()],
        "motion_input": [model.network.motion_input.weight.grad],
        "trunk_layer_0": [
            model.network.transformer.layers[0].self_attn.in_proj_weight.grad
        ],
    }
    group_records: Dict[str, object] = {}
    for name, gradients in groups.items():
        records = [
            record(f"{name}[{index}]", gradient)
            for index, gradient in enumerate(gradients)
        ]
        group_records[name] = {
            "parameters": records,
            "finite": all(item["finite"] for item in records),
            "nonzero": all(item["nonzero"] for item in records),
        }
    result["relation_groups"] = group_records
    failed = sorted(
        name for name, value in group_records.items()
        if not value["finite"] or not value["nonzero"]
    )
    if failed:
        raise FloatingPointError(
            f"D2-AE activated relation gradients must be finite/nonzero: {failed}"
        )
    if result["gate_value"] == 0.0 and field._gate_override is None:
        raise FloatingPointError("D2-AE gate did not activate after the initial alpha update")
    return result


def _validate_d2ae_model_instance(
    model: torch.nn.Module,
    *,
    require_zero_alpha: bool,
) -> Dict[str, object]:
    field = model.network.sparse_relation_field
    if field is None:
        raise ValueError("D2-AE model instance is missing the sparse relation field")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    relation_parameters = sum(parameter.numel() for parameter in field.parameters())
    checks = {
        "base_parameters": total_parameters - relation_parameters == D2AE_BASE_PARAMETER_COUNT,
        "relation_parameters": relation_parameters == SPARSE_RELATION_PARAMETER_COUNT,
        "total_parameters": total_parameters == D2AE_TOTAL_PARAMETER_COUNT,
        "parameter_increase_below_limit": (
            relation_parameters / (total_parameters - relation_parameters) <= 0.015
        ),
        "diagnostic_variant_full": field._diagnostic_variant == "full",
        "gate_override_absent": field._gate_override is None,
        "capture_disabled": field._capture is False,
        "alpha_finite": bool(torch.isfinite(field.alpha.detach())),
        "alpha_zero_when_required": (
            not require_zero_alpha or float(field.alpha.detach().item()) == 0.0
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AE model instance contract mismatch: {failed}")
    return {
        "checks": checks,
        "base_parameters": total_parameters - relation_parameters,
        "relation_parameters": relation_parameters,
        "total_parameters": total_parameters,
        "parameter_increase_fraction": (
            relation_parameters / (total_parameters - relation_parameters)
        ),
    }


def _primary_validation_model(
    cfg: DictConfig,
    model: DistributedDataParallel,
    ema_models: Mapping[str, torch.nn.Module],
) -> torch.nn.Module:
    if _uses_author_update_rule(cfg):
        if ema_models:
            raise ValueError("author-update validation forbids EMA models")
        return model.module
    return ema_models["0.9999"]


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {
        key: batch[key].to(device, non_blocking=True)
        for key in (
            "x", "text_embedding", "object_bps", "goals", "progress", "rest_human_offsets",
            "terminal_window", "rest_object_points", "world_to_local_rotation",
            "object_rotation_reference",
        )
    }
    if "d2z_near_ground_gate" in batch:
        moved["d2z_near_ground_gate"] = batch["d2z_near_ground_gate"].to(
            device, non_blocking=True
        )
    if "d2ab_floor_m" in batch:
        moved["d2ab_floor_m"] = batch["d2ab_floor_m"].to(
            device, non_blocking=True
        )
    if "local_object_bps" in batch:
        moved["local_object_bps"] = batch["local_object_bps"].to(
            device, non_blocking=True
        )
    return moved


def _forward_losses(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: Dict[str, torch.Tensor],
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg: DictConfig,
    *,
    generator: Optional[torch.Generator] = None,
    audit_digest=None,
) -> Dict[str, torch.Tensor]:
    clean = batch["x"]
    timesteps = torch.randint(
        0, REPRESENTATION.diffusion_steps, (clean.shape[0],), device=clean.device, generator=generator,
    )
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    if audit_digest is not None:
        audit_digest.update(
            clean[:, (0, -1), :8].detach().contiguous().cpu().numpy().tobytes()
        )
        audit_digest.update(timesteps.detach().contiguous().cpu().numpy().tobytes())
        audit_digest.update(
            noise[:, (0, -1), :8].detach().contiguous().cpu().numpy().tobytes()
        )
    noisy = diffusion.q_sample(clean, timesteps, noise)
    model_arguments = (
        noisy,
        timesteps,
        batch["text_embedding"],
        batch["object_bps"],
        batch["goals"],
        normalize_progress(batch["progress"]),
    )
    if _is_d2ad(cfg):
        prediction = model(
            *model_arguments,
            local_object_bps=batch["local_object_bps"],
        )
    elif _is_d2ae(cfg):
        prediction = model(
            *model_arguments,
            rest_object_points=batch["rest_object_points"],
            world_to_local_rotation=batch["world_to_local_rotation"],
            object_rotation_reference=batch["object_rotation_reference"],
            position_minimum=minimum,
            position_maximum=maximum,
            object_minimum=object_minimum,
            object_maximum=object_maximum,
        )
    else:
        prediction = model(*model_arguments)
    positional = (
        prediction,
        clean,
        batch["goals"],
        batch["rest_human_offsets"],
        parents,
        minimum,
        maximum,
        object_minimum,
        object_maximum,
        batch["terminal_window"],
        batch["rest_object_points"],
        batch["world_to_local_rotation"],
        batch["object_rotation_reference"],
    )
    weights = {
        "fk_weight": float(cfg.fk_weight),
        "object_surface_weight": float(cfg.object_surface_weight),
        "velocity_weight": float(cfg.velocity_weight),
        "goal_weight": float(cfg.goal_weight),
    }
    if _is_d2z(cfg):
        return d2z_hoi_training_losses(
            *positional,
            batch["d2z_near_ground_gate"],
            **weights,
        )
    if _is_d2ab(cfg):
        return d2ab_hoi_training_losses(
            *positional,
            batch["d2ab_floor_m"],
            **weights,
        )
    return hoi_training_losses(
        *positional,
        **weights,
        fk_foot_temporal_routing=bool(cfg.get("fk_foot_temporal_routing", False)),
        routed_foot_residual_multiplier=float(
            cfg.get("routed_foot_residual_multiplier", 1.0)
        ),
    )


@torch.no_grad()
def _ema_update(ema: torch.nn.Module, source: torch.nn.Module, decay: float) -> None:
    for target_parameter, source_parameter in zip(ema.parameters(), source.parameters()):
        target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
    for target_buffer, source_buffer in zip(ema.buffers(), source.buffers()):
        target_buffer.copy_(source_buffer)


@torch.no_grad()
def _validate(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg: DictConfig,
    processed_windows: int,
) -> Dict[str, object]:
    was_training = model.training
    model.eval()
    local_limit = int(cfg.validation_windows) // world_size
    if local_limit * world_size != int(cfg.validation_windows):
        raise ValueError("validation_windows must be divisible by world size")
    totals = torch.zeros(len(LOSS_KEYS) + 1, dtype=torch.float64, device=minimum.device)
    generator = torch.Generator(device=minimum.device)
    generator.manual_seed(int(cfg.seed) * 1000003 + processed_windows + rank)
    seen = 0
    while seen < local_limit:
        previous_seen = seen
        for raw_batch in loader:
            if seen >= local_limit:
                break
            batch = _move_batch(raw_batch, minimum.device)
            remaining = local_limit - seen
            if batch["x"].shape[0] > remaining:
                batch = {key: value[:remaining] for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                losses = _forward_losses(
                    model, diffusion, batch, parents, minimum, maximum,
                    object_minimum, object_maximum, cfg, generator=generator,
                )
            count = batch["x"].shape[0]
            for index, key in enumerate(LOSS_KEYS):
                totals[index] += losses[key].detach().double() * count
            totals[-1] += count
            seen += count
        if seen == previous_seen:
            raise RuntimeError("internal validation loader produced no batches")
    torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    if int(totals[-1].item()) != int(cfg.validation_windows):
        raise RuntimeError(f"validated {int(totals[-1])} windows, expected {cfg.validation_windows}")
    result = {
        key: float((totals[index] / totals[-1]).item()) for index, key in enumerate(LOSS_KEYS)
    }
    result.update({
        "processed_windows": processed_windows,
        "validation_windows": int(totals[-1].item()),
        "finite": all(math.isfinite(value) for value in result.values()),
    })
    model.train(was_training)
    return result


def _rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng(value: Dict[str, object]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state(value["cuda"])


def _checkpoint_value(
    cfg: DictConfig,
    model: DistributedDataParallel,
    ema_models: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: torch.cuda.amp.GradScaler,
    *,
    world_size: int,
    processed_windows: int,
    optimizer_updates: int,
    amp_overflow_skips: int,
    epoch: int,
    batches_consumed_in_epoch: int,
    rng_pattern: str,
    weight_initialization: Mapping[str, object],
) -> Dict[str, object]:
    value = {
        "schema_version": 2,
        "checkpoint_type": "hoi_prior_phase1b",
        "window_state_codec": "state-compositional-v1",
        "expert": "hoi",
        "initialization": "random",
        "run_id": str(cfg.run_id),
        "seed": int(cfg.seed),
        "git_commit": _git_commit(Path(str(cfg.repo_root)).resolve()),
        "processed_windows": processed_windows,
        "processed_frames": processed_windows * REPRESENTATION.window_frames,
        "optimizer_updates": optimizer_updates,
        "amp_overflow_skips": amp_overflow_skips,
        "epoch": epoch,
        "batches_consumed_in_epoch": batches_consumed_in_epoch,
        "world_size": world_size,
        "effective_batch_size": int(cfg.effective_batch_size),
        "model_config": _model_config(cfg),
        "resume_contract": _resume_contract(cfg),
        "data_contract_sha256": str(cfg.data_contract_sha256),
        "split_sha256": _sha256(Path(str(cfg.split_manifest)).resolve()),
        "model": model.module.state_dict(),
        "ema_models": {key: item.state_dict() for key, item in ema_models.items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": scaler.state_dict() if bool(cfg.amp) else None,
        "optimization_contract": _optimization_contract(cfg),
        "primary_weight_variant": str(
            cfg.get(
                "primary_weight_variant",
                "online" if _uses_author_update_rule(cfg) else "ema_0.9999",
            )
        ),
        "rng_pattern": rng_pattern,
        "weight_initialization": dict(weight_initialization),
    }
    if _is_d2ac(cfg):
        value["architecture_variant"] = HOI_ARCHITECTURE_D2AC
        value["interaction_adapter_contract"] = {
            "bps_sha256": D2AC_BPS_SHA256,
            "assignment_sha256": ASSIGNMENT_SHA256,
            "adapter_parameters": ADAPTER_PARAMETER_COUNT,
            "alpha_initial": 0.0,
            "placement": "after_transformer_layer_4_before_layers_5_to_8",
        }
    if _is_d2ad(cfg):
        value["architecture_variant"] = HOI_ARCHITECTURE_D2AD
        value["interaction_adapter_contract"] = {
            "bps_sha256": D2AC_BPS_SHA256,
            "assignment_sha256": ASSIGNMENT_SHA256,
            "adapter_parameters": ADAPTER_PARAMETER_COUNT,
            "alpha_initial": 0.0,
            "placement": "after_transformer_layer_4_before_layers_5_to_8",
            "basis_coordinate_system": LOCAL_BASIS_COORDINATE_SYSTEM,
            "basis_yup_tensor_sha256": BPS_YUP_TENSOR_SHA256,
            "rest_mesh_manifest_sha256": REST_MESH_MANIFEST_SHA256,
            "object_mapping_sha256": OBJECT_MAPPING_SHA256,
            "query_backend": "scipy.spatial.cKDTree.query",
            "query_parameters": {"k": 1, "eps": 0.0, "p": 2},
            "query_workers": int(cfg.local_bps_query_workers),
            "full_rest_mesh": True,
            "mesh_subsample": False,
            "stored_per_window_local_bps": False,
        }
    if _is_d2ae(cfg):
        value["architecture_variant"] = HOI_ARCHITECTURE_D2AE
        value["sparse_relation_contract"] = (
            model.module.network.sparse_relation_field.contract_metadata()
        )
    if ema_models:
        # Retain the legacy name for pre-D2-T official evaluator compatibility.
        value["ema_model"] = ema_models["0.9999"].state_dict()
    return value


def _save_checkpoint(
    rank: int,
    world_size: int,
    cfg: DictConfig,
    model: DistributedDataParallel,
    ema_models: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: torch.cuda.amp.GradScaler,
    *,
    processed_windows: int,
    optimizer_updates: int,
    amp_overflow_skips: int,
    epoch: int,
    batches_consumed_in_epoch: int,
    checkpoint_hashes: List[Dict[str, object]],
    weight_initialization: Mapping[str, object],
) -> Path:
    checkpoint_dir = Path(str(cfg.checkpoint_dir)).resolve()
    checkpoint_path = checkpoint_dir / f"{cfg.run_id}_windows{processed_windows:09d}.pth"
    rng_path = checkpoint_dir / f"{checkpoint_path.stem}.rank{rank}.rng.pth"
    _atomic_torch_save(rng_path, _rng_state())
    torch.distributed.barrier()
    if rank == 0:
        if checkpoint_path.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {checkpoint_path}")
        value = _checkpoint_value(
            cfg, model, ema_models, optimizer, scheduler, scaler,
            world_size=world_size,
            processed_windows=processed_windows,
            optimizer_updates=optimizer_updates,
            amp_overflow_skips=amp_overflow_skips,
            epoch=epoch,
            batches_consumed_in_epoch=batches_consumed_in_epoch,
            rng_pattern=f"{checkpoint_path.stem}.rank{{rank}}.rng.pth",
            weight_initialization=weight_initialization,
        )
        _atomic_torch_save(checkpoint_path, value)
        checkpoint_hashes.append({
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "processed_windows": processed_windows,
        })
    torch.distributed.barrier()
    return checkpoint_path


def _validate_d2ae_random_origin_checkpoint(
    checkpoint: Mapping[str, object],
    expected_initial_model_state_sha256: str,
) -> Dict[str, object]:
    """Reject a D2-AE resume artifact without exact random-origin provenance."""
    initialization = checkpoint.get("weight_initialization")
    expected_initialization_keys = {
        "mode",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_model_state_sha256",
        "initial_model_state_sha256",
        "restored_components",
        "old_optimizer_states_loaded",
        "old_ema_models_loaded",
        "old_scheduler_states_loaded",
        "old_scaler_states_loaded",
        "old_rng_states_loaded",
    }
    checks = {
        "schema_version": checkpoint.get("schema_version") == 2,
        "checkpoint_type": checkpoint.get("checkpoint_type") == "hoi_prior_phase1b",
        "window_state_codec": checkpoint.get("window_state_codec")
        == "state-compositional-v1",
        "expert": checkpoint.get("expert") == "hoi",
        "initialization": checkpoint.get("initialization") == "random",
        "architecture_variant": checkpoint.get("architecture_variant")
        == HOI_ARCHITECTURE_D2AE,
        "weight_initialization_mapping": isinstance(initialization, Mapping),
        "weight_initialization_keys": (
            isinstance(initialization, Mapping)
            and set(initialization) == expected_initialization_keys
        ),
        "weight_initialization_mode": (
            isinstance(initialization, Mapping)
            and initialization.get("mode") == "random"
        ),
        "weight_initialization_sources_absent": (
            isinstance(initialization, Mapping)
            and all(
                initialization.get(name) is None
                for name in (
                    "source_checkpoint",
                    "source_checkpoint_sha256",
                    "source_model_state_sha256",
                )
            )
        ),
        "restored_components_empty": (
            isinstance(initialization, Mapping)
            and initialization.get("restored_components") == []
        ),
        "old_state_load_counts_zero": (
            isinstance(initialization, Mapping)
            and all(
                initialization.get(name) == 0
                for name in (
                    "old_optimizer_states_loaded",
                    "old_ema_models_loaded",
                    "old_scheduler_states_loaded",
                    "old_scaler_states_loaded",
                    "old_rng_states_loaded",
                )
            )
        ),
        "initial_model_state_sha256_exact": (
            isinstance(initialization, Mapping)
            and initialization.get("initial_model_state_sha256")
            == expected_initial_model_state_sha256
        ),
        "ema_absent": checkpoint.get("ema_models") == {},
        "primary_online": checkpoint.get("primary_weight_variant") == "online",
        "model_present": isinstance(checkpoint.get("model"), Mapping),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "D2-AE resume checkpoint random-origin provenance mismatch: "
            + ", ".join(failed)
        )
    return {"checks": checks, "weight_initialization": dict(initialization)}


def _load_resume(
    rank: int,
    cfg: DictConfig,
    model: DistributedDataParallel,
    ema_models: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: torch.cuda.amp.GradScaler,
) -> Dict[str, object]:
    path = Path(str(cfg.resume_checkpoint)).resolve()
    checkpoint = torch.load(path, map_location=f"cuda:{rank}")
    if checkpoint.get("checkpoint_type") != "hoi_prior_phase1b":
        raise ValueError("resume checkpoint is not a Phase 1B HOIPrior checkpoint")
    if _is_d2ac(cfg):
        adapter_contract = checkpoint.get("interaction_adapter_contract")
        if (
            checkpoint.get("architecture_variant") != HOI_ARCHITECTURE_D2AC
            or not isinstance(adapter_contract, dict)
            or adapter_contract.get("bps_sha256") != D2AC_BPS_SHA256
            or adapter_contract.get("assignment_sha256") != ASSIGNMENT_SHA256
            or adapter_contract.get("adapter_parameters") != ADAPTER_PARAMETER_COUNT
        ):
            raise ValueError("D2-AC resume checkpoint architecture/provenance mismatch")
    elif _is_d2ad(cfg):
        adapter_contract = checkpoint.get("interaction_adapter_contract")
        if (
            checkpoint.get("architecture_variant") != HOI_ARCHITECTURE_D2AD
            or not isinstance(adapter_contract, dict)
            or adapter_contract.get("bps_sha256") != D2AC_BPS_SHA256
            or adapter_contract.get("assignment_sha256") != ASSIGNMENT_SHA256
            or adapter_contract.get("adapter_parameters") != ADAPTER_PARAMETER_COUNT
            or adapter_contract.get("basis_coordinate_system")
            != LOCAL_BASIS_COORDINATE_SYSTEM
            or adapter_contract.get("basis_yup_tensor_sha256")
            != BPS_YUP_TENSOR_SHA256
            or adapter_contract.get("rest_mesh_manifest_sha256")
            != REST_MESH_MANIFEST_SHA256
            or adapter_contract.get("object_mapping_sha256")
            != OBJECT_MAPPING_SHA256
            or adapter_contract.get("query_backend")
            != "scipy.spatial.cKDTree.query"
            or adapter_contract.get("query_parameters")
            != {"k": 1, "eps": 0.0, "p": 2}
            or adapter_contract.get("query_workers")
            != int(cfg.local_bps_query_workers)
            or adapter_contract.get("full_rest_mesh") is not True
            or adapter_contract.get("mesh_subsample") is not False
            or adapter_contract.get("stored_per_window_local_bps") is not False
        ):
            raise ValueError("D2-AD resume checkpoint architecture/provenance mismatch")
    elif _is_d2ae(cfg):
        _validate_d2ae_random_origin_checkpoint(
            checkpoint,
            _state_dict_sha256(model.module.state_dict()),
        )
        try:
            validate_sparse_relation_contract(
                checkpoint.get("sparse_relation_contract")
            )
        except ValueError as error:
            raise ValueError(
                "D2-AE resume checkpoint architecture/provenance mismatch"
            ) from error
    elif checkpoint.get("architecture_variant") in {
        HOI_ARCHITECTURE_D2AC,
        HOI_ARCHITECTURE_D2AD,
        HOI_ARCHITECTURE_D2AE,
    }:
        raise ValueError("variant checkpoint cannot resume a base HOIPrior run")
    repo = Path(str(cfg.repo_root)).resolve()
    current_commit = _git_commit(repo)
    checkpoint_commit = str(checkpoint.get("git_commit"))
    resume_provenance = _resume_commit_provenance(
        cfg, checkpoint_commit, current_commit, repo,
    )
    if checkpoint.get("resume_contract") != _resume_contract(cfg):
        raise ValueError("resume checkpoint training contract mismatch")
    for key, expected in (
        ("run_id", str(cfg.run_id)),
        ("seed", int(cfg.seed)),
        ("world_size", int(cfg.num_gpus)),
        ("effective_batch_size", int(cfg.effective_batch_size)),
        ("model_config", _model_config(cfg)),
        ("data_contract_sha256", str(cfg.data_contract_sha256)),
    ):
        if checkpoint.get(key) != expected:
            raise ValueError(f"resume checkpoint {key} mismatch: {checkpoint.get(key)!r} != {expected!r}")
    model.module.load_state_dict(checkpoint["model"], strict=True)
    checkpoint_emas = checkpoint.get("ema_models")
    if not isinstance(checkpoint_emas, dict) or set(checkpoint_emas) != set(ema_models):
        raise ValueError("resume checkpoint EMA variants mismatch")
    for key, ema in ema_models.items():
        ema.load_state_dict(checkpoint_emas[key], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint_scheduler = checkpoint.get("scheduler")
    if scheduler is None:
        if checkpoint_scheduler is not None:
            raise ValueError("resume checkpoint unexpectedly contains scheduler state")
    else:
        if not isinstance(checkpoint_scheduler, dict):
            raise ValueError("resume checkpoint is missing scheduler state")
        scheduler.load_state_dict(checkpoint_scheduler)
    checkpoint_scaler = checkpoint.get("scaler")
    if bool(cfg.amp):
        if not isinstance(checkpoint_scaler, dict):
            raise ValueError("resume checkpoint is missing AMP scaler state")
        scaler.load_state_dict(checkpoint_scaler)
    elif checkpoint_scaler is not None:
        raise ValueError("FP32 resume checkpoint unexpectedly contains AMP scaler state")
    rng_path = path.parent / checkpoint["rng_pattern"].format(rank=rank)
    _restore_rng(torch.load(rng_path, map_location="cpu"))
    return {
        "processed_windows": int(checkpoint["processed_windows"]),
        "optimizer_updates": int(checkpoint["optimizer_updates"]),
        "amp_overflow_skips": int(checkpoint.get("amp_overflow_skips", 0)),
        "epoch": int(checkpoint["epoch"]),
        "batches_consumed_in_epoch": int(checkpoint["batches_consumed_in_epoch"]),
        "resume_checkpoint_git_commit": checkpoint_commit,
        "resume_commit_provenance": resume_provenance,
    }


def _worker(rank: int, cfg: DictConfig) -> None:
    world_size = int(cfg.num_gpus)
    _validate_fk_foot_temporal_routing_mode(cfg)
    _validate_d2t_contract(cfg, world_size)
    _validate_d2u_contract(cfg, world_size)
    _validate_d2v_contract(cfg, world_size)
    _validate_d2x_contract(cfg, world_size)
    _validate_d2y_contract(cfg, world_size)
    _validate_d2z_contract(cfg, world_size)
    _validate_d2ab_contract(cfg, world_size)
    _validate_d2ac_contract(cfg, world_size)
    _validate_d2ad_contract(cfg, world_size)
    d2ae_lifecycle_contract = _validate_d2ae_contract(cfg, world_size)
    _validate_author_update_execution_host(cfg)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    seed = int(cfg.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    effective = int(cfg.batch_size) * world_size * int(cfg.gradient_accumulation_steps)
    if effective != int(cfg.effective_batch_size):
        raise ValueError(
            f"effective batch mismatch: {cfg.batch_size} x {world_size} x "
            f"{cfg.gradient_accumulation_steps} = {effective}, configured {cfg.effective_batch_size}"
        )
    if int(cfg.max_processed_windows) % effective:
        raise ValueError("max_processed_windows must be divisible by effective_batch_size")
    if int(cfg.warmup_windows) % effective:
        raise ValueError("warmup_windows must be divisible by effective_batch_size")
    if int(cfg.window_frames) != REPRESENTATION.window_frames:
        raise ValueError("HOIPrior window_frames contract mismatch")
    if int(cfg.history_frames) != REPRESENTATION.history_frames:
        raise ValueError("HOIPrior history_frames contract mismatch")
    if int(cfg.diffusion_steps) != REPRESENTATION.diffusion_steps:
        raise ValueError("HOIPrior diffusion_steps contract mismatch")
    if int(cfg.max_consecutive_amp_overflows) < 0:
        raise ValueError("max_consecutive_amp_overflows must be non-negative")
    if {
        "dim_model": int(cfg.dim_model),
        "num_heads": int(cfg.num_heads),
        "num_layers": int(cfg.num_layers),
    } != {"dim_model": 512, "num_heads": 16, "num_layers": 8}:
        raise ValueError("HOIPrior architecture must remain 512-wide, 16-head, 8-layer")
    if cfg.init_checkpoint not in (None, "", False):
        raise ValueError("HOIPrior training initialization must be random; init_checkpoint is forbidden")
    d2m_candidate = None if cfg.d2m_candidate in (None, "", False) else str(cfg.d2m_candidate)
    configured_weights = {
        "fk": float(cfg.fk_weight),
        "object_surface": float(cfg.object_surface_weight),
        "velocity": float(cfg.velocity_weight),
        "terminal_goal": float(cfg.goal_weight),
    }
    if d2m_candidate is None:
        locked_weights = _locked_loss_weights(cfg)
        if configured_weights != locked_weights:
            raise ValueError(
                "Phase 1B loss weights do not match the registered training mode"
            )
        if cfg.weight_init_checkpoint not in (None, "", False):
            raise ValueError("weight-only initialization requires a registered D2-M candidate")
    else:
        from priors.optimizer_reset import (
            CANDIDATES,
            EFFECTIVE_BATCH_SIZE,
            OPTIMIZER_UPDATES,
            PROCESSED_WINDOWS,
            SOURCE_OPTIMIZER_LR,
            WEIGHTS,
        )

        if d2m_candidate not in CANDIDATES or configured_weights != WEIGHTS[d2m_candidate]:
            raise ValueError("D2-M candidate loss weights do not match the locked contract")
        if (
            world_size != 4
            or int(cfg.batch_size) != 768
            or int(cfg.gradient_accumulation_steps) != 1
            or int(cfg.effective_batch_size) != EFFECTIVE_BATCH_SIZE
            or int(cfg.max_processed_windows) != PROCESSED_WINDOWS
            or int(cfg.max_processed_windows) // int(cfg.effective_batch_size) != OPTIMIZER_UPDATES
            or float(cfg.learning_rate) != SOURCE_OPTIMIZER_LR
            or int(cfg.warmup_windows) != 0
            or float(cfg.minimum_lr_ratio) != 1.0
            or not bool(cfg.d2m_rng_audit)
        ):
            raise ValueError("D2-M optimizer-reset smoke budget/LR/RNG contract mismatch")
    if world_size not in {1, 4}:
        raise ValueError("Phase 1B supports one-GPU functional smoke or four-GPU worker execution")

    split_manifest = str(Path(str(cfg.split_manifest)).resolve())
    d2z_gate_audit_path = (
        str(Path(str(cfg.d2z_gate_audit_path)).resolve()) if _is_d2z(cfg) else None
    )
    d2z_gate_audit_sha256 = str(cfg.d2z_gate_audit_sha256) if _is_d2z(cfg) else None
    d2ab_support_metadata_path = (
        str(Path(str(cfg.d2ab_support_metadata_path)).resolve())
        if _is_d2ab(cfg) else None
    )
    d2ab_support_metadata_sha256 = (
        str(cfg.d2ab_support_metadata_sha256) if _is_d2ab(cfg) else None
    )
    dataset_class = (
        D2ADPriorWindowDataset if _is_d2ad(cfg)
        else (
            D2ZPriorWindowDataset if _is_d2z(cfg)
            else (
                D2ABPriorWindowDataset if _is_d2ab(cfg)
                else PriorWindowDataset
            )
        )
    )
    d2z_dataset_kwargs = (
        {
            "gate_audit_path": d2z_gate_audit_path,
            "gate_audit_sha256": d2z_gate_audit_sha256,
        }
        if _is_d2z(cfg) else {}
    )
    d2ab_dataset_kwargs = (
        {
            "support_metadata_path": d2ab_support_metadata_path,
            "support_metadata_sha256": d2ab_support_metadata_sha256,
        }
        if _is_d2ab(cfg) else {}
    )
    train_dataset = dataset_class(
        str(cfg.repo_root), "hoi", partition="train", limit=int(cfg.dataset_limit),
        split_manifest=split_manifest, **d2z_dataset_kwargs, **d2ab_dataset_kwargs,
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(cfg.seed), drop_last=True,
    )
    train_collator = (
        D2ADBatchCollator(
            str(cfg.repo_root),
            query_workers=int(cfg.local_bps_query_workers),
        )
        if _is_d2ad(cfg) else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        sampler=train_sampler,
        drop_last=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.num_workers) > 0,
        collate_fn=train_collator,
    )
    validation_loader = None
    if int(cfg.validation_windows):
        validation_dataset = dataset_class(
            str(cfg.repo_root), "hoi", partition="internal_validation",
            split_manifest=split_manifest, **d2z_dataset_kwargs, **d2ab_dataset_kwargs,
        )
        validation_sampler = DistributedSampler(
            validation_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
        )
        validation_collator = (
            D2ADBatchCollator(
                str(cfg.repo_root),
                query_workers=int(cfg.local_bps_query_workers),
            )
            if _is_d2ad(cfg) else None
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(cfg.validation_batch_size),
            sampler=validation_sampler,
            drop_last=False,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
            persistent_workers=int(cfg.num_workers) > 0,
            collate_fn=validation_collator,
        )

    model = build_expert(
        "hoi", init_checkpoint=cfg.init_checkpoint, dim_model=int(cfg.dim_model),
        num_heads=int(cfg.num_heads), num_layers=int(cfg.num_layers),
        architecture_variant=str(
            cfg.get("hoi_architecture_variant", HOI_ARCHITECTURE_BASE)
        ),
    ).to(device)
    weight_initialization = _load_weight_initialization(cfg, model)
    model = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
    ema_decays = [float(value) for value in cfg.ema_decays]
    if not _uses_author_update_rule(cfg) and ema_decays != [0.999, 0.9999]:
        raise ValueError("Phase 1B remediation requires EMA decays 0.999 and 0.9999")
    if _uses_author_update_rule(cfg) and ema_decays:
        raise ValueError("author-update modes forbid EMA models")
    ema_models = {
        str(decay): copy.deepcopy(model.module).requires_grad_(False).eval()
        for decay in ema_decays
    }
    diffusion = GaussianDiffusion(int(cfg.diffusion_steps)).to(device)
    optimizer = _build_optimizer(cfg, model.parameters())
    total_updates = int(cfg.max_processed_windows) // effective
    warmup_updates = int(cfg.warmup_windows) // effective
    scheduler = _build_scheduler(cfg, optimizer, total_updates, warmup_updates)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp))
    initial_optimizer_state_count = len(optimizer.state)
    parents = torch.as_tensor(get_smpl_parents(use_joints24=True), device=device, dtype=torch.long)
    norm = np.load(Path(str(cfg.repo_root)) / "data/train/norm.npy")
    minimum = torch.as_tensor(norm[0], device=device, dtype=torch.float32)
    maximum = torch.as_tensor(norm[1], device=device, dtype=torch.float32)
    object_minimum = torch.as_tensor(norm[2], device=device, dtype=torch.float32)
    object_maximum = torch.as_tensor(norm[3], device=device, dtype=torch.float32)

    state = {
        "processed_windows": 0,
        "optimizer_updates": 0,
        "amp_overflow_skips": 0,
        "epoch": 0,
        "batches_consumed_in_epoch": 0,
        "resume_checkpoint_git_commit": None,
        "resume_commit_provenance": None,
    }
    resumed_from = None
    if cfg.resume_checkpoint not in (None, "", False):
        state = _load_resume(rank, cfg, model, ema_models, optimizer, scheduler, scaler)
        resumed_from = str(Path(str(cfg.resume_checkpoint)).resolve())
    d2ae_model_contract = (
        _validate_d2ae_model_instance(
            model.module,
            require_zero_alpha=resumed_from is None,
        )
        if _is_d2ae(cfg) else None
    )
    processed_windows = state["processed_windows"]
    optimizer_updates = state["optimizer_updates"]
    amp_overflow_skips = state["amp_overflow_skips"]
    start_epoch = state["epoch"]
    resume_batches = state["batches_consumed_in_epoch"]

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    compute_update_seconds: List[float] = []
    loss_sums = {key: 0.0 for key in LOSS_KEYS}
    loss_observations = 0
    pending_loss_sums = {key: 0.0 for key in LOSS_KEYS}
    pending_loss_observations = 0
    consecutive_amp_overflows = 0
    initial_grad_scale = float(scaler.get_scale())
    validation_records: List[Dict[str, object]] = []
    checkpoint_hashes: List[Dict[str, object]] = []
    training_rng_digest = hashlib.sha256() if bool(cfg.d2m_rng_audit) else None
    interaction_gradient_audit_path = (
        Path(str(cfg.output_dir)).resolve() / "interaction_gradient_audit.json"
    )
    interaction_gradient_audit: Dict[str, object] = {}
    if (_is_d2ac(cfg) or _is_d2ad(cfg)) and interaction_gradient_audit_path.is_file():
        interaction_gradient_audit = json.loads(
            interaction_gradient_audit_path.read_text(encoding="utf-8")
        )
    sparse_relation_gradient_audit_path = (
        Path(str(cfg.output_dir)).resolve() / "sparse_relation_gradient_audit.json"
    )
    sparse_relation_gradient_audit: Dict[str, object] = {}
    if _is_d2ae(cfg) and sparse_relation_gradient_audit_path.is_file():
        sparse_relation_gradient_audit = json.loads(
            sparse_relation_gradient_audit_path.read_text(encoding="utf-8")
        )
    local_bps_build_seconds = 0.0
    local_bps_build_batches = 0
    optimizer.zero_grad(set_to_none=True)
    paused = False
    last_checkpoint_windows = -1
    epoch = start_epoch
    batches_consumed = resume_batches

    while processed_windows < int(cfg.max_processed_windows):
        train_sampler.set_epoch(epoch)
        micro_in_accumulation = 0
        group_start = None
        for batch_index, raw_batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batches:
                continue
            if _is_d2ad(cfg):
                local_bps_build_seconds += float(raw_batch["local_bps_build_seconds"])
                local_bps_build_batches += 1
            if micro_in_accumulation == 0 and bool(cfg.profile_every_update):
                torch.cuda.synchronize(device)
                group_start = time.perf_counter()
            batch = _move_batch(raw_batch, device)
            boundary = micro_in_accumulation + 1 == int(cfg.gradient_accumulation_steps)
            sync_context = contextlib.nullcontext() if boundary else model.no_sync()
            with sync_context:
                with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                    losses = _forward_losses(
                        model, diffusion, batch, parents, minimum, maximum,
                        object_minimum, object_maximum, cfg,
                        audit_digest=training_rng_digest,
                    )
                    loss = losses["total"] / int(cfg.gradient_accumulation_steps)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite HOIPrior loss at update {optimizer_updates}")
                if bool(cfg.amp):
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            for key in LOSS_KEYS:
                pending_loss_sums[key] += float(losses[key].detach())
            pending_loss_observations += 1
            micro_in_accumulation += 1
            batches_consumed = batch_index + 1
            if not boundary:
                continue

            if bool(cfg.amp):
                scaler.unscale_(optimizer)
            key_gradient = model.module.network.motion_input.weight.grad
            if key_gradient is None:
                raise FloatingPointError("missing key HOIPrior gradient")
            local_nonfinite = torch.tensor(
                int(any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )),
                dtype=torch.int32,
                device=device,
            )
            torch.distributed.all_reduce(local_nonfinite, op=torch.distributed.ReduceOp.MAX)
            if int(local_nonfinite.item()):
                if not bool(cfg.amp):
                    raise FloatingPointError("non-finite HOIPrior gradient with AMP disabled")
                amp_overflow_skips += 1
                consecutive_amp_overflows += 1
                scaler.update(new_scale=float(scaler.get_scale()) * 0.5)
                optimizer.zero_grad(set_to_none=True)
                pending_loss_sums = {key: 0.0 for key in LOSS_KEYS}
                pending_loss_observations = 0
                micro_in_accumulation = 0
                group_start = None
                if consecutive_amp_overflows > int(cfg.max_consecutive_amp_overflows):
                    raise FloatingPointError(
                        f"AMP gradient overflow persisted for {consecutive_amp_overflows} groups"
                    )
                continue
            if not torch.any(key_gradient != 0):
                raise FloatingPointError("zero key HOIPrior gradient")
            if _is_d2ac(cfg) or _is_d2ad(cfg):
                if (
                    optimizer_updates == 0
                    and "initial_zero_gate_alpha_gradient" not in interaction_gradient_audit
                ):
                    interaction_gradient_audit["initial_zero_gate_alpha_gradient"] = (
                        _d2ac_gradient_audit(
                            model.module, require_adapter_paths=False,
                        )
                    )
                elif (
                    optimizer_updates == 1
                    and "activated_adapter_gradients" not in interaction_gradient_audit
                ):
                    interaction_gradient_audit["activated_adapter_gradients"] = (
                        _d2ac_gradient_audit(
                            model.module, require_adapter_paths=True,
                        )
                    )
                    interaction_gradient_audit.update({
                        "schema_version": 1,
                        "run_id": str(cfg.run_id),
                        "seed": int(cfg.seed),
                        "optimizer_updates_observed": [0, 1],
                        "probe_or_override_used": False,
                    })
                    if rank == 0:
                        _atomic_json(
                            interaction_gradient_audit_path,
                            interaction_gradient_audit,
                        )
            if _is_d2ae(cfg):
                if (
                    optimizer_updates == 0
                    and "initial_zero_gate_alpha_gradient"
                    not in sparse_relation_gradient_audit
                ):
                    sparse_relation_gradient_audit[
                        "initial_zero_gate_alpha_gradient"
                    ] = _d2ae_gradient_audit(
                        model.module,
                        require_relation_paths=False,
                    )
                elif (
                    optimizer_updates == 1
                    and "activated_relation_gradients"
                    not in sparse_relation_gradient_audit
                ):
                    sparse_relation_gradient_audit[
                        "activated_relation_gradients"
                    ] = _d2ae_gradient_audit(
                        model.module,
                        require_relation_paths=True,
                    )
                    sparse_relation_gradient_audit.update({
                        "schema_version": 1,
                        "run_id": str(cfg.run_id),
                        "seed": int(cfg.seed),
                        "optimizer_updates_observed": [0, 1],
                        "probe_or_override_used": False,
                    })
                    if rank == 0:
                        _atomic_json(
                            sparse_relation_gradient_audit_path,
                            sparse_relation_gradient_audit,
                        )
            if bool(cfg.get("gradient_clipping", True)):
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.gradient_clip_norm))
            else:
                gradient_norm = _gradient_l2_norm(model.parameters())
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite HOIPrior gradient norm")
            if bool(cfg.amp):
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            for decay in ema_decays:
                _ema_update(ema_models[str(decay)], model.module, decay)
            optimizer_updates += 1
            processed_windows += effective
            for key in LOSS_KEYS:
                loss_sums[key] += pending_loss_sums[key]
                pending_loss_sums[key] = 0.0
            loss_observations += pending_loss_observations
            pending_loss_observations = 0
            consecutive_amp_overflows = 0
            micro_in_accumulation = 0
            if group_start is not None:
                torch.cuda.synchronize(device)
                compute_update_seconds.append(time.perf_counter() - group_start)

            if (
                validation_loader is not None
                and int(cfg.validation_interval_windows)
                and processed_windows % int(cfg.validation_interval_windows) == 0
            ):
                validation_records.append(_validate(
                    rank, world_size, _primary_validation_model(cfg, model, ema_models), diffusion,
                    validation_loader, parents,
                    minimum, maximum, object_minimum, object_maximum, cfg, processed_windows,
                ))
            if (
                int(cfg.checkpoint_interval_windows)
                and processed_windows % int(cfg.checkpoint_interval_windows) == 0
            ):
                _save_checkpoint(
                    rank, world_size, cfg, model, ema_models, optimizer, scheduler, scaler,
                    processed_windows=processed_windows,
                    optimizer_updates=optimizer_updates,
                    amp_overflow_skips=amp_overflow_skips,
                    epoch=epoch,
                    batches_consumed_in_epoch=batches_consumed,
                    checkpoint_hashes=checkpoint_hashes,
                    weight_initialization=weight_initialization,
                )
                last_checkpoint_windows = processed_windows
            pause_at = int(cfg.pause_after_windows) if cfg.pause_after_windows is not None else 0
            if pause_at and processed_windows >= pause_at and processed_windows < int(cfg.max_processed_windows):
                if last_checkpoint_windows != processed_windows:
                    _save_checkpoint(
                        rank, world_size, cfg, model, ema_models, optimizer, scheduler, scaler,
                        processed_windows=processed_windows,
                        optimizer_updates=optimizer_updates,
                        amp_overflow_skips=amp_overflow_skips,
                        epoch=epoch,
                        batches_consumed_in_epoch=batches_consumed,
                        checkpoint_hashes=checkpoint_hashes,
                        weight_initialization=weight_initialization,
                    )
                    last_checkpoint_windows = processed_windows
                paused = True
                break
            if processed_windows >= int(cfg.max_processed_windows):
                break
        if paused or processed_windows >= int(cfg.max_processed_windows):
            break
        epoch += 1
        resume_batches = 0

    final_d2ae_model_contract = None
    if _is_d2ae(cfg):
        required_audit = {
            "initial_zero_gate_alpha_gradient",
            "activated_relation_gradients",
            "schema_version",
            "run_id",
            "seed",
            "optimizer_updates_observed",
            "probe_or_override_used",
        }
        missing_audit = sorted(
            required_audit - set(sparse_relation_gradient_audit)
        )
        if missing_audit:
            raise RuntimeError(
                "D2-AE training ended without the locked gradient audit: "
                + ", ".join(missing_audit)
            )
        final_d2ae_model_contract = _validate_d2ae_model_instance(
            model.module,
            require_zero_alpha=False,
        )

    if not paused and validation_loader is not None:
        if not validation_records or validation_records[-1]["processed_windows"] != processed_windows:
            validation_records.append(_validate(
                rank, world_size, _primary_validation_model(cfg, model, ema_models), diffusion,
                validation_loader, parents,
                minimum, maximum, object_minimum, object_maximum, cfg, processed_windows,
            ))
    if last_checkpoint_windows != processed_windows:
        terminal_checkpoint = _save_checkpoint(
            rank, world_size, cfg, model, ema_models, optimizer, scheduler, scaler,
            processed_windows=processed_windows,
            optimizer_updates=optimizer_updates,
            amp_overflow_skips=amp_overflow_skips,
            epoch=epoch,
            batches_consumed_in_epoch=batches_consumed,
            checkpoint_hashes=checkpoint_hashes,
            weight_initialization=weight_initialization,
        )
    else:
        terminal_checkpoint = Path(str(cfg.checkpoint_dir)).resolve() / (
            f"{cfg.run_id}_windows{processed_windows:09d}.pth"
        )

    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    local = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
            wall_seconds,
            sum(compute_update_seconds),
            len(compute_update_seconds),
            amp_overflow_skips,
            initial_grad_scale,
            float(scaler.get_scale()),
            local_bps_build_seconds,
            local_bps_build_batches,
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local)
    loss_vector = torch.tensor(
        [loss_sums[key] for key in LOSS_KEYS] + [loss_observations],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(loss_vector, op=torch.distributed.ReduceOp.SUM)
    audit_hashes = None
    if training_rng_digest is not None:
        audit_bytes = bytes.fromhex(training_rng_digest.hexdigest())
        audit_tensor = torch.tensor(list(audit_bytes), dtype=torch.uint8, device=device)
        gathered_audits = [torch.zeros_like(audit_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_audits, audit_tensor)
        audit_hashes = [
            bytes(int(value) for value in item.cpu().tolist()).hex()
            for item in gathered_audits
        ]
    if rank == 0:
        values = [item.cpu().tolist() for item in gathered]
        device_total = torch.cuda.get_device_properties(device).total_memory
        status = "paused" if paused else "completed"
        state_record = {
            "schema_version": 1,
            "status": status,
            "run_id": str(cfg.run_id),
            "seed": int(cfg.seed),
            "processed_windows": processed_windows,
            "processed_frames": processed_windows * REPRESENTATION.window_frames,
            "optimizer_updates": optimizer_updates,
            "amp_overflow_skips": amp_overflow_skips,
            "terminal_checkpoint": str(terminal_checkpoint),
            "terminal_checkpoint_sha256": _sha256(terminal_checkpoint),
            "resume_checkpoint": resumed_from,
            "resume_checkpoint_git_commit": state.get("resume_checkpoint_git_commit"),
            "resume_commit_provenance": state.get("resume_commit_provenance"),
        }
        _atomic_json(Path(str(cfg.state_path)).resolve(), state_record, overwrite=True)
        if not paused:
            optimizer_steps = [
                int(value["step"].item() if torch.is_tensor(value["step"]) else value["step"])
                for value in optimizer.state.values()
                if "step" in value
            ]
            averaged_losses = {
                key: float(loss_vector[index].item() / max(loss_vector[-1].item(), 1.0))
                for index, key in enumerate(LOSS_KEYS)
            }
            metrics = {
                **state_record,
                "status": "stable",
                "expert": "hoi",
                "initialization": "random",
                "training_start": weight_initialization["mode"],
                "released_checkpoint_used": False,
                "git_commit": _git_commit(Path(str(cfg.repo_root)).resolve()),
                "world_size": world_size,
                "gpu_name": torch.cuda.get_device_name(device),
                "micro_batch_per_gpu": int(cfg.batch_size),
                "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
                "effective_batch_size": effective,
                "learning_rate": float(cfg.learning_rate),
                "optimization_contract": _optimization_contract(cfg),
                "optimizer_class": optimizer.__class__.__name__,
                "scheduler_class": None if scheduler is None else scheduler.__class__.__name__,
                "gradient_clipping_enabled": bool(cfg.get("gradient_clipping", True)),
                "primary_weight_variant": str(
                    cfg.get(
                        "primary_weight_variant",
                        "online" if _uses_author_update_rule(cfg) else "ema_0.9999",
                    )
                ),
                "warmup_windows": int(cfg.warmup_windows),
                "warmup_updates": warmup_updates,
                "epochs_equivalent": processed_windows / len(train_dataset),
                "parameter_count": sum(parameter.numel() for parameter in model.module.parameters()),
                "key_gradient_present": True,
                "amp_overflow_skips_by_rank": [int(value[5]) for value in values],
                "initial_grad_scale_by_rank": [value[6] for value in values],
                "final_grad_scale_by_rank": [value[7] for value in values],
                "loss_finite": all(math.isfinite(value) for value in averaged_losses.values()),
                "mean_training_losses": averaged_losses,
                "validation": validation_records,
                "peak_memory_allocated_bytes_by_rank": [int(value[0]) for value in values],
                "peak_memory_reserved_bytes_by_rank": [int(value[1]) for value in values],
                "memory_headroom_bytes_by_rank": [device_total - int(value[1]) for value in values],
                "device_total_memory_bytes": device_total,
                "wall_seconds_by_rank": [value[2] for value in values],
                "wall_seconds": max(value[2] for value in values),
                "throughput_windows_per_second": processed_windows / max(value[2] for value in values),
                "throughput_frames_per_second": processed_windows * REPRESENTATION.window_frames / max(value[2] for value in values),
                "local_bps_build_seconds_by_rank": (
                    [value[8] for value in values] if _is_d2ad(cfg) else None
                ),
                "local_bps_build_batches_by_rank": (
                    [int(value[9]) for value in values] if _is_d2ad(cfg) else None
                ),
                "local_bps_condition_windows_per_second_by_rank": (
                    [
                        (
                            int(value[9]) * int(cfg.batch_size) / value[8]
                            if value[8] > 0.0 else None
                        )
                        for value in values
                    ]
                    if _is_d2ad(cfg) else None
                ),
                "mean_profiled_update_seconds_by_rank": [
                    value[3] / value[4] if value[4] else None for value in values
                ],
                "timing_cuda_synchronized": True,
                "checkpoint_hashes": checkpoint_hashes,
                "weight_initialization": weight_initialization,
                "initial_optimizer_state_count": initial_optimizer_state_count,
                "terminal_optimizer_state_count": len(optimizer.state),
                "terminal_optimizer_step_min": min(optimizer_steps) if optimizer_steps else None,
                "terminal_optimizer_step_max": max(optimizer_steps) if optimizer_steps else None,
                "training_rng_audit_schema": (
                    "per-rank SHA256(clean[:,{0,-1},:8], timesteps, q_noise[:,{0,-1},:8])"
                    if audit_hashes is not None else None
                ),
                "training_rng_sha256_by_rank": audit_hashes,
                "d2m_candidate": d2m_candidate,
                "model_config": _model_config(cfg),
                "ema_decays": ema_decays,
                "loss_weights": {
                    "fk": float(cfg.fk_weight),
                    "object_surface": float(cfg.object_surface_weight),
                    "velocity": float(cfg.velocity_weight),
                    "terminal_object_goal": float(cfg.goal_weight),
                },
                "loss_routing": _loss_routing_contract(cfg),
                "interaction_gradient_audit": (
                    interaction_gradient_audit
                    if (_is_d2ac(cfg) or _is_d2ad(cfg)) else None
                ),
                "interaction_adapter": (
                    {
                        "architecture_variant": (
                            HOI_ARCHITECTURE_D2AD
                            if _is_d2ad(cfg) else HOI_ARCHITECTURE_D2AC
                        ),
                        "alpha": float(
                            model.module.network.interaction_adapter.alpha.detach().item()
                        ),
                        "gate": float(torch.tanh(
                            model.module.network.interaction_adapter.alpha.detach()
                        ).item()),
                        "contract": model.module.network.interaction_adapter.contract_metadata(),
                    }
                    if (_is_d2ac(cfg) or _is_d2ad(cfg)) else None
                ),
                "sparse_relation_gradient_audit": (
                    sparse_relation_gradient_audit if _is_d2ae(cfg) else None
                ),
                "d2ae_lifecycle_contract": (
                    d2ae_lifecycle_contract if _is_d2ae(cfg) else None
                ),
                "sparse_relation_field": (
                    {
                        "architecture_variant": HOI_ARCHITECTURE_D2AE,
                        "alpha": float(
                            model.module.network.sparse_relation_field.alpha.detach().item()
                        ),
                        "gate": float(torch.tanh(
                            model.module.network.sparse_relation_field.alpha.detach()
                        ).item()),
                        "contract": (
                            model.module.network.sparse_relation_field.contract_metadata()
                        ),
                        "initial_model_instance_contract": d2ae_model_contract,
                        "final_model_instance_contract": final_d2ae_model_contract,
                        "diagnostic_variant": (
                            model.module.network.sparse_relation_field._diagnostic_variant
                        ),
                        "gate_override": (
                            model.module.network.sparse_relation_field._gate_override
                        ),
                        "capture_enabled": (
                            model.module.network.sparse_relation_field._capture
                        ),
                        "builder": {
                            "backend": "pure_pytorch",
                            "runtime_device": "gpu",
                            "source": "current_diffusion_state_x_t",
                            "cpu_dynamic_geometry": False,
                            "collator_dynamic_geometry": False,
                            "stored_relation": False,
                            "full_mesh_query": False,
                            "point_features_shape_per_batch": [
                                int(cfg.batch_size), 4, 3, 100, 4,
                            ],
                            "encoded_points_shape_per_batch": [
                                int(cfg.batch_size), 4, 3, 100, 128,
                            ],
                            "pooled_blocks_shape_per_batch": [
                                int(cfg.batch_size), 4, 3, 256,
                            ],
                            "relation_vectors_shape_per_batch": [
                                int(cfg.batch_size), 4, 512,
                            ],
                        },
                    }
                    if _is_d2ae(cfg) else None
                ),
                "terminal_model_state_sha256": (
                    _state_dict_sha256(model.module.state_dict())
                    if _is_d2ae(cfg) else None
                ),
                "local_bps_builder": (
                    {
                        "basis_coordinate_system": LOCAL_BASIS_COORDINATE_SYSTEM,
                        "basis_yup_tensor_sha256": BPS_YUP_TENSOR_SHA256,
                        "rest_mesh_manifest_sha256": REST_MESH_MANIFEST_SHA256,
                        "object_mapping_sha256": OBJECT_MAPPING_SHA256,
                        "query_backend": "scipy.spatial.cKDTree.query",
                        "query_parameters": {"k": 1, "eps": 0.0, "p": 2},
                        "query_workers": int(cfg.local_bps_query_workers),
                        "full_rest_mesh": True,
                        "mesh_subsample": False,
                        "stored_per_window_local_bps": False,
                    }
                    if _is_d2ad(cfg) else None
                ),
                "support_metadata": (
                    {
                        "path": str(Path(str(cfg.d2ab_support_metadata_path)).resolve()),
                        "sha256": str(cfg.d2ab_support_metadata_sha256),
                    }
                    if _is_d2ab(cfg) else None
                ),
                "window_state_codec": "state-compositional-v1",
                "bps_sha256": BPS_SHA256,
                "representation": REPRESENTATION.as_dict(),
                "data_contract_sha256": str(cfg.data_contract_sha256),
                "split_manifest": split_manifest,
                "split_sha256": _sha256(Path(split_manifest)),
            }
            _atomic_json(Path(str(cfg.metrics_path)).resolve(), metrics)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@hydra.main(version_base=None, config_path="config", config_name="config_train_hoi_prior")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg), flush=True)
    _validate_fk_foot_temporal_routing_mode(cfg)
    _validate_d2t_contract(cfg, int(cfg.num_gpus))
    _validate_d2u_contract(cfg, int(cfg.num_gpus))
    _validate_d2v_contract(cfg, int(cfg.num_gpus))
    _validate_d2x_contract(cfg, int(cfg.num_gpus))
    _validate_d2y_contract(cfg, int(cfg.num_gpus))
    _validate_d2z_contract(cfg, int(cfg.num_gpus))
    _validate_d2ab_contract(cfg, int(cfg.num_gpus))
    _validate_d2ac_contract(cfg, int(cfg.num_gpus))
    _validate_d2ad_contract(cfg, int(cfg.num_gpus))
    _validate_d2ae_contract(cfg, int(cfg.num_gpus))
    _validate_author_update_execution_host(cfg)
    if not torch.cuda.is_available() or torch.cuda.device_count() < int(cfg.num_gpus):
        raise RuntimeError(f"requires {cfg.num_gpus} visible CUDA devices")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    torch.multiprocessing.spawn(_worker, args=(cfg,), nprocs=int(cfg.num_gpus), join=True)


if __name__ == "__main__":
    main()
