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
    HOI_ARCHITECTURE_D2AF,
    HOI_ARCHITECTURE_D2AG,
    build_expert,
)
from priors.representation import REPRESENTATION
from priors.sparse_relation import (
    BASE_PARAMETER_COUNT as D2AE_BASE_PARAMETER_COUNT,
    D2AG_SELF_CONDITION_PROBABILITY,
    D2AG_VARIABLE_ANCHORS,
    PARAMETER_INCREASE_FRACTION as D2AE_PARAMETER_INCREASE_FRACTION,
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
    TOTAL_PARAMETER_COUNT as D2AE_TOTAL_PARAMETER_COUNT,
    build_d2ag_relation_source,
    diffusion_reliability_contract_metadata,
    selfcond_relation_source_contract_metadata,
    validate_diffusion_reliability_contract,
    validate_selfcond_relation_source_contract,
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
D2AF_FORMAL_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-sqrt-alpha-bar-reliability"
    r"(?P<retry>-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
D2AF_PERFORMANCE_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-performance-benchmark"
    r"(?P<retry>-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
D2AF_PERFORMANCE_WAIVER_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-performance-waiver-s42-(?P<date>[0-9]{8})$"
)
D2AF_MINIMUM_THROUGHPUT = 3179.689863044761
D2AF_MAXIMUM_ETA_HOURS = 5.367399778519349
D2AF_WAIVED_BENCHMARK_RUN_ID = (
    "p1-hoi-d2af-performance-benchmark-s42-20260729"
)
D2AF_WAIVED_FORMAL_RUN_ID = (
    "p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729"
)
D2AF_WAIVED_BENCHMARK_SHA256 = (
    "53e9842d0522cf456a86eedc25d2a972cd00db3fb067113ff25f31f6117e1f33"
)
D2AF_WAIVED_ELIGIBILITY_SHA256 = (
    "c52c0536423d7a17101829cb2b020316b9c6e0f7aa2cf39f33b984ffb39896b4"
)
D2AF_WAIVED_SOURCE_COMMIT = "1c6c3058478411361bf3e73830f900f660ae516b"
D2AF_WAIVED_SOURCE_CONTRACT_SHA256 = (
    "68269a2cac8eaf6fd2b55b139bb2be5b5dbafde6e7f22496f5a894f18b843145"
)
D2AF_WAIVED_THROUGHPUT = 2089.8443630127094
D2AF_WAIVED_ETA_HOURS = 8.166477355310539
D2AF_PERFORMANCE_FAILURE_CLASSIFICATION = (
    "diffusion-reliability-performance-negative-stop"
)
D2AF_PERFORMANCE_WAIVER_CLASSIFICATION = (
    "user-authorized-performance-waiver"
)
D2AF_CHECKPOINT_RACE_CONTINUATION_RUN_ID = (
    "p1-hoi-d2af-checkpoint-race-continuation-s42-20260729"
)
D2AF_CHECKPOINT_RACE_CONTINUATION_CLASSIFICATION = (
    "ddp-checkpoint-sidecar-existence-race-operational-continuation"
)
D2AF_CHECKPOINT_RACE_CONTINUATION_RELATIVE_PATH = (
    "experiments/contracts/"
    "p1_hoi_d2af_checkpoint_race_continuation_s42_20260729.json"
)
D2AF_CHECKPOINT_RACE_SOURCE_COMMIT = (
    "7202d32a7375e7197886c4f873688fd472e2c803"
)
D2AF_CHECKPOINT_RACE_SOURCE_FORMAL_CONTRACT_SHA256 = (
    "299d7a900c6a96264dd698c50ef476ea78d2b2efdfbb3b0e375d27d99101cc3e"
)
D2AF_CHECKPOINT_RACE_CHECKPOINT_BASENAME = (
    "p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729"
    "_windows006144000.pth"
)
D2AF_CHECKPOINT_RACE_CHECKPOINT_SHA256 = (
    "3c94f7344991cb38aab37fd8356cabe83a84b449d10505e0e46341490605287e"
)
D2AF_CHECKPOINT_RACE_CHECKPOINT_BYTES = 361283695
D2AF_CHECKPOINT_RACE_FAILURE_SHA256 = (
    "a66fec685afb5cbb4079619de9417b7171af7e29244723f1deac9d4ba306d1b1"
)
D2AF_CHECKPOINT_RACE_PARTIAL_ARCHIVE_SHA256 = (
    "b5573764eceb388f6a28f10b4ed89b44bbbcdd430213dad490f6c8b5caa7f9dd"
)
D2AF_CHECKPOINT_RACE_PARTIAL_ARCHIVE_FILES = 3
D2AF_CHECKPOINT_RACE_PARTIAL_ARCHIVE_BYTES = 45977
D2AF_CHECKPOINT_RACE_MANIFEST_SHA256 = (
    "985192f686de2d4330cb82c826b648a08d12b7ed55c0bd4c8d196951d05b589b"
)
D2AF_CHECKPOINT_RACE_RNG_SHA256 = {
    0: "ebc379497baa4da38c71b5d100ccb179afd6cbf7f629f6d9ba4cd0bf3abfaaae",
    1: "ac0184e9746b55fc1e6bde4bfba6f6038951587d782e520fd2884e89036b2ecc",
    2: "91ddfbd38b8781dd82f112180ade468372e164075b679b65c8368e102ae24229",
    3: "f5063b69ff77d837a9f223744b826e0f93176004b955037b5538b502a32d353d",
}
D2AF_WAIVER_ALLOWED_CHANGED_PATHS = frozenset(
    {
        "code/config/config_train_hoi_prior.yaml",
        "code/config/config_train_hoi_prior_d2af.yaml",
        "code/train_hoi_prior.py",
        "docs/EXPERIMENT_PLAN.md",
        "experiments/registry.jsonl",
        "tests/test_hoi_d2af.py",
    }
)
D2AF_CHECKPOINT_RACE_IMPLEMENTATION_ALLOWED_PATHS = frozenset(
    {
        "code/config/config_train_hoi_prior.yaml",
        "code/config/config_train_hoi_prior_d2af.yaml",
        "code/train_hoi_prior.py",
        "docs/EXPERIMENT_PLAN.md",
        "experiments/registry.jsonl",
        "tests/test_hoi_d2af.py",
        "tests/test_hoi_d2af_lifecycle_cpu.py",
    }
)
D2AF_CHECKPOINT_RACE_EXECUTION_ALLOWED_PATHS = frozenset(
    {
        *D2AF_CHECKPOINT_RACE_IMPLEMENTATION_ALLOWED_PATHS,
        D2AF_CHECKPOINT_RACE_CONTINUATION_RELATIVE_PATH,
    }
)
D2AF_FORMAL_SOURCE_SCOPES = (
    "code",
    "tools/benchmark_hoi_d2af.py",
    "tools/benchmark_hoi_d2ae.py",
    "tools/smoke_hoi_d2af.py",
    "tools/smoke_hoi_d2ae.py",
    "tools/diagnose_hoi_d2af.py",
    "tools/diagnose_hoi_d2ae.py",
    "tools/run_hoi_d2af_eligibility.py",
    "tools/smoke_hoi_d2ac.py",
    "tools/capture_hoi_worker_preflight.py",
    "tools/experiment.py",
)
D2AG_FORMAL_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ag-selfcond-relation-source"
    r"(?P<retry>-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
D2AG_PERFORMANCE_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ag-performance-benchmark"
    r"(?P<retry>-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
# Registered as D2-AG's own gate: 85% of the sealed D2-X formal throughput
# 3243.0357134915853 windows/s (EP:7090-7098).  Numerically equal to the D2-AE
# pair by coincidence of form; the D2-AF 95%-of-predecessor values
# (3179.689863044761 / 5.367399778519349) are explicitly inapplicable here.
D2AG_MINIMUM_THROUGHPUT = 2756.580356467847
D2AG_MAXIMUM_ETA_HOURS = 6.20
D2AG_PERFORMANCE_FAILURE_CLASSIFICATION = (
    "selfcond-relation-source-performance-negative-stop"
)
D2AG_CONTRACT_FAILURE_CLASSIFICATION = (
    "selfcond-relation-source-contract-failure-stop"
)
D2AG_PERFORMANCE_WAIVER_CLASSIFICATION = (
    "user-authorized-performance-waiver"
)
D2AG_PERFORMANCE_WAIVER_STATUS = "failed-waived"
D2AG_FORBIDDEN_WAIVED_STATUS = "performance-gate-passed"
D2AG_PERFORMANCE_WAIVER_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ag-performance-waiver-s42-(?P<date>[0-9]{8})$"
)
D2AG_PERFORMANCE_WAIVER_RELATIVE_PATH = (
    "experiments/contracts/"
    "p1_hoi_d2ag_performance_waiver_s42_20260731.json"
)
# The one user-authorized D2-AG failure.  Every value is the measured
# benchmark record; none of them is a target the trainer may relax.
D2AG_WAIVED_BENCHMARK_RUN_ID = (
    "p1-hoi-d2ag-performance-benchmark-r2-s42-20260731"
)
D2AG_WAIVED_FORMAL_RUN_ID = (
    "p1-hoi-d2ag-selfcond-relation-source-s42-20260731"
)
D2AG_WAIVED_BENCHMARK_SHA256 = (
    "ad5b052d850a6978d2ce64022ecc0328ab73f32ffa7a72446a4a16bdf2c19cae"
)
D2AG_WAIVED_THROUGHPUT = 2172.2037135137825
D2AG_WAIVED_ETA_HOURS = 7.8568444388944645
D2AG_WAIVED_SOURCE_COMMIT = "ada2d84223ecbf76f5ed9bbd313f5ac6dfce2cbb"
D2AG_WAIVED_SOURCE_CONTRACT_SHA256 = (
    "55ff307986e1c1e0ff94286b1fadec681b7cb3fe478da7b8ce5da670de84ee88"
)
D2AG_WAIVER_ALLOWED_CHANGED_PATHS = frozenset(
    {
        "code/config/config_train_hoi_prior.yaml",
        "code/config/config_train_hoi_prior_d2ag.yaml",
        "code/train_hoi_prior.py",
        "tools/run_hoi_d2ag_native_evaluation.py",
        "tests/test_hoi_d2ag.py",
        "tests/test_hoi_d2ag_lifecycle_cpu.py",
        "tests/test_hoi_d2ag_eval.py",
        "docs/EXPERIMENT_PLAN.md",
        "experiments/registry.jsonl",
        D2AG_PERFORMANCE_WAIVER_RELATIVE_PATH,
    }
)
# The exact benchmark checks the waiver may excuse.  Anything else failing
# invalidates the waiver, so a genuine scientific failure cannot pass.
D2AG_WAIVED_BENCHMARK_FAILED_CHECKS = (
    "classification", "eta", "formal_authorized", "status", "throughput",
)
D2AG_FORMAL_SOURCE_SCOPES = (
    "code",
    "tools/benchmark_hoi_d2ag.py",
    "tools/benchmark_hoi_d2ae.py",
    "tools/smoke_hoi_d2ag.py",
    "tools/smoke_hoi_d2ae.py",
    "tools/diagnose_hoi_d2ag.py",
    "tools/diagnose_hoi_d2ae.py",
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


def _validate_d2af_formal_run_id(
    run_id: str,
    *,
    require_actual_date: bool = True,
) -> Dict[str, object]:
    match = D2AF_FORMAL_RUN_ID_RE.fullmatch(str(run_id))
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if match is None or (
        require_actual_date and match.group("date") != actual_date
    ):
        raise ValueError("D2-AF formal run id must use the locked stem and actual date")
    return {
        "run_id": str(run_id),
        "date": match.group("date"),
        "date_is_actual": match.group("date") == actual_date,
        "retry": match.group("retry") is not None,
    }


def _validate_d2ag_formal_run_id(
    run_id: str,
    *,
    require_actual_date: bool = True,
) -> Dict[str, object]:
    match = D2AG_FORMAL_RUN_ID_RE.fullmatch(str(run_id))
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if match is None or (
        require_actual_date and match.group("date") != actual_date
    ):
        raise ValueError("D2-AG formal run id must use the locked stem and actual date")
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


def _git_commit_is_ancestor(repo: Path, value: object) -> bool:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        return False
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", value, _git_commit(repo)],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


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


def _d2ag_formal_source_contract(repo: Path) -> Dict[str, object]:
    """Hash the exact tracked D2-AG runtime tree used by benchmark/formal."""
    repo = repo.resolve()
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *D2AG_FORMAL_SOURCE_SCOPES],
        cwd=repo,
    )
    relative_paths = sorted(
        item.decode("utf-8") for item in output.split(b"\0") if item
    )
    if not relative_paths:
        raise ValueError("D2-AG formal source contract resolved no tracked files")
    records = [
        {"path": relative, "sha256": _sha256(repo / relative)}
        for relative in relative_paths
    ]
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": "git-ls-files-path-content-sha256-v1",
        "scopes": list(D2AG_FORMAL_SOURCE_SCOPES),
        "tracked_file_count": len(records),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _d2ag_formal_source_contract_at_commit(
    repo: Path,
    commit: str,
) -> Dict[str, object]:
    """Hash the registered D2-AG runtime tree from an immutable Git object."""
    repo = repo.resolve()
    if re.fullmatch(r"[0-9a-f]{40}", str(commit)) is None:
        raise ValueError("D2-AG source-contract commit must be a full Git object id")
    try:
        output = subprocess.check_output(
            [
                "git", "ls-tree", "-r", "-z", "--name-only", str(commit),
                "--", *D2AG_FORMAL_SOURCE_SCOPES,
            ],
            cwd=repo,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("D2-AG source-contract commit is not readable") from error
    relative_paths = sorted(
        item.decode("utf-8") for item in output.split(b"\0") if item
    )
    if not relative_paths:
        raise ValueError("D2-AG committed source contract resolved no tracked files")
    records = []
    for relative in relative_paths:
        try:
            content = subprocess.check_output(
                ["git", "show", f"{commit}:{relative}"],
                cwd=repo,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"D2-AG committed source path is not readable: {relative}"
            ) from error
        records.append({
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": "git-ls-files-path-content-sha256-v1",
        "scopes": list(D2AG_FORMAL_SOURCE_SCOPES),
        "tracked_file_count": len(records),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _d2ag_source_transition(
    repo: Path,
    source_commit: str,
    target_commit: str,
) -> Dict[str, object]:
    """Resolve the exact validator-only Git transition the waiver authorizes."""
    commit_pattern = re.compile(r"^[0-9a-f]{40}$")
    if (
        commit_pattern.fullmatch(str(source_commit)) is None
        or commit_pattern.fullmatch(str(target_commit)) is None
    ):
        raise ValueError("D2-AG waiver transition requires full Git object ids")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, target_commit],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("D2-AG waiver target is not a descendant of benchmark source")
    if not _git_commit_is_ancestor(repo, target_commit):
        raise ValueError("D2-AG waiver target is not an ancestor of current HEAD")
    changed_paths = tuple(
        line
        for line in subprocess.check_output(
            ["git", "diff", "--name-only", source_commit, target_commit],
            cwd=str(repo),
            text=True,
        ).splitlines()
        if line
    )
    unexpected = sorted(set(changed_paths) - D2AG_WAIVER_ALLOWED_CHANGED_PATHS)
    if unexpected:
        raise ValueError(
            "D2-AG waiver transition changes non-authorized paths: "
            + ", ".join(unexpected)
        )
    diff_bytes = subprocess.check_output(
        ["git", "diff", "--binary", source_commit, target_commit],
        cwd=str(repo),
    )
    return {
        "source_commit": source_commit,
        "target_commit": target_commit,
        "changed_paths": list(changed_paths),
        "diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
    }


def _d2af_formal_source_contract(repo: Path) -> Dict[str, object]:
    """Hash the exact tracked D2-AF runtime tree used by benchmark/formal."""
    repo = repo.resolve()
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *D2AF_FORMAL_SOURCE_SCOPES],
        cwd=repo,
    )
    relative_paths = sorted(
        item.decode("utf-8") for item in output.split(b"\0") if item
    )
    if not relative_paths:
        raise ValueError("D2-AF formal source contract resolved no tracked files")
    records = [
        {"path": relative, "sha256": _sha256(repo / relative)}
        for relative in relative_paths
    ]
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": "git-ls-files-path-content-sha256-v1",
        "scopes": list(D2AF_FORMAL_SOURCE_SCOPES),
        "tracked_file_count": len(records),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _d2af_formal_source_contract_at_commit(
    repo: Path,
    commit: str,
) -> Dict[str, object]:
    """Hash the registered D2-AF runtime tree from an immutable Git object."""
    repo = repo.resolve()
    if re.fullmatch(r"[0-9a-f]{40}", str(commit)) is None:
        raise ValueError("D2-AF source-contract commit must be a full Git object id")
    try:
        output = subprocess.check_output(
            [
                "git", "ls-tree", "-r", "-z", "--name-only", str(commit),
                "--", *D2AF_FORMAL_SOURCE_SCOPES,
            ],
            cwd=repo,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("D2-AF source-contract commit is not readable") from error
    relative_paths = sorted(
        item.decode("utf-8") for item in output.split(b"\0") if item
    )
    if not relative_paths:
        raise ValueError("D2-AF committed source contract resolved no tracked files")
    records = []
    for relative in relative_paths:
        try:
            content = subprocess.check_output(
                ["git", "show", f"{commit}:{relative}"],
                cwd=repo,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"D2-AF committed source path is not readable: {relative}"
            ) from error
        records.append({
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": "git-ls-files-path-content-sha256-v1",
        "scopes": list(D2AF_FORMAL_SOURCE_SCOPES),
        "tracked_file_count": len(records),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _d2af_source_transition(
    repo: Path,
    source_commit: str,
    target_commit: str,
) -> Dict[str, object]:
    """Resolve the exact governance/validator-only Git transition."""
    commit_pattern = re.compile(r"^[0-9a-f]{40}$")
    if (
        commit_pattern.fullmatch(str(source_commit)) is None
        or commit_pattern.fullmatch(str(target_commit)) is None
    ):
        raise ValueError("D2-AF waiver transition requires full Git object ids")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, target_commit],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("D2-AF waiver target is not a descendant of benchmark source")
    if not _git_commit_is_ancestor(repo, target_commit):
        raise ValueError("D2-AF waiver target is not an ancestor of current HEAD")
    changed_paths = tuple(
        line
        for line in subprocess.check_output(
            ["git", "diff", "--name-only", source_commit, target_commit],
            cwd=str(repo),
            text=True,
        ).splitlines()
        if line
    )
    unexpected = sorted(set(changed_paths) - D2AF_WAIVER_ALLOWED_CHANGED_PATHS)
    if unexpected:
        raise ValueError(
            "D2-AF waiver transition changes non-authorized paths: "
            + ", ".join(unexpected)
        )
    diff_bytes = subprocess.check_output(
        ["git", "diff", "--binary", source_commit, target_commit],
        cwd=str(repo),
    )
    return {
        "source_commit": source_commit,
        "target_commit": target_commit,
        "changed_paths": list(changed_paths),
        "diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
    }


def _d2af_checkpoint_race_source_transition(
    repo: Path,
    source_commit: str,
    target_commit: str,
) -> Dict[str, object]:
    """Resolve the one operational D2-AF checkpoint-race implementation diff."""
    commit_pattern = re.compile(r"^[0-9a-f]{40}$")
    if (
        commit_pattern.fullmatch(str(source_commit)) is None
        or commit_pattern.fullmatch(str(target_commit)) is None
    ):
        raise ValueError(
            "D2-AF checkpoint-race transition requires full Git object ids"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, target_commit],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError(
            "D2-AF checkpoint-race target is not a descendant of its source"
        )
    if not _git_commit_is_ancestor(repo, target_commit):
        raise ValueError(
            "D2-AF checkpoint-race target is not an ancestor of current HEAD"
        )
    changed_paths = tuple(
        line
        for line in subprocess.check_output(
            ["git", "diff", "--name-only", source_commit, target_commit],
            cwd=str(repo),
            text=True,
        ).splitlines()
        if line
    )
    unexpected = sorted(
        set(changed_paths) - D2AF_CHECKPOINT_RACE_IMPLEMENTATION_ALLOWED_PATHS
    )
    if unexpected:
        raise ValueError(
            "D2-AF checkpoint-race transition changes non-authorized paths: "
            + ", ".join(unexpected)
        )
    diff_bytes = subprocess.check_output(
        ["git", "diff", "--binary", source_commit, target_commit],
        cwd=str(repo),
    )
    return {
        "source_commit": source_commit,
        "target_commit": target_commit,
        "changed_paths": list(changed_paths),
        "diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
    }


def _git_transition(
    repo: Path,
    source_commit: str,
    target_commit: str,
) -> Dict[str, object]:
    """Return a byte-exact Git transition without applying a path policy."""
    changed_paths = [
        line
        for line in subprocess.check_output(
            ["git", "diff", "--name-only", source_commit, target_commit],
            cwd=str(repo),
            text=True,
        ).splitlines()
        if line
    ]
    diff_bytes = subprocess.check_output(
        ["git", "diff", "--binary", source_commit, target_commit],
        cwd=str(repo),
    )
    return {
        "source_commit": source_commit,
        "target_commit": target_commit,
        "changed_paths": changed_paths,
        "diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
    }


def _sha256_path_record(path: Path) -> Dict[str, object]:
    """Hash a file/directory using the reportable-manifest tree algorithm."""
    path = path.resolve()
    if not path.exists():
        raise ValueError(f"artifact path does not exist: {path}")
    if path.is_symlink():
        raise ValueError(f"top-level artifact must not be a symlink: {path}")
    if path.is_file():
        return {
            "kind": "file",
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        if child.is_symlink():
            raise ValueError(f"artifact tree contains a symlink: {child}")
        relative = child.relative_to(path).as_posix().encode("utf-8")
        file_hash = _sha256(child).encode("ascii")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_hash)
        files += 1
        total_bytes += child.stat().st_size
    return {
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "files": files,
        "bytes": total_bytes,
    }


def _validate_tracked_d2ag_waiver_path(repo: Path, path: Path) -> str:
    """Require the immutable waiver to be a tracked experiments/contracts file."""
    try:
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "D2-AG performance waiver must be inside the repository"
        ) from error
    if not relative.startswith("experiments/contracts/"):
        raise ValueError("D2-AG performance waiver must be under experiments/contracts")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError("D2-AG performance waiver must be tracked by Git")
    return relative


def _validate_tracked_d2af_waiver_path(repo: Path, path: Path) -> str:
    """Require the immutable waiver to be a tracked experiments/contracts file."""
    try:
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("D2-AF performance waiver must be inside the repository") from error
    if not relative.startswith("experiments/contracts/"):
        raise ValueError("D2-AF performance waiver must be under experiments/contracts")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError("D2-AF performance waiver must be tracked by Git")
    return relative


def _validate_tracked_d2af_checkpoint_race_path(repo: Path, path: Path) -> str:
    """Require the operational continuation to be the one tracked contract."""
    try:
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "D2-AF checkpoint-race continuation must be inside the repository"
        ) from error
    if relative != D2AF_CHECKPOINT_RACE_CONTINUATION_RELATIVE_PATH:
        raise ValueError(
            "D2-AF checkpoint-race continuation path is not the registered contract"
        )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError(
            "D2-AF checkpoint-race continuation must be tracked by Git"
        )
    return relative


def _validate_d2af_checkpoint_race_continuation(
    cfg: DictConfig,
    *,
    repo: Path,
    waiver_target_contract: Mapping[str, object],
) -> Dict[str, object]:
    """Validate the one exact same-run operational continuation."""
    if cfg.resume_checkpoint in (None, "", False):
        raise ValueError(
            "D2-AF checkpoint-race continuation requires its exact resume checkpoint"
        )
    path_value = cfg.get("d2af_checkpoint_race_continuation_path")
    configured_sha256 = cfg.get("d2af_checkpoint_race_continuation_sha256")
    if path_value in (None, "", False) or configured_sha256 in (None, "", False):
        raise ValueError(
            "D2-AF changed-source resume requires a sealed checkpoint-race contract"
        )
    path = Path(str(path_value))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(
            "D2-AF checkpoint-race continuation path must be an existing absolute file"
        )
    actual_sha256 = _sha256(path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(configured_sha256)) is None
        or actual_sha256 != str(configured_sha256)
    ):
        raise ValueError("D2-AF checkpoint-race continuation SHA-256 mismatch")
    relative_path = _validate_tracked_d2af_checkpoint_race_path(repo, path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "D2-AF checkpoint-race continuation is not valid JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise ValueError(
            "D2-AF checkpoint-race continuation must be a JSON object"
        )

    resume_path = Path(str(cfg.resume_checkpoint)).resolve()
    run_dir = resume_path.parent.parent
    checkpoint_dir = run_dir / "checkpoints"
    expected_checkpoint = checkpoint_dir / D2AF_CHECKPOINT_RACE_CHECKPOINT_BASENAME
    failure_path = run_dir / "operational_checkpoint_race_failure.json"
    manifest_path = run_dir / "manifest.json"
    partial_archive_path = (
        run_dir / "operational_failures/checkpoint_race_windows009216000"
    )
    if resume_path != expected_checkpoint.resolve() or not resume_path.is_file():
        raise ValueError(
            "D2-AF checkpoint-race continuation resume checkpoint is not exact"
        )
    try:
        failure = (
            json.loads(failure_path.read_text(encoding="utf-8"))
            if failure_path.is_file() else None
        )
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else None
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "D2-AF checkpoint-race run artifacts are not valid JSON"
        ) from error
    partial_archive = (
        _sha256_path_record(partial_archive_path)
        if partial_archive_path.is_dir() else None
    )
    checkpoint_binding = value.get("checkpoint")
    failure_binding = value.get("failure")
    partial_binding = value.get("partial_archive")
    source_transition_binding = value.get("source_transition")
    execution_binding = value.get("execution_transition")
    continuation = value.get("continuation")
    scientific = value.get("scientific_conditions")
    if not isinstance(source_transition_binding, Mapping):
        raise ValueError(
            "D2-AF checkpoint-race continuation source transition is missing"
        )
    implementation_target = str(
        source_transition_binding.get("target_implementation_commit", "")
    )
    implementation_transition = _d2af_checkpoint_race_source_transition(
        repo,
        D2AF_CHECKPOINT_RACE_SOURCE_COMMIT,
        implementation_target,
    )
    source_contract = _d2af_formal_source_contract_at_commit(
        repo, D2AF_CHECKPOINT_RACE_SOURCE_COMMIT,
    )
    target_contract = _d2af_formal_source_contract_at_commit(
        repo, implementation_target,
    )
    current_commit = _git_commit(repo)
    current_contract = _d2af_formal_source_contract(repo)
    execution_transition = _git_transition(
        repo, D2AF_CHECKPOINT_RACE_SOURCE_COMMIT, current_commit,
    )
    post_implementation_transition = _git_transition(
        repo, implementation_target, current_commit,
    )
    unexpected_execution_paths = sorted(
        set(execution_transition["changed_paths"])
        - D2AF_CHECKPOINT_RACE_EXECUTION_ALLOWED_PATHS
    )
    unexpected_post_paths = sorted(
        set(post_implementation_transition["changed_paths"])
        - {
            D2AF_CHECKPOINT_RACE_CONTINUATION_RELATIVE_PATH,
            "docs/EXPERIMENT_PLAN.md",
            "experiments/registry.jsonl",
        }
    )
    rng_bindings = (
        checkpoint_binding.get("rng_sidecars")
        if isinstance(checkpoint_binding, Mapping) else None
    )
    rng_records = []
    for rank in range(4):
        rng_path = checkpoint_dir / (
            f"{expected_checkpoint.stem}.rank{rank}.rng.pth"
        )
        rng_records.append({
            "rank": rank,
            "basename": rng_path.name,
            "sha256": _sha256(rng_path) if rng_path.is_file() else None,
            "bytes": rng_path.stat().st_size if rng_path.is_file() else None,
        })
    partial_checkpoint_files = sorted(
        item.name
        for item in checkpoint_dir.glob(
            f"{cfg.run_id}_windows009216000*"
        )
    )

    checks = {
        "schema_version": value.get("schema_version") == 1,
        "run_id": value.get("run_id")
        == D2AF_CHECKPOINT_RACE_CONTINUATION_RUN_ID,
        "status": value.get("status") == "authorized",
        "classification": value.get("classification")
        == D2AF_CHECKPOINT_RACE_CONTINUATION_CLASSIFICATION,
        "formal_run_id": value.get("formal_run_id") == str(cfg.run_id)
        == D2AF_WAIVED_FORMAL_RUN_ID,
        "seed": value.get("seed") == int(cfg.seed) == 42,
        "tracked_contract": relative_path
        == D2AF_CHECKPOINT_RACE_CONTINUATION_RELATIVE_PATH,
        "manifest": isinstance(manifest, Mapping)
        and _sha256(manifest_path) == D2AF_CHECKPOINT_RACE_MANIFEST_SHA256
        and manifest.get("experiment_id") == str(cfg.run_id)
        and manifest.get("status") == "running"
        and manifest.get("ended_at") is None
        and manifest.get("metrics") is None
        and manifest.get("git", {}).get("commit")
        == D2AF_CHECKPOINT_RACE_SOURCE_COMMIT,
        "checkpoint_binding": isinstance(checkpoint_binding, Mapping)
        and resume_path == expected_checkpoint.resolve()
        and checkpoint_binding.get("basename")
        == D2AF_CHECKPOINT_RACE_CHECKPOINT_BASENAME
        and checkpoint_binding.get("sha256")
        == _sha256(resume_path)
        == D2AF_CHECKPOINT_RACE_CHECKPOINT_SHA256
        and checkpoint_binding.get("bytes") == resume_path.stat().st_size
        == D2AF_CHECKPOINT_RACE_CHECKPOINT_BYTES
        and checkpoint_binding.get("processed_windows") == 6144000
        and checkpoint_binding.get("optimizer_updates") == 3000
        and checkpoint_binding.get("source_commit")
        == D2AF_CHECKPOINT_RACE_SOURCE_COMMIT,
        "rng_sidecars": isinstance(rng_bindings, list)
        and rng_bindings == rng_records
        and all(
            record["sha256"] == D2AF_CHECKPOINT_RACE_RNG_SHA256[record["rank"]]
            for record in rng_records
        ),
        "failure_binding": isinstance(failure_binding, Mapping)
        and isinstance(failure, Mapping)
        and failure_binding.get("relative_path")
        == "operational_checkpoint_race_failure.json"
        and failure_binding.get("sha256")
        == _sha256(failure_path)
        == D2AF_CHECKPOINT_RACE_FAILURE_SHA256
        and failure.get("run_id") == str(cfg.run_id)
        and failure.get("status") == "operational-failure-preserved"
        and failure.get("classification")
        == "ddp-checkpoint-sidecar-existence-race-operational-failure"
        and failure.get("return_code") == 1
        and failure.get("failure_progress", {}).get(
            "processed_windows_attempted"
        ) == 9216000
        and failure.get("failure_progress", {}).get(
            "optimizer_updates_attempted"
        ) == 4500
        and failure.get("last_complete_resume_checkpoint", {}).get("sha256")
        == D2AF_CHECKPOINT_RACE_CHECKPOINT_SHA256,
        "partial_archive": isinstance(partial_binding, Mapping)
        and partial_archive == {
            "kind": "directory",
            "sha256": D2AF_CHECKPOINT_RACE_PARTIAL_ARCHIVE_SHA256,
            "files": D2AF_CHECKPOINT_RACE_PARTIAL_ARCHIVE_FILES,
            "bytes": D2AF_CHECKPOINT_RACE_PARTIAL_ARCHIVE_BYTES,
        }
        and partial_binding.get("relative_path")
        == "operational_failures/checkpoint_race_windows009216000"
        and partial_binding.get("sha256")
        == D2AF_CHECKPOINT_RACE_PARTIAL_ARCHIVE_SHA256
        and partial_binding.get("files")
        == D2AF_CHECKPOINT_RACE_PARTIAL_ARCHIVE_FILES
        and partial_binding.get("bytes")
        == D2AF_CHECKPOINT_RACE_PARTIAL_ARCHIVE_BYTES
        and partial_checkpoint_files == [],
        "source_identity": source_transition_binding.get("source_commit")
        == D2AF_CHECKPOINT_RACE_SOURCE_COMMIT,
        "implementation_transition": (
            source_transition_binding.get("changed_paths")
            == implementation_transition["changed_paths"]
            and source_transition_binding.get("diff_sha256")
            == implementation_transition["diff_sha256"]
        ),
        "source_formal_contract": source_contract
        == source_transition_binding.get("source_formal_contract")
        == waiver_target_contract
        and source_contract.get("sha256")
        == D2AF_CHECKPOINT_RACE_SOURCE_FORMAL_CONTRACT_SHA256,
        "target_formal_contract": target_contract
        == source_transition_binding.get("target_formal_contract")
        == current_contract,
        "execution_transition": isinstance(execution_binding, Mapping)
        and execution_binding.get("allowed_changed_paths")
        == sorted(D2AF_CHECKPOINT_RACE_EXECUTION_ALLOWED_PATHS)
        and execution_binding.get("target_must_equal_current_head") is True
        and execution_binding.get(
            "diff_sha256_bound_in_resolved_config"
        ) is True
        and not unexpected_execution_paths
        and not unexpected_post_paths
        and bool(cfg.get("resume_commit_transition_authorized", False))
        and str(cfg.get("resume_source_commit"))
        == D2AF_CHECKPOINT_RACE_SOURCE_COMMIT
        and str(cfg.get("resume_target_commit")) == current_commit
        and str(cfg.get("resume_transition_diff_sha256"))
        == execution_transition["diff_sha256"],
        "continuation": isinstance(continuation, Mapping)
        and continuation.get("same_run_only") is True
        and continuation.get("new_formal_run") is False
        and continuation.get("from_random_restart") is False
        and continuation.get("resume_processed_windows") == 6144000
        and continuation.get("resume_optimizer_updates") == 3000
        and continuation.get("target_processed_windows") == 61440000
        and continuation.get("target_optimizer_updates") == 30000
        and continuation.get("accepted_lineage_optimizer_updates") == 30000
        and continuation.get("actual_total_gpu_optimizer_updates") == 31500
        and continuation.get("checkpoint_selection") is False
        and continuation.get("budget_extension") is False,
        "scientific_conditions": isinstance(scientific, Mapping)
        and all(
            scientific.get(name) is False
            for name in (
                "model_math_changed",
                "relation_builder_or_routing_changed",
                "loss_or_optimizer_changed",
                "batch_or_budget_changed",
                "data_loader_or_worker_configuration_changed",
                "evaluation_protocol_changed",
            )
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "D2-AF checkpoint-race continuation contract mismatch: "
            + ", ".join(failed)
        )
    return {
        "path": str(path.resolve()),
        "relative_path": relative_path,
        "sha256": actual_sha256,
        "run_id": str(value["run_id"]),
        "classification": D2AF_CHECKPOINT_RACE_CONTINUATION_CLASSIFICATION,
        "resume_checkpoint": str(resume_path),
        "resume_checkpoint_sha256": D2AF_CHECKPOINT_RACE_CHECKPOINT_SHA256,
        "failure_path": str(failure_path),
        "failure_sha256": D2AF_CHECKPOINT_RACE_FAILURE_SHA256,
        "partial_archive_path": str(partial_archive_path),
        "partial_archive_sha256": D2AF_CHECKPOINT_RACE_PARTIAL_ARCHIVE_SHA256,
        "source_transition": implementation_transition,
        "execution_transition": execution_transition,
        "source_formal_contract": source_contract,
        "target_formal_contract": target_contract,
        "checks": checks,
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


def _validate_d2ag_performance_waiver(
    cfg: DictConfig,
    *,
    benchmark: Mapping[str, object],
    benchmark_sha256: str,
    repo: Path,
) -> Dict[str, object]:
    """Validate the one exact post-failure, user-authorized D2-AG formal run.

    This mirrors the D2-AF waiver without a clean-signal eligibility premise,
    which D2-AG never registered.  The waiver may only excuse the measured
    throughput/ETA shortfall of one exact benchmark: it binds that benchmark's
    run id and SHA-256, keeps the failed status/classification, and must never
    assert ``performance-gate-passed``.
    """
    path_value = cfg.get("d2ag_performance_waiver_path")
    configured_sha256 = cfg.get("d2ag_performance_waiver_sha256")
    if path_value in (None, "", False) or configured_sha256 in (None, "", False):
        raise ValueError(
            "D2-AG failed performance benchmark requires an explicit sealed waiver"
        )
    path = Path(str(path_value))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(
            "D2-AG performance waiver path must be an existing absolute file"
        )
    actual_sha256 = _sha256(path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(configured_sha256)) is None
        or actual_sha256 != str(configured_sha256)
    ):
        raise ValueError("D2-AG performance waiver SHA-256 mismatch")
    tracked_relative = _validate_tracked_d2ag_waiver_path(repo, path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("D2-AG performance waiver is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("D2-AG performance waiver must be a JSON object")

    waiver_match = D2AG_PERFORMANCE_WAIVER_RUN_ID_RE.fullmatch(
        str(value.get("run_id", ""))
    )
    formal_match = D2AG_FORMAL_RUN_ID_RE.fullmatch(str(cfg.run_id))
    benchmark_binding = value.get("benchmark")
    authorization = value.get("authorization")
    transition_binding = value.get("source_transition")
    preexisting = value.get("preexisting_formal_artifacts")
    non_speed = value.get("non_speed_contracts")
    if not isinstance(transition_binding, Mapping):
        raise ValueError("D2-AG performance waiver source transition is missing")
    source_commit = str(transition_binding.get("source_commit", ""))
    target_commit = str(transition_binding.get("target_commit", ""))
    transition = _d2ag_source_transition(repo, source_commit, target_commit)
    source_contract = _d2ag_formal_source_contract_at_commit(repo, source_commit)
    target_contract = _d2ag_formal_source_contract_at_commit(repo, target_commit)
    current_contract = _d2ag_formal_source_contract(repo)
    is_resume = cfg.resume_checkpoint not in (None, "", False)
    binding_labels = (
        (
            benchmark_binding.get("status"),
            benchmark_binding.get("classification"),
        )
        if isinstance(benchmark_binding, Mapping) else (None, None)
    )
    checkpoint_dir = Path(str(cfg.checkpoint_dir)).resolve()
    initial_outputs_absent = (
        not Path(str(cfg.metrics_path)).resolve().exists()
        and not Path(str(cfg.state_path)).resolve().exists()
        and (
            not checkpoint_dir.exists()
            or not any(checkpoint_dir.glob("*.pth"))
        )
    )
    checks = {
        "schema_version": value.get("schema_version") == 1,
        "status": value.get("status") == "authorized",
        "classification": value.get("classification")
        == D2AG_PERFORMANCE_WAIVER_CLASSIFICATION,
        "run_id": waiver_match is not None,
        "date_binding": waiver_match is not None
        and formal_match is not None
        and waiver_match.group("date") == formal_match.group("date"),
        "formal_run_id": value.get("formal_run_id") == str(cfg.run_id)
        == D2AG_WAIVED_FORMAL_RUN_ID,
        "seed": value.get("seed") == int(cfg.seed) == 42,
        "tracked_contract": tracked_relative
        == D2AG_PERFORMANCE_WAIVER_RELATIVE_PATH,
        # The waiver is bound to one exact measured benchmark.  It must keep
        # the failed status/classification and the measured numbers, and it
        # must acknowledge D2-AG's own registered thresholds.
        "benchmark_binding": isinstance(benchmark_binding, Mapping)
        and benchmark_binding.get("run_id") == D2AG_WAIVED_BENCHMARK_RUN_ID
        == benchmark.get("run_id")
        and benchmark_binding.get("sha256") == benchmark_sha256
        == D2AG_WAIVED_BENCHMARK_SHA256
        and benchmark_binding.get("status") == benchmark.get("status") == "failed"
        and benchmark_binding.get("classification")
        == benchmark.get("classification")
        == D2AG_PERFORMANCE_FAILURE_CLASSIFICATION
        and benchmark_binding.get("throughput_windows_per_second")
        == benchmark.get("throughput_windows_per_second")
        == D2AG_WAIVED_THROUGHPUT
        and benchmark_binding.get("full_budget_eta_hours")
        == benchmark.get("full_budget_eta_hours")
        == D2AG_WAIVED_ETA_HOURS
        and benchmark_binding.get("minimum_throughput_windows_per_second")
        == benchmark.get("minimum_throughput_windows_per_second")
        == D2AG_MINIMUM_THROUGHPUT
        and benchmark_binding.get("maximum_full_budget_eta_hours")
        == benchmark.get("maximum_full_budget_eta_hours")
        == D2AG_MAXIMUM_ETA_HOURS
        and benchmark_binding.get("formal_training_authorized") is False
        and benchmark.get("formal_training_authorized") is False
        and benchmark_binding.get("failed_checks")
        == sorted(D2AG_WAIVED_BENCHMARK_FAILED_CHECKS)
        and benchmark_binding.get("non_speed_contracts_passed") is True,
        # A waived run must never be labelled as a passing gate.
        "no_passed_claim": D2AG_FORBIDDEN_WAIVED_STATUS not in (
            value.get("status"),
            value.get("classification"),
            *binding_labels,
        ),
        "source_identity": source_commit == D2AG_WAIVED_SOURCE_COMMIT
        == benchmark.get("identity", {}).get("git_commit"),
        "source_contract": source_contract
        == benchmark.get("formal_source_contract")
        == transition_binding.get("source_formal_contract")
        and source_contract.get("sha256")
        == D2AG_WAIVED_SOURCE_CONTRACT_SHA256,
        "target_contract": target_contract
        == transition_binding.get("target_formal_contract")
        == current_contract,
        "transition": transition_binding.get("diff_sha256")
        == transition["diff_sha256"]
        and transition_binding.get("changed_paths")
        == transition["changed_paths"]
        and set(transition["changed_paths"]).issubset(
            D2AG_WAIVER_ALLOWED_CHANGED_PATHS
        ),
        "authorization": isinstance(authorization, Mapping)
        and authorization.get("user_authorized_after_full_failure_disclosure")
        is True
        and authorization.get("user_accepted_full_budget_eta_hours")
        == D2AG_WAIVED_ETA_HOURS
        and authorization.get("formal_runs_maximum") == 1
        and authorization.get("random_initialization") is True
        and authorization.get("benchmark_retry_authorized") is False
        and authorization.get("execution_sweep_authorized") is False
        and authorization.get("benchmark_reclassification_authorized") is False
        and authorization.get("benchmark_classification_unchanged") is True
        and authorization.get("history_rewritten") is False
        and authorization.get("training_conditions_unchanged") is True
        and authorization.get("runtime_status_label")
        == D2AG_PERFORMANCE_WAIVER_STATUS
        and authorization.get("forbidden_status_label")
        == D2AG_FORBIDDEN_WAIVED_STATUS
        and authorization.get("d2af_waiver_inherited") is False
        and authorization.get("bound_benchmark_run_ids")
        == [D2AG_WAIVED_BENCHMARK_RUN_ID]
        and authorization.get("bound_formal_run_ids")
        == [D2AG_WAIVED_FORMAL_RUN_ID]
        and bool(cfg.get("profile_every_update")) is True,
        # Every non-speed contract must still be true in the waiver record.
        "non_speed_contracts": isinstance(non_speed, Mapping)
        and non_speed.get("all_rank_contract_pass") is True
        and non_speed.get("memory_headroom_pass") is True
        and non_speed.get("contention_pass") is True
        and non_speed.get("losses_finite") is True
        and non_speed.get("gradients_finite") is True
        and non_speed.get("selfcond_estimate_forward_measured") is True
        and non_speed.get("invalid_if_any_non_speed_contract_fails") is True,
        "preexisting_artifacts": isinstance(preexisting, Mapping)
        and preexisting.get("formal_output_directory_existed") is False
        and preexisting.get("training_state_existed") is False
        and preexisting.get("training_metrics_existed") is False
        and preexisting.get("checkpoint_count") == 0,
        "initial_outputs": is_resume or initial_outputs_absent,
        "fresh_or_same_run_resume": (
            cfg.resume_checkpoint in (None, "", False)
            or Path(str(cfg.resume_checkpoint)).name.startswith(
                f"{cfg.run_id}_windows"
            )
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "D2-AG performance waiver contract mismatch: " + ", ".join(failed)
        )
    return {
        "path": str(path.resolve()),
        "relative_path": tracked_relative,
        "sha256": actual_sha256,
        "run_id": str(value["run_id"]),
        "status": D2AG_PERFORMANCE_WAIVER_STATUS,
        "classification": D2AG_PERFORMANCE_WAIVER_CLASSIFICATION,
        "formal_run_id": str(cfg.run_id),
        "benchmark_run_id": D2AG_WAIVED_BENCHMARK_RUN_ID,
        "benchmark_sha256": benchmark_sha256,
        "source_transition": transition,
        "source_formal_contract": source_contract,
        "target_formal_contract": current_contract,
        "checks": checks,
    }


def _validate_d2ag_performance_gate(cfg: DictConfig) -> Dict[str, object]:
    """Require a passing benchmark or the one exact hash-bound failed-run waiver.

    The D2-AG floor is registered independently (85% of the sealed D2-X formal
    throughput).  The D2-AF waiver is run-id bound and is not reachable here.
    Without a waiver the behaviour is unchanged: any failed check stops formal
    training.  With one, only the measured throughput/ETA shortfall of one
    exact benchmark is excused, and the result is reported as ``failed-waived``.
    """
    path_value = cfg.get("d2ag_performance_benchmark_path")
    configured_sha256 = cfg.get("d2ag_performance_benchmark_sha256")
    if path_value in (None, "", False) or configured_sha256 in (None, "", False):
        raise ValueError("D2-AG formal training requires a sealed performance benchmark")
    path = Path(str(path_value))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("D2-AG performance benchmark path must be an existing absolute file")
    configured_sha256 = str(configured_sha256)
    actual_sha256 = _sha256(path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", configured_sha256) is None
        or actual_sha256 != configured_sha256
    ):
        raise ValueError("D2-AG performance benchmark SHA-256 mismatch")
    try:
        benchmark = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("D2-AG performance benchmark is not valid JSON") from error
    if not isinstance(benchmark, Mapping):
        raise ValueError("D2-AG performance benchmark must be a JSON object")

    repo = Path(str(cfg.repo_root)).resolve()
    identity = benchmark.get("identity")
    benchmark_run_id = str(benchmark.get("run_id", ""))
    benchmark_run_match = D2AG_PERFORMANCE_RUN_ID_RE.fullmatch(benchmark_run_id)
    formal_run_id = str(benchmark.get("formal_run_id", ""))
    configured_formal_run_id = str(cfg.run_id)
    configured_formal_match = D2AG_FORMAL_RUN_ID_RE.fullmatch(
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
        and float(throughput) >= D2AG_MINIMUM_THROUGHPUT,
        "eta": numeric_metrics and float(eta_hours) <= D2AG_MAXIMUM_ETA_HOURS,
        "throughput_threshold": benchmark.get(
            "minimum_throughput_windows_per_second"
        ) == D2AG_MINIMUM_THROUGHPUT,
        "eta_threshold": benchmark.get("maximum_full_budget_eta_hours")
        == D2AG_MAXIMUM_ETA_HOURS,
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
        "selfcond_estimate_forward_measured": benchmark.get(
            "selfcond_estimate_forward_measured"
        ) is True,
        "selection_probability": benchmark.get("selection_probability")
        == D2AG_SELF_CONDITION_PROBABILITY,
        "waiver_absent": benchmark.get("performance_waiver") in (None, False),
        "identity_mapping": isinstance(identity, Mapping),
        "identity_clean": isinstance(identity, Mapping)
        and identity.get("worktree_clean") is True,
        "benchmark_commit": commit_valid,
        "benchmark_ancestor": ancestor,
        "source_contract": benchmark.get("formal_source_contract")
        == _d2ag_formal_source_contract(repo),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    waiver_requested = (
        cfg.get("d2ag_performance_waiver_path") not in (None, "", False)
        or cfg.get("d2ag_performance_waiver_sha256") not in (None, "", False)
    )
    if not failed:
        if waiver_requested:
            raise ValueError(
                "D2-AG performance benchmark contract mismatch: waiver_absent"
            )
        return {
            "status": "performance-gate-passed",
            "classification": "performance-gate-passed",
            "original_gate_passed": True,
            "formal_authorization": "registered-performance-gate",
            "path": str(path.resolve()),
            "sha256": actual_sha256,
            "run_id": str(benchmark["run_id"]),
            "formal_run_id": formal_run_id,
            "git_commit": benchmark_commit,
            "throughput_windows_per_second": float(throughput),
            "full_budget_eta_hours": float(eta_hours),
            "memory_headroom_min_bytes": int(headroom),
            "minimum_throughput_windows_per_second": D2AG_MINIMUM_THROUGHPUT,
            "maximum_full_budget_eta_hours": D2AG_MAXIMUM_ETA_HOURS,
            "formal_source_contract": dict(benchmark["formal_source_contract"]),
            "performance_waiver": None,
            "checks": checks,
        }

    # Below this point the benchmark failed.  Only the one user-authorized,
    # hash-bound waiver can proceed, only for the exact measured shortfall of
    # one exact benchmark, and only while every other contract still passes.
    expected_waived_failures = set(D2AG_WAIVED_BENCHMARK_FAILED_CHECKS)
    # The validator-only transition the waiver authorizes rewrites tracked
    # source, so the benchmark's source contract no longer equals the current
    # tree.  The waiver re-establishes it against both committed trees.
    permitted = (
        expected_waived_failures,
        expected_waived_failures | {"source_contract"},
    )
    if set(failed) not in permitted:
        raise ValueError(
            "D2-AG performance benchmark contract mismatch: " + ", ".join(failed)
        )
    if (
        actual_sha256 != D2AG_WAIVED_BENCHMARK_SHA256
        or benchmark.get("run_id") != D2AG_WAIVED_BENCHMARK_RUN_ID
        or benchmark.get("formal_run_id") != D2AG_WAIVED_FORMAL_RUN_ID
        or benchmark.get("status") != "failed"
        or benchmark.get("classification")
        != D2AG_PERFORMANCE_FAILURE_CLASSIFICATION
        or benchmark.get("throughput_windows_per_second") != D2AG_WAIVED_THROUGHPUT
        or benchmark.get("full_budget_eta_hours") != D2AG_WAIVED_ETA_HOURS
        or benchmark.get("formal_training_authorized") is not False
    ):
        raise ValueError("D2-AG performance benchmark is not the exact waived failure")
    waiver = _validate_d2ag_performance_waiver(
        cfg,
        benchmark=benchmark,
        benchmark_sha256=actual_sha256,
        repo=repo,
    )
    benchmark_checks = dict(checks)
    # Re-established by the waiver against the committed source/target trees.
    benchmark_checks["source_contract"] = True
    authorization_checks = {
        "benchmark_failure_exact": set(failed) in permitted,
        "benchmark_non_speed_contracts": all(
            passed
            for name, passed in benchmark_checks.items()
            if name not in expected_waived_failures
        ),
        "waiver": all(waiver["checks"].values()),
        "training_conditions_unchanged": bool(cfg.get("profile_every_update")),
        "status_not_reported_as_passed": (
            D2AG_PERFORMANCE_WAIVER_STATUS != D2AG_FORBIDDEN_WAIVED_STATUS
        ),
    }
    failed_authorization = sorted(
        name for name, passed in authorization_checks.items() if not passed
    )
    if failed_authorization:
        raise ValueError(
            "D2-AG performance waiver authorization mismatch: "
            + ", ".join(failed_authorization)
        )
    return {
        "status": D2AG_PERFORMANCE_WAIVER_STATUS,
        "classification": D2AG_PERFORMANCE_WAIVER_CLASSIFICATION,
        "original_gate_passed": False,
        "formal_authorization": "explicit-single-run-waiver",
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "run_id": str(benchmark["run_id"]),
        "formal_run_id": formal_run_id,
        "git_commit": benchmark_commit,
        "throughput_windows_per_second": float(throughput),
        "full_budget_eta_hours": float(eta_hours),
        "memory_headroom_min_bytes": int(headroom),
        "minimum_throughput_windows_per_second": D2AG_MINIMUM_THROUGHPUT,
        "maximum_full_budget_eta_hours": D2AG_MAXIMUM_ETA_HOURS,
        "benchmark_status": str(benchmark["status"]),
        "benchmark_classification": str(benchmark["classification"]),
        "benchmark_failed_checks": sorted(expected_waived_failures),
        "formal_source_contract": dict(waiver["target_formal_contract"]),
        "benchmark_formal_source_contract": dict(
            benchmark["formal_source_contract"]
        ),
        "performance_waiver": waiver,
        "benchmark_checks": benchmark_checks,
        "checks": authorization_checks,
    }


def _validate_d2af_eligibility_gate(
    cfg: DictConfig,
    *,
    expected_formal_source_contract: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Require the immutable no-checkpoint clean-signal premise gate."""
    path_value = cfg.get("d2af_clean_signal_eligibility_path")
    configured_sha256 = cfg.get("d2af_clean_signal_eligibility_sha256")
    if path_value in (None, "", False) or configured_sha256 in (None, "", False):
        raise ValueError("D2-AF formal training requires a sealed eligibility gate")
    path = Path(str(path_value))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("D2-AF eligibility path must be an existing absolute file")
    actual_sha256 = _sha256(path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(configured_sha256)) is None
        or actual_sha256 != str(configured_sha256)
    ):
        raise ValueError("D2-AF eligibility SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("D2-AF eligibility is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("D2-AF eligibility must be a JSON object")
    identity = value.get("identity")
    selection = value.get("selection")
    schedule = value.get("schedule")
    gates = value.get("gates")
    repo = Path(str(cfg.repo_root)).resolve()
    formal_source_contract = (
        dict(expected_formal_source_contract)
        if expected_formal_source_contract is not None
        else _d2af_formal_source_contract(repo)
    )

    def prerequisite_binding(
        binding: object,
        *,
        run_id_pattern: str,
        status: str,
        classification: str,
    ) -> bool:
        if not isinstance(binding, Mapping):
            return False
        binding_path = binding.get("path")
        resolved_path = binding.get("resolved_config_path")
        checks_value = binding.get("checks")
        return bool(
            isinstance(binding_path, str)
            and Path(binding_path).is_absolute()
            and re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", "")))
            is not None
            and re.fullmatch(run_id_pattern, str(binding.get("run_id", "")))
            is not None
            and binding.get("status") == status
            and binding.get("classification") == classification
            and _git_commit_is_ancestor(repo, binding.get("git_commit"))
            and binding.get("formal_source_contract") == formal_source_contract
            and isinstance(resolved_path, str)
            and Path(resolved_path).is_absolute()
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(binding.get("resolved_config_sha256", "")),
            ) is not None
            and isinstance(checks_value, Mapping)
            and bool(checks_value)
            and all(bool(item) for item in checks_value.values())
        )

    noise_streams = value.get("noise_streams")
    noise_stream_contract = bool(
        isinstance(noise_streams, Mapping)
        and set(noise_streams) == {"0", "249", "499"}
        and all(
            isinstance(noise_streams[str(timestep)], Mapping)
            and noise_streams[str(timestep)].get("seed")
            == 42 + 1_000_003 * timestep
            and noise_streams[str(timestep)].get("device") == "cpu"
            and noise_streams[str(timestep)].get("dtype") == "torch.float32"
            and noise_streams[str(timestep)].get("shape_per_window")
            == [16, 232]
            and noise_streams[str(timestep)].get("values")
            == 29382 * 16 * 232
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(noise_streams[str(timestep)].get("sha256", "")),
            ) is not None
            for timestep in (0, 249, 499)
        )
    )
    sparse_assets = value.get("sparse_assets")
    checks = {
        "schema_version": value.get("schema_version") == 1,
        "status": value.get("status") == "passed",
        "classification": value.get("classification")
        == "clean-signal-premise-passed",
        "run_id": re.fullmatch(
            r"p1-hoi-d2af-clean-signal-eligibility"
            r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}",
            str(value.get("run_id", "")),
        ) is not None,
        "seed": value.get("seed") == 42,
        "checkpoint_loads": value.get("checkpoint_loads") == 0,
        "model_created": value.get("model_created") is False,
        "optimizer_created": value.get("optimizer_created") is False,
        "official_test_used": value.get("official_test_used") is False,
        "selection": isinstance(selection, Mapping)
        and selection.get("partition") == "internal_validation"
        and selection.get("sequences") == 216
        and selection.get("windows") == 29382
        and selection.get("global_indices_sha256")
        == "eab0bde2dc2ddad7ce2cc1817973ca46b9adaf24b1c906307f865930aeb11eb9"
        and selection.get("sequence_names_sha256")
        == "472768c85c6d6c5b682a31a4d40a879d7a1e3d0b16085923c153db1045223fd8",
        "schedule": schedule
        == diffusion_reliability_contract_metadata()["schedule"],
        "gates": isinstance(gates, Mapping)
        and gates.get("c249_minus_c0_ci_lower_gt_zero") is True
        and gates.get("c499_minus_c249_ci_lower_gt_zero") is True
        and gates.get("anchor0_prescaling_max_abs_le_1e_minus_6") is True,
        "source_contract": value.get("formal_source_contract")
        == formal_source_contract,
        "identity": isinstance(identity, Mapping)
        and _git_commit_is_ancestor(repo, identity.get("git_commit"))
        and identity.get("worktree_clean") is True,
        "authority_cpu_contract": prerequisite_binding(
            value.get("authority_cpu_contract"),
            run_id_pattern=(
                r"p1-hoi-d2af-cpu-contract"
                r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}"
            ),
            status="passed",
            classification="cpu-contract-passed",
        ),
        "functional_smoke": prerequisite_binding(
            value.get("functional_smoke"),
            run_id_pattern=(
                r"p1-hoi-d2af-gpu-functional-smoke"
                r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}"
            ),
            status="stable",
            classification="functional-smoke-passed",
        ),
        "prerequisite_source_contract": (
            value.get("prerequisite_source_contract_match") is True
        ),
        "noise_streams": noise_stream_contract,
        "sparse_assets": isinstance(sparse_assets, Mapping)
        and sparse_assets.get("mapping_sha256") == SPARSE_POINT_MAPPING_SHA256
        and sparse_assets.get("manifest_sha256") == SPARSE_POINT_MANIFEST_SHA256
        and sparse_assets.get("stacked_tensor_sha256")
        == SPARSE_POINT_TENSOR_SHA256,
        "resolved_config": isinstance(value.get("resolved_config_path"), str)
        and Path(str(value["resolved_config_path"])).is_absolute()
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(value.get("resolved_config_sha256", "")),
        ) is not None
        and value.get("resolved_config_has_unresolved_interpolation") is False,
        "formal_authorized": value.get("formal_training_authorized") is True,
        "performance_authorized": (
            value.get("performance_benchmark_authorized") is True
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "D2-AF clean-signal eligibility contract mismatch: "
            + ", ".join(failed)
        )
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "run_id": str(value["run_id"]),
        "formal_source_contract": dict(value["formal_source_contract"]),
        "authority_cpu_contract": dict(value["authority_cpu_contract"]),
        "functional_smoke": dict(value["functional_smoke"]),
        "checks": checks,
    }


def _validate_d2af_performance_waiver(
    cfg: DictConfig,
    *,
    benchmark: Mapping[str, object],
    benchmark_sha256: str,
    eligibility: Mapping[str, object],
    repo: Path,
) -> Dict[str, object]:
    """Validate the one exact post-failure, user-authorized formal-run waiver."""
    path_value = cfg.get("d2af_performance_waiver_path")
    configured_sha256 = cfg.get("d2af_performance_waiver_sha256")
    if path_value in (None, "", False) or configured_sha256 in (None, "", False):
        raise ValueError(
            "D2-AF failed performance benchmark requires an explicit sealed waiver"
        )
    path = Path(str(path_value))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("D2-AF performance waiver path must be an existing absolute file")
    actual_sha256 = _sha256(path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(configured_sha256)) is None
        or actual_sha256 != str(configured_sha256)
    ):
        raise ValueError("D2-AF performance waiver SHA-256 mismatch")
    tracked_relative = _validate_tracked_d2af_waiver_path(repo, path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("D2-AF performance waiver is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("D2-AF performance waiver must be a JSON object")

    waiver_match = D2AF_PERFORMANCE_WAIVER_RUN_ID_RE.fullmatch(
        str(value.get("run_id", ""))
    )
    formal_match = D2AF_FORMAL_RUN_ID_RE.fullmatch(str(cfg.run_id))
    benchmark_binding = value.get("benchmark")
    authorization = value.get("authorization")
    transition_binding = value.get("source_transition")
    preexisting = value.get("preexisting_formal_artifacts")
    if not isinstance(transition_binding, Mapping):
        raise ValueError("D2-AF performance waiver source transition is missing")
    source_commit = str(transition_binding.get("source_commit", ""))
    target_commit = str(transition_binding.get("target_commit", ""))
    transition = _d2af_source_transition(repo, source_commit, target_commit)
    source_contract = _d2af_formal_source_contract_at_commit(repo, source_commit)
    target_contract = _d2af_formal_source_contract_at_commit(repo, target_commit)
    current_contract = _d2af_formal_source_contract(repo)
    is_resume = cfg.resume_checkpoint not in (None, "", False)
    operational_continuation = None
    if is_resume and current_contract != target_contract:
        operational_continuation = _validate_d2af_checkpoint_race_continuation(
            cfg,
            repo=repo,
            waiver_target_contract=target_contract,
        )
    checkpoint_dir = Path(str(cfg.checkpoint_dir)).resolve()
    initial_outputs_absent = (
        not Path(str(cfg.metrics_path)).resolve().exists()
        and not Path(str(cfg.state_path)).resolve().exists()
        and (
            not checkpoint_dir.exists()
            or not any(checkpoint_dir.glob("*.pth"))
        )
    )
    checks = {
        "schema_version": value.get("schema_version") == 1,
        "status": value.get("status") == "authorized",
        "classification": value.get("classification")
        == D2AF_PERFORMANCE_WAIVER_CLASSIFICATION,
        "run_id": waiver_match is not None,
        "date_binding": waiver_match is not None
        and formal_match is not None
        and waiver_match.group("date") == formal_match.group("date"),
        "formal_run_id": value.get("formal_run_id") == str(cfg.run_id)
        == D2AF_WAIVED_FORMAL_RUN_ID,
        "seed": value.get("seed") == int(cfg.seed) == 42,
        "tracked_contract": tracked_relative
        == "experiments/contracts/"
        "p1_hoi_d2af_performance_waiver_s42_20260729.json",
        "benchmark_binding": isinstance(benchmark_binding, Mapping)
        and benchmark_binding.get("run_id") == D2AF_WAIVED_BENCHMARK_RUN_ID
        == benchmark.get("run_id")
        and benchmark_binding.get("sha256") == benchmark_sha256
        == D2AF_WAIVED_BENCHMARK_SHA256
        and benchmark_binding.get("status") == benchmark.get("status") == "failed"
        and benchmark_binding.get("classification")
        == benchmark.get("classification")
        == D2AF_PERFORMANCE_FAILURE_CLASSIFICATION
        and benchmark_binding.get("throughput_windows_per_second")
        == benchmark.get("throughput_windows_per_second")
        == D2AF_WAIVED_THROUGHPUT
        and benchmark_binding.get("full_budget_eta_hours")
        == benchmark.get("full_budget_eta_hours")
        == D2AF_WAIVED_ETA_HOURS
        and benchmark_binding.get("formal_training_authorized") is False
        and benchmark.get("formal_training_authorized") is False
        and benchmark_binding.get("failed_checks")
        == ["classification", "eta", "formal_authorized", "status", "throughput"]
        and benchmark_binding.get("non_speed_contracts_passed") is True,
        "eligibility_binding": value.get("eligibility_sha256")
        == eligibility.get("sha256")
        == D2AF_WAIVED_ELIGIBILITY_SHA256,
        "source_identity": source_commit == D2AF_WAIVED_SOURCE_COMMIT
        == benchmark.get("identity", {}).get("git_commit"),
        "source_contract": source_contract
        == benchmark.get("formal_source_contract")
        == transition_binding.get("source_formal_contract")
        and source_contract.get("sha256")
        == D2AF_WAIVED_SOURCE_CONTRACT_SHA256,
        "target_contract": (
            target_contract
            == transition_binding.get("target_formal_contract")
            and (
                current_contract == target_contract
                or (
                    isinstance(operational_continuation, Mapping)
                    and operational_continuation.get(
                        "source_formal_contract"
                    ) == target_contract
                    and operational_continuation.get(
                        "target_formal_contract"
                    ) == current_contract
                )
            )
        ),
        "transition": transition_binding.get("diff_sha256")
        == transition["diff_sha256"]
        and transition_binding.get("changed_paths")
        == transition["changed_paths"]
        and set(transition["changed_paths"]).issubset(
            D2AF_WAIVER_ALLOWED_CHANGED_PATHS
        ),
        "authorization": isinstance(authorization, Mapping)
        and authorization.get("user_accepted_full_budget_eta_hours")
        == D2AF_WAIVED_ETA_HOURS
        and authorization.get("formal_runs_maximum") == 1
        and authorization.get("random_initialization") is True
        and authorization.get("benchmark_retry_authorized") is False
        and authorization.get("execution_sweep_authorized") is False
        and authorization.get("benchmark_reclassification_authorized") is False
        and authorization.get("training_conditions_unchanged") is True
        and authorization.get("profile_every_update") is True
        and bool(cfg.get("profile_every_update")) is True,
        "preexisting_artifacts": isinstance(preexisting, Mapping)
        and preexisting.get("formal_output_directory_existed") is False
        and preexisting.get("training_state_existed") is False
        and preexisting.get("training_metrics_existed") is False
        and preexisting.get("checkpoint_count") == 0,
        "initial_outputs": is_resume or initial_outputs_absent,
        "fresh_or_same_run_resume": (
            cfg.resume_checkpoint in (None, "", False)
            or Path(str(cfg.resume_checkpoint)).name.startswith(
                f"{cfg.run_id}_windows"
            )
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "D2-AF performance waiver contract mismatch: " + ", ".join(failed)
        )
    return {
        "path": str(path.resolve()),
        "relative_path": tracked_relative,
        "sha256": actual_sha256,
        "run_id": str(value["run_id"]),
        "classification": D2AF_PERFORMANCE_WAIVER_CLASSIFICATION,
        "formal_run_id": str(cfg.run_id),
        "source_transition": transition,
        "source_formal_contract": source_contract,
        "target_formal_contract": current_contract,
        "original_waiver_target_formal_contract": target_contract,
        "operational_continuation": operational_continuation,
        "checks": checks,
    }


def _validate_d2af_performance_gate(cfg: DictConfig) -> Dict[str, object]:
    """Require a passing benchmark or the one exact hash-bound failed-run waiver."""
    path_value = cfg.get("d2af_performance_benchmark_path")
    configured_sha256 = cfg.get("d2af_performance_benchmark_sha256")
    if path_value in (None, "", False) or configured_sha256 in (None, "", False):
        raise ValueError("D2-AF formal training requires a sealed performance benchmark")
    path = Path(str(path_value))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("D2-AF performance benchmark path must be an existing absolute file")
    actual_sha256 = _sha256(path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(configured_sha256)) is None
        or actual_sha256 != str(configured_sha256)
    ):
        raise ValueError("D2-AF performance benchmark SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("D2-AF performance benchmark is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("D2-AF performance benchmark must be a JSON object")
    identity = value.get("identity")
    benchmark_match = D2AF_PERFORMANCE_RUN_ID_RE.fullmatch(
        str(value.get("run_id", ""))
    )
    formal_match = D2AF_FORMAL_RUN_ID_RE.fullmatch(str(cfg.run_id))
    throughput = value.get("throughput_windows_per_second")
    eta = value.get("full_budget_eta_hours")
    headroom = value.get("memory_headroom_min_bytes")
    required = value.get("memory_headroom_required_bytes")
    numeric = (
        isinstance(throughput, (int, float))
        and not isinstance(throughput, bool)
        and math.isfinite(float(throughput))
        and isinstance(eta, (int, float))
        and not isinstance(eta, bool)
        and math.isfinite(float(eta))
    )
    repo = Path(str(cfg.repo_root)).resolve()
    benchmark_source = value.get("formal_source_contract")
    if not isinstance(benchmark_source, Mapping):
        raise ValueError("D2-AF performance benchmark source contract is missing")
    eligibility = _validate_d2af_eligibility_gate(
        cfg,
        expected_formal_source_contract=benchmark_source,
    )
    checks = {
        "schema_version": value.get("schema_version") == 1,
        "status": value.get("status") == "passed",
        "classification": value.get("classification") == "performance-gate-passed",
        "run_id": benchmark_match is not None,
        "formal_run_id": formal_match is not None
        and benchmark_match is not None
        and value.get("formal_run_id") == str(cfg.run_id)
        and formal_match.group("date") == benchmark_match.group("date"),
        "seed": value.get("seed") == 42,
        "world_size": value.get("world_size") == 4,
        "micro_batch": value.get("micro_batch_per_gpu") == 512,
        "effective_batch": value.get("effective_batch_size") == 2048,
        "updates": value.get("warmup_updates") == 64
        and value.get("measured_updates") == 256
        and value.get("total_updates") == 320
        and value.get("measured_windows") == 524288,
        "throughput": numeric
        and float(throughput) >= D2AF_MINIMUM_THROUGHPUT,
        "eta": numeric and float(eta) <= D2AF_MAXIMUM_ETA_HOURS,
        "thresholds": value.get("minimum_throughput_windows_per_second")
        == D2AF_MINIMUM_THROUGHPUT
        and value.get("maximum_full_budget_eta_hours")
        == D2AF_MAXIMUM_ETA_HOURS,
        "memory": isinstance(headroom, int)
        and isinstance(required, int)
        and headroom >= required
        and required >= 2 * 1024**3
        and value.get("memory_headroom_pass") is True,
        "finite": value.get("losses_finite") is True
        and value.get("gradients_finite") is True,
        "gpu_only": value.get("relation_gpu_only") is True
        and value.get("cpu_dynamic_geometry") is False
        and value.get("relation_build_device") == "cuda",
        "timing": value.get("cuda_timing_synchronized") is True,
        "optimizer": value.get("optimizer") == "FP32 Adam"
        and value.get("optimizer_updates") == 320,
        "checkpoint_io": value.get("checkpoint_loads") == 0
        and value.get("checkpoint_writes") == 0
        and value.get("benchmark_weights_reusable") is False,
        "all_rank_contract": value.get("all_rank_contract_pass") is True,
        "four_rank_identity": value.get("four_rank_schedule_hash_pass") is True
        and value.get("four_rank_initial_model_hash_pass") is True,
        "contention": value.get("contention_pass") is True,
        "schedule": value.get("sqrt_alpha_bar_sha256")
        == diffusion_reliability_contract_metadata()["schedule"][
            "sqrt_alpha_bar_sha256"
        ],
        "eligibility": value.get("eligibility_sha256") == eligibility["sha256"],
        "identity": isinstance(identity, Mapping)
        and _git_commit_is_ancestor(repo, identity.get("git_commit"))
        and identity.get("worktree_clean") is True,
        "formal_authorized": value.get("formal_training_authorized") is True,
        "sweep": value.get("sweep_authorized_on_failure") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if not failed:
        waiver_absent = (
            cfg.get("d2af_performance_waiver_path") in (None, "", False)
            and cfg.get("d2af_performance_waiver_sha256") in (None, "", False)
        )
        direct_source = benchmark_source == _d2af_formal_source_contract(repo)
        if not waiver_absent or not direct_source:
            extra = []
            if not waiver_absent:
                extra.append("waiver_absent")
            if not direct_source:
                extra.append("source_contract")
            raise ValueError(
                "D2-AF performance benchmark contract mismatch: "
                + ", ".join(extra)
            )
        checks["source_contract"] = True
        checks["waiver_absent"] = True
        return {
            "status": "performance-gate-passed",
            "classification": "performance-gate-passed",
            "original_gate_passed": True,
            "formal_authorization": "registered-performance-gate",
            "path": str(path.resolve()),
            "sha256": actual_sha256,
            "run_id": str(value["run_id"]),
            "formal_run_id": str(value["formal_run_id"]),
            "throughput_windows_per_second": float(throughput),
            "full_budget_eta_hours": float(eta),
            "memory_headroom_min_bytes": int(headroom),
            "benchmark_formal_source_contract": dict(benchmark_source),
            "formal_source_contract": dict(benchmark_source),
            "eligibility": eligibility,
            "waiver": None,
            "benchmark_checks": dict(checks),
            "checks": dict(checks),
        }

    expected_waived_failures = {
        "status", "classification", "throughput", "eta", "formal_authorized",
    }
    if set(failed) != expected_waived_failures:
        raise ValueError(
            "D2-AF performance benchmark contract mismatch: " + ", ".join(failed)
        )
    if (
        actual_sha256 != D2AF_WAIVED_BENCHMARK_SHA256
        or value.get("run_id") != D2AF_WAIVED_BENCHMARK_RUN_ID
        or value.get("formal_run_id") != D2AF_WAIVED_FORMAL_RUN_ID
        or value.get("status") != "failed"
        or value.get("classification")
        != D2AF_PERFORMANCE_FAILURE_CLASSIFICATION
        or value.get("throughput_windows_per_second") != D2AF_WAIVED_THROUGHPUT
        or value.get("full_budget_eta_hours") != D2AF_WAIVED_ETA_HOURS
        or value.get("formal_training_authorized") is not False
    ):
        raise ValueError("D2-AF performance benchmark is not the exact waived failure")
    waiver = _validate_d2af_performance_waiver(
        cfg,
        benchmark=value,
        benchmark_sha256=actual_sha256,
        eligibility=eligibility,
        repo=repo,
    )
    checks["source_contract"] = True
    authorization_checks = {
        "benchmark_failure_exact": set(failed) == expected_waived_failures,
        "benchmark_non_speed_contracts": all(
            passed
            for name, passed in checks.items()
            if name not in expected_waived_failures
        ),
        "waiver": all(waiver["checks"].values()),
        "training_conditions_unchanged": bool(cfg.get("profile_every_update")),
    }
    return {
        "status": "failed-waived",
        "classification": D2AF_PERFORMANCE_WAIVER_CLASSIFICATION,
        "original_gate_passed": False,
        "formal_authorization": "explicit-single-run-waiver",
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "run_id": str(value["run_id"]),
        "formal_run_id": str(value["formal_run_id"]),
        "throughput_windows_per_second": float(throughput),
        "full_budget_eta_hours": float(eta),
        "memory_headroom_min_bytes": int(headroom),
        "benchmark_status": str(value["status"]),
        "benchmark_classification": str(value["classification"]),
        "benchmark_failed_checks": sorted(expected_waived_failures),
        "benchmark_formal_source_contract": dict(benchmark_source),
        "formal_source_contract": dict(waiver["target_formal_contract"]),
        "eligibility": eligibility,
        "waiver": waiver,
        "benchmark_checks": checks,
        "checks": authorization_checks,
    }


_RESUME_TRANSITION_ALLOWED_PATHS = frozenset(
    {
        "code/config/config_train_hoi_prior.yaml",
        "code/config/config_train_hoi_prior_d2af.yaml",
        "code/train_hoi_prior.py",
        "docs/EXPERIMENT_PLAN.md",
        "experiments/registry.jsonl",
        "tests/test_hoi_d2ab.py",
        "tests/test_hoi_d2af.py",
        "tests/test_hoi_d2af_lifecycle_cpu.py",
        D2AF_CHECKPOINT_RACE_CONTINUATION_RELATIVE_PATH,
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
        or bool(cfg.get("d2af_sqrt_alpha_bar_reliability", False))
        or bool(cfg.get("d2ag_selfcond_relation_source", False))
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
        "d2af_sqrt_alpha_bar_reliability": bool(
            cfg.get("d2af_sqrt_alpha_bar_reliability", False)
        ),
        "d2ag_selfcond_relation_source": bool(
            cfg.get("d2ag_selfcond_relation_source", False)
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
    if _is_d2af(cfg):
        contract["d2af_sqrt_alpha_bar_reliability"] = True
        contract["architecture_variant"] = str(cfg.get("hoi_architecture_variant"))
        contract["d2af_sparse_relation_parameters"] = SPARSE_RELATION_PARAMETER_COUNT
        contract["d2af_sparse_point_mapping_sha256"] = SPARSE_POINT_MAPPING_SHA256
        contract["d2af_sparse_point_manifest_sha256"] = SPARSE_POINT_MANIFEST_SHA256
        contract["d2af_sparse_point_tensor_sha256"] = SPARSE_POINT_TENSOR_SHA256
        contract["d2af_schedule"] = diffusion_reliability_contract_metadata()["schedule"]
        contract["d2af_clean_signal_eligibility_path"] = str(
            Path(str(cfg.d2af_clean_signal_eligibility_path)).resolve()
        )
        contract["d2af_clean_signal_eligibility_sha256"] = str(
            cfg.d2af_clean_signal_eligibility_sha256
        )
        contract["d2af_performance_benchmark_path"] = str(
            Path(str(cfg.d2af_performance_benchmark_path)).resolve()
        )
        contract["d2af_performance_benchmark_sha256"] = str(
            cfg.d2af_performance_benchmark_sha256
        )
        contract["d2af_performance_waiver_path"] = (
            None
            if cfg.get("d2af_performance_waiver_path") in (None, "", False)
            else str(
                Path(str(cfg.d2af_performance_waiver_path)).resolve()
            )
        )
        contract["d2af_performance_waiver_sha256"] = (
            None
            if cfg.get("d2af_performance_waiver_sha256") in (None, "", False)
            else str(cfg.d2af_performance_waiver_sha256)
        )
    if _is_d2ag(cfg):
        contract["d2ag_selfcond_relation_source"] = True
        contract["architecture_variant"] = str(cfg.get("hoi_architecture_variant"))
        contract["d2ag_sparse_relation_parameters"] = SPARSE_RELATION_PARAMETER_COUNT
        contract["d2ag_sparse_point_mapping_sha256"] = SPARSE_POINT_MAPPING_SHA256
        contract["d2ag_sparse_point_manifest_sha256"] = SPARSE_POINT_MANIFEST_SHA256
        contract["d2ag_sparse_point_tensor_sha256"] = SPARSE_POINT_TENSOR_SHA256
        contract["d2ag_selection_probability"] = D2AG_SELF_CONDITION_PROBABILITY
        contract["d2ag_variable_anchors"] = list(D2AG_VARIABLE_ANCHORS)
        contract["d2ag_performance_benchmark_path"] = str(
            Path(str(cfg.d2ag_performance_benchmark_path)).resolve()
        )
        contract["d2ag_performance_benchmark_sha256"] = str(
            cfg.d2ag_performance_benchmark_sha256
        )
        contract["d2ag_performance_waiver_path"] = (
            None
            if cfg.get("d2ag_performance_waiver_path") in (None, "", False)
            else str(
                Path(str(cfg.d2ag_performance_waiver_path)).resolve()
            )
        )
        contract["d2ag_performance_waiver_sha256"] = (
            None
            if cfg.get("d2ag_performance_waiver_sha256") in (None, "", False)
            else str(cfg.d2ag_performance_waiver_sha256)
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


def _is_d2af(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2af_sqrt_alpha_bar_reliability", False))


def _is_d2ag(cfg: DictConfig) -> bool:
    return bool(cfg.get("d2ag_selfcond_relation_source", False))


def _is_sparse_relation(cfg: DictConfig) -> bool:
    return _is_d2ae(cfg) or _is_d2af(cfg) or _is_d2ag(cfg)


def _uses_author_update_rule(cfg: DictConfig) -> bool:
    return (
        _is_d2t(cfg) or _is_d2u(cfg) or _is_d2v(cfg)
        or _is_d2x(cfg) or _is_d2y(cfg) or _is_d2z(cfg) or _is_d2ab(cfg)
        or _is_d2ac(cfg) or _is_d2ad(cfg) or _is_sparse_relation(cfg)
    )


def _validate_fk_foot_temporal_routing_mode(cfg: DictConfig) -> None:
    routing = bool(cfg.get("fk_foot_temporal_routing", False))
    multiplier = float(cfg.get("routed_foot_residual_multiplier", 1.0))
    gating = bool(cfg.get("immutable_gt_near_ground_gating", False))
    if routing and not (
        _is_d2x(cfg) or _is_d2y(cfg) or _is_d2z(cfg) or _is_d2ab(cfg)
        or _is_d2ac(cfg) or _is_d2ad(cfg) or _is_sparse_relation(cfg)
    ):
        raise ValueError(
            "FK-foot temporal routing is restricted to registered "
            "D2-X/D2-Y/D2-Z/D2-AB/D2-AC/D2-AD/D2-AE/D2-AF/D2-AG modes"
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
    if _is_d2af(cfg) and not routing:
        raise ValueError("D2-AF reliability routing requires D2-X FK-foot routing")
    if _is_d2ag(cfg) and not routing:
        raise ValueError(
            "D2-AG selfcond relation source requires D2-X FK-foot routing"
        )
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
    if sum(
        int(value)
        for value in (
            _is_d2ac(cfg), _is_d2ad(cfg), _is_d2ae(cfg), _is_d2af(cfg),
            _is_d2ag(cfg),
        )
    ) > 1:
        raise ValueError(
            "D2-AC, D2-AD, D2-AE, D2-AF and D2-AG modes are mutually exclusive"
        )
    if _is_d2ac(cfg):
        if architecture_variant != HOI_ARCHITECTURE_D2AC:
            raise ValueError("D2-AC requires the registered interaction-adapter architecture")
    elif _is_d2ad(cfg):
        if architecture_variant != HOI_ARCHITECTURE_D2AD:
            raise ValueError("D2-AD requires the registered local-frame adapter architecture")
    elif _is_d2ae(cfg):
        if architecture_variant != HOI_ARCHITECTURE_D2AE:
            raise ValueError("D2-AE requires the registered sparse-relation architecture")
    elif _is_d2af(cfg):
        if architecture_variant != HOI_ARCHITECTURE_D2AF:
            raise ValueError("D2-AF requires the registered reliability architecture")
    elif _is_d2ag(cfg):
        if architecture_variant != HOI_ARCHITECTURE_D2AG:
            raise ValueError(
                "D2-AG requires the registered selfcond-relation-source architecture"
            )
    elif architecture_variant != HOI_ARCHITECTURE_BASE:
        raise ValueError(
            "HOIPrior architecture variants are forbidden outside registered "
            "D2-AC/D2-AD/D2-AE/D2-AF/D2-AG"
        )


def _locked_loss_weights(cfg: DictConfig) -> Dict[str, float]:
    if (
        _is_d2u(cfg) or _is_d2v(cfg) or _is_d2x(cfg)
        or _is_d2y(cfg) or _is_d2z(cfg) or _is_d2ab(cfg)
        or _is_d2ac(cfg) or _is_d2ad(cfg) or _is_sparse_relation(cfg)
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
    if _is_d2af(cfg):
        contract.update({
            "d2af_sqrt_alpha_bar_reliability": True,
            "architecture_variant": HOI_ARCHITECTURE_D2AF,
            "placement": "after_motion_input_before_condition_concat_position_and_full_trunk",
            "writeback": (
                "motion + sqrt_alpha_bar[current_timestep] "
                "* tanh(alpha) * routed_relation"
            ),
            "global_bps_token_preserved": True,
            "temporal_anchors": [0, 5, 10, 15],
            "roles": ["left_hand", "right_hand", "pelvis"],
            "role_joints": [24, 26, 0],
            "rest_object_points": [100, 3],
            "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
            "manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
            "stacked_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
            "sparse_relation_parameters": SPARSE_RELATION_PARAMETER_COUNT,
            "schedule": diffusion_reliability_contract_metadata()["schedule"],
            "current_state_only": True,
            "current_timestep_only": True,
            "clean_target_used": False,
            "future_gt_used": False,
            "scene_used": False,
            "stored_relation_used": False,
            "loss_or_snr_weighting": False,
            "d2ab_predicted_support_no_slip": False,
        })
    if _is_d2ag(cfg):
        contract.update({
            "d2ag_selfcond_relation_source": True,
            "architecture_variant": HOI_ARCHITECTURE_D2AG,
            "placement": "after_motion_input_before_condition_concat_position_and_full_trunk",
            "writeback": "motion + tanh(alpha) * routed_relation(relation_source)",
            "global_bps_token_preserved": True,
            "temporal_anchors": [0, 5, 10, 15],
            "variable_anchors": list(D2AG_VARIABLE_ANCHORS),
            "variable_anchor_source": "detached_model_x0_hat",
            "history_anchor_source": "current_noisy_state",
            "selection_probability": D2AG_SELF_CONDITION_PROBABILITY,
            "unselected_sample_source": "current_noisy_state",
            "relation_zero_branch": False,
            "relation_exposure_fraction": 1.0,
            "roles": ["left_hand", "right_hand", "pelvis"],
            "role_joints": [24, 26, 0],
            "rest_object_points": [100, 3],
            "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
            "manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
            "stacked_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
            "sparse_relation_parameters": SPARSE_RELATION_PARAMETER_COUNT,
            "current_state_only": False,
            "clean_target_used": False,
            "future_gt_used": False,
            "scene_used": False,
            "stored_relation_used": False,
            "sqrt_alpha_bar_attenuation": False,
            "loss_or_snr_weighting": False,
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


def _validate_d2af_contract(
    cfg: DictConfig,
    world_size: int,
    *,
    require_eligibility_gate: bool = True,
    require_performance_gate: bool = True,
) -> Optional[Dict[str, object]]:
    if not _is_d2af(cfg):
        return None
    resume_value = cfg.resume_checkpoint
    is_resume = resume_value not in (None, "", False)
    run_id_contract = _validate_d2af_formal_run_id(
        str(cfg.run_id),
        require_actual_date=not is_resume,
    )
    resume_allowed = (
        resume_value in (None, "", False)
        or (
            Path(str(resume_value)).name.startswith(f"{cfg.run_id}_windows")
            and Path(str(resume_value)).suffix == ".pth"
        )
    )
    split_path = Path(str(cfg.split_manifest)).resolve()
    exact = {
        "mode": str(cfg.mode) == "d2af-sqrt-alpha-bar-reliability",
        "all_predecessor_modes_off": not any((
            _is_d2t(cfg), _is_d2u(cfg), _is_d2v(cfg), _is_d2x(cfg),
            _is_d2y(cfg), _is_d2z(cfg), _is_d2ab(cfg), _is_d2ac(cfg),
            _is_d2ad(cfg), _is_d2ae(cfg),
        )),
        "subphase": str(cfg.subphase) == "1B-D2-AF0",
        "run_id": run_id_contract["run_id"] == str(cfg.run_id),
        "run_id_date": is_resume or run_id_contract["date_is_actual"],
        "seed": int(cfg.seed) == 42,
        "architecture_variant": (
            str(cfg.get("hoi_architecture_variant")) == HOI_ARCHITECTURE_D2AF
        ),
        "model_config": _model_config(cfg) == {
            "dim_model": 512,
            "num_heads": 16,
            "num_layers": 8,
            "architecture_variant": HOI_ARCHITECTURE_D2AF,
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
        "locked_weights": {
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
        "old_auxiliary_inputs_absent": all(
            cfg.get(name) in (None, "", False)
            for name in (
                "d2z_gate_audit_path",
                "d2z_gate_audit_sha256",
                "d2ab_support_metadata_path",
                "d2ab_support_metadata_sha256",
            )
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
        "split_sha256": split_path.is_file() and _sha256(split_path)
        == "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e",
        "parameter_contract": (
            D2AE_BASE_PARAMETER_COUNT == 29673448
            and SPARSE_RELATION_PARAMETER_COUNT == 413953
            and D2AE_TOTAL_PARAMETER_COUNT == 30087401
            and D2AE_PARAMETER_INCREASE_FRACTION <= 0.015
        ),
        "asset_hashes": all(
            len(value) == 64
            for value in (
                SPARSE_POINT_MAPPING_SHA256,
                SPARSE_POINT_MANIFEST_SHA256,
                SPARSE_POINT_TENSOR_SHA256,
            )
        ),
        "schedule_contract": (
            diffusion_reliability_contract_metadata()["architecture_variant"]
            == HOI_ARCHITECTURE_D2AF
        ),
        "eligibility_gate_binding": (
            require_eligibility_gate
            or (
                cfg.get("d2af_clean_signal_eligibility_path") in (None, "", False)
                and cfg.get("d2af_clean_signal_eligibility_sha256")
                in (None, "", False)
            )
        ),
        "performance_gate_binding": (
            require_performance_gate
            or (
                cfg.get("d2af_performance_benchmark_path") in (None, "", False)
                and cfg.get("d2af_performance_benchmark_sha256")
                in (None, "", False)
                and cfg.get("d2af_performance_waiver_path")
                in (None, "", False)
                and cfg.get("d2af_performance_waiver_sha256")
                in (None, "", False)
            )
        ),
        "checkpoint_race_continuation_binding": (
            is_resume
            or (
                cfg.get("d2af_checkpoint_race_continuation_path")
                in (None, "", False)
                and cfg.get("d2af_checkpoint_race_continuation_sha256")
                in (None, "", False)
            )
        ),
        "profile_every_update": bool(cfg.get("profile_every_update")) is True,
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-AF reliability contract mismatch: {failed}")
    eligibility_gate = (
        _validate_d2af_eligibility_gate(cfg)
        if require_eligibility_gate and not require_performance_gate else None
    )
    performance_gate = (
        _validate_d2af_performance_gate(cfg)
        if require_performance_gate else None
    )
    return {
        "run_id": run_id_contract,
        "eligibility_gate_required": require_eligibility_gate,
        "performance_gate_required": require_performance_gate,
        "performance_gate": performance_gate,
        "eligibility_gate": (
            performance_gate["eligibility"]
            if performance_gate is not None else eligibility_gate
        ),
        "schedule": diffusion_reliability_contract_metadata()["schedule"],
    }


def _validate_d2ag_contract(
    cfg: DictConfig,
    world_size: int,
    *,
    require_performance_gate: bool = True,
) -> Optional[Dict[str, object]]:
    if not _is_d2ag(cfg):
        return None
    resume_value = cfg.resume_checkpoint
    is_resume = resume_value not in (None, "", False)
    run_id_contract = _validate_d2ag_formal_run_id(
        str(cfg.run_id),
        require_actual_date=not is_resume,
    )
    resume_allowed = (
        resume_value in (None, "", False)
        or (
            Path(str(resume_value)).name.startswith(f"{cfg.run_id}_windows")
            and Path(str(resume_value)).suffix == ".pth"
        )
    )
    split_path = Path(str(cfg.split_manifest)).resolve()
    exact = {
        "mode": str(cfg.mode) == "d2ag-selfcond-relation-source",
        "all_predecessor_modes_off": not any((
            _is_d2t(cfg), _is_d2u(cfg), _is_d2v(cfg), _is_d2x(cfg),
            _is_d2y(cfg), _is_d2z(cfg), _is_d2ab(cfg), _is_d2ac(cfg),
            _is_d2ad(cfg), _is_d2ae(cfg), _is_d2af(cfg),
        )),
        "subphase": str(cfg.subphase) == "1B-D2-AG0",
        "run_id": run_id_contract["run_id"] == str(cfg.run_id),
        "run_id_date": is_resume or run_id_contract["date_is_actual"],
        "seed": int(cfg.seed) == 42,
        "architecture_variant": (
            str(cfg.get("hoi_architecture_variant")) == HOI_ARCHITECTURE_D2AG
        ),
        "model_config": _model_config(cfg) == {
            "dim_model": 512,
            "num_heads": 16,
            "num_layers": 8,
            "architecture_variant": HOI_ARCHITECTURE_D2AG,
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
        "locked_weights": {
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
        "old_auxiliary_inputs_absent": all(
            cfg.get(name) in (None, "", False)
            for name in (
                "d2z_gate_audit_path",
                "d2z_gate_audit_sha256",
                "d2ab_support_metadata_path",
                "d2ab_support_metadata_sha256",
                "d2ae_performance_benchmark_path",
                "d2ae_performance_benchmark_sha256",
                "d2af_clean_signal_eligibility_path",
                "d2af_clean_signal_eligibility_sha256",
                "d2af_performance_benchmark_path",
                "d2af_performance_benchmark_sha256",
                "d2af_performance_waiver_path",
                "d2af_performance_waiver_sha256",
                "d2af_checkpoint_race_continuation_path",
                "d2af_checkpoint_race_continuation_sha256",
            )
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
        "split_sha256": split_path.is_file() and _sha256(split_path)
        == "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e",
        "parameter_contract": (
            D2AE_BASE_PARAMETER_COUNT == 29673448
            and SPARSE_RELATION_PARAMETER_COUNT == 413953
            and D2AE_TOTAL_PARAMETER_COUNT == 30087401
            and D2AE_PARAMETER_INCREASE_FRACTION <= 0.015
        ),
        "asset_hashes": all(
            len(value) == 64
            for value in (
                SPARSE_POINT_MAPPING_SHA256,
                SPARSE_POINT_MANIFEST_SHA256,
                SPARSE_POINT_TENSOR_SHA256,
            )
        ),
        "selfcond_contract": (
            selfcond_relation_source_contract_metadata()["architecture_variant"]
            == HOI_ARCHITECTURE_D2AG
        ),
        "selection_probability": D2AG_SELF_CONDITION_PROBABILITY == 0.5,
        "sqrt_alpha_bar_attenuation_off": (
            selfcond_relation_source_contract_metadata()[
                "sqrt_alpha_bar_attenuation"
            ] is False
        ),
        "relation_zero_branch_absent": (
            selfcond_relation_source_contract_metadata()[
                "relation_zero_branch"
            ] is False
        ),
        "throughput_gate_independent_of_d2af": (
            D2AG_MINIMUM_THROUGHPUT != D2AF_MINIMUM_THROUGHPUT
            and D2AG_MAXIMUM_ETA_HOURS != D2AF_MAXIMUM_ETA_HOURS
        ),
        "performance_gate_binding": (
            require_performance_gate
            or (
                cfg.get("d2ag_performance_benchmark_path") in (None, "", False)
                and cfg.get("d2ag_performance_benchmark_sha256")
                in (None, "", False)
                and cfg.get("d2ag_performance_waiver_path")
                in (None, "", False)
                and cfg.get("d2ag_performance_waiver_sha256")
                in (None, "", False)
            )
        ),
        "profile_every_update": bool(cfg.get("profile_every_update")) is True,
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(
            f"D2-AG selfcond relation source contract mismatch: {failed}"
        )
    performance_gate = (
        _validate_d2ag_performance_gate(cfg)
        if require_performance_gate else None
    )
    waiver = (
        None if performance_gate is None
        else performance_gate.get("performance_waiver")
    )
    return {
        "run_id": run_id_contract,
        "performance_gate_required": require_performance_gate,
        "performance_gate": performance_gate,
        "performance_gate_state": (
            None if performance_gate is None
            else str(performance_gate["status"])
        ),
        "performance_gate_classification": (
            None if performance_gate is None
            else str(performance_gate["classification"])
        ),
        "performance_waiver_path": (
            None if waiver is None else str(waiver["path"])
        ),
        "performance_waiver_sha256": (
            None if waiver is None else str(waiver["sha256"])
        ),
        "performance_waiver_run_id": (
            None if waiver is None else str(waiver["run_id"])
        ),
        "d2af_waiver_inherited": False,
        "selection_probability": D2AG_SELF_CONDITION_PROBABILITY,
        "minimum_throughput_windows_per_second": D2AG_MINIMUM_THROUGHPUT,
        "maximum_full_budget_eta_hours": D2AG_MAXIMUM_ETA_HOURS,
        "contract": selfcond_relation_source_contract_metadata(),
    }


def _validate_author_update_execution_host(cfg: DictConfig) -> None:
    if not _uses_author_update_rule(cfg):
        return
    modes = (
        (_is_d2ag(cfg), "D2-AG"),
        (_is_d2af(cfg), "D2-AF"),
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
        raise ValueError("sparse-relation gradient audit requires the relation field")

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
        raise FloatingPointError("sparse-relation alpha gradient must be finite and nonzero")
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
            f"activated sparse-relation gradients must be finite/nonzero: {failed}"
        )
    if result["gate_value"] == 0.0 and field._gate_override is None:
        raise FloatingPointError("sparse-relation gate did not activate")
    return result


def _validate_d2ae_model_instance(
    model: torch.nn.Module,
    *,
    require_zero_alpha: bool,
    expected_diffusion_reliability: bool = False,
    expected_selfcond_relation_source: bool = False,
) -> Dict[str, object]:
    field = model.network.sparse_relation_field
    if field is None:
        raise ValueError("model instance is missing the sparse relation field")
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
        "rho_override_absent": field._rho_override is None,
        "diffusion_reliability": (
            field.diffusion_reliability is expected_diffusion_reliability
        ),
        "selfcond_relation_source": (
            field.selfcond_relation_source is expected_selfcond_relation_source
        ),
        "schedule_buffer": (
            (
                field.sqrt_alpha_bar is not None
                and field.sqrt_alpha_bar.shape == (500,)
                and field.sqrt_alpha_bar.dtype == torch.float32
            )
            if expected_diffusion_reliability
            else field.sqrt_alpha_bar is None
        ),
        "relation_source_estimate_cleared": (
            field.relation_source_estimate is False
        ),
        "capture_disabled": field._capture is False,
        "alpha_finite": bool(torch.isfinite(field.alpha.detach())),
        "alpha_zero_when_required": (
            not require_zero_alpha or float(field.alpha.detach().item()) == 0.0
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"sparse-relation model instance contract mismatch: {failed}")
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


def _d2ag_mask_seed(cfg: DictConfig, processed_windows: int, rank: int) -> int:
    """Registered D2-AG Bernoulli seed (EP:6955-6968).

    Structurally identical to the validation derivation at ``_validate``, so the
    mask is reproducible from ``(seed, processed_windows, rank)`` alone and needs
    no extra checkpoint state for a bit-exact resume.
    """
    return int(cfg.seed) * 1_000_003 + int(processed_windows) + int(rank)


def _d2ag_selection_mask(
    cfg: DictConfig,
    batch_size: int,
    device: torch.device,
    processed_windows: int,
    rank: int,
) -> torch.Tensor:
    """Draw the per-sample D2-AG mask from a dedicated generator.

    The generator is deliberately independent of the global stream: formal
    training draws ``timesteps``/``noise`` from the global RNG, so a shared
    stream would make global RNG consumption depend on ``mask.sum()``.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(_d2ag_mask_seed(cfg, processed_windows, rank))
    return torch.rand(
        (int(batch_size),), device=device, generator=generator,
    ) < D2AG_SELF_CONDITION_PROBABILITY


def _d2ag_relation_source_arguments(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    relation_metadata: Mapping[str, torch.Tensor],
    cfg: DictConfig,
    *,
    processed_windows: int,
    rank: int,
    observer=None,
) -> Dict[str, torch.Tensor]:
    """Build the D2-AG per-sample relation source for one training micro-batch.

    Selected rows take the detached ``x0_hat`` of one extra ``torch.no_grad()``
    forward at the same timestep; that estimate forward itself uses the current
    D2-AE ``x_t`` source.  Unselected rows keep ``x_t`` verbatim, so the field is
    always active and there is no relation-zero branch.
    """
    mask = _d2ag_selection_mask(
        cfg, noisy.shape[0], noisy.device, processed_windows, rank,
    )
    index = mask.nonzero(as_tuple=True)[0]
    # Reach the unwrapped module so the estimate forward never touches DDP
    # buffer sync or ``require_forward_param_sync``; ``no_sync()`` is already
    # owned by gradient accumulation and must not be overloaded here.
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    estimate: Optional[torch.Tensor] = None
    if int(index.numel()):
        selected = {
            key: value.index_select(0, index)
            for key, value in (
                ("noisy", noisy),
                ("timesteps", timesteps),
                ("text_embedding", batch["text_embedding"]),
                ("object_bps", batch["object_bps"]),
                ("goals", batch["goals"]),
                ("progress", normalize_progress(batch["progress"])),
                ("rest_object_points", relation_metadata["rest_object_points"]),
                (
                    "world_to_local_rotation",
                    relation_metadata["world_to_local_rotation"],
                ),
                (
                    "object_rotation_reference",
                    relation_metadata["object_rotation_reference"],
                ),
            )
        }
        was_training = inner.training
        # eval() disables the active trunk dropout, so the estimate consumes no
        # global RNG and matches the sampler's execution mode.
        inner.eval()
        if observer is not None:
            observer("begin", int(index.numel()))
        try:
            with torch.no_grad():
                estimate = inner(
                    selected["noisy"],
                    selected["timesteps"],
                    selected["text_embedding"],
                    selected["object_bps"],
                    selected["goals"],
                    selected["progress"],
                    rest_object_points=selected["rest_object_points"],
                    world_to_local_rotation=selected["world_to_local_rotation"],
                    object_rotation_reference=selected[
                        "object_rotation_reference"
                    ],
                    position_minimum=relation_metadata["position_minimum"],
                    position_maximum=relation_metadata["position_maximum"],
                    object_minimum=relation_metadata["object_minimum"],
                    object_maximum=relation_metadata["object_maximum"],
                    relation_source_estimate=True,
                )
        finally:
            inner.train(was_training)
            if observer is not None:
                observer("end", int(index.numel()))
    return {
        "relation_source": build_d2ag_relation_source(
            noisy, estimate, index=None if estimate is None else index,
        ),
    }


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
    processed_windows: int = 0,
    rank: int = 0,
    selfcond_observer=None,
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
    relation_metadata = {
        "rest_object_points": batch["rest_object_points"],
        "world_to_local_rotation": batch["world_to_local_rotation"],
        "object_rotation_reference": batch["object_rotation_reference"],
        "position_minimum": minimum,
        "position_maximum": maximum,
        "object_minimum": object_minimum,
        "object_maximum": object_maximum,
    }
    selfcond_arguments: Dict[str, torch.Tensor] = {}
    if _is_d2ag(cfg):
        selfcond_arguments = _d2ag_relation_source_arguments(
            model,
            batch,
            noisy,
            timesteps,
            relation_metadata,
            cfg,
            processed_windows=processed_windows,
            rank=rank,
            observer=selfcond_observer,
        )
    if _is_d2ad(cfg):
        prediction = model(
            *model_arguments,
            local_object_bps=batch["local_object_bps"],
        )
    elif _is_sparse_relation(cfg):
        prediction = model(
            *model_arguments,
            **relation_metadata,
            **selfcond_arguments,
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
                    processed_windows=processed_windows, rank=rank,
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
    if _is_d2af(cfg):
        value["architecture_variant"] = HOI_ARCHITECTURE_D2AF
        value["diffusion_reliability_contract"] = (
            model.module.network.sparse_relation_field.contract_metadata()
        )
    if _is_d2ag(cfg):
        value["architecture_variant"] = HOI_ARCHITECTURE_D2AG
        value["selfcond_relation_source_contract"] = (
            model.module.network.sparse_relation_field.contract_metadata()
        )
    if ema_models:
        # Retain the legacy name for pre-D2-T official evaluator compatibility.
        value["ema_model"] = ema_models["0.9999"].state_dict()
    return value


def _checkpoint_collision_preflight(
    rank: int,
    *,
    checkpoint_path: Path,
    rng_path: Path,
    device: torch.device,
) -> None:
    """Synchronize rank-local checkpoint collision checks before any write."""
    local_collisions = []
    if rng_path.exists():
        local_collisions.append(rng_path)
    if rank == 0 and checkpoint_path.exists():
        local_collisions.append(checkpoint_path)
    collision_flag = torch.tensor(
        [1 if local_collisions else 0],
        dtype=torch.int32,
        device=device,
    )
    torch.distributed.all_reduce(
        collision_flag,
        op=torch.distributed.ReduceOp.MAX,
    )
    if int(collision_flag.item()) != 0:
        detail = (
            ", ".join(str(path) for path in local_collisions)
            if local_collisions
            else "another rank detected an existing checkpoint artifact"
        )
        raise FileExistsError(
            "refusing to overwrite checkpoint or rank-local RNG sidecar: "
            + detail
        )
    # The collective reports a global pass; this explicit barrier keeps every
    # rank behind the preflight until no rank can still be checking peer-visible
    # filesystem state.
    torch.distributed.barrier()


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
    device = next(model.module.parameters()).device
    _checkpoint_collision_preflight(
        rank,
        checkpoint_path=checkpoint_path,
        rng_path=rng_path,
        device=device,
    )
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
    *,
    expected_architecture_variant: str = HOI_ARCHITECTURE_D2AE,
) -> Dict[str, object]:
    """Reject a sparse-relation resume without exact random-origin provenance."""
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
        == expected_architecture_variant,
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
            "sparse-relation resume checkpoint random-origin provenance mismatch: "
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
    elif _is_d2af(cfg):
        _validate_d2ae_random_origin_checkpoint(
            checkpoint,
            _state_dict_sha256(model.module.state_dict()),
            expected_architecture_variant=HOI_ARCHITECTURE_D2AF,
        )
        if checkpoint.get("selfcond_relation_source_contract") is not None:
            raise ValueError(
                "D2-AF resume checkpoint architecture/provenance mismatch"
            )
        try:
            validate_diffusion_reliability_contract(
                checkpoint.get("diffusion_reliability_contract")
            )
        except ValueError as error:
            raise ValueError(
                "D2-AF resume checkpoint architecture/provenance mismatch"
            ) from error
    elif _is_d2ag(cfg):
        _validate_d2ae_random_origin_checkpoint(
            checkpoint,
            _state_dict_sha256(model.module.state_dict()),
            expected_architecture_variant=HOI_ARCHITECTURE_D2AG,
        )
        if (
            checkpoint.get("sparse_relation_contract") is not None
            or checkpoint.get("diffusion_reliability_contract") is not None
        ):
            raise ValueError(
                "D2-AG resume checkpoint architecture/provenance mismatch"
            )
        try:
            validate_selfcond_relation_source_contract(
                checkpoint.get("selfcond_relation_source_contract")
            )
        except ValueError as error:
            raise ValueError(
                "D2-AG resume checkpoint architecture/provenance mismatch"
            ) from error
    elif checkpoint.get("architecture_variant") in {
        HOI_ARCHITECTURE_D2AC,
        HOI_ARCHITECTURE_D2AD,
        HOI_ARCHITECTURE_D2AE,
        HOI_ARCHITECTURE_D2AF,
        HOI_ARCHITECTURE_D2AG,
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
    d2af_lifecycle_contract = _validate_d2af_contract(cfg, world_size)
    d2ag_lifecycle_contract = _validate_d2ag_contract(cfg, world_size)
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
    sparse_relation_model_contract = (
        _validate_d2ae_model_instance(
            model.module,
            require_zero_alpha=resumed_from is None,
            expected_diffusion_reliability=_is_d2af(cfg),
            expected_selfcond_relation_source=_is_d2ag(cfg),
        )
        if _is_sparse_relation(cfg) else None
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
    if _is_sparse_relation(cfg) and sparse_relation_gradient_audit_path.is_file():
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
                        processed_windows=processed_windows, rank=rank,
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
            if _is_sparse_relation(cfg):
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

    final_sparse_relation_model_contract = None
    if _is_sparse_relation(cfg):
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
                "sparse-relation training ended without the locked gradient audit: "
                + ", ".join(missing_audit)
            )
        final_sparse_relation_model_contract = _validate_d2ae_model_instance(
            model.module,
            require_zero_alpha=False,
            expected_diffusion_reliability=_is_d2af(cfg),
            expected_selfcond_relation_source=_is_d2ag(cfg),
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
                    sparse_relation_gradient_audit
                    if _is_sparse_relation(cfg) else None
                ),
                "d2ae_lifecycle_contract": (
                    d2ae_lifecycle_contract if _is_d2ae(cfg) else None
                ),
                "d2af_lifecycle_contract": (
                    d2af_lifecycle_contract if _is_d2af(cfg) else None
                ),
                "d2ag_lifecycle_contract": (
                    d2ag_lifecycle_contract if _is_d2ag(cfg) else None
                ),
                "sparse_relation_field": (
                    {
                        "architecture_variant": (
                            HOI_ARCHITECTURE_D2AG
                            if _is_d2ag(cfg)
                            else HOI_ARCHITECTURE_D2AF
                            if _is_d2af(cfg) else HOI_ARCHITECTURE_D2AE
                        ),
                        "alpha": float(
                            model.module.network.sparse_relation_field.alpha.detach().item()
                        ),
                        "gate": float(torch.tanh(
                            model.module.network.sparse_relation_field.alpha.detach()
                        ).item()),
                        "contract": (
                            model.module.network.sparse_relation_field.contract_metadata()
                        ),
                        "initial_model_instance_contract": (
                            sparse_relation_model_contract
                        ),
                        "final_model_instance_contract": (
                            final_sparse_relation_model_contract
                        ),
                        "diagnostic_variant": (
                            model.module.network.sparse_relation_field._diagnostic_variant
                        ),
                        "gate_override": (
                            model.module.network.sparse_relation_field._gate_override
                        ),
                        "rho_override": (
                            model.module.network.sparse_relation_field._rho_override
                        ),
                        "diffusion_reliability": (
                            model.module.network.sparse_relation_field.diffusion_reliability
                        ),
                        "selfcond_relation_source": (
                            model.module.network.sparse_relation_field.selfcond_relation_source
                        ),
                        "selection_probability": (
                            D2AG_SELF_CONDITION_PROBABILITY
                            if _is_d2ag(cfg) else None
                        ),
                        "mask_generator_seed_formula": (
                            "cfg.seed * 1000003 + processed_windows + rank"
                            if _is_d2ag(cfg) else None
                        ),
                        "capture_enabled": (
                            model.module.network.sparse_relation_field._capture
                        ),
                        "builder": {
                            "backend": "pure_pytorch",
                            "runtime_device": "gpu",
                            "source": (
                                "variable_anchors_detached_model_x0_hat_history_current_x_t"
                                if _is_d2ag(cfg)
                                else "current_diffusion_state_x_t"
                            ),
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
                    if _is_sparse_relation(cfg) else None
                ),
                "terminal_model_state_sha256": (
                    _state_dict_sha256(model.module.state_dict())
                    if _is_sparse_relation(cfg) else None
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
    _validate_d2af_contract(cfg, int(cfg.num_gpus))
    _validate_author_update_execution_host(cfg)
    if not torch.cuda.is_available() or torch.cuda.device_count() < int(cfg.num_gpus):
        raise RuntimeError(f"requires {cfg.num_gpus} visible CUDA devices")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    torch.multiprocessing.spawn(_worker, args=(cfg,), nprocs=int(cfg.num_gpus), join=True)


if __name__ == "__main__":
    main()
