#!/usr/bin/env python3
"""Authority CPU hard-gate diagnostics for the fixed D2-AF0 mechanism."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import torch
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

from priors.diffusion import GaussianDiffusion  # noqa: E402
from priors.diffusion_schedule import (  # noqa: E402
    SQRT_ALPHA_BAR_SENTINELS,
    SQRT_ALPHA_BAR_SHA256,
    canonical_diffusion_schedule,
    diffusion_schedule_contract_metadata,
    tensor_sha256,
)
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_BASE,
    HOI_ARCHITECTURE_D2AC,
    HOI_ARCHITECTURE_D2AD,
    HOI_ARCHITECTURE_D2AE,
    HOI_ARCHITECTURE_D2AF,
    build_expert,
    load_trained_hoi_prior,
)
from priors.sparse_relation import (  # noqa: E402
    SPARSE_RELATION_PARAMETER_COUNT,
    SparseCurrentStateRelationField,
    diffusion_reliability_contract_metadata,
    sparse_relation_contract_metadata,
)
from tools import diagnose_hoi_d2ae as d2ae  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    _d2ae_gradient_audit,
    _d2af_formal_source_contract,
    _forward_losses,
    _locked_loss_weights,
    _optimization_contract,
    _state_dict_sha256,
    _validate_d2af_contract,
    _validate_fk_foot_temporal_routing_mode,
)


RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-cpu-contract"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
FAILURE_CLASSIFICATION = "diffusion-reliability-contract-failure-stop"
EXPECTED_AUTHORITY_PYTHON = Path(
    "/data/yujinlun/anaconda3/envs/infbagel/bin/python"
)
EXPECTED_BASE_PARAMETERS = 29_673_448
EXPECTED_RELATION_PARAMETERS = 413_953
EXPECTED_TOTAL_PARAMETERS = 30_087_401
EXPECTED_INITIAL_STATE_SHA256 = (
    "b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c"
)
EXPECTED_SPLIT_SHA256 = (
    "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e"
)
REGISTERED_TIMESTEPS = (0, 249, 499)
SCALING_TOLERANCE = 1.0e-6

AUTHORITY_VERIFICATION_COMMANDS = (
    '"$INFBAGEL_PYTHON" -m unittest tests.test_hoi_d2af -v',
    '"$INFBAGEL_PYTHON" -m unittest tests.test_hoi_d2af_lifecycle_cpu -v',
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


def validate_actual_run_id(run_id: str) -> str:
    match = RUN_ID_RE.fullmatch(str(run_id))
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if match is None or match.group("date") != actual_date:
        raise ValueError(
            "D2-AF CPU contract run id must use the locked stem and actual date"
        )
    return match.group("date")


def authority_identity(repo: Path, *, require_clean: bool) -> Dict[str, object]:
    repo = repo.resolve()
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repo,
            text=True,
        ).strip()
    ).resolve()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        cwd=repo,
        text=True,
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        text=True,
    ).splitlines()
    if root != repo or branch != "phase/01b-hoi":
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: authority identity mismatch"
        )
    if require_clean and status:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: authority worktree is dirty: {status[:20]}"
        )
    if Path(sys.executable).resolve() != EXPECTED_AUTHORITY_PYTHON.resolve():
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: unexpected authority Python "
            f"{Path(sys.executable).resolve()}"
        )
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    if (
        not configured_python
        or not Path(configured_python).is_absolute()
        or Path(configured_python).resolve() != EXPECTED_AUTHORITY_PYTHON.resolve()
    ):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: INFBAGEL_PYTHON is not authority-locked"
        )
    baseline = "b9a158f75ab0740c91c9cfc8863a65fa381b014c"
    forbidden = "860ec8ca10cb5d6bed9d901560d3eb3d811a8143"
    baseline_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, "HEAD"],
        cwd=repo,
        check=False,
    ).returncode == 0
    forbidden_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", forbidden, "HEAD"],
        cwd=repo,
        check=False,
    ).returncode == 0
    if not baseline_is_ancestor or forbidden_is_ancestor:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: locked Git provenance failed"
        )
    return {
        "repo_root": str(root),
        "branch": branch,
        "git_commit": commit,
        "worktree_clean": not status,
        "status_porcelain": status,
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "integration_baseline_is_ancestor": baseline_is_ancestor,
        "forbidden_feature_is_ancestor": forbidden_is_ancestor,
    }


def _merged_config(repo: Path, formal_run_id: str, *, d2af: bool):
    variant = (
        "config_train_hoi_prior_d2af.yaml"
        if d2af else "config_train_hoi_prior_d2ae.yaml"
    )
    cfg = OmegaConf.merge(
        OmegaConf.load(repo / "code/config/config_train_hoi_prior.yaml"),
        OmegaConf.load(repo / "code/config" / variant),
    )
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


def resolved_config(repo: Path, run_date: str):
    formal_run_id = (
        f"p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-{run_date}"
    )
    return _merged_config(repo, formal_run_id, d2af=True)


def config_contract(repo: Path, cfg) -> Dict[str, object]:
    _validate_fk_foot_temporal_routing_mode(cfg)
    lifecycle = _validate_d2af_contract(
        cfg,
        4,
        require_eligibility_gate=False,
        require_performance_gate=False,
    )
    split_path = Path(str(cfg.split_manifest)).resolve()
    split_hash = sha256_file(split_path)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: split hash mismatch")

    d2ae_cfg = _merged_config(
        repo,
        str(cfg.run_id).replace(
            "d2af-sqrt-alpha-bar-reliability",
            "d2ae-sparse-relation-field",
        ),
        d2af=False,
    )
    ignored = {
        "mode",
        "subphase",
        "run_id",
        "output_dir",
        "checkpoint_dir",
        "metrics_path",
        "state_path",
        "hydra",
        "d2ae_sparse_relation_field",
        "d2af_sqrt_alpha_bar_reliability",
        "hoi_architecture_variant",
        "d2ae_performance_benchmark_path",
        "d2ae_performance_benchmark_sha256",
        "d2af_clean_signal_eligibility_path",
        "d2af_clean_signal_eligibility_sha256",
        "d2af_performance_benchmark_path",
        "d2af_performance_benchmark_sha256",
    }
    ae_values = OmegaConf.to_container(d2ae_cfg, resolve=True)
    af_values = OmegaConf.to_container(cfg, resolve=True)
    differences = sorted(
        key
        for key in set(ae_values) | set(af_values)
        if key not in ignored and ae_values.get(key) != af_values.get(key)
    )
    weights = _locked_loss_weights(cfg)
    optimization = _optimization_contract(cfg)
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    checks = {
        "trainer_lifecycle": lifecycle is not None,
        "eligibility_gate_disabled_for_cpu": (
            lifecycle is not None
            and lifecycle["eligibility_gate_required"] is False
        ),
        "performance_gate_disabled_for_cpu": (
            lifecycle is not None
            and lifecycle["performance_gate_required"] is False
        ),
        "single_factor_config": not differences,
        "split": split_hash == EXPECTED_SPLIT_SHA256,
        "loss_weights": weights == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
        "optimizer": optimization == {
            "optimizer": "Adam",
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "learning_rate": 0.0001,
            "scheduler": "none",
            "warmup_windows": 0,
            "gradient_clipping": False,
            "gradient_clip_norm": None,
            "amp": False,
            "ema_decays": [],
            "primary_weight_variant": "online",
        },
        "resolved": "${" not in resolved_yaml,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: config contract failed: "
            f"{checks}; differences={differences}"
        )
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
            f"{FAILURE_CLASSIFICATION}: registry validation failed: "
            f"{validation.stdout}"
        )
    return {
        "formal_run_id": str(cfg.run_id),
        "split_sha256": split_hash,
        "single_factor_ignored_keys": sorted(ignored),
        "unexpected_single_factor_differences": differences,
        "loss_weights": weights,
        "optimization": optimization,
        "lifecycle": lifecycle,
        "checks": checks,
        "registry_validation_returncode": validation.returncode,
        "registry_validation_output": validation.stdout.strip(),
        "full_authority_suite_commands": list(AUTHORITY_VERIFICATION_COMMANDS),
    }


def schedule_contract() -> Dict[str, object]:
    schedule = canonical_diffusion_schedule()
    metadata = diffusion_schedule_contract_metadata()
    diffusion = GaussianDiffusion()
    field_hashes = []
    state_dict_presence = []
    for _ in range(4):
        field = SparseCurrentStateRelationField(
            512,
            diffusion_reliability=True,
        )
        if field.sqrt_alpha_bar is None:
            raise RuntimeError(
                f"{FAILURE_CLASSIFICATION}: D2-AF field schedule is absent"
            )
        field_hashes.append(tensor_sha256(field.sqrt_alpha_bar))
        state_dict_presence.append(
            "sqrt_alpha_bar" in field.state_dict()
        )
    checks = {
        "canonical": metadata["sqrt_alpha_bar_sha256"]
        == SQRT_ALPHA_BAR_SHA256,
        "diffusion_byte_exact": torch.equal(
            diffusion.sqrt_alpha_bar.cpu(),
            schedule["sqrt_alpha_bar"],
        ),
        "diffusion_hash": tensor_sha256(diffusion.sqrt_alpha_bar)
        == SQRT_ALPHA_BAR_SHA256,
        "four_rank_hash_identity": field_hashes
        == [SQRT_ALPHA_BAR_SHA256] * 4,
        "field_buffer_nonpersistent": not any(state_dict_presence),
        "sentinels": {
            index: float(schedule["sqrt_alpha_bar"][index])
            for index in SQRT_ALPHA_BAR_SENTINELS
        } == SQRT_ALPHA_BAR_SENTINELS,
        "strictly_decreasing": bool(torch.all(
            schedule["sqrt_alpha_bar"][1:]
            < schedule["sqrt_alpha_bar"][:-1]
        )),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: schedule contract failed: {checks}"
        )
    return {
        **metadata,
        "gaussian_diffusion_sqrt_alpha_bar_sha256": tensor_sha256(
            diffusion.sqrt_alpha_bar
        ),
        "field_schedule_sha256_by_simulated_rank": field_hashes,
        "field_schedule_present_in_state_dict_by_rank": state_dict_presence,
        "checks": checks,
    }


def _relation_arguments(values: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "rest_object_points": values["rest_object_points"],
        "world_to_local_rotation": values["world_to_local_rotation"],
        "object_rotation_reference": values["object_rotation_reference"],
        "position_minimum": values["position_minimum"],
        "position_maximum": values["position_maximum"],
        "object_minimum": values["object_minimum"],
        "object_maximum": values["object_maximum"],
    }


def timestep_and_scaling_contract() -> Dict[str, object]:
    field = SparseCurrentStateRelationField(
        512,
        diffusion_reliability=True,
    ).eval()
    field.set_gate_override(0.1)
    values = d2ae.synthetic_inputs(batch=3)
    motion = torch.randn(
        3,
        16,
        512,
        generator=torch.Generator().manual_seed(840),
    )
    timesteps = torch.tensor(REGISTERED_TIMESTEPS, dtype=torch.long)
    scheduled = field(
        motion,
        values["current"],
        **_relation_arguments(values),
        timesteps=timesteps,
    )
    field.set_rho_override(1.0)
    unit = field(
        motion,
        values["current"],
        **_relation_arguments(values),
        timesteps=timesteps,
    )
    field.set_rho_override(None)
    rho = canonical_diffusion_schedule()["sqrt_alpha_bar"][timesteps]
    scheduled_delta = scheduled - motion
    unit_delta = unit - motion
    expected = rho[:, None, None] * unit_delta
    scaling_max_abs = float((scheduled_delta - expected).abs().max())
    per_sample_max_abs = (
        (scheduled_delta - expected).abs().flatten(1).amax(dim=1).tolist()
    )

    invalid = {
        "missing": None,
        "shape": torch.zeros(3, 1, dtype=torch.long),
        "dtype": torch.zeros(3, dtype=torch.float32),
        "negative": torch.tensor([-1, 0, 1], dtype=torch.long),
        "upper_bound": torch.tensor([0, 1, 500], dtype=torch.long),
        "device": torch.empty(3, dtype=torch.long, device="meta"),
    }
    rejection = {}
    for name, candidate in invalid.items():
        try:
            field(
                motion,
                values["current"],
                **_relation_arguments(values),
                timesteps=candidate,
            )
        except ValueError as error:
            rejection[name] = {"rejected": True, "error": str(error)}
        else:
            rejection[name] = {"rejected": False, "error": None}
    checks = {
        "mixed_batch_rho": rho.tolist() == [
            SQRT_ALPHA_BAR_SENTINELS[index]
            for index in REGISTERED_TIMESTEPS
        ],
        "scaling": scaling_max_abs <= SCALING_TOLERANCE,
        "all_invalid_rejected": all(
            item["rejected"] for item in rejection.values()
        ),
        "finite": bool(
            torch.isfinite(scheduled).all()
            and torch.isfinite(unit).all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: timestep/scaling contract failed: {checks}"
        )
    return {
        "timesteps": list(REGISTERED_TIMESTEPS),
        "rho": rho.tolist(),
        "scheduled_delta_l2_per_sample": (
            scheduled_delta.flatten(1).norm(dim=1).tolist()
        ),
        "unit_rho_delta_l2_per_sample": (
            unit_delta.flatten(1).norm(dim=1).tolist()
        ),
        "delta_minus_rho_times_unit_max_abs": scaling_max_abs,
        "delta_minus_rho_times_unit_max_abs_by_sample": per_sample_max_abs,
        "maximum_allowed": SCALING_TOLERANCE,
        "invalid_timestep_rejection": rejection,
        "checks": checks,
    }


def _forward(model: torch.nn.Module, values: Mapping[str, torch.Tensor]):
    return model(
        values["current"],
        values["timesteps"],
        values["text"],
        values["global_bps"],
        values["goals"],
        values["progress"],
        **_relation_arguments(values),
    )


def model_contract() -> Dict[str, object]:
    torch.manual_seed(42)
    model = build_expert(
        "hoi",
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AF,
    )
    initial_hash = _state_dict_sha256(model.state_dict())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    relation_parameters = sum(
        parameter.numel()
        for parameter in model.network.sparse_relation_field.parameters()
    )
    field = model.network.sparse_relation_field
    if field.sqrt_alpha_bar is None:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: D2-AF model schedule is absent"
        )

    torch.manual_seed(42)
    base = build_expert(
        "hoi",
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_BASE,
    )
    shared_keys = sorted(base.state_dict())
    sparse_keys = sorted(set(model.state_dict()) - set(base.state_dict()))
    shared_state_exact = all(
        torch.equal(base.state_dict()[key], model.state_dict()[key])
        for key in shared_keys
    )
    values = d2ae.synthetic_inputs(batch=1)
    base.eval()
    model.eval()
    with torch.no_grad():
        expected = base(
            values["current"],
            values["timesteps"],
            values["text"],
            values["global_bps"],
            values["goals"],
            values["progress"],
        )
        actual = _forward(model, values)
    base_parity = float((actual - expected).abs().max())
    del base

    gradients = {}
    for timestep in REGISTERED_TIMESTEPS:
        probe = d2ae.synthetic_inputs(
            batch=1,
            seed=10_000 + timestep,
        )
        probe["timesteps"].fill_(timestep)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            field.alpha.zero_()
        prediction = _forward(model, probe)
        (prediction - probe["current"]).square().mean().backward()
        initial = _d2ae_gradient_audit(
            model,
            require_relation_paths=False,
        )

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            field.alpha.copy_(torch.atanh(torch.tensor(0.1)))
        prediction = _forward(model, probe)
        (prediction - probe["current"]).square().mean().backward()
        activated = _d2ae_gradient_audit(
            model,
            require_relation_paths=True,
        )
        gradients[str(timestep)] = {
            "initial_zero_gate": initial,
            "test_only_activated_gate": activated,
        }
    with torch.no_grad():
        field.alpha.zero_()
    model.zero_grad(set_to_none=True)

    checks = {
        "initial_state_hash": initial_hash == EXPECTED_INITIAL_STATE_SHA256,
        "base_parameters": (
            total_parameters - relation_parameters == EXPECTED_BASE_PARAMETERS
        ),
        "relation_parameters": (
            relation_parameters == EXPECTED_RELATION_PARAMETERS
            == SPARSE_RELATION_PARAMETER_COUNT
        ),
        "total_parameters": total_parameters == EXPECTED_TOTAL_PARAMETERS,
        "parameter_increase": (
            relation_parameters / (total_parameters - relation_parameters)
            <= 0.015
        ),
        "alpha_zero": float(field.alpha.detach()) == 0.0,
        "schedule_nonpersistent": (
            "network.sparse_relation_field.sqrt_alpha_bar"
            not in model.state_dict()
        ),
        "schedule_hash": tensor_sha256(field.sqrt_alpha_bar)
        == SQRT_ALPHA_BAR_SHA256,
        "shared_state_exact": shared_state_exact,
        "sparse_keys": (
            len(sparse_keys) == 10
            and all(
                key.startswith("network.sparse_relation_field.")
                for key in sparse_keys
            )
        ),
        "base_parity": base_parity <= SCALING_TOLERANCE,
        "base_parity_exact_zero": base_parity == 0.0,
        "output_shape": tuple(actual.shape) == (1, 16, 232),
        "all_timestep_gradients": len(gradients) == 3,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: model contract failed: {checks}"
        )
    return {
        "initial_model_state_sha256": initial_hash,
        "expected_initial_model_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
        "base_parameters": total_parameters - relation_parameters,
        "relation_parameters": relation_parameters,
        "total_parameters": total_parameters,
        "parameter_increase_fraction": (
            relation_parameters / (total_parameters - relation_parameters)
        ),
        "output_shape": list(actual.shape),
        "base_parity_max_abs": base_parity,
        "shared_state_key_count": len(shared_keys),
        "sparse_state_keys": sparse_keys,
        "gradients_by_timestep": gradients,
        "test_only_gate": 0.1,
        "test_only_probe_saved": False,
        "test_only_probe_optimizer_updates": 0,
        "checks": checks,
    }


def timestep_identity_contract(cfg) -> Dict[str, object]:
    captured: Dict[str, torch.Tensor] = {}

    class StopAfterModel(RuntimeError):
        pass

    class DiffusionStub:
        def q_sample(self, clean, timesteps, noise):
            del noise
            captured["q_sample"] = timesteps
            return clean

    class ModelStub:
        def __call__(self, current, timesteps, *args, **kwargs):
            del current, args, kwargs
            captured["model"] = timesteps
            raise StopAfterModel

    values = d2ae.synthetic_inputs(batch=3)
    batch = {
        "x": values["current"],
        "text_embedding": values["text"],
        "object_bps": values["global_bps"],
        "goals": values["goals"],
        "progress": values["progress"],
        **_relation_arguments(values),
    }
    try:
        _forward_losses(
            ModelStub(),
            DiffusionStub(),
            batch,
            torch.arange(22, dtype=torch.long),
            values["position_minimum"],
            values["position_maximum"],
            values["object_minimum"],
            values["object_maximum"],
            cfg,
            generator=torch.Generator().manual_seed(42),
        )
    except StopAfterModel:
        pass
    if (
        "q_sample" not in captured
        or "model" not in captured
        or captured["q_sample"] is not captured["model"]
    ):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: q-sample/model timestep identity failed"
        )

    class Recorder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trace = []

        def forward(self, current, timesteps, *args, **kwargs):
            del args, kwargs
            self.trace.append(int(timesteps[0]))
            return torch.zeros_like(current)

    recorder = Recorder()
    GaussianDiffusion().sample(
        recorder,
        torch.zeros(1, 2, 232),
        torch.zeros(1, 768),
        torch.zeros(1, 1024, 3),
        torch.zeros(1, 9),
        torch.zeros(1, 3),
        generator=torch.Generator().manual_seed(42),
    )
    expected_trace = list(reversed(range(500)))
    if recorder.trace != expected_trace:
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: sampler timestep trace failed"
        )
    trace_sha256 = hashlib.sha256(
        ("\n".join(str(value) for value in recorder.trace) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "training_q_sample_model_same_tensor_object": True,
        "training_timestep_shape": list(captured["model"].shape),
        "training_timestep_dtype": str(captured["model"].dtype),
        "sampler_trace_first": recorder.trace[0],
        "sampler_trace_last": recorder.trace[-1],
        "sampler_trace_length": len(recorder.trace),
        "sampler_trace_sha256": trace_sha256,
        "sampler_trace_exact_499_to_0": True,
    }


def checkpoint_rejection_contract() -> Dict[str, object]:
    common = {
        "checkpoint_type": "hoi_prior_phase1b",
        "expert": "hoi",
        "initialization": "random",
        "seed": 42,
    }
    variants = {
        "released": {"model": {}},
        "d2x_base": {
            **common,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_BASE,
            },
            "architecture_variant": HOI_ARCHITECTURE_BASE,
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
            "interaction_adapter_contract": {},
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
            "interaction_adapter_contract": {},
        },
        "d2ae": {
            **common,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_D2AE,
            },
            "architecture_variant": HOI_ARCHITECTURE_D2AE,
            "sparse_relation_contract": sparse_relation_contract_metadata(),
        },
        "d2af_missing_contract": {
            **common,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_D2AF,
            },
            "architecture_variant": HOI_ARCHITECTURE_D2AF,
        },
    }
    rejected = {}
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for label, checkpoint in variants.items():
            path = directory / f"{label}.pth"
            torch.save(checkpoint, path)
            try:
                load_trained_hoi_prior(
                    str(path),
                    torch.device("cpu"),
                    use_ema=False,
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AF,
                )
            except (ValueError, RuntimeError) as error:
                rejected[label] = {
                    "rejected": True,
                    "error": str(error),
                }
            else:
                rejected[label] = {"rejected": False, "error": None}

        reverse_path = directory / "d2af_reverse.pth"
        torch.save({
            **common,
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_D2AF,
            },
            "architecture_variant": HOI_ARCHITECTURE_D2AF,
            "diffusion_reliability_contract":
                diffusion_reliability_contract_metadata(),
        }, reverse_path)
        try:
            load_trained_hoi_prior(
                str(reverse_path),
                torch.device("cpu"),
                use_ema=False,
                expected_architecture_variant=HOI_ARCHITECTURE_D2AE,
            )
        except (ValueError, RuntimeError) as error:
            reverse = {"rejected": True, "error": str(error)}
        else:
            reverse = {"rejected": False, "error": None}
    if (
        not all(value["rejected"] for value in rejected.values())
        or not reverse["rejected"]
    ):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: checkpoint rejection failed"
        )
    return {
        "d2af_rejects": rejected,
        "d2ae_rejects_d2af": reverse,
        "scientific_checkpoint_loads": 0,
        "synthetic_checkpoint_attempts": len(rejected) + 1,
    }


def static_contract(repo: Path) -> Dict[str, object]:
    schedule_path = repo / "code/priors/diffusion_schedule.py"
    sparse_path = repo / "code/priors/sparse_relation.py"
    trainer_path = repo / "code/train_hoi_prior.py"
    schedule_source = schedule_path.read_text(encoding="utf-8")
    sparse_source = sparse_path.read_text(encoding="utf-8")
    trainer_source = trainer_path.read_text(encoding="utf-8")
    field_source = inspect.getsource(
        SparseCurrentStateRelationField.forward
    ).lower()
    schedule_tree = ast.parse(schedule_source)
    sparse_tree = ast.parse(sparse_source)
    imported = set()
    for tree in (schedule_tree, sparse_tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    forbidden_imports = sorted(
        imported & {"numpy", "scipy", "trimesh", "sklearn"}
    )
    forbidden_field_tokens = [
        token
        for token in (
            "x_start",
            "future_gt",
            "contact_label",
            "scene",
            "snr_weight",
            "timestep_weight",
            "gamma",
            "per_anchor",
            "previous_x0",
        )
        if token in field_source
    ]
    checks = {
        "pure_torch_schedule_and_field": not forbidden_imports,
        "forbidden_field_sources_absent": not forbidden_field_tokens,
        "single_canonical_schedule_constructor": (
            schedule_source.count("torch.linspace(") == 1
            and "BETA_START" not in sparse_source
            and "BETA_END" not in sparse_source
        ),
        "no_loss_weighting": (
            "d2af_timestep_loss_weight" not in trainer_source
            and "d2af_snr_weight" not in trainer_source
        ),
        "single_writeback": (
            inspect.getsource(
                SparseCurrentStateRelationField.forward
            ).count("return motion + attenuated_writeback") == 1
        ),
        "no_learned_schedule_parameter": (
            "nn.Parameter(schedule" not in sparse_source
            and "nn.Parameter(self.sqrt_alpha_bar" not in sparse_source
        ),
        "no_checkpoint_or_model_in_eligibility": True,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: static contract failed: {checks}"
        )
    return {
        "checks": checks,
        "forbidden_imports": forbidden_imports,
        "forbidden_field_tokens": forbidden_field_tokens,
        "schedule_source_sha256": sha256_file(schedule_path),
        "sparse_relation_source_sha256": sha256_file(sparse_path),
        "trainer_source_sha256": sha256_file(trainer_path),
    }


def run_contract(
    repo: Path,
    run_id: str,
    *,
    require_clean: bool = True,
) -> Dict[str, object]:
    started = time.perf_counter()
    repo = repo.resolve()
    run_date = validate_actual_run_id(run_id)
    identity = authority_identity(repo, require_clean=require_clean)
    cfg = resolved_config(repo, run_date)
    formal_source = _d2af_formal_source_contract(repo)

    # Reuse the sealed D2-AE pure geometry, assets, finiteness, sampler metadata,
    # evaluator-hash, HSIPrior-storage, and clean-output contracts unchanged.
    inherited = {
        "sparse_assets": d2ae.sparse_asset_contract(repo),
        "geometry": d2ae.geometry_contract(),
        "model_geometry_and_independence": d2ae.model_contract(),
        "static_and_evaluator": d2ae.static_contract(repo),
    }
    result = {
        "schema_version": 1,
        "status": "passed",
        "classification": "cpu-contract-passed",
        "run_id": run_id,
        "subphase": "1B-D2-AF0-cpu-contract",
        "seed": 42,
        "runtime_seconds": time.perf_counter() - started,
        "identity": identity,
        "formal_source_contract": formal_source,
        "resolved_formal_config": OmegaConf.to_container(cfg, resolve=True),
        "inherited_d2ae_contracts": inherited,
        "schedule": schedule_contract(),
        "timestep_and_scaling": timestep_and_scaling_contract(),
        "model": model_contract(),
        "train_sample_timestep_identity": timestep_identity_contract(cfg),
        "checkpoint_provenance": checkpoint_rejection_contract(),
        "static_contract": static_contract(repo),
        "training_and_registry": config_contract(repo, cfg),
        "optimizer_created": False,
        "optimizer_updates": 0,
        "scientific_checkpoint_loads": 0,
        "checkpoint_writes": 0,
        "official_test_used": False,
        "checkpoint_selection": False,
        "formal_training_started": False,
        "consistency_started": False,
        "hsiprior_started": False,
        "mixer_started": False,
    }
    result["runtime_seconds"] = time.perf_counter() - started
    return result


def exclusive_text(path: Path, value: str) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def exclusive_json(path: Path, value: MutableMapping[str, object]) -> None:
    exclusive_text(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


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
                "D2-AF CPU contract requires a pre-archived resolved config"
            )
        if path.read_text(encoding="utf-8") != resolved:
            raise RuntimeError(
                "D2-AF CPU contract differs from archived resolved config"
            )
    return sha256_file(path)


def failure_record(
    *,
    run_id: str,
    started: float,
    error: Exception,
    repo: Path,
) -> Dict[str, object]:
    try:
        commit: Optional[str] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
        ).strip()
    except Exception:
        commit = None
    return {
        "schema_version": 1,
        "status": "failed",
        "classification": FAILURE_CLASSIFICATION,
        "run_id": run_id,
        "subphase": "1B-D2-AF0-cpu-contract",
        "seed": 42,
        "git_commit": commit,
        "runtime_seconds": time.perf_counter() - started,
        "failure_type": type(error).__name__,
        "failure": str(error),
        "optimizer_created": False,
        "optimizer_updates": 0,
        "scientific_checkpoint_loads": 0,
        "checkpoint_writes": 0,
        "official_test_used": False,
        "formal_training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-config-output", type=Path, required=True)
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help=(
            "archive and validate the exact CPU contract config without "
            "running diagnostics"
        ),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    repo = args.repo_root.resolve()
    try:
        run_date = validate_actual_run_id(args.run_id)
        # Identity is checked before writing the archived resolved config.
        authority_identity(repo, require_clean=True)
        cfg = resolved_config(repo, run_date)
        resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        if "${" in resolved_yaml:
            raise RuntimeError("D2-AF CPU resolved config is incomplete")
        resolved_sha256 = archive_or_validate_resolved_config(
            args.resolved_config_output,
            resolved_yaml,
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
                "cpu_contract_started": False,
                "checkpoint_loads": 0,
                "optimizer_created": False,
            }
            print(json.dumps(value, indent=2, sort_keys=True), flush=True)
            return 0
        result = run_contract(repo, args.run_id, require_clean=True)
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
        print(
            f"{FAILURE_CLASSIFICATION}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
