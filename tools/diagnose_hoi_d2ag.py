#!/usr/bin/env python3
"""Authority CPU hard-gate diagnostics for the fixed D2-AG0 mechanism."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional

import torch
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

from priors.diffusion import GaussianDiffusion, prepare_clean_x0  # noqa: E402
from priors.diffusion_schedule import (  # noqa: E402
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
    HOI_ARCHITECTURE_D2AG,
    build_expert,
    load_trained_hoi_prior,
)
from priors.representation import REPRESENTATION  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    D2AG_HIGH_T_SELF_CONDITION_CUTOFF,
    D2AG_SELF_CONDITION_PROBABILITY,
    D2AG_VARIABLE_ANCHORS,
    SPARSE_RELATION_PARAMETER_COUNT,
    SparseCurrentStateRelationField,
    build_d2ag_relation_source,
    diffusion_reliability_contract_metadata,
    selfcond_relation_source_contract_metadata,
    sparse_relation_contract_metadata,
)
from tools import diagnose_hoi_d2ae as d2ae  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    D2AF_MAXIMUM_ETA_HOURS,
    D2AF_MINIMUM_THROUGHPUT,
    D2AG_MAXIMUM_ETA_HOURS,
    D2AG_MINIMUM_THROUGHPUT,
    _d2ae_gradient_audit,
    _d2ag_formal_source_contract,
    _d2ag_mask_seed,
    _d2ag_selection_mask,
    _forward_losses,
    _locked_loss_weights,
    _optimization_contract,
    _state_dict_sha256,
    _validate_d2ag_contract,
    _validate_fk_foot_temporal_routing_mode,
)


RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ag-cpu-contract"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
FAILURE_CLASSIFICATION = "selfcond-relation-source-contract-failure-stop"
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
PARITY_TOLERANCE = 1.0e-6
DIFFUSION_STEPS = 500

AUTHORITY_VERIFICATION_COMMANDS = (
    '"$INFBAGEL_PYTHON" -m unittest tests.test_hoi_d2ag -v',
    '"$INFBAGEL_PYTHON" -m unittest tests.test_hoi_d2ag_eval -v',
    '"$INFBAGEL_PYTHON" -m unittest tests.test_hoi_d2ag_lifecycle_cpu -v',
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
            "D2-AG CPU contract run id must use the locked stem and actual date"
        )
    return match.group("date")


def authority_identity(repo: Path, *, require_clean: bool) -> Dict[str, object]:
    repo = repo.resolve()
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


def _merged_config(repo: Path, formal_run_id: str, *, variant: str):
    cfg = OmegaConf.merge(
        OmegaConf.load(repo / "code/config/config_train_hoi_prior.yaml"),
        OmegaConf.load(repo / "code/config" / f"config_train_hoi_prior_{variant}.yaml"),
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
    formal_run_id = f"p1-hoi-d2ag-selfcond-relation-source-s42-{run_date}"
    return _merged_config(repo, formal_run_id, variant="d2ag")


def config_contract(repo: Path, cfg) -> Dict[str, object]:
    _validate_fk_foot_temporal_routing_mode(cfg)
    lifecycle = _validate_d2ag_contract(cfg, 4, require_performance_gate=False)
    split_path = Path(str(cfg.split_manifest)).resolve()
    split_hash = sha256_file(split_path)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"{FAILURE_CLASSIFICATION}: split hash mismatch")

    run_date = str(cfg.run_id).rsplit("-", 1)[-1]
    peers = {
        "d2ae": _merged_config(
            repo,
            f"p1-hoi-d2ae-sparse-relation-field-s42-{run_date}",
            variant="d2ae",
        ),
        "d2af": _merged_config(
            repo,
            f"p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-{run_date}",
            variant="d2af",
        ),
    }
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
        "d2ag_selfcond_relation_source",
        "hoi_architecture_variant",
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
        "d2ag_performance_benchmark_path",
        "d2ag_performance_benchmark_sha256",
    }
    ag_values = OmegaConf.to_container(cfg, resolve=True)
    differences = {}
    for label, peer in peers.items():
        peer_values = OmegaConf.to_container(peer, resolve=True)
        differences[label] = sorted(
            key
            for key in set(peer_values) | set(ag_values)
            if key not in ignored and peer_values.get(key) != ag_values.get(key)
        )
    weights = _locked_loss_weights(cfg)
    optimization = _optimization_contract(cfg)
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    checks = {
        "trainer_lifecycle": lifecycle is not None,
        "performance_gate_disabled_for_cpu": (
            lifecycle is not None
            and lifecycle["performance_gate_required"] is False
        ),
        "eligibility_gate_absent": (
            cfg.get("d2ag_clean_signal_eligibility_path") is None
            and cfg.get("d2ag_clean_signal_eligibility_sha256") is None
        ),
        "single_factor_config_vs_d2ae": not differences["d2ae"],
        "single_factor_config_vs_d2af": not differences["d2af"],
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
        "throughput_gate_independent_of_d2af": (
            D2AG_MINIMUM_THROUGHPUT != D2AF_MINIMUM_THROUGHPUT
            and D2AG_MAXIMUM_ETA_HOURS != D2AF_MAXIMUM_ETA_HOURS
        ),
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
        "minimum_throughput_windows_per_second": D2AG_MINIMUM_THROUGHPUT,
        "maximum_full_budget_eta_hours": D2AG_MAXIMUM_ETA_HOURS,
        "checks": checks,
        "registry_validation_returncode": validation.returncode,
        "registry_validation_output": validation.stdout.strip(),
        "full_authority_suite_commands": list(AUTHORITY_VERIFICATION_COMMANDS),
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


def schedule_absence_contract() -> Dict[str, object]:
    """D2-AG must not register or read the D2-AF reliability schedule buffer."""
    diffusion = GaussianDiffusion()
    fields: List[SparseCurrentStateRelationField] = [
        SparseCurrentStateRelationField(512, selfcond_relation_source=True)
        for _ in range(4)
    ]
    reliability = SparseCurrentStateRelationField(512, diffusion_reliability=True)
    combined_rejected = False
    try:
        SparseCurrentStateRelationField(
            512, diffusion_reliability=True, selfcond_relation_source=True,
        )
    except ValueError:
        combined_rejected = True
    rho_override_rejected = False
    try:
        fields[0].set_rho_override(1.0)
    except ValueError:
        rho_override_rejected = True
    checks = {
        "field_schedule_absent_on_every_rank": all(
            field.sqrt_alpha_bar is None for field in fields
        ),
        "field_schedule_absent_from_state_dict": all(
            "sqrt_alpha_bar" not in field.state_dict() for field in fields
        ),
        "d2af_field_still_registers_schedule": (
            reliability.sqrt_alpha_bar is not None
        ),
        "combined_reliability_and_selfcond_rejected": combined_rejected,
        "rho_override_rejected": rho_override_rejected,
        "diffusion_schedule_unchanged": tensor_sha256(diffusion.sqrt_alpha_bar)
        == SQRT_ALPHA_BAR_SHA256,
        "canonical_schedule_hash": (
            diffusion_schedule_contract_metadata()["sqrt_alpha_bar_sha256"]
            == SQRT_ALPHA_BAR_SHA256
        ),
        "contract_declares_attenuation_off": (
            selfcond_relation_source_contract_metadata()[
                "sqrt_alpha_bar_attenuation"
            ] is False
            and selfcond_relation_source_contract_metadata()[
                "schedule_buffer_registered"
            ] is False
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: schedule absence contract failed: {checks}"
        )
    return {
        "gaussian_diffusion_sqrt_alpha_bar_sha256": tensor_sha256(
            diffusion.sqrt_alpha_bar
        ),
        "field_schedule_registered_by_simulated_rank": [
            field.sqrt_alpha_bar is not None for field in fields
        ],
        "checks": checks,
    }


def relation_source_contract() -> Dict[str, object]:
    """Train/sample shared-builder parity, history pin, and detachment."""
    values = d2ae.synthetic_inputs(batch=4)
    current = values["current"]
    estimate = torch.randn(
        current.shape, generator=torch.Generator().manual_seed(931),
    )
    estimate.requires_grad_(True)
    index = torch.tensor([0, 2], dtype=torch.long)
    full = build_d2ag_relation_source(current, estimate)
    subset = build_d2ag_relation_source(
        current, estimate.detach().index_select(0, index), index=index,
    )
    passthrough = build_d2ag_relation_source(current, None)
    history = REPRESENTATION.history_frames
    variable = list(D2AG_VARIABLE_ANCHORS)
    unselected = torch.tensor([1, 3], dtype=torch.long)
    rejections = {}
    for name, call in (
        ("index_without_estimate", lambda: build_d2ag_relation_source(
            current, None, index=index,
        )),
        ("wrong_current_shape", lambda: build_d2ag_relation_source(
            current[:, :, :4],
        )),
        ("batch_mismatch", lambda: build_d2ag_relation_source(
            current, estimate.detach()[:2],
        )),
        ("index_row_mismatch", lambda: build_d2ag_relation_source(
            current, estimate.detach()[:1], index=index,
        )),
        ("index_out_of_range", lambda: build_d2ag_relation_source(
            current,
            estimate.detach().index_select(0, index),
            index=torch.tensor([0, 99], dtype=torch.long),
        )),
        ("index_wrong_dtype", lambda: build_d2ag_relation_source(
            current,
            estimate.detach().index_select(0, index),
            index=index.float(),
        )),
    ):
        try:
            call()
        except ValueError as error:
            rejections[name] = {"rejected": True, "error": str(error)}
        else:
            rejections[name] = {"rejected": False, "error": None}
    checks = {
        "passthrough_is_current": passthrough is current,
        "full_history_pin_exact": float(
            (full[:, :history] - current[:, :history]).abs().amax()
        ) == 0.0,
        "subset_history_pin_exact": float(
            (subset[:, :history] - current[:, :history]).abs().amax()
        ) == 0.0,
        "full_variable_anchors_from_estimate": float(
            (
                full[:, variable] - estimate.detach()[:, variable]
            ).abs().amax()
        ) == 0.0,
        "subset_selected_rows_from_estimate": float(
            (
                subset.index_select(0, index)[:, variable]
                - estimate.detach().index_select(0, index)[:, variable]
            ).abs().amax()
        ) == 0.0,
        "subset_unselected_rows_are_current": float(
            (
                subset.index_select(0, unselected)
                - current.index_select(0, unselected)
            ).abs().amax()
        ) == 0.0,
        "estimate_detached_in_source": (
            full.grad_fn is None and not full.requires_grad
        ),
        "all_invalid_rejected": all(
            item["rejected"] for item in rejections.values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: relation source contract failed: {checks}"
        )
    return {
        "history_frames": history,
        "variable_anchors": variable,
        "selected_index": index.tolist(),
        "unselected_index": unselected.tolist(),
        "invalid_input_rejection": rejections,
        "checks": checks,
    }


def _forward(
    model: torch.nn.Module,
    values: Mapping[str, torch.Tensor],
    *,
    relation_source: Optional[torch.Tensor] = None,
    relation_source_estimate: bool = False,
):
    extra: Dict[str, object] = {}
    if relation_source is not None:
        extra["relation_source"] = relation_source
    if relation_source_estimate:
        extra["relation_source_estimate"] = True
    return model(
        values["current"],
        values["timesteps"],
        values["text"],
        values["global_bps"],
        values["goals"],
        values["progress"],
        **_relation_arguments(values),
        **extra,
    )


def model_contract() -> Dict[str, object]:
    """Parameter counts, seed-42 identity, D2-AE parity and gradient audits."""
    torch.manual_seed(42)
    model = build_expert(
        "hoi",
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AG,
    )
    initial_hash = _state_dict_sha256(model.state_dict())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    field = model.network.sparse_relation_field
    relation_parameters = sum(
        parameter.numel() for parameter in field.parameters()
    )

    torch.manual_seed(42)
    base = build_expert(
        "hoi",
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_BASE,
    )
    torch.manual_seed(42)
    peer = build_expert(
        "hoi",
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AE,
    )
    shared_keys = sorted(base.state_dict())
    sparse_keys = sorted(set(model.state_dict()) - set(base.state_dict()))
    shared_state_exact = all(
        torch.equal(base.state_dict()[key], model.state_dict()[key])
        for key in shared_keys
    )
    peer_key_set_identical = sorted(peer.state_dict()) == sorted(
        model.state_dict()
    )
    values = d2ae.synthetic_inputs(batch=2)
    base.eval()
    peer.eval()
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
        peer_actual = _forward(peer, values)
        estimate = torch.randn(
            values["current"].shape,
            generator=torch.Generator().manual_seed(4242),
        )
        source = build_d2ag_relation_source(values["current"], estimate)
        selfcond_actual = _forward(model, values, relation_source=source)
        # An estimate pass with no explicit source is a D2-AE-equivalent forward.
        estimate_actual = _forward(model, values, relation_source_estimate=True)
    base_parity = float((actual - expected).abs().max())
    peer_parity = float((actual - peer_actual).abs().max())
    estimate_parity = float((estimate_actual - peer_actual).abs().max())
    # At the registered exact-zero alpha initialization tanh(alpha) is exactly
    # zero, so the writeback vanishes and the relation source cannot move the
    # output.  The source-sensitivity claim is therefore only meaningful with an
    # activated gate; both facts are asserted separately below.
    selfcond_delta_at_zero_alpha = float((selfcond_actual - peer_actual).abs().max())
    with torch.no_grad():
        field.alpha.copy_(torch.atanh(torch.tensor(0.1)))
        activated_current = _forward(model, values)
        activated_selfcond = _forward(model, values, relation_source=source)
        field.alpha.zero_()
    selfcond_delta_activated = float(
        (activated_selfcond - activated_current).abs().max()
    )
    del base, peer

    gradients = {}
    for timestep in REGISTERED_TIMESTEPS:
        probe = d2ae.synthetic_inputs(batch=2, seed=20_000 + timestep)
        probe["timesteps"].fill_(timestep)
        probe_estimate = torch.randn(
            probe["current"].shape,
            generator=torch.Generator().manual_seed(70_000 + timestep),
        )
        probe_source = build_d2ag_relation_source(
            probe["current"],
            probe_estimate.index_select(0, torch.tensor([0], dtype=torch.long)),
            index=torch.tensor([0], dtype=torch.long),
        )
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            field.alpha.zero_()
        prediction = _forward(model, probe, relation_source=probe_source)
        (prediction - probe["current"]).square().mean().backward()
        initial = _d2ae_gradient_audit(model, require_relation_paths=False)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            field.alpha.copy_(torch.atanh(torch.tensor(0.1)))
        prediction = _forward(model, probe, relation_source=probe_source)
        (prediction - probe["current"]).square().mean().backward()
        activated = _d2ae_gradient_audit(model, require_relation_paths=True)
        gradients[str(timestep)] = {
            "initial_zero_gate": initial,
            "test_only_activated_gate": activated,
            "partial_selection_index": [0],
        }
    with torch.no_grad():
        field.alpha.zero_()
    model.zero_grad(set_to_none=True)

    # ``x0_hat`` detachment: perturbing the relation source must not reach the
    # estimating parameters.  A grad-carrying estimate is rejected by design at
    # the builder, so verify the built source carries no graph at all.
    graph_estimate = _forward(model, values)
    graph_source = build_d2ag_relation_source(values["current"], graph_estimate)
    detached_source = (
        graph_source.grad_fn is None and not graph_source.requires_grad
    )

    restriction = {}
    for label, variant in (
        ("base", HOI_ARCHITECTURE_BASE),
        ("d2ae", HOI_ARCHITECTURE_D2AE),
        ("d2af", HOI_ARCHITECTURE_D2AF),
    ):
        torch.manual_seed(42)
        other = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=variant,
        ).eval()
        other_values = (
            values if variant != HOI_ARCHITECTURE_BASE
            else {**values, **{key: None for key in _relation_arguments(values)}}
        )
        for probe_name, keyword in (
            ("relation_source", {"relation_source": source}),
            ("relation_source_estimate", {"relation_source_estimate": True}),
        ):
            try:
                with torch.no_grad():
                    other(
                        other_values["current"],
                        other_values["timesteps"],
                        other_values["text"],
                        other_values["global_bps"],
                        other_values["goals"],
                        other_values["progress"],
                        **{
                            key: other_values[key]
                            for key in _relation_arguments(values)
                        },
                        **keyword,
                    )
            except (ValueError, RuntimeError) as error:
                restriction[f"{label}_{probe_name}"] = {
                    "rejected": True, "error": str(error),
                }
            else:
                restriction[f"{label}_{probe_name}"] = {
                    "rejected": False, "error": None,
                }
        del other

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
        "schedule_buffer_absent": field.sqrt_alpha_bar is None,
        "shared_state_exact": shared_state_exact,
        "state_keys_identical_to_d2ae": peer_key_set_identical,
        "sparse_keys": (
            len(sparse_keys) == 10
            and all(
                key.startswith("network.sparse_relation_field.")
                for key in sparse_keys
            )
        ),
        "base_parity": base_parity <= PARITY_TOLERANCE,
        "base_parity_exact_zero": base_parity == 0.0,
        "d2ae_parity_exact_zero": peer_parity == 0.0,
        "estimate_pass_equals_d2ae_exact_zero": estimate_parity == 0.0,
        "selfcond_source_inert_at_zero_alpha": (
            selfcond_delta_at_zero_alpha == 0.0
        ),
        "selfcond_source_changes_output_when_activated": (
            selfcond_delta_activated > 0.0
        ),
        "x0_hat_detached": detached_source,
        "output_shape": tuple(actual.shape) == (2, 16, 232),
        "all_timestep_gradients": len(gradients) == 3,
        "predecessor_variants_reject_selfcond_arguments": all(
            item["rejected"] for item in restriction.values()
        ),
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
        "d2ae_parity_max_abs": peer_parity,
        "estimate_pass_minus_d2ae_max_abs": estimate_parity,
        "selfcond_minus_d2ae_max_abs_at_zero_alpha": (
            selfcond_delta_at_zero_alpha
        ),
        "selfcond_minus_current_max_abs_activated_gate": (
            selfcond_delta_activated
        ),
        "shared_state_key_count": len(shared_keys),
        "sparse_state_keys": sparse_keys,
        "gradients_by_timestep": gradients,
        "predecessor_argument_restriction": restriction,
        "test_only_gate": 0.1,
        "test_only_probe_saved": False,
        "test_only_probe_optimizer_updates": 0,
        "checks": checks,
    }


def diagnostic_gate_contract() -> Dict[str, object]:
    """The five registered internal gates, applied inside the field forward."""
    field = SparseCurrentStateRelationField(
        512, selfcond_relation_source=True,
    ).eval()
    field.set_gate_override(0.1)
    values = d2ae.synthetic_inputs(batch=3)
    current = values["current"]
    estimate = torch.randn(
        current.shape, generator=torch.Generator().manual_seed(515),
    )
    source = build_d2ag_relation_source(current, estimate)
    motion = torch.randn(
        3, 16, 512, generator=torch.Generator().manual_seed(616),
    )
    # One sample below the cutoff and two above, so the high-t gate must produce
    # a per-sample mixture rather than a batch-wide switch.
    timesteps = torch.tensor([0, 249, 499], dtype=torch.long)
    low = timesteps < D2AG_HIGH_T_SELF_CONDITION_CUTOFF

    def run(variant: str, *, relation_source, steps=timesteps):
        field.set_diagnostic_variant(variant)
        with torch.no_grad():
            return field(
                motion,
                current,
                **_relation_arguments(values),
                timesteps=steps,
                relation_source=relation_source,
            )

    full_selfcond = run("full", relation_source=source)
    # Same inputs without ``timesteps``; captured here, while the gate override
    # is still active, so the comparison isolates ``timesteps`` alone.
    full_selfcond_no_timesteps = run("full", relation_source=source, steps=None)
    full_current = run("full", relation_source=None)
    substituted = run("source_substituted_xt", relation_source=source)
    high_t = run("high_t_restricted", relation_source=source)
    displaced = run("object_displaced_counterfactual", relation_source=source)
    temporal = run("temporal_correspondence_permuted", relation_source=source)
    role = run("left_right_role_swapped", relation_source=source)
    field.set_diagnostic_variant("full")

    high_t_requires_timesteps = False
    try:
        run("high_t_restricted", relation_source=source, steps=None)
    except ValueError:
        high_t_requires_timesteps = True
    field.set_diagnostic_variant("full")
    high_t_rejects_wrong_shape = False
    try:
        run(
            "high_t_restricted",
            relation_source=source,
            steps=timesteps[:2],
        )
    except ValueError:
        high_t_rejects_wrong_shape = True
    field.set_diagnostic_variant("full")
    unknown_variant_rejected = False
    try:
        field.set_diagnostic_variant("relation_gate_ablated_and_displaced")
    except ValueError:
        unknown_variant_rejected = True
    field.set_gate_override(None)
    field.set_diagnostic_variant("full")

    # Gate 3 magnitude check: the registered delta is metric-space 0.10 m on x
    # only, applied after denormalization, to object translation channels
    # 216:219 of frames 5/10/15 alone.
    scale = (
        values["object_maximum"] - values["object_minimum"]
    ).reshape(1, 1, 3)
    expected_normalized = 2.0 * torch.tensor([0.10, 0.0, 0.0]) / scale
    variable = list(D2AG_VARIABLE_ANCHORS)
    field.set_diagnostic_variant("object_displaced_counterfactual")
    with torch.no_grad():
        displaced_source = field._diagnostic_relation_source(
            source,
            current,
            timesteps=timesteps,
            object_minimum=values["object_minimum"],
            object_maximum=values["object_maximum"],
        )
    field.set_diagnostic_variant("full")
    observed_delta = displaced_source[:, variable, 216:219] - source[
        :, variable, 216:219
    ]
    delta_max_abs = float(
        (observed_delta - expected_normalized).abs().amax()
    )
    untouched_channels_max_abs = float(
        torch.cat(
            (
                displaced_source[..., :216] - source[..., :216],
                displaced_source[..., 219:] - source[..., 219:],
            ),
            dim=-1,
        ).abs().amax()
    )
    anchor_zero_untouched_max_abs = float(
        (displaced_source[:, 0] - source[:, 0]).abs().amax()
    )
    field.set_diagnostic_variant("high_t_restricted")
    with torch.no_grad():
        high_t_source = field._diagnostic_relation_source(
            source,
            current,
            timesteps=timesteps,
            object_minimum=values["object_minimum"],
            object_maximum=values["object_maximum"],
        )
    field.set_diagnostic_variant("full")
    checks = {
        "source_substitution_equals_current_source": float(
            (substituted - full_current).abs().amax()
        ) == 0.0,
        "source_substitution_differs_from_full": float(
            (substituted - full_selfcond).abs().amax()
        ) > 0.0,
        "high_t_low_rows_keep_self_conditioning": float(
            (
                high_t_source[low] - source[low]
            ).abs().amax()
        ) == 0.0,
        "high_t_high_rows_fall_back_to_current": float(
            (
                high_t_source[~low] - current[~low]
            ).abs().amax()
        ) == 0.0,
        "high_t_high_rows_are_not_zeroed": bool(
            float(high_t_source[~low].abs().amax()) > 0.0
        ),
        "high_t_output_between_full_and_substituted": (
            float((high_t - full_selfcond).abs().amax()) > 0.0
            and float((high_t - substituted).abs().amax()) > 0.0
        ),
        "high_t_requires_timesteps": high_t_requires_timesteps,
        "high_t_rejects_wrong_timestep_shape": high_t_rejects_wrong_shape,
        "object_delta_is_registered_metric_0.10m_x": (
            delta_max_abs <= PARITY_TOLERANCE
        ),
        "object_delta_leaves_other_channels_exact": (
            untouched_channels_max_abs == 0.0
        ),
        "object_delta_leaves_anchor_zero_exact": (
            anchor_zero_untouched_max_abs == 0.0
        ),
        "object_displacement_changes_output": float(
            (displaced - full_selfcond).abs().amax()
        ) > 0.0,
        "temporal_permutation_changes_output": float(
            (temporal - full_selfcond).abs().amax()
        ) > 0.0,
        "role_swap_changes_output": float(
            (role - full_selfcond).abs().amax()
        ) > 0.0,
        "timesteps_diagnostic_only_for_full": float(
            (full_selfcond_no_timesteps - full_selfcond).abs().amax()
        ) == 0.0,
        "unknown_variant_rejected": unknown_variant_rejected,
    }
    field.set_diagnostic_variant("full")
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: diagnostic gate contract failed: {checks}"
        )
    return {
        "high_t_cutoff": D2AG_HIGH_T_SELF_CONDITION_CUTOFF,
        "probe_timesteps": timesteps.tolist(),
        "object_displacement_m": [0.10, 0.0, 0.0],
        "object_displacement_space": "metric_after_denormalization",
        "object_displacement_channels": [216, 219],
        "object_displacement_frames": variable,
        "object_delta_minus_registered_max_abs": delta_max_abs,
        "gate_ablation_variant_used": False,
        "checks": checks,
    }


def sampler_contract() -> Dict[str, object]:
    """First-step source identity, ``prev_x0`` provenance and timestep trace."""
    observations: List[Dict[str, object]] = []

    class Recorder(torch.nn.Module):
        architecture_variant = HOI_ARCHITECTURE_D2AG

        def __init__(self) -> None:
            super().__init__()
            self.trace: List[int] = []
            self.raw_outputs: List[torch.Tensor] = []
            self.inputs: List[torch.Tensor] = []

        def forward(self, current, timesteps, *args, **kwargs):
            del args
            source = kwargs.get("relation_source")
            self.trace.append(int(timesteps[0]))
            self.inputs.append(current.detach().clone())
            observations.append({
                "timestep": int(timesteps[0]),
                "relation_source_is_none": source is None,
                "source_sha256": (
                    None if source is None
                    else d2ae.tensor_sha256(source.detach())
                ),
                "history_pin_exact": (
                    None if source is None else float(
                        (
                            source[:, :REPRESENTATION.history_frames]
                            - current[:, :REPRESENTATION.history_frames]
                        ).abs().amax()
                    ) == 0.0
                ),
            })
            raw = torch.randn(
                current.shape,
                generator=torch.Generator().manual_seed(
                    9_000 + int(timesteps[0])
                ),
            )
            # Give the raw output an out-of-SO(3) rotation block and a history
            # block that differs from the fixed history, so a ``prev_x0`` taken
            # after ``prepare_clean_x0`` would be detectably different.
            raw[..., 219:228] = raw[..., 219:228] * 3.0 + 1.0
            self.raw_outputs.append(raw.detach().clone())
            return raw

    recorder = Recorder()
    values = d2ae.synthetic_inputs(batch=1)
    fixed_history = torch.zeros(1, REPRESENTATION.history_frames, 232)
    GaussianDiffusion().sample(
        recorder,
        fixed_history,
        values["text"][:1],
        values["global_bps"][:1],
        values["goals"][:1],
        values["progress"][:1],
        **{key: value[:1] if value.ndim > 1 else value
           for key, value in _relation_arguments(values).items()},
        generator=torch.Generator().manual_seed(42),
    )
    expected_trace = list(reversed(range(DIFFUSION_STEPS)))
    trace_sha256 = hashlib.sha256(
        ("\n".join(str(value) for value in recorder.trace) + "\n").encode("utf-8")
    ).hexdigest()
    # Every step after the first must receive exactly the shared builder applied
    # to the previous step's raw ``x0_hat``, and never the ``prepare_clean_x0``
    # output (which restores history and closes rotations on SO(3)).
    # Honest scope note: with the registered ``object_so3_x0=False`` sampler the
    # only thing ``prepare_clean_x0`` changes is the two history frames, which
    # the shared builder re-pins to ``x_t`` regardless.  Raw and history-restored
    # sources are therefore byte-identical there, so that comparison alone cannot
    # discriminate.  The SO(3) branch is what makes the registered "raw x0_hat"
    # provenance observable, so both are checked.
    raw_source_matches = []
    history_restored_source_matches = []
    so3_source_matches = []
    so3_distinguishable = []
    for index in range(1, len(observations)):
        previous_raw = recorder.raw_outputs[index - 1]
        current_state = recorder.inputs[index]
        expected_raw_source = build_d2ag_relation_source(
            current_state, previous_raw,
        )
        expected_history_restored = build_d2ag_relation_source(
            current_state,
            prepare_clean_x0(previous_raw.clone(), fixed_history),
        )
        expected_so3 = build_d2ag_relation_source(
            current_state,
            prepare_clean_x0(
                previous_raw.clone(), fixed_history, object_so3_x0=True,
            ),
        )
        observed = observations[index]["source_sha256"]
        raw_source_matches.append(
            observed == d2ae.tensor_sha256(expected_raw_source)
        )
        history_restored_source_matches.append(
            observed == d2ae.tensor_sha256(expected_history_restored)
        )
        so3_source_matches.append(observed == d2ae.tensor_sha256(expected_so3))
        so3_distinguishable.append(
            d2ae.tensor_sha256(expected_so3)
            != d2ae.tensor_sha256(expected_raw_source)
        )
    checks = {
        "trace_exact_499_to_0": recorder.trace == expected_trace,
        "trace_length": len(recorder.trace) == DIFFUSION_STEPS,
        "first_step_source_is_none": (
            observations[0]["relation_source_is_none"] is True
            and observations[0]["timestep"] == DIFFUSION_STEPS - 1
        ),
        "only_first_step_lacks_source": sum(
            1 for row in observations if row["relation_source_is_none"]
        ) == 1,
        "every_later_step_pins_history": all(
            row["history_pin_exact"] is True
            for row in observations[1:]
        ),
        "prev_x0_is_previous_raw_x0_hat": (
            len(raw_source_matches) == DIFFUSION_STEPS - 1
            and all(raw_source_matches)
        ),
        "history_restoration_is_indistinguishable_by_construction": all(
            history_restored_source_matches
        ),
        "so3_projection_would_be_observable": all(so3_distinguishable),
        "prev_x0_is_not_so3_projected": not any(so3_source_matches),
        "field_entered_on_every_step": len(observations) == DIFFUSION_STEPS,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: sampler contract failed: {checks}"
        )
    return {
        "diffusion_steps": DIFFUSION_STEPS,
        "sampler_trace_first": recorder.trace[0],
        "sampler_trace_last": recorder.trace[-1],
        "sampler_trace_length": len(recorder.trace),
        "sampler_trace_sha256": trace_sha256,
        "first_reverse_step_source": "current_noisy_state",
        "later_reverse_step_source": "previous_step_raw_x0_hat",
        "prepare_clean_x0_applied_to_relation_source": False,
        "relation_source_so3_projected": False,
        "steps_without_relation_source": 1,
        "checks": checks,
    }


def training_rng_contract(cfg) -> Dict[str, object]:
    """Mask generator independence, eval-mode discipline and timestep identity."""
    captured: Dict[str, object] = {}

    class StopAfterModel(RuntimeError):
        pass

    class DiffusionStub:
        def q_sample(self, clean, timesteps, noise):
            captured["q_sample"] = timesteps
            captured["noise"] = noise.detach().clone()
            return clean

    class ModelStub(torch.nn.Module):
        architecture_variant = HOI_ARCHITECTURE_D2AG

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, current, timesteps, *args, **kwargs):
            del args
            self.calls += 1
            if bool(kwargs.get("relation_source_estimate")):
                captured["estimate_batch"] = int(current.shape[0])
                captured["estimate_training"] = bool(self.training)
                return torch.zeros_like(current)
            captured["model"] = timesteps
            captured["graph_batch"] = int(current.shape[0])
            captured["graph_training"] = bool(self.training)
            captured["relation_source"] = kwargs.get("relation_source")
            raise StopAfterModel

    values = d2ae.synthetic_inputs(batch=8)
    batch = {
        "x": values["current"],
        "text_embedding": values["text"],
        "object_bps": values["global_bps"],
        "goals": values["goals"],
        "progress": values["progress"],
        **_relation_arguments(values),
    }
    observations: List[Dict[str, object]] = []

    def observer(stage: str, selected: int) -> None:
        observations.append({"stage": stage, "selected_count": int(selected)})

    def one_step(*, processed_windows: int, rank: int) -> Dict[str, object]:
        stub = ModelStub().train()
        observations.clear()
        captured.clear()
        torch.manual_seed(1234)
        state_before = torch.get_rng_state().clone()
        try:
            _forward_losses(
                stub,
                DiffusionStub(),
                batch,
                torch.arange(22, dtype=torch.long),
                values["position_minimum"],
                values["position_maximum"],
                values["object_minimum"],
                values["object_maximum"],
                cfg,
                processed_windows=processed_windows,
                rank=rank,
                selfcond_observer=observer,
            )
        except StopAfterModel:
            pass
        return {
            "timesteps": captured["q_sample"].clone(),
            "noise_sha256": d2ae.tensor_sha256(captured["noise"]),
            "state_before_sha256": d2ae.tensor_sha256(state_before),
            "state_after_sha256": d2ae.tensor_sha256(torch.get_rng_state()),
            "estimate_batch": captured.get("estimate_batch"),
            "estimate_training": captured.get("estimate_training"),
            "graph_training": captured.get("graph_training"),
            "same_timestep_object": captured["q_sample"] is captured["model"],
            "observations": list(observations),
            "module_training_after": bool(stub.training),
            "calls": stub.calls,
        }

    baseline = one_step(processed_windows=0, rank=0)
    repeat = one_step(processed_windows=0, rank=0)
    other_rank = one_step(processed_windows=0, rank=1)
    later = one_step(processed_windows=2048, rank=0)
    masks = {
        f"windows{windows}_rank{rank}": _d2ag_selection_mask(
            cfg, 512, torch.device("cpu"), windows, rank,
        )
        for windows, rank in ((0, 0), (0, 1), (2048, 0), (2048, 1))
    }
    fractions = {
        name: float(mask.float().mean()) for name, mask in masks.items()
    }
    seeds = {
        f"windows{windows}_rank{rank}": _d2ag_mask_seed(cfg, windows, rank)
        for windows, rank in ((0, 0), (0, 1), (2048, 0), (2048, 1))
    }
    distinct_masks = len({
        d2ae.tensor_sha256(mask) for mask in masks.values()
    })
    checks = {
        "q_sample_and_model_share_the_timestep_tensor": (
            baseline["same_timestep_object"] is True
        ),
        "global_rng_independent_of_mask_across_ranks": (
            baseline["noise_sha256"] == other_rank["noise_sha256"]
            and torch.equal(baseline["timesteps"], other_rank["timesteps"])
        ),
        "global_rng_independent_of_mask_across_windows": (
            baseline["noise_sha256"] == later["noise_sha256"]
            and torch.equal(baseline["timesteps"], later["timesteps"])
        ),
        "global_rng_consumption_is_deterministic": (
            baseline["state_after_sha256"] == repeat["state_after_sha256"]
            == other_rank["state_after_sha256"] == later["state_after_sha256"]
        ),
        "masks_differ_across_windows_and_ranks": distinct_masks == 4,
        "mask_seed_derivation_registered": seeds == {
            "windows0_rank0": 42 * 1_000_003,
            "windows0_rank1": 42 * 1_000_003 + 1,
            "windows2048_rank0": 42 * 1_000_003 + 2048,
            "windows2048_rank1": 42 * 1_000_003 + 2049,
        },
        "mask_probability_is_half": all(
            abs(value - D2AG_SELF_CONDITION_PROBABILITY) < 0.10
            for value in fractions.values()
        ),
        "estimate_pass_runs_in_eval": baseline["estimate_training"] is False,
        "graph_pass_runs_in_train": baseline["graph_training"] is True,
        "training_flag_restored": baseline["module_training_after"] is True,
        "estimate_pass_uses_selected_subset_only": (
            baseline["estimate_batch"] is not None
            and 0 < int(baseline["estimate_batch"]) < 8
        ),
        "observer_brackets_one_pass": [
            row["stage"] for row in baseline["observations"]
        ] == ["begin", "end"],
        "two_forwards_per_update": baseline["calls"] == 2,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: training RNG contract failed: {checks}"
        )
    return {
        "mask_seed_derivation": "cfg.seed * 1000003 + processed_windows + rank",
        "mask_seeds": seeds,
        "mask_selected_fractions": fractions,
        "selection_probability": D2AG_SELF_CONDITION_PROBABILITY,
        "training_timestep_shape": list(baseline["timesteps"].shape),
        "training_timestep_dtype": str(baseline["timesteps"].dtype),
        "global_noise_sha256_by_case": {
            "windows0_rank0": baseline["noise_sha256"],
            "windows0_rank1": other_rank["noise_sha256"],
            "windows2048_rank0": later["noise_sha256"],
        },
        "estimate_selected_batch_by_case": {
            "windows0_rank0": baseline["estimate_batch"],
            "windows0_rank1": other_rank["estimate_batch"],
            "windows2048_rank0": later["estimate_batch"],
        },
        "checks": checks,
    }


def checkpoint_rejection_contract() -> Dict[str, object]:
    """Fail-closed loader provenance in both directions."""
    common = {
        "checkpoint_type": "hoi_prior_phase1b",
        "expert": "hoi",
        "initialization": "random",
        "seed": 42,
    }

    def model_config(variant: str) -> Dict[str, object]:
        return {
            "dim_model": 512,
            "num_heads": 16,
            "num_layers": 8,
            "architecture_variant": variant,
        }

    variants = {
        "released": {"model": {}},
        "d2x_base": {
            **common,
            "model_config": model_config(HOI_ARCHITECTURE_BASE),
            "architecture_variant": HOI_ARCHITECTURE_BASE,
        },
        "d2ac": {
            **common,
            "model_config": model_config(HOI_ARCHITECTURE_D2AC),
            "architecture_variant": HOI_ARCHITECTURE_D2AC,
            "interaction_adapter_contract": {},
        },
        "d2ad": {
            **common,
            "model_config": model_config(HOI_ARCHITECTURE_D2AD),
            "architecture_variant": HOI_ARCHITECTURE_D2AD,
            "interaction_adapter_contract": {},
        },
        "d2ae": {
            **common,
            "model_config": model_config(HOI_ARCHITECTURE_D2AE),
            "architecture_variant": HOI_ARCHITECTURE_D2AE,
            "sparse_relation_contract": sparse_relation_contract_metadata(),
        },
        "d2af": {
            **common,
            "model_config": model_config(HOI_ARCHITECTURE_D2AF),
            "architecture_variant": HOI_ARCHITECTURE_D2AF,
            "diffusion_reliability_contract": (
                diffusion_reliability_contract_metadata()
            ),
        },
        "d2ag_missing_contract": {
            **common,
            "model_config": model_config(HOI_ARCHITECTURE_D2AG),
            "architecture_variant": HOI_ARCHITECTURE_D2AG,
        },
        "d2ag_partial_contract": {
            **common,
            "model_config": model_config(HOI_ARCHITECTURE_D2AG),
            "architecture_variant": HOI_ARCHITECTURE_D2AG,
            "selfcond_relation_source_contract": {
                key: value
                for key, value in
                selfcond_relation_source_contract_metadata().items()
                if key != "variable_anchor_source"
            },
        },
        "d2ag_altered_probability": {
            **common,
            "model_config": model_config(HOI_ARCHITECTURE_D2AG),
            "architecture_variant": HOI_ARCHITECTURE_D2AG,
            "selfcond_relation_source_contract": {
                **selfcond_relation_source_contract_metadata(),
                "selection_probability": 0.25,
            },
        },
    }
    rejected = {}
    reverse = {}
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
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AG,
                )
            except (ValueError, RuntimeError) as error:
                rejected[label] = {"rejected": True, "error": str(error)}
            else:
                rejected[label] = {"rejected": False, "error": None}

        reverse_path = directory / "d2ag_reverse.pth"
        torch.save({
            **common,
            "model_config": model_config(HOI_ARCHITECTURE_D2AG),
            "architecture_variant": HOI_ARCHITECTURE_D2AG,
            "selfcond_relation_source_contract": (
                selfcond_relation_source_contract_metadata()
            ),
        }, reverse_path)
        for label, variant in (
            ("d2ae_rejects_d2ag", HOI_ARCHITECTURE_D2AE),
            ("d2af_rejects_d2ag", HOI_ARCHITECTURE_D2AF),
            ("base_rejects_d2ag", HOI_ARCHITECTURE_BASE),
        ):
            try:
                load_trained_hoi_prior(
                    str(reverse_path),
                    torch.device("cpu"),
                    use_ema=False,
                    expected_architecture_variant=variant,
                )
            except (ValueError, RuntimeError) as error:
                reverse[label] = {"rejected": True, "error": str(error)}
            else:
                reverse[label] = {"rejected": False, "error": None}
    if (
        not all(value["rejected"] for value in rejected.values())
        or not all(value["rejected"] for value in reverse.values())
    ):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: checkpoint rejection failed: "
            f"{rejected}; {reverse}"
        )
    return {
        "d2ag_rejects": rejected,
        "predecessors_reject_d2ag": reverse,
        "scientific_checkpoint_loads": 0,
        "synthetic_checkpoint_attempts": len(rejected) + len(reverse),
    }


def static_contract(repo: Path) -> Dict[str, object]:
    """Forbidden-source static scan over the D2-AG runtime path."""
    sparse_path = repo / "code/priors/sparse_relation.py"
    trainer_path = repo / "code/train_hoi_prior.py"
    diffusion_path = repo / "code/priors/diffusion.py"
    models_path = repo / "code/priors/models.py"
    sparse_source = sparse_path.read_text(encoding="utf-8")
    trainer_source = trainer_path.read_text(encoding="utf-8")
    diffusion_source = diffusion_path.read_text(encoding="utf-8")
    field_source = inspect.getsource(
        SparseCurrentStateRelationField.forward
    ).lower()
    sampler_source = inspect.getsource(GaussianDiffusion.sample)
    builder_source = inspect.getsource(build_d2ag_relation_source)
    sparse_tree = ast.parse(sparse_source)
    imported = set()
    for node in ast.walk(sparse_tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
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
            "stored_relation",
            "per_anchor",
        )
        if token in field_source
    ]
    history_pin_sites = sparse_source.count(
        "torch.cat((current[:, :history], source[:, history:]), dim=1)"
    )
    checks = {
        "pure_torch_field": not forbidden_imports,
        "forbidden_field_sources_absent": not forbidden_field_tokens,
        "single_shared_source_builder": (
            "build_d2ag_relation_source" in trainer_source
            and "build_d2ag_relation_source" in diffusion_source
        ),
        "history_pin_only_in_shared_builder": (
            history_pin_sites == 1
            and "current[:, :history]" in builder_source
        ),
        "sampler_prev_x0_is_raw": (
            "prev_x0 = clean.detach()" in sampler_source
            and sampler_source.index("prev_x0 = clean.detach()")
            < sampler_source.index("clean = prepare_clean_x0(")
        ),
        "sampler_prev_x0_starts_none": (
            "prev_x0: Optional[torch.Tensor] = None" in sampler_source
        ),
        "sampler_future_gt_absent": "future_gt" not in sampler_source,
        "sampler_scene_absent": "Scene" not in sampler_source,
        "sampler_stored_relation_absent": (
            "stored_relation" not in sampler_source
        ),
        "sampler_cpu_dynamic_geometry_absent": all(
            token not in sampler_source
            for token in ("cKDTree", "scipy", "cdist", "full_mesh")
        ),
        "no_loss_or_snr_weighting": (
            "d2ag_timestep_loss_weight" not in trainer_source
            and "d2ag_snr_weight" not in trainer_source
        ),
        "single_writeback": (
            inspect.getsource(SparseCurrentStateRelationField.forward).count(
                "return motion + attenuated_writeback"
            ) == 1
        ),
        "no_relation_zero_branch_for_selfcond": (
            "relation_zero" not in field_source
        ),
        "probability_is_a_module_constant": (
            "D2AG_SELF_CONDITION_PROBABILITY = 0.5" in sparse_source
        ),
        "no_learned_or_scheduled_probability": all(
            token not in sparse_source and token not in trainer_source
            for token in (
                "nn.Parameter(D2AG",
                "probability_schedule",
                "learned_selection_probability",
            )
        ),
        "no_field_schedule_registration_for_selfcond": (
            "selfcond_relation_source=True" not in sparse_source
            or "register_buffer(\"sqrt_alpha_bar\"" in sparse_source
        ),
        "estimate_flag_is_plain_attribute": (
            "self.relation_source_estimate = False" in sparse_source
            and "register_buffer(\"relation_source_estimate\""
            not in sparse_source
        ),
        "models_forwards_timesteps_for_d2af_and_d2ag": (
            "HOI_ARCHITECTURE_D2AF," in models_path.read_text(encoding="utf-8")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{FAILURE_CLASSIFICATION}: static contract failed: {checks}"
        )
    return {
        "checks": checks,
        "forbidden_imports": forbidden_imports,
        "forbidden_field_tokens": forbidden_field_tokens,
        "history_pin_sites": history_pin_sites,
        "sparse_relation_source_sha256": sha256_file(sparse_path),
        "trainer_source_sha256": sha256_file(trainer_path),
        "diffusion_source_sha256": sha256_file(diffusion_path),
        "models_source_sha256": sha256_file(models_path),
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
    formal_source = _d2ag_formal_source_contract(repo)

    # Reuse the sealed D2-AE pure geometry, asset, finiteness, sampler-metadata,
    # evaluator-hash, HSIPrior-storage and clean-output contracts unchanged.
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
        "subphase": "1B-D2-AG0-cpu-contract",
        "seed": 42,
        "runtime_seconds": time.perf_counter() - started,
        "identity": identity,
        "formal_source_contract": formal_source,
        "resolved_formal_config": OmegaConf.to_container(cfg, resolve=True),
        "inherited_d2ae_contracts": inherited,
        "selfcond_relation_source_contract": (
            selfcond_relation_source_contract_metadata()
        ),
        "schedule_absence": schedule_absence_contract(),
        "relation_source": relation_source_contract(),
        "model": model_contract(),
        "diagnostic_gates": diagnostic_gate_contract(),
        "sampler": sampler_contract(),
        "training_rng_and_eval_mode": training_rng_contract(cfg),
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
    exclusive_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


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
                "D2-AG CPU contract requires a pre-archived resolved config"
            )
        if path.read_text(encoding="utf-8") != resolved:
            raise RuntimeError(
                "D2-AG CPU contract differs from archived resolved config"
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
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        ).strip()
    except Exception:
        commit = None
    return {
        "schema_version": 1,
        "status": "failed",
        "classification": FAILURE_CLASSIFICATION,
        "run_id": run_id,
        "subphase": "1B-D2-AG0-cpu-contract",
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
            raise RuntimeError("D2-AG CPU resolved config is incomplete")
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
            run_id=args.run_id, started=started, error=error, repo=repo,
        )
        failure_path = args.output.resolve().parent / "failure.json"
        if not failure_path.exists():
            exclusive_json(failure_path, failure)
        if not args.output.resolve().exists():
            exclusive_json(args.output, failure)
        print(f"{FAILURE_CLASSIFICATION}: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
