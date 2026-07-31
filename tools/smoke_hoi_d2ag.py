#!/usr/bin/env python3
"""Registered one-GPU, no-update real-data functional smoke for D2-AG0."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.diffusion_schedule import (  # noqa: E402
    SQRT_ALPHA_BAR_SHA256,
    tensor_sha256,
)
from priors.losses import hoi_training_losses  # noqa: E402
from priors.models import HOI_ARCHITECTURE_D2AG, build_expert  # noqa: E402
from priors.representation import REPRESENTATION  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    D2AG_SELF_CONDITION_PROBABILITY,
    D2AG_VARIABLE_ANCHORS,
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
    TOTAL_PARAMETER_COUNT,
    build_d2ag_relation_source,
    build_sparse_relation_geometry,
    selfcond_relation_source_contract_metadata,
)
from tools.smoke_hoi_d2ac import (  # noqa: E402
    _atomic_json,
    _gpu_contention,
    _gradient_record,
)
from tools.smoke_hoi_d2ae import (  # noqa: E402
    _atomic_text,
    _model_arguments,
    _sha256_file,
    _tensor_summary,
)
from train_hoi_prior import (  # noqa: E402
    LOSS_KEYS,
    _d2ag_formal_source_contract,
    _d2ag_relation_source_arguments,
    _d2ag_selection_mask,
    _d2ae_gradient_audit,
    _move_batch,
    _state_dict_sha256,
    _validate_author_update_execution_host,
    _validate_d2ag_contract,
    _validate_fk_foot_temporal_routing_mode,
)


RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ag-gpu-functional-smoke"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
EXPECTED_BATCH_SIZE = 8
REGISTERED_TIMESTEPS = (0, 249, 499, 0, 249, 499, 0, 499)
DISTINCT_TIMESTEPS = (0, 249, 499)
EXPECTED_INITIAL_MODEL_SHA256 = (
    "b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c"
)
PARITY_MAX_ABS_TOLERANCE = 1.0e-6
# Batch-of-1 versus batch-of-N parity is *not* a registered contract; the plan
# never states it.  Its floor is pure cuBLAS GEMM-shape behavior: measured on
# infbagel-4gpu/RTX 3090 (torch 1.13.1+cu117) a *pure D2-AE* variant with no
# self-conditioning at all gives 7.152557373046875e-07 for the identical
# comparison, and eval-mode D2-AG gives 6.55e-07/7.15e-07.  1e-6 leaves only a
# 1.4x margin over that floor, so this gate is set to 1e-5 (~14x the measured
# floor).  It keeps full discriminating power: a genuine per-sample selection
# failure moves a row by ``selected_min_l2`` ~ 7.1, six orders of magnitude up.
BATCH_SHAPE_PARITY_MAX_ABS_TOLERANCE = 1.0e-5
# The parity comparisons run with the gate forced to this value.  At the
# registered ``alpha=0`` initialization ``tanh(alpha)`` is exactly zero
# (``code/priors/sparse_relation.py:579``, ``:811``, ``:814``), so the writeback
# vanishes and the relation source cannot move the output at all; comparing
# sources under a zero gate is vacuous.  ``tools/diagnose_hoi_d2ag.py:613-616``
# makes the same point for the CPU gate.
PROBE_PARITY_GATE = 0.1
# Fixed descriptive mask for the per-sample source-selection probe.  It is not
# the registered training mask; the registered Bernoulli draw is exercised
# separately through ``_d2ag_selection_mask``.
PROBE_SELECTION_MASK = (True, False, True, False, True, False, True, False)
FAILURE_CLASSIFICATION = "selfcond-relation-source-contract-failure-stop"


def _validate_actual_run_id(run_id: str) -> str:
    match = RUN_ID_RE.fullmatch(str(run_id))
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if match is None or match.group("date") != actual_date:
        raise ValueError(
            "D2-AG functional smoke run id must use the locked stem and actual date"
        )
    return match.group("date")


def _formal_run_id_for_date(date: str) -> str:
    return f"p1-hoi-d2ag-selfcond-relation-source-s42-{date}"


def _resolved_config(repo: Path, formal_run_id: str):
    cfg = OmegaConf.merge(
        OmegaConf.load(repo / "code/config/config_train_hoi_prior.yaml"),
        OmegaConf.load(repo / "code/config/config_train_hoi_prior_d2ag.yaml"),
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


def _resolved_workload_config(
    cfg,
    *,
    repo: Path,
    run_id: str,
    expected_commit: str,
    formal_source_contract: dict,
    output: Path,
    resolved_config_output: Path,
) -> str:
    contract = selfcond_relation_source_contract_metadata()
    value = {
        "schema_version": 1,
        "lifecycle": "d2ag_single_gpu_functional_smoke",
        "run_id": run_id,
        "formal_run_id": str(cfg.run_id),
        "expected_git_commit": expected_commit,
        "formal_source_contract": formal_source_contract,
        "repo_root": str(repo),
        "python": str(Path(sys.executable).resolve()),
        "output": str(output.resolve()),
        "resolved_config_output": str(resolved_config_output.resolve()),
        "workload": {
            "host": "node01",
            "visible_gpus": 1,
            "device": "cuda:0",
            "gpu_model": "RTX 3090",
            "real_data": True,
            "partition": "train",
            "batch_size": EXPECTED_BATCH_SIZE,
            "timesteps": list(DISTINCT_TIMESTEPS),
            "mixed_batch_timesteps": list(REGISTERED_TIMESTEPS),
            "seed": 42,
            "random_initialization": True,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "checkpoint_loads": 0,
            "checkpoint_writes": 0,
            "variable_anchor_source": "detached_model_x0_hat",
            "unselected_variable_anchor_source": "current_noisy_state",
            "history_anchor_source": "current_noisy_state",
            "self_condition_probability": D2AG_SELF_CONDITION_PROBABILITY,
            "relation_field_always_active": True,
            "sqrt_alpha_bar_attenuation": False,
        },
        "contracts": {
            "architecture_variant": HOI_ARCHITECTURE_D2AG,
            "expected_initial_model_state_sha256": EXPECTED_INITIAL_MODEL_SHA256,
            "selfcond_relation_source_contract": contract,
            "parity_max_abs_tolerance": PARITY_MAX_ABS_TOLERANCE,
            "per_timestep_gradient_audit": list(DISTINCT_TIMESTEPS),
            "probe_selection_mask": list(PROBE_SELECTION_MASK),
        },
        "formal_training_config_reference": OmegaConf.to_container(
            cfg, resolve=True,
        ),
    }
    resolved = OmegaConf.to_yaml(OmegaConf.create(value), resolve=True)
    if "${" in resolved:
        raise RuntimeError("D2-AG functional smoke resolved config is incomplete")
    return resolved


def _verify_worker(repo: Path, expected_commit: str) -> dict:
    if socket.gethostname() != "node01":
        raise RuntimeError("D2-AG functional smoke is restricted to node01")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-AG functional smoke requires INFBAGEL_WORKER_EXPERT=hoi")
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    if not configured_python or not Path(configured_python).is_absolute():
        raise RuntimeError("D2-AG functional smoke requires an absolute INFBAGEL_PYTHON")
    if Path(sys.executable).resolve() != Path(configured_python).resolve():
        raise RuntimeError(
            f"worker Python mismatch: {sys.executable} != {configured_python}"
        )
    root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=repo, text=True,
    ).strip()).resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        text=True,
    ).splitlines()
    if root != repo or commit != expected_commit or status:
        raise RuntimeError(
            f"D2-AG worker Git identity mismatch: root={root}, commit={commit}, "
            f"status={status[:20]}"
        )
    return {
        "repo_root": str(root),
        "git_commit": commit,
        "worktree_clean": True,
        "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _losses(
    prediction: torch.Tensor,
    clean: torch.Tensor,
    batch: dict,
    parents: torch.Tensor,
    cfg,
) -> dict:
    return hoi_training_losses(
        prediction,
        clean,
        batch["goals"],
        batch["rest_human_offsets"],
        parents,
        batch["position_minimum"],
        batch["position_maximum"],
        batch["object_minimum"],
        batch["object_maximum"],
        batch["terminal_window"],
        batch["rest_object_points"],
        batch["world_to_local_rotation"],
        batch["object_rotation_reference"],
        fk_weight=float(cfg.fk_weight),
        object_surface_weight=float(cfg.object_surface_weight),
        velocity_weight=float(cfg.velocity_weight),
        goal_weight=float(cfg.goal_weight),
        fk_foot_temporal_routing=True,
        routed_foot_residual_multiplier=1.0,
    )


def _forward(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    batch: dict,
    *,
    relation_source: torch.Tensor = None,
) -> torch.Tensor:
    extra = {} if relation_source is None else {"relation_source": relation_source}
    return model(
        noisy,
        timesteps,
        batch["text_embedding"],
        batch["object_bps"],
        batch["goals"],
        normalize_progress(batch["progress"]),
        **_model_arguments(batch, noisy),
        **extra,
    )


def _activated_gradient_probe(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: int,
    batch: dict,
    parents: torch.Tensor,
    cfg,
    *,
    probe_mask: torch.Tensor,
) -> dict:
    """One activated-gate probe with the registered partial source selection."""
    model.zero_grad(set_to_none=True)
    timesteps = torch.full(
        (clean.shape[0],), timestep, device=clean.device, dtype=torch.long,
    )
    noisy = diffusion.q_sample(clean, timesteps, noise)
    index = probe_mask.nonzero(as_tuple=True)[0]
    with torch.no_grad():
        estimate = _forward(
            model.eval(), noisy.index_select(0, index),
            timesteps.index_select(0, index),
            {
                key: (
                    value.index_select(0, index)
                    if torch.is_tensor(value) and value.shape[:1] == noisy.shape[:1]
                    else value
                )
                for key, value in batch.items()
            },
        )
    model.train()
    relation_source = build_d2ag_relation_source(noisy, estimate, index=index)
    prediction = _forward(
        model, noisy, timesteps, batch, relation_source=relation_source,
    )
    losses = _losses(prediction, clean, batch, parents, cfg)
    values = {key: float(losses[key].detach().item()) for key in LOSS_KEYS}
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError(
            f"non-finite D2-AG activated probe loss at timestep {timestep}"
        )
    losses["total"].backward()
    torch.cuda.synchronize(clean.device)
    return {
        "timestep": timestep,
        "selected_rows": index.tolist(),
        "losses": values,
        "gradients": _d2ae_gradient_audit(model, require_relation_paths=True),
    }


def _estimator_source_contract(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    batch: dict,
    device: torch.device,
    probe_mask: torch.Tensor,
) -> dict:
    """Per-sample source selection, history pin, RNG and eval-mode contract.

    The registered assertions (EP:7031-7043) are: unselected rows must equal the
    D2-AE-style ``x_t``-source forward (the field stays active, so the reference
    is *not* the bare trunk), selected rows must differ, ``relation_source[:, :2]``
    must be exactly ``noisy[:, :2]``, and the estimate pass must consume no global
    RNG while running under ``eval()`` and restoring the prior mode.

    Two properties of the measurement, both established on the worker GPU rather
    than assumed:

    * The numerical parity comparisons must run under ``eval()``.  The ambient
      mode here is ``train()`` (``main`` at :646) and the trunk carries an active
      ``dropout=0.1`` (``code/priors/models.py:126``), so every forward draws its
      own dropout mask.  Measured on infbagel-4gpu/RTX 3090, running the
      *identical* reference forward twice in train mode differs by 0.762 max abs,
      which swamps the 1e-6 contract and made the comparison meaningless.  Under
      ``eval()`` the same unselected comparison is exactly 0.0.
    * They must run with an activated gate.  At the registered ``alpha=0``
      initialization ``tanh(alpha)`` is exactly zero, so the writeback vanishes
      and the relation source cannot move the output; unselected equality would
      then hold trivially even if per-sample selection were broken, and
      ``selected_differs_from_d2ae`` would be passing on dropout noise alone
      (measured ``selected_min_l2`` 9.18 in train mode versus 0.0 in eval mode at
      zero gate).  With the gate forced to ``PROBE_PARITY_GATE`` the selected
      rows move by ``selected_min_l2`` ~ 7.1 while unselected rows stay bitwise
      identical, so the check now has real discriminating power.

    The estimate pass itself deliberately stays in the ambient ``train()`` mode so
    that the registered eval-enter/restore, RNG and detach assertions remain
    meaningful; only the parity comparisons are moved into the deterministic
    context.
    """
    field = model.network.sparse_relation_field
    index = probe_mask.nonzero(as_tuple=True)[0]
    observed_training: list = []
    observed_estimate_flag: list = []

    def watch(module, _inputs) -> None:
        observed_training.append(bool(model.network.training))
        observed_estimate_flag.append(bool(module.relation_source_estimate))

    handle = field.register_forward_pre_hook(watch)
    try:
        # Pass 0: the ambient graph pass, in train mode with the estimate flag
        # clear.  Its output is not compared numerically; the deterministic
        # reference below is.
        with torch.no_grad():
            _forward(model, noisy, timesteps, batch)
        cpu_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(device).clone()
        was_training = model.network.training
        inner = model.network
        inner_module = model
        estimate_batch = {
            key: (
                value.index_select(0, index)
                if torch.is_tensor(value) and value.shape[:1] == noisy.shape[:1]
                else value
            )
            for key, value in batch.items()
        }
        inner_module.eval()
        try:
            with torch.no_grad():
                estimate = model(
                    noisy.index_select(0, index),
                    timesteps.index_select(0, index),
                    estimate_batch["text_embedding"],
                    estimate_batch["object_bps"],
                    estimate_batch["goals"],
                    normalize_progress(estimate_batch["progress"]),
                    **_model_arguments(estimate_batch, noisy),
                    relation_source_estimate=True,
                )
        finally:
            inner_module.train(was_training)
        cpu_after = torch.get_rng_state().clone()
        cuda_after = torch.cuda.get_rng_state(device).clone()
        relation_source = build_d2ag_relation_source(noisy, estimate, index=index)
        parity_was_training = model.network.training
        parity_gate = PROBE_PARITY_GATE
        inner_module.eval()
        field.set_gate_override(parity_gate)
        try:
            with torch.no_grad():
                reference = _forward(model, noisy, timesteps, batch)
                # Same inputs, same shapes: the instrument's own floor.  Any
                # non-zero value here means the parity context is not
                # deterministic and no comparison below can be trusted.
                reference_repeat = _forward(model, noisy, timesteps, batch)
                selected = _forward(
                    model, noisy, timesteps, batch,
                    relation_source=relation_source,
                )
                single_rows = [
                    _forward(
                        model,
                        noisy[row: row + 1],
                        timesteps[row: row + 1],
                        {
                            key: (
                                value[row: row + 1]
                                if torch.is_tensor(value)
                                and value.shape[:1] == noisy.shape[:1]
                                else value
                            )
                            for key, value in batch.items()
                        },
                        relation_source=relation_source[row: row + 1],
                    )
                    for row in index.tolist()
                ]
        finally:
            field.set_gate_override(None)
            inner_module.train(parity_was_training)
    finally:
        handle.remove()
    unselected = (~probe_mask).nonzero(as_tuple=True)[0]
    reference_run_to_run_max_abs = float(
        (reference_repeat - reference).abs().amax().item()
    )
    unselected_max_abs = float(
        (selected.index_select(0, unselected) - reference.index_select(0, unselected))
        .abs().amax().item()
    ) if unselected.numel() else 0.0
    selected_min_l2 = float(
        (selected.index_select(0, index) - reference.index_select(0, index))
        .flatten(1).norm(dim=1).amin().item()
    ) if index.numel() else 0.0
    per_row_max_abs = max(
        [
            float((single - selected[row: row + 1]).abs().amax().item())
            for single, row in zip(single_rows, index.tolist())
        ] or [0.0]
    )
    history = REPRESENTATION.history_frames
    history_max_abs = float(
        (relation_source[:, :history] - noisy[:, :history]).abs().amax().item()
    )
    with torch.no_grad():
        anchor_zero_source = build_sparse_relation_geometry(
            relation_source, **_model_arguments(batch, noisy),
        )["features"][:, 0]
        anchor_zero_current = build_sparse_relation_geometry(
            noisy, **_model_arguments(batch, noisy),
        )["features"][:, 0]
    anchor_zero_max_abs = float(
        (anchor_zero_source - anchor_zero_current).abs().amax().item()
    )
    variable_delta_l2 = float(
        (
            relation_source[:, list(D2AG_VARIABLE_ANCHORS)]
            - noisy[:, list(D2AG_VARIABLE_ANCHORS)]
        ).flatten(1).norm(dim=1).sum().item()
    )
    checks = {
        # Reference and mixed-source forwards use the *same* batch shape, so every
        # Linear/attention/LayerNorm op is row-independent and an unselected row
        # cannot be perturbed by another row's data.  Exact equality is therefore
        # structural, and it is what the GPU measures (0.0).  EP:7035-7036 only
        # requires max abs <= 1e-6, so the bitwise form is strictly stronger; both
        # are asserted so that a future 1e-7-scale drift is diagnosable instead of
        # failing under a name that overstates the registered contract.
        "unselected_equals_d2ae_bitwise": unselected_max_abs == 0.0,
        "unselected_equals_d2ae_within_registered_tolerance": (
            unselected_max_abs <= PARITY_MAX_ABS_TOLERANCE
        ),
        "selected_differs_from_d2ae": selected_min_l2 > 0.0,
        # Renamed from ``selection_is_per_sample``: this compares batch-of-1 with
        # batch-of-N, which changes the GEMM shape and hence the accumulation
        # order, so it measures cuBLAS batch-shape behavior rather than any
        # registered D2-AG property.  See BATCH_SHAPE_PARITY_MAX_ABS_TOLERANCE for
        # the measured floor and the tolerance rationale.
        "selection_is_per_sample_within_batch_shape_tolerance": (
            per_row_max_abs <= BATCH_SHAPE_PARITY_MAX_ABS_TOLERANCE
        ),
        # Self-check on the instrument: the parity context must be deterministic,
        # otherwise every comparison above is measuring dropout noise instead of
        # the mechanism.  This fails loudly if the parity block ever loses its
        # ``eval()`` bracket.
        "parity_reference_is_deterministic": reference_run_to_run_max_abs == 0.0,
        "parity_gate_activated": parity_gate > 0.0,
        "history_pin_exact_zero": history_max_abs == 0.0,
        "anchor_zero_geometry_exact": anchor_zero_max_abs == 0.0,
        "variable_anchors_moved": variable_delta_l2 > 0.0,
        "estimate_consumes_no_cpu_rng": bool(torch.equal(cpu_before, cpu_after)),
        "estimate_consumes_no_cuda_rng": bool(
            torch.equal(cuda_before, cuda_after)
        ),
        "estimate_ran_in_eval": observed_training[1:2] == [False],
        "estimate_flag_observed_by_hook": observed_estimate_flag[:2]
        == [False, True],
        "exactly_one_estimate_forward": sum(observed_estimate_flag) == 1,
        "module_restored_to_train": bool(model.network.training),
        "estimate_detached": estimate.grad_fn is None
        and not estimate.requires_grad,
        "relation_source_detached": relation_source.grad_fn is None
        and not relation_source.requires_grad,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            f"D2-AG per-sample source contract failed: {failed}"
        )
    return {
        "probe_selection_mask": probe_mask.tolist(),
        "selected_rows": index.tolist(),
        "unselected_rows": unselected.tolist(),
        "selected_count": int(index.numel()),
        "unselected_vs_d2ae_max_abs": unselected_max_abs,
        "selected_vs_d2ae_min_l2": selected_min_l2,
        "single_row_parity_max_abs": per_row_max_abs,
        "single_row_parity_tolerance": BATCH_SHAPE_PARITY_MAX_ABS_TOLERANCE,
        "registered_parity_tolerance": PARITY_MAX_ABS_TOLERANCE,
        "parity_reference_run_to_run_max_abs": reference_run_to_run_max_abs,
        "parity_context": "eval_mode_gate_activated",
        "parity_gate": parity_gate,
        "history_pin_max_abs": history_max_abs,
        "anchor_zero_geometry_max_abs": anchor_zero_max_abs,
        "variable_anchor_delta_l2_sum": variable_delta_l2,
        "estimate_forward_count": int(sum(observed_estimate_flag)),
        "estimate_ran_in_eval": True,
        "module_restored_to_train": True,
        "field_pass_estimate_flags": observed_estimate_flag,
        "field_pass_training_flags": observed_training,
        "estimate_summary": _tensor_summary(estimate),
        "relation_source_summary": _tensor_summary(relation_source),
        "checks": checks,
    }


# Tokens are assembled at runtime so that naming them in a check does not make
# the check's own module text a false positive.
_SCHEDULE_ATTRIBUTE_TOKEN = "field." + "sqrt_alpha" + "_bar"
_RHO_OVERRIDE_TOKEN = "set_rho" + "_override"
_NO_SYNC_CALL_TOKEN = "." + "no_sync" + "("


def schedule_dereference_sites(source: str) -> list:
    """Return every schedule-attribute use on the field that is not a None guard.

    D2-AG never registers the D2-AF reliability schedule buffer, so that field
    attribute is always ``None``.  Asserting its absence is legitimate; reading
    it is an ``AttributeError`` waiting to happen (plan section 2.6-6).
    """
    sites = []
    start = 0
    while True:
        index = source.find(_SCHEDULE_ATTRIBUTE_TOKEN, start)
        if index < 0:
            return sites
        start = index + len(_SCHEDULE_ATTRIBUTE_TOKEN)
        tail = source[start: start + 16]
        if not (tail.startswith(" is None") or tail.startswith(" is not None")):
            sites.append(source[max(0, index - 40): start + 16])


def _static_source_contract() -> dict:
    """Cheap source-level assertions that need no CUDA."""
    estimator_source = inspect.getsource(_d2ag_relation_source_arguments)
    mask_source = inspect.getsource(_d2ag_selection_mask)
    smoke_source = Path(__file__).resolve().read_text(encoding="utf-8")
    dereferences = schedule_dereference_sites(smoke_source)
    checks = {
        "estimator_unwraps_ddp": "model.module" in estimator_source
        and "DistributedDataParallel" in estimator_source,
        "estimator_avoids_no_sync": _NO_SYNC_CALL_TOKEN not in estimator_source,
        "estimator_uses_eval_and_finally": "inner.eval()" in estimator_source
        and "finally" in estimator_source
        and "inner.train(was_training)" in estimator_source,
        "estimator_is_no_grad": "torch.no_grad()" in estimator_source,
        "mask_uses_dedicated_generator": "torch.Generator(" in mask_source
        and "generator=generator" in mask_source,
        "no_field_schedule_dereference": not dereferences,
        "no_rho_override_call": f"{_RHO_OVERRIDE_TOKEN}(" not in smoke_source,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            f"D2-AG static source contract failed: {failed}; "
            f"schedule_dereferences={dereferences[:3]}"
        )
    return {"checks": checks, "schedule_dereference_sites": dereferences}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-config-output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="archive and validate the exact workload config without touching CUDA",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    if args.batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError("registered D2-AG functional smoke batch size is exactly 8")
    run_date = _validate_actual_run_id(args.run_id)
    formal_run_id = _formal_run_id_for_date(run_date)

    identity = _verify_worker(repo, args.expected_commit)
    formal_source_contract = _d2ag_formal_source_contract(repo)
    cfg = _resolved_config(repo, formal_run_id)
    _validate_fk_foot_temporal_routing_mode(cfg)
    _validate_d2ag_contract(cfg, 4, require_performance_gate=False)
    _validate_author_update_execution_host(cfg)
    static_contract = _static_source_contract()
    resolved_yaml = _resolved_workload_config(
        cfg,
        repo=repo,
        run_id=args.run_id,
        expected_commit=args.expected_commit,
        formal_source_contract=formal_source_contract,
        output=args.output,
        resolved_config_output=args.resolved_config_output,
    )
    if args.resolve_only:
        _atomic_text(args.resolved_config_output, resolved_yaml)
    else:
        resolved_path = args.resolved_config_output.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(
                "D2-AG functional smoke requires a pre-archived resolved config"
            )
        if resolved_path.read_text(encoding="utf-8") != resolved_yaml:
            raise RuntimeError(
                "D2-AG functional smoke workload differs from archived config"
            )
    resolved_sha256 = _sha256_file(args.resolved_config_output)
    if args.resolve_only:
        print(json.dumps({
            "schema_version": 1,
            "status": "resolved-config-archived",
            "run_id": args.run_id,
            "resolved_config_path": str(args.resolved_config_output.resolve()),
            "resolved_config_sha256": resolved_sha256,
            "static_contract": static_contract,
            "gpu_workload_started": False,
        }, indent=2, sort_keys=True), flush=True)
        return 0

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "D2-AG functional smoke requires exactly one visible CUDA device"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if "RTX 3090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("D2-AG functional smoke requires an RTX 3090")
    contention_before = _gpu_contention()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = build_expert(
        "hoi",
        init_checkpoint=None,
        dim_model=int(cfg.dim_model),
        num_heads=int(cfg.num_heads),
        num_layers=int(cfg.num_layers),
        architecture_variant=HOI_ARCHITECTURE_D2AG,
    )
    initial_model_sha256 = _state_dict_sha256(model.state_dict())
    if initial_model_sha256 != EXPECTED_INITIAL_MODEL_SHA256:
        raise RuntimeError(
            "D2-AG seed-42 initial model-state hash differs from D2-AE/D2-AF"
        )
    if sum(parameter.numel() for parameter in model.parameters()) != TOTAL_PARAMETER_COUNT:
        raise RuntimeError("D2-AG total parameter count is not exact")
    field = model.network.sparse_relation_field
    if field.sqrt_alpha_bar is not None:
        raise RuntimeError("D2-AG must not register the D2-AF schedule buffer")
    model = model.to(device).train()
    field = model.network.sparse_relation_field

    dataset = PriorWindowDataset(
        str(repo),
        "hoi",
        partition="train",
        split_manifest=str(cfg.split_manifest),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    raw_batch = next(iter(loader))
    if "local_object_bps" in raw_batch:
        raise RuntimeError("D2-AG functional smoke received CPU dynamic geometry")
    batch = _move_batch(raw_batch, device)
    batch.update({
        "position_minimum": torch.as_tensor(
            dataset.minimum, device=device, dtype=torch.float32,
        ),
        "position_maximum": torch.as_tensor(
            dataset.maximum, device=device, dtype=torch.float32,
        ),
        "object_minimum": torch.as_tensor(
            dataset.object_minimum, device=device, dtype=torch.float32,
        ),
        "object_maximum": torch.as_tensor(
            dataset.object_maximum, device=device, dtype=torch.float32,
        ),
    })
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )
    diffusion = GaussianDiffusion(int(cfg.diffusion_steps)).to(device)
    clean = batch["x"]
    timesteps = torch.tensor(
        REGISTERED_TIMESTEPS, device=device, dtype=torch.long,
    )
    generator = torch.Generator(device=device).manual_seed(42)
    noise = torch.randn(clean.shape, device=device, generator=generator)
    noisy = diffusion.q_sample(clean, timesteps, noise)
    probe_mask = torch.tensor(
        PROBE_SELECTION_MASK, device=device, dtype=torch.bool,
    )

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        geometry = build_sparse_relation_geometry(
            noisy, **_model_arguments(batch, noisy),
        )
        encoded = field.point_encoder(geometry["features"])
        pooled = torch.cat(
            (encoded.mean(dim=-2), encoded.amax(dim=-2)), dim=-1,
        )
        relation = field._relation_vectors(pooled)
        routed = relation.index_select(1, field.routing_slots)
        motion = model.network.motion_input(noisy)
        field.set_gate_override(0.1)
        field.set_capture(True)
        current_motion = field(
            motion,
            noisy,
            **_model_arguments(batch, noisy),
            timesteps=timesteps,
        )
        runtime_snapshot = field.snapshot()
        # ``timesteps`` is diagnostic-only for the production ``full`` variant:
        # the field output must be bitwise identical without it.
        timestep_free_motion = field(
            motion,
            noisy,
            **_model_arguments(batch, noisy),
        )
        field.set_capture(False)
        field.set_gate_override(None)
        writeback = current_motion - motion
        timesteps_diagnostic_only_max_abs = float(
            (timestep_free_motion - current_motion).abs().amax().item()
        )

    relation_summaries = {
        "surface": _tensor_summary(geometry["surface"]),
        "features": _tensor_summary(geometry["features"]),
        "encoded_points": _tensor_summary(encoded),
        "pooled_blocks": _tensor_summary(pooled),
        "relation_vectors": _tensor_summary(relation),
        "routed_relation": _tensor_summary(routed),
        "writeback": _tensor_summary(writeback),
    }
    if (
        not all(value["finite"] for value in relation_summaries.values())
        or not all(
            str(value["device"]).startswith("cuda")
            for value in relation_summaries.values()
        )
    ):
        raise FloatingPointError(
            "D2-AG relation values are non-finite or not GPU-native"
        )
    if runtime_snapshot is None or not {
        "writeback_norm",
        "relation_source_minus_current_l2",
        "relation_source_history_max_abs",
        "relation_source_estimate",
        "relation_source_is_current",
    } <= set(runtime_snapshot):
        raise RuntimeError("D2-AG functional smoke did not capture source values")
    if timesteps_diagnostic_only_max_abs != 0.0:
        raise RuntimeError(
            "D2-AG production field output must not depend on timesteps"
        )

    source_selection = _estimator_source_contract(
        model, noisy, timesteps, batch, device, probe_mask,
    )
    registered_mask = _d2ag_selection_mask(cfg, args.batch_size, device, 0, 0)
    registered_mask_alternate = _d2ag_selection_mask(
        cfg, args.batch_size, device, int(cfg.effective_batch_size), 0,
    )
    if registered_mask.dtype != torch.bool or registered_mask.shape != (
        args.batch_size,
    ):
        raise RuntimeError("D2-AG registered selection mask shape is invalid")

    estimator_observations: list = []

    def observer(stage: str, selected: int) -> None:
        estimator_observations.append({
            "stage": stage,
            "selected_count": int(selected),
            "model_training": bool(model.network.training),
        })

    peak_before_estimator = int(torch.cuda.max_memory_allocated(device))
    prediction = _forward(model, noisy, timesteps, batch)
    prediction.retain_grad()
    losses = _losses(prediction, clean, batch, parents, cfg)
    loss_values = {
        key: float(losses[key].detach().item()) for key in LOSS_KEYS
    }
    if not all(math.isfinite(value) for value in loss_values.values()):
        raise FloatingPointError(f"non-finite D2-AG smoke loss: {loss_values}")
    losses["total"].backward()
    torch.cuda.synchronize(device)
    initial_audit = _d2ae_gradient_audit(model, require_relation_paths=False)
    initial_gradients = {
        "motion_input_weight": _gradient_record(
            model.network.motion_input.weight.grad,
        ),
        "transformer_first_parameter": _gradient_record(
            next(model.network.transformer.parameters()).grad,
        ),
        "prediction": _gradient_record(prediction.grad),
    }
    model.zero_grad(set_to_none=True)

    # One registered training-shaped step through the real trainer entry point,
    # so the smoke exercises the Bernoulli draw, the estimate forward and the
    # graph forward exactly as formal training will.
    training_step_mask = _d2ag_selection_mask(cfg, args.batch_size, device, 0, 0)
    training_losses = None
    with torch.enable_grad():
        from train_hoi_prior import _forward_losses  # noqa: PLC0415

        training_losses = _forward_losses(
            model,
            diffusion,
            batch,
            parents,
            batch["position_minimum"],
            batch["position_maximum"],
            batch["object_minimum"],
            batch["object_maximum"],
            cfg,
            generator=torch.Generator(device=device).manual_seed(42),
            processed_windows=0,
            rank=0,
            selfcond_observer=observer,
        )
    training_loss_values = {
        key: float(training_losses[key].detach().item()) for key in LOSS_KEYS
    }
    if not all(math.isfinite(value) for value in training_loss_values.values()):
        raise FloatingPointError(
            f"non-finite D2-AG trainer-path loss: {training_loss_values}"
        )
    training_losses["total"].backward()
    torch.cuda.synchronize(device)
    peak_after_estimator = int(torch.cuda.max_memory_allocated(device))
    trainer_path_audit = _d2ae_gradient_audit(model, require_relation_paths=False)
    model.zero_grad(set_to_none=True)
    if [row["stage"] for row in estimator_observations] != ["begin", "end"]:
        raise RuntimeError("D2-AG estimate observer did not bracket one pass")
    if estimator_observations[0]["selected_count"] != int(
        training_step_mask.sum().item()
    ):
        raise RuntimeError(
            "D2-AG observed selected count differs from the registered mask"
        )
    if not model.network.training:
        raise RuntimeError("D2-AG trainer path left the module in eval mode")

    with torch.no_grad():
        field.alpha.copy_(torch.atanh(torch.tensor(0.1, device=device)))
    per_timestep_activated_gradients = [
        _activated_gradient_probe(
            model,
            diffusion,
            clean,
            noise,
            timestep,
            batch,
            parents,
            cfg,
            probe_mask=probe_mask,
        )
        for timestep in DISTINCT_TIMESTEPS
    ]
    with torch.no_grad():
        field.alpha.zero_()
    model.zero_grad(set_to_none=True)

    diffusion_schedule_sha256 = tensor_sha256(diffusion.sqrt_alpha_bar)
    if diffusion_schedule_sha256 != SQRT_ALPHA_BAR_SHA256:
        raise RuntimeError("D2-AG diffusion schedule hash mismatch")

    total_memory = torch.cuda.get_device_properties(device).total_memory
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    contention_after = _gpu_contention()
    contract = selfcond_relation_source_contract_metadata()
    result = {
        "schema_version": 1,
        "status": "stable",
        "classification": "functional-smoke-passed",
        "run_id": args.run_id,
        "formal_run_id": formal_run_id,
        "subphase": "1B-D2-AG0-gpu-functional-smoke",
        "seed": 42,
        "identity": identity,
        "formal_source_contract": formal_source_contract,
        "resolved_config_path": str(args.resolved_config_output.resolve()),
        "resolved_config_sha256": resolved_sha256,
        "resolved_config_has_unresolved_interpolation": False,
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(device),
        "visible_cuda_devices": torch.cuda.device_count(),
        "batch_size": args.batch_size,
        "timesteps": list(DISTINCT_TIMESTEPS),
        "mixed_batch_timesteps": list(REGISTERED_TIMESTEPS),
        "variable_anchor_source": "detached_model_x0_hat",
        "unselected_variable_anchor_source": "current_noisy_state",
        "history_anchor_source": "current_noisy_state",
        "self_condition_probability": D2AG_SELF_CONDITION_PROBABILITY,
        "relation_field_always_active": True,
        "relation_zero_branch": False,
        "unselected_equals_d2ae_bitwise": bool(
            source_selection["checks"]["unselected_equals_d2ae_bitwise"]
        ),
        "first_reverse_step_source": "x_t",
        "first_reverse_step_matches_d2ae": True,
        "sqrt_alpha_bar_attenuation": False,
        "field_schedule_buffer_registered": False,
        "selfcond_relation_source_contract": contract,
        "relation_build_device": "cuda:0",
        "relation_gpu_only": True,
        "relation_intermediates": relation_summaries,
        "timesteps_diagnostic_only_max_abs": timesteps_diagnostic_only_max_abs,
        "per_sample_source": source_selection,
        "registered_selection_mask": {
            "seed_derivation": "cfg.seed * 1000003 + processed_windows + rank",
            "probability": D2AG_SELF_CONDITION_PROBABILITY,
            "mask_at_zero_windows": registered_mask.tolist(),
            "selected_count_at_zero_windows": int(registered_mask.sum().item()),
            "mask_at_one_update": registered_mask_alternate.tolist(),
            "masks_differ_across_processed_windows": not bool(
                torch.equal(registered_mask, registered_mask_alternate)
            ),
        },
        "trainer_path_step": {
            "losses": training_loss_values,
            "loss_finite": True,
            "estimator_observations": estimator_observations,
            "estimator_forward_count": 1,
            "estimator_selected_count": int(training_step_mask.sum().item()),
            "estimator_peak_memory_delta_bytes": (
                peak_after_estimator - peak_before_estimator
            ),
            "gradients": trainer_path_audit,
            "module_restored_to_train": True,
        },
        "runtime_snapshot": {
            key: value.tolist() for key, value in runtime_snapshot.items()
        },
        "diffusion_schedule_sha256": diffusion_schedule_sha256,
        "sqrt_alpha_bar_sha256": SQRT_ALPHA_BAR_SHA256,
        "losses": loss_values,
        "loss_finite": True,
        "initial_alpha_gradient": initial_audit,
        "initial_gradients": initial_gradients,
        "test_only_activated_gradients_by_timestep": (
            per_timestep_activated_gradients
        ),
        "test_only_gate": 0.1,
        "test_only_probe_saved": False,
        "static_contract": static_contract,
        "initialization": "random",
        "initial_model_state_sha256": initial_model_sha256,
        "expected_initial_model_state_sha256": EXPECTED_INITIAL_MODEL_SHA256,
        "total_parameter_count": TOTAL_PARAMETER_COUNT,
        "sparse_relation_parameter_count": SPARSE_RELATION_PARAMETER_COUNT,
        "sparse_point_mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
        "sparse_point_manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
        "sparse_point_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_loads": 0,
        "checkpoint_writes": 0,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": total_memory,
        "memory_headroom_bytes": total_memory - peak_reserved,
        "cuda_timing_synchronized": True,
        "contention_before": contention_before,
        "contention_after": contention_after,
        "checkpoint_selected": False,
        "formal_training_started": False,
        "consistency_started": False,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"{FAILURE_CLASSIFICATION}: {error}", file=sys.stderr, flush=True)
        raise
