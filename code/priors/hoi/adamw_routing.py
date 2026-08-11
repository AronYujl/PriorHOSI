"""Sealed AdamW counterfactual routing for the Phase 1B D2-K0 audit."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .gradient_clipping import (
    AUXILIARY_COMPONENTS,
    BASE_COMPONENTS,
    BLOCK_SIZE,
    FIELD_COMPONENTS,
    GATE_TIMESTEPS,
    GRADIENT_CLIP_NORM,
    LOSS_COMPONENTS,
    PRIMARY_WINDOWS,
    TIMESTEPS,
    clipping_replay,
)
from .gradient_routing import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CHECKPOINTS,
    PARAMETER_GROUPS,
    _cosine,
    _norm,
    _relative_l2,
    _sum_gradients,
)
from .remediation import selection_sha256, stable_digest


DIRECTIONS: Tuple[str, ...] = (
    "clipped_total",
    "historical_moment",
    "current_gradient",
    "weight_decay",
    "adamw_full",
)
EXPECTED_PRIMARY_SHA256 = "747c0b1c881e150a8ccdb8675044a877b1ab32f615169ea9e3577dcff0a3f90a"
EXPECTED_OPTIMIZER = {
    "R-1024": {"step": 6000, "lr": 1e-5, "initial_lr": 1e-4},
    "R-3072": {"step": 2000, "lr": 2.9999999999999997e-5, "initial_lr": 3e-4},
}


def select_fresh_primary(dataset) -> Dict[str, object]:
    """Scan from D0 global rank 768 and take 128 fresh nonterminal windows."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-K selection is internal-validation only")
    ranked = []
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        name = str(dataset.scene_names[sequence])
        pi = int(dataset.language["pi"][global_index])
        terminal = int(dataset.ends[global_index]) == int(dataset.seq_ends[sequence]) - 1
        ranked.append((
            stable_digest(f"42:hoi-remediation-window:{name}:{pi}"),
            name, pi, position, int(global_index), bool(terminal),
        ))
    ranked.sort()
    prior = {row[-2] for row in ranked[:768]}
    rows = []
    skipped_terminal_ranks = []
    selected_ranks = []
    for rank, row in enumerate(ranked[768:], start=768):
        if row[-1]:
            skipped_terminal_ranks.append(rank)
            continue
        rows.append(row)
        selected_ranks.append(rank)
        if len(rows) == PRIMARY_WINDOWS:
            break
    if len(rows) != PRIMARY_WINDOWS:
        raise ValueError("D2-K could not select 128 fresh nonterminal windows")
    positions = [row[-3] for row in rows]
    global_indices = [row[-2] for row in rows]
    if set(global_indices) & prior:
        raise AssertionError("D2-K selection overlaps D2-H0/D2-I0/D2-J0")
    return {
        "positions": positions,
        "global_indices": global_indices,
        "sha256": selection_sha256(global_indices),
        "terminal_windows": sum(int(row[-1]) for row in rows),
        "selected_ranks": selected_ranks,
        "skipped_terminal_ranks": skipped_terminal_ranks,
        "prior_global_indices": prior,
    }


def optimizer_state_sha256(optimizer_state: Mapping[str, object]) -> str:
    """Hash exact AdamW metadata, parameter ids and tensor state."""
    digest = hashlib.sha256()
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if not isinstance(groups, list) or not isinstance(states, dict):
        raise ValueError("invalid optimizer state mapping")
    for group in groups:
        metadata = {key: value for key, value in group.items() if key != "params"}
        digest.update(json.dumps(metadata, sort_keys=True, default=str).encode("utf-8"))
        digest.update(json.dumps(list(group["params"])).encode("ascii"))
    for parameter_id in sorted(states):
        digest.update(str(parameter_id).encode("ascii"))
        state = states[parameter_id]
        for key in sorted(state):
            digest.update(key.encode("utf-8"))
            value = state[key]
            if torch.is_tensor(value):
                tensor = value.detach().contiguous().cpu()
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def validate_optimizer_contract(
    checkpoint_name: str,
    optimizer_state: Mapping[str, object],
    parameters: Sequence[torch.nn.Parameter],
) -> Dict[str, object]:
    """Validate the single-group AdamW state and exact parameter ordering."""
    if checkpoint_name not in EXPECTED_OPTIMIZER:
        raise ValueError(f"unexpected D2-K checkpoint name: {checkpoint_name}")
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise ValueError("D2-K requires one complete AdamW parameter group")
    group = groups[0]
    parameter_ids = tuple(group.get("params", ()))
    if parameter_ids != tuple(range(len(parameters))):
        raise ValueError("D2-K optimizer parameter order is not exact sequential model order")
    if set(states) != set(parameter_ids) or len(states) != 119 or len(parameters) != 119:
        raise ValueError("D2-K optimizer state must cover all 119 model parameters")
    expected = EXPECTED_OPTIMIZER[checkpoint_name]
    expected_group = {
        "lr": expected["lr"], "initial_lr": expected["initial_lr"],
        "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01,
        "amsgrad": False, "maximize": False,
    }
    for key, value in expected_group.items():
        if group.get(key) != value:
            raise ValueError(f"D2-K optimizer {key} mismatch: {group.get(key)!r} != {value!r}")
    steps = set()
    for parameter_id, parameter in zip(parameter_ids, parameters):
        state = states[parameter_id]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("D2-K AdamW state keys are incomplete")
        if state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape:
            raise ValueError("D2-K AdamW moment shape does not match model parameter")
        steps.add(int(state["step"]))
    if steps != {expected["step"]}:
        raise ValueError(f"D2-K optimizer step mismatch: {steps}")
    return {
        "parameter_count": len(parameters),
        "state_count": len(states),
        "step": expected["step"],
        "next_step": expected["step"] + 1,
        "lr": float(group["lr"]),
        "initial_lr": float(group["initial_lr"]),
        "betas": list(group["betas"]),
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group["amsgrad"]),
        "maximize": bool(group["maximize"]),
    }


def mapped_optimizer_states(
    optimizer_state: Mapping[str, object],
    parameters: Sequence[torch.nn.Parameter],
) -> Tuple[Dict[str, object], ...]:
    """Copy sealed moments to each parameter device without mutating the checkpoint mapping."""
    group = optimizer_state["param_groups"][0]
    result = []
    for parameter_id, parameter in zip(group["params"], parameters):
        state = optimizer_state["state"][parameter_id]
        result.append({
            "step": int(state["step"]),
            "exp_avg": state["exp_avg"].detach().to(parameter.device).clone(),
            "exp_avg_sq": state["exp_avg_sq"].detach().to(parameter.device).clone(),
        })
    return tuple(result)


def mapped_state_sha256(states: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for state in states:
        digest.update(str(int(state["step"])).encode("ascii"))
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state[key].detach().contiguous().cpu()
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def adamw_directions(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
    states: Sequence[Mapping[str, object]],
    group: Mapping[str, object],
) -> Dict[str, object]:
    """Construct the exact next AdamW gradient-like descent direction without writes."""
    if not (len(gradients) == len(parameters) == len(states)):
        raise ValueError("D2-K AdamW inputs have inconsistent lengths")
    preclip_norm = _norm(gradients, tuple(range(len(gradients))))
    clipping = clipping_replay(preclip_norm, GRADIENT_CLIP_NORM)
    coefficient = clipping["clip_coefficient"]
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])
    clipped = []
    historical = []
    current = []
    decay = []
    direct_full = []
    decomposed_full = []
    with torch.no_grad():
        for gradient, parameter, state in zip(gradients, parameters, states):
            if gradient is None:
                clipped.append(None)
                historical.append(None)
                current.append(None)
                decay.append(None)
                direct_full.append(None)
                decomposed_full.append(None)
                continue
            gradient = gradient.detach()
            clipped_gradient = gradient * coefficient
            next_step = int(state["step"]) + 1
            first_correction = 1.0 - beta1 ** next_step
            second_correction = 1.0 - beta2 ** next_step
            historical_numerator = beta1 * state["exp_avg"]
            current_numerator = (1.0 - beta1) * clipped_gradient
            next_second = beta2 * state["exp_avg_sq"] + (1.0 - beta2) * clipped_gradient.square()
            denominator = (next_second / second_correction).sqrt().add(eps)
            historical_direction = (historical_numerator / first_correction) / denominator
            current_direction = (current_numerator / first_correction) / denominator
            decay_direction = weight_decay * parameter.detach()
            full = ((historical_numerator + current_numerator) / first_correction) / denominator
            full = full + decay_direction
            replay = historical_direction + current_direction + decay_direction
            clipped.append(clipped_gradient)
            historical.append(historical_direction)
            current.append(current_direction)
            decay.append(decay_direction)
            direct_full.append(full)
            decomposed_full.append(replay)
    return {
        "directions": {
            "clipped_total": tuple(clipped),
            "historical_moment": tuple(historical),
            "current_gradient": tuple(current),
            "weight_decay": tuple(decay),
            "adamw_full": tuple(direct_full),
        },
        "clipping": clipping,
        "adamw_decomposition_relative_l2": _relative_l2(decomposed_full, direct_full),
        "stored_lr": float(group["lr"]),
    }


def routing_geometry(
    base_gradients: Mapping[str, Sequence[torch.Tensor | None]],
    direct_total: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
    states: Sequence[Mapping[str, object]],
    optimizer_group: Mapping[str, object],
    parameter_groups: Mapping[str, Sequence[int]],
) -> Dict[str, object]:
    """Build all D2-K field gradients and clipped/AdamW direction routing records."""
    if tuple(base_gradients) != BASE_COMPONENTS:
        raise ValueError(f"expected {BASE_COMPONENTS}, got {tuple(base_gradients)}")
    gradients = dict(base_gradients)
    gradients["human_reconstruction"] = _sum_gradients(
        gradients, ("joint_position", "joint_rotation"),
    )
    gradients["object_reconstruction"] = _sum_gradients(
        gradients, ("object_translation", "object_rotation"),
    )
    gradients["reconstruction"] = _sum_gradients(
        gradients, ("human_reconstruction", "object_reconstruction", "contact"),
    )
    gradients["auxiliary_sum"] = _sum_gradients(gradients, AUXILIARY_COMPONENTS)
    replay_total = _sum_gradients(gradients, ("reconstruction", "auxiliary_sum"))
    gradients["total"] = tuple(direct_total)
    gradient_replay = _relative_l2(replay_total, direct_total)
    update = adamw_directions(direct_total, parameters, states, optimizer_group)
    directions = update.pop("directions")
    group_records = {}
    for group_name in PARAMETER_GROUPS:
        indices = parameter_groups[group_name]
        loss_norms = {name: _norm(gradients[name], indices) for name in LOSS_COMPONENTS}
        direction_norms = {name: _norm(directions[name], indices) for name in DIRECTIONS}
        direction_loss_cosine = {
            direction: {
                loss: _cosine(directions[direction], gradients[loss], indices)
                for loss in LOSS_COMPONENTS
            }
            for direction in DIRECTIONS
        }
        direction_cosine = {
            first: {second: _cosine(directions[first], directions[second], indices) for second in DIRECTIONS}
            for first in DIRECTIONS
        }
        group_records[group_name] = {
            "loss_gradient_l2_norm": loss_norms,
            "direction_l2_norm": direction_norms,
            "direction_loss_cosine": direction_loss_cosine,
            "direction_cosine": direction_cosine,
            "adamw_minus_clipped_efficiency": {
                loss: (
                    direction_loss_cosine["adamw_full"][loss]["value"]
                    - direction_loss_cosine["clipped_total"][loss]["value"]
                )
                for loss in LOSS_COMPONENTS
            },
        }
    finite = bool(
        math.isfinite(gradient_replay)
        and math.isfinite(update["adamw_decomposition_relative_l2"])
        and all(math.isfinite(value) for value in update["clipping"].values())
        and all(
            math.isfinite(value)
            for record in group_records.values()
            for values in (record["loss_gradient_l2_norm"], record["direction_l2_norm"])
            for value in values.values()
        )
        and all(
            math.isfinite(value["value"])
            for record in group_records.values()
            for matrix in (record["direction_loss_cosine"], record["direction_cosine"])
            for row in matrix.values()
            for value in row.values()
        )
    )
    return {
        "groups": group_records,
        "total_gradient_formula_relative_l2": float(gradient_replay),
        **update,
        "finite": finite,
    }


def _bootstrap(values: Sequence[float]) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("D2-K bootstrap requires finite one-dimensional values")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    sampled = array[indices].mean(axis=1)
    lower, upper = np.quantile(sampled, (0.025, 0.975))
    return {
        "estimate": float(array.mean()),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "blocks": array.tolist(),
    }


def mechanism_gate(candidates: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    checkpoint_results = {}
    for checkpoint in CHECKPOINTS:
        candidate = candidates.get(checkpoint)
        if candidate is None:
            checkpoint_results[checkpoint] = {"passed": False, "missing": True}
            continue
        timestep_results = {}
        for timestep in GATE_TIMESTEPS:
            blocks = candidate["timesteps"][str(timestep)]["blocks"]
            records = [block["groups"]["all_parameters"] for block in blocks]
            human_delta = _bootstrap([
                record["adamw_minus_clipped_efficiency"]["human_reconstruction"]
                for record in records
            ])
            adamw_human = _bootstrap([
                record["direction_loss_cosine"]["adamw_full"]["human_reconstruction"]["value"]
                for record in records
            ])
            adamw_object = _bootstrap([
                record["direction_loss_cosine"]["adamw_full"]["object_reconstruction"]["value"]
                for record in records
            ])
            checks = {
                "finite": bool(all(block["finite"] for block in blocks)),
                "gradient_formula_replay": max(
                    block["total_gradient_formula_relative_l2"] for block in blocks
                ) <= 1e-5,
                "clip_formula_replay": max(
                    block["clipping"]["formula_replay_max_abs"] for block in blocks
                ) <= 1e-5,
                "adamw_formula_replay": max(
                    block["adamw_decomposition_relative_l2"] for block in blocks
                ) <= 1e-5,
                "human_delta_bootstrap_lower": human_delta["bootstrap_95_ci"][0] >= 0.05,
                "adamw_human_bootstrap_lower": adamw_human["bootstrap_95_ci"][0] >= 0.15,
                "adamw_object_bootstrap_lower": adamw_object["bootstrap_95_ci"][0] >= 0.15,
            }
            timestep_results[str(timestep)] = {
                "passed": all(checks.values()),
                "checks": checks,
                "adamw_minus_clipped_human_efficiency": human_delta,
                "adamw_human_efficiency": adamw_human,
                "adamw_object_efficiency": adamw_object,
            }
        checks = {
            "all_finite": bool(candidate.get("finite", False)),
            "model_state_unchanged": candidate.get("model_state_sha256_before") == candidate.get("model_state_sha256_after"),
            "optimizer_state_unchanged": candidate.get("optimizer_state_sha256_before") == candidate.get("optimizer_state_sha256_after"),
            "mapped_optimizer_state_unchanged": candidate.get("mapped_state_sha256_before") == candidate.get("mapped_state_sha256_after"),
            "parameter_grad_buffers_clear": bool(candidate.get("parameter_grad_buffers_clear", False)),
            "optimizer_contract_exact": bool(candidate.get("optimizer_contract_exact", False)),
            "all_gate_timesteps": all(record["passed"] for record in timestep_results.values()),
        }
        checkpoint_results[checkpoint] = {
            "passed": all(checks.values()), "checks": checks, "timesteps": timestep_results,
        }
    passed = all(checkpoint_results.get(name, {}).get("passed", False) for name in CHECKPOINTS)
    return {
        "passed": passed,
        "classification": (
            "adamw-human-routing-rescue-positive-stop" if passed
            else "adamw-human-routing-rescue-negative-stop"
        ),
        "checkpoint_results": checkpoint_results,
        "training_authorized": False,
        "training_started": False,
    }
