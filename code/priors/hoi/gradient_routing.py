"""Frozen-checkpoint gradient geometry for the Phase 1B D2-I0 audit."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch

from .remediation import selection_sha256, stable_digest


TIMESTEPS: Tuple[int, ...] = (0, 1, 10, 50, 100, 250, 499)
GATE_TIMESTEPS: Tuple[int, ...] = (250, 499)
CHECKPOINTS: Tuple[str, ...] = ("R-1024", "R-3072")
BASE_COMPONENTS: Tuple[str, ...] = (
    "human_reconstruction",
    "object_reconstruction",
    "contact",
    "weighted_fk",
    "weighted_object_surface",
    "weighted_velocity",
    "terminal_goal",
)
LOSS_COMPONENTS: Tuple[str, ...] = (
    "human_reconstruction",
    "object_reconstruction",
    "contact",
    "reconstruction",
    "weighted_fk",
    "weighted_object_surface",
    "weighted_velocity",
    "terminal_goal",
    "auxiliary_sum",
    "total",
)
PARAMETER_GROUPS: Tuple[str, ...] = (
    "all_parameters",
    "time",
    "text",
    "bps",
    "goal_progress",
    "motion_input",
    "transformer",
    "output",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
PRIMARY_WINDOWS = 128
TERMINAL_WINDOWS = 64
BLOCK_SIZE = 16
EXPECTED_PRIMARY_SHA256 = "cefbee34d09cf7db3015e7dc1aacb2d17259608ae27f466c4cc7a11f3c1714c3"
EXPECTED_TERMINAL_SHA256 = "43acfcbcbfd6755e2bd66a991b5314805eee7a246a5b9783ec596f7a95c7fc21"


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:15], 16)


def select_fresh_holdouts(dataset) -> Dict[str, object]:
    """Select preregistered D2-H-disjoint primary and terminal cohorts."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-I selections are internal-validation only")
    ranked = []
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        name = str(dataset.scene_names[sequence])
        pi = int(dataset.language["pi"][global_index])
        ranked.append((
            stable_digest(f"42:hoi-remediation-window:{name}:{pi}"),
            name, pi, position, int(global_index),
        ))
    ranked.sort()
    if len(ranked) < 640:
        raise ValueError("D2-I requires at least 640 ranked internal windows")
    d2h_global = {value[-1] for value in ranked[:512]}
    primary_rows = ranked[512:640]
    primary = [value[-2] for value in primary_rows]
    primary_global = [value[-1] for value in primary_rows]
    terminal_count = sum(
        int(
            int(dataset.ends[index])
            == int(dataset.seq_ends[int(dataset.sequence_ids[index])]) - 1
        )
        for index in primary_global
    )
    terminal_rows = []
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        global_index = int(global_index)
        sequence = int(dataset.sequence_ids[global_index])
        if global_index in d2h_global or int(dataset.ends[global_index]) != int(dataset.seq_ends[sequence]) - 1:
            continue
        name = str(dataset.scene_names[sequence])
        pi = int(dataset.language["pi"][global_index])
        terminal_rows.append((
            stable_digest(f"42:d2i-terminal-fresh:{name}:{pi}"),
            name, pi, position, global_index,
        ))
    terminal_rows.sort()
    if len(terminal_rows) < TERMINAL_WINDOWS:
        raise ValueError(f"only {len(terminal_rows)} fresh terminal windows are available")
    terminal_rows = terminal_rows[:TERMINAL_WINDOWS]
    terminal = [value[-2] for value in terminal_rows]
    terminal_global = [value[-1] for value in terminal_rows]
    result = {
        "primary": {
            "positions": primary,
            "global_indices": primary_global,
            "sha256": selection_sha256(primary_global),
            "terminal_windows": terminal_count,
        },
        "terminal": {
            "positions": terminal,
            "global_indices": terminal_global,
            "sha256": selection_sha256(terminal_global),
            "terminal_windows": len(terminal),
        },
        "d2h_global_indices": d2h_global,
    }
    if set(primary_global) & d2h_global or set(terminal_global) & d2h_global:
        raise AssertionError("D2-I selection overlaps D2-H0")
    return result


def state_dict_sha256(model: torch.nn.Module) -> str:
    """Hash names, metadata and exact bytes without serializing a checkpoint."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_group_indices(model: torch.nn.Module) -> Tuple[Tuple[torch.nn.Parameter, ...], Dict[str, Tuple[int, ...]]]:
    named = tuple((name, value) for name, value in model.named_parameters() if value.requires_grad)
    parameters = tuple(value for _, value in named)
    prefixes = {
        "time": "network.time.",
        "text": "network.text.",
        "bps": "network.bps.",
        "goal_progress": "network.goal_progress.",
        "motion_input": "network.motion_input.",
        "transformer": "network.transformer.",
    }
    groups: Dict[str, Tuple[int, ...]] = {
        "all_parameters": tuple(range(len(named))),
    }
    for group, prefix in prefixes.items():
        groups[group] = tuple(index for index, (name, _) in enumerate(named) if name.startswith(prefix))
    groups["output"] = tuple(
        index for index, (name, _) in enumerate(named)
        if name.startswith("network.output.") or name.startswith("network.output_norm.")
    )
    missing = [name for name in PARAMETER_GROUPS if not groups.get(name)]
    if missing:
        raise ValueError(f"empty D2-I parameter groups: {missing}")
    return parameters, groups


def _sum_gradients(
    gradients: Mapping[str, Sequence[torch.Tensor | None]],
    names: Iterable[str],
) -> Tuple[torch.Tensor | None, ...]:
    names = tuple(names)
    width = len(next(iter(gradients.values())))
    result = []
    for index in range(width):
        values = [gradients[name][index] for name in names if gradients[name][index] is not None]
        result.append(sum(values[1:], values[0].clone()) if values else None)
    return tuple(result)


def _norm(gradients: Sequence[torch.Tensor | None], indices: Sequence[int]) -> float:
    squared = sum(
        float(gradients[index].detach().double().square().sum())
        for index in indices if gradients[index] is not None
    )
    return math.sqrt(squared)


def _cosine(
    first: Sequence[torch.Tensor | None],
    second: Sequence[torch.Tensor | None],
    indices: Sequence[int],
) -> Dict[str, object]:
    first_norm = _norm(first, indices)
    second_norm = _norm(second, indices)
    if not first_norm or not second_norm:
        return {"value": 0.0, "defined": False}
    dot = sum(
        float((first[index].detach().double() * second[index].detach().double()).sum())
        for index in indices if first[index] is not None and second[index] is not None
    )
    return {"value": dot / (first_norm * second_norm), "defined": True}


def _relative_l2(
    first: Sequence[torch.Tensor | None], second: Sequence[torch.Tensor | None],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for left, right in zip(first, second):
        if left is None and right is None:
            continue
        if left is None:
            left = torch.zeros_like(right)
        if right is None:
            right = torch.zeros_like(left)
        numerator += float((left.detach().double() - right.detach().double()).square().sum())
        denominator += float(right.detach().double().square().sum())
    return math.sqrt(numerator / max(denominator, np.finfo(np.float64).tiny))


def gradient_geometry(
    base_gradients: Mapping[str, Sequence[torch.Tensor | None]],
    direct_total: Sequence[torch.Tensor | None],
    groups: Mapping[str, Sequence[int]],
) -> Dict[str, object]:
    """Build all derived gradients, norms, cosines and formula replay checks."""
    if tuple(base_gradients) != BASE_COMPONENTS:
        raise ValueError(f"expected ordered base components {BASE_COMPONENTS}, got {tuple(base_gradients)}")
    gradients = dict(base_gradients)
    gradients["reconstruction"] = _sum_gradients(
        gradients, ("human_reconstruction", "object_reconstruction", "contact"),
    )
    gradients["auxiliary_sum"] = _sum_gradients(
        gradients, ("weighted_fk", "weighted_object_surface", "weighted_velocity", "terminal_goal"),
    )
    replay_total = _sum_gradients(gradients, ("reconstruction", "auxiliary_sum"))
    gradients["total"] = tuple(direct_total)
    replay_error = _relative_l2(replay_total, direct_total)
    group_records = {}
    for group in PARAMETER_GROUPS:
        indices = groups[group]
        norms = {name: _norm(gradients[name], indices) for name in LOSS_COMPONENTS}
        cosine = {
            first: {
                second: _cosine(gradients[first], gradients[second], indices)
                for second in LOSS_COMPONENTS
            }
            for first in LOSS_COMPONENTS
        }
        group_records[group] = {
            "gradient_l2_norm": norms,
            "cosine_matrix": cosine,
            "total_over_reconstruction_norm_ratio": (
                norms["total"] / max(norms["reconstruction"], np.finfo(np.float64).tiny)
            ),
            "total_reconstruction_cosine": cosine["total"]["reconstruction"],
            "gradient_cancellation_index": (
                sum(norms[name] for name in BASE_COMPONENTS)
                / max(norms["total"], np.finfo(np.float64).tiny)
            ),
        }
    finite = bool(
        math.isfinite(replay_error)
        and all(
            math.isfinite(value)
            for record in group_records.values()
            for value in record["gradient_l2_norm"].values()
        )
        and all(
            math.isfinite(value["value"])
            for record in group_records.values()
            for row in record["cosine_matrix"].values()
            for value in row.values()
        )
    )
    return {
        "groups": group_records,
        "total_gradient_formula_relative_l2": replay_error,
        "finite": finite,
    }


def _bootstrap(values: Sequence[float], *, geometric: bool) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("D2-I bootstrap requires finite one-dimensional block values")
    if geometric and bool((array <= 0).any()):
        raise ValueError("D2-I geometric bootstrap requires positive values")
    transformed = np.log(array) if geometric else array
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    sampled = transformed[indices].mean(axis=1)
    estimate = transformed.mean()
    if geometric:
        sampled = np.exp(sampled)
        estimate = math.exp(float(estimate))
    lower, upper = np.quantile(sampled, (0.025, 0.975))
    return {
        "estimate": float(estimate),
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
            blocks = candidate["cohorts"]["primary"]["timesteps"][str(timestep)]["blocks"]
            ratios = [block["groups"]["all_parameters"]["total_over_reconstruction_norm_ratio"] for block in blocks]
            cosines = [block["groups"]["all_parameters"]["total_reconstruction_cosine"]["value"] for block in blocks]
            ratio = _bootstrap(ratios, geometric=True)
            cosine = _bootstrap(cosines, geometric=False)
            checks = {
                "finite": bool(all(block["finite"] for block in blocks)),
                "formula_replay": bool(
                    max(block["total_gradient_formula_relative_l2"] for block in blocks) <= 1e-5
                ),
                "ratio_geometric_mean": ratio["estimate"] >= 20.0,
                "ratio_bootstrap_lower": ratio["bootstrap_95_ci"][0] >= 10.0,
                "cosine_bootstrap_upper": cosine["bootstrap_95_ci"][1] <= 0.25,
            }
            timestep_results[str(timestep)] = {
                "passed": all(checks.values()),
                "checks": checks,
                "total_over_reconstruction_norm_ratio": ratio,
                "total_reconstruction_cosine": cosine,
                "maximum_formula_replay_relative_l2": max(
                    block["total_gradient_formula_relative_l2"] for block in blocks
                ),
            }
        state_unchanged = (
            candidate.get("state_dict_sha256_before")
            == candidate.get("state_dict_sha256_after")
        )
        all_finite = bool(candidate.get("finite", False))
        checks = {
            "all_finite": all_finite,
            "state_dict_unchanged": state_unchanged,
            "all_gate_timesteps": all(record["passed"] for record in timestep_results.values()),
        }
        checkpoint_results[checkpoint] = {
            "passed": all(checks.values()),
            "checks": checks,
            "timesteps": timestep_results,
        }
    passed = all(checkpoint_results.get(name, {}).get("passed", False) for name in CHECKPOINTS)
    return {
        "passed": passed,
        "classification": (
            "weighted-objective-gradient-dominance-positive-stop"
            if passed else "weighted-objective-gradient-dominance-negative-stop"
        ),
        "checkpoint_results": checkpoint_results,
        "training_authorized": False,
        "training_started": False,
    }
