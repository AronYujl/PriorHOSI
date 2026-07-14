"""Deterministic internal-only diagnostics for the Phase 1B remediation gate."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .representation import REPRESENTATION


D0_TIMESTEPS: Tuple[int, ...] = (0, 1, 10, 50, 100, 250, 499)


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_internal_triples(dataset, count: int = 128) -> List[Tuple[int, int, int]]:
    """Select the preregistered sequence-disjoint D0 triples.

    Dataset positions, rather than global language-window indices, are returned.
    Each triple begins at ``pi=0`` and advances by the 42 source frames emitted
    by a 16-frame/2-history rollout window.  Sequence selection is exactly the
    hash ordering preregistered in ``docs/EXPERIMENT_PLAN.md``.
    """
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D0 sequence selection is internal-validation only")
    by_sequence: Dict[int, Dict[int, int]] = defaultdict(dict)
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        pi = int(dataset.language["pi"][global_index])
        by_sequence[sequence][pi] = position
    eligible = []
    for sequence, positions in by_sequence.items():
        if all(pi in positions for pi in (0, 42, 84)):
            name = str(dataset.scene_names[sequence])
            eligible.append((stable_digest("42:hoi-remediation:" + name), name, sequence, positions))
    eligible.sort(key=lambda value: (value[0], value[1], value[2]))
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} internal sequences form a three-window rollout")
    return [tuple(value[3][pi] for pi in (0, 42, 84)) for value in eligible[:count]]


def select_teacher_windows(dataset, count: int = 512) -> List[int]:
    """Select a stable set of internal windows for every D0 timestep."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D0 teacher-window selection is internal-validation only")
    ranked = []
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        name = str(dataset.scene_names[sequence])
        pi = int(dataset.language["pi"][global_index])
        key = stable_digest(f"42:hoi-remediation-window:{name}:{pi}")
        ranked.append((key, name, pi, position))
    ranked.sort()
    if len(ranked) < count:
        raise ValueError(f"only {len(ranked)} internal windows are available")
    return [value[-1] for value in ranked[:count]]


def deterministic_derangement(count: int, *, device=None) -> torch.Tensor:
    if count < 2:
        raise ValueError("permutation sensitivity needs at least two samples")
    return torch.roll(torch.arange(count, device=device), shifts=1)


def field_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return non-history MSE for each locked 232-D representation field."""
    if prediction.shape != target.shape or prediction.shape[-1] != REPRESENTATION.dimension:
        raise ValueError(f"expected matching [...,232] tensors, got {prediction.shape}/{target.shape}")
    return {
        field.name: (
            prediction[:, REPRESENTATION.history_frames:, field.slice]
            - target[:, REPRESENTATION.history_frames:, field.slice]
        ).square().mean()
        for field in REPRESENTATION.fields
    }


def selection_sha256(values: Iterable[object]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
