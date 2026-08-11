"""Shared expert construction contract for the State-Compositional priors.

FROZEN CROSS-BRANCH CONTRACT.  This module holds the pieces that both experts
and the future mixer must agree on:

* ``_time_embedding`` / ``_PriorNetwork`` - the minimal shared backbone;
* ``build_expert`` - the single construction entry point, carrying the
  random-initialization guard that forbids released-checkpoint initialization,
  so neither expert branch can bypass it;
* ``assert_parameter_independence`` - the generic two-module independence check.

Per-expert construction uses lazy imports inside ``build_expert``'s branches, so
this module never depends on ``priors.hoi`` or ``priors.hsi`` at import time and
an expert branch that has deleted the other expert's package still imports.
Every body below was moved verbatim from the pre-split ``priors/models.py`` at
commit c77b9d8.
"""

import math
from typing import Optional

import torch
from torch import nn

from .representation import REPRESENTATION

# Mirrors ``priors.hoi.models.HOI_ARCHITECTURE_BASE``.  The literal is repeated
# rather than imported because ``core`` must not depend on an expert package;
# ``tests/core/test_expert_contract.py`` asserts the two stay equal.
DEFAULT_ARCHITECTURE_VARIANT = "base"


def _time_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    value = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return value if width % 2 == 0 else torch.nn.functional.pad(value, (0, 1))


class _PriorNetwork(nn.Module):
    def __init__(self, condition_width: int, dim_model: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.input = nn.Linear(REPRESENTATION.dimension, dim_model)
        self.condition = nn.Sequential(nn.Linear(condition_width, dim_model), nn.SiLU(), nn.Linear(dim_model, dim_model))
        layer = nn.TransformerEncoderLayer(
            d_model=dim_model, nhead=num_heads, dim_feedforward=dim_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(dim_model, REPRESENTATION.dimension)
        self.dim_model = dim_model

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        token = self.input(noisy)
        token = token + self.condition(condition)[:, None] + _time_embedding(timesteps, self.dim_model)[:, None]
        return self.output(self.transformer(token))


def build_expert(
    expert: str, *, init_checkpoint: Optional[str] = None, dim_model: int = 256,
    num_heads: int = 8, num_layers: int = 4,
    architecture_variant: str = DEFAULT_ARCHITECTURE_VARIANT,
    bps_path: Optional[str] = None,
) -> nn.Module:
    if init_checkpoint not in (None, "", False):
        raise ValueError(
            "HOIPrior/HSIPrior must be randomly initialized; released InfBaGel checkpoint initialization is forbidden"
        )
    if expert == "hoi":
        from ..hoi.models import HOIPrior
        return HOIPrior(
            dim_model,
            num_heads,
            num_layers,
            architecture_variant=architecture_variant,
            bps_path=bps_path,
        )
    if expert == "hsi":
        if architecture_variant != DEFAULT_ARCHITECTURE_VARIANT:
            raise ValueError("HOI architecture variants are forbidden for HSIPrior")
        from ..hsi.models import HSIPrior
        return HSIPrior(dim_model, num_heads, num_layers)
    raise ValueError(f"unknown expert: {expert}")


def assert_parameter_independence(first: nn.Module, second: nn.Module) -> None:
    first_parameters = list(first.parameters())
    second_parameters = list(second.parameters())
    if {id(value) for value in first_parameters} & {id(value) for value in second_parameters}:
        raise AssertionError("experts share Parameter objects")
    def pointer(value: torch.Tensor) -> int:
        storage = value.untyped_storage() if hasattr(value, "untyped_storage") else value.storage()
        return storage.data_ptr()
    first_storage = {pointer(value) for value in first_parameters}
    second_storage = {pointer(value) for value in second_parameters}
    if first_storage & second_storage:
        raise AssertionError("experts share parameter storage")
