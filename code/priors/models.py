"""Independent randomly initialized HOI and HSI expert scaffolds."""

import math
from typing import Iterable, Optional

import torch
from torch import nn

from .representation import REPRESENTATION


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


class HOIPrior(nn.Module):
    """HOI API intentionally has no scene argument or scene encoder."""
    def __init__(self, dim_model: int = 256, num_heads: int = 8, num_layers: int = 4) -> None:
        super().__init__()
        self.text = nn.Linear(768, 128)
        self.bps = nn.Linear(1024 * 3, 128)
        self.goal_progress = nn.Linear(12, 64)
        self.network = _PriorNetwork(320, dim_model, num_heads, num_layers)

    def forward(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, text_embedding: torch.Tensor,
        object_bps: torch.Tensor, goals: torch.Tensor, progress: torch.Tensor,
    ) -> torch.Tensor:
        condition = torch.cat((
            self.text(text_embedding), self.bps(object_bps.flatten(1)),
            self.goal_progress(torch.cat((goals, progress), dim=-1)),
        ), dim=-1)
        return self.network(noisy, timesteps, condition)


class HSIPrior(nn.Module):
    """HSI consumes real scene occupancy features and exposes no object condition."""
    def __init__(self, dim_model: int = 256, num_heads: int = 8, num_layers: int = 4,
                 scene_grid_size: int = 8) -> None:
        super().__init__()
        self.text = nn.Linear(768, 128)
        if scene_grid_size == 8:
            self.scene = nn.Sequential(nn.Flatten(), nn.Linear(8 * 8 * 8, 128))
        else:
            self.scene = nn.Sequential(
                nn.Unflatten(1, (1, scene_grid_size, scene_grid_size, scene_grid_size)),
                nn.Conv3d(1, 32, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(128, 128),
            )
        self.goal_progress = nn.Linear(12, 64)
        self.network = _PriorNetwork(320, dim_model, num_heads, num_layers)

    def forward(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, text_embedding: torch.Tensor,
        scene_condition: torch.Tensor, goals: torch.Tensor, progress: torch.Tensor,
    ) -> torch.Tensor:
        condition = torch.cat((
            self.text(text_embedding), self.scene(scene_condition.flatten(1)),
            self.goal_progress(torch.cat((goals, progress), dim=-1)),
        ), dim=-1)
        return self.network(noisy, timesteps, condition)


def build_expert(
    expert: str, *, init_checkpoint: Optional[str] = None, dim_model: int = 256,
    num_heads: int = 8, num_layers: int = 4, scene_grid_size: int = 8,
) -> nn.Module:
    if init_checkpoint not in (None, "", False):
        raise ValueError(
            "HOIPrior/HSIPrior must be randomly initialized; released InfBaGel checkpoint initialization is forbidden"
        )
    if expert == "hoi":
        return HOIPrior(dim_model, num_heads, num_layers)
    if expert == "hsi":
        return HSIPrior(dim_model, num_heads, num_layers, scene_grid_size)
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
