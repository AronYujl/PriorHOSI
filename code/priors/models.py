"""Independent randomly initialized HOI and HSI experts.

The HOI network is deliberately scene-free.  Its condition tokens mirror the
capacity of the released single-model Transformer while exposing only the
Phase 1A HOI contract: language/progress, dynamic-object BPS, and object goal.
"""

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

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


class _HOICleanMotionNetwork(nn.Module):
    """Condition-token Transformer that predicts the clean 232-D window."""

    def __init__(self, dim_model: int, num_heads: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.motion_input = nn.Linear(REPRESENTATION.dimension, dim_model)
        self.text = nn.Sequential(
            nn.Linear(768, dim_model), nn.SiLU(), nn.Linear(dim_model, dim_model),
        )
        self.bps = nn.Sequential(
            nn.Linear(1024 * 3, 768), nn.SiLU(), nn.Linear(768, dim_model),
        )
        self.goal_progress = nn.Sequential(
            nn.Linear(12, dim_model), nn.SiLU(), nn.Linear(dim_model, dim_model),
        )
        self.time = nn.Sequential(
            nn.Linear(dim_model, dim_model), nn.SiLU(), nn.Linear(dim_model, dim_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.position = nn.Parameter(torch.zeros(1, REPRESENTATION.window_frames + 4, dim_model))
        nn.init.normal_(self.position, std=0.02)
        self.output_norm = nn.LayerNorm(dim_model)
        self.output = nn.Linear(dim_model, REPRESENTATION.dimension)
        self.dim_model = dim_model

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        text_embedding: torch.Tensor,
        object_bps: torch.Tensor,
        goals: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        if noisy.ndim != 3 or noisy.shape[1:] != (
            REPRESENTATION.window_frames, REPRESENTATION.dimension,
        ):
            raise ValueError(f"expected noisy [B,16,232], got {tuple(noisy.shape)}")
        if text_embedding.ndim == 3 and text_embedding.shape[1] == 1:
            text_embedding = text_embedding[:, 0]
        conditions = torch.stack((
            self.time(_time_embedding(timesteps, self.dim_model)),
            self.text(text_embedding),
            self.bps(object_bps.reshape(object_bps.shape[0], -1)),
            self.goal_progress(torch.cat((goals, progress), dim=-1)),
        ), dim=1)
        motion = self.motion_input(noisy)
        tokens = torch.cat((conditions, motion), dim=1) + self.position
        encoded = self.transformer(tokens)
        return self.output(self.output_norm(encoded[:, -REPRESENTATION.window_frames:]))


class HOIPrior(nn.Module):
    """HOI API intentionally has no scene argument or scene encoder."""
    def __init__(self, dim_model: int = 256, num_heads: int = 8, num_layers: int = 4) -> None:
        super().__init__()
        self.network = _HOICleanMotionNetwork(dim_model, num_heads, num_layers)

    def forward(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, text_embedding: torch.Tensor,
        object_bps: torch.Tensor, goals: torch.Tensor, progress: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(noisy, timesteps, text_embedding, object_bps, goals, progress)


class HSIPrior(nn.Module):
    """HSI consumes real scene occupancy features and exposes no object condition."""
    def __init__(self, dim_model: int = 256, num_heads: int = 8, num_layers: int = 4) -> None:
        super().__init__()
        self.text = nn.Linear(768, 128)
        self.scene = nn.Linear(8 * 8 * 8, 128)
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
    num_heads: int = 8, num_layers: int = 4,
) -> nn.Module:
    if init_checkpoint not in (None, "", False):
        raise ValueError(
            "HOIPrior/HSIPrior must be randomly initialized; released InfBaGel checkpoint initialization is forbidden"
        )
    if expert == "hoi":
        return HOIPrior(dim_model, num_heads, num_layers)
    if expert == "hsi":
        return HSIPrior(dim_model, num_heads, num_layers)
    raise ValueError(f"unknown expert: {expert}")


def load_trained_hoi_prior(
    checkpoint_path: str, device: torch.device, *, use_ema: bool = True,
    weight_variant: Optional[str] = None,
) -> Tuple[HOIPrior, Dict[str, object]]:
    """Strictly load a Phase 1B checkpoint for evaluation or same-run resume.

    Released InfBaGel state dictionaries do not carry this checkpoint schema and
    are rejected before any parameter is loaded.
    """
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_type") != "hoi_prior_phase1b":
        raise ValueError("checkpoint is not a Phase 1B HOIPrior checkpoint; released InfBaGel initialization is forbidden")
    if checkpoint.get("expert") != "hoi" or checkpoint.get("initialization") != "random":
        raise ValueError("invalid HOIPrior checkpoint provenance")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("HOIPrior checkpoint is missing model_config")
    model = build_expert(
        "hoi",
        init_checkpoint=None,
        dim_model=int(model_config["dim_model"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
    )
    if weight_variant is None:
        weight_variant = "ema_0.9999" if use_ema else "online"
    if weight_variant == "online":
        state_key = "model"
        state = checkpoint.get(state_key)
    elif weight_variant in {"ema_0.999", "ema_0.9999"}:
        decay = weight_variant[len("ema_"):]
        ema_models = checkpoint.get("ema_models")
        if isinstance(ema_models, dict) and decay in ema_models:
            state_key = f"ema_models[{decay}]"
            state = ema_models[decay]
        elif decay == "0.9999":
            state_key = "ema_model"
            state = checkpoint.get(state_key)
        else:
            state_key = f"ema_models[{decay}]"
            state = None
    else:
        raise ValueError(f"unknown HOIPrior weight variant: {weight_variant}")
    if not isinstance(state, dict):
        raise ValueError(f"HOIPrior checkpoint is missing {state_key} weights")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    metadata = {
        key: checkpoint.get(key) for key in (
            "schema_version", "checkpoint_type", "expert", "initialization", "run_id",
            "seed", "git_commit", "processed_windows", "processed_frames", "optimizer_updates",
            "model_config", "data_contract_sha256", "split_sha256", "window_state_codec",
        )
    }
    metadata["weights"] = state_key
    metadata["weight_variant"] = weight_variant
    metadata["path"] = str(path)
    return model, metadata


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
