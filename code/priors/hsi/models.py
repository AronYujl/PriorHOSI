"""Independent randomly initialized HSI expert.

Mirrored here read-only for the HOI branch: ``phase/01c-hsi`` owns this file.
``HSIPrior`` consumes real scene occupancy and exposes no object condition; it
is built through ``priors.core.expert_api.build_expert("hsi", ...)`` so the
random-initialization guard cannot be bypassed.
"""

import torch
from torch import nn

from ..core.expert_api import _PriorNetwork


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
