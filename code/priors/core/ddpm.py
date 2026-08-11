"""Expert-agnostic DDPM core for the State-Compositional priors.

FROZEN CROSS-BRANCH CONTRACT.  This module holds the schedule buffers, the
forward noising step with its two-frame history pinning, and the production
reverse posterior.  Nothing here knows about objects, scenes or any single
expert, so HOIPrior, HSIPrior and the mixer all consume the same registered
coefficients.  The expert-specific reverse loop lives in each expert package
(for HOI: ``priors.hoi.diffusion.GaussianDiffusion.sample``).

Every line below was moved verbatim from the pre-split ``priors/diffusion.py``
at commit c77b9d8; changing it invalidates the sealed D2-X / D2-AI / W3
comparison points on both expert branches at once.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .diffusion_schedule import canonical_diffusion_schedule
from .representation import REPRESENTATION


def normalize_progress(progress: torch.Tensor) -> torch.Tensor:
    """Map raw ``pi/end_pi/seq_length`` to the Phase 1A model condition."""
    if progress.shape[-1] != 3:
        raise ValueError(f"expected progress [...,3], got {tuple(progress.shape)}")
    denominator = progress[..., 2:3].clamp_min(1.0)
    return torch.cat((progress[..., :2] / denominator, torch.log1p(denominator) / 10.0), dim=-1)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = values.gather(0, timesteps)
    return result.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Linear-beta x0-prediction diffusion with fixed two-frame history."""

    def __init__(self, timesteps: int = 500) -> None:
        super().__init__()
        schedule = canonical_diffusion_schedule(timesteps)
        betas = schedule["betas"]
        alphas = schedule["alphas"]
        alpha_bar = schedule["alpha_bar"]
        alpha_bar_previous = torch.nn.functional.pad(alpha_bar[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alpha_bar", schedule["sqrt_alpha_bar"])
        self.register_buffer(
            "sqrt_one_minus_alpha_bar",
            schedule["sqrt_one_minus_alpha_bar"],
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", posterior_variance.clamp_min(1e-20).log())
        self.register_buffer(
            "posterior_mean_coef1", betas * alpha_bar_previous.sqrt() / (1.0 - alpha_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2", (1.0 - alpha_bar_previous) * alphas.sqrt() / (1.0 - alpha_bar),
        )
        self.timesteps = timesteps

    def q_sample(
        self, clean: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = (
            _extract(self.sqrt_alpha_bar, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alpha_bar, timesteps, clean.shape) * noise
        )
        noisy[:, :REPRESENTATION.history_frames] = clean[:, :REPRESENTATION.history_frames]
        return noisy

    def posterior_mean(
        self,
        current: torch.Tensor,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Return the production ``q(x_{t-1} | x_t, x_0)`` posterior mean.

        Keeping this formula in one helper makes the paired D2-H diagnostic and
        the production sampler consume the exact same registered coefficients.
        The helper deliberately performs no projection, clamp, or conditioning.
        """
        if current.shape != clean.shape:
            raise ValueError(
                f"posterior current/clean shapes differ: {tuple(current.shape)}/{tuple(clean.shape)}"
            )
        if current.ndim != 3 or current.shape[1:] != (
            REPRESENTATION.window_frames, REPRESENTATION.dimension,
        ):
            raise ValueError(f"expected posterior state [B,16,232], got {tuple(current.shape)}")
        if timesteps.shape != (current.shape[0],) or timesteps.dtype != torch.long:
            raise ValueError(f"expected long posterior timesteps [B], got {timesteps.shape}/{timesteps.dtype}")
        if bool((timesteps < 0).any()) or bool((timesteps >= self.timesteps).any()):
            raise ValueError("posterior timestep is outside the registered diffusion schedule")
        return (
            _extract(self.posterior_mean_coef1, timesteps, current.shape) * clean
            + _extract(self.posterior_mean_coef2, timesteps, current.shape) * current
        )

    def posterior_sample(
        self,
        current: torch.Tensor,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        fixed_history: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one production reverse posterior step with explicit noise.

        ``noise`` is explicit so paired diagnostics can prove identity instead
        of relying on generator call order.  At timestep zero the registered
        posterior variance is zero, so a zero noise tensor preserves the
        production sampler's historical no-draw behavior.
        """
        if noise.shape != current.shape:
            raise ValueError(f"posterior noise shape mismatch: {tuple(noise.shape)}/{tuple(current.shape)}")
        expected_history = (
            current.shape[0], REPRESENTATION.history_frames, REPRESENTATION.dimension,
        )
        if fixed_history.shape != expected_history:
            raise ValueError(
                f"expected fixed history {expected_history}, got {tuple(fixed_history.shape)}"
            )
        mean = self.posterior_mean(current, clean, timesteps)
        result = mean + (
            0.5 * _extract(self.posterior_log_variance, timesteps, current.shape)
        ).exp() * noise
        result[:, :REPRESENTATION.history_frames] = fixed_history
        return result
