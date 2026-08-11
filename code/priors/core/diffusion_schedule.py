"""Canonical 500-step float32 diffusion schedule shared by HOIPrior paths."""

from __future__ import annotations

import hashlib
from typing import Dict

import torch


DIFFUSION_STEPS = 500
BETA_START = 0.0001
BETA_END = 0.02
BETA_SHA256 = "496ec54f35af6fe7b92417f7da8b442f31c9c0070bfdd62dbb16fefc426c8f3e"
ALPHA_BAR_SHA256 = "55f162cebbe109c67a75b00a10a1d23ea85fb1d18df9a372a3e237df5a8f48d4"
SQRT_ALPHA_BAR_SHA256 = "5d25c63d6618c77cc31976ee9e2c5645aa41653030fca210594a05254323b440"
SQRT_ALPHA_BAR_SENTINELS = {
    0: 0.9999499917030334,
    100: 0.8995221257209778,
    249: 0.5297974348068237,
    400: 0.19632703065872192,
    499: 0.0797039046883583,
}


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a CPU-contiguous tensor using its raw dtype-preserving bytes."""
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def canonical_diffusion_schedule(timesteps: int = DIFFUSION_STEPS) -> Dict[str, torch.Tensor]:
    """Build the locked linear-beta schedule without consulting model state."""
    if int(timesteps) != DIFFUSION_STEPS:
        raise ValueError(f"HOIPrior diffusion must use {DIFFUSION_STEPS} steps")
    betas = torch.linspace(
        BETA_START,
        BETA_END,
        DIFFUSION_STEPS,
        dtype=torch.float32,
        device="cpu",
    )
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": alpha_bar.sqrt(),
        "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar).sqrt(),
    }


def validate_canonical_diffusion_schedule(
    schedule: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    """Fail closed if a caller did not receive the registered float32 schedule."""
    required = {
        "betas",
        "alphas",
        "alpha_bar",
        "sqrt_alpha_bar",
        "sqrt_one_minus_alpha_bar",
    }
    if set(schedule) != required:
        raise ValueError(f"canonical diffusion schedule keys mismatch: {sorted(schedule)}")
    for name in required:
        value = schedule[name]
        if (
            not torch.is_tensor(value)
            or value.shape != (DIFFUSION_STEPS,)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"invalid canonical diffusion schedule tensor: {name}")
    hashes = {
        "betas": tensor_sha256(schedule["betas"]),
        "alpha_bar": tensor_sha256(schedule["alpha_bar"]),
        "sqrt_alpha_bar": tensor_sha256(schedule["sqrt_alpha_bar"]),
    }
    expected = {
        "betas": BETA_SHA256,
        "alpha_bar": ALPHA_BAR_SHA256,
        "sqrt_alpha_bar": SQRT_ALPHA_BAR_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"canonical diffusion schedule hash mismatch: {hashes}")
    rho = schedule["sqrt_alpha_bar"]
    if not bool(torch.all(rho[1:] < rho[:-1])):
        raise ValueError("sqrt(alpha_bar) must be strictly decreasing")
    for index, expected_value in SQRT_ALPHA_BAR_SENTINELS.items():
        if float(rho[index].item()) != expected_value:
            raise ValueError(f"sqrt(alpha_bar) sentinel mismatch at {index}")
    return {
        "steps": DIFFUSION_STEPS,
        "dtype": "torch.float32",
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "beta_sha256": BETA_SHA256,
        "alpha_bar_sha256": ALPHA_BAR_SHA256,
        "sqrt_alpha_bar_sha256": SQRT_ALPHA_BAR_SHA256,
        "sqrt_alpha_bar_sentinels": {
            str(index): value
            for index, value in SQRT_ALPHA_BAR_SENTINELS.items()
        },
        "strictly_decreasing": True,
    }


def diffusion_schedule_contract_metadata() -> Dict[str, object]:
    schedule = canonical_diffusion_schedule()
    return validate_canonical_diffusion_schedule(schedule)
