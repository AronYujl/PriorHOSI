"""Expert views of one world, including the diffusion path of a known empty object."""

import torch

from priors.core.diffusion_schedule import canonical_diffusion_schedule


def empty_motion_view(current, sigma, noise):
    """The single-level empty-modality view retained by the first diagnostic."""
    view = current.clone()
    view[:, :2, 216:] = 0
    view[:, 2:, 216:] = sigma * noise[:, 2:]
    return view


def masked_object_arguments(common):
    """Keep the complete geometric query, masking only Unet object tokens."""
    arguments = list(common)
    arguments[13] = torch.zeros_like(arguments[13])  # is_object after motion argument
    return tuple(arguments)


def empty_forward_trajectory(betas, innovations):
    """Exact linear Gaussian forward recurrence from clean zero, vectorized."""
    alpha_bar = (1 - betas).cumprod(0)
    shape = (-1,) + (1,) * (innovations.ndim - 1)
    increments = (betas / alpha_bar).sqrt().reshape(shape) * innovations
    return alpha_bar.sqrt().reshape(shape) * increments.cumsum(0)


class KnownEmptyObjectView:
    """Read a known-zero forward trajectory in reverse alongside the human chain."""

    def __init__(self):
        self.betas = canonical_diffusion_schedule()['betas']

    def begin_window(self, current, window_seed):
        generator = torch.Generator(device=current.device)
        generator.manual_seed((int(window_seed) + 2147483647) % (2**63 - 1))
        innovations = torch.randn(
            len(self.betas), current.shape[0], current.shape[1] - 2, 16,
            dtype=current.dtype, device=current.device, generator=generator,
        )
        self.trajectory = empty_forward_trajectory(
            self.betas.to(current.device, current.dtype), innovations,
        )

    def for_step(self, current, step):
        view = current.clone()
        view[:, :2, 216:] = 0
        view[:, 2:, 216:] = self.trajectory[step]
        return view
