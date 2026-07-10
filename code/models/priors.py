"""Independent HOI and HSI diffusion priors.

The two priors intentionally share the human-motion representation and the
underlying denoiser implementation, but expose different state and condition
contracts.  Keeping those contracts explicit prevents scene information from
leaking into the HOI prior and prevents dummy object channels from being
learned by the HSI prior.
"""

from dataclasses import asdict, dataclass

import torch

from models.infbagel import Unet


@dataclass(frozen=True)
class PriorSpec:
    name: str
    state_dim: int
    human_dim: int = 216
    joint_position_dim: int = 84
    joint_rotation_dim: int = 132
    object_translation_dim: int = 0
    object_rotation_dim: int = 0
    contact_dim: int = 0
    uses_scene: bool = False
    uses_object: bool = False

    def to_dict(self):
        return asdict(self)


HOI_PRIOR_SPEC = PriorSpec(
    name="hoi",
    state_dim=232,
    object_translation_dim=3,
    object_rotation_dim=9,
    contact_dim=4,
    uses_scene=False,
    uses_object=True,
)

HSI_PRIOR_SPEC = PriorSpec(
    name="hsi",
    state_dim=216,
    uses_scene=True,
    uses_object=False,
)

PRIOR_SPECS = {
    HOI_PRIOR_SPEC.name: HOI_PRIOR_SPEC,
    HSI_PRIOR_SPEC.name: HSI_PRIOR_SPEC,
}


class _IndependentPrior(Unet):
    prior_spec = None

    def __init__(self, **kwargs):
        if self.prior_spec is None:
            raise TypeError("_IndependentPrior must be specialized with a prior_spec")

        kwargs["dim_input"] = self.prior_spec.state_dim
        kwargs["dim_output"] = self.prior_spec.state_dim
        super().__init__(**kwargs)

    def forward(self, x, *args, **kwargs):
        if x.shape[-1] != self.prior_spec.state_dim:
            raise ValueError(
                f"{self.prior_spec.name.upper()} prior expects "
                f"{self.prior_spec.state_dim} channels, got {x.shape[-1]}"
            )
        return super().forward(x, *args, **kwargs)

    def prior_metadata(self):
        return self.prior_spec.to_dict()


class HOIPrior(_IndependentPrior):
    """Scene-independent human-object interaction prior.

    Scene conditioning is disabled in the constructor, not merely dropped at
    training time.  This makes the intended causal boundary auditable from a
    checkpoint's architecture.
    """

    prior_spec = HOI_PRIOR_SPEC

    def __init__(self, **kwargs):
        kwargs.update(
            load_scene=False,
            load_scene_goal=False,
            load_object_goal=True,
            is_mix=False,
            scene_type=None,
        )
        super().__init__(**kwargs)


class HSIPrior(_IndependentPrior):
    """Object-free human-scene interaction prior trained on real scenes."""

    prior_spec = HSI_PRIOR_SPEC

    def __init__(self, **kwargs):
        kwargs.update(
            load_scene=True,
            load_scene_goal=True,
            load_object_goal=False,
            # This activates the LINGO routing: locomotion uses pelvis_goal,
            # while sit/lie interactions use scene_goal.
            is_mix=True,
        )
        super().__init__(**kwargs)

    def forward(self, x, cond, timesteps, text_emb, pelvis_goal, scene_goal,
                is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length,
                need_pi, object_goal, is_object, obj_bps_data, occ_list,
                occ_pos, **kwargs):
        if torch.any(is_object):
            raise ValueError("HSI prior received object-bearing samples")
        return super().forward(
            x, cond, timesteps, text_emb, pelvis_goal, scene_goal, is_loco,
            need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi,
            object_goal, is_object, obj_bps_data, occ_list, occ_pos, **kwargs
        )
