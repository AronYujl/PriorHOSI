import json
import math
import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch import ViT
from tqdm import tqdm
from utils import *
from guidance_loss import apply_hsi_guidance_loss
import pytorch3d.transforms as transforms
from priors.hsi.penetration import (
    DEFAULT_LINGO_MESH_ROOT,
    SceneSDFBank,
    resolve_sdf_dtype,
)
from priors.hsi.diagnostics import (
    FUTURE_OCC_OFFSETS,
    FutureOccCenterTelemetry,
    select_future_occ_centers,
    validate_future_occ_mode,
)

@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def cap_guidance_increment(gradient, cap):
    """Clip a guidance increment to an L2 norm of ``cap``, one norm PER SAMPLE.

    ``gradient *= min(1, cap / (||gradient|| + eps))``, reduced over every non-batch
    dimension so the branch cannot key on sample 0 and make the result depend on how a
    batch was laid out.  A sample already under ``cap`` is multiplied by exactly 1.0 and
    therefore comes back bitwise unchanged.  ``cap=None`` is the identity.
    """
    if cap is None:
        return gradient
    norm = gradient.flatten(1).norm(dim=1).view(-1, *([1] * (gradient.ndim - 1)))
    return gradient * torch.clamp(float(cap) / (norm + 1e-12), max=1.0)


HSI_GUIDANCE_ENERGIES = ("voxel", "sdf")


def validate_hsi_guidance_energy(value):
    value = str(value)
    if value not in HSI_GUIDANCE_ENERGIES:
        raise ValueError(
            "hsi_guidance_energy must be one of %s, got %r"
            % (", ".join(HSI_GUIDANCE_ENERGIES), value)
        )
    return value


def hsi_guidance_frame_weights(length, *, device, dtype, enabled):
    """Return the frozen flat-to-ramp guidance schedule, or ``None`` when off."""
    if not enabled:
        return None
    length = int(length)
    weights = torch.ones(length, device=device, dtype=dtype)
    weights[:4] = 0.0
    weights[4] = 1.0 / 3.0
    weights[5] = 2.0 / 3.0
    return weights


def apply_hsi_sdf_guidance_loss(
    human_jnts,
    scene_flag,
    sdf_bank,
    *,
    weight=20000.0,
    margin=0.03,
    floor_height=0.02,
    history_frames=2,
):
    """Mesh-SDF hinge energy on finite, in-bounds generated FK joints."""
    sdf, out_of_bounds = sdf_bank.signed_distance(human_jnts, scene_flag)
    scorable = torch.isfinite(sdf) & ~out_of_bounds
    scorable &= human_jnts[..., 1] >= float(floor_height)
    scorable[:, : int(history_frames)] = False
    depth = torch.clamp(-(sdf + float(margin)), min=0.0)
    if bool(scorable.any()):
        return float(weight) * (depth.square()[scorable]).mean()
    return human_jnts.sum() * 0.0


OBJECT_CHANNEL_SLICE = slice(216, 232)
HSI_CHAIN_REBASE_MODES = ("off", "c1", "c2", "c3")


def zero_object_x0(model_output, is_object, enabled):
    """Zero channels 216:232 of a predicted x0, PER SAMPLE, for scene-only rows.

    ``enabled=False`` returns the very same object, so a build carrying this knob
    reproduces the released arithmetic bitwise and no extra tensor op runs.

    When enabled, a row with ``is_object=False`` gets exactly zero in the 16 object
    channels (com 216:219, rot 219:228, contact 228:232) and is untouched in the 216
    human channels; a row with ``is_object=True`` comes back bitwise unchanged.  The
    mask is built from ``is_object`` alone and applied per sample, so the result for
    one row never depends on which rows shared its batch.

    Why this exists: HSI training never supervises those channels with anything but
    zero (``is_object`` is pinned False, so ``x_start[:, :, 216:232]`` is identically
    0 and ``p_losses`` masks the three object terms out), yet ``self.out`` emits all
    232 channels and ``embedding_input`` consumes all 232 at the next reverse step.
    B-v2 learned to predict ~0 there (dead-row norm 9.223e-3) because the unmasked
    loss still pushed it to the zero target; the P17-OC arm's E0 severed that
    gradient, so those rows stayed at ``nn.Linear`` init (norm 0.5761) and the
    reverse chain now drives x[216:232] toward a non-zero prediction the trunk never
    saw at low t.  Zeroing the PREDICTED x0 -- not x itself -- is the form that
    matches training: the DDPM posterior then walks those channels along exactly the
    q_sample(0) path, O(1) noise at high t decaying to 0, instead of imposing 0 at
    high t where training had sqrt(1-abar_t)*noise.
    """
    if not enabled:
        return model_output
    if not torch.is_tensor(is_object):
        is_object = torch.as_tensor(
            is_object, dtype=torch.bool, device=model_output.device
        )
    flags = is_object.reshape(-1).to(device=model_output.device, dtype=torch.bool)
    if flags.numel() == 1 and model_output.shape[0] != 1:
        flags = flags.expand(model_output.shape[0])
    keep = flags.to(model_output.dtype).view(-1, *([1] * (model_output.ndim - 1)))
    out = model_output.clone()
    out[..., OBJECT_CHANNEL_SLICE] = out[..., OBJECT_CHANNEL_SLICE] * keep
    return out


def rebase_model_output(
    model_output,
    x,
    mode,
    oracle_frame2=None,
    *,
    timestep=None,
    min_timestep=0,
):
    """Rigidly align predicted future channels to the fixed two-frame history."""
    mode = str(mode)
    if mode == "off" or (
        timestep is not None and int(timestep) < int(min_timestep)
    ):
        return model_output
    if mode not in HSI_CHAIN_REBASE_MODES:
        raise ValueError("unknown HSI chain rebase mode %r" % mode)

    out = model_output.clone()
    if mode == "c2":
        if oracle_frame2 is None:
            raise RuntimeError("c2 requires an oracle frame-2 position")
        position_base = oracle_frame2
    else:
        position_base = 2.0 * x[:, 1, :84] - x[:, 0, :84]
    position_delta = position_base - model_output[:, 2, :84]
    out[:, 2:, :84] += position_delta[:, None]

    if mode == "c3":
        rotation_base = 2.0 * x[:, 1, 84:216] - x[:, 0, 84:216]
        rotation_delta = rotation_base - model_output[:, 2, 84:216]
        out[:, 2:, 84:216] += rotation_delta[:, None]
    return out


class Sampler:
    def __init__(self, device, mask_ind, emb_f, batch_size, channel, auto_regre_num, timesteps, ddim_timesteps, cm_timesteps, **kwargs):
        self.device = device
        self.mask_ind = mask_ind
        self.emb_f = emb_f
        self.batch_size = batch_size
        self.channel = channel
        self.auto_regre_num = auto_regre_num
        self.timesteps = timesteps
        self.ddim_timesteps = ddim_timesteps
        self.scene_type = kwargs.get('scene_type', None)
        self.temp_voxel_num = kwargs.get('temp_voxel_num', 3)  # new param, controls number of temporal voxels, default 3 for backward compatibility
        self.get_scheduler()
        self.solver = DDIMSolver(self.alpha_cumprod.numpy(), self.timesteps, self.ddim_timesteps).to(self.device)
        self.cm_timesteps = cm_timesteps
        self.w = kwargs.get('w', 0)
        self.is_mix = kwargs.get('is_mix', False)
        # None = off, and p_sample then emits exactly the released arithmetic.
        # See config_sample_infbagel_lingo_hsi.yaml: hsi_guidance_norm_cap.
        cap = kwargs.get('hsi_guidance_norm_cap', None)
        self.hsi_guidance_norm_cap = None if cap is None else float(cap)
        # Two independent dose knobs on the same diffusion-path increment, both off by
        # default.  The 2026-08-23 norm-cap smoke showed the per-step MAGNITUDE is not
        # the defect -- B's increments are smaller than C's at every percentile to p99 --
        # so what is left to attack is how many of them accumulate per window.
        dose = kwargs.get('hsi_guidance_dose_scale', None)
        self.hsi_guidance_dose_scale = None if dose is None else float(dose)
        self.hsi_guidance_alpha_decay = bool(kwargs.get('hsi_guidance_alpha_decay', False))
        self.hsi_guidance_posterior_coef1 = bool(kwargs.get('hsi_guidance_posterior_coef1', False))
        self.hsi_guidance_frame_ramp = bool(kwargs.get('hsi_guidance_frame_ramp', False))
        self.hsi_guidance_energy = validate_hsi_guidance_energy(
            kwargs.get('hsi_guidance_energy', 'voxel')
        )
        self.hsi_guidance_sdf_weight = float(
            kwargs.get('hsi_guidance_sdf_weight', 20000.0)
        )
        self.hsi_guidance_sdf_margin = float(
            kwargs.get('hsi_guidance_sdf_margin', 0.03)
        )
        self.hsi_guidance_sdf_floor_height = float(
            kwargs.get('hsi_guidance_sdf_floor_height', 0.02)
        )
        # False by default reproduces pre-589ac7f / B-v2 / C-v4; True is for
        # P17-OC and later checkpoints trained with the occ_list transpose fix.
        self.occ_permute_fix = bool(kwargs.get('occ_permute_fix', False))
        # Training/inference input-distribution repair on the 16 object channels,
        # OFF by default (False = the released arithmetic, bitwise).  Diffusion path
        # only: cm_sample is the consistency path and C is neither modified nor
        # retrained.  See zero_object_x0 and
        # config_sample_infbagel_lingo_hsi.yaml: hsi_zero_object_x0.
        self.hsi_zero_object_x0 = bool(kwargs.get('hsi_zero_object_x0', False))
        self.hsi_chain_rebase_mode = str(kwargs.get('hsi_chain_rebase_mode', 'off'))
        if self.hsi_chain_rebase_mode not in HSI_CHAIN_REBASE_MODES:
            raise ValueError(
                "hsi_chain_rebase_mode must be one of %s"
                % ", ".join(HSI_CHAIN_REBASE_MODES)
            )
        self.hsi_chain_rebase_min_timestep = int(
            kwargs.get('hsi_chain_rebase_min_timestep', 0)
        )
        self._hsi_chain_rebase_oracle_frame2 = None
        self.hsi_future_occ_mode = validate_future_occ_mode(
            kwargs.get('hsi_future_occ_mode', 'predicted')
        )
        self.hsi_future_occ_jitter_scale = float(
            kwargs.get('hsi_future_occ_jitter_scale', 0.2)
        )
        self._hsi_future_occ_oracle_local = None
        self._hsi_future_occ_telemetry = None
        self._p_sample_trace_timestep = None
        self._p_sample_trace = None
        # B-match seam term, OFF by default (0.0 = the released arithmetic, and
        # p_losses then takes a branch it cannot distinguish from the old code).
        # When > 0 it reweights the first two GENERATED frames of the position
        # channel only -- the two frames that carry the whole measured seam
        # transient.  See docs/plan/PHASE_1C_HSI.md 2026-08-25 (third section).
        self.seam_loss_weight = float(kwargs.get('seam_loss_weight', 0.0) or 0.0)
        self.fullbody_seam_loss_weight = float(
            kwargs.get('fullbody_seam_loss_weight', 0.0) or 0.0
        )
        _w = kwargs.get('loss_w_jpos', 1.0)
        self.loss_w_jpos = 1.0 if _w is None else float(_w)
        _pen_weight = kwargs.get('pen_loss_weight', 0.0)
        self.pen_loss_weight = 0.0 if _pen_weight is None else float(_pen_weight)
        _pen_delta = kwargs.get('pen_delta', 0.03)
        self.pen_delta = 0.03 if _pen_delta is None else float(_pen_delta)
        _pen_floor_height = kwargs.get('pen_floor_height', 0.02)
        self.pen_floor_height = (
            0.02 if _pen_floor_height is None else float(_pen_floor_height)
        )
        self.pen_sdf_cache = kwargs.get('pen_sdf_cache', None) or None
        self.pen_sdf_dtype = resolve_sdf_dtype(kwargs.get('pen_sdf_dtype', torch.float16))
        self.geometry_loss_fp32 = bool(kwargs.get('geometry_loss_fp32', False))
        self.pen_sdf_bank = None
        
    def set_dataset_and_model(self, dataset, student_model, teacher_model=None, target_model=None):
        self.dataset = dataset
        if dataset.load_scene:
            self.grid = dataset.create_meshgrid(batch_size=self.batch_size).to(self.device)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.target_model = target_model
        nb_voxels = dataset.nb_voxels
        self.occ_idx = torch.arange(0, nb_voxels[1], 1).to(self.device)

    def begin_hsi_future_occ_episode(self):
        if self._hsi_future_occ_telemetry is not None:
            raise RuntimeError("future-occ telemetry episode already active")
        device = next(self.student_model.parameters()).device
        self._hsi_future_occ_telemetry = FutureOccCenterTelemetry(self.timesteps, device)

    def set_hsi_future_occ_oracle(self, oracle_local):
        expected = (self.batch_size, len(FUTURE_OCC_OFFSETS), 3)
        if tuple(oracle_local.shape) != expected:
            raise ValueError("future-occ oracle must have shape %s" % (expected,))
        if self._hsi_future_occ_oracle_local is not None:
            raise RuntimeError("future-occ oracle was not cleared after the previous window")
        self._hsi_future_occ_oracle_local = oracle_local

    def clear_hsi_future_occ_oracle(self):
        self._hsi_future_occ_oracle_local = None

    def finish_hsi_future_occ_episode(self):
        if self._hsi_future_occ_oracle_local is not None:
            raise RuntimeError("cannot finish future-occ episode with a live window oracle")
        if self._hsi_future_occ_telemetry is None:
            raise RuntimeError("future-occ telemetry episode is not active")
        report = self._hsi_future_occ_telemetry.report()
        self._hsi_future_occ_telemetry = None
        return report

    def abort_hsi_future_occ_episode(self):
        self._hsi_future_occ_oracle_local = None
        self._hsi_future_occ_telemetry = None

    def set_hsi_chain_rebase_oracle(self, frame2):
        self._hsi_chain_rebase_oracle_frame2 = frame2

    def clear_hsi_chain_rebase_oracle(self):
        self._hsi_chain_rebase_oracle_frame2 = None

    def begin_p_sample_trace(self, timestep):
        if self._p_sample_trace_timestep is not None:
            raise RuntimeError("p-sample trace already active")
        timestep = int(timestep)
        if not 0 <= timestep < int(self.timesteps):
            raise ValueError("p-sample trace timestep is out of range")
        self._p_sample_trace_timestep = timestep
        self._p_sample_trace = None

    def consume_p_sample_trace(self):
        if self._p_sample_trace_timestep is None:
            raise RuntimeError("p-sample trace is not active")
        if self._p_sample_trace is None:
            raise RuntimeError(
                "p-sample loop did not visit trace timestep %d"
                % self._p_sample_trace_timestep
            )
        trace = self._p_sample_trace
        self._p_sample_trace_timestep = None
        self._p_sample_trace = None
        return trace

    def abort_p_sample_trace(self):
        self._p_sample_trace_timestep = None
        self._p_sample_trace = None

    def _hsi_guidance_loss(self, human_jnts, scene_flag):
        if self.hsi_guidance_energy == 'voxel':
            return apply_hsi_guidance_loss(
                human_jnts, scene_flag, self.dataset.get_nearest_free_voxel
            )
        return apply_hsi_sdf_guidance_loss(
            human_jnts,
            scene_flag,
            self._get_pen_sdf_bank(),
            weight=self.hsi_guidance_sdf_weight,
            margin=self.hsi_guidance_sdf_margin,
            floor_height=self.hsi_guidance_sdf_floor_height,
            history_frames=self.auto_regre_num,
        )

    def _compute_human_joints(self, predicted_noise, joints, mat, rest_human_offsets):
        global_jpos = transform_points(
            self.dataset.denormalize_torch(predicted_noise[:, :, :84]), mat
        ).reshape(joints.shape[0], -1, 28, 3)

        # FK to get joint positions.
        curr_seq_local_jpos = rest_human_offsets[:, None].repeat(
            1, global_jpos.shape[1], 1, 1
        )  # [b, t, 24, 3]
        curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3)  # [b*t, 24, 3]
        curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]

        global_jrot_6d = predicted_noise[:, :, 84:216].reshape(
            joints.shape[0], -1, 22, 6
        )
        global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d)  # [b, t, 22, 3, 3]
        global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat

        local_jrot_mat = self.dataset.quat_ik_torch(
            global_jrot_mat.reshape(-1, 22, 3, 3)
        )  # [b*t, 22, 3, 3]
        _, human_jnts = self.dataset.quat_fk_torch(
            local_jrot_mat, curr_seq_local_jpos
        )  # [b*t, 24, 3]
        human_jnts = human_jnts.reshape(joints.shape[0], -1, 24, 3)  # [b, t, 24, 3]
        return global_jpos, human_jnts

    def _compute_fullbody_seam_loss(self, predicted_joints, target_joints):
        n = int(self.auto_regre_num)
        predicted_seam = torch.cat(
            [target_joints[:, n - 2:n], predicted_joints[:, n:n + 2]], dim=1
        )
        target_seam = target_joints[:, n - 2:n + 2]
        predicted_acceleration = (
            predicted_seam[:, 2:] - 2.0 * predicted_seam[:, 1:-1]
            + predicted_seam[:, :-2]
        )
        target_acceleration = (
            target_seam[:, 2:] - 2.0 * target_seam[:, 1:-1]
            + target_seam[:, :-2]
        )
        return F.mse_loss(predicted_acceleration, target_acceleration)

    def _get_pen_sdf_bank(self):
        if self.pen_sdf_bank is None:
            split_scene_names = None
            if self.dataset.split_manifest is not None:
                with open(self.dataset.split_manifest, "r") as f:
                    split = json.load(f)
                split_scene_names = {
                    str(name)
                    for name in split[self.dataset.split_partition]["scenes"]
                }

            flag_to_name = {}
            source_scene_names = set()
            for scene_name, unified_flag in self.dataset.unified_scene_dict.items():
                unified_flag = int(unified_flag)
                scene_name = str(scene_name)
                if self.dataset.unified_scene_source[unified_flag] != "lingo":
                    continue
                if split_scene_names is not None and scene_name not in split_scene_names:
                    continue
                flag_to_name[unified_flag] = scene_name
                source_scene_names.add(
                    scene_name[:-7]
                    if scene_name.endswith("_mirror")
                    else scene_name
                )

            if split_scene_names is not None:
                expected_source_scene_names = {
                    name[:-7] if name.endswith("_mirror") else name
                    for name in split_scene_names
                }
                if source_scene_names != expected_source_scene_names:
                    missing = sorted(expected_source_scene_names - source_scene_names)
                    unexpected = sorted(source_scene_names - expected_source_scene_names)
                    raise RuntimeError(
                        "split partition scene selection produced "
                        f"{len(source_scene_names)} source scenes, expected "
                        f"{len(expected_source_scene_names)}; "
                        f"missing={missing}, unexpected={unexpected}"
                    )

            self.pen_sdf_bank = SceneSDFBank.from_scene_flags(
                flag_to_name,
                dataset_root=self.dataset.lingo_dataset.folder,
                mesh_root=DEFAULT_LINGO_MESH_ROOT,
                cache_dir=self.pen_sdf_cache,
                dtype=self.pen_sdf_dtype,
                device=self.device,
                require_cache=True,
            )
        return self.pen_sdf_bank

    def get_scheduler(self):
        betas = linear_beta_schedule(timesteps=self.timesteps)

        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.betas = betas

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        self.alpha_cumprod = alphas_cumprod
    
    def sample_cfg_scale_mixed(self, batch_size, device, uncond_prob=0.1, w_max=2.0):
        """Mixed sampling strategy: with 10% probability sample w=-1, with 90% probability sample uniformly from [0, w_max]
        
        Args:
            batch_size: batch size
            device: device
            uncond_prob: probability of unconditional generation
            w_max: maximum CFG scale for conditional generation
            
        Returns:
            w: CFG scale [batch_size, 1]
            is_uncond: flag for unconditional generation [batch_size]
        """
        is_uncond = torch.rand(batch_size) < uncond_prob
        
        # generate w values
        w = torch.zeros(batch_size, 1, device=device)
        w[is_uncond] = -1.0
        if (~is_uncond).any():
            w[~is_uncond] = torch.rand((~is_uncond).sum(), 1, device=device) * w_max
        
        return w, is_uncond
        
    def q_sample(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _get_temp_frame_indices(self, temp_voxel_num, seq_length=15):
        """
        Dynamically generate temporal frame indices based on temp_voxel_num
        
        Args:
            temp_voxel_num: required number of temporal voxels
            seq_length: total sequence length, default 15
        
        Returns:
            List[int]: list of temporal frame indices
        """
        if temp_voxel_num == 0:
            return []
        elif temp_voxel_num == 1:
            return [seq_length // 2]  # middle frame, i.e. [8]
        elif temp_voxel_num == 2:
            return [8, 15]
        elif temp_voxel_num == 3:
            return [5, 10, 15]  # original implementation, kept for backward compatibility
        else:
            # for other counts, distribute uniformly
            if temp_voxel_num > seq_length - 1:
                temp_voxel_num = seq_length - 1
            indices = []
            for i in range(temp_voxel_num):
                idx = (i + 1) * seq_length // (temp_voxel_num + 1)
                indices.append(min(idx, seq_length - 1))
            return indices

    def _compute_occ(self, noisy_input, x_start, joints, mat, scene_flag, object_points, pelvis_goal, scene_goal, object_goal, is_loco, is_object, need_pelvis_dir, obj_rot_mat_ref):
        with torch.no_grad():
            x_orig = transform_points(self.dataset.denormalize_torch(noisy_input[:, :, :joints.shape[-1]]), mat)
            mat_for_query = mat.clone()
            target_ind = self.mask_ind if self.mask_ind != -1 else 0
            mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
            mat_for_query[:, 1, 3] = 0
            query_points = transform_points(self.grid, mat_for_query)
            occ = self.dataset.get_occ_for_points(query_points, object_points, scene_flag).float()

            nb_voxels = self.dataset.nb_voxels
            occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

            if self.scene_type in ['plane_two', 'occ_two', 'occ_temp']:
                mat_for_query_goal = mat.clone()

                # handle pelvis goal in the need_pelvis_dir case
                pelvis_goal_copy = pelvis_goal.clone()
                if self.is_mix:
                    # mix: non-loco (sit/lie) uses scene_goal (scene goal) instead of pelvis; no loco normalization
                    pelvis_goal_copy[torch.logical_not(is_loco)] = scene_goal[torch.logical_not(is_loco)]
                else:
                    # handle pelvis goal in the is_loco case
                    pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                pelvis_goal_orig = transform_points(pelvis_goal_copy.unsqueeze(1), mat).squeeze(1)

                # handle object goal in the is_object case - no rotation needed
                object_goal_copy = object_goal.clone()
                # object_goal_copy[is_object] = object_goal_copy[is_object] / (torch.norm(object_goal_copy[is_object], dim=-1, keepdim=True) + 1e-6) * 0.8
                object_goal_orig = transform_points(object_goal_copy.unsqueeze(1), mat).squeeze(1)

                # set goal position based on need_pelvis_dir and is_object
                mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir] # need_pelvis_dir: inter_scene, is_loco, is_object
                mat_for_query_goal[is_object, :3, 3] = object_goal_orig[is_object] # is_object: inter_object
                mat_for_query_goal[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3] = mat_for_query[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3].clone()
                mat_for_query_goal[:, 1, 3] = 0.

                query_points = transform_points(self.grid, mat_for_query_goal)
                occ_goal = self.dataset.get_occ_for_points(query_points, None, scene_flag)
                nb_voxels = self.dataset.nb_voxels
                occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                end_goal_pos = torch.zeros(self.batch_size, 2).to(self.device)
                end_goal_pos[need_pelvis_dir] = pelvis_goal_copy[need_pelvis_dir].reshape(-1, 3)[:, [0, 2]]
                end_goal_pos[is_object] = object_goal_copy[is_object].reshape(-1, 3)[:, [0, 2]]

            occ_pos = torch.zeros(0, self.batch_size, 2).to(self.device)
            occ_pos = torch.cat([occ_pos, end_goal_pos[None]], dim=0)

            occ_list = torch.zeros(0, nb_voxels[1], nb_voxels[0], nb_voxels[2]).to(self.device)
            if self.occ_permute_fix:
                occ_list = torch.cat([occ_list, occ.permute(0, 2, 1, 3)], dim=0)
            else:
                occ_list = torch.cat([occ_list, occ], dim=0)
            occ_temp = None
            if self.scene_type == 'occ_temp':
                object_points_temp = object_points.clone()
                pred_obj_rot_mat_rel = noisy_input[:, :, 219:228].reshape(joints.shape[0], -1, 3, 3)

                pred_obj_rot_mat_rel_aa = transforms.matrix_to_axis_angle(pred_obj_rot_mat_rel) # [b, t, 3]
                # std_per_dim = torch.tensor([0.5, 1.5, 0.5], device=pred_obj_rot_mat_rel_aa.device).view(1, 1, 3)
                # perturb = torch.randn_like(pred_obj_rot_mat_rel_aa) * std_per_dim
                # pred_obj_rot_mat_rel_aa = pred_obj_rot_mat_rel_aa + perturb
                pred_obj_rot_mat_rel = transforms.axis_angle_to_matrix(pred_obj_rot_mat_rel_aa)

                obj_rot_mat_ref_temp = obj_rot_mat_ref.unsqueeze(1).repeat(1, pred_obj_rot_mat_rel.shape[1], 1, 1)
                pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref_temp # [b, t, 3, 3]
                pred_obj_rot_mat = pred_obj_rot_mat @ pred_obj_rot_mat[:, 0:1, :, :].transpose(2, 3)

                pred_obj_trans = noisy_input[:, :, 216:219] # [b, t, 3]
                pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)
                pred_obj_trans = pred_obj_trans - pred_obj_trans[:, 0:1, :]

                # perturb = (torch.rand_like(pred_obj_trans) - 0.5) * 0.4  # ∈ [-0.2, 0.2]
                # pred_obj_trans = pred_obj_trans + perturb

                object_points_temp = object_points_temp.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 1024, 3]
                object_points_temp = torch.matmul(pred_obj_rot_mat, object_points_temp.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 1024, 3]

                x_denorm = self.dataset.denormalize_torch(x_start[:, :, :joints.shape[-1]])
                perturb = (
                    torch.rand_like(x_denorm) - 0.5
                ) * self.hsi_future_occ_jitter_scale
                x_denorm = x_denorm + perturb

                # dynamically obtain temporal frame indices
                temp_indices = self._get_temp_frame_indices(self.temp_voxel_num)

                # only loop when temporal voxels exist
                for i in temp_indices:
                    x0_orig = transform_points(x_denorm, mat)
                    mat_for_query = mat.clone()
                    target_ind = self.mask_ind if self.mask_ind != -1 else 0

                    mat_for_query[:, :3, 3] = x0_orig[:, i, target_ind * 3: target_ind * 3 + 3]
                    mat_for_query[:, 1, 3] = 0
                    query_points = transform_points(self.grid, mat_for_query)

                    occ_pos = torch.cat([occ_pos, x_denorm[:, i, [0, 2]][None]], dim=0)

                    occ_temp = self.dataset.get_occ_for_points(query_points, object_points_temp[:, i, :, :], scene_flag)

                    nb_voxels = self.dataset.nb_voxels
                    occ_temp = occ_temp.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
                    occ_temp = occ_temp.permute(0, 2, 1, 3)

                    occ_list = torch.cat([occ_list, occ_temp], dim=0)

            if self.scene_type == 'occ':
                occ = occ.permute(0, 2, 1, 3)
            elif self.scene_type == 'plane':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
            elif self.scene_type == 'plane_two':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                occ_goal = occ_goal.permute(0, 1, 3, 2)
                occ_goal_cnt = occ_goal * self.occ_idx
                occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_two':
                occ = occ.permute(0, 2, 1, 3)
                occ_goal = occ_goal.permute(0, 2, 1, 3)
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_temp':
                occ = occ_goal.permute(0, 2, 1, 3)

        return occ, occ_list, occ_pos

    def consistency_loss(self, x_start, joints, mat, scene_flag, mask, t, text_emb, pelvis_goal, scene_goal, object_goal, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points=None, noise=None, loss_type='l2'):
        update_ema(self.target_model.parameters(), self.student_model.parameters(), 0.95)

        if noise is None:
            noise = torch.randn_like(x_start)
        
        noise = noise.to(x_start.device, dtype=torch.float32)
        noise[mask] = 0.

        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        topk = (self.timesteps // self.ddim_timesteps)
        index = torch.randint(0, self.ddim_timesteps, (x_start.shape[0],), device=x_start.device).long()
        
        start_timestep = self.solver.ddim_timesteps[index]
        timesteps = start_timestep - topk
        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

        inference_indices = np.linspace(
                    0, len(self.solver.ddim_timesteps), num=self.cm_timesteps, endpoint=False
                )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = (
            torch.from_numpy(inference_indices).long().to(timesteps.device)
        )
        
        # Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timestep)
        c_skip_start, c_out_start = [append_dims(x, x_start.ndim) for x in [c_skip_start, c_out_start]]
        
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, x_start.ndim) for x in [c_skip, c_out]]

        # Add noise to the latents according to the noise magnitude at each timestep
        x_start_noisy = self.q_sample(x_start=x_start, t=start_timestep, noise=noise)
        x_start_noisy[mask] = x_start[mask]
        if self.is_mix:
            # mix: keep object dims (216:) as GT for non-object (scene-only) samples
            x_start_noisy[torch.logical_not(is_object), :, 216:] = x_start[torch.logical_not(is_object), :, 216:]

        if self.dataset.load_scene:
            occ, occ_list, occ_pos = self._compute_occ(x_start_noisy, x_start, joints, mat, scene_flag, object_points, pelvis_goal, scene_goal, object_goal, is_loco, is_object, need_pelvis_dir, obj_rot_mat_ref)
        else:
            occ = None
        
        # sample CFG scale
        w, is_uncond = self.sample_cfg_scale_mixed(x_start.shape[0], x_start.device)
        
        # Student model prediction (with CFG scale)
        pred_x_0 = self.student_model(x_start_noisy, occ, start_timestep, text_emb, pelvis_goal, scene_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, cfg_scale=w)

        sqrt_one_minus_alphas_cumprod_t = extract(
                self.sqrt_one_minus_alphas_cumprod, start_timestep, x_start_noisy.shape
            )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, start_timestep, x_start_noisy.shape)
        
        noise_pred = (x_start_noisy - sqrt_alphas_cumprod_t * pred_x_0) / sqrt_one_minus_alphas_cumprod_t

        model_pred = pred_x_0

        model_pred = c_skip_start * x_start_noisy + c_out_start * model_pred

        # Use the ODE solver to predict the kth step in the augmented PF-ODE trajectory after
        with torch.no_grad():
            # conditional prediction
            cond_pred = self.teacher_model(x_start_noisy, occ, start_timestep, text_emb, pelvis_goal, scene_goal,
                                          is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi,
                                          object_goal, is_object, obj_bps_data, occ_list, occ_pos, 
                                          is_sample=True, is_uncondition=False)
            
            # unconditional prediction
            uncond_pred = self.teacher_model(x_start_noisy, occ, start_timestep, text_emb, pelvis_goal, scene_goal,
                                            is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi,
                                            object_goal, is_object, obj_bps_data, occ_list, occ_pos, 
                                            is_sample=True, is_uncondition=True)
            
            # unified CFG formula, automatically handles all w values
            # w = -1: teacher_pred_x0 = cond_pred + (-1) * (cond_pred - uncond_pred) = uncond_pred
            # w = 0:  teacher_pred_x0 = cond_pred + 0 * (cond_pred - uncond_pred) = cond_pred
            # w > 0:  teacher_pred_x0 = cond_pred + w * (cond_pred - uncond_pred) = CFG enhancement
            teacher_pred_x0 = cond_pred + w.unsqueeze(-1) * (cond_pred - uncond_pred)
            
            teacher_noise_pred = (x_start_noisy - sqrt_alphas_cumprod_t * teacher_pred_x0) / sqrt_one_minus_alphas_cumprod_t
            x_prev = self.solver.ddim_step(teacher_pred_x0, teacher_noise_pred, index)
            
            x_prev[mask] = x_start[mask].to(x_prev.dtype)

            # Pass w so the target is w-dependent; otherwise its target is w-invariant and
            # drives the newly trainable cfg embedding back toward zero. Without cfg_scale,
            # the forward else branch applies a random 10% scene ablation with no
            # self.training guard; passing w selects the per-row cfg_scale == -1 mask path,
            # matching student. Do not pass is_sample: target training uses that same path.
            target_pred_x0 = self.target_model(x_prev, occ, timesteps, text_emb, pelvis_goal, scene_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, cfg_scale=w)
            
            sqrt_one_minus_alphas_cumprod_t = extract(
                    self.sqrt_one_minus_alphas_cumprod, timesteps, x_prev.shape
                )
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, timesteps, x_prev.shape)
            
            target_noise_pred = (x_prev - sqrt_alphas_cumprod_t * target_pred_x0) / sqrt_one_minus_alphas_cumprod_t

            target = target_pred_x0

            target = c_skip * x_prev + c_out * target

        # Calculate loss
        mask_inv = torch.logical_not(mask)
        if loss_type == 'l1':
            loss = F.l1_loss(model_pred[mask_inv].float(), target[mask_inv].float())
        elif loss_type == 'l2':
            loss = F.mse_loss(model_pred[mask_inv].float(), target[mask_inv].float())
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(model_pred[mask_inv].float(), target[mask_inv].float())
        else:
            raise NotImplementedError()

        # add object loss (obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts)
        if self.dataset.use_object_keypoints:
            hand_idx_28 = [20, 21, 25, 27]
            hand_idx_24 = [20, 21, 22, 23]
            foot_idx = [7, 8, 10, 11]
            
            gt_global_jpos = transform_points(self.dataset.denormalize_torch(joints), mat).reshape(joints.shape[0], -1, 28, 3)
            gt_global_hand_jpos = gt_global_jpos[:, :, hand_idx_28, :]
            gt_global_foot_jpos = gt_global_jpos[:, :, foot_idx, :]

            model_pred[mask] = x_start[mask]
            
            global_jpos = transform_points(self.dataset.denormalize_torch(model_pred[:, :, :84]), mat).reshape(joints.shape[0], -1, 28, 3)

            # FK to get joint positions.
            curr_seq_local_jpos = rest_human_offsets[:, None].repeat(1, global_jpos.shape[1], 1, 1) # [b, t, 24, 3]
            curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [b*t, 24, 3]
            curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]

            global_jrot_6d = model_pred[:, :, 84:216].reshape(joints.shape[0], -1, 22, 6)
            global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d) # [b, t, 22, 3, 3]
            global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat
            
            local_jrot_mat = self.dataset.quat_ik_torch(global_jrot_mat.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]
            _, human_jnts = self.dataset.quat_fk_torch(local_jrot_mat, curr_seq_local_jpos) # [b*t, 24, 3]
            human_jnts = human_jnts.reshape(joints.shape[0], -1, 24, 3) # [b, t, 24, 3]

            pred_global_hand_jpos = human_jnts[:, :, hand_idx_24, :]
            pred_global_foot_jpos = human_jnts[:, :, foot_idx, :] # [b, t, 4, 3]

            mask_fk = torch.ones(mask_inv.shape[0], self.dataset.max_window_size, 4, 3, dtype=torch.bool).to(mask_inv.device)
            mask_fk[:, :self.auto_regre_num, :, :] = False
            fk_hand_loss = F.mse_loss(pred_global_hand_jpos[mask_fk], gt_global_hand_jpos[mask_fk])
            fk_foot_loss = F.mse_loss(pred_global_foot_jpos[mask_fk], gt_global_foot_jpos[mask_fk])
            loss_fk = fk_hand_loss + fk_foot_loss
            
            model_mean = model_pred # x_start
            pred_obj_rot_mat_rel = model_mean[:, :, 219:228].reshape(joints.shape[0], -1, 3, 3)
            obj_rot_mat_ref = obj_rot_mat_ref.unsqueeze(1).repeat(1, pred_obj_rot_mat_rel.shape[1], 1, 1)
            pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref # [b, t, 3, 3]

            pred_obj_trans = model_mean[:, :, 216:219] # [b, t, 3]
            pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)

            rest_pose_obj_nn_pts = rest_pose_obj_nn_pts.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 100, 3]
            pred_seq_obj_kpts = torch.matmul(pred_obj_rot_mat, rest_pose_obj_nn_pts.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 100, 3]
            
            mask_points = torch.ones(mask_inv.shape[0], self.dataset.max_window_size, 100, 3, dtype=torch.bool).to(mask_inv.device)
            mask_points[:, :self.auto_regre_num, :, :] = False
            mask_points = torch.logical_and(mask_points, is_object.to(mask_inv.device, dtype=torch.bool).reshape(-1, 1, 1, 1))

            if mask_points.any():
                if loss_type == 'l1':
                    loss_object = F.l1_loss(transformed_obj_verts[mask_points], pred_seq_obj_kpts[mask_points])
                elif loss_type == 'l2':
                    loss_object = F.mse_loss(transformed_obj_verts[mask_points], pred_seq_obj_kpts[mask_points])
                elif loss_type == "huber":
                    loss_object = F.smooth_l1_loss(transformed_obj_verts[mask_points], pred_seq_obj_kpts[mask_points])
                else:
                    raise NotImplementedError()
            else:
                loss_object = None

        else: 
            loss_object = None
            loss_fk = None

        return dict(loss_consistency=loss, loss_object=loss_object, loss_fk=loss_fk)

    @torch.no_grad()
    def cm_sample_loop(self, fixed_points, mat, scene_flag, text_emb, pelvis_goal, scene_goal, object_goal, \
                    need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, guidance_fn, guidance_scale, object_only=False, w=None, obj_rot_mat_prefix=None):
        self.batch_size = fixed_points.shape[0]
        device = next(self.student_model.parameters()).device
        shape = (self.batch_size, self.dataset.max_window_size, self.channel)
        points = torch.randn(shape, device=device, dtype=torch.float32)

        if self.auto_regre_num > 0:
            self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)
        imgs = []
        occs = []
        x0 = []
        inference_indices = np.linspace(-1, len(self.solver.ddim_timesteps) - 1, num=self.cm_timesteps + 1, endpoint=True)
        inference_indices = (
                    torch.from_numpy(np.floor(inference_indices).astype(np.int64)).long().to(device)
                )
        inference_indices = inference_indices[1:]
        t_index = len(inference_indices) - 1
        x0.append(points)
        for i in tqdm(reversed(inference_indices), desc='sampling loop time step', total=len(inference_indices)):
            model_used = self.student_model
            points, occ, pred_x_0 = self.cm_sample(model_used, x0[-1], points, fixed_points, mat, scene_flag,
                                        torch.full((self.batch_size,), i, device=device, dtype=torch.long), t_index,
                                        text_emb, pelvis_goal, scene_goal, object_goal, need_scene,
                                        need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, obj_rot_mat_prefix, guidance_fn, guidance_scale, object_only, w)
            if self.auto_regre_num > 0:
                self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)

            points_orig = points
            imgs.append(points_orig)
            x0.append(pred_x_0)
            if occ is not None:
                occs.append(occ.cpu().numpy())

            t_index -= 1

        return imgs, occs

    @torch.no_grad()
    def _compute_occ_sample(self, x, x0, mat, scene_flag, object_points,
                            pelvis_goal, scene_goal, object_goal,
                            is_loco, is_object, need_pelvis_dir, obj_rot_mat_ref,
                            object_only, obj_rest_verts, seq_name_dict, obj_rot_mat_prefix,
                            diffusion_timestep=None):
        if self.dataset.load_scene:
            x_orig = transform_points(self.dataset.denormalize_torch(x[:, :, :84]), mat)
            mat_for_query = mat.clone()
            target_ind = self.mask_ind if self.mask_ind != -1 else 0
            mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
            mat_for_query[:, 1, 3] = 0
            
            self.grid = self.dataset.create_meshgrid(batch_size=self.batch_size).to(self.device)

            query_points = transform_points(self.grid, mat_for_query)
            occ = self.dataset.get_occ_for_points(query_points, object_points, scene_flag)
            nb_voxels = self.dataset.nb_voxels
            occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
            
            if object_only:
                occ[occ == 1] = 0.

            if self.is_mix and torch.logical_not(is_object).any():
                occ[torch.logical_not(is_object)][occ == 2] = 1.

            if self.scene_type in ['plane_two', 'occ_two', 'occ_temp']:
                mat_for_query_goal = mat.clone()
                
                # handle pelvis goal in the is_loco case
                pelvis_goal_copy = pelvis_goal.clone()
                if self.is_mix:
                    # mix: non-loco (sit/lie) uses scene_goal (scene goal) instead of pelvis
                    pelvis_goal_copy[torch.logical_not(is_loco)] = scene_goal[torch.logical_not(is_loco)]
                else:
                    pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (
                                torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                pelvis_goal_orig = transform_points(pelvis_goal_copy.reshape(pelvis_goal_copy.shape[0], 1, 3), mat).squeeze(1)

                # handle object goal in the is_object case - no rotation needed
                object_goal_copy = object_goal.clone()
                object_goal_orig = transform_points(object_goal_copy.reshape(object_goal_copy.shape[0], 1, 3), mat).squeeze(1)

                mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir]
                mat_for_query_goal[is_object, :3, 3] = object_goal_orig[is_object]
                mat_for_query_goal[torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3, 3] = mat_for_query[
                                                                                torch.logical_not(torch.logical_or(need_pelvis_dir, is_object)), :3,
                                                                                3].clone()
                mat_for_query_goal[:, 1, 3] = 0.
                query_points_goal = transform_points(self.grid, mat_for_query_goal)
                occ_goal = self.dataset.get_occ_for_points(query_points_goal, object_points, scene_flag)

                if object_only:
                    occ_goal[occ_goal == 1] = 0.

                if self.is_mix and torch.logical_not(is_object).any():
                    occ_goal[torch.logical_not(is_object)][occ_goal == 2] = 1.

                nb_voxels = self.dataset.nb_voxels
                occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                end_goal_pos = torch.zeros(self.batch_size, 2).to(self.device)
                end_goal_pos[need_pelvis_dir] = pelvis_goal_copy[need_pelvis_dir].reshape(-1, 3)[:, [0, 2]]
                end_goal_pos[is_object] = object_goal_copy[is_object].reshape(-1, 3)[:, [0, 2]]
            
            occ_pos = torch.zeros(0, self.batch_size, 2).to(self.device)
            occ_pos = torch.cat([occ_pos, end_goal_pos[None]], dim=0)
                
            occ_list = torch.zeros(0, nb_voxels[1], nb_voxels[0], nb_voxels[2]).to(self.device)
            if self.occ_permute_fix:
                occ_list = torch.cat([occ_list, occ.permute(0, 2, 1, 3)], dim=0)
            else:
                occ_list = torch.cat([occ_list, occ], dim=0)
            occ_temp = None
            if self.scene_type == 'occ_temp':
                if self.dataset.vis:
                    # object_rot_mat = x0[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                    # object_trans_orig = x0[:, :, 216:219] # [b, t, 3]
                    object_rot_mat = x[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                    object_trans_orig = x[:, :, 216:219] # [b, t, 3]
                    object_trans_orig = transform_points(self.dataset.denormalize_torch(object_trans_orig, is_object=True), mat)

                    obj_name = seq_name_dict[0].split('_')[1]
                    pred_obj_rot_mat_seg = (obj_rot_mat_prefix[None] @ object_rot_mat[:, :, :].reshape(-1, 3, 3) @ obj_rot_mat_ref).reshape(-1, 3, 3)
                    pred_seq_com_pos_seg = object_trans_orig[:, :, :].reshape(-1, 3)
                    obj_rest_verts_seg = load_object_geometry_w_rest_geo(pred_obj_rot_mat_seg, pred_seq_com_pos_seg, obj_rest_verts[obj_name])
                    indices = torch.randperm(obj_rest_verts_seg.shape[1])[:1024]
                    object_points_temp = obj_rest_verts_seg[:, indices, :].reshape(1, -1, 1024, 3)
                else:
                    object_points_temp = object_points.clone()
                    pred_obj_rot_mat_rel = x[:, :, 219:228].reshape(x.shape[0], -1, 3, 3)
                    
                    obj_rot_mat_ref_temp = obj_rot_mat_ref.unsqueeze(1).repeat(1, pred_obj_rot_mat_rel.shape[1], 1, 1)
                    pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref_temp # [b, t, 3, 3]
                    pred_obj_rot_mat = pred_obj_rot_mat @ pred_obj_rot_mat[:, 0:1, :, :].transpose(2, 3)

                    pred_obj_trans = x[:, :, 216:219] # [b, t, 3]
                    pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)
                    pred_obj_trans = pred_obj_trans - pred_obj_trans[:, 0:1, :]

                    object_points_temp = object_points_temp.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 1024, 3]
                    object_points_temp = torch.matmul(pred_obj_rot_mat, object_points_temp.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 1024, 3]

                x_denorm = self.dataset.denormalize_torch(x0[:, :, :84])
                    
                # dynamically obtain temporal frame indices
                temp_indices = self._get_temp_frame_indices(self.temp_voxel_num)
                if tuple(temp_indices) != FUTURE_OCC_OFFSETS:
                    if self.hsi_future_occ_mode != 'predicted':
                        raise ValueError(
                            "future-occ GT modes require offsets %s, sampler produced %s"
                            % (FUTURE_OCC_OFFSETS, tuple(temp_indices))
                        )
                target_ind = self.mask_ind if self.mask_ind != -1 else 0
                predicted_centers_local = x_denorm[
                    :, list(temp_indices), target_ind * 3: target_ind * 3 + 3
                ]
                oracle_local = self._hsi_future_occ_oracle_local
                if oracle_local is None:
                    if self.hsi_future_occ_mode != 'predicted':
                        raise RuntimeError(
                            "hsi_future_occ_mode=%s requires a window-scoped oracle"
                            % self.hsi_future_occ_mode
                        )
                    crop_centers_local = predicted_centers_local
                    coordinate_centers_local = predicted_centers_local
                else:
                    crop_centers_local, coordinate_centers_local = select_future_occ_centers(
                        predicted_centers_local, oracle_local, self.hsi_future_occ_mode
                    )
                    if diffusion_timestep is not None and self._hsi_future_occ_telemetry is not None:
                        self._hsi_future_occ_telemetry.record(
                            diffusion_timestep, predicted_centers_local, oracle_local
                        )
                
                # only loop when temporal voxels exist
                for temp_position, i in enumerate(temp_indices):
                    mat_for_query = mat.clone()
                    if self.hsi_future_occ_mode in ('gt_crop', 'gt_both'):
                        crop_center_orig = transform_points(
                            crop_centers_local[:, temp_position:temp_position + 1], mat
                        ).squeeze(1)
                        mat_for_query[:, :3, 3] = crop_center_orig
                    else:
                        x0_orig = transform_points(x_denorm, mat)
                        mat_for_query[:, :3, 3] = x0_orig[
                            :, i, target_ind * 3: target_ind * 3 + 3
                        ]
                    mat_for_query[:, 1, 3] = 0
                    query_points = transform_points(self.grid, mat_for_query)

                    if self.hsi_future_occ_mode in ('gt_coordinate', 'gt_both'):
                        coordinate = coordinate_centers_local[:, temp_position, [0, 2]]
                    else:
                        coordinate = x_denorm[:, i, [0, 2]]
                    occ_pos = torch.cat([occ_pos, coordinate[None]], dim=0)

                    occ_temp = self.dataset.get_occ_for_points(query_points, object_points_temp[:, i, :, :], scene_flag)
                    
                    if object_only:
                        occ_temp[occ_temp == 1] = 0.

                    if self.is_mix and torch.logical_not(is_object).any():
                        occ_temp[torch.logical_not(is_object)][occ_temp == 2] = 1.
                    
                    nb_voxels = self.dataset.nb_voxels
                    occ_temp = occ_temp.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()
                    occ_temp = occ_temp.permute(0, 2, 1, 3)

                    occ_list = torch.cat([occ_list, occ_temp], dim=0)

            if self.scene_type == 'occ':
                occ = occ.permute(0, 2, 1, 3)
            elif self.scene_type == 'plane':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
            elif self.scene_type == 'plane_two':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                occ_goal = occ_goal.permute(0, 1, 3, 2)
                occ_goal_cnt = occ_goal * self.occ_idx
                occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_two':
                occ = occ.permute(0, 2, 1, 3)
                occ_goal = occ_goal.permute(0, 2, 1, 3)
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_temp':
                occ = occ_goal.permute(0, 2, 1, 3)

        else:
            occ = None
        return occ, occ_list, occ_pos

    def cm_sample(self, model, x0, x, fixed_points, mat, scene_flag, t, t_index,
                 text_emb, pelvis_goal, scene_goal, object_goal, need_scene,
                 need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, obj_rot_mat_prefix, guidance_fn, guidance_scale, object_only=False, w=None):
        occ, occ_list, occ_pos = self._compute_occ_sample(x, x0, mat, scene_flag, object_points, pelvis_goal, scene_goal, object_goal, is_loco, is_object, need_pelvis_dir, obj_rot_mat_ref, object_only, obj_rest_verts, seq_name_dict, obj_rot_mat_prefix, t_index)

        # if w is None, set the w value based on t_index
        if w is None:
            is_uncondition = False
            w = torch.zeros((self.batch_size, 1), device=x.device)
        elif isinstance(w, (int, float)):
            if w == -1:
                is_uncondition = True
            else:
                is_uncondition = False
            w = torch.full((self.batch_size, 1), w, device=x.device)
        
        if t_index > 0:
            start_timestep = self.solver.ddim_timesteps[t]
            model_output = model(x, occ, start_timestep, text_emb, pelvis_goal, scene_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=True, is_uncondition=is_uncondition, cfg_scale=w)
            
            c_skip, c_out = scalings_for_boundary_conditions(start_timestep)
            c_skip, c_out = [append_dims(item, x.ndim) for item in [c_skip, c_out]]
            
            pred_x_0 = c_skip * x + c_out * model_output
            self.set_fixed_points(pred_x_0, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)
            
            noise = torch.randn(x.shape).to(x.device)

            inference_indices = np.linspace(
                0, len(self.solver.ddim_timesteps), num=self.cm_timesteps, endpoint=False
            )
            inference_indices = np.floor(inference_indices).astype(np.int64)
            inference_indices = (
                torch.from_numpy(inference_indices).long().to(self.solver.ddim_timesteps.device)
            )
            expanded_timestep_index = t.unsqueeze(1).expand(
                -1, inference_indices.size(0)
            )
            last_valid_index = (expanded_timestep_index >= inference_indices).flip(dims=[1]).long().argmax(dim=1)
            last_valid_index = inference_indices.size(0) - 1 - last_valid_index
            timestep_index = inference_indices[last_valid_index]
            alpha_cumprod_prev = extract_into_tensor(
                self.solver.ddim_alpha_cumprods_prev, timestep_index, pred_x_0.shape
            ).float()
            x_prev = alpha_cumprod_prev.sqrt() * pred_x_0 + (1.0 - alpha_cumprod_prev).sqrt() * noise

            if guidance_fn is None:
                return x_prev.float(), occ, pred_x_0
            
            with torch.enable_grad():
                x_start = pred_x_0.detach().requires_grad_(True)
                end_timesteps = self.solver.ddim_timesteps_prev[timestep_index]

                global_jpos = x_start[:, :, :84].reshape(self.batch_size, self.dataset.max_window_size, 84)
                global_jpos = transform_points(self.dataset.denormalize_torch(global_jpos), mat).reshape(self.batch_size, self.dataset.max_window_size, 28, 3)

                # FK to get joint positions.
                rest_human_offsets, transl, betas, gender = human_dict['rest_human_offsets'], human_dict['transl'], human_dict['betas'], human_dict['gender']
                
                curr_seq_local_jpos = rest_human_offsets # [b, t, 24, 3]
                curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [b*t, 24, 3]
                curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]

                global_jrot_6d = x_start[:, :, 84:216].reshape(self.batch_size, self.dataset.max_window_size, 22, 6)
                global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d) # [b, t, 22, 3, 3]
                global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat

                local_jrot_mat = self.dataset.quat_ik_torch(global_jrot_mat.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]
                _, human_jnts = self.dataset.quat_fk_torch(local_jrot_mat, curr_seq_local_jpos) # [b*t, 24, 3]
                human_jnts = human_jnts.reshape(self.batch_size, -1, 24, 3) # [b, t, 24, 3]

                # transl, betas = transl.reshape(-1, 3), betas.reshape(-1, 16)
                
                # root_trans = yup_to_zup(global_jpos.reshape(-1, 28, 3)[:, 0, :] + transl)
                # pose_pred = yup_to_zup(transforms.matrix_to_axis_angle(local_jrot_mat).reshape(-1, 22, 3))
                
                # verts, joints = run_smplx_model(pose_pred, root_trans, betas, 'male', joints_ind=None)
                # verts, joints = zup_to_yup(verts), zup_to_yup(joints)
                # verts = verts.reshape(self.batch_size, self.dataset.max_window_size, -1, 3)

                if not is_object.any():
                    # scene-only batch: human-scene penetration guidance (no object geometry)
                    loss = apply_hsi_guidance_loss(human_jnts, scene_flag, self.dataset.get_nearest_free_voxel)
                else:
                    pred_seq_com_pos = x_start[:, :, 216:219].reshape(self.batch_size, self.dataset.max_window_size, 3)
                    pred_seq_com_pos = transform_points(self.dataset.denormalize_torch(pred_seq_com_pos, is_object=True), mat)

                    object_rot_mat = x_start[:, :, 219:228].reshape(self.batch_size, self.dataset.max_window_size, 3, 3) # B X 16 X 3 X 3

                    if self.dataset.vis:
                        pred_obj_rot_mat = (obj_rot_mat_prefix @ object_rot_mat.reshape(self.batch_size, -1, 3, 3) @ obj_rot_mat_ref)
                    else:
                        pred_obj_rot_mat = (object_rot_mat.reshape(self.batch_size, -1, 3, 3) @ obj_rot_mat_ref)

                    contact_labels = x_start[:, :, 228:232].reshape(self.batch_size, self.dataset.max_window_size, 4)

                    obj_verts = torch.zeros(0, self.dataset.max_window_size, 10000, 3).to(self.device)
                    obj_normals = torch.zeros(0, self.dataset.max_window_size, 10000, 3).to(self.device)

                    for seg_id in range(self.batch_size):
                        obj_name = seq_name_dict[seg_id].split('_')[1]
                        pred_obj_rot_mat_seg = pred_obj_rot_mat[seg_id].reshape(-1, 3, 3)
                        pred_seq_com_pos_seg = pred_seq_com_pos[seg_id].reshape(-1, 3)
                        obj_rest_verts_seg, obj_rest_normals_seg = load_object_geometry_w_rest_geo_and_normals(pred_obj_rot_mat_seg, pred_seq_com_pos_seg, obj_rest_verts[obj_name], obj_vert_normals[obj_name])
                        obj_rest_verts_seg = obj_rest_verts_seg.reshape(1, self.dataset.max_window_size, -1, 3) # 1 X T X Nv X 3
                        obj_rest_normals_seg = obj_rest_normals_seg.reshape(1, self.dataset.max_window_size, -1, 3) # 1 X T X Nv X 3
                        num_obj_verts = obj_rest_verts_seg.shape[2]
                        if num_obj_verts > 10000:
                            # randomly select indices of 10000 points
                            indices = torch.randperm(num_obj_verts)[:10000]
                            obj_rest_verts_seg = obj_rest_verts_seg[:, :, indices, :].reshape(1, self.dataset.max_window_size, 10000, 3)
                            obj_rest_normals_seg = obj_rest_normals_seg[:, :, indices, :].reshape(1, self.dataset.max_window_size, 10000, 3)
                        obj_verts = torch.cat([obj_verts, obj_rest_verts_seg], dim=0)
                        obj_normals = torch.cat([obj_normals, obj_rest_normals_seg], dim=0)

                    assert obj_verts.shape[0] == self.batch_size

                    loss = guidance_fn(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels, scene_flag, self.dataset.get_nearest_free_voxel)

                gradient = torch.autograd.grad(-loss, x_start, retain_graph=True)[0] * guidance_scale
                # penetration_gradient = torch.autograd.grad(-penetration_loss, x_start)[0]
                
                alpha_cumprod = extract(self.alpha_cumprod, end_timesteps, x_start.shape)
                
                x_prev = x_prev + gradient # * (1 - alpha_cumprod)
        else:
            start_timestep = self.solver.ddim_timesteps[t]

            c_skip, c_out = scalings_for_boundary_conditions(start_timestep)
            c_skip, c_out = [append_dims(item, x.ndim) for item in [c_skip, c_out]]

            pred_x_0 = self.student_model(x, occ, start_timestep, text_emb, pelvis_goal, scene_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=True, is_uncondition=is_uncondition, cfg_scale=w)

            pred_x_0 = c_skip * x + c_out * pred_x_0
        
            noise_pred = torch.randn(x.shape).to(x.device)

            x_prev, end_timesteps = self.solver.ddim_style_multiphase_pred(
                    pred_x_0, noise_pred, t, self.cm_timesteps
                )

        return x_prev.float(), occ, pred_x_0

    def p_losses(self, x_start, joints, mat, scene_flag, mask, t, text_emb, pelvis_goal, scene_goal, object_goal, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # ensure all inputs are float type and on the correct device
        noise = noise.to(x_start.device, dtype=torch.float32)
        mask = mask.to(x_start.device, dtype=torch.bool)

        # set the noise in the masked region to 0
        noise[mask] = 0.

        # generate noise data
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy[mask] = x_start[mask] # test
        if self.is_mix:
            # mix: keep object dims (216:) as GT for non-object (scene-only) samples
            x_noisy[torch.logical_not(is_object), :, 216:] = x_start[torch.logical_not(is_object), :, 216:]
        # print('x noisy in mask with scale')

        if self.dataset.load_scene:
            occ, occ_list, occ_pos = self._compute_occ(x_noisy, x_start, joints, mat, scene_flag, object_points, pelvis_goal, scene_goal, object_goal, is_loco, is_object, need_pelvis_dir, obj_rot_mat_ref)
        else:
            occ = None

        # use the model to predict noise
        predicted_noise = self.student_model(x_noisy, occ, t, text_emb, pelvis_goal, scene_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos)
        predicted_noise = rebase_model_output(
            predicted_noise, x_noisy, self.hsi_chain_rebase_mode
        )

        # compute the loss
        mask_inv = torch.logical_not(mask)

        loss_jpos = F.mse_loss(x_start[:, :, :84][mask_inv[:, :, :84]], predicted_noise[:, :, :84][mask_inv[:, :, :84]])
        loss_jrot = F.l1_loss(x_start[:, :, 84:216][mask_inv[:, :, 84:216]], predicted_noise[:, :, 84:216][mask_inv[:, :, 84:216]])

        mask_obj = torch.logical_and(
            mask_inv[:, :, 216:232],
            is_object.to(mask_inv.device, dtype=torch.bool).reshape(-1, 1, 1),
        )
        mask_otrans = mask_obj[:, :, 0:3]
        if mask_otrans.any():
            loss_otrans = F.mse_loss(
                x_start[:, :, 216:219][mask_otrans],
                predicted_noise[:, :, 216:219][mask_otrans],
            )
        else:
            loss_otrans = x_start.new_zeros(())

        mask_orot = mask_obj[:, :, 3:12]
        if mask_orot.any():
            loss_orot = F.l1_loss(
                x_start[:, :, 219:228][mask_orot],
                predicted_noise[:, :, 219:228][mask_orot],
            )
        else:
            loss_orot = x_start.new_zeros(())

        mask_contact = mask_obj[:, :, 12:16]
        if mask_contact.any():
            loss_contact = F.l1_loss(
                x_start[:, :, 228:232][mask_contact],
                predicted_noise[:, :, 228:232][mask_contact],
            )
        else:
            loss_contact = x_start.new_zeros(())

        loss = self.loss_w_jpos * loss_jpos + loss_jrot + loss_otrans + loss_orot + loss_contact

        # The history frames are clean GT at every step (get_mask p=1.0), so the
        # seam second-order residual is exactly the first generated frame's
        # position residual: a_hat[1] - a_star[1] == p_hat[2] - p[2].  Weighting
        # these two frames IS the acceleration match, not an approximation to it.
        # Per-element mean over the slice, matching all five terms above.
        loss_seam = None
        if self.seam_loss_weight > 0.0:
            n = int(self.auto_regre_num)
            loss_seam = F.mse_loss(x_start[:, n:n + 2, :84], predicted_noise[:, n:n + 2, :84])
            loss = loss + self.seam_loss_weight * loss_seam

        human_jnts = None
        loss_fullbody_seam = None
        if self.fullbody_seam_loss_weight > 0.0:
            with torch.autocast(device_type=predicted_noise.device.type, enabled=False):
                geometry_prediction = predicted_noise.float()
                geometry_target = x_start.float()
                geometry_joints = joints.float()
                geometry_mat = mat.float()
                geometry_offsets = rest_human_offsets.float()
                _, human_jnts = self._compute_human_joints(
                    geometry_prediction, geometry_joints, geometry_mat, geometry_offsets
                )
                _, target_human_jnts = self._compute_human_joints(
                    geometry_target, geometry_joints, geometry_mat, geometry_offsets
                )
                loss_fullbody_seam = self._compute_fullbody_seam_loss(
                    human_jnts, target_human_jnts
                )
            loss = loss + self.fullbody_seam_loss_weight * loss_fullbody_seam

        # add object loss (obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts)
        if self.dataset.use_object_keypoints:
            hand_idx_28 = [20, 21, 25, 27]
            hand_idx_24 = [20, 21, 22, 23]
            foot_idx = [7, 8, 10, 11]
            
            if self.geometry_loss_fp32:
                with torch.autocast(device_type=predicted_noise.device.type, enabled=False):
                    geometry_prediction = predicted_noise.float()
                    geometry_joints = joints.float()
                    geometry_mat = mat.float()
                    geometry_offsets = rest_human_offsets.float()
                    gt_global_jpos = transform_points(
                        self.dataset.denormalize_torch(geometry_joints), geometry_mat
                    ).reshape(joints.shape[0], -1, 28, 3)
                    if human_jnts is None:
                        global_jpos, human_jnts = self._compute_human_joints(
                            geometry_prediction, geometry_joints, geometry_mat, geometry_offsets
                        )
            else:
                gt_global_jpos = transform_points(self.dataset.denormalize_torch(joints), mat).reshape(joints.shape[0], -1, 28, 3)
                global_jpos, human_jnts = self._compute_human_joints(
                    predicted_noise, joints, mat, rest_human_offsets
                )
            gt_global_hand_jpos = gt_global_jpos[:, :, hand_idx_28, :]
            gt_global_foot_jpos = gt_global_jpos[:, :, foot_idx, :]

            pred_global_hand_jpos = human_jnts[:, :, hand_idx_24, :]
            pred_global_foot_jpos = human_jnts[:, :, foot_idx, :] # [b, t, 4, 3]

            mask_fk = torch.ones(mask_inv.shape[0], self.dataset.max_window_size, 4, 3, dtype=torch.bool).to(mask_inv.device)
            mask_fk[:, :self.auto_regre_num, :, :] = False
            # The released code built mask_fk here and then never applied it, while
            # consistency_loss (:404-405) does.  The masked frames are the
            # auto_regre_num history frames, which set_fixed_points overwrites at
            # every sampling step and which mask_inv already excludes from all five
            # base losses, so the unmasked term supervised an output nobody reads --
            # and made loss_fk exactly (T - auto_regre_num) / T = 0.875x the
            # consistency stage's on identical geometry, so the two stages'
            # loss_w_fk values were not comparable.
            fk_hand_loss = F.mse_loss(pred_global_hand_jpos[mask_fk], gt_global_hand_jpos[mask_fk])
            fk_foot_loss = F.mse_loss(pred_global_foot_jpos[mask_fk], gt_global_foot_jpos[mask_fk])
            loss_fk = fk_hand_loss + fk_foot_loss
            
            model_mean = predicted_noise # x_start
            pred_obj_rot_mat_rel = model_mean[:, :, 219:228].reshape(joints.shape[0], -1, 3, 3)
            obj_rot_mat_ref = obj_rot_mat_ref.unsqueeze(1).repeat(1, pred_obj_rot_mat_rel.shape[1], 1, 1)
            pred_obj_rot_mat = pred_obj_rot_mat_rel @ obj_rot_mat_ref # [b, t, 3, 3]

            pred_obj_trans = model_mean[:, :, 216:219] # [b, t, 3]
            pred_obj_trans = transform_points(self.dataset.denormalize_torch(pred_obj_trans, is_object=True), mat)

            rest_pose_obj_nn_pts = rest_pose_obj_nn_pts.unsqueeze(1).repeat(1, pred_obj_rot_mat.shape[1], 1, 1) # [b, t, 100, 3]
            pred_seq_obj_kpts = torch.matmul(pred_obj_rot_mat, rest_pose_obj_nn_pts.transpose(-2,-1)).transpose(-2,-1) + pred_obj_trans.unsqueeze(-2) # [b, t, 100, 3]
            
            # transformed_obj_verts = self.dataset.normalize_torch(transformed_obj_verts, is_object=True)
            # pred_seq_obj_kpts = self.dataset.normalize_torch(pred_seq_obj_kpts, is_object=True)
            
            mask_points = torch.ones(mask_inv.shape[0], self.dataset.max_window_size, 100, 3, dtype=torch.bool).to(mask_inv.device)
            mask_points[:, :self.auto_regre_num, :, :] = False
            mask_points = torch.logical_and(mask_points, is_object.to(mask_inv.device, dtype=torch.bool).reshape(-1, 1, 1, 1))

            if mask_points.any():
                loss_object = F.smooth_l1_loss(transformed_obj_verts[mask_points], pred_seq_obj_kpts[mask_points])
            else:
                loss_object = None

        else: 
            loss_object = None
            loss_fk = None

        loss_pen = None
        if self.pen_loss_weight > 0.0:
            if human_jnts is None:
                if self.geometry_loss_fp32:
                    with torch.autocast(device_type=predicted_noise.device.type, enabled=False):
                        _, human_jnts = self._compute_human_joints(
                            predicted_noise.float(), joints.float(), mat.float(),
                            rest_human_offsets.float()
                        )
                else:
                    _, human_jnts = self._compute_human_joints(
                        predicted_noise, joints, mat, rest_human_offsets
                    )

            pen_sdf_bank = self._get_pen_sdf_bank()
            sdf, m_out_of_bounds = pen_sdf_bank.signed_distance(human_jnts, scene_flag)
            m_floor = human_jnts[..., 1] >= self.pen_floor_height
            m_hist = torch.ones_like(m_floor, dtype=torch.bool)
            m_hist[:, :int(self.auto_regre_num)] = False
            m_finite = torch.isfinite(sdf)
            m_inbound = torch.logical_not(m_out_of_bounds)
            m_scorable = m_floor & m_hist & m_finite & m_inbound

            d = torch.clamp(-(sdf + self.pen_delta), min=0.0)
            if m_scorable.any():
                loss_pen = (d ** 2)[m_scorable].mean()
            else:
                loss_pen = predicted_noise.new_zeros(())
            loss = loss + self.pen_loss_weight * loss_pen

        if occ_list is not None:
            del occ_list
        if occ is not None:
            del occ
                
        return dict(
            loss=loss,
            loss_object=loss_object,
            loss_fk=loss_fk,
            loss_seam=loss_seam,
            loss_fullbody_seam=loss_fullbody_seam,
            loss_pen=loss_pen,
            loss_jpos=loss_jpos,
            loss_jrot=loss_jrot,
        )

    @torch.no_grad()
    def p_sample_loop(self, fixed_points, mat, scene_flag, text_emb, pelvis_goal, scene_goal, object_goal, \
                    need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, guidance_fn, guidance_scale, obj_rot_mat_prefix=None, object_only=False):
        self.batch_size = fixed_points.shape[0]
        device = next(self.student_model.parameters()).device
        shape = (self.batch_size, self.dataset.max_window_size, self.channel)
        points = torch.randn(shape, device=device)

        if self.auto_regre_num > 0:
            self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)
        imgs = []
        occs = []
        x0 = []
        x0.append(points)
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            model_used = self.student_model

            points, occ, pred_x0 = self.p_sample(model_used, x0[-1], points, fixed_points, mat, scene_flag,
                                        torch.full((self.batch_size,), i, device=device, dtype=torch.long), i,
                                        text_emb, pelvis_goal, scene_goal, object_goal, need_scene,
                                        need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, guidance_fn, guidance_scale, obj_rot_mat_prefix, object_only)
            if self.auto_regre_num > 0:
                self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)

            points_orig = points
            imgs.append(points_orig)
            x0.append(pred_x0)
            if occ is not None:
                occs.append(occ.cpu().numpy())

        return imgs, occs

    @torch.no_grad()
    def p_sample(self, model, x0, x, fixed_points, mat, scene_flag, t, t_index,
                 text_emb, pelvis_goal, scene_goal, object_goal, need_scene,
                 need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, guidance_fn, guidance_scale, obj_rot_mat_prefix=None, object_only=False):
        occ, occ_list, occ_pos = self._compute_occ_sample(x, x0, mat, scene_flag, object_points, pelvis_goal, scene_goal, object_goal, is_loco, is_object, need_pelvis_dir, obj_rot_mat_ref, object_only, obj_rest_verts, seq_name_dict, obj_rot_mat_prefix, t_index)

        cond_model_output = model(x, occ, t, text_emb, pelvis_goal, scene_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=True)

        uncond_model_output = model(x, occ, t, text_emb, pelvis_goal, scene_goal, is_loco, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=True, is_uncondition=True)

        model_output = cond_model_output + self.w * (cond_model_output - uncond_model_output)

        # After CFG, before the posterior mean: the 16 object channels of the
        # predicted x0 are forced to their training value (exactly 0) on scene-only
        # rows.  Off by default and then the identity object, so a sealed guided cell
        # reproduces bitwise.  It changes NO random draw -- torch.randn_like below is
        # called with the same shape in the same order either way -- so the A/B
        # differs only in the arithmetic on those 16 channels and in whatever the
        # trunk then makes of them at the next step.
        model_output = zero_object_x0(model_output, is_object, self.hsi_zero_object_x0)
        model_output = rebase_model_output(
            model_output,
            x,
            self.hsi_chain_rebase_mode,
            self._hsi_chain_rebase_oracle_frame2,
            timestep=t_index,
            min_timestep=self.hsi_chain_rebase_min_timestep,
        )
        if self._p_sample_trace_timestep == int(t_index):
            if self._p_sample_trace is not None:
                raise RuntimeError("p-sample trace timestep was visited more than once")
            self._p_sample_trace = model_output.detach().clone()

        model_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * model_output +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )

        if t_index == 0:
            return model_mean, occ, model_output
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            # return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x), occ
            model_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
            x_prev = model_mean + (0.5 * model_log_variance).exp() * torch.randn_like(x)

            if guidance_fn is None:
                return x_prev, occ, model_output

            with torch.enable_grad():
                x_start = model_output.detach().requires_grad_(True)

                global_jpos = x_start[:, :, :84].reshape(self.batch_size, self.dataset.max_window_size, 84)
                global_jpos = transform_points(self.dataset.denormalize_torch(global_jpos), mat).reshape(self.batch_size, self.dataset.max_window_size, 28, 3)

                # FK to get joint positions.
                rest_human_offsets, transl, betas, gender = human_dict['rest_human_offsets'], human_dict['transl'], human_dict['betas'], human_dict['gender']
                
                curr_seq_local_jpos = rest_human_offsets # [b, t, 24, 3]
                curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [b*t, 24, 3]
                curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(-1, 28, 3)[:, 0, :]

                global_jrot_6d = x_start[:, :, 84:216].reshape(self.batch_size, self.dataset.max_window_size, 22, 6)
                global_jrot_mat = transforms.rotation_6d_to_matrix(global_jrot_6d) # [b, t, 22, 3, 3]
                global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat

                local_jrot_mat = self.dataset.quat_ik_torch(global_jrot_mat.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]
                _, human_jnts = self.dataset.quat_fk_torch(local_jrot_mat, curr_seq_local_jpos) # [b*t, 24, 3]
                human_jnts = human_jnts.reshape(self.batch_size, -1, 24, 3) # [b, t, 24, 3]

                if not is_object.any():
                    # scene-only batch: human-scene penetration guidance (no object geometry)
                    loss = self._hsi_guidance_loss(human_jnts, scene_flag)
                else:
                    pred_seq_com_pos = x_start[:, :, 216:219].reshape(self.batch_size, self.dataset.max_window_size, 3)
                    pred_seq_com_pos = transform_points(self.dataset.denormalize_torch(pred_seq_com_pos, is_object=True), mat)

                    object_rot_mat = x_start[:, :, 219:228].reshape(self.batch_size, self.dataset.max_window_size, 3, 3) # B X 16 X 3 X 3

                    if self.dataset.vis:
                        pred_obj_rot_mat = (obj_rot_mat_prefix @ object_rot_mat.reshape(self.batch_size, -1, 3, 3) @ obj_rot_mat_ref)
                    else:
                        pred_obj_rot_mat = (object_rot_mat.reshape(self.batch_size, -1, 3, 3) @ obj_rot_mat_ref)

                    contact_labels = x_start[:, :, 228:232].reshape(self.batch_size, self.dataset.max_window_size, 4)

                    obj_verts = torch.zeros(0, self.dataset.max_window_size, 10000, 3).to(self.device)
                    obj_normals = torch.zeros(0, self.dataset.max_window_size, 10000, 3).to(self.device)

                    for seg_id in range(self.batch_size):
                        obj_name = seq_name_dict[seg_id].split('_')[1]
                        pred_obj_rot_mat_seg = pred_obj_rot_mat[seg_id].reshape(-1, 3, 3)
                        pred_seq_com_pos_seg = pred_seq_com_pos[seg_id].reshape(-1, 3)
                        obj_rest_verts_seg, obj_rest_normals_seg = load_object_geometry_w_rest_geo_and_normals(pred_obj_rot_mat_seg, pred_seq_com_pos_seg, obj_rest_verts[obj_name], obj_vert_normals[obj_name])
                        obj_rest_verts_seg = obj_rest_verts_seg.reshape(1, self.dataset.max_window_size, -1, 3) # 1 X T X Nv X 3
                        obj_rest_normals_seg = obj_rest_normals_seg.reshape(1, self.dataset.max_window_size, -1, 3) # 1 X T X Nv X 3
                        num_obj_verts = obj_rest_verts_seg.shape[2]
                        if num_obj_verts > 10000:
                            # randomly select indices of 10000 points
                            indices = torch.randperm(num_obj_verts)[:10000]
                            obj_rest_verts_seg = obj_rest_verts_seg[:, :, indices, :].reshape(1, self.dataset.max_window_size, 10000, 3)
                            obj_rest_normals_seg = obj_rest_normals_seg[:, :, indices, :].reshape(1, self.dataset.max_window_size, 10000, 3)
                        obj_verts = torch.cat([obj_verts, obj_rest_verts_seg], dim=0)
                        obj_normals = torch.cat([obj_normals, obj_rest_normals_seg], dim=0)

                    assert obj_verts.shape[0] == self.batch_size

                    loss = guidance_fn(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels, scene_flag, self.dataset.get_nearest_free_voxel)

                gradient = torch.autograd.grad(-loss, x_start, retain_graph=True)[0] * guidance_scale
                if self.hsi_guidance_posterior_coef1:
                    gradient = gradient * extract(self.posterior_mean_coef1, t, x.shape)
                frame_weights = hsi_guidance_frame_weights(
                    gradient.shape[1],
                    device=gradient.device,
                    dtype=gradient.dtype,
                    enabled=self.hsi_guidance_frame_ramp,
                )
                if frame_weights is not None:
                    gradient = gradient * frame_weights.view(1, -1, 1)
                # Per-step trust region on the guidance increment, per sample so the
                # branch cannot key on sample 0 and break layout neutrality.  Off by
                # default; the released path adds the increment unnormalised 499 times
                # per window, which the 2026-08-23 2x2 tied to 100% of the >5g root
                # accelerations.  Diffusion path only -- cm_sample is untouched.
                cap = self.hsi_guidance_norm_cap
                if cap is not None:
                    gradient = cap_guidance_increment(gradient, cap)
                # Order: trust region on the raw increment, then the schedule decay,
                # then the constant dose scale.  Each is independent and off by default.
                if self.hsi_guidance_alpha_decay:
                    # The factor the released author wrote and commented out on the
                    # consistency path.  alpha_cumprod -> 1 as t -> 0, so this SUPPRESSES
                    # guidance on the late low-noise refinement steps and keeps it on the
                    # early high-noise ones.  Measured effect on total per-window
                    # displacement over the frozen worst-20: 0.596x.
                    gradient = gradient * (1.0 - extract(self.alpha_cumprod, t, x.shape))
                if self.hsi_guidance_dose_scale is not None:
                    gradient = gradient * self.hsi_guidance_dose_scale
                x_prev = x_prev + gradient

            return x_prev, occ, model_output
    

    def set_fixed_points(self, img, goal, fixed_points, mat, joint_id, fix_mode, fix_goal):
        '''
        set fixed points of goal and prefix frames

        img: [b, max_window_size, 3 * joint_num]
        fixed_points: [b, auto_regre_num, 3 * joint_num]

        '''

        if goal is not None and fix_goal:
            goal_len = goal.shape[1]
            goal = self.dataset.normalize_torch(transform_points(goal, torch.inverse(mat)))

            img[:, -goal_len:, joint_id * 3] = goal[:, :, 0]
            if joint_id != 0:
                img[:, -goal_len:, joint_id * 3 + 1] = goal[:, :, 1]
            img[:, -goal_len:, joint_id * 3 + 2] = goal[:, :, 2]

        if fixed_points is not None and fix_mode:
            img[:, :fixed_points.shape[1], :] = fixed_points

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=500, ddim_timesteps=25):
        self.step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * self.step_ratio
        ).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_timesteps_prev = np.asarray([0] + self.ddim_timesteps[:-1].tolist())
        # self.ddim_alpha_cumprods_prev = np.asarray(
        #     [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        # )
        self.ddim_alpha_cumprods_prev = np.asarray(
            [1.0] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_timesteps_prev = torch.from_numpy(self.ddim_timesteps_prev).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_timesteps_prev = self.ddim_timesteps_prev.to(device)

        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

    def ddim_style_multiphase_pred(self, pred_x0, pred_noise, timestep_index, multiphase):
        inference_indices = np.linspace(
            0, len(self.ddim_timesteps), num=multiphase, endpoint=False
        )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = (
            torch.from_numpy(inference_indices).long().to(self.ddim_timesteps.device)
        )
        expanded_timestep_index = timestep_index.unsqueeze(1).expand(
            -1, inference_indices.size(0)
        )
        valid_indices_mask = expanded_timestep_index >= inference_indices
        last_valid_index = valid_indices_mask.flip(dims=[1]).long().argmax(dim=1)
        last_valid_index = inference_indices.size(0) - 1 - last_valid_index
        timestep_index = inference_indices[last_valid_index]
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt

        return x_prev, self.ddim_timesteps_prev[timestep_index]


class Unet(nn.Module):
    def __init__(
            self,
            dim_model,
            num_heads,
            num_layers,
            dropout_p,
            dim_input,
            dim_output,
            nb_voxels=None,
            temp_voxel_num=3,
            free_p=0.1,
            load_scene=True,
            load_language=True,
            load_scene_goal=True,
            load_pelvis_goal=True,
            load_object_goal=True,
            is_mix=False,
            language_feature_dim=768,
            scene_type=None,
            **kwargs
    ):
        super().__init__()

        self.dim_model = dim_model
        self.load_scene = load_scene
        self.load_language = load_language
        self.load_scene_goal = load_scene_goal
        self.load_pelvis_goal = load_pelvis_goal
        self.load_object_goal = load_object_goal
        self.is_mix = is_mix
        self.scene_type = scene_type
        self.temp_voxel_num = temp_voxel_num  # store the number of temporal voxels

        if self.scene_type == 'plane':
            vit_channels = 1
        elif self.scene_type == 'occ':
            vit_channels = nb_voxels[1]
        elif self.scene_type == 'plane_two':
            vit_channels = 2
        elif self.scene_type == 'occ_two':
            vit_channels = 2*nb_voxels[1]
        elif self.scene_type == 'occ_temp':
            vit_channels = nb_voxels[1]

        if self.load_scene:
            self.scene_embedding = ViT(
                image_size=nb_voxels[0],
                patch_size=8,
                channels=vit_channels,
                num_classes=dim_model,
                dim=512,
                depth=6,
                heads=16,
                mlp_dim=1024,
                dropout=0.1,
                emb_dropout=0.1
            )
        self.free_p = free_p
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )

        self.embedding_input = nn.Linear(dim_input, dim_model)
        self.embedding_output = nn.Linear(dim_output, dim_model)

        if self.load_language:
            self.embedding_language = LanguageEncoder(dim_output=dim_model, dim_input=language_feature_dim)

        if self.load_scene_goal:
            self.embedding_scene_goal = GoalEncoder(mode='scene', dim_output=dim_model)

        if self.load_pelvis_goal:
            self.embedding_pelvis_goal = GoalEncoder(mode='pelvis', dim_output=dim_model)

        if self.load_object_goal:
            self.embedding_object_goal = GoalEncoder(mode='object', dim_output=dim_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_model,
                                                   dropout=dropout_p,
                                                   activation="gelu")

        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers
        )

        self.out = nn.Linear(dim_model, dim_output)

        self.embed_timestep = TimestepEmbedder(self.dim_model, self.positional_encoder)

        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024*3, out_features=768),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=self.dim_model),
            )
        
        # add the CFG scale embedding module
        self.cfg_scale_embedding = CFGScaleEmbedding(dim_model)
        
        self.division_term = torch.exp(
            torch.arange(0, dim_model//2, 2).float() * (-math.log(10000.0)) / (dim_model//2))  # 1000^(2i/dim_model)
        # self.register_buffer("division_term", division_term)

    def encode_2d_coordinate(self, pos, dim_model=512):
        # pos: [b, 2]
        # dim_model: int
        # return: [b, dim_model]
        pe_x = torch.zeros(pos.shape[0], dim_model//2).to(pos.device)
        pe_x[:, 0::2] = torch.sin(pos[:, 0:1] * self.division_term.to(pos.device))
        pe_x[:, 1::2] = torch.cos(pos[:, 0:1] * self.division_term.to(pos.device))

        pe_y = torch.zeros(pos.shape[0], dim_model//2).to(pos.device)
        pe_y[:, 0::2] = torch.sin(pos[:, 1:] * self.division_term.to(pos.device))
        pe_y[:, 1::2] = torch.cos(pos[:, 1:] * self.division_term.to(pos.device))

        return torch.cat((pe_x, pe_y), dim=1)[:, None, :] / math.sqrt(dim_model // 2)

    def forward(self, x, cond, timesteps, text_emb, pelvis_goal, scene_goal, is_loco, need_scene, need_pelvis_dir, \
                pi, end_pi, seq_length, need_pi, object_goal, is_object, obj_bps_data, occ_list, occ_pos, is_sample=False, is_uncondition=False, mask_timestep=10, cfg_scale=None):
        """
        Forward function, ensures all inputs have the correct type and device
        
        Args:
            x: input human and object motion [batch_size, seq_len, dim_input]
            cond: scene condition
            timesteps: timestep
            text_emb: text embedding
            pelvis_goal: pelvis goal position
            scene_goal: scene-interaction goal position
            is_loco: locomotion flag (mix: routes scene/pelvis goal; non-mix: unused)
            need_scene: scene-needed flag
            need_pelvis_dir: pelvis-direction-needed flag
            pi: progress indicator
            need_pi: progress-indicator-needed flag
            object_goal: object goal position
            is_object: object-present flag
            obj_bps_data: object BPS data [batch_size, 1024, 3]
        
        Returns:
            output: predicted noise
        """
        # ensure all inputs are float type
        x = x.to(dtype=torch.float32)
        self.batch_size = x.shape[0]
        
        if cond is not None:
            cond = cond.to(dtype=torch.float32)
        timesteps = timesteps.to(dtype=torch.long)
        text_emb = text_emb.to(dtype=torch.float32)
        pelvis_goal = pelvis_goal.to(dtype=torch.float32)
        scene_goal = scene_goal.to(dtype=torch.float32)
        object_goal = object_goal.to(dtype=torch.float32)
        obj_bps_data = obj_bps_data.to(dtype=torch.float32)
        
        t_emb = self.embed_timestep(timesteps)  # [b, 1, d]
        
        # if a CFG scale is provided, add the CFG embedding
        if cfg_scale is not None:
            if int(timesteps[0]) == 499 or is_uncondition: # todo: this should adapt to the timestep
                cfg_scale = torch.full((self.batch_size, 1), -1.0, device=x.device)
            cfg_emb = self.cfg_scale_embedding(cfg_scale)
            # add the CFG embedding to the timestep embedding
            t_emb = t_emb + cfg_emb.unsqueeze(1)

        if not self.load_scene:
            scene_emb = torch.zeros_like(t_emb)
            scene_emb_0 = torch.zeros_like(t_emb)
            scene_emb_1 = torch.zeros_like(t_emb)
            scene_emb_2 = torch.zeros_like(t_emb)
            scene_emb_3 = torch.zeros_like(t_emb)
        else:
            scene_emb = self.scene_embedding(cond).reshape(-1, 1, self.dim_model)
            scene_emb += self.encode_2d_coordinate(occ_pos[0], self.dim_model)

            if self.scene_type == 'occ_temp':
                scene_all = self.scene_embedding(occ_list).reshape(-1, 1, self.dim_model)
                
                # dynamically handle scene embedding
                scene_embs = []
                if self.temp_voxel_num == 0:
                    # current frame only, no temporal voxels
                    scene_emb_0 = scene_all
                    scene_emb_0 += self.encode_2d_coordinate(torch.zeros(self.batch_size, 2).to(occ_pos.device), self.dim_model)
                    scene_embs = [scene_emb_0]
                elif self.temp_voxel_num == 1:
                    # current frame + 1 temporal voxel
                    scene_emb_0 = scene_all[0:scene_all.shape[0]//2]
                    scene_emb_1 = scene_all[scene_all.shape[0]//2:]
                    scene_emb_0 += self.encode_2d_coordinate(torch.zeros(self.batch_size, 2).to(occ_pos.device), self.dim_model)
                    scene_emb_1 += self.encode_2d_coordinate(occ_pos[1], self.dim_model)
                    scene_embs = [scene_emb_0, scene_emb_1]
                elif self.temp_voxel_num == 2:
                    # current frame + 2 temporal voxels
                    scene_emb_0 = scene_all[0:scene_all.shape[0]//3]
                    scene_emb_1 = scene_all[scene_all.shape[0]//3:scene_all.shape[0]//3*2]
                    scene_emb_2 = scene_all[scene_all.shape[0]//3*2:]
                    
                    scene_emb_0 += self.encode_2d_coordinate(torch.zeros(self.batch_size, 2).to(occ_pos.device), self.dim_model)
                    scene_emb_1 += self.encode_2d_coordinate(occ_pos[1], self.dim_model)
                    scene_emb_2 += self.encode_2d_coordinate(occ_pos[2], self.dim_model)
                    scene_embs = [scene_emb_0, scene_emb_1, scene_emb_2]
                elif self.temp_voxel_num == 3:
                    # current frame + 3 temporal voxels (original implementation)
                    scene_emb_0 = scene_all[0:scene_all.shape[0]//4]
                    scene_emb_1 = scene_all[scene_all.shape[0]//4:scene_all.shape[0]//2]
                    scene_emb_2 = scene_all[scene_all.shape[0]//2:scene_all.shape[0]//4*3]
                    scene_emb_3 = scene_all[scene_all.shape[0]//4*3:scene_all.shape[0]]
                    
                    scene_emb_0 += self.encode_2d_coordinate(torch.zeros(self.batch_size, 2).to(occ_pos.device), self.dim_model)
                    scene_emb_1 += self.encode_2d_coordinate(occ_pos[1], self.dim_model)
                    scene_emb_2 += self.encode_2d_coordinate(occ_pos[2], self.dim_model)
                    scene_emb_3 += self.encode_2d_coordinate(occ_pos[3], self.dim_model)
                    scene_embs = [scene_emb_0, scene_emb_1, scene_emb_2, scene_emb_3]

                # handle dropout during sampling (temporal voxels only)
                if is_sample:
                    if int(timesteps[0]) == 499 or is_uncondition:
                        for i in range(1, len(scene_embs)):
                            scene_embs[i] = torch.zeros_like(t_emb)
                # when cfg_scale=-1, mask the scene condition (unconditional generation), Training
                elif cfg_scale is not None:
                    is_uncond = (cfg_scale == -1).squeeze(1)
                    if is_uncond.any():
                        mask = is_uncond.unsqueeze(1).unsqueeze(2)
                        for i in range(1, len(scene_embs)):
                            scene_embs[i] = torch.where(mask, torch.zeros_like(scene_embs[i]), scene_embs[i])
                else:
                    prob_mask = (torch.rand(scene_embs[0].size(0), 1, 1, device=scene_embs[0].device) < 0.1)
                    for i in range(1, len(scene_embs)):
                        scene_embs[i] = torch.where(prob_mask, torch.zeros_like(scene_embs[i]), scene_embs[i])
                
                # keep the original variable names for backward compatibility
                if self.temp_voxel_num == 0:
                    scene_emb_0 = scene_embs[0]
                    scene_emb_1 = torch.zeros_like(t_emb)
                    scene_emb_2 = torch.zeros_like(t_emb)
                    scene_emb_3 = torch.zeros_like(t_emb)
                elif self.temp_voxel_num == 1:
                    scene_emb_0 = scene_embs[0]
                    scene_emb_1 = scene_embs[1]
                    scene_emb_2 = torch.zeros_like(t_emb)
                    scene_emb_3 = torch.zeros_like(t_emb)
                elif self.temp_voxel_num == 2:
                    scene_emb_0 = scene_embs[0]
                    scene_emb_1 = scene_embs[1]
                    scene_emb_2 = scene_embs[2]
                    scene_emb_3 = torch.zeros_like(t_emb)
                elif self.temp_voxel_num == 3:
                    scene_emb_0 = scene_embs[0]
                    scene_emb_1 = scene_embs[1]
                    scene_emb_2 = scene_embs[2]
                    scene_emb_3 = scene_embs[3]

            else:
                scene_emb_0 = torch.zeros_like(t_emb)
                scene_emb_1 = torch.zeros_like(t_emb)
                scene_emb_2 = torch.zeros_like(t_emb)
                scene_emb_3 = torch.zeros_like(t_emb)

            not_need_scene = torch.logical_not(need_scene)
            scene_emb[not_need_scene] = 0.
            scene_emb_0[not_need_scene] = 0.
            scene_emb_1[not_need_scene] = 0.
            scene_emb_2[not_need_scene] = 0.
            scene_emb_3[not_need_scene] = 0.

        if not self.load_language:
            language_emb = torch.zeros_like(t_emb)
        else:
            language_emb = self.embedding_language(text_emb, pi, end_pi, seq_length, need_pi)

        # NOTE: slot-7 arg `is_loco` selects goal routing. In mix mode it gates scene vs pelvis
        # goal; in non-mix mode scene-goal conditioning is unused. self.is_mix picks the trained routing.
        if not self.load_scene_goal:
            scene_goal_emb = torch.zeros_like(t_emb)
        else:
            scene_goal_emb = self.embedding_scene_goal(scene_goal)
            if self.is_mix:
                # mix: scene_goal = scene-interaction (sit/lie) goal, active only for non-loco
                scene_goal_emb[is_loco] = 0.
            else:
                # non-mix: scene-goal conditioning is unused (legacy is_pick was always False,
                # so every row was zeroed); preserve that exact behavior.
                scene_goal_emb[:] = 0.

        if not self.load_pelvis_goal:
            pelvis_goal_emb = torch.zeros_like(t_emb)
        else:
            pelvis_goal_emb = self.embedding_pelvis_goal(pelvis_goal)
            not_need_pelvis_dir = torch.logical_not(need_pelvis_dir)
            pelvis_goal_emb[not_need_pelvis_dir] = 0.
            if self.is_mix:
                # mix: pelvis goal active only for loco
                pelvis_goal_emb[torch.logical_not(is_loco)] = 0.

        if not self.load_object_goal:
            object_goal_emb = torch.zeros_like(t_emb)
        else:
            object_goal_emb = self.embedding_object_goal(object_goal)
            not_need_object = torch.logical_not(is_object)
            object_goal_emb[not_need_object] = 0.

        if not self.load_object_goal:
            obj_bps_data_emb = torch.zeros_like(t_emb)
        else:
            # ensure obj_bps_data is float type
            obj_bps_data = obj_bps_data.float()
            obj_bps_data_emb = obj_bps_data.reshape(-1, 1024*3)
            obj_bps_data_emb = self.bps_encoder(obj_bps_data_emb)
            obj_bps_data_emb = obj_bps_data_emb.reshape(-1, 1, self.dim_model)
            # for samples that do not need an object, set the object BPS feature to 0
            not_need_object = torch.logical_not(is_object)
            obj_bps_data_emb[not_need_object] = 0.

        t_emb = t_emb.permute(1, 0, 2)
        scene_emb = scene_emb.permute(1, 0, 2)
        scene_emb_0 = scene_emb_0.permute(1, 0, 2)
        scene_emb_1 = scene_emb_1.permute(1, 0, 2)
        scene_emb_2 = scene_emb_2.permute(1, 0, 2)
        scene_emb_3 = scene_emb_3.permute(1, 0, 2)
        language_emb = language_emb.permute(1, 0, 2)
        scene_goal_emb = scene_goal_emb.permute(1, 0, 2)
        pelvis_goal_emb = pelvis_goal_emb.permute(1, 0, 2)
        object_goal_emb = object_goal_emb.permute(1, 0, 2)
        obj_bps_data_emb = obj_bps_data_emb.permute(1, 0, 2)

        scene_emb = t_emb + scene_emb
        scene_emb_0 = t_emb + scene_emb_0
        scene_emb_1 = t_emb + scene_emb_1
        scene_emb_2 = t_emb + scene_emb_2
        scene_emb_3 = t_emb + scene_emb_3
        language_emb = t_emb + language_emb
        scene_goal_emb = t_emb + scene_goal_emb
        pelvis_goal_emb = t_emb + pelvis_goal_emb
        object_goal_emb = t_emb + object_goal_emb
        obj_bps_data_emb = t_emb + obj_bps_data_emb

        x = x.permute(1, 0, 2)
        x = self.embedding_input(x) * math.sqrt(self.dim_model)
        
        if self.scene_type == 'occ_temp':
            if self.temp_voxel_num == 0:
                # current frame only
                x = torch.cat((scene_emb, language_emb, scene_goal_emb, pelvis_goal_emb, 
                              object_goal_emb, obj_bps_data_emb, scene_emb_0, x), dim=0)
            elif self.temp_voxel_num == 1:
                # current frame + 1 temporal voxel
                x = torch.cat((scene_emb, language_emb, scene_goal_emb, pelvis_goal_emb, 
                              object_goal_emb, obj_bps_data_emb, scene_emb_0, x, 
                              scene_emb_1), dim=0)
            elif self.temp_voxel_num == 2:
                # current frame + 2 temporal voxels
                x = torch.cat((scene_emb, language_emb, scene_goal_emb, pelvis_goal_emb, 
                              object_goal_emb, obj_bps_data_emb, scene_emb_0, scene_emb_1, scene_emb_2, x), dim=0)
            elif self.temp_voxel_num == 3:
                # current frame + 3 temporal voxels (original implementation)
                x = torch.cat((scene_emb, language_emb, scene_goal_emb, pelvis_goal_emb, 
                              object_goal_emb, obj_bps_data_emb, scene_emb_0, x[0:5], 
                              scene_emb_1, x[5:10], scene_emb_2, x[10:15], 
                              scene_emb_3, x[15:]), dim=0)
        else:
            x = torch.cat((scene_emb, language_emb, scene_goal_emb, pelvis_goal_emb, object_goal_emb, obj_bps_data_emb, x), dim=0)
        
        x = self.positional_encoder(x)
        x = self.transformer(x)

        if self.scene_type == 'occ_temp':
            if self.temp_voxel_num == 0:
                # output indices: skip the first 7 (6 embeddings + 1 scene_emb_0)
                expected_len = 23  # 6 + 1 + 16
                assert x.shape[0] == expected_len, f"Expected {expected_len} but got {x.shape[0]}"
                x_index = list(range(7, 23))
            elif self.temp_voxel_num == 1:
                # output indices: skip the embedding and scene_emb positions
                expected_len = 24  # 6 + 2 + 16
                assert x.shape[0] == expected_len, f"Expected {expected_len} but got {x.shape[0]}"
                x_index = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
            elif self.temp_voxel_num == 2:
                # output indices: skip the embedding and scene_emb positions
                expected_len = 25  # 6 + 3 + 16
                assert x.shape[0] == expected_len, f"Expected {expected_len} but got {x.shape[0]}"
                x_index = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            elif self.temp_voxel_num == 3:
                # original implementation
                expected_len = 26  # 6 + 4 + 16
                assert x.shape[0] == expected_len, f"Expected {expected_len} but got {x.shape[0]}"
                x_index = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25]
            
            output = self.out(x)[x_index]
        else:
            output = self.out(x)[6:]
        output = output.permute(1, 0, 2)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).reshape(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pos_encoding[timesteps])


class CFGScaleEmbedding(nn.Module):
    """Fourier embedding module for the CFG scale"""
    def __init__(self, dim_model, max_period=10000, init_scale=0.0001):
        super().__init__()
        self.dim_model = dim_model
        
        # Fourier frequencies
        half_dim = dim_model // 2
        self.freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim) / half_dim
        )
        
        # projection layer
        self.proj = nn.Linear(dim_model, dim_model)
        # small-value initialization
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_scale)
        nn.init.constant_(self.proj.bias, 0.0)
        
    def forward(self, w):
        """
        Args:
            w: CFG scale [batch_size, 1]
        Returns:
            w_emb: CFG embedding [batch_size, dim_model]
        """
        # expand the w dimension
        w = w.unsqueeze(-1) * 1000  # [batch_size, 1, 1]
        
        # compute Fourier features
        self.freqs = self.freqs.to(w.device)
        args = w * self.freqs  # [batch_size, 1, half_dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [batch_size, 1, dim_model]
        embedding = embedding.squeeze(1)  # [batch_size, dim_model]
        
        # pass through the zero-initialized projection layer
        w_emb = self.proj(embedding)
        
        return w_emb


class DynamicProgressEmbedding(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder, dropout_p=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(dropout_p)
        # MLP that fuses start and end position information
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.sequence_pos_encoder = sequence_pos_encoder
        
    def forward(self, timesteps_start, timesteps_end, seq_length):
        # obtain the encodings of the start and end positions
        start_encoding = self.sequence_pos_encoder.pos_encoding[timesteps_start]  # [B, D]
        end_encoding = self.sequence_pos_encoder.pos_encoding[timesteps_end]      # [B, D]
        len_encoding = self.sequence_pos_encoder.pos_encoding[seq_length]

        # fuse the two position encodings
        combined = torch.cat([start_encoding, end_encoding, len_encoding], dim=-1)  # [B, 3D]
        progress_embedding = self.fusion(combined)      # [B, D]
        
        return self.dropout(progress_embedding)


class LanguageEncoder(nn.Module):
    def __init__(self, dim_output, dim_input, **kwargs):
        super().__init__()
        self.dim_model = dim_output

        self.embedding_input1 = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

        self.embedding_input2 = nn.Sequential(
            nn.Linear(dim_output, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

        self.positional_encoder = PositionalEncoding(
            dim_model=dim_output, dropout_p=0.1, max_len=5000
        )

        self.embed_pi = DynamicProgressEmbedding(dim_output, self.positional_encoder)

    def forward(self, x, pi, end_pi, seq_length, need_pi):
        # x.shape: [b, 1, 768]

        x = self.embedding_input1(x)
        pi = self.embed_pi(pi, end_pi, seq_length)

        # normalization
        pi = pi / np.sqrt(self.dim_model // 2)
        not_need_pi = torch.logical_not(need_pi)
        pi[not_need_pi] = 0.
        x = x + pi
        x = self.embedding_input2(x)
        return x

class GoalEncoder(nn.Module):
    def __init__(self, mode, dim_output, **kwargs):
        super().__init__()

        self.mode = mode
        if mode == 'pelvis':
            self.embedding_input = nn.Sequential(nn.Linear(2, dim_output),
                                                    nn.SiLU(inplace=False),
                                                    nn.Linear(dim_output, dim_output))
        elif mode == 'scene':
            self.embedding_input = nn.Sequential(nn.Linear(3, dim_output),
                                                    nn.SiLU(inplace=False),
                                                    nn.Linear(dim_output, dim_output))
        elif mode == 'object':
            self.embedding_input = nn.Sequential(nn.Linear(3, dim_output),
                                                    nn.SiLU(inplace=False),
                                                    nn.Linear(dim_output, dim_output))

    def forward(self, x):
        # x.shape: [b, 3] (includes object_goal)
        if self.mode == 'pelvis':
            x = x[..., [0, 2]]  # use only the x and z coordinates
        x = self.embedding_input(x)
        x = x.reshape(-1, 1, x.shape[-1])
        return x
