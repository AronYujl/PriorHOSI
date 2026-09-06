"""One reverse chain, two experts, composed at every step.

The naive alternative -- let each expert run its own 500-step chain and average
the two finished windows -- is wrong, and not marginally.  A diffusion model's
output distribution is multimodal; the mean of two independent samples from a
multimodal distribution is generally not itself a sample.  Averaging one expert's
"walk left around the table" with the other's "walk right" produces a motion that
neither expert would ever emit and that goes through the table.  priorMDM
(2303.01418) and MixerMDM (2504.01019) both compose per denoising step for this
reason, and the preregistered operator is written on x_hat_0 -- a per-step
quantity -- not on the finished window.

So there is ONE chain.  At each reverse step both experts see the same ``current``
state, each predicts its own x_hat_0 under its own inference convention, the two
predictions are blended by ``compose_x0``, and the shared posterior advances the
single chain from the blend.  The experts stay coupled because they never see
diverging inputs.

The two inference conventions are NOT interchangeable and each expert keeps its
own:

* HOIPrior calls its model ONCE per step.  It has no classifier-free guidance at
  inference -- ``priors/hoi/diffusion.py`` has no ``w`` -- and it applies
  ``prepare_clean_x0`` to restore the two history frames and close the object
  rotation channel on SO(3).
* HSIPrior is the released ``models.infbagel.Sampler`` architecture, and calls
  its model TWICE, combining ``cond + w * (cond - uncond)``.  Its "uncond" pass
  is not unconditional: ``models/infbagel.py:1554`` zeroes only the TEMPORAL
  scene embeddings, keeping the static scene, the text and the goals.  So its CFG
  amplifies dynamic scene perception specifically, which is exactly the term
  HOIPrior has no analogue for.  Dropping it would silently de-scene the HSI
  expert.

With CG and relational correction off, ``G == 0`` reduces this to HOIPrior alone,
bitwise, with no HSI checkpoint
loaded, and ``tests/phase2`` asserts that against ``HOIPriorSampler.p_sample_loop``
rather than trusting the arithmetic.  It also holds with the HSI expert LOADED and
running -- measured on 7 real HOSI-test episodes against the sealed anchor row, all
15 metrics identical, which is what rules out the composed loop perturbing HOI's
chain through a shared RNG stream
(``.claude/scratch/phase2-hsi-wiring/anchor_comparison_smoke_g0.json``).

One asymmetry is NOT a free choice and is exposed as ``hsi_object_voxel_mode``:
HOSI-test has a manipulated object, HSIPrior's training data never did, and the
occupancy alphabet has a distinct value for it.  See ``_resolve_object_voxels``.
"""

from typing import Optional

import torch
from pytorch3d import transforms

from guidance_loss import apply_hsi_guidance_loss
from priors.core.representation import REPRESENTATION
from priors.core.window_codec import project_to_so3
from priors.hoi.diffusion import prepare_clean_x0
from utils import transform_points

from .composition import (
    ExpertOutputs,
    _validate_channel_mask,
    compose_x0,
    gate_is_identity,
)
from .input_views import masked_object_arguments


class HOSIComposedSampler:
    """Drive one reverse chain through both experts, composing x_hat_0 per step.

    Presents the released ``Sampler.p_sample_loop`` signature so the HOSI
    evaluator calls it unchanged.  ``hsi_sampler`` may be None, in which case the
    gate must be identically 0 and this is the HOI-alone anchor path.
    """

    def __init__(self, hoi_adapter, hsi_sampler=None, gate=None, state=None,
                 channel_mask='human', hsi_object_voxel_mode='occupied',
                 body_composer=None, hsi_guidance_scale=1.0,
                 inference_engineering=False, input_diagnostic=None,
                 hsi_input_view=None, relational_corrector=None, scene_editor=None):
        if state is not None:
            raise NotImplementedError(
                'HOSIComposedSampler accepts `state` only as a reserved '
                'parameter; the LLM state machine is not implemented'
            )
        if hsi_object_voxel_mode not in ('occupied', 'object'):
            raise ValueError(
                "hsi_object_voxel_mode must be 'occupied' or 'object', got "
                f'{hsi_object_voxel_mode!r}'
            )
        # Validate the mask HERE, not 500 steps into the first window.  A row
        # configured with `mixer_channel_mask: null` used to run to completion and
        # produce an invalid object channel; now the sampler refuses to construct.
        # A scalar probe stands in for the expert output: the check needs only a
        # device, a dtype and the 232-channel extent.
        _validate_channel_mask(channel_mask, torch.zeros(REPRESENTATION.dimension))
        self.hoi_adapter = hoi_adapter
        self.hsi_sampler = hsi_sampler
        self.gate = 0 if gate is None else gate
        self.body_composer = body_composer
        self.channel_mask = channel_mask
        self.hsi_object_voxel_mode = hsi_object_voxel_mode
        self.hsi_guidance_scale = float(hsi_guidance_scale)
        self.inference_engineering = bool(inference_engineering)
        self.input_diagnostic = input_diagnostic
        self.hsi_input_view = hsi_input_view
        self.relational_corrector = relational_corrector
        self.scene_editor = scene_editor
        if scene_editor is not None and scene_editor.enabled:
            if (not gate_is_identity(self.gate, 0) or body_composer is not None
                    or relational_corrector is not None or input_diagnostic is not None
                    or self._hsi_posterior_guidance_enabled()):
                raise ValueError('scene editor requires pure HOI generation and HSI CG off')
            if hsi_sampler is None:
                raise ValueError('scene editor requires the scene-query sampler')
        self._posterior_coefficient_cache = None
        if self.hsi_guidance_scale < 0:
            raise ValueError('hsi_guidance_scale must be non-negative')
        self.dataset = None
        self.student_model = None
        self.compose_calls = 0
        self.hsi_object_voxels_remapped = 0
        self.hsi_guidance_calls = 0
        # Keep per-step telemetry on device. Converting each value to a Python
        # scalar would force 499 CUDA synchronizations per window and turn audit
        # instrumentation into a large, protocol-changing latency tax.
        self._hsi_guidance_nonfinite_steps = None
        self._hsi_guidance_gradient_max_abs = None
        self._hsi_guidance_gradient_square_sum = None
        self.hsi_guidance_gradient_values = 0
        if body_composer is not None and not gate_is_identity(self.gate, 0):
            raise ValueError(
                'body_composer replaces the raw gate, so gate must remain the '
                'identity value 0; combining both operators is ambiguous'
            )
        if body_composer is not None and hsi_sampler is None:
            raise ValueError('a body_composer needs both expert samplers')
        if hsi_sampler is None and not gate_is_identity(self.gate, 0):
            raise ValueError(
                'a non-zero gate needs an HSI sampler; pass hsi_sampler or use '
                'gate=0 for the HOI-alone anchor'
            )

    @property
    def inner_hoi(self):
        """The real ``HOIPriorSampler`` behind the adapter."""
        return self.hoi_adapter.inner

    @property
    def timesteps(self):
        # HOIPriorSampler holds the step count on its diffusion object, not on
        # itself; the released Sampler holds it directly. Read HOI's, because
        # HOI's diffusion is what advances the composed chain.
        return self.inner_hoi.diffusion.timesteps

    def set_dataset_and_model(self, dataset, model, hsi_model=None):
        """Wire both experts.

        ``model`` is HOIPrior's network and goes through the adapter, which hands
        the expert a scene-blind view of the dataset.  ``hsi_model`` is the
        released-architecture network for the HSI expert and is given to
        ``hsi_sampler`` with the FULL dataset, because that expert's whole job is
        to read the scene.
        """
        self.dataset = dataset
        self.student_model = model
        self.hoi_adapter.set_dataset_and_model(dataset, model)
        if self.hsi_sampler is not None:
            if hsi_model is None:
                raise ValueError('an HSI sampler needs an HSI model')
            self.hsi_sampler.set_dataset_and_model(dataset, hsi_model)
        if self.scene_editor is not None and self.scene_editor.enabled:
            model.eval().requires_grad_(False)
            hsi_model.eval().requires_grad_(False)
            if not torch.equal(self.inner_hoi.diffusion.betas.cpu(), self.hsi_sampler.betas.cpu()):
                raise ValueError('scene teacher diffusion schedules differ')

    def audit_dict(self):
        audit = dict(self.inner_hoi.audit_dict())
        audit['composition'] = {
            'gate': self._describe_gate(),
            'operator': (
                self.body_composer.describe() if self.body_composer is not None
                else {'kind': 'raw_channel_blend'}
            ),
            'channel_mask': (
                self.channel_mask if isinstance(self.channel_mask, str) else 'tensor'
            ),
            # Raw/kinematic composition preserves HOI object channels. Relational
            # correction also transforms the object with the human; its own audit
            # records that physical provenance and the exact contact restoration.
            'object_channels_from_hoi': (self.relational_corrector is None
                                         and (self.scene_editor is None or not self.scene_editor.enabled)),
            'compose_calls': self.compose_calls,
            'hsi_expert_loaded': self.hsi_sampler is not None,
            'per_step_composition': True,
            'hsi_object_voxel_mode': self.hsi_object_voxel_mode,
            'hsi_input_view': (
                'known_empty_forward_trajectory' if self.hsi_input_view is not None
                else 'shared_full_state'
            ),
            'hsi_object_voxels_remapped': self.hsi_object_voxels_remapped,
            # Anchor occupancy reads the shared noisy chain (`current`); temporal
            # occupancy reads the previous composed clean prediction.  There is
            # deliberately no private-HSI-pelvis switch.
            'scene_query_pelvis': 'shared_current_and_previous_composed_x0',
            'hsi_posterior_guidance': {
                'enabled': self._hsi_posterior_guidance_enabled(),
                'energy': 'guidance_loss.apply_hsi_guidance_loss',
                'evaluated_on': 'actual_composed_clean_body',
                'posterior_coef1': self._hsi_posterior_guidance_enabled(),
                'guidance_scale': self.hsi_guidance_scale,
                'steps': '499_through_1' if self._hsi_posterior_guidance_enabled() else None,
                'calls': self.hsi_guidance_calls,
                'nonfinite_steps': self._audit_scalar(
                    self._hsi_guidance_nonfinite_steps, int,
                ),
                'gradient_max_abs': self._audit_scalar(
                    self._hsi_guidance_gradient_max_abs, float,
                ),
                'gradient_rms': (
                    (self._audit_scalar(
                        self._hsi_guidance_gradient_square_sum, float,
                    ) / self.hsi_guidance_gradient_values) ** 0.5
                    if self.hsi_guidance_gradient_values else 0.0
                ),
                'object_contact_dependency': False,
                'history_restored_after_update': True,
            },
        }
        if self.relational_corrector is not None:
            audit['composition']['relational_correction'] = self.relational_corrector.audit_dict()
        if self.scene_editor is not None:
            audit['composition']['scene_edit'] = self.scene_editor.audit_dict()
        return audit

    @staticmethod
    def _audit_scalar(value, cast):
        if value is None:
            return cast(0)
        return cast(value.detach().cpu().item())

    def _describe_gate(self):
        if isinstance(self.gate, torch.Tensor):
            return {
                'kind': 'tensor',
                'shape': list(self.gate.shape),
                'min': float(self.gate.min()),
                'max': float(self.gate.max()),
            }
        if hasattr(self.gate, 'describe'):
            return self.gate.describe()
        return {'kind': 'scalar', 'value': float(self.gate)}

    def cm_sample_loop(self, *_, **__):
        raise NotImplementedError(
            'the composed sampler has no consistency-model path: HOIPrior was '
            'never distilled, so there is no few-step HOI x_hat_0 to compose'
        )

    # Both single-expert loops carry this (priors/hoi/diffusion.py:490,
    # models/infbagel.py:1137) and the composed loop needs it for the same
    # reason: without it the HSI expert's two forward passes build a graph, the
    # returned window comes back with requires_grad set, and the evaluator dies
    # on `.cpu().numpy()` 500 steps later.  It does not disable HOIPrior's
    # guidance, which opens its own `torch.enable_grad()` around a detached copy
    # (priors/hoi/inference_guidance.py:612).
    @torch.no_grad()
    def p_sample_loop(self, fixed_points, mat, scene_flag, text_emb, pelvis_goal,
                      scene_goal, object_goal, need_scene, need_pelvis_dir, pi,
                      end_pi, seq_length, need_pi, is_loco, is_object,
                      obj_bps_data, object_points, obj_rot_mat_ref,
                      obj_rest_verts, obj_vert_normals=None, seq_name_dict=None,
                      human_dict=None, guidance_fn=None, guidance_scale=None,
                      object_only=False, obj_rot_mat_prefix=None,
                      ground_truth_contact=None, state=None):
        if state is not None:
            raise NotImplementedError(
                'p_sample_loop accepts `state` only as a reserved parameter; the '
                'LLM state machine is not implemented'
            )
        if guidance_fn is not None:
            raise ValueError(
                'the composed sampler takes no per-call guidance_fn: each '
                "expert's guidance is its own preregistered sampler state"
            )
        batch = fixed_points.shape[0]
        device = fixed_points.device

        # HOI's conditioning is loop-invariant, so it is built once.  This also
        # advances the expert's sample_calls and constructs its guidance object,
        # which is why it must happen exactly once per window -- the same
        # contract p_sample_loop has.
        hoi_arguments = self.inner_hoi.prepare_sample_arguments(
            fixed_points, mat, text_emb, pelvis_goal, object_goal, pi, end_pi,
            seq_length, is_object, obj_bps_data, obj_rot_mat_ref, obj_rest_verts,
            seq_name_dict, ground_truth_contact=ground_truth_contact,
        )
        generator = hoi_arguments.pop('generator')
        object_so3_x0 = hoi_arguments.pop('object_so3_x0')
        hoi_guidance = hoi_arguments.pop('guidance', None)
        local_object_bps = hoi_arguments.pop('local_object_bps', None)

        diffusion = self.inner_hoi.diffusion
        shape = (batch, REPRESENTATION.window_frames, REPRESENTATION.dimension)
        current = torch.randn(shape, device=device, generator=generator)
        current[:, :REPRESENTATION.history_frames] = fixed_points

        hsi_context = None
        if self.hsi_sampler is not None:
            hsi_context = self._hsi_context(
                batch, mat, scene_flag, text_emb, pelvis_goal, scene_goal,
                object_goal, need_scene, need_pelvis_dir, pi, end_pi, seq_length,
                need_pi, is_loco, is_object, obj_bps_data, object_points,
                obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict,
                obj_rot_mat_prefix, object_only,
            )

        # The released HSI loop carries the PREVIOUS step's x_hat_0 forward: its
        # `x0` list starts at the initial noise and `_compute_occ_sample` reads
        # `x0[:, :, :84]` to place the temporal occupancy queries.  Under
        # composition the chain's own best estimate of the finished motion is the
        # COMPOSED prediction, so that is what the occupancy sees. Feeding HSI's
        # private prediction instead would let the scene queries follow a
        # trajectory the chain is not taking. Initialized to `current` to match
        # the released `x0.append(points)`, which happens after the history fix.
        previous_x0 = current
        if self.hsi_input_view is not None:
            self.hsi_input_view.begin_window(current, generator.initial_seed())
        if self.input_diagnostic is not None:
            self.input_diagnostic.begin_window(current)
        imgs = []
        for step in reversed(range(diffusion.timesteps)):
            timesteps = torch.full((batch,), step, dtype=torch.long, device=device)
            hoi_clean = self._hoi_x0(
                diffusion, current, timesteps, fixed_points, hoi_arguments,
                local_object_bps, object_so3_x0,
            )
            if self.input_diagnostic is not None:
                self.input_diagnostic.observe(
                    self, current, previous_x0, timesteps, hsi_context,
                    human_dict['rest_human_offsets'], hoi_clean,
                )
            # Skip the HSI expert on any step whose gate discards it entirely.
            # This is free of numerical consequence, and provably so: the G == 0
            # row already reproduced a sealed anchor produced WITHOUT the HSI
            # expert in the process at all, so HSI's two forward passes draw
            # nothing from the stream HOIPrior draws from, and removing them
            # cannot move HOI's chain.  Re-measured after adding the skip: still
            # bitwise equal to the sealed row on all 15 metrics over 7 episodes.
            #
            # It is worth 10x, not the 3x that counting network calls predicts
            # (HSI calls twice per step to HOI's once): 58.69 s/episode to 5.89 s
            # on the same 7 episodes.  So the HSI expert's per-step cost here is
            # dominated by `_compute_occ_sample`, which runs four 32,768-point
            # occupancy queries against the scene grid every step, not by its
            # network.  That is the price of every G > 0 row and the reason a
            # schedule gate's zero region is worth skipping properly.
            #
            # A CALLABLE gate cannot be resolved without both predictions, so this
            # applies only to a constant gate. `_step_gate_is_zero` decides.
            hsi_clean = None
            if self.hsi_sampler is not None and not self._step_gate_is_zero(step):
                hsi_clean = self._hsi_x0(
                    current, previous_x0, timesteps, hsi_context,
                )

            outputs = ExpertOutputs(hoi=hoi_clean, hsi=hsi_clean)
            if self.body_composer is None:
                gate = self._gate_for_step(step, current, hoi_clean, hsi_clean)
                clean = compose_x0(
                    outputs, gate, channel_mask=self.channel_mask,
                )
            else:
                if human_dict is None or 'rest_human_offsets' not in human_dict:
                    raise ValueError(
                        'kinematic body composition needs '
                        'human_dict[rest_human_offsets]'
                    )
                clean = self.body_composer(
                    outputs, dataset=self.dataset,
                    rest_human_offsets=human_dict['rest_human_offsets'],
                    fixed_points=fixed_points,
                )
            if self.relational_corrector is not None:
                clean = self.relational_corrector.correct(
                    self, current, previous_x0, timesteps, hsi_context,
                    human_dict['rest_human_offsets'], clean, step,
                )
            self.compose_calls += 1
            previous_x0 = clean

            if step:
                noise = torch.randn(shape, device=device, generator=generator)
            else:
                noise = torch.zeros_like(current)
            current = diffusion.posterior_sample(
                current, clean, timesteps, noise, fixed_points,
            )
            if step and self._hsi_posterior_guidance_enabled():
                current = self._apply_hsi_posterior_guidance(
                    current, clean, timesteps, hsi_context, human_dict,
                    fixed_points,
                )
            if hoi_guidance is not None and step:
                current = hoi_guidance.apply(current, clean, fixed_points, step)
            imgs.append(current)

        current[..., 219:228] = project_to_so3(
            current[..., 219:228].reshape(batch, REPRESENTATION.window_frames, 3, 3)
        ).reshape(batch, REPRESENTATION.window_frames, 9)
        if self.scene_editor is not None and self.scene_editor.enabled:
            current[:, :REPRESENTATION.history_frames] = fixed_points
            current = self.scene_editor.edit(
                self, current, hoi_arguments, local_object_bps, hsi_context,
                human_dict['rest_human_offsets'], generator.initial_seed(),
            )
        self.inner_hoi._update_audit(current)
        imgs[-1] = current
        return imgs, []

    def _hoi_x0(self, diffusion, current, timesteps, fixed_points, arguments,
                local_object_bps, object_so3_x0):
        """HOIPrior's x_hat_0 for one step: one forward, then prepare_clean_x0.

        The D2-AG self-conditioning relation source is deliberately NOT wired
        here.  It feeds the PREVIOUS step's raw x0_hat back in, and under
        composition the previous step's x0_hat is the BLEND, not this expert's
        own prediction -- so a composed run would feed the expert a
        self-conditioning signal it was never trained on.  P15, the settled
        HOIPrior, is not a D2-AG architecture, so this path is unused; a D2-AG
        checkpoint raises in prepare_sample_arguments' relation check instead of
        being silently mis-conditioned.
        """
        clean = self._hoi_raw_x0(current, timesteps, arguments, local_object_bps)
        return prepare_clean_x0(clean, fixed_points, object_so3_x0=object_so3_x0)

    def _hoi_raw_x0(self, current, timesteps, arguments, local_object_bps):
        """Raw HOI network head with the already-prepared window conditions."""
        model = self.inner_hoi.student_model
        forward_arguments = dict(arguments)
        if local_object_bps is not None:
            clean = model(
                current, timesteps, forward_arguments['text_embedding'],
                forward_arguments['object_bps'], forward_arguments['goals'],
                forward_arguments['progress'], local_object_bps=local_object_bps,
            )
        elif 'rest_object_points' in forward_arguments:
            clean = model(
                current, timesteps, forward_arguments.pop('text_embedding'),
                forward_arguments.pop('object_bps'), forward_arguments.pop('goals'),
                forward_arguments.pop('progress'), **forward_arguments,
            )
        else:
            clean = model(
                current, timesteps, forward_arguments['text_embedding'],
                forward_arguments['object_bps'], forward_arguments['goals'],
                forward_arguments['progress'],
            )
        return clean

    def _hsi_context(self, batch, mat, scene_flag, text_emb, pelvis_goal,
                     scene_goal, object_goal, need_scene, need_pelvis_dir, pi,
                     end_pi, seq_length, need_pi, is_loco, is_object,
                     obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts,
                     obj_vert_normals, seq_name_dict, obj_rot_mat_prefix,
                     object_only):
        """Everything the released sampler's per-step call needs, minus the state.

        Occupancy is NOT hoisted: ``_compute_occ_sample`` reads the current state
        (that is the dynamic-perception mechanism), so it is recomputed per step
        inside ``_hsi_x0``.
        """
        self.hsi_sampler.batch_size = batch
        context = {
            'mat': mat, 'scene_flag': scene_flag, 'text_emb': text_emb,
            'pelvis_goal': pelvis_goal, 'scene_goal': scene_goal,
            'object_goal': object_goal, 'need_scene': need_scene,
            'need_pelvis_dir': need_pelvis_dir, 'pi': pi, 'end_pi': end_pi,
            'seq_length': seq_length, 'need_pi': need_pi, 'is_loco': is_loco,
            'is_object': is_object, 'obj_bps_data': obj_bps_data,
            'object_points': object_points, 'obj_rot_mat_ref': obj_rot_mat_ref,
            'obj_rest_verts': obj_rest_verts, 'obj_vert_normals': obj_vert_normals,
            'seq_name_dict': seq_name_dict, 'obj_rot_mat_prefix': obj_rot_mat_prefix,
            'object_only': object_only,
        }
        if self.inference_engineering:
            context['static_occ_cache'] = {}
        return context

    def _hsi_x0(self, current, previous_x0, timesteps, context):
        """HSIPrior's x_hat_0 for one step: occupancy, then cond and uncond.

        Mirrors ``models/infbagel.py`` p_sample's model calls exactly, including
        ``cond + w * (cond - uncond)``, and stops before its posterior step --
        the composed chain owns the posterior.

        ``_compute_occ_sample`` reads BOTH states and they are not
        interchangeable.  ``current`` (its ``x``) centres the anchor query and
        carries the object pose that builds the temporal object point cloud;
        ``previous_x0`` (its ``x0``) supplies the human joints at frames 5/10/15
        that place the three temporal occupancy queries.  The released loop
        passes ``x0[-1]``, i.e. the previous step's prediction, so passing
        ``current`` for both would query the scene along a NOISY trajectory at
        high timesteps -- pure noise on the first step.
        """
        cond, uncond = self._hsi_prediction_pair(current, previous_x0, timesteps, context)
        return cond + self.hsi_sampler.w * (cond - uncond)

    def _hsi_prediction_pair(self, current, previous_x0, timesteps, context):
        common = self._hsi_model_arguments(current, previous_x0, timesteps, context)
        if self.hsi_input_view is not None:
            current = self.hsi_input_view.for_step(current, int(timesteps[0]))
            common = masked_object_arguments(common)
        return self._hsi_predict_pair(current, common)

    def _hsi_model_arguments(self, current, previous_x0, timesteps, context):
        """Query the full world once, independently of denoiser input views."""
        sampler = self.hsi_sampler
        occ_arguments = (
            current, previous_x0, context['mat'], context['scene_flag'],
            context['object_points'], context['pelvis_goal'],
            context['scene_goal'], context['object_goal'], context['is_loco'],
            context['is_object'], context['need_pelvis_dir'],
            context['obj_rot_mat_ref'], context['object_only'],
            context['obj_rest_verts'], context['seq_name_dict'],
            context['obj_rot_mat_prefix'],
        )
        if self.inference_engineering:
            occ, occ_list, occ_pos = sampler._compute_occ_sample(
                *occ_arguments, inference_engineering=True,
                static_cache=context['static_occ_cache'],
            )
        else:
            occ, occ_list, occ_pos = sampler._compute_occ_sample(*occ_arguments)
        occ, occ_list = self._resolve_object_voxels(occ, occ_list)
        return (
            occ, timesteps, context['text_emb'], context['pelvis_goal'],
            context['scene_goal'], context['is_loco'], context['need_scene'],
            context['need_pelvis_dir'], context['pi'], context['end_pi'],
            context['seq_length'], context['need_pi'], context['object_goal'],
            context['is_object'], context['obj_bps_data'], occ_list, occ_pos,
        )

    def _hsi_predict(self, current, common):
        cond, uncond = self._hsi_predict_pair(current, common)
        return cond + self.hsi_sampler.w * (cond - uncond)

    def _hsi_predict_pair(self, current, common):
        model = self.hsi_sampler.student_model
        cond = model(current, *common, is_sample=True)
        uncond = model(current, *common, is_sample=True, is_uncondition=True)
        return cond, uncond

    def _hsi_posterior_guidance_enabled(self):
        """Whether the frozen R2-CG posterior rule is active.

        Historical composed rows have no such sampler property and therefore
        remain byte-identical. Enabling the rule without an HSI expert is a
        configuration error rather than a silently ignored recipe.
        """
        return bool(
            self.hsi_sampler is not None
            and getattr(self.hsi_sampler, 'hsi_guidance_posterior_coef1', False)
        )

    def _apply_hsi_posterior_guidance(self, current, clean, timesteps,
                                      context, human_dict, fixed_points):
        """Apply native R2-CG human-scene guidance to the shared posterior.

        Native HSIPrior differentiates its scene energy with respect to x0 and
        adds ``posterior_mean_coef1(t) * gradient`` to x_{t-1}. Composition owns
        the posterior, so the energy is evaluated on the clean body the shared
        chain will actually follow. The noisy state may leave the FK manifold;
        the next clean prediction is reconstructed by the selected composer.
        """
        if context is None or human_dict is None:
            raise ValueError(
                'R2-CG composed guidance needs HSI context and human_dict'
            )
        if 'rest_human_offsets' not in human_dict:
            raise ValueError(
                'R2-CG composed guidance needs '
                'human_dict[rest_human_offsets]'
            )

        with torch.enable_grad():
            guided_clean = clean.detach().requires_grad_(True)
            batch, frames = guided_clean.shape[:2]
            global_position = transform_points(
                self.dataset.denormalize_torch(guided_clean[..., :84]),
                context['mat'],
            ).reshape(batch, frames, 28, 3)

            offsets = human_dict['rest_human_offsets']
            if offsets.ndim == 3:
                offsets = offsets[:, None].expand(-1, frames, -1, -1)
            expected = (batch, frames, 24, 3)
            if tuple(offsets.shape) != expected:
                raise ValueError(
                    f'R2-CG expected rest offsets {expected}, got '
                    f'{tuple(offsets.shape)}'
                )
            local_position = offsets.reshape(-1, 24, 3).clone()
            local_position[:, 0] = global_position[..., 0, :].reshape(-1, 3)

            global_rotation = transforms.rotation_6d_to_matrix(
                guided_clean[..., 84:216].reshape(batch, frames, 22, 6)
            )
            global_rotation = (
                context['mat'][:, None, None, :3, :3] @ global_rotation
            )
            local_rotation = self.dataset.quat_ik_torch(
                global_rotation.reshape(-1, 22, 3, 3)
            )
            _, human_joints = self.dataset.quat_fk_torch(
                local_rotation, local_position,
            )
            human_joints = human_joints.reshape(batch, frames, 24, 3)
            loss = apply_hsi_guidance_loss(
                human_joints, context['scene_flag'],
                self.dataset.get_nearest_free_voxel,
            )
            gradient = torch.autograd.grad(-loss, guided_clean)[0]
            gradient = gradient * self.hsi_guidance_scale

        if self.inference_engineering:
            source = self.hsi_sampler.posterior_mean_coef1
            cache_key = (
                source.data_ptr(), source._version, gradient.device,
                gradient.dtype,
            )
            if (
                self._posterior_coefficient_cache is None
                or self._posterior_coefficient_cache[0] != cache_key
            ):
                schedule = source.to(
                    device=gradient.device, dtype=gradient.dtype,
                )
                self._posterior_coefficient_cache = (cache_key, schedule)
            coefficient = self._posterior_coefficient_cache[1].gather(
                -1, timesteps,
            ).reshape(batch, 1, 1)
        else:
            coefficient = self.hsi_sampler.posterior_mean_coef1.gather(
                -1, timesteps.cpu(),
            ).reshape(batch, 1, 1).to(
                device=gradient.device, dtype=gradient.dtype,
            )
        update = coefficient * gradient

        self.hsi_guidance_calls += 1
        nonfinite_step = torch.logical_not(torch.isfinite(update)).any().to(
            dtype=torch.int64,
        )
        gradient_max = gradient.detach().abs().max()
        gradient_square_sum = gradient.detach().double().square().sum()
        if self._hsi_guidance_nonfinite_steps is None:
            self._hsi_guidance_nonfinite_steps = nonfinite_step
            self._hsi_guidance_gradient_max_abs = gradient_max
            self._hsi_guidance_gradient_square_sum = gradient_square_sum
        else:
            self._hsi_guidance_nonfinite_steps = (
                self._hsi_guidance_nonfinite_steps + nonfinite_step
            )
            self._hsi_guidance_gradient_max_abs = torch.maximum(
                self._hsi_guidance_gradient_max_abs, gradient_max,
            )
            self._hsi_guidance_gradient_square_sum = (
                self._hsi_guidance_gradient_square_sum + gradient_square_sum
            )
        self.hsi_guidance_gradient_values += gradient.numel()

        result = current + update
        # Native p_sample_loop restores fixed history immediately after p_sample.
        # Do the same after the shared update, so guidance never becomes history.
        result[:, :REPRESENTATION.history_frames] = fixed_points
        return result

    def _resolve_object_voxels(self, occ, occ_list):
        """Decide what the manipulated object looks like to the HSI expert.

        The occupancy alphabet is 0 free / 1 occupied / 2 object.  HSIPrior was
        trained LINGO-only, and under `lingo_only` every sample's object_points is
        the 999.0 sentinel (datasets/infbagel_mix.py:471), which falls out of
        bounds, clamps to voxel 0 and is written as 2 there.  So the value 2
        reached its scene ViT as at most ONE spurious corner voxel per grid.

        On HOSI-test the object is real.  Measured on one episode, the three
        TEMPORAL grids carry 225-239 voxels at 2 -- and `add_object_voxel: false`
        does not prevent it, because the evaluator sets `cfg.vis = True` and
        `_compute_occ_sample:706` then rebuilds the object from `obj_rest_verts`
        regardless of the flag.  Those temporal grids are exactly the embeddings
        HSI's CFG amplifies.

        The choice matters, measured with the real P17-OC weights on one window:
        remapping moves x_hat_0 by 0.039-0.057 m mean and up to 0.158 m max in
        joint position at t in {1, 100, 250, 400}, and by exactly zero at t = 499
        (where models/infbagel.py:1554 zeroes the temporal embeddings on both
        passes, so the only grids carrying a 2 have no effect -- an independent
        confirmation that the whole effect travels through the temporal channels).
        Evidence: .claude/scratch/phase2-hsi-wiring/occ2_sensitivity.json.

        DECIDED 2026-08-30 (user): 'occupied' for every row, 'object' demoted to a
        later ablation.  'occupied' rewrites 2 to 1, so the object becomes ordinary
        occupied geometry -- what LINGO furniture looked like in training -- and the
        input is in distribution.  'object' passes the 2 through, which is the
        released arithmetic and what a model trained with `is_mix=True` would
        expect; the released InfBaGel checkpoint is such a model, HSIPrior is not.

        The decision rests on the two facts above and on nothing else.  It does NOT
        claim 'occupied' generates better motion -- no measurement says that yet,
        and the ablation is what would. What it claims is that the input is in
        distribution and the alternative is not, and that the measured difference
        lives entirely in the temporal channels where that shift is (zero at
        t=499, where those embeddings are zeroed on both CFG passes).
        """
        if self.hsi_object_voxel_mode == 'object':
            return occ, occ_list
        remapped = 0
        if occ is not None:
            remapped += int((occ == 2).sum())
            occ = torch.where(occ == 2, torch.ones_like(occ), occ)
        if occ_list is not None:
            remapped += int((occ_list == 2).sum())
            occ_list = torch.where(
                occ_list == 2, torch.ones_like(occ_list), occ_list,
            )
        self.hsi_object_voxels_remapped += remapped
        return occ, occ_list

    def _step_gate_is_zero(self, step):
        """Can this step's gate be known to discard HSI before computing it?

        Only ever used to SKIP work, so it must be conservative: a gate whose
        value cannot be known without both predictions answers False and the
        expert runs.  A gate that reads the expert outputs is inherently in that
        category -- resolving it is what needs them.

        Gates that depend on the step alone advertise
        ``is_identically_zero_at(step)``, which lets a schedule that starts at 0
        cost what it looks like it costs rather than paying for a prediction the
        blend throws away.
        """
        if self.body_composer is not None:
            return False
        if callable(self.gate):
            decide = getattr(self.gate, 'is_identically_zero_at', None)
            if decide is None:
                return False
            return bool(decide(step))
        return gate_is_identity(self.gate, 0)

    def _gate_for_step(self, step, current, hoi_clean, hsi_clean):
        """Resolve the gate for one step.

        A plain scalar or tensor is used as is.  A callable gate is given the
        step and both experts' predictions, which is the MixerMDM modularity
        property: the gate reads only expert OUTPUTS, never their weights or
        their internal features, so either expert can be replaced without
        retraining it.
        """
        if callable(self.gate):
            return self.gate(
                step=step, current=current, hoi=hoi_clean, hsi=hsi_clean,
            )
        return self.gate
