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

``G == 0`` reduces this to HOIPrior alone, bitwise, with no HSI checkpoint
loaded, and ``tests/phase2`` asserts that against ``HOIPriorSampler.p_sample_loop``
rather than trusting the arithmetic.
"""

from typing import Optional

import torch

from priors.core.representation import REPRESENTATION
from priors.core.window_codec import project_to_so3
from priors.hoi.diffusion import prepare_clean_x0

from .composition import ExpertOutputs, compose_x0, gate_is_identity


class HOSIComposedSampler:
    """Drive one reverse chain through both experts, composing x_hat_0 per step.

    Presents the released ``Sampler.p_sample_loop`` signature so the HOSI
    evaluator calls it unchanged.  ``hsi_sampler`` may be None, in which case the
    gate must be identically 0 and this is the HOI-alone anchor path.
    """

    def __init__(self, hoi_adapter, hsi_sampler=None, gate=None, state=None,
                 channel_mask='human'):
        if state is not None:
            raise NotImplementedError(
                'HOSIComposedSampler accepts `state` only as a reserved '
                'parameter; the LLM state machine is not implemented'
            )
        self.hoi_adapter = hoi_adapter
        self.hsi_sampler = hsi_sampler
        self.gate = 0 if gate is None else gate
        self.channel_mask = channel_mask
        self.dataset = None
        self.student_model = None
        self.compose_calls = 0
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
        return self.inner_hoi.timesteps

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

    def audit_dict(self):
        audit = dict(self.inner_hoi.audit_dict())
        audit['composition'] = {
            'gate': self._describe_gate(),
            'channel_mask': (
                self.channel_mask if isinstance(self.channel_mask, str)
                else ('tensor' if self.channel_mask is not None else None)
            ),
            'compose_calls': self.compose_calls,
            'hsi_expert_loaded': self.hsi_sampler is not None,
            'per_step_composition': True,
        }
        return audit

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

        imgs = []
        for step in reversed(range(diffusion.timesteps)):
            timesteps = torch.full((batch,), step, dtype=torch.long, device=device)
            hoi_clean = self._hoi_x0(
                diffusion, current, timesteps, fixed_points, hoi_arguments,
                local_object_bps, object_so3_x0,
            )
            hsi_clean = None
            if self.hsi_sampler is not None:
                hsi_clean = self._hsi_x0(current, timesteps, hsi_context)

            gate = self._gate_for_step(step, current, hoi_clean, hsi_clean)
            clean = compose_x0(
                ExpertOutputs(hoi=hoi_clean, hsi=hsi_clean), gate,
                channel_mask=self.channel_mask,
            )
            self.compose_calls += 1

            if step:
                noise = torch.randn(shape, device=device, generator=generator)
            else:
                noise = torch.zeros_like(current)
            current = diffusion.posterior_sample(
                current, clean, timesteps, noise, fixed_points,
            )
            if hoi_guidance is not None and step:
                current = hoi_guidance.apply(current, clean, fixed_points, step)
            imgs.append(current)

        current[..., 219:228] = project_to_so3(
            current[..., 219:228].reshape(batch, REPRESENTATION.window_frames, 3, 3)
        ).reshape(batch, REPRESENTATION.window_frames, 9)
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
        return prepare_clean_x0(clean, fixed_points, object_so3_x0=object_so3_x0)

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
        return {
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

    def _hsi_x0(self, current, timesteps, context):
        """HSIPrior's x_hat_0 for one step: occupancy, then cond and uncond.

        Mirrors ``models/infbagel.py`` p_sample's model calls exactly, including
        ``cond + w * (cond - uncond)``, and stops before its posterior step --
        the composed chain owns the posterior.
        """
        sampler = self.hsi_sampler
        model = sampler.student_model
        occ, occ_list, occ_pos = sampler._compute_occ_sample(
            current, current, context['mat'], context['scene_flag'],
            context['object_points'], context['pelvis_goal'],
            context['scene_goal'], context['object_goal'], context['is_loco'],
            context['is_object'], context['need_pelvis_dir'],
            context['obj_rot_mat_ref'], context['object_only'],
            context['obj_rest_verts'], context['seq_name_dict'],
            context['obj_rot_mat_prefix'],
        )
        common = (
            occ, timesteps, context['text_emb'], context['pelvis_goal'],
            context['scene_goal'], context['is_loco'], context['need_scene'],
            context['need_pelvis_dir'], context['pi'], context['end_pi'],
            context['seq_length'], context['need_pi'], context['object_goal'],
            context['is_object'], context['obj_bps_data'], occ_list, occ_pos,
        )
        cond = model(current, *common, is_sample=True)
        uncond = model(current, *common, is_sample=True, is_uncondition=True)
        return cond + sampler.w * (cond - uncond)

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
