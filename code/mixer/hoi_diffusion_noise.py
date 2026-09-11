"""DNO on a frozen HOI generator, with current-motion HSI teacher targets."""
import json
import sys
import time
from pathlib import Path

import torch
from pytorch3d import transforms
from torch.utils.checkpoint import checkpoint

from priors.core.window_codec import project_to_so3
from .body_projection import native_rest_offsets, NativeSceneDifferential
from .diffusion_noise import ddim_transition, native_linear_interpolation, native_quaternion_interpolation
from .hsi_motion_target import human_goal_context, rotated_scene_context, observation_outside
from .input_views import empty_motion_view, masked_object_arguments
from .kinematic_composition import _local_from_global, _forward_kinematics
from .surface_edit import FEET, HANDS, object_frame_hands, native_hand_distances, decode_body
from .continuation_outcomes import write_json


def reframe_hoi(value, dataset, old, new):
    """Change both human and object frames while preserving the world motion."""
    old_mat, new_mat = old['mat'], new['mat']
    rotation = new_mat[:, :3, :3].transpose(-1, -2) @ old_mat[:, :3, :3]
    shift = (new_mat[:, :3, :3].transpose(-1, -2) @
             (old_mat[:, :3, 3] - new_mat[:, :3, 3])[..., None]).squeeze(-1)
    points = dataset.denormalize_torch(value[..., :84]).reshape(*value.shape[:2], 28, 3)
    points = (rotation[:, None, None] @ points[..., None]).squeeze(-1) + shift[:, None, None]
    human = rotation[:, None, None] @ transforms.rotation_6d_to_matrix(value[..., 84:216].reshape(*value.shape[:2], 22, 6))
    obj = dataset.denormalize_torch(value[..., 216:219], is_object=True)
    obj = (rotation[:, None] @ obj[..., None]).squeeze(-1) + shift[:, None]
    relative = value[..., 219:228].reshape(*value.shape[:2], 3, 3)
    world = old['obj_rot_mat_prefix'][:, None] @ relative @ old['obj_rot_mat_ref'][:, None]
    relative = new['obj_rot_mat_prefix'][:, None].transpose(-1, -2) @ world @ new['obj_rot_mat_ref'][:, None].transpose(-1, -2)
    return torch.cat((dataset.normalize_torch(points).flatten(2),
                      transforms.matrix_to_rotation_6d(human).flatten(2),
                      dataset.normalize_torch(obj, is_object=True), relative.flatten(2), value[..., 228:]), -1)


class HOIDDIM:
    """One deterministic HOI rollout; later windows consume generated history."""
    def __init__(self, teacher, windows, source, model, task, ordinal, settings):
        self.teacher, self.sampler, self.dataset = teacher, teacher.sampler, teacher.dataset
        self.windows, self.source, self.settings = windows, source, settings
        self.task, self.ordinal = task, ordinal
        self.device = source['pose'].device
        self.alpha = self.sampler.inner_hoi.diffusion.sqrt_alpha_bar.square()
        self.times = torch.linspace(0, 499, settings['ddim_steps']).round().long().tolist()[::-1]
        self.clean = torch.cat([w['clean'] for w in windows])
        self.mats = torch.cat([w['context']['mat'] for w in windows])
        sequence = self.dataset.ori_sequence_idx[task['data_idx']]
        self.translation_offset = torch.as_tensor(self.dataset.transl[sequence], device=self.device)
        self.rest_offsets = native_rest_offsets(model, source['betas']).detach()
        self.hoi_calls = 0
        self.last_teacher = {}

    def initial_latent(self):
        generator = torch.Generator(device=self.device).manual_seed(42 + self.ordinal * 1000003)
        return torch.randn(1, 232, 1, len(self.windows) * 14, generator=generator, device=self.device)

    def predict(self, value, history, index, step):
        current = torch.cat((history, value[:, 2:]), 1)
        window = self.windows[index]
        timestep = torch.full((1,), step, device=value.device, dtype=torch.long)
        def forward(x, t=timestep, arguments=window['arguments'], local_bps=window['local_bps']):
            self.hoi_calls += 1
            return self.sampler._hoi_raw_x0(x, t, arguments, local_bps)
        clean = checkpoint(forward, current, use_reentrant=False) if torch.is_grad_enabled() else forward(current)
        return torch.cat((history, clean[:, 2:]), 1)

    def decode(self, latent):
        future = latent[0, :, 0].transpose(0, 1).reshape(len(self.windows), 14, 232)
        predictions = []
        for i, window in enumerate(self.windows):
            history = self.clean[:1, :2] if i == 0 else reframe_hoi(
                predictions[-1][:, -2:], self.dataset, self.windows[i-1]['context'], window['context'])
            value = torch.cat((history, future[i:i+1]), 1)
            for j, step in enumerate(self.times):
                clean = self.predict(value, history, i, step)
                next_alpha = self.alpha[self.times[j+1]] if j+1 < len(self.times) else self.alpha.new_tensor(1.)
                value = ddim_transition(value, clean, self.alpha[step], next_alpha)
            obj = project_to_so3(value[:, 2:, 219:228].reshape(1, 14, 3, 3)).flatten(2)
            value = torch.cat((value[:, 2:, :219], obj, value[:, 2:, 228:]), -1)
            predictions.append(torch.cat((history, value), 1))
        return torch.cat(predictions)

    @torch.enable_grad()
    def sample(self, latent):
        """Use the optimization forward path for reference and final readouts.

        PyTorch's frozen Transformer has a separate inference fast path. A
        grad-enabled latent keeps its arithmetic identical to the DNO decoder.
        """
        return self.decode(latent.detach().requires_grad_(True)).detach()

    def world_windows(self, prediction):
        positions = self.dataset.denormalize_torch(prediction[..., :84]).reshape(-1, 16, 28, 3)
        points = (self.mats[:, None, None, :3, :3] @ positions[..., None]).squeeze(-1) + self.mats[:, None, None, :3, 3]
        rotation = self.mats[:, None, None, :3, :3] @ transforms.rotation_6d_to_matrix(prediction[..., 84:216].reshape(-1, 16, 22, 6))
        obj = self.dataset.denormalize_torch(prediction[..., 216:219], is_object=True)
        obj = (self.mats[:, None, :3, :3] @ obj[..., None]).squeeze(-1) + self.mats[:, None, :3, 3]
        prefixes = torch.cat([w['context']['obj_rot_mat_prefix'] for w in self.windows])
        references = torch.cat([w['context']['obj_rot_mat_ref'] for w in self.windows])
        obj_rot = prefixes[:, None] @ prediction[..., 219:228].reshape(-1, 16, 3, 3) @ references[:, None]
        return dict(points_world=points, global_rotation=rotation,
                    object_translation_world=obj, object_rotation_world=obj_rot)

    def world(self, prediction):
        windows = self.world_windows(prediction)
        result = {k: torch.cat([v[0]] + [w[2:] for w in v[1:]]) for k, v in windows.items()}
        result['global_rot_6d'] = transforms.matrix_to_rotation_6d(result.pop('global_rotation'))
        return result

    def native(self, prediction):
        world = self.world(prediction)
        rotation = transforms.rotation_6d_to_matrix(world['global_rot_6d'])
        local = _local_from_global(rotation)
        quaternion = native_quaternion_interpolation(transforms.matrix_to_quaternion(local))
        pose = transforms.matrix_to_axis_angle(transforms.quaternion_to_matrix(quaternion))
        translation = native_linear_interpolation(world['points_world'][:, 0]) + self.translation_offset
        object_translation = native_linear_interpolation(world['object_translation_world'])
        object_quaternion = transforms.matrix_to_quaternion(world['object_rotation_world'])[:, None]
        object_rotation = transforms.quaternion_to_matrix(native_quaternion_interpolation(object_quaternion)[:, 0])
        object_rotation = torch.cat((object_rotation[:-3], world['object_rotation_world'][-1:].expand(3, -1, -1)))
        return pose, translation, object_translation, object_rotation

    def coarse_body(self, prediction):
        world = self.world_windows(prediction)
        local = _local_from_global(world['global_rotation'])
        offsets = self.rest_offsets.expand(*local.shape[:2], 24, 3).clone()
        offsets[..., 0, :] = offsets[..., 0, :] + world['points_world'][..., 0, :] + self.translation_offset
        return _forward_kinematics(local, offsets)[1]

    @torch.no_grad()
    def teacher_target(self, prediction, view, iteration, capture=False):
        current = prediction.detach()
        level = self.settings['hsi_level']
        diffusion = self.sampler.inner_hoi.diffusion
        timestep = torch.full((1,), level, device=self.device, dtype=torch.long)
        targets, coverage, bundles = [], [], []
        for i, window in enumerate(self.windows):
            seed = 42 + 300000000 + self.ordinal * 100000 + i * 1000 + iteration
            generator = torch.Generator(device=self.device).manual_seed(seed)
            clean = current[i:i+1]
            noise = torch.randn(clean.shape, device=self.device, dtype=clean.dtype, generator=generator)
            noisy = diffusion.q_sample(clean, timestep, noise)
            noisy = empty_motion_view(noisy, diffusion.sqrt_one_minus_alpha_bar[level], noise[..., 216:])
            context = human_goal_context(window['context'])
            if view == 'wrong':
                context = rotated_scene_context(context, self.task['start_location'])
            with torch.random.fork_rng(devices=[self.device.index]):
                torch.manual_seed(seed)
                common = masked_object_arguments(self.sampler._hsi_model_arguments(clean, clean, timestep, context))
            self.teacher.calls += 1
            target = self.teacher.model(noisy, *common, is_sample=True)
            target = torch.cat((clean[:, :2], target[:, 2:]), 1)
            targets.append(target)
            coverage.append(observation_outside(self.dataset, common, context, clean, self.sampler.hsi_sampler.emb_f))
            if capture:
                bundles.append(dict(noisy=noisy.cpu(), arguments=[v.cpu() for v in common], target=target.cpu()))
        target = self.coarse_body(torch.cat(targets)).detach()
        self.last_teacher = dict(view=view, level=level, outside_fraction=sum(coverage)/len(coverage),
                                 outside_fraction_by_window=coverage)
        return target, bundles

    def teacher_loss(self, prediction, target):
        return (self.coarse_body(prediction)[:, 2:] - target[:, 2:]).square().mean() / .05**2


def root_local_body(pose, joints):
    rotation = transforms.axis_angle_to_matrix(pose[:, 0])
    return (rotation[:, None].transpose(-1, -2) @ (joints-joints[:, :1])[..., None]).squeeze(-1)


class _JointPhysicalLoss(torch.autograd.Function):
    """Native surfaces with exact first derivatives, including object motion."""
    @staticmethod
    def forward(ctx, pose, translation, object_translation, object_rotation, objective):
        from utils import run_smplx_model, SMPLX_JOINTS_28
        values = (pose, translation, object_translation, object_rotation)
        derivatives = [torch.zeros_like(v) for v in values]
        sums = pose.new_zeros(len(objective.term_names))
        weights = sums.new_tensor(objective.weights)
        with torch.enable_grad():
            for lo in range(0, len(pose), 24):
                start, stop = max(lo-2, 0), min(lo+24, len(pose))
                p, t, op, orm = [v[start:stop].detach().requires_grad_(True) for v in values]
                vertices, joints = run_smplx_model(p, t, objective.source['betas'], objective.source['gender'],
                    joints_ind=SMPLX_JOINTS_28, smpl_model=objective.model)
                terms = objective.chunk_terms(p, joints, vertices, op, orm, start, lo, stop)
                gradients = torch.autograd.grad((terms*weights).sum(), (p, t, op, orm))
                for result, gradient in zip(derivatives, gradients):
                    result[start:stop] += gradient
                sums += terms.detach()
        ctx.save_for_backward(*derivatives)
        objective.last_terms = sums.cpu().tolist()
        return (sums*weights).sum()

    @staticmethod
    def backward(ctx, upstream):
        return tuple(upstream*v for v in ctx.saved_tensors) + (None,)


class JointMotionObjective:
    term_names = ('body', 'hand', 'stance_height', 'stance_speed', 'local_velocity',
                  'trajectory_acceleration', 'goal', 'human_scene', 'object_scene', 'domain')

    @torch.no_grad()
    def __init__(self, source, model, object_vertices, sdf, info, task, baseline, scales):
        from types import SimpleNamespace
        from test_infbagel_hosi import _subsample_seed
        from utils import run_smplx_model, SMPLX_JOINTS_28
        self.source, self.model, self.task, self.scales = source, model, task, scales
        self.length = len(source['pose'])
        self.object_vertices = object_vertices
        generator = torch.Generator().manual_seed(_subsample_seed(42, task['scene_name'], task['test_idx']))
        indices = torch.randperm(len(object_vertices), generator=generator)[:10475].to(object_vertices.device)
        self.scene_object_vertices = object_vertices[indices]
        self.scene = NativeSceneDifferential(SimpleNamespace(translation=source['translation']), model, sdf, info)
        self.lower = self.scene.centroid-self.scene.extent/2
        self.upper = self.scene.centroid+self.scene.extent/2
        self.floor = baseline['feet_height']/100
        self.local_reference = root_local_body(source['pose'], source['joints'])
        self.hand_reference = object_frame_hands(source['joints'], source['object_translation'], source['object_rotation'])
        self.reference_chunks = {}
        self.source_fk_reference_max_error_m = 0.
        for lo in range(0, self.length, 24):
            start, stop = max(lo-2, 0), min(lo+24, self.length)
            reference = run_smplx_model(source['pose'][start:stop], source['translation'][start:stop],
                source['betas'], source['gender'], joints_ind=SMPLX_JOINTS_28, smpl_model=model)
            self.reference_chunks[start, stop] = reference
            self.source_fk_reference_max_error_m = max(self.source_fk_reference_max_error_m,
                float((reference[1]-source['joints'][start:stop]).abs().max()))
        self.hand_mask = torch.cat([native_hand_distances(source['joints'][lo:lo+24],
            self.objects(source['object_translation'][lo:lo+24], source['object_rotation'][lo:lo+24], object_vertices)) < .05
            for lo in range(0, self.length, 24)])
        self.stance = source['joints'][:, FEET, 1] < self.floor + source['pose'].new_tensor((.08, .08, .04, .04))
        self.stance_pairs = self.stance[1:] & self.stance[:-1]
        self.counts = dict(hand=int(self.hand_mask.sum()), stance=int(self.stance.sum()), pairs=int(self.stance_pairs.sum()))
        self.hs_scale = max(baseline['scene_human_penetration_s_mean'], 1.)
        self.os_scale = max(baseline['scene_obj_penetration_s_mean'], 1.)
        self.weights = [1.]*len(self.term_names)
        self.last_terms = None

    def objects(self, position, rotation, points):
        return (rotation[:, None] @ points[None, :, :, None]).squeeze(-1) + position[:, None]

    def outside(self, points):
        return ((self.lower-points).clamp_min(0) + (points-self.upper).clamp_min(0)).norm(dim=-1)

    def chunk_terms(self, pose, joints, vertices, object_position, object_rotation, start, lo, stop):
        source, scales, length = self.source, self.scales, self.length
        offset = lo-start
        reference_vertices, reference_joints = self.reference_chunks[start, stop]
        local = root_local_body(pose, joints)
        local_delta = local-root_local_body(source['pose'][start:stop], reference_joints)
        body = local_delta[offset:].square().sum()/(length*28*3*scales['body_m']**2)
        hands = object_frame_hands(joints, object_position, object_rotation)-object_frame_hands(
            reference_joints, source['object_translation'][start:stop], source['object_rotation'][start:stop])
        hand = (hands[offset:].square()*self.hand_mask[lo:stop, :, None]).sum()/(max(self.counts['hand'], 1)*3*scales['hand_m']**2)
        feet = joints[:, FEET]
        ref_feet = reference_joints[:, FEET]
        height = ((feet[offset:, :, 1]-ref_feet[offset:, :, 1]).square()*self.stance[lo:stop]).sum()/(max(self.counts['stance'], 1)*scales['stance_height_m']**2)
        first_link = max(lo, 1)-start-1
        speed = ((feet[1:, :, (0, 2)]-feet[:-1, :, (0, 2)])*30).norm(dim=-1)
        ref_speed = ((ref_feet[1:, :, (0, 2)]-ref_feet[:-1, :, (0, 2)])*30).norm(dim=-1)
        slip = ((speed-ref_speed).clamp_min(0)[first_link:].square()*self.stance_pairs[max(lo, 1)-1:stop-1]).sum()/(max(self.counts['pairs'], 1)*scales['stance_speed_m_s']**2)
        local_velocity = ((local_delta[1:]-local_delta[:-1])*30)[first_link:].square().sum()/((length-1)*28*3*scales['local_velocity_m_s']**2)
        route = torch.stack((joints[:, 0]-reference_joints[:, 0],
                             object_position-source['object_translation'][start:stop]), 1)
        first_second = max(lo, 2)-start-2
        acceleration = ((route[2:]-2*route[1:-1]+route[:-2])*900)[first_second:].square().sum()/((length-2)*2*3*scales['trajectory_acceleration_m_s2']**2)
        goal = body.new_zeros(())
        if stop == length:
            human_error = joints[-1, 0, (0, 2)]-joints.new_tensor(self.task['pelvis_goal'])[[0, 2]]
            object_error = object_position[-1]-object_position.new_tensor(self.task['object_goal'])
            goal = (human_error.square().sum()+object_error.square().sum())/(5*scales['goal_m']**2)
        objects = self.objects(object_position[offset:], object_rotation[offset:], self.scene_object_vertices)
        ref_objects = self.objects(source['object_translation'][lo:stop], source['object_rotation'][lo:stop], self.scene_object_vertices)
        hs = self.scene.frame_sums(vertices[offset:]).sum()/(length*self.hs_scale)
        os = self.scene.frame_sums(objects).sum()/(length*self.os_scale)
        outside_h = (self.outside(vertices[offset:])-self.outside(reference_vertices[offset:])).clamp_min(0).square().sum()/(length*vertices.shape[1])
        outside_o = (self.outside(objects)-self.outside(ref_objects)).clamp_min(0).square().sum()/(length*objects.shape[1])
        return torch.stack((body, hand, height, slip, local_velocity, acceleration, goal, hs, os,
                            (outside_h+outside_o)/scales['domain_m']**2))

    def __call__(self, pose, translation, object_translation, object_rotation):
        return _JointPhysicalLoss.apply(pose, translation, object_translation, object_rotation, self)

    def term_record(self):
        return dict(zip(self.term_names, self.last_terms))


def manipulation_contact_mask(distances, object_position, targets):
    """Fix a main hand and fill its manipulation interval from source geometry."""
    contact = distances < .05
    counts = contact.sum(0)
    tied = counts == counts.max()
    main = int(distances.mean(0).masked_fill(~tied, float('inf')).argmin())
    moving = torch.zeros(len(distances), dtype=torch.bool, device=distances.device)
    moving[1:] = (object_position[1:]-object_position[:-1]).norm(dim=-1)*30 > targets['object_motion_speed_m_s']
    active = torch.where(contact.any(-1) | moving)[0]
    mask = contact.clone()
    interval = None
    if len(active):
        interval = (int(active[0]), int(active[-1])+1)
        mask[interval[0]:interval[1], main] = True
    mask[:3] = False
    return mask, main, interval


class MetricMotionObjective(JointMotionObjective):
    """Repair grasp, collision and stance quality on the generated native motion."""
    term_names = ('body', 'contact', 'grasp_velocity', 'stance_height', 'stance_speed',
                  'local_velocity', 'trajectory_acceleration', 'goal', 'human_scene',
                  'object_scene', 'human_object', 'human_clearance', 'object_clearance', 'domain')

    @torch.no_grad()
    def __init__(self, source, model, object_vertices, sdf, info, task, baseline, scales,
                 targets, object_sdf, object_info):
        super().__init__(source, model, object_vertices, sdf, info, task, baseline, scales)
        self.targets = targets
        device = source['pose'].device
        self.object_sdf = torch.as_tensor(object_sdf, dtype=torch.float32, device=device)[None]
        self.object_centroid = torch.as_tensor(object_info['centroid'], dtype=torch.float32, device=device)[None]
        self.object_extent = torch.as_tensor(object_info['extents'], dtype=torch.float32, device=device)[None]
        self.distances = torch.cat([native_hand_distances(source['joints'][lo:lo+24], self.objects(
            source['object_translation'][lo:lo+24], source['object_rotation'][lo:lo+24], object_vertices))
            for lo in range(0, self.length, 24)])
        self.contact_mask, self.main_hand, self.contact_interval = manipulation_contact_mask(
            self.distances, source['object_translation'], targets)
        self.contact_pairs = self.contact_mask[1:] & self.contact_mask[:-1]
        self.contact_count, self.contact_pair_count = int(self.contact_mask.sum()), int(self.contact_pairs.sum())

    def object_signed(self, vertices, position, rotation):
        from eval_metrics import compute_signed_distances
        from utils import yup_to_zup, yup_to_zup_rotation_matrix
        local = yup_to_zup(vertices)-yup_to_zup(position)[:, None]
        local = (yup_to_zup_rotation_matrix(rotation).transpose(1, 2) @ local.transpose(1, 2)).transpose(1, 2)
        return compute_signed_distances(self.object_sdf, self.object_centroid, self.object_extent, local)

    def scene_signed(self, points):
        normalized = (points.float()-self.scene.centroid.reshape(1, 1, 3))/(self.scene.extent/2)
        values = torch.nn.functional.grid_sample(self.scene.sdf,
            normalized[:, :, [2, 1, 0]].reshape(1, -1, 1, 1, 3), padding_mode='border', align_corners=True)
        return values.reshape(points.shape[:2])*self.scene.extent/2

    def chunk_terms(self, pose, joints, vertices, object_position, object_rotation, start, lo, stop):
        source, scales, targets, length = self.source, self.scales, self.targets, self.length
        offset = lo-start
        reference_vertices, reference_joints = self.reference_chunks[start, stop]
        local = root_local_body(pose, joints)
        local_delta = local-root_local_body(source['pose'][start:stop], reference_joints)
        body = local_delta[offset:].square().sum()/(length*28*3*scales['body_m']**2)
        objects_full = self.objects(object_position[offset:], object_rotation[offset:], self.object_vertices)
        distance = native_hand_distances(joints[offset:], objects_full)
        contact = ((distance-targets['contact_distance_m']).clamp_min(0).square()*
                   self.contact_mask[lo:stop]).sum()/(max(self.contact_count, 1)*targets['contact_scale_m']**2)
        first_link = max(lo, 1)-start-1
        hands = object_frame_hands(joints, object_position, object_rotation)
        grasp_velocity = (((hands[1:]-hands[:-1])*30)[first_link:].square().sum(-1)*
            self.contact_pairs[max(lo, 1)-1:stop-1]).sum()/(max(self.contact_pair_count, 1)*3*targets['grasp_speed_m_s']**2)
        feet, ref_feet = joints[:, FEET], reference_joints[:, FEET]
        height = ((feet[offset:, :, 1]-ref_feet[offset:, :, 1]).square()*self.stance[lo:stop]).sum()/(max(self.counts['stance'], 1)*scales['stance_height_m']**2)
        velocity = (feet[1:, :, (0, 2)]-feet[:-1, :, (0, 2)])*30
        slip = (velocity[first_link:].square().sum(-1)*self.stance_pairs[max(lo, 1)-1:stop-1]).sum()/(max(self.counts['pairs'], 1)*scales['stance_speed_m_s']**2)
        local_velocity = ((local_delta[1:]-local_delta[:-1])*30)[first_link:].square().sum()/((length-1)*28*3*scales['local_velocity_m_s']**2)
        route = torch.stack((joints[:, 0]-reference_joints[:, 0], object_position-source['object_translation'][start:stop]), 1)
        first_second = max(lo, 2)-start-2
        acceleration = ((route[2:]-2*route[1:-1]+route[:-2])*900)[first_second:].square().sum()/((length-2)*2*3*scales['trajectory_acceleration_m_s2']**2)
        goal = body.new_zeros(())
        if stop == length:
            human_error = joints[-1, 0, (0, 2)]-joints.new_tensor(self.task['pelvis_goal'])[[0, 2]]
            object_error = object_position[-1]-object_position.new_tensor(self.task['object_goal'])
            goal = (human_error.square().sum()+object_error.square().sum())/(5*scales['goal_m']**2)
        objects = self.objects(object_position[offset:], object_rotation[offset:], self.scene_object_vertices)
        ref_objects = self.objects(source['object_translation'][lo:stop], source['object_rotation'][lo:stop], self.scene_object_vertices)
        signed_h, signed_o = self.scene_signed(vertices[offset:]), self.scene_signed(objects)
        hs = (-signed_h).clamp_min(0).sum()/(length*self.hs_scale)
        os = (-signed_o).clamp_min(0).sum()/(length*self.os_scale)
        human_object = (-self.object_signed(vertices[offset:], object_position[offset:], object_rotation[offset:])).clamp_min(0).sum()/(length*targets['human_object_sum_m'])
        clearance = [targets['clearance_weight']*(targets['clearance_margin_m']-signed.amin(-1)).clamp_min(0).square().sum()/(length*targets['clearance_scale_m']**2) for signed in (signed_h, signed_o)]
        outside_h = (self.outside(vertices[offset:])-self.outside(reference_vertices[offset:])).clamp_min(0).square().sum()/(length*vertices.shape[1])
        outside_o = (self.outside(objects)-self.outside(ref_objects)).clamp_min(0).square().sum()/(length*objects.shape[1])
        return torch.stack((body, contact, grasp_velocity, height, slip, local_velocity,
                            acceleration, goal, hs, os, human_object, *clearance,
                            (outside_h+outside_o)/scales['domain_m']**2))


@torch.no_grad()
def motion_measures(objective, motion):
    source = objective.source
    joints, original = motion['joints'], source['joints']
    local = root_local_body(motion['pose'], joints)
    relative = object_frame_hands(joints, motion['object_translation'], motion['object_rotation'])
    hand_errors = (relative-objective.hand_reference).norm(dim=-1)
    distances = torch.cat([native_hand_distances(joints[lo:lo+24], objective.objects(
        motion['object_translation'][lo:lo+24], motion['object_rotation'][lo:lo+24], objective.object_vertices))
        for lo in range(0, len(joints), 24)])
    hand_count = objective.counts['hand']
    feet = joints[:, FEET]
    support = feet[..., 1] < objective.floor + joints.new_tensor((.08, .08, .04, .04))
    rotation = transforms.axis_angle_to_matrix(motion['pose'])
    original_rotation = transforms.axis_angle_to_matrix(source['pose'])
    rotation_delta = transforms.matrix_to_axis_angle(original_rotation.transpose(-1, -2) @ rotation).norm(dim=-1)
    seam_indices = torch.arange(48, len(joints), 42, device=joints.device)
    speed = (joints[1:]-joints[:-1]).norm(dim=-1)*3000
    paths = torch.stack((joints[:, 0], motion['object_translation']), 1)
    acceleration = (paths[2:]-2*paths[1:-1]+paths[:-2]).norm(dim=-1)*90000
    objects = objective.objects(motion['object_translation'], motion['object_rotation'], objective.scene_object_vertices)
    return dict(
        native_body28_mean_displacement_cm=float((joints-original).norm(dim=-1).mean()*100),
        root_local_body_mean_drift_cm=float((local-objective.local_reference).norm(dim=-1).mean()*100),
        active_hand_mean_drift_cm=float((hand_errors*objective.hand_mask).sum()/max(hand_count, 1)*100),
        source_hand_contact_retention=float(((distances < .05)&objective.hand_mask).sum()/max(hand_count, 1)),
        active_hand_samples=hand_count, stance_samples=objective.counts['stance'],
        source_floor_support_fraction=float(support.float().mean()),
        nonroot_rotation_mean_change_deg=float(rotation_delta[:, 1:].mean()*180/torch.pi),
        root_rotation_mean_change_deg=float(rotation_delta[:, 0].mean()*180/torch.pi),
        root_max_change_m=float((joints[:, 0]-original[:, 0]).norm(dim=-1).max()),
        object_mean_change_cm=float((motion['object_translation']-source['object_translation']).norm(dim=-1).mean()*100),
        root_path_m=float((paths[1:, 0]-paths[:-1, 0]).norm(dim=-1).sum()),
        object_path_m=float((paths[1:, 1]-paths[:-1, 1]).norm(dim=-1).sum()),
        seam_speed_mean_cm_s=float(speed[seam_indices-1].mean()) if len(seam_indices) else 0.,
        seam_count=len(seam_indices), trajectory_acceleration_mean_cm_s2=float(acceleration.mean()),
        root_acceleration_mean_cm_s2=float(acceleration[:, 0].mean()),
        object_acceleration_mean_cm_s2=float(acceleration[:, 1].mean()),
        initial_body_max_error_m=float((joints[:3]-original[:3]).abs().max()),
        initial_object_max_error_m=float((motion['object_translation'][:3]-source['object_translation'][:3]).abs().max()),
        initial_object_rotation_max_error=float((motion['object_rotation'][:3]-source['object_rotation'][:3]).abs().max()),
        human_outside_fraction=float((objective.outside(motion['verts']) > 0).float().mean()),
        object_outside_fraction=float((objective.outside(objects) > 0).float().mean()))


@torch.enable_grad()
def gradient_audit(decoder, objective, latent, view, iteration):
    value = latent.detach().requires_grad_(True)
    prediction = decoder.decode(value)
    native = decoder.native(prediction)
    records, gradients = {}, {}
    weights = list(objective.weights)
    for i, name in enumerate(objective.term_names):
        objective.weights = [float(j == i) for j in range(len(weights))]
        loss = objective(*native)
        gradient, = torch.autograd.grad(loss, value, retain_graph=True)
        gradients[name] = gradient.detach()
        records[name] = dict(value=objective.last_terms[i], gradient_norm=float(gradient.norm()))
    objective.weights = weights
    physical = sum(weight*gradients[name] for weight, name in zip(weights, objective.term_names))
    if view is not None:
        target, _ = decoder.teacher_target(prediction, view, iteration)
        loss = decoder.teacher_loss(prediction, target)
        gradient, = torch.autograd.grad(loss, value)
        gradients['hsi'] = gradient.detach()
        records['hsi'] = dict(value=float(loss.detach()), gradient_norm=float(gradient.norm()), weight=decoder.settings['hsi_weight'])
    else:
        gradients['hsi'] = torch.zeros_like(value)
    scene = gradients['human_scene']+gradients['object_scene']
    product = scene.norm()*gradients['hsi'].norm()
    total = physical+decoder.settings['hsi_weight']*gradients['hsi']
    for gradient in gradients.values():
        assert torch.isfinite(gradient).all(), 'nonfinite HOI DNO objective derivative'
    groups = {'human_root': slice(0, 3), 'human_rotation': slice(84, 216),
              'object_translation': slice(216, 219), 'object_rotation': slice(219, 228)}
    return dict(terms=records, physical_gradient_norm=float(physical.norm()),
                total_without_latent_penalty_norm=float(total.norm()),
                scene_hsi_cosine=float((scene*gradients['hsi']).sum()/product) if float(product) > 0 else None,
                group_gradient_norms={name: float(total[:, block].norm()) for name, block in groups.items()}), gradients


def _optimizer_history(hist):
    names = ('step', 'lr', 'loss', 'loss_diff', 'loss_decorrelate', 'grad_norm', 'diff_norm')
    return [{k: float(row[k][0]) for k in names} for row in hist]


@torch.enable_grad()
def optimize_hoi_latent(decoder, objective, initial, view, settings, destination, stage, resume=False):
    sys.path.insert(0, str(destination['repository']))
    from dno import DNO, DNOOptions
    traces, history = [], []
    weight = 0. if view is None else settings['hsi_weight']

    def generate(latent):
        return decoder.decode(latent).unsqueeze(0)

    def criterion(value):
        prediction = value[0]
        physical = objective(*decoder.native(prediction))
        terms = objective.term_record()
        hsi = physical.new_zeros(())
        if weight:
            target, _ = decoder.teacher_target(prediction, view, engine.step_count)
            hsi = decoder.teacher_loss(prediction, target)
        traces.append(dict(iteration=engine.step_count, **terms, hsi=float(hsi.detach()),
                           hsi_weight=weight, hsi_query=None if view is None else dict(decoder.last_teacher)))
        return (physical+weight*hsi).reshape(1)

    options = DNOOptions(num_opt_steps=settings['editing_steps'], lr=settings['lr'],
                         lr_warm_up_steps=settings['warmup'], decorrelate_scale=0.,
                         diff_penalty_scale=settings['diff_penalty_scale'], perturb_scale=0.)
    torch.manual_seed(42)
    engine = DNO(generate, criterion, initial, options)
    old = sorted(destination['path'].glob(stage+'-step*.pt')) if resume else []
    if old:
        saved = torch.load(old[-1], map_location=initial.device, weights_only=False)
        assert saved['options'] == vars(options) and saved['commit'] == destination['commit']
        assert torch.equal(saved['initial'], initial)
        with torch.no_grad():
            engine.current_z.copy_(saved['latent'])
        engine.optimizer.load_state_dict(saved['optimizer'])
        engine.step_count = saved['step']
        traces, history = saved['traces'], saved['history']
        torch.set_rng_state(saved['cpu_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), initial.device)

    def finite_gradient(gradient):
        assert torch.isfinite(gradient).all(), 'nonfinite gradient through HOI generator'
        return gradient
    engine.current_z.register_hook(finite_gradient)
    torch.cuda.synchronize(initial.device)
    started = time.perf_counter()
    while engine.step_count < settings['editing_steps']:
        count = min(settings['checkpoint_every'], settings['editing_steps']-engine.step_count)
        engine(count)
        history += _optimizer_history(engine.hist)
        engine.hist.clear()
        checkpoint_path = destination['path']/f'{stage}-step{engine.step_count:04d}.pt'
        with checkpoint_path.open('xb') as handle:
            torch.save(dict(latent=engine.current_z.detach().cpu(), initial=engine.start_z.cpu(),
                optimizer=engine.optimizer.state_dict(), step=engine.step_count, options=vars(options),
                traces=traces, history=history, view=view, commit=destination['commit'],
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(initial.device)), handle)
        torch.cuda.synchronize(initial.device)
        peak = torch.cuda.max_memory_allocated(initial.device)/1024**3
        assert peak <= destination['memory_limit'], peak
        print(json.dumps(dict(task=decoder.ordinal, arm=stage, step=engine.step_count,
            loss=history[-1]['loss'], gradient_norm=history[-1]['grad_norm'],
            seconds=time.perf_counter()-started, peak_memory_gib=peak, checkpoint=str(checkpoint_path))), flush=True)
    trace_path = destination['path']/(stage+'-trace.json')
    if not trace_path.exists():
        write_json(trace_path, dict(terms=traces, optimizer=history, seconds=time.perf_counter()-started))
    return engine.current_z.detach()


def hoi_dno_task(teacher, windows, ddpm_source, model, sdf, info, evaluate, baseline,
                 task, ordinal, protocol, destination, commit, resume=False,
                 object_sdf=None, object_info=None, previous_inputs=None, replay_from=None):
    from utils import run_smplx_model, SMPLX_JOINTS_28
    started = time.perf_counter()
    decoder = HOIDDIM(teacher, windows, ddpm_source, model, task, ordinal, protocol['method'])
    initial = decoder.initial_latent()
    with torch.no_grad():
        torch.cuda.synchronize(decoder.device); began = time.perf_counter()
        prediction = decoder.sample(initial)
        if previous_inputs is not None:
            previous = torch.load(previous_inputs, map_location=decoder.device, weights_only=False)
            assert torch.equal(initial, previous['initial'])
            assert torch.equal(prediction, previous['source_prediction']), 'previous DDIM source differs'
        torch.cuda.synchronize(decoder.device); generation_seconds = time.perf_counter()-began
        source = dict(ddpm_source, **dict(zip(('pose', 'translation', 'object_translation', 'object_rotation'), decoder.native(prediction))))
        source['verts'], source['joints'] = decode_body(source, model)
        source_metrics = evaluate(source)
        began = time.perf_counter(); replay = decoder.sample(initial)
        torch.cuda.synchronize(decoder.device); replay_seconds = time.perf_counter()-began
        assert torch.equal(prediction, replay), 'same HOI noise must reproduce the source exactly'
        world_windows = decoder.world_windows(prediction)
        coarse_pose = transforms.matrix_to_axis_angle(_local_from_global(world_windows['global_rotation'])).reshape(-1, 22, 3)
        coarse_translation = (world_windows['points_world'][..., 0, :]+decoder.translation_offset).reshape(-1, 3)
        native_coarse = []
        for lo in range(0, len(coarse_pose), 24):
            _, joints = run_smplx_model(coarse_pose[lo:lo+24], coarse_translation[lo:lo+24], source['betas'],
                source['gender'], joints_ind=SMPLX_JOINTS_28, smpl_model=model)
            native_coarse.append(joints[:, list(range(22))+[24, 26]])
        coarse_fk_error = float((torch.cat(native_coarse).reshape(-1, 16, 24, 3)-decoder.coarse_body(prediction)).abs().max())
        assert coarse_fk_error <= 1e-5, coarse_fk_error
    object_vertices = teacher.dataset.obj_rest_verts[task['object_name']]
    targets = protocol['method'].get('metric_targets')
    arguments = (source, model, object_vertices, sdf, info, task, source_metrics,
                 protocol['method']['physical_scales'])
    objective = (JointMotionObjective(*arguments) if targets is None else
                 MetricMotionObjective(*arguments, targets, object_sdf, object_info))
    metrics = {}

    @torch.no_grad()
    def record(name, motion, native_metrics=None, predicted=None):
        row = dict(evaluate(motion) if native_metrics is None else native_metrics, **motion_measures(objective, motion))
        state = {k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in motion.items() if k != 'verts'}
        if predicted is not None:
            state['prediction'] = predicted.detach().cpu()
        path = destination/(name+'.pt')
        if not path.exists():
            with path.open('xb') as handle: torch.save(state, handle)
        path = destination/(name+'-metrics.json')
        if not path.exists(): write_json(path, row)
        metrics[name] = row
        print(json.dumps(dict(task=ordinal, stage=name, hs=row['scene_human_penetration_s_mean'],
            os=row['scene_obj_penetration_s_mean'], contact=row['contact_percent'],
            hand_cm=row['active_hand_mean_drift_cm'])), flush=True)
        return row

    record('DDPM_reference', ddpm_source, baseline)
    record('source', source, source_metrics, prediction)
    postprocess = {}
    if targets is not None:
        mask_path = destination/'contact-targets.pt'
        if not mask_path.exists():
            with mask_path.open('xb') as handle:
                torch.save(dict(mask=objective.contact_mask.cpu(), pairs=objective.contact_pairs.cpu(),
                    source_distances_m=objective.distances.cpu(), source_contact=objective.hand_mask.cpu(),
                    source_stance=objective.stance.cpu(), main_hand=objective.main_hand,
                    interval=objective.contact_interval, count=objective.contact_count,
                    pair_count=objective.contact_pair_count, targets=targets), handle)
        for name, motion in [('DDPM_reference', ddpm_source), ('source', source)]:
            postprocess[name] = postprocess_hoi_motion(name, motion, metrics[name], model,
                object_vertices, sdf, info, task, evaluate, destination, record, resume)
    inputs_path = destination/'inputs.pt'
    if inputs_path.exists():
        old_inputs = torch.load(inputs_path, map_location='cpu', weights_only=False)
        assert old_inputs['commit'] == commit
        assert torch.equal(old_inputs['initial'], initial.cpu())
        assert torch.equal(old_inputs['source_prediction'], prediction.cpu())
    else:
        with inputs_path.open('xb') as handle:
            torch.save(dict(initial=initial.cpu(), source_prediction=prediction.cpu(), commit=commit,
                contexts=[{k:v.cpu() if torch.is_tensor(v) else v for k,v in w['context'].items()
                           if k not in ('obj_rest_verts', 'obj_vert_normals', 'static_occ_cache')} for w in windows],
                hoi_arguments=[{k:v.cpu() for k,v in w['arguments'].items()} for w in windows]), handle)
    physical_path = destination/'source-physical.json'
    if not physical_path.exists():
        objective(*decoder.native(prediction))
        terms = objective.term_record()
        source_hs_error = abs(terms['human_scene']*objective.hs_scale-source_metrics['scene_human_penetration_s_mean'])
        source_os_error = abs(terms['object_scene']*objective.os_scale-source_metrics['scene_obj_penetration_s_mean'])
        normalized_error = dict(
            human=abs(terms['human_scene']-source_metrics['scene_human_penetration_s_mean']/objective.hs_scale),
            object=abs(terms['object_scene']-source_metrics['scene_obj_penetration_s_mean']/objective.os_scale))
        assert max(normalized_error.values()) <= 1e-5, normalized_error
        if targets is not None:
            human_object = terms['human_object']*targets['human_object_sum_m']
            normalized_error['human_object'] = abs(human_object-source_metrics['human_pen_loss_infbagel'])/max(source_metrics['human_pen_loss_infbagel'], 1.)
            assert normalized_error['human_object'] <= 1e-5, normalized_error
        write_json(physical_path, dict(terms=terms, counts=objective.counts, native_hs_error=source_hs_error,
            native_os_error=source_os_error, normalized_scene_error=normalized_error,
            source_fk_reference_max_error_m=objective.source_fk_reference_max_error_m,
            coarse_fk_native_max_error_m=coarse_fk_error, same_noise_replay_exact=True))
    audits = {}
    for name, value, view, iteration in [('source_correct', initial, 'correct', 0), ('source_wrong', initial, 'wrong', 0)]:
        path = destination/(name+'-gradient.json')
        if path.exists():
            audits[name] = json.loads(path.read_text())
        else:
            audits[name], gradients = gradient_audit(decoder, objective, value, view, iteration)
            write_json(path, audits[name])
            with (destination/(name+'-gradients.pt')).open('xb') as handle:
                torch.save({k:v.cpu() for k,v in gradients.items()}, handle)
    query_path = destination/'source-teacher.pt'
    if not query_path.exists():
        correct, c = decoder.teacher_target(prediction, 'correct', 0, capture=True)
        wrong, w = decoder.teacher_target(prediction, 'wrong', 0, capture=True)
        for first, second in zip(c, w):
            assert torch.equal(first['noisy'], second['noisy'])
            assert all(torch.equal(first['arguments'][i], second['arguments'][i]) for i in range(17) if i not in (0, 15))
        with query_path.open('xb') as handle:
            torch.save(dict(correct=c, wrong=w, fk_target_difference_rms_m=float((correct-wrong).square().mean().sqrt())), handle)
    settings = dict(repository=protocol['dno_repository'], path=destination, commit=commit,
                    memory_limit=protocol['execution']['peak_memory_gib'])
    for name in protocol['method']['arms']:
        view = {'G': None, 'C': 'correct', 'W': 'wrong'}[name]
        if replay_from is None:
            value = optimize_hoi_latent(decoder, objective, initial, view, protocol['method'], settings, name, resume)
        else:
            saved = torch.load(replay_from/(name+'-step0300.pt'), map_location=decoder.device, weights_only=False)
            assert saved['step'] == protocol['method']['editing_steps'] and torch.equal(saved['initial'], initial)
            value = saved['latent']
        with torch.no_grad():
            predicted = decoder.sample(value)
            motion = dict(source, **dict(zip(('pose', 'translation', 'object_translation', 'object_rotation'), decoder.native(predicted))))
            motion['verts'], motion['joints'] = decode_body(motion, model)
        record(name, motion, predicted=predicted)
        if targets is not None:
            postprocess[name] = postprocess_hoi_motion(name, motion, metrics[name], model,
                object_vertices, sdf, info, task, evaluate, destination, record, resume)
        path = destination/(name+'-gradient.json')
        if path.exists():
            audits[name] = json.loads(path.read_text())
        else:
            audits[name], gradients = gradient_audit(decoder, objective, value, view, protocol['method']['editing_steps'])
            write_json(path, audits[name])
            with (destination/(name+'-gradients.pt')).open('xb') as handle:
                torch.save({k:v.cpu() for k,v in gradients.items()}, handle)
    cache_replay = None
    if replay_from is not None:
        cache_replay = compare_hoi_dno_replay(destination, replay_from, metrics)
    torch.cuda.synchronize(decoder.device)
    return dict(task=ordinal, scene=task['scene_name'], object=task['object_name'],
        means=metrics, gradients=audits, postprocess=postprocess, cache_replay=cache_replay,
        windows=len(windows), native_frames=len(source['pose']),
        source_replay_exact=True, coarse_fk_native_max_error_m=coarse_fk_error,
        source_fk_reference_max_error_m=objective.source_fk_reference_max_error_m,
        source_generation_seconds=generation_seconds,
        source_replay_seconds=replay_seconds, seconds=time.perf_counter()-started,
        hoi_calls=decoder.hoi_calls, hsi_calls=teacher.calls,
        peak_memory_gib=torch.cuda.max_memory_allocated(decoder.device)/1024**3)


def compare_hoi_dno_replay(destination, previous, metrics):
    """Compare the current decoder/chain with the immutable cached outputs."""
    cached = json.loads((previous/'metrics.json').read_text())
    metric_abs, metric_normalized, tensor_max = 0., 0., 0.
    tensor_exact = True
    for arm, values in metrics.items():
        for key, value in values.items():
            reference = cached['means'][arm][key]
            difference = abs(float(value)-float(reference))
            metric_abs = max(metric_abs, difference)
            metric_normalized = max(metric_normalized, difference/max(abs(reference), 1.))
        old = torch.load(previous/(arm+'.pt'), map_location='cpu', weights_only=False)
        new = torch.load(destination/(arm+'.pt'), map_location='cpu', weights_only=False)
        for key in ('pose', 'translation', 'joints', 'object_translation', 'object_rotation'):
            tensor_max = max(tensor_max, float((old[key]-new[key]).abs().max()))
            tensor_exact = tensor_exact and torch.equal(old[key], new[key])
    assert tensor_max <= 1e-5 and metric_normalized <= 1e-5, (tensor_max, metric_abs, metric_normalized)
    return dict(previous=str(previous), previous_commit=cached['commit'], stages=len(metrics),
                tensor_max_abs=tensor_max, tensor_exact=tensor_exact,
                metric_max_abs=metric_abs, metric_max_normalized=metric_normalized,
                optimization_steps=0, passed=True)


def postprocess_hoi_motion(name, source, before, model, object_vertices, sdf, info,
                           task, evaluate, destination, record, resume):
    """Apply the frozen relation20 and terminal rule to each complete raw arm."""
    from .surface_edit import SurfaceProblem, apply_terminal_repair
    device = source['pose'].device
    torch.cuda.synchronize(device); began = time.perf_counter()
    relation_name, final_name = name+'_relation', name+'_final'
    relation_path = destination/(relation_name+'.pt')
    solver_path = destination/(relation_name+'-solver.pt')
    if resume and solver_path.exists() and relation_path.exists():
        relation = torch.load(relation_path, map_location=device, weights_only=False)
        relation['verts'], _ = decode_body(relation, model)
        solve = torch.load(solver_path, map_location='cpu', weights_only=False)
    else:
        problem = SurfaceProblem(source, model, object_vertices, sdf, info,
                                 before['feet_height']/100, task, 'relation_20', .2, 20)
        with torch.no_grad():
            risk = 0.
            for lo in range(0, len(source['joints']), 24):
                hi = min(lo+24, len(source['joints']))
                for vertices in (source['verts'][lo:hi], problem.objects(
                        source['object_translation'][lo:hi], source['object_rotation'][lo:hi])):
                    risk += float((-problem.signed(vertices)).clamp_min(0).sum()/(len(source['joints'])*vertices.shape[1]))
        if risk == 0:
            relation, solve = source, dict(trace=[], steps=0, optimization_seconds=0., best_iteration=0, parameters=None)
        else:
            relation, solve = problem.solve(20 if risk <= .005 else 40)
        solve['source_mean_depth_m'] = risk
    row = record(relation_name, relation)
    if not solver_path.exists():
        with solver_path.open('xb') as handle: torch.save(solve, handle)
    torch.cuda.synchronize(device); relation_seconds = time.perf_counter()-began
    final, terminal, candidate, parameters = apply_terminal_repair(
        relation, row, model, object_vertices, sdf, info, task, evaluate)
    record(final_name, final, terminal['metrics'])
    candidate_path = destination/(name+'-terminal-candidate.pt')
    if not candidate_path.exists():
        with candidate_path.open('xb') as handle:
            torch.save(dict(motion={k:v.detach().cpu() if torch.is_tensor(v) else v
                                   for k,v in candidate.items() if k != 'verts'},
                            parameters=parameters, **terminal), handle)
    return dict(relation={k:v for k,v in solve.items() if k != 'parameters'},
                relation_seconds_including_evaluation=relation_seconds, terminal=terminal)


def edit_protections(current, source):
    hand = source['active_hand_samples'] > 0
    return dict(contact=current['contact_percent'] >= source['contact_percent']-.02,
        support=current['source_floor_support_fraction'] >= source['source_floor_support_fraction']-.02,
        foot_sliding=current['foot_sliding'] <= source['foot_sliding']+.02,
        hand_retention=(current['source_hand_contact_retention'] >= .95) if hand else None,
        hand_drift=(current['active_hand_mean_drift_cm'] <= 1.) if hand else None,
        local_body=current['root_local_body_mean_drift_cm'] <= 5.,
        local_rotation=current['nonroot_rotation_mean_change_deg'] <= 10.,
        seam=current['seam_speed_mean_cm_s'] <= source['seam_speed_mean_cm_s']+10.,
        acceleration=current['trajectory_acceleration_mean_cm_s2'] <= source['trajectory_acceleration_mean_cm_s2']+100.,
        initial=max(current['initial_body_max_error_m'], current['initial_object_max_error_m'],
                    current['initial_object_rotation_max_error']) <= 1e-5)


def collect_hoi_dno_records(run_root, task_manifest, protocol=None):
    """Include each task once, retaining the old run identity for cached tasks."""
    tasks = json.loads(Path(task_manifest).read_text())['tasks']
    records = [json.loads(p.read_text()) for p in sorted(Path(run_root).glob('lanes/*/task-*/metrics.json'))]
    if protocol is not None and 'reuse_completed_run' in protocol:
        root = Path(task_manifest).resolve().parents[2]
        cached_ids = {t['canonical_ordinal'] for t in json.loads((root/protocol['reuse_task_manifest']).read_text())['tasks']}
        for p in sorted((root/protocol['reuse_completed_run']).glob('lanes/*/task-*/metrics.json')):
            row = json.loads(p.read_text())
            if row['task'] in cached_ids:
                row['reused_from'] = str(p)
                records.append(row)
        base_arms = ['DDPM_reference', 'source']+protocol['method']['arms']
        stages = {a+suffix for a in base_arms for suffix in ('', '_relation', '_final')}
        for row in records:
            row['means'] = {k:v for k,v in row['means'].items() if k in stages}
            row['postprocess'] = {k:v for k,v in row['postprocess'].items() if k in base_arms}
    assert len(records) == len(tasks) and {r['task'] for r in records} == {t['canonical_ordinal'] for t in tasks}
    return tasks, records if protocol is None else sorted(records, key=lambda r:r['task'])


def compare_released_hosi(records, tasks, reference_path, device):
    """Match the released system by task identity and report all12 table metrics."""
    from .scene_calibration import paired_local_metrics
    keys = ('xy_points_err', 'end_obj_trans_err', 'completed', 'foot_sliding',
            'contact_percent', 'human_pen_loss_infbagel', 'scene_human_penetration_s_mean',
            'scene_human_penetration_s_max', 'scene_human_penetration_frame_ratio',
            'scene_obj_penetration_s_mean', 'scene_obj_penetration_s_max', 'scene_obj_penetration_frame_ratio')
    old = json.loads(Path(reference_path).read_text())['individual_metrics']
    index = {(r['scene_name'], r['test_idx'], r['object_name']):r for r in old}
    assert len(index) == len(old)
    by_task = {'July_released': {str(t['canonical_ordinal']):
        {k:float(index[t['scene_name'], t['test_idx'], t['object_name']][k]) for k in keys} for t in tasks}}
    for arm in ('G_final', 'C_final'):
        by_task[arm] = {str(r['task']):{k:float(r['means'][arm][k]) for k in keys} for r in records}
    names = {str(t['canonical_ordinal']):t['scene_name'] for t in tasks}
    def average(rows):
        return {k:sum(r[k] for r in rows)/len(rows) for k in keys}
    by_scene = {a:{scene:average([v for tid,v in data.items() if names[tid] == scene])
        for scene in sorted(set(names.values()))} for a,data in by_task.items()}
    means = {a:average(list(data.values())) for a,data in by_task.items()}
    comparisons = {}
    for arm in ('G_final', 'C_final'):
        comparisons[arm] = {}
        for key in keys:
            delta = means[arm][key]-means['July_released'][key]
            gain = delta if key in ('completed', 'contact_percent') else -delta
            comparisons[arm][key] = 'better' if gain > 0 else 'worse' if gain < 0 else 'equal'
    return dict(tasks=len(tasks), scenes=len(by_scene['July_released']), source=str(reference_path),
        identity_keys=['scene_name', 'test_idx', 'object_name'], means=means, point_comparison=comparisons,
        all12_better=all(v == 'better' for v in comparisons['C_final'].values()),
        contrasts={a:{unit:paired_local_metrics(data['July_released'], data[a], device)
            for unit,data in [('task', by_task), ('scene', by_scene)]} for a in ('G_final', 'C_final')},
        comparison_scope='Historical system-level comparison with different model representation, sampling and compute; fixed task identities, all historical development use retained.')


def summarize_hoi_dno(run_root, task_manifest, device='cuda:0', protocol=None):
    from .scene_calibration import paired_local_metrics
    run_root = Path(run_root)
    tasks, records = collect_hoi_dno_records(run_root, task_manifest, protocol)
    arms = tuple(records[0]['means'])
    edits = tuple(a for a in ('G', 'C', 'W') if a in arms)
    timed_records = [r for r in records if 'reused_from' not in r]
    keys = tuple(records[0]['means']['source'])
    scenes = sorted({r['scene'] for r in records})
    def mean(rows):
        return {k:sum(float(r[k]) for r in rows)/len(rows) for k in keys}
    # Completion is a measured 0/1 outcome; numeric discovery excludes bools.
    by_task = {a:{str(r['task']):{k:float(r['means'][a][k]) for k in keys} for r in records} for a in arms}
    by_scene = {a:{s:mean([r['means'][a] for r in records if r['scene'] == s]) for s in scenes} for a in arms}
    means = {a:mean([r['means'][a] for r in records]) for a in arms}
    pairs = [('source', 'DDPM_reference')] + [(a, 'source') for a in edits] + [('C', 'G'), ('C', 'DDPM_reference')]
    if 'W' in edits:
        pairs.append(('C', 'W'))
    if 'C_final' in arms:
        for stage in ('relation', 'final'):
            pairs += [(a+'_'+stage, 'source_'+stage) for a in edits]
            pairs += [('C_'+stage, b+'_'+stage) for b in ('G', 'W', 'DDPM_reference') if b in arms]
        pairs += [(a+'_final', a) for a in ('DDPM_reference', 'source')+edits]
    contrasts = {a+'__minus__'+b:{unit:paired_local_metrics(values[b], values[a], device)
        for unit, values in [('task', by_task), ('scene', by_scene)]} for a,b in pairs}
    protections = {a:{str(r['task']):edit_protections(r['means'][a], r['means']['source']) for r in records}
                   for a in edits}
    hand_tasks = [str(r['task']) for r in records if r['means']['source']['active_hand_samples'] > 0]
    conditional_hands = {a:{key:sum(by_task[a][t][key] for t in hand_tasks)/len(hand_tasks)
        if hand_tasks else None for key in ('active_hand_mean_drift_cm', 'source_hand_contact_retention')} for a in arms}
    c, b, g, old = [means[a] for a in ('C', 'source', 'G', 'DDPM_reference')]
    w = means.get('W')
    hs, os = 'scene_human_penetration_s_mean', 'scene_obj_penetration_s_mean'
    aggregate = edit_protections(c, b)
    aggregate['initial'] = all(row['initial'] for row in protections['C'].values())
    aggregate.update(hand_retention=bool(hand_tasks) and conditional_hands['C']['source_hand_contact_retention'] >= .95,
        hand_drift=bool(hand_tasks) and conditional_hands['C']['active_hand_mean_drift_cm'] <= 1.,
        contact_vs_ddpm=c['contact_percent'] >= old['contact_percent']-.02,
        completed_vs_source=c['completed'] >= b['completed'], completed_vs_ddpm=c['completed'] >= old['completed'])
    conditions = dict(source_hs_gain=c[hs] <= .9*b[hs], extra_hsi_hs=c[hs] <= .99*g[hs],
        correct_scene=(c[hs]-w[hs] <= -.005*b[hs]) if w is not None else None, object_scene_source=c[os] <= 1.01*b[os],
        object_scene_geometry=c[os] <= 1.01*g[os], protection=all(aggregate.values()))
    directions = {a+'__minus__'+b:dict(
        improved=sum(r['means'][a][hs] < r['means'][b][hs] for r in records),
        worsened=sum(r['means'][a][hs] > r['means'][b][hs] for r in records),
        equal=sum(r['means'][a][hs] == r['means'][b][hs] for r in records)) for a,b in pairs}
    summary = dict(tasks=len(records), scenes=len(scenes), windows=sum(r['windows'] for r in records),
        native_frames=sum(r['native_frames'] for r in records), means=means, contrasts=contrasts,
        aggregate_protection=aggregate, protections=protections, conditional_hand_means=conditional_hands,
        active_hand_task_count=len(hand_tasks), conditions=conditions, utility=all(conditions.values()) if w is not None else None,
        hs_directions=directions, protection_pass_counts={a:sum(all(v for v in row.values() if v is not None)
            for row in values.values()) for a,values in protections.items()},
        hoi_calls=sum(r['hoi_calls'] for r in timed_records), hsi_calls=sum(r['hsi_calls'] for r in timed_records),
        task_seconds_sum=sum(r['seconds'] for r in timed_records), peak_memory_gib=max(r['peak_memory_gib'] for r in records),
        source_replay_exact=all(r['source_replay_exact'] for r in records),
        test_set_development=True, timing_comparison_valid=False)
    if 'C_final' in arms:
        c, b, g, old = [means[a+'_final'] for a in ('C', 'source', 'G', 'DDPM_reference')]
        w = means.get('W_final')
        def metric_protection(current, reference):
            result = edit_protections(current, reference)
            del result['hand_drift']
            result.update(completion=current['completed'] >= reference['completed'],
                hs=current[hs] <= 1.01*reference[hs], os=current[os] <= 1.01*reference[os],
                hs_frames=current['scene_human_penetration_frame_ratio'] <= reference['scene_human_penetration_frame_ratio']+.002,
                os_frames=current['scene_obj_penetration_frame_ratio'] <= reference['scene_obj_penetration_frame_ratio']+.002)
            return result
        final_protections = {a:{str(r['task']):metric_protection(r['means'][a+'_final'], r['means']['source_final'])
            for r in records} for a in edits}
        final_aggregate = metric_protection(c, b)
        final_aggregate['completion_ddpm'] = c['completed'] >= old['completed']
        final_aggregate['initial'] = all(r['initial'] for r in final_protections['C'].values())
        final_aggregate['hand_retention'] = (conditional_hands['C_final']['source_hand_contact_retention'] >= .95) if hand_tasks else None
        progress = dict(contact=c['contact_percent'] >= b['contact_percent']+.02,
            human_object=c['human_pen_loss_infbagel'] <= .9*b['human_pen_loss_infbagel'],
            foot_sliding=c['foot_sliding'] <= .9*b['foot_sliding'],
            protection=all(v for v in final_aggregate.values() if v is not None))
        scene_conditions = dict(extra_hsi_hs=c[hs] <= .99*g[hs],
                                correct_scene=(c[hs]-w[hs] <= -.005*b[hs]) if w is not None else None)
        objects = sorted({r['object'] for r in records})
        summary.update(raw_conditions=summary['conditions'], raw_utility=summary['utility'],
            conditions=progress, utility=all(progress.values()), hsi_conditions=scene_conditions,
            hsi_scene_utility=all(scene_conditions.values()) if w is not None else None,
            correct_wrong_trajectory_comparison_available=w is not None,
            final_aggregate_protection=final_aggregate,
            final_protections=final_protections,
            final_protection_pass_counts={a:sum(all(v for v in r.values() if v is not None)
                for r in rows.values()) for a,rows in final_protections.items()},
            object_means={obj:{a:mean([r['means'][a] for r in records if r['object']==obj]) for a in arms} for obj in objects},
            relation_seconds_sum=sum(v['relation_seconds_including_evaluation'] for r in timed_records for v in r['postprocess'].values()),
            terminal_seconds_sum=sum(v['terminal']['solver']['arm_seconds_including_evaluation'] for r in timed_records for v in r['postprocess'].values()))
    if protocol is not None and 'reuse_completed_run' in protocol:
        summary.update(reused_tasks=len(records)-len(timed_records), newly_executed_tasks=len(timed_records),
            resource_accounting='Execution counters exclude cached28 historical work and separately recorded replay validation.',
            strata={name:dict(tasks=len(group), means={a:mean([r['means'][a] for r in group]) for a in arms})
                    for name,group in [('reused', [r for r in records if 'reused_from' in r]), ('new', timed_records)]})
        if 'released_reference' in protocol:
            summary['released_comparison'] = compare_released_hosi(records, tasks, protocol['released_reference'], device)
    output = run_root/'analysis'; output.mkdir()
    write_json(output/'summary.json', summary); write_json(output/'records.json', records)
    for arm in arms:
        for unit, values in [('task', by_task[arm]), ('scene', by_scene[arm])]:
            write_json(output/f'{arm}-{unit}.json', dict(metrics=values))
    return summary
