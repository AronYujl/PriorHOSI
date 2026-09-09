"""External pretrained bridge inference in each model's own Python environment.

Input/output NPZ files carry Y-up, metre, pelvis-position conventions. This
module is launched as a leaf file by the native Hydra dispatcher with only the
external repo on PYTHONPATH. CondMDI owns a top-level ``utils`` package; package
execution would also import the native mixer's eager dependencies.
"""

import argparse
import copy
import json
import importlib.metadata
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def bone_lengths(joints):
    return torch.cat((torch.zeros_like(joints[..., :1, 0]),
        (joints[..., 1:, :] - joints[..., list(PARENTS[1:]), :]).norm(dim=-1)), -1)


def retarget_positions(joints, lengths, root):
    """Preserve each world bone direction while changing its length."""
    points = [root]
    for j in range(1, 22):
        direction = F.normalize(joints[..., j, :] - joints[..., PARENTS[j], :], dim=-1)
        points.append(points[PARENTS[j]] + direction * lengths[..., j, None])
    return torch.stack(points, -2)


def native_fk(rotation, neutral, root):
    """Apply model local rotations to the source body's actual rest skeleton."""
    rotations, positions = [rotation[:, :, 0]], [root]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        rotations.append(rotations[parent] @ rotation[:, :, joint])
        offset = (neutral[:, joint]-neutral[:, parent])[:, None, :, None]
        positions.append(positions[parent]+(rotations[parent] @ offset).squeeze(-1))
    return torch.stack(positions, -2)


def kimodo_generate(data, root, out, steps, device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from kimodo.constraints import FullBodyConstraintSet, EndEffectorConstraintSet
    from kimodo.motion_rep.feature_utils import compute_heading_angle
    from kimodo.postprocess import post_process_motion

    checkpoint = root / 'Kimodo-SMPLX-RP-v1'
    config = OmegaConf.merge(OmegaConf.load(checkpoint / 'config.yaml'),
        dict(checkpoint_dir=str(checkpoint), text_encoder=None))
    model_config = OmegaConf.to_container(config, resolve=True)
    model_config.pop('checkpoint_dir')
    model = instantiate(model_config, device=device).eval()
    skeleton = model.skeleton
    native_lengths = bone_lengths(data['neutral_joints'])
    model_lengths = bone_lengths(skeleton.neutral_joints)
    scale = (model_lengths[5] + model_lengths[8]) / (native_lengths[:, 5] + native_lengths[:, 8])
    roots = data['root_positions'] * scale[:, None, None]
    global_rot, positions, _ = skeleton.fk(data['local_rot_mats'], roots)
    indices = torch.where(data['known'][0])[0]
    constraints = [[
        FullBodyConstraintSet(skeleton, indices, positions[i, indices], global_rot[i, indices]),
        EndEffectorConstraintSet(skeleton, indices, positions[i, indices], global_rot[i, indices],
            smooth_root_2d=roots[i, indices][:, [0, 2]],
            joint_names=['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot'])]
        for i in range(len(roots))]
    lengths = torch.full((len(roots),), roots.shape[1], device=device, dtype=torch.long)
    observed, mask = model.motion_rep.create_conditions_from_constraints_batched(
        constraints, lengths, to_normalize=True, device=device)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    encoded = model._generate(texts=None, max_frames=roots.shape[1], num_denoising_steps=steps,
        pad_mask=torch.ones(roots.shape[:2], device=device, dtype=torch.bool),
        first_heading_angle=compute_heading_angle(positions, skeleton)[:, 0],
        motion_mask=mask, observed_motion=observed, cfg_weight=2.0, cfg_type='regular',
        text_feat=torch.zeros(len(roots), 1, 4096, device=device),
        text_pad_mask=torch.zeros(len(roots), 1, dtype=torch.bool, device=device))
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - started
    raw = model.motion_rep.inverse(encoded, is_normalized=True, return_numpy=False)
    np.savez(out / 'raw_model.npz', **{k: v.detach().cpu().numpy() for k, v in raw.items()})
    started = time.perf_counter()
    cpu_constraints = [[constraint.to('cpu') for constraint in sample] for sample in constraints]
    corrected = post_process_motion(raw['local_rot_mats'].cpu(), raw['root_positions'].cpu(),
        raw['foot_contacts'].cpu(), copy.deepcopy(skeleton).cpu(), cpu_constraints, root_margin=.04)
    corrected = {k: v.to(device) for k, v in corrected.items()}
    torch.cuda.synchronize(device)
    postprocess_seconds = time.perf_counter() - started
    native_roots = corrected['root_positions'] / scale[:, None, None]
    native_positions = native_fk(corrected['local_rot_mats'], data['neutral_joints'], native_roots)
    # FK with the original body rotations supplies the reversible shape mapping.
    mapped_input = native_fk(data['local_rot_mats'], data['neutral_joints'], data['root_positions'])
    np.savez(out / 'prediction.npz', local_rot_mats=corrected['local_rot_mats'].detach().cpu().numpy(),
        root_positions=native_roots.detach().cpu().numpy(),
        target_joints=native_positions.detach().cpu().numpy(),
        input_model_joints=positions.detach().cpu().numpy(),
        input_model_root=roots.detach().cpu().numpy(), scale=scale.detach().cpu().numpy(),
        foot_contacts=raw['foot_contacts'].detach().cpu().numpy(), fps=np.array(30))
    metrics = dict(model='Kimodo-SMPLX-RP-v1', checkpoint=str(checkpoint / 'model.safetensors'),
        generation_seconds=generation_seconds, postprocess_seconds=postprocess_seconds,
        steps=steps, samples=len(roots), frames=roots.shape[1], seed=42, text_condition='official empty text',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        shape_roundtrip_max_m=float((mapped_input-data['joints']).norm(dim=-1).max()),
        raw_endpoint_position_max_m=float((raw['posed_joints'][:, indices]-positions[:, indices]).norm(dim=-1).max()),
        corrected_endpoint_position_max_m=float((corrected['posed_joints'][:, indices]-positions[:, indices]).norm(dim=-1).max()))
    return metrics


def humanml_features(positions):
    """Absolute-root HumanML3D features, using the released IK conventions on GPU."""
    from data_loaders.humanml.common.quaternion import qbetween, qinv, qmul, qrot, quaternion_to_cont6d
    from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain

    b, t, j, _ = positions.shape
    raw = positions.new_tensor(t2m_raw_offsets)
    across = F.normalize(positions[:, :, 1]-positions[:, :, 2] + positions[:, :, 17]-positions[:, :, 16], dim=-1)
    up = torch.zeros_like(across); up[..., 1] = 1
    forward = torch.cross(up, across, dim=-1)
    # scipy gaussian_filter1d(sigma=20, truncate=4, mode='nearest').
    distance = torch.arange(-80, 81, device=positions.device, dtype=positions.dtype)
    kernel = torch.exp(-.5*(distance/20)**2); kernel = kernel/kernel.sum()
    forward = F.conv1d(F.pad(forward.transpose(1, 2), (80, 80), mode='replicate'),
        kernel[None, None].expand(3, 1, -1), groups=3).transpose(1, 2)
    forward = F.normalize(forward, dim=-1)
    target = torch.zeros_like(forward); target[..., 2] = 1
    root_quat = qbetween(forward.reshape(-1, 3), target.reshape(-1, 3)).reshape(b, t, 4)
    root_quat[:, 0] = root_quat.new_tensor([1, 0, 0, 0])
    quats = torch.zeros(b, t, j, 4, device=positions.device); quats[..., 0] = 1
    quats[:, :, 0] = root_quat
    for chain in t2m_kinematic_chain:
        rotation = root_quat
        for parent, child in zip(chain[:-1], chain[1:]):
            direction = F.normalize(positions[:, :, child]-positions[:, :, parent], dim=-1)
            global_rotation = qbetween(raw[child].expand(b*t, -1), direction.reshape(-1, 3)).reshape(b, t, 4)
            quats[:, :, child] = qmul(qinv(rotation), global_rotation)
            rotation = qmul(rotation, quats[:, :, child])
    local = positions.clone()
    local[..., 0] -= positions[:, :, :1, 0]
    local[..., 2] -= positions[:, :, :1, 2]
    local = qrot(root_quat[:, :, None].expand(-1, -1, j, -1), local)
    velocity = torch.cat((positions[:, 1:]-positions[:, :-1], torch.zeros_like(positions[:, :1])), 1)
    local_velocity = qrot(root_quat[:, :, None].expand(-1, -1, j, -1), velocity)
    contacts = (velocity[:, :, [7, 10, 8, 11]].square().sum(-1) < .002).float()
    angle = torch.atan2(root_quat[..., 2:3], root_quat[..., :1])
    roots = torch.cat((angle, positions[:, :, 0, [0, 2, 1]]), -1)
    return torch.cat((roots, local[:, :, 1:].flatten(2), quaternion_to_cont6d(quats[:, :, 1:]).flatten(2),
        local_velocity.flatten(2), contacts), -1)


def condmdi_generate(data, root, out, device):
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils.parser_util import DataOptions, ModelOptions, DiffusionOptions, TrainingOptions
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    checkpoint = root / 'diffusion-motion-inbetweening/save/condmdi_random_joints'
    args_dict = {}
    for cls in (DataOptions, ModelOptions, DiffusionOptions, TrainingOptions):
        args_dict.update(vars(cls()))
    args_dict.update(json.loads((checkpoint / 'args.json').read_text()))
    args = SimpleNamespace(**args_dict)
    model, diffusion = create_model_and_diffusion(args, SimpleNamespace(dataset=SimpleNamespace()))
    load_saved_model(model, str(checkpoint / 'model000750000.pt'))
    model = model.to(device).eval().requires_grad_(False)
    # 20Hz timestamps on the common 0..2 second interval.
    time_indices = torch.arange(41, device=device)*1.5
    low, high = time_indices.floor().long(), time_indices.ceil().long()
    weight = (time_indices-low)[None, :, None, None]
    positions = data['joints'][:, low]*(1-weight) + data['joints'][:, high]*weight
    native_lengths = bone_lengths(data['neutral_joints'])
    reference = torch.from_numpy(np.load('dataset/000021.npy')).float().to(device).reshape(-1, 22, 3)[0]
    model_lengths = bone_lengths(reference)
    scale = (model_lengths[5]+model_lengths[8]) / (native_lengths[:, 5]+native_lengths[:, 8])
    roots = positions[:, :, 0]*scale[:, None, None]
    model_positions = retarget_positions(positions, model_lengths, roots)
    floor = model_positions[:, 0, :, 1].amin(-1)
    model_positions[..., 1] -= floor[:, None, None]
    features = humanml_features(model_positions)
    mean = torch.from_numpy(np.load('dataset/HumanML3D_abs/Mean.npy')).float().to(device)
    std = torch.from_numpy(np.load('dataset/HumanML3D_abs/Std.npy')).float().to(device)
    inputs = ((features-mean)/std).permute(0, 2, 1).unsqueeze(2)
    known = (time_indices <= 9) | (time_indices >= 51)
    mask = known[None, None, None].expand_as(inputs)
    kwargs = dict(obs_x0=inputs, obs_mask=mask, y=dict(
        text=['']*len(inputs), uncond=True,
        mask=torch.ones(len(inputs), 1, 1, 41, dtype=torch.bool, device=device),
        lengths=torch.full((len(inputs),), 41, device=device, dtype=torch.long),
        diffusion_steps=1000, imputate=True, stop_imputation_at=0,
        replacement_distribution='conditional', inpainted_motion=inputs,
        inpainting_mask=mask, reconstruction_guidance=False))
    reconstructed = recover_from_ric(features, 22, abs_3d=True)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        samples = diffusion.p_sample_loop(model, inputs.shape, clip_denoised=False,
            model_kwargs=kwargs, progress=True)
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter()-started
    features_out = samples[:, :, 0].permute(0, 2, 1)*std+mean
    joints = recover_from_ric(features_out, 22, abs_3d=True)
    np.savez(out / 'raw_model.npz', features=features_out.cpu().numpy(),
        joints=joints.cpu().numpy(), input_features=features.cpu().numpy(), known=known.cpu().numpy())
    joints[..., 1] += floor[:, None, None]
    native_roots = joints[:, :, 0]/scale[:, None, None]
    native_positions = retarget_positions(joints, native_lengths[:, None], native_roots)
    np.savez(out / 'prediction.npz', target_joints=native_positions.cpu().numpy(),
        root_positions=native_roots.cpu().numpy(), scale=scale.cpu().numpy(), floor=floor.cpu().numpy(), fps=np.array(20))
    return dict(model='CondMDI random joints', checkpoint=str(checkpoint / 'model000750000.pt'),
        generation_seconds=generation_seconds, postprocess_seconds=0., steps=1000,
        samples=len(inputs), frames=41, seed=42, text_condition='official uncond flag',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        representation_roundtrip_max_m=float((reconstructed-model_positions).norm(dim=-1).max()),
        endpoint_feature_max_error=float((features_out[:, known]-features[:, known]).abs().max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['kimodo', 'condmdi'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(4)
    external_commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    external_status = subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()
    args.output.mkdir(parents=True)
    data = {k: torch.from_numpy(v).to(args.device) for k, v in np.load(args.input).items()}
    function = kimodo_generate if args.model == 'kimodo' else condmdi_generate
    kwargs = dict(data=data, root=args.root, out=args.output, device=args.device)
    if args.model == 'kimodo':
        kwargs['steps'] = args.steps
    metrics = function(**kwargs)
    metrics.update(python=sys.executable, python_version=sys.version, torch_version=torch.__version__,
        cuda_version=torch.version.cuda, device=args.device, working_directory=str(Path.cwd()),
        external_git_commit=external_commit, external_git_status=external_status,
        external_git_commit_at_completion=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dependencies=sorted(f'{p.metadata["Name"]}=={p.version}' for p in importlib.metadata.distributions() if p.metadata.get('Name')))
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
