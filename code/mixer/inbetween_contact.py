"""Contact correction on the final native body with immutable bridge contexts."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .continuation_outcomes import write_json
from .inbetween import motion_metrics
from .standing_transition import scene_measures
from .surface_edit import decode_body, load_object_sdf


def contact_intervals(contacts, minimum_frames):
    """Keep complete predicted intervals; never infer contact from edited speed."""
    mask = contacts.clone().bool()
    intervals = []
    for foot in range(mask.shape[1]):
        padded = torch.cat((mask.new_zeros(1), mask[:, foot], mask.new_zeros(1))).int()
        changes = padded[1:] - padded[:-1]
        for start, stop in zip(torch.where(changes == 1)[0].tolist(), torch.where(changes == -1)[0].tolist()):
            retained = stop-start >= minimum_frames
            intervals.append(dict(foot=foot, start=start, stop=stop, retained=retained))
            if not retained:
                mask[start:stop, foot] = False
    return mask, intervals


def fixed_context_motion(source, free_pose, free_translation):
    return dict(source, pose=torch.cat((source['pose'][:10], free_pose, source['pose'][51:])),
                translation=torch.cat((source['translation'][:10], free_translation, source['translation'][51:])))


def surface_patches(model, target_vertices, count):
    dominant = model.lbs_weights.argmax(-1)
    patches = []
    for joint in (7, 10, 8, 11):
        region = torch.where(dominant == joint)[0]
        patches.append(region[target_vertices[region, 1].argsort()[:count]])
    return torch.stack(patches)


def foot_points(vertices, patches):
    points = vertices[:, patches]
    return points.mean(2)[..., [0, 2]], points[..., 1].amin(-1)


def differentiable_body(motion, model):
    from utils import run_smplx_model, SMPLX_JOINTS_28
    return run_smplx_model(motion['pose'], motion['translation'], motion['betas'], motion['gender'],
        joints_ind=SMPLX_JOINTS_28, smpl_model=model)


def scene_distances(vertices, sdf, info):
    extent = float(max(info['extents']))
    query = (vertices-vertices.new_tensor(info['centroid']))/(extent/2)
    distances = torch.nn.functional.grid_sample(sdf, query[..., [2, 1, 0]].reshape(1, -1, 1, 1, 3),
        padding_mode='border', align_corners=True).reshape(vertices.shape[:2])*(extent/2)
    return distances


def object_distances(vertices, motion, sdf, info):
    from eval_metrics import compute_signed_distances
    relative = (motion['object_rotation'][10:51].transpose(-1, -2) @
        (vertices-motion['object_translation'][10:51, None]).transpose(-1, -2)).transpose(-1, -2)
    return compute_signed_distances(sdf, relative.new_tensor(info['centroid'])[None],
        relative.new_tensor(info['extents'])[None], relative)


def contact_measures(vertices, patches, mask):
    horizontal, height = foot_points(vertices, patches)
    pairs = mask[9:51] & mask[10:52]
    speed = (horizontal[10:52]-horizontal[9:51]).norm(dim=-1)*30
    selected = mask[10:51]
    return dict(fixed_contact_speed_m_s=float(speed[pairs].mean()) if pairs.any() else None,
        fixed_contact_height_abs_m=float(height[10:51][selected].abs().mean()) if selected.any() else None,
        fixed_contact_pairs=int(pairs.sum()), fixed_contact_points=int(selected.sum()),
        free_ground_depth_max_m=float((-vertices[10:51, :, 1].amin()).clamp_min(0)),
        free_ground_depth_mean_m=float((-vertices[10:51, :, 1].amin(-1)).clamp_min(0).mean()))


def rotation_continuity_terms(pose, reference_rotations):
    """Constrain angular motion as well as joint positions on SO(3)."""
    rotations = transforms.axis_angle_to_matrix(pose)
    difference = transforms.matrix_to_axis_angle(
        rotations[10:51] @ reference_rotations[10:51].transpose(-1, -2))
    steps = transforms.matrix_to_axis_angle(rotations[1:] @ rotations[:-1].transpose(-1, -2))
    return dict(rotation=.1*(difference/.15).square().mean(),
        angular_acceleration=((steps[1:]-steps[:-1])/(3*np.pi/180)).square().mean(),
        angular_seam=((steps[[9, 50]]-steps[[8, 51]])/(np.pi/180)).square().mean())


def correct_motion(source, model, patches, mask, sdf, info, object_sdf, object_info, cfg, dest):
    """One fixed-budget optimization; evaluate the final iterate."""
    settings = cfg.inbetween.contact
    with torch.no_grad():
        reference_vertices, reference_joints = decode_body(source, model)
        reference_rotations = transforms.axis_angle_to_matrix(source['pose'])
    pose = source['pose'][10:51].clone().requires_grad_(True)
    translation = source['translation'][10:51].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pose, translation], lr=settings.learning_rate)
    reference_acceleration = reference_joints[2:]-2*reference_joints[1:-1]+reference_joints[:-2]
    pairs = mask[9:51] & mask[10:52]
    selected = mask[10:51]
    pair_count = int(pairs.sum())
    point_count = int(selected.sum())
    trace = []
    torch.cuda.reset_peak_memory_stats(source['pose'].device)
    torch.cuda.synchronize(source['pose'].device)
    started = time.perf_counter()
    for iteration in range(settings.iterations):
        motion = fixed_context_motion(source, pose, translation)
        vertices, joints = differentiable_body(motion, model)
        acceleration = joints[2:]-2*joints[1:-1]+joints[:-2]
        horizontal, height = foot_points(vertices, patches)
        free = vertices[10:51]
        floor_depth = (-free[..., 1].amin(-1)).clamp_min(0)
        scene_depth = (-scene_distances(free, sdf, info).amin(-1)-.005).clamp_min(0)
        object_depth = (-object_distances(free, source, object_sdf, object_info).amin(-1)-.005).clamp_min(0)
        terms = dict(
            joint=((joints[10:51]-reference_joints[10:51])/.05).square().mean(),
            acceleration=((acceleration-reference_acceleration)/.002).square().mean(),
            seam=5*(acceleration[[8, 50]]/.001).square().mean(),
            floor=20*(floor_depth/.002).square().mean(),
            scene=2*(scene_depth/.005).square().mean(),
            object=2*(object_depth/.005).square().mean(),
            contact_height=2*((height[10:51]/.01).square()*selected).sum()/point_count if point_count else height.sum()*0,
            contact_velocity=5*(((horizontal[10:52]-horizontal[9:51])/.001).square().sum(-1)*pairs).sum()/(2*pair_count)
                if pair_count else horizontal.sum()*0)
        terms.update(rotation_continuity_terms(motion['pose'], reference_rotations))
        loss = sum(terms.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trace.append(dict(iteration=iteration+1, loss=float(loss.detach()),
            **{name: float(value.detach()) for name, value in terms.items()}))
        if iteration == 0 or (iteration+1) % 100 == 0:
            print(json.dumps(dict(task=dest.name, **trace[-1])), flush=True)
    with torch.no_grad():
        result = fixed_context_motion(source, pose.detach(), translation.detach())
        vertices, result['joints'] = decode_body(result, model)
    torch.cuda.synchronize(source['pose'].device)
    audit = dict(iterations=settings.iterations, seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(source['pose'].device),
        joint_correction_mean_m=float((result['joints']-reference_joints).norm(dim=-1).mean()),
        joint_correction_max_m=float((result['joints']-reference_joints).norm(dim=-1).max()),
        root_translation_correction_max_m=float((result['translation']-source['translation']).norm(dim=-1).max()),
        rotation_correction_max_deg=float(transforms.matrix_to_axis_angle(
            transforms.axis_angle_to_matrix(result['pose']) @ transforms.axis_angle_to_matrix(source['pose']).transpose(-1, -2)
        ).norm(dim=-1).max()*180/np.pi))
    write_json(dest/'optimization.json', dict(audit=audit, trace=trace))
    return result, vertices, audit


def export_motion(dest, motion, source_full, row, status):
    cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in motion.items()}
    torch.save(dict(motion=cpu, source=row, model='kimodo_contact', status=status), dest/'motion.pt')
    arrays = {k: cpu[k].numpy() for k in ('pose', 'translation', 'joints', 'betas', 'object_translation', 'object_rotation')}
    arrays.update(gender=np.array(row['gender']), fps=np.array(30))
    np.savez(dest/'motion.npz', **arrays)
    full = dict(np.load(source_full))
    for key in ('pose', 'translation', 'joints', 'object_translation', 'object_rotation'):
        full[key] = np.concatenate((full[key][:-51], arrays[key][10:]))
    np.savez(dest/'full_motion.npz', **full)
    return dict(total_frames=len(full['pose']), source_frames=len(full['pose'])-51, new_frames=51,
        transition_start_frame=int(full['transition_start_frame']))


def run_contact(cfg, root):
    import trimesh
    from utils import create_smplx_model, zup_to_yup
    from .scene_calibration import paired_local_metrics
    output, inputs = Path(cfg.hosi_output_dir), Path(cfg.inbetween.input_dir)
    baseline = Path(cfg.inbetween.contact.baseline)
    index = json.loads((inputs/'index.json').read_text())
    prediction = np.load(baseline/'kimodo/prediction.npz')
    (output/'kimodo').symlink_to(baseline/'kimodo', target_is_directory=True)
    (output/'kimodo_contact').mkdir()
    models, records = {}, []
    for ordinal, row in enumerate(index['tasks']):
        dest = output/'kimodo_contact'/f'task-{row["task"]:03d}'
        dest.mkdir()
        source = torch.load(baseline/'kimodo'/dest.name/'motion.pt', map_location=cfg.device, weights_only=False)['motion']
        condition = torch.load(inputs/dest.name/'condition.pt', map_location=cfg.device, weights_only=False)['motion']
        if row['gender'] not in models:
            models[row['gender']] = create_smplx_model(row['gender'], torch.device(cfg.device)).eval().requires_grad_(False)
        model = models[row['gender']]
        sdf_root = root/'data/hosi_test/Scene_sdf'
        sdf = torch.from_numpy(np.load(sdf_root/(row['scene']+'_sdf.npy'))).float().to(cfg.device)[None, None]
        info = json.loads((sdf_root/(row['scene']+'_sdf_info.json')).read_text())
        object_sdf, object_info = load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files', row['object'])
        object_sdf = torch.from_numpy(object_sdf).float().to(cfg.device)[None]
        object_vertices = torch.as_tensor(zup_to_yup(np.asarray(trimesh.load_mesh(
            root/'data/test/rest_object_geo'/(row['object']+'.ply')).vertices)), device=cfg.device, dtype=torch.float32)
        with torch.no_grad():
            vertices, source['joints'] = decode_body(source, model)
            patches = surface_patches(model, vertices[51], cfg.inbetween.contact.patch_vertices)
            mask, intervals = contact_intervals(torch.as_tensor(prediction['foot_contacts'][ordinal], device=cfg.device),
                cfg.inbetween.contact.minimum_contact_frames)
            points, _ = foot_points(vertices, patches)
            for interval in intervals:
                touches_both = interval['start'] <= 9 and interval['stop'] > 51
                interval['spans_both_boundaries'] = touches_both
                interval['endpoint_horizontal_difference_m'] = float((points[51, interval['foot']]-points[9, interval['foot']]).norm()) if touches_both else None
            endpoints = {label: scene_measures(vertices[frame:frame+1], sdf, info) for label, frame in [('source', 9), ('target', 51)]}
            feasible = all(v['scene_penetration_mean_m'] <= .005 and v['scene_penetration_max_m'] <= .05
                and v['scene_outside_fraction'] == 0 for v in endpoints.values())
            before = motion_metrics(source, vertices, condition, cfg, object_vertices, sdf, info, object_sdf, object_info)
            before.update(contact_measures(vertices, patches, mask))
        write_json(dest/'contacts.json', dict(patch_vertices=patches.tolist(), fixed_mask=mask.tolist(), intervals=intervals))
        status = 'corrected' if feasible else 'infeasible_endpoint'
        if feasible:
            corrected, corrected_vertices, audit = correct_motion(source, model, patches, mask, sdf, info,
                object_sdf, object_info, cfg, dest)
        else:
            corrected, corrected_vertices, audit = source, vertices, dict(iterations=0, seconds=0.)
        with torch.no_grad():
            after = motion_metrics(corrected, corrected_vertices, condition, cfg, object_vertices, sdf, info, object_sdf, object_info)
            after.update(contact_measures(corrected_vertices, patches, mask))
            audit['object_translation_max_change_m'] = float((corrected['object_translation']-source['object_translation']).abs().max())
            audit['object_rotation_max_change'] = float((corrected['object_rotation']-source['object_rotation']).abs().max())
        assembly = export_motion(dest, corrected, baseline/'kimodo'/dest.name/'full_motion.npz', row, status)
        record = dict(task=row['task'], scene=row['scene'], object=row['object'], status=status,
            endpoints=endpoints, before=before, after=after, audit=audit, assembly=assembly)
        write_json(dest/'metrics.json', record)
        records.append(record)
        print(json.dumps(record), flush=True)
    statistics = output/'analysis'
    statistics.mkdir()
    summaries, comparisons = {}, {}
    for subset in ('all', 'feasible'):
        selected = [r for r in records if subset == 'all' or r['status'] == 'corrected']
        paired = {}
        summaries[subset] = {}
        for arm in ('before', 'after'):
            values = {f'task-{r["task"]:03d}': {k: float(v) for k, v in r[arm].items() if isinstance(v, (int, float))}
                for r in selected}
            write_json(statistics/f'{subset}_{arm}_per_sequence_metrics.json', dict(metrics=values, sequence_count=len(values)))
            summaries[subset][arm] = {k: float(np.mean([v[k] for v in values.values()])) for k in next(iter(values.values()))}
            paired[arm] = values
        comparisons[subset] = paired_local_metrics(paired['before'], paired['after'], cfg.device)
    first, second = summaries['feasible']['before'], summaries['feasible']['after']
    gates = dict(ground_reduced=second['free_ground_depth_max_m'] < first['free_ground_depth_max_m'],
        contact_speed_reduced=second['fixed_contact_speed_m_s'] < first['fixed_contact_speed_m_s'],
        scene_pass_preserved=second['geometry_success'] >= first['geometry_success'],
        entry_seam_preserved=second['entry_velocity_jump_mean_m_s'] <= first['entry_velocity_jump_mean_m_s']+.02,
        exit_seam_preserved=second['exit_velocity_jump_mean_m_s'] <= first['exit_velocity_jump_mean_m_s']+.02,
        contexts_exact=all(r['after']['endpoint_joint_max_error_m'] < 1e-5 for r in records),
        finite=all(r['after']['finite'] for r in records))
    write_json(output/'metrics.json', dict(records=records, means=summaries, paired=comparisons, gates=gates,
        correction_gate=all(gates.values()), selected_generator='kimodo', seed=42,
        corrected=sum(r['status'] == 'corrected' for r in records), rejected=sum(r['status'] == 'infeasible_endpoint' for r in records)))
