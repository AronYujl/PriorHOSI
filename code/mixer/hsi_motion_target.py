"""Full-condition HSI motion targets lifted into joint native surface editing."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .candidate_selection import _task_data, recover_score_context
from .input_views import KnownEmptyObjectView, masked_object_arguments
from .surface_edit import edit_envelope, yaw_matrix
from .continuation_outcomes import write_json


def root_target_loss(control,target,length,weight):
    scale = control.new_tensor((.05,.05,math.radians(10)))
    return weight*((control-target)/scale).square().sum()/(3*length)


def planar_yaw(rotation):
    """Closest world-Y rotation; inputs may also contain pitch and roll."""
    return torch.atan2(rotation[...,0,2]-rotation[...,2,0],
                       rotation[...,0,0]+rotation[...,2,2])


def root_heading_delta(prediction,source,dataset,mat):
    positions = dataset.denormalize_torch(prediction[...,:84]).reshape(*prediction.shape[:2],28,3)
    original = dataset.denormalize_torch(source[...,:84]).reshape(*source.shape[:2],28,3)
    rotation = mat[:,None,:3,:3]
    delta = (rotation@(positions[:,:,0]-original[:,:,0])[...,None]).squeeze(-1)
    predicted_rotation = transforms.rotation_6d_to_matrix(prediction[...,84:90])
    source_rotation = transforms.rotation_6d_to_matrix(source[...,84:90])
    relative = rotation@predicted_rotation@source_rotation.transpose(-1,-2)@rotation.transpose(-1,-2)
    return torch.stack((delta[...,0],delta[...,2],planar_yaw(relative)),-1)


def mean_targets(draws):
    result = draws.mean(0)
    result[...,2] = torch.atan2(draws[...,2].sin().mean(0),draws[...,2].cos().mean(0))
    return result


def native_target(coarse,scale=3):
    bounds = coarse.new_tensor((.2,.2,math.radians(20)))
    bounded = coarse.clamp(-bounds,bounds)
    indices = torch.arange(len(coarse)*scale,device=coarse.device)
    lower = indices//scale
    upper = (lower+1).clamp_max(len(coarse)-1)
    fraction = (indices%scale).to(coarse.dtype)/scale
    target = bounded[lower]*(1-fraction[:,None])+bounded[upper]*fraction[:,None]
    return target*edit_envelope(len(target),target.device)[:,None]


def rotated_scene_context(context,start):
    """One fixed world rotation about the task start for every scene query."""
    changed = dict(context)
    mat = context['mat'].clone()
    rotation = yaw_matrix(mat.new_tensor([math.pi/2]))[0]
    pivot = torch.as_tensor(start,device=mat.device,dtype=mat.dtype)
    mat[:,:3,:3] = rotation@mat[:,:3,:3]
    mat[:,:3,3] = pivot+(rotation@(mat[:,:3,3]-pivot)[...,None]).squeeze(-1)
    changed['mat'] = mat
    changed['obj_rot_mat_prefix'] = rotation@context['obj_rot_mat_prefix']
    if 'static_occ_cache' in changed:
        changed['static_occ_cache'] = {}
    return changed


def human_goal_context(context):
    """HSI uses its human-goal query while retaining the actual object geometry."""
    changed = dict(context)
    changed['is_object'] = torch.zeros_like(context['is_object'])
    if 'static_occ_cache' in changed:
        changed['static_occ_cache'] = {}
    return changed


def observation_outside(dataset,common,context,clean,anchor_frame):
    """Full native query-grid coverage, for the rotated-scene stress control."""
    from utils import transform_points
    positions = common[16].reshape(-1,2)
    centres = torch.zeros(len(positions)+1,1,3,device=positions.device)
    centres[0,0] = dataset.denormalize_torch(clean[...,:84]).reshape(1,16,28,3)[0,anchor_frame,0]
    centres[1:,0,0] = positions[:,0]
    centres[1:,0,2] = positions[:,1]
    mat = context['mat'].expand(len(centres),-1,-1).clone()
    world = transform_points(centres,mat)[:,0]
    mat[:,:3,3] = world;mat[:,1,3] = 0
    grid = dataset.create_meshgrid(batch_size=1).to(positions.device)
    points = transform_points(grid.expand(len(centres),-1,-1),mat)
    bounds = dataset.scene_grid_torch.to(points)
    return ((points<bounds[:3]) | (points>=bounds[3:6])).any(-1).float().mean().item()


class MotionTargetTeacher:
    def __init__(self,cfg,protocol):
        import hydra
        from omegaconf import OmegaConf
        from utils import init_model
        self.cfg,self.protocol = cfg,protocol
        self.device = torch.device(cfg.device)
        model_cfg = OmegaConf.merge(cfg.model.infbagel,dict(ckpt=str(cfg.hsi_ckpt_path)))
        self.model = init_model(model_cfg,device=self.device,eval=True).eval().requires_grad_(False)
        self.sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
        self.calls = 0

    def set_dataset(self,dataset):
        self.dataset = dataset
        self.sampler.dataset = dataset
        self.sampler.hsi_sampler.set_dataset_and_model(dataset,self.model)

    def recover_window(self,saved,snapshot,world,index,task,data,trajectory,points):
        from .scene_calibration import recover_temporal_window
        _,c,audit,geometry_context,offsets = recover_temporal_window(
            self.dataset,saved,snapshot,world,task,points,self.device)
        context,error = recover_score_context(self.sampler,self.cfg,saved,index,task,data,
                                               trajectory,geometry_context,offsets)
        return c['edited'],context,audit,error

    @torch.no_grad()
    def query_task(self,saved,task,ordinal,destination):
        from astar import get_path
        cfg,sampler,dataset = self.cfg,self.sampler,self.dataset
        data = _task_data(dataset,task)
        trajectory = get_path(np.asarray(task['start_location'])[[0,2]],
                              np.asarray(task['pelvis_goal'])[[0,2]],dataset)
        points = dataset.obj_rest_verts[task['object_name']]
        points = points[torch.linspace(0,len(points)-1,128,device=self.device).long()][None]
        length = saved['stitched']['points_world'].reshape(-1,28,3).shape[0]
        coarse = {name:torch.zeros(length,3,device=self.device) for name in ('correct','wrong')}
        covered = torch.zeros(length,device=self.device,dtype=torch.long)
        rows,predictions = [],[]
        forward_seconds = dict(correct=0.,wrong=0.)
        torch.cuda.synchronize(self.device);started = time.perf_counter()
        for index,(snapshot,world) in enumerate(zip(saved['corrections'],saved['windows'])):
            clean,context,audit,error = self.recover_window(saved,snapshot,world,index,task,data,trajectory,points)
            legacy_context = context
            context = human_goal_context(context)
            level = self.protocol['teacher']['level']
            timestep = torch.full((len(clean),),level,device=self.device,dtype=torch.long)
            contexts = dict(correct=context,wrong=rotated_scene_context(context,task['start_location']))
            common = {}
            scene_seed = 42+800000000+ordinal*1000+index
            for name,query_context in contexts.items():
                with torch.random.fork_rng(devices=[self.device.index]):
                    torch.set_rng_state(torch.Generator().manual_seed(scene_seed).get_state())
                    common[name] = masked_object_arguments(sampler._hsi_model_arguments(
                        clean,clean,timestep,query_context))
            for k in range(17):
                if k not in (0,15) and not torch.equal(common['correct'][k],common['wrong'][k]):
                    raise AssertionError('wrong-scene query changed non-scene argument '+str(k))
            with torch.random.fork_rng(devices=[self.device.index]):
                torch.set_rng_state(torch.Generator().manual_seed(scene_seed).get_state())
                legacy = masked_object_arguments(sampler._hsi_model_arguments(clean,clean,timestep,legacy_context))
            for k in range(17):
                if k not in (0,16) and not torch.equal(common['correct'][k],legacy[k]):
                    raise AssertionError('human-goal query changed a non-goal argument '+str(k))
            human_goal = common['correct'][3]
            expected_goal = human_goal/(human_goal.norm(dim=-1,keepdim=True)+1e-6)*.8
            goal_error = float((common['correct'][16][0]-expected_goal[:,[0,2]]).abs().max())
            if goal_error>1e-6:
                raise AssertionError('HSI scene goal query differs from native human goal')
            samples = {name:[] for name in common}
            raw = {name:[] for name in common}
            noisy_inputs = []
            for draw in range(self.protocol['teacher']['draws']):
                seed = 42+900000000+ordinal*10000+index*100+draw
                generator = torch.Generator(device=self.device).manual_seed(seed)
                noise = torch.randn(clean.shape,device=self.device,generator=generator)
                noisy = sampler.hsi_sampler.q_sample(clean,timestep,noise)
                noisy[:,:2] = clean[:,:2]
                empty = KnownEmptyObjectView();empty.begin_window(clean,seed)
                view = empty.for_step(noisy,level)
                noisy_inputs.append(view.cpu())
                for name,arguments in common.items():
                    torch.cuda.synchronize(self.device);forward_started=time.perf_counter()
                    prediction = self.model(view,*arguments,is_sample=True)
                    torch.cuda.synchronize(self.device)
                    forward_seconds[name] += time.perf_counter()-forward_started
                    self.calls += 1
                    if not torch.isfinite(prediction[...,:216]).all():
                        raise FloatingPointError('nonfinite HSI motion target')
                    delta = root_heading_delta(prediction,clean,dataset,context['mat'])[0]
                    samples[name].append(delta)
                    raw[name].append(prediction.cpu())
            indices = torch.arange(2,16,device=self.device)+14*index
            covered[indices] += 1
            for name in common:
                coarse[name][indices] = mean_targets(torch.stack(samples[name]))[2:]
            difference = coarse['correct'][indices]-coarse['wrong'][indices]
            row = dict(window=index,frames=indices.cpu().tolist(),source_recovery=audit,
                context_world_error_m=error,scene_seed=scene_seed,level=level,
                goal_query=dict(native_human_max_error_m=goal_error,
                    legacy_human_gap_m=float((legacy[16][0]-expected_goal[:,[0,2]]).norm()),
                    occupancy_changed_fraction=float((common['correct'][0]!=legacy[0]).float().mean()),
                    other_arguments_exact=True,dynamic_object_observations_exact=True),
                root_scene_difference_rms_m=float(difference[:,:2].square().mean().sqrt()),
                heading_scene_difference_mean_deg=float(difference[:,2].abs().mean()*180/math.pi),
                observations_changed_fraction={str(k):float((common['correct'][k]!=common['wrong'][k]).float().mean())
                                              for k in (0,15)},
                outside_fraction={name:observation_outside(dataset,arguments,contexts[name],clean,sampler.hsi_sampler.emb_f)
                                  for name,arguments in common.items()},
                raw_root_rms_m={name:float(torch.stack(value)[:,2:,:2].square().mean().sqrt()) for name,value in samples.items()},
                draw_root_difference_rms_m={name:float((value[0][2:,:2]-value[1][2:,:2]).square().mean().sqrt())
                                           for name,value in samples.items()})
            rows.append(row)
            predictions.append(dict(window=index,source=clean.cpu(),mat=context['mat'].cpu(),
                inputs=noisy_inputs,predictions=raw,
                arguments={name:[v.cpu() for v in args] for name,args in common.items()},
                targets={name:torch.stack(value).cpu() for name,value in samples.items()}))
        if not torch.equal(covered[:2],torch.zeros_like(covered[:2])) or not (covered[2:]==1).all():
            raise AssertionError('HSI target frame ownership differs from native source')
        targets = {name:native_target(value) for name,value in coarse.items()}
        torch.cuda.synchronize(self.device)
        metadata = dict(task=ordinal,windows=len(rows),hsi_calls=len(rows)*4,
            seconds=time.perf_counter()-started,forward_seconds=forward_seconds,
            checkpoint=str(cfg.hsi_ckpt_path),queries=rows,
            target_bound_fraction={name:float((value.abs()>value.new_tensor((.2,.2,math.radians(20)))).float().mean())
                                   for name,value in coarse.items()},
            target_rms_m={name:float(value[:,:2].square().mean().sqrt()) for name,value in targets.items()},
            target_mean_heading_deg={name:float(value[:,2].abs().mean()*180/math.pi) for name,value in targets.items()})
        write_json(destination/'teacher.json',metadata)
        with (destination/'teacher.pt').open('xb') as handle:
            torch.save(dict(coarse={k:v.cpu() for k,v in coarse.items()},
                            native={k:v.cpu() for k,v in targets.items()},windows=predictions),handle)
        print(json.dumps(dict(task=ordinal,hsi_teacher=metadata['target_rms_m'],hsi_seconds=metadata['seconds'])),flush=True)
        return targets


def target_alignment(source,result,targets):
    delta = result['joints'][:,0]-source['joints'][:,0]
    rotation = transforms.axis_angle_to_matrix(result['pose'][:,0])
    original = transforms.axis_angle_to_matrix(source['pose'][:,0])
    current = torch.stack((delta[:,0],delta[:,2],planar_yaw(rotation@original.transpose(-1,-2))),-1)
    return dict(root_change_rms_m=float(current[:,:2].square().mean().sqrt()),
        heading_change_mean_deg=float(current[:,2].abs().mean()*180/math.pi),
        distance={name:float(root_target_loss(current,target,len(current),1.)) for name,target in targets.items()})


def summarize_motion_targets(run_root,task_manifest,device='cuda:7'):
    from .scene_calibration import paired_local_metrics
    root = Path(run_root)
    expected = {t['canonical_ordinal'] for t in json.loads(Path(task_manifest).read_text())['tasks']}
    records = [json.loads(p.read_text()) for p in sorted(root.glob('lanes/*/task-*/complete.json'))]
    if len(records)!=len(expected) or {r['task'] for r in records}!=expected:
        raise ValueError('incomplete HSI target campaign')
    arms = list(records[0]['arms'])
    tasks = {a:{str(r['task']):{k:float(v) for k,v in r['arms'][a]['metrics'].items()} for r in records} for a in arms}
    def average(rows):
        return {k:sum(v[k] for v in rows)/len(rows) for k in rows[0]}
    names = {str(r['task']):r['scene'] for r in records}
    means = {a:average(list(rows.values())) for a,rows in tasks.items()}
    scenes = {a:{s:average([v for k,v in rows.items() if names[k]==s]) for s in sorted(set(names.values()))}
              for a,rows in tasks.items()}
    baseline,correct,wrong,independent = 'relation_20','relation_hsi_20','relation_wrong_20','independent_hsi_20'
    pairs = [(baseline,correct),(wrong,correct),(independent,correct),('source',baseline)]
    pairs += [(a,a+'_terminal') for a in (baseline,correct,wrong,independent)]
    pairs += [(a+'_terminal',correct+'_terminal') for a in (baseline,wrong,independent)]
    paired = {u:{b+'-minus-'+a:paired_local_metrics(rows[a],rows[b],device) for a,b in pairs}
              for u,rows in (('task',tasks),('scene',scenes))}
    b,c,w = means[baseline],means[correct],means[wrong]
    hs,oskey = 'scene_human_penetration_s_mean','scene_obj_penetration_s_mean'
    gates = dict(extra_HS=c[hs]<=.99*b[hs],scene_dependence=c[hs]<=w[hs]-.005*b[hs],
        OS=c[oskey]<=1.01*b[oskey],FS=c['foot_sliding']<=b['foot_sliding']+.01,
        contact=c['contact_percent']>=b['contact_percent']-.002,completion=c['completed']>=b['completed'])
    audit = [r['arms'][correct]['audit'] for r in records]
    gates['relation'] = max(a['hand_object_max_error_m'] for a in audit)<=1e-4
    for k in ('initial_max_error_m','final_max_error_m','initial_object_max_error_m','final_object_max_error_m'):
        gates[k] = max(a[k] for a in audit)<=1e-5
    teachers = [json.loads(p.read_text()) for p in sorted(root.glob('lanes/*/task-*/teacher.json'))]
    result = dict(task_count=len(records),scene_count=len(set(names.values())),means=means,paired=paired,
        full469_entry=all(gates.values()),gates=gates,
        changed_tasks={a:sum(r['arms'][a]['audit']['mean_joint_change_mm']>0 for r in records) for a in arms},
        terminal_recovered={a:sum(not r['arms'][a]['metrics']['completed'] and
            r['arms'][a+'_terminal']['metrics']['completed'] for r in records) for a in (baseline,correct,wrong,independent)},
        teacher_calls=sum(r['hsi_calls'] for r in teachers),teacher_seconds=sum(r['seconds'] for r in teachers),
        baseline_recovery_max_error=max(r['arms'][baseline]['baseline_metric_error'] for r in records),
        test_set_development=True,training_allowed=False)
    out = root/'analysis';out.mkdir()
    for name,value in [('tasks',tasks),('scenes',scenes),('means',means),('paired',paired),('summary',result)]:
        write_json(out/(name+'.json'),value)
    return result


def encode_native_window(dataset,points,global_rotation,object_translation,object_rotation,contact,mat,reference):
    """Encode current world geometry in the native window and object frames."""
    rotation = mat[:,None,:3,:3].transpose(-1,-2)
    local_points = (rotation[:,:,None]@(points-mat[:,None,None,:3,3])[...,None]).squeeze(-1)
    local_object = (rotation@(object_translation-mat[:,None,:3,3])[...,None]).squeeze(-1)
    human_rotation = rotation[:,:,None]@global_rotation
    relative_object = object_rotation@reference[:,None].transpose(-1,-2)
    return torch.cat((dataset.normalize_torch(local_points).flatten(2),
        transforms.matrix_to_rotation_6d(human_rotation).flatten(2),
        dataset.normalize_torch(local_object,is_object=True),relative_object.flatten(2),contact),-1)


def lift_native_prediction(base,reference,prediction):
    """Transfer teacher differences, cancelling the base's coarse roundtrip."""
    original = transforms.axis_angle_to_matrix(base['pose'].double())
    clean = transforms.axis_angle_to_matrix(reference['pose'].double())
    predicted = transforms.axis_angle_to_matrix(prediction['pose'].double())
    delta = transforms.axis_angle_to_matrix(transforms.matrix_to_axis_angle(clean.transpose(-1,-2)@predicted))
    target = original@delta
    root_delta = transforms.axis_angle_to_matrix(transforms.matrix_to_axis_angle(predicted[:,0]@clean[:,0].transpose(-1,-2)))
    target[:,0] = root_delta@original[:,0]
    pose = base['pose']+(transforms.matrix_to_axis_angle(target)-transforms.matrix_to_axis_angle(original)).to(base['pose'])
    translation = base['translation']+(prediction['translation']-reference['translation'])
    return pose,translation


def native_coarse_pose(global_rot_6d,dataset):
    """Decode the encoded samples themselves, before native time interpolation."""
    rotation=transforms.rotation_6d_to_matrix(global_rot_6d.double().reshape(-1,22,6))
    local=dataset.quat_ik_torch(rotation)
    return transforms.matrix_to_axis_angle(local).to(global_rot_6d)


class CurrentMotionTeacher(MotionTargetTeacher):
    """Fresh full HSI targets conditioned on the current native geometry."""
    def recover_window(self,saved,snapshot,world,index,task,data,trajectory,points):
        return self.current_windows[index]

    @torch.no_grad()
    def query_current_task(self,saved,task,ordinal,destination,source,projector,smpl_cache):
        from astar import get_path
        from test_infbagel_hosi import get_mat,decode_sample_window
        from .continuation_outcomes import native_tracks
        from .surface_edit import decode_body
        dataset,cfg = self.dataset,self.cfg
        data = _task_data(dataset,task)
        trajectory = get_path(np.asarray(task['start_location'])[[0,2]],np.asarray(task['pelvis_goal'])[[0,2]],dataset)
        seq = dataset.ori_sequence_idx[task['data_idx']]
        dataset_translation = torch.as_tensor(dataset.transl[seq],device=self.device,dtype=source['pose'].dtype)
        root_bias = projector.offsets[0]+dataset_translation
        positions = source['joints'][::3]-root_bias
        global_rotation,_ = projector.forward(projector.rotation,source['translation'])
        global_rotation = global_rotation[::3]
        object_translation = source['object_translation'][::3]
        object_rotation = source['object_rotation'][::3]
        start = int(dataset.start_ind[task['data_idx']])
        reference = torch.as_tensor(dataset.object_rot_mat[start],device=self.device,dtype=source['pose'].dtype)[None]
        prefix = torch.eye(3,device=self.device)[None]
        offsets = torch.as_tensor(dataset.rest_human_offsets[seq],device=self.device,dtype=source['pose'].dtype)
        corrected = dict(saved,corrections=[],windows=[])
        geometry_contexts=[];encoding_audits=[]
        for index,snapshot in enumerate(saved['corrections']):
            lo,hi = index*14,index*14+16
            points = positions[None,lo:hi]
            mat = get_mat(cfg,points.flatten(2),0)
            clean = encode_native_window(dataset,points,global_rotation[None,lo:hi],
                object_translation[None,lo:hi],object_rotation[None,lo:hi],
                snapshot['edited'].to(self.device)[...,228:232],mat,reference)
            decoded = decode_sample_window(cfg,clean,dataset,mat)
            world = dict(points_world=decoded['points_orig'],global_rot_6d=decoded['global_rot_6d'],
                object_translation_world=decoded['obj_trans_orig'],
                object_rotation_world=(decoded['object_rot_mat'].reshape(1,16,3,3)@reference[:,None]).reshape(1,16,9))
            errors = dict(position_world_max_error_m=float((world['points_world'].reshape(1,16,28,3)-points).abs().max()),
                rotation_world_max_error=float((transforms.rotation_6d_to_matrix(world['global_rot_6d'].reshape(1,16,22,6))-global_rotation[None,lo:hi]).abs().max()),
                object_world_max_error_m=float((world['object_translation_world']-object_translation[None,lo:hi]).abs().max()),
                object_rotation_world_max_error=float((world['object_rotation_world'].reshape(1,16,3,3)-object_rotation[None,lo:hi]).abs().max()))
            assert max(errors.values())<=1e-5,errors
            old = saved['windows'][index]['points_world'].to(self.device).reshape(1,16,28,3)
            errors['world_position_difference_from_legacy_cm'] = float((points-old).norm(dim=-1).mean()*100)
            encoding_audits.append(errors)
            corrected['corrections'].append(dict(edited=clean))
            corrected['windows'].append(world)
            geometry_contexts.append(dict(mat=mat,obj_rot_mat_ref=reference,obj_rot_mat_prefix=prefix))
        self.current_windows=[]
        for index in range(len(corrected['windows'])):
            context,error = recover_score_context(self.sampler,cfg,corrected,index,task,data,trajectory,geometry_contexts[index],offsets)
            self.current_windows.append((corrected['corrections'][index]['edited'],context,encoding_audits[index],error))
        self.query_task(corrected,task,ordinal,destination)
        teacher = torch.load(destination/'teacher.pt',map_location=self.device,weights_only=False)
        world = dict(points_world=positions.cpu(),
            global_rot_6d=transforms.matrix_to_rotation_6d(global_rotation).cpu(),
            object_translation_world=object_translation.cpu(),object_rotation_world=object_rotation.cpu())
        # Decode actual encoded channels, including their finite-precision roundtrip.
        for index,w in enumerate(corrected['windows']):
            keep=slice(0,16) if index==0 else slice(2,16)
            dest=slice(0,16) if index==0 else slice(index*14+2,index*14+16)
            world['points_world'][dest]=w['points_world'].reshape(-1,28,3)[keep].cpu()
            world['global_rot_6d'][dest]=w['global_rot_6d'].reshape(-1,22,6)[keep].cpu()
        model=smpl_cache[source['gender']]
        coarse_motion=dict(source,pose=native_coarse_pose(world['global_rot_6d'],dataset).to(self.device),
                           translation=world['points_world'][:,0].to(self.device)+dataset_translation)
        _,coarse_joints=decode_body(coarse_motion,model)
        encoded_coarse_error=float((coarse_joints-source['joints'][::3]).abs().max())
        assert encoded_coarse_error<=1e-4,encoded_coarse_error
        roundtrip = native_tracks(cfg,dataset,world,task,True,smpl_cache,body_parameters=True)
        coarse_error = float((roundtrip['joints'][::3]-source['joints'][::3]).abs().max())
        pose,translation = lift_native_prediction(source,roundtrip,roundtrip)
        identity_error = max(float((pose-source['pose']).abs().max()),float((translation-source['translation']).abs().max()))
        assert identity_error<=1e-6,identity_error
        cache=dict(source=dict(joints=source['joints']))
        lifted_states={};raw_states={}
        for view in ('correct','wrong'):
            for draw in range(2):
                predicted_world={k:v.clone() for k,v in world.items()}
                for w in teacher['windows']:
                    index=w['window'];dest=slice(index*14+2,index*14+16)
                    decoded=decode_sample_window(cfg,w['predictions'][view][draw],dataset,w['mat'])
                    predicted_world['points_world'][dest]=decoded['points_orig'].reshape(-1,28,3)[2:].cpu()
                    predicted_world['global_rot_6d'][dest]=decoded['global_rot_6d'].reshape(-1,22,6)[2:].cpu()
                predicted=native_tracks(cfg,dataset,predicted_world,task,True,smpl_cache,body_parameters=True)
                pose,translation=lift_native_prediction(source,roundtrip,predicted)
                lifted=dict(source,pose=pose,translation=translation)
                lifted['verts'],lifted['joints']=decode_body(lifted,model)
                key=f'{view}_full_draw{draw}'
                cache[key]=lifted
                lifted_states[key]={k:v.cpu() for k,v in lifted.items() if torch.is_tensor(v) and k!='verts'}
                raw_states[key]={k:v.cpu() for k,v in predicted.items() if torch.is_tensor(v) and k!='verts'}
        audit=dict(hsi_calls=len(corrected['windows'])*4,root_reconstruction_bias_m=root_bias.tolist(),
            encoded_windows=encoding_audits,encoded_coarse_body_max_error_m=encoded_coarse_error,
            native_coarse_body_max_error_m=coarse_error,
            native_all_frame_body_roundtrip_max_error_m=float((roundtrip['joints']-source['joints']).abs().max()),
            native_all_frame_body_roundtrip_mean_error_cm=float((roundtrip['joints']-source['joints']).norm(dim=-1).mean()*100),
            identity_lift_max_error=identity_error)
        write_json(destination/'current_motion_query.json',audit)
        with (destination/'teacher_body.pt').open('xb') as handle:
            torch.save(dict(roundtrip={k:v.cpu() for k,v in roundtrip.items() if torch.is_tensor(v) and k!='verts'},
                            raw_predictions=raw_states,lifted_targets=lifted_states),handle)
        return cache,audit
