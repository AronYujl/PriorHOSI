"""Complete native-motion editing with shared rigid manipulation transforms."""
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .continuation_outcomes import native_tracks, write_json


LEGS = (1, 2, 4, 5, 7, 8)
ARMS = (16, 17, 18, 19, 20, 21)
FEET = (7, 8, 10, 11)
HANDS = (24, 26)


def load_object_sdf(directory,name):
    """Use the native evaluator's object key for the author's multi-suffix assets."""
    paths = {path.name.split('.')[0]:path for path in Path(directory).glob('*.npy')}
    path = paths[name]
    return np.load(path),json.loads(path.with_suffix('.json').read_text())


def smoothstep(u):
    u = u.clamp(0, 1)
    return u**3 * (10 + u * (-15 + 6*u))


def spline_basis(length, device, spacing=15):
    """Uniform cubic B-spline; all rows partition unity, including the endpoints."""
    time_index = torch.arange(length, device=device, dtype=torch.float32)/spacing
    cell = time_index.floor().long()
    u = time_index-cell
    weights = torch.stack(((1-u)**3, 3*u**3-6*u**2+4,
                           -3*u**3+3*u**2+3*u+1, u**3), -1)/6
    basis = torch.zeros(length, int(cell[-1])+4, device=device)
    basis.scatter_(1, cell[:, None]+torch.arange(4, device=device), weights)
    return basis


def edit_envelope(length, device):
    t = torch.arange(length, device=device, dtype=torch.float32)
    return smoothstep((t-5)/15)*smoothstep((length-3-t)/15)


def yaw_matrix(yaw):
    c, s = yaw.cos(), yaw.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c,z,s,z,o,z,-s,z,c), -1).reshape(-1,3,3)


def shared_object_transform(pivot, translation, rotation, shift, yaw):
    r = yaw_matrix(yaw)
    return (pivot+(r@(translation-pivot)[..., None]).squeeze(-1)+shift,
            r@rotation)


def object_frame_hands(joints, position, rotation):
    return (rotation[:, None].transpose(-1,-2) @
            (joints[:, HANDS]-position[:, None])[..., None]).squeeze(-1)


def terminal_displacements(joints, obj, task):
    """Use supplied goals with their original planar-human/3D-object semantics."""
    pelvis = joints[-1,0].clone(); pelvis[1] = 0
    human_delta = pelvis.new_tensor(task['pelvis_goal'])-pelvis
    object_delta = obj.new_tensor(task['object_goal'])-obj[-1]
    def correction(delta):
        distance = delta.norm()
        if float(distance) < .1:
            return torch.zeros_like(delta)
        return delta/distance*torch.clamp(distance-.08, max=.05)
    h, o = correction(human_delta), correction(object_delta)
    envelope = terminal_envelope(len(joints),joints.device)
    return envelope[:, None]*h, envelope[:, None]*o


def terminal_envelope(length,device):
    t = torch.arange(length,device=device,dtype=torch.float32)
    begin = max(5,length-33)
    return smoothstep((t-begin)/(length-3-begin))


class SurfaceProblem:
    """Native SMPL-X surfaces, source foot trajectories and full object geometry."""
    def __init__(self, source, model, object_vertices, sdf, info, floor, task,
                 arm, bound=.1, yaw_degrees=10):
        self.source, self.model, self.object_vertices = source, model, object_vertices
        self.task, self.arm = task, arm
        self.terminal = arm == 'terminal'
        self.device = source['pose'].device
        self.length = len(source['pose'])
        self.basis = spline_basis(self.length, self.device)
        self.envelope = edit_envelope(self.length, self.device)
        self.joint_ids = LEGS+ARMS if self.terminal else LEGS
        self.dimensions = len(self.joint_ids)*3 if self.terminal else 24
        self.bound, self.yaw_bound = bound, math.radians(yaw_degrees)
        self.rotations = transforms.axis_angle_to_matrix(source['pose'])
        self.rotation_logs = transforms.matrix_to_axis_angle(self.rotations)
        self.sdf = torch.as_tensor(sdf, dtype=torch.float32, device=self.device)[None]
        self.centroid = torch.as_tensor(info['centroid'], dtype=torch.float32, device=self.device)[None]
        self.extents = torch.as_tensor(info['extents'], dtype=torch.float32, device=self.device)[None]
        self.lower, self.upper = self.centroid[0]-self.extents.max()/2, self.centroid[0]+self.extents.max()/2
        self.foot_mask = source['joints'][:, FEET, 1] < floor+source['pose'].new_tensor((.08,.08,.04,.04))
        self.foot_pairs = self.foot_mask[1:] & self.foot_mask[:-1]
        self.position_count = int(self.foot_mask.sum())
        self.velocity_count = int(self.foot_pairs.sum())
        self.hand_reference = object_frame_hands(source['joints'], source['object_translation'], source['object_rotation'])
        self.hand_mask = torch.zeros(self.length,2,dtype=torch.bool,device=self.device)
        if self.terminal:
            self.human_shift, self.object_shift = terminal_displacements(source['joints'], source['object_translation'], task)
            for lo in range(0,self.length,24):
                hi = min(lo+24,self.length)
                vertices = self.objects(source['object_translation'][lo:hi], source['object_rotation'][lo:hi])
                distances = torch.cdist(source['joints'][lo:hi,HANDS],vertices).amin(-1)
                self.hand_mask[lo:hi] = distances < .05
            self.envelope = terminal_envelope(self.length,self.device)
            self.basis[-3:] = self.basis[-3].clone()
        self.hand_count = int(self.hand_mask.sum())
        self.model.eval().requires_grad_(False)

    def signed(self, points):
        from eval_metrics import compute_signed_distances
        return compute_signed_distances(self.sdf,self.centroid,self.extents,points)

    def objects(self, position, rotation):
        vertices = self.object_vertices[None].repeat(len(rotation),1,1)
        return (rotation.bmm(vertices.transpose(1,2))+position[:,:,None]).transpose(1,2)

    def outside(self, points):
        return ((self.lower-points).clamp_min(0)+(points-self.upper).clamp_min(0)).norm(dim=-1)

    def parameters(self):
        return torch.zeros(self.basis.shape[1],self.dimensions,device=self.device,requires_grad=True)

    def chunk(self, parameters, lo, hi):
        from utils import run_smplx_model, SMPLX_JOINTS_28
        field = (self.basis[lo:hi]@parameters).tanh()*self.envelope[lo:hi,None]
        pose = self.source['pose'][lo:hi].clone()
        translation = self.source['translation'][lo:hi].clone()
        object_position = self.source['object_translation'][lo:hi]
        object_rotation = self.source['object_rotation'][lo:hi]
        if self.terminal:
            translation = translation+self.human_shift[lo:hi]
            object_position = object_position+self.object_shift[lo:hi]
            local = field
            normalized = field
        else:
            human_control = field[:,:3]
            object_control = human_control if self.arm.startswith('relation') else field[:,3:6]
            shift = torch.stack((human_control[:,0],torch.zeros_like(field[:,0]),human_control[:,1]),-1)*self.bound
            object_shift = torch.stack((object_control[:,0],torch.zeros_like(field[:,0]),object_control[:,1]),-1)*self.bound
            yaw = human_control[:,2]*self.yaw_bound
            rotated = yaw_matrix(yaw)@self.rotations[lo:hi,0]
            pose[:,0] = pose[:,0]+transforms.matrix_to_axis_angle(rotated)-self.rotation_logs[lo:hi,0]
            translation = translation+shift
            object_position,object_rotation = shared_object_transform(
                self.source['joints'][lo:hi,0],object_position,object_rotation,
                object_shift,object_control[:,2]*self.yaw_bound)
            local = field[:,6:]
            normalized = torch.cat((human_control,object_control,local),-1)
        delta = transforms.axis_angle_to_matrix(local.reshape(-1,len(self.joint_ids),3)*math.radians(20))
        original = self.rotations[lo:hi,self.joint_ids]
        pose[:,self.joint_ids] = (pose[:,self.joint_ids]+
            transforms.matrix_to_axis_angle(original@delta)-self.rotation_logs[lo:hi,self.joint_ids])
        fixed = self.envelope[lo:hi]==0
        pose = torch.where(fixed[:,None,None],self.source['pose'][lo:hi],pose)
        translation = torch.where(fixed[:,None],self.source['translation'][lo:hi],translation)
        object_position = torch.where(fixed[:,None],self.source['object_translation'][lo:hi],object_position)
        object_rotation = torch.where(fixed[:,None,None],self.source['object_rotation'][lo:hi],object_rotation)
        vertices,joints = run_smplx_model(pose,translation,self.source['betas'],self.source['gender'],
                                         joints_ind=SMPLX_JOINTS_28,smpl_model=self.model)
        vertices = torch.where(fixed[:,None,None],self.source['verts'][lo:hi],vertices)
        joints = torch.where(fixed[:,None,None],self.source['joints'][lo:hi],joints)
        return dict(pose=pose,translation=translation,verts=vertices,joints=joints,
                    object_translation=object_position,object_rotation=object_rotation,normalized=normalized)

    def objective(self, parameters, backward=False):
        totals = dict(surface=0.,support_position=0.,support_velocity=0.,residual=0.,temporal=0.,domain=0.,contact=0.)
        for start in range(0,self.length,24):
            lo,hi = max(0,start-1),min(start+24,self.length)
            cut = start-lo
            state = self.chunk(parameters,lo,hi)
            terms = {name: parameters.new_zeros(()) for name in totals}
            object_vertices = self.objects(state['object_translation'][cut:],state['object_rotation'][cut:])
            source_objects = self.objects(self.source['object_translation'][start:hi],self.source['object_rotation'][start:hi])
            for vertices,reference in ((state['verts'][cut:],self.source['verts'][start:hi]),(object_vertices,source_objects)):
                terms['surface'] = terms['surface']+(-self.signed(vertices)).clamp_min(0).sum()/(self.length*vertices.shape[1]*.01)
                increase = (self.outside(vertices)-self.outside(reference)).clamp_min(0)
                terms['domain'] = terms['domain']+10*increase.square().sum()/(self.length*vertices.shape[1]*.01**2)
            delta = state['joints'][:,FEET]-self.source['joints'][lo:hi,FEET]
            if self.position_count:
                terms['support_position'] = (delta[cut:].square().sum(-1)*self.foot_mask[start:hi]).sum()/(3*self.position_count*.02**2)
            if self.velocity_count:
                difference = (delta[1:]-delta[:-1])*30
                terms['support_velocity'] = .2*(difference.square().sum(-1)*self.foot_pairs[lo:hi-1]).sum()/(3*self.velocity_count*.03**2)
            field = state['normalized']
            terms['residual'] = .05*field[cut:].square().sum()/(self.length*field.shape[1])
            terms['temporal'] = .05*(field[1:]-field[:-1]).square().sum()/((self.length-1)*field.shape[1])
            if self.hand_count:
                relative = object_frame_hands(state['joints'][cut:],state['object_translation'][cut:],state['object_rotation'][cut:])
                terms['contact'] = ((relative-self.hand_reference[start:hi]).square().sum(-1)*self.hand_mask[start:hi]).sum()/(3*self.hand_count*.01**2)
            value = sum(terms.values())
            if backward:
                value.backward()
            for name,value in terms.items():
                totals[name] += float(value.detach())
        return totals

    @torch.no_grad()
    def decode(self, parameters):
        parts = [self.chunk(parameters,lo,min(lo+24,self.length)) for lo in range(0,self.length,24)]
        result = {key:torch.cat([p[key] for p in parts]) for key in parts[0] if key!='normalized'}
        result.update(betas=self.source['betas'],gender=self.source['gender'])
        return result

    def solve(self, steps):
        parameters = self.parameters()
        optimizer = torch.optim.Adam([parameters],lr=.05)
        best,best_value,best_iteration = parameters.detach().clone(),float('inf'),0
        trace = []
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for iteration in range(steps+1):
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(iteration<steps):
                terms = self.objective(parameters,backward=iteration<steps)
            value = sum(terms.values())
            if not math.isfinite(value):
                raise FloatingPointError('nonfinite surface-edit objective')
            if value<best_value:
                best,best_value,best_iteration = parameters.detach().clone(),value,iteration
            record = dict(iteration=iteration,objective=value,terms=terms)
            if iteration<steps:
                if not torch.isfinite(parameters.grad).all():
                    raise FloatingPointError('nonfinite surface-edit gradient')
                record['gradient_norm'] = float(parameters.grad.norm())
                optimizer.step()
            trace.append(record)
            if iteration%10==0:
                print(json.dumps(dict(surface_iteration=iteration,objective=value,best=best_value)),flush=True)
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter()-started
        if best_iteration==0 and not self.terminal:
            result = self.source
        else:
            result = self.decode(best)
        return result,dict(trace=trace,best_iteration=best_iteration,best_objective=best_value,
                          optimization_seconds=seconds,steps=steps,parameters=best.cpu())


def native_metrics(tracks, task, object_vertices, object_sdf, object_info, sdf, info, faces, seed):
    from test_infbagel_hosi import compute_metrics_for_sample, _subsample_seed
    vertices = object_vertices[None].repeat(len(tracks['object_rotation']),1,1)
    obj = (tracks['object_rotation'].bmm(vertices.transpose(1,2))+tracks['object_translation'][:,:,None]).transpose(1,2)
    name = task['object_name']; key = task['scene_name']+'_sdf'
    metrics = compute_metrics_for_sample(tracks['joints'].flatten(1),tracks['object_translation'],tracks['object_rotation'],
        task,{name:object_vertices},{name:object_sdf},{name:object_info},None,
        tracks['verts'],tracks['joints'].clone(),obj,name,{key:sdf},{key:info},faces,
        subsample_seed=_subsample_seed(seed,task['scene_name'],task['test_idx']))
    metrics = {k:float(v) for k,v in metrics.items()}
    metrics['completed'] = metrics['xy_points_err']<10 and metrics['end_obj_trans_err']<10
    return metrics


def motion_audit(source, result):
    relative = object_frame_hands(result['joints'],result['object_translation'],result['object_rotation'])
    original = object_frame_hands(source['joints'],source['object_translation'],source['object_rotation'])
    return dict(hand_object_max_error_m=float((relative-original).norm(dim=-1).max()),
        initial_max_error_m=float((result['joints'][:6]-source['joints'][:6]).abs().max()),
        final_max_error_m=float((result['joints'][-3:]-source['joints'][-3:]).abs().max()),
        initial_object_max_error_m=float((result['object_translation'][:6]-source['object_translation'][:6]).abs().max()),
        final_object_max_error_m=float((result['object_translation'][-3:]-source['object_translation'][-3:]).abs().max()),
        mean_joint_change_mm=float((result['joints']-source['joints']).norm(dim=-1).mean()*1000),
        root_path_m=float((result['joints'][1:,0]-result['joints'][:-1,0]).norm(dim=-1).sum()),
        mean_joint_frame_m=float((result['joints'][1:]-result['joints'][:-1]).norm(dim=-1).mean()))


def terminal_acceptance(before, after):
    failures = []
    if before['completed'] or not after['completed']: failures.append('completion_recovery')
    if after['contact_percent']<before['contact_percent']-.02: failures.append('contact')
    if after['foot_sliding']>before['foot_sliding']+.02: failures.append('foot_sliding')
    for key in ('scene_human_penetration_s_mean','scene_obj_penetration_s_mean'):
        if after[key]>before[key]+.05: failures.append(key)
    return not failures,failures


def run_surface_tasks(cfg):
    """Hydra native-evaluation entry; every saved task is independently resumable."""
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from test_infbagel_hosi import seed_everything
    if cfg.get('run_id') and subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('reportable surface editing requires a clean worktree')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    protocol_path = Path(cfg.surface_edit.protocol)
    root = protocol_path.resolve().parents[2]
    protocol = json.loads(protocol_path.read_text())
    task_manifest = root/protocol['task_manifest'] if cfg.surface_edit.task_manifest is None else Path(cfg.surface_edit.task_manifest)
    tasks = json.loads(task_manifest.read_text())['tasks']
    if cfg.surface_edit.task_ids is not None:
        tasks = [t for t in tasks if t['canonical_ordinal'] in cfg.surface_edit.task_ids]
    out = Path(cfg.hosi_output_dir); out.mkdir(parents=True,exist_ok=False)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)),out/'resolved.yaml')
    seed_everything(42)
    device = torch.device(cfg.device)
    source_root = root/protocol['source_run'] if cfg.surface_edit.source_root is None else Path(cfg.surface_edit.source_root)
    dataset = None; current_scene = None; smpl_cache = {}; records = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for item in tasks:
        scene = item['scene_name']; ordinal = item['canonical_ordinal']
        if current_scene!=scene:
            dc = OmegaConf.merge(cfg.dataset,dict(device=str(device),vis=True,load_object_payload=False,test_scene_name=scene))
            dataset = InfBaGelDataset(**dc)
            dataset.obj_rest_verts = {k:v.to(device) for k,v in dataset.obj_rest_verts.items()}
            native = json.loads((root/'data/hosi_test/data'/(scene+'.json')).read_text())
            key = scene+'_sdf'
            sdf_root = root/'data/hosi_test/Scene_sdf'
            sdf = np.load(sdf_root/(key+'.npy')); info = json.loads((sdf_root/(key+'_info.json')).read_text())
            current_scene = scene
        task = dict(native[item['test_idx']],test_idx=item['test_idx'])
        path, = source_root.glob(f"{protocol['source_arm']}-shard*/episode-motion-{ordinal:03d}.pt")
        saved = torch.load(path,map_location='cpu',weights_only=False)
        world = {k:torch.as_tensor(v).reshape(-1,28,3) if k=='points_world' else torch.as_tensor(v)
                 for k,v in saved['stitched'].items()}
        with torch.no_grad():
            source = native_tracks(cfg,dataset,world,task,True,smpl_cache,body_parameters=True)
        source = {k:v.detach() if torch.is_tensor(v) else v for k,v in source.items()}
        object_vertices = dataset.obj_rest_verts[item['object_name']]
        object_sdf_root = root/'data/object/rest_object_sdf_256_npy_files'
        object_sdf,object_info = load_object_sdf(object_sdf_root,item['object_name'])
        model = smpl_cache[source['gender']]
        def evaluate(tracks):
            return native_metrics(tracks,task,object_vertices,object_sdf,object_info,sdf,info,model.faces,42)
        source_metrics = evaluate(source)
        old_metrics = json.loads(path.with_name(f'episode-audit-{ordinal:03d}.json').read_text())['metrics']
        joint_error = float((source['joints'].cpu()-saved['evaluated_joints_world']).abs().max())
        metric_error = max(abs(float(source_metrics[k])-float(old_metrics[k])) for k in source_metrics)
        if joint_error>1e-5 or metric_error>1e-5:
            raise AssertionError(f'native source recovery differs: joints={joint_error}, metrics={metric_error}')
        dest = out/f'task-{ordinal:03d}'; dest.mkdir()
        write_json(dest/'source.json',dict(metrics=source_metrics,motion_path=str(path),joint_error_m=joint_error,metric_error=metric_error))
        task_rows = dict(source=dict(metrics=source_metrics,audit=motion_audit(source,source)))
        for arm in cfg.surface_edit.arms:
            torch.cuda.synchronize(device)
            arm_started = time.perf_counter()
            base = source
            before = source_metrics
            if arm=='terminal' and cfg.surface_edit.terminal_source is not None:
                previous = Path(cfg.surface_edit.terminal_source)
                selected = (str(cfg.surface_edit.terminal_arm) if cfg.surface_edit.terminal_arm is not None
                            else json.loads((previous/'selection.json').read_text())['arm'])
                if selected!='source':
                    prior, = previous.glob(f'lanes/*/task-{ordinal:03d}/{selected}.pt')
                    data = torch.load(prior,map_location=device,weights_only=False)
                    base = dict(data['motion'],betas=source['betas'],gender=source['gender'])
                    base['verts'],_ = decode_body(base,model)
                    before = data['metrics']
            if arm=='terminal' and before['completed']:
                result = base; solve = dict(trace=[],steps=0,optimization_seconds=0.,best_iteration=0,parameters=None)
            else:
                bound = .2 if arm.endswith('_20') else .1
                problem = SurfaceProblem(base,model,object_vertices,sdf,info,float(before['feet_height'])/100,
                                         task,arm,bound,20 if bound==.2 else 10)
                with torch.no_grad():
                    risk = 0.
                    for lo in range(0,len(base['joints']),24):
                        hi = min(lo+24,len(base['joints']))
                        for vertices in (base['verts'][lo:hi],problem.objects(base['object_translation'][lo:hi],base['object_rotation'][lo:hi])):
                            risk += float((-problem.signed(vertices)).clamp_min(0).sum()/(len(base['joints'])*vertices.shape[1]))
                steps = 20 if arm=='terminal' or risk<=.005 else 40
                if risk==0 and arm!='terminal':
                    result = base; solve = dict(trace=[],steps=0,optimization_seconds=0.,best_iteration=0,parameters=None)
                else:
                    result,solve = problem.solve(steps)
                solve['source_mean_depth_m'] = risk
            candidate_metrics = evaluate(result)
            accepted,reasons = (terminal_acceptance(before,candidate_metrics) if arm=='terminal' and not before['completed'] else (True,[]))
            candidate = result
            if not accepted: result = base
            metrics = candidate_metrics if accepted else before
            audit = motion_audit(base,result)
            motion = {k:v.detach().cpu() for k,v in result.items() if torch.is_tensor(v) and k not in ('verts','betas')}
            candidate_motion = None if accepted else {k:v.detach().cpu() for k,v in candidate.items() if torch.is_tensor(v) and k not in ('verts','betas')}
            torch.cuda.synchronize(device)
            solve['arm_seconds_including_evaluation'] = time.perf_counter()-arm_started
            payload = dict(motion=motion,metrics=metrics,input_metrics=before,candidate_motion=candidate_motion,candidate_metrics=candidate_metrics,
                           solver=solve,audit=audit,accepted=accepted,rejection_reasons=reasons)
            with (dest/(arm+'.pt')).open('xb') as f: torch.save(payload,f)
            trace = {k:v for k,v in solve.items() if k!='parameters'}
            row = dict(metrics=metrics,input_metrics=before,candidate_metrics=candidate_metrics,audit=audit,solver=trace,accepted=accepted,rejection_reasons=reasons)
            write_json(dest/(arm+'.json'),row)
            task_rows[arm] = row
            print(json.dumps(dict(task=ordinal,arm=arm,metrics=metrics,seconds=solve['optimization_seconds']),allow_nan=False),flush=True)
        records.append(dict(task=ordinal,scene=scene,object=item['object_name'],arms=task_rows))
        write_json(dest/'complete.json',records[-1])
    torch.cuda.synchronize(device)
    write_json(out/'metrics.json',dict(git_commit=commit,live_head_at_completion=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tasks=records,seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        hoi_calls=0,hsi_calls=0))


@torch.no_grad()
def decode_body(motion,model):
    from utils import run_smplx_model, SMPLX_JOINTS_28
    parts = [run_smplx_model(motion['pose'][lo:lo+24],motion['translation'][lo:lo+24],motion['betas'],motion['gender'],
                           joints_ind=SMPLX_JOINTS_28,smpl_model=model) for lo in range(0,len(motion['pose']),24)]
    return torch.cat([p[0] for p in parts]),torch.cat([p[1] for p in parts])


def summarize_surface(run_root, task_manifest, device='cuda:7', terminal=False, frozen_arm=None):
    """Task/scene paired native tables, fixed development selection and all failures."""
    from .scene_calibration import paired_local_metrics
    run_root = Path(run_root)
    expected = {r['canonical_ordinal'] for r in json.loads(Path(task_manifest).read_text())['tasks']}
    records = [json.loads(p.read_text()) for p in sorted(run_root.glob('lanes/*/task-*/complete.json'))]
    if len(records)!=len(expected) or {r['task'] for r in records}!=expected:
        raise ValueError('incomplete or duplicate surface-edit task coverage')
    arms = list(records[0]['arms'])
    tasks = {a:{str(r['task']):r['arms'][a]['metrics'] for r in records} for a in arms}
    if 'terminal' in arms:
        tasks['terminal_input'] = {str(r['task']):r['arms']['terminal']['input_metrics'] for r in records}
    table_arms = list(tasks)
    names = {str(r['task']):r['scene'] for r in records}
    def average(rows):
        return {k:sum(float(v[k]) for v in rows)/len(rows) for k in rows[0]}
    scenes = {a:{s:average([row for task,row in tasks[a].items() if names[task]==s])
                 for s in sorted(set(names.values()))} for a in table_arms}
    means = {a:average(list(tasks[a].values())) for a in table_arms}
    contrasts = {}
    pairs = [('source',a) for a in arms if a!='source']
    pairs += [(a.replace('relation','independent'),a) for a in arms
              if a.startswith('relation') and a.replace('relation','independent') in arms]
    if 'terminal' in arms: pairs += [('terminal_input','terminal')]
    for unit,data in (('task',tasks),('scene',scenes)):
        contrasts[unit] = {b+'-minus-'+a:paired_local_metrics(data[a],data[b],device) for a,b in pairs}
    base = means['source']; eligibility = {}
    for arm in (a for a in arms if a.startswith('relation')):
        row = means[arm]; failures = []
        if row['scene_obj_penetration_s_mean']>.9*base['scene_obj_penetration_s_mean']: failures.append('OS_10_percent')
        if row['scene_human_penetration_s_mean']>base['scene_human_penetration_s_mean']: failures.append('HS')
        if row['contact_percent']<base['contact_percent']-.02: failures.append('contact')
        if row['completed']<base['completed']: failures.append('completion')
        if row['foot_sliding']>base['foot_sliding']+.02: failures.append('foot_sliding')
        audit = [r['arms'][arm]['audit'] for r in records]
        if max(a['hand_object_max_error_m'] for a in audit)>1e-4: failures.append('relation_invariant')
        for key in ('initial_max_error_m','final_max_error_m','initial_object_max_error_m','final_object_max_error_m'):
            if max(a[key] for a in audit)>1e-5: failures.append(key)
        eligibility[arm] = dict(eligible=not failures,failures=failures)
    available = [a for a in eligibility if eligibility[a]['eligible']]
    selected = min(available,key=lambda a:(means[a]['scene_obj_penetration_s_mean'],a)) if available else 'source'
    if frozen_arm is not None:
        selected = frozen_arm
    selection = dict(arm=selected,full469_entry=bool(available),eligibility=eligibility,
                     selection_scope=('fixed before full469 outcomes' if frozen_arm is not None else
                                      'registered development point estimates; all task/scene intervals reported'))
    directory = run_root/'analysis'; directory.mkdir()
    for name,value in [('tasks',tasks),('scenes',scenes),('means',means),('paired',contrasts),('selection',selection)]:
        write_json(directory/(name+'.json'),value)
    write_json(run_root/'selection.json',selection)
    result = dict(task_count=len(records),scene_count=len(set(names.values())),means=means,
        selection=selection,contrasts=contrasts,
        changed_tasks={a:sum(r['arms'][a]['audit']['mean_joint_change_mm']>0 for r in records) for a in arms},
        optimization_seconds={a:sum(r['arms'][a].get('solver',{}).get('optimization_seconds',0) for r in records) for a in arms},
        accepted_tasks={a:sum(r['arms'][a].get('accepted',True) for r in records) for a in arms},
        source_references='Sealed native P15/ArmB full motions; same per-episode seed42 protocol.',
        paper_reference=dict(method='InfBaGel paper Hybrid HOSI-test',HS=3.17,OS=12.45,FS=.15,contact_percent=76.96,success_percent=81.45,
                             comparison='User-authorized direct external aggregate comparison; no paired InfBaGel significance.'),
        test_set_development=True,training_allowed=False)
    write_json(directory/'summary.json',result)
    return result


def render_surface(run_root,source_root,selected=None,tasks=(14,329,371,375,420)):
    """Fixed complete native-joint animations and native-metric comparisons."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    import trimesh
    from utils import zup_to_yup
    run_root,source_root = Path(run_root),Path(source_root)
    report = json.loads((run_root/'analysis/summary.json').read_text())
    if selected is None:
        selected = report['selection']['arm']
    if selected=='source': selected='relation_20'
    alternatives = [selected.replace('relation','independent'),selected]
    if 'terminal' in report['means']: alternatives=['terminal']
    out = run_root/'visualizations';out.mkdir()
    means = report['means']; labels = list(means)
    fig,axes = plt.subplots(1,5,figsize=(16,4))
    for ax,key,title in zip(axes,('scene_human_penetration_s_mean','scene_obj_penetration_s_mean','foot_sliding','contact_percent','completed'),
                            ('Human-scene penetration','Object-scene penetration','Foot sliding (cm)','Contact fraction','Completion fraction')):
        ax.bar(range(len(labels)),[means[a][key] for a in labels]);ax.set_title(title,fontsize=10)
        ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,rotation=55,ha='right',fontsize=8)
    fig.tight_layout();fig.savefig(out/'native_metrics.png',dpi=160);plt.close(fig)
    parents=(-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19)
    root=Path(__file__).resolve().parents[2]
    for task_id in tasks:
        dest, = run_root.glob(f'lanes/*/task-{task_id:03d}')
        source_path, = source_root.glob(f'B0_hoi-shard*/episode-motion-{task_id:03d}.pt')
        saved=torch.load(source_path,map_location='cpu',weights_only=False)
        human=[saved['evaluated_joints_world'].numpy()]
        from utils import interp_object
        op,orr=interp_object(saved['stitched']['object_translation_world'],saved['stitched']['object_rotation_world'],3)
        objects=[(op,orr.reshape(-1,3,3))]
        for arm in alternatives:
            data=torch.load(dest/(arm+'.pt'),map_location='cpu',weights_only=False)['motion']
            human.append(data['joints'].numpy());objects.append((data['object_translation'].numpy(),data['object_rotation'].numpy()))
        mesh=trimesh.load_mesh(root/'data/object/rest_object_geo'/(saved['object_name']+'.ply'))
        vertices=zup_to_yup(np.asarray(mesh.vertices))[np.linspace(0,len(mesh.vertices)-1,256).astype(int)]
        object_points=[(r@vertices.T).transpose(0,2,1)+p[:,None] for p,r in objects]
        scene=saved['scene_name']+'_sdf'
        sdf=np.load(root/'data/hosi_test/Scene_sdf'/(scene+'.npy'))
        info=json.loads((root/'data/hosi_test/Scene_sdf'/(scene+'_info.json')).read_text())
        indices=np.argwhere(np.abs(sdf)<.007)
        indices=indices[::max(1,len(indices)//3000)]
        scene_points=(indices/(np.array(sdf.shape)-1)*2-1)*max(info['extents'])/2+np.array(info['centroid'])
        all_points=np.concatenate([v.reshape(-1,3) for v in human+object_points])
        low,high=all_points.min(0)-.3,all_points.max(0)+.3
        fig=plt.figure(figsize=(6*len(human),5))
        axes=[fig.add_subplot(1,len(human),i+1,projection='3d') for i in range(len(human))]
        def draw(frame):
            for index,ax in enumerate(axes):
                ax.clear(); joints=human[index][frame]
                ax.scatter(scene_points[:,0],scene_points[:,2],scene_points[:,1],s=.2,c='gray',alpha=.12)
                points=object_points[index][frame];ax.scatter(points[:,0],points[:,2],points[:,1],s=2,c='orange')
                for j,parent in enumerate(parents):
                    if parent>=0:
                        line=joints[[parent,j]];ax.plot(line[:,0],line[:,2],line[:,1],c='royalblue',lw=2)
                ax.scatter(joints[list(HANDS),0],joints[list(HANDS),2],joints[list(HANDS),1],s=18,c='red')
                ax.set(xlim=(low[0],high[0]),ylim=(low[2],high[2]),zlim=(low[1],high[1]),
                       title=f"{(['source']+alternatives)[index]} | task {task_id} | {frame/30:.2f}s")
                ax.set_box_aspect((high-low)[[0,2,1]]);ax.view_init(20,-65)
            fig.suptitle('Native SMPL-X joints; surface metrics evaluated separately',fontsize=10)
        writer=FFMpegWriter(fps=10,bitrate=1800)
        with writer.saving(fig,str(out/f'task-{task_id:03d}.mp4'),dpi=90):
            for frame in range(0,len(human[0]),3): draw(frame);writer.grab_frame()
        for frame in np.linspace(0,len(human[0])-1,4).astype(int):
            draw(frame);fig.savefig(out/f'task-{task_id:03d}-frame-{frame:04d}.png',dpi=110)
        plt.close(fig)
