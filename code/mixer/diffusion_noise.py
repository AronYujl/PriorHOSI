"""Frozen HSI DDIM latent editing with the author's local DNO optimizer."""
import json
import sys
import time
from pathlib import Path

import torch
from pytorch3d import transforms
from torch.utils.checkpoint import checkpoint

from .body_projection import NATIVE_ANCHORS, NativeSceneDifferential, native_increment_pose
from .hsi_motion_target import lift_native_prediction
from .input_views import KnownEmptyObjectView


def ddim_transition(value, clean, alpha, next_alpha):
    noise = (value-alpha.sqrt()*clean)/(1-alpha).sqrt()
    return next_alpha.sqrt()*clean+(1-next_alpha).sqrt()*noise


def native_quaternion_interpolation(quaternion, scale=3):
    """Native SLERP/LERP convention, evaluated only on each branch's domain."""
    first = quaternion[:-1, None].expand(-1, scale, -1, -1)
    second = quaternion[1:, None].expand_as(first)
    fraction = torch.arange(scale, device=quaternion.device, dtype=quaternion.dtype)[None, :, None, None]/scale
    fraction = fraction.expand(*first.shape[:-1], 1)
    first, second, fraction = first.reshape(-1, 4), second.reshape(-1, 4), fraction.reshape(-1, 1)
    dot = (first*second).sum(-1, keepdim=True)
    first = torch.where(dot < 0, -first, first)
    dot = (first*second).sum(-1, keepdim=True)
    linear = (dot > 1-1e-6).squeeze(-1)
    result = torch.empty_like(first)
    # The native near-parallel branch weights the second endpoint by1-t.
    result[linear] = first[linear]*fraction[linear]+second[linear]*(1-fraction[linear])
    curved = ~linear
    omega = torch.acos(dot[curved].clamp(-1, 1))
    result[curved] = (first[curved]*torch.sin((1-fraction[curved])*omega)
                      +second[curved]*torch.sin(fraction[curved]*omega))/torch.sin(omega)
    result = result/result.norm(dim=-1, keepdim=True)
    return torch.cat((result.reshape(-1, quaternion.shape[1], 4),
                      quaternion[-1:].expand(scale, -1, -1)))


def native_linear_interpolation(value, scale=3):
    index = torch.arange(len(value)*scale, device=value.device)
    lo = index//scale
    hi = (lo+1).clamp_max(len(value)-1)
    weight = ((index % scale)/scale).reshape(-1, *([1]*(value.ndim-1)))
    return value[lo]+weight*(value[hi]-value[lo])


class HSIDDIM:
    def __init__(self, teacher, windows, source, task, ordinal, settings):
        self.model, self.dataset = teacher.model, teacher.dataset
        self.teacher, self.windows, self.source = teacher, windows, source
        self.settings = settings
        self.device = source['pose'].device
        self.alpha = teacher.sampler.hsi_sampler.alpha_cumprod.to(self.device)
        self.clean = torch.cat([w['source'].to(self.device)[..., :216] for w in windows])
        self.mats = [w['mat'].to(self.device) for w in windows]
        self.arguments = [{k: [a.to(self.device) for a in args] for k, args in w['arguments'].items()}
                          for w in windows]
        self.empty = []
        for i, w in enumerate(windows):
            empty = KnownEmptyObjectView()
            empty.begin_window(w['source'].to(self.device), 42+900000000+ordinal*10000+i*100)
            self.empty.append(empty)
        sequence = self.dataset.ori_sequence_idx[task['data_idx']]
        self.dataset_translation = torch.as_tensor(self.dataset.transl[sequence], device=self.device)
        self.source_decode = self.raw_pose(self.clean)

    def reframe(self, value, old, new):
        points = self.dataset.denormalize_torch(value[..., :84]).reshape(1, -1, 28, 3)
        rotation = new[:, :3, :3].transpose(-1, -2)@old[:, :3, :3]
        shift = (new[:, :3, :3].transpose(-1, -2)@(old[:, :3, 3]-new[:, :3, 3])[..., None]).squeeze(-1)
        points = (rotation[:, None, None]@points[..., None]).squeeze(-1)+shift[:, None, None]
        human = rotation[:, None, None]@transforms.rotation_6d_to_matrix(value[..., 84:].reshape(1, -1, 22, 6))
        return torch.cat((self.dataset.normalize_torch(points).flatten(2),
                          transforms.matrix_to_rotation_6d(human).flatten(2)), -1)

    def predict(self, current, history, window, view, step):
        current = torch.cat((history, current[:, 2:]), 1)
        whole = torch.cat((current, current.new_zeros(1, 16, 16)), -1)
        whole = self.empty[window].for_step(whole, step)
        args = list(self.arguments[window][view])
        args[1] = torch.full((1,), step, device=self.device, dtype=torch.long)

        def forward(value, arguments=tuple(args)):
            self.teacher.calls += 1
            return self.model(value, *arguments, is_sample=True)[..., :216]

        prediction = checkpoint(forward, whole, use_reentrant=False) if torch.is_grad_enabled() else forward(whole)
        return torch.cat((history, prediction[:, 2:]), 1)

    def decode(self, latent, view, *, steps=None, source_history=False):
        future = latent[0, :, 0].transpose(0, 1).reshape(len(self.windows), 14, 216)
        times = torch.linspace(0, 499, steps or self.settings['ddim_steps']).round().long().tolist()[::-1]
        predictions = []
        for i in range(len(self.windows)):
            history = self.clean[i:i+1, :2] if i == 0 or source_history else self.reframe(predictions[-1][:, -2:], self.mats[i-1], self.mats[i])
            value = torch.cat((history, future[i:i+1]), 1)
            for j, step in enumerate(times):
                clean = self.predict(value, history, i, view, step)
                next_alpha = self.alpha[times[j+1]] if j+1 < len(times) else self.alpha.new_tensor(1.)
                value = ddim_transition(value, clean, self.alpha[step], next_alpha)
            predictions.append(torch.cat((history, value[:, 2:]), 1))
        return torch.cat(predictions)

    @torch.no_grad()
    def invert(self):
        times = torch.linspace(0, 499, self.settings['inversion_steps']).round().long().tolist()
        latents = []
        for i in range(len(self.windows)):
            value = self.clean[i:i+1].clone()
            history = value[:, :2]
            for j, step in enumerate(times):
                if j == len(times)-1 and self.settings.get('inversion_terminal') == 'model':
                    break
                clean = self.predict(value, history, i, 'correct', step)
                next_alpha = self.alpha[times[j+1]] if j+1 < len(times) else self.alpha.new_tensor(0.)
                value = ddim_transition(value, clean, self.alpha[step], next_alpha)
            latents.append(value[:, 2:])
        return torch.cat(latents, 1).transpose(1, 2).unsqueeze(2)

    def raw_pose(self, prediction):
        roots, rotations = [], []
        for i, value in enumerate(prediction):
            mat = self.mats[i][0]
            positions = self.dataset.denormalize_torch(value[..., :84]).reshape(16, 28, 3)
            world_root = (mat[:3, :3]@positions[:, 0, :, None]).squeeze(-1)+mat[:3, 3]
            global_rotation = mat[:3, :3]@transforms.rotation_6d_to_matrix(value[..., 84:].reshape(16, 22, 6))
            keep = slice(None) if i == 0 else slice(2, None)
            roots.append(world_root[keep]); rotations.append(global_rotation[keep])
        global_rotation = torch.cat(rotations)
        local = self.dataset.quat_ik_torch(global_rotation)
        quaternion = transforms.matrix_to_quaternion(local)
        interpolated = native_quaternion_interpolation(quaternion)
        pose = transforms.matrix_to_axis_angle(transforms.quaternion_to_matrix(interpolated))
        translation = native_linear_interpolation(torch.cat(roots))+self.dataset_translation
        return dict(pose=pose, translation=translation)

    def pose(self, prediction):
        raw = self.raw_pose(prediction)
        pose, translation = lift_native_prediction(self.source, self.source_decode, raw)
        fixed = torch.zeros(len(pose), device=pose.device, dtype=torch.bool)
        fixed[:6] = True; fixed[-3:] = True
        return (torch.where(fixed[:, None, None], self.source['pose'], pose),
                torch.where(fixed[:, None], self.source['translation'], translation))


class _PhysicalLoss(torch.autograd.Function):
    """Exact first derivatives accumulated in native24-frame SMPL-X blocks."""
    @staticmethod
    def forward(ctx, pose, translation, objective):
        from utils import run_smplx_model, SMPLX_JOINTS_28
        source, length = objective.source, len(pose)
        grad_pose, grad_translation = torch.zeros_like(pose), torch.zeros_like(translation)
        sums = pose.new_zeros(len(objective.term_names))
        weights = pose.new_tensor(objective.weights)
        with torch.enable_grad():
            for lo in range(0, length, 24):
                start, stop = max(lo-1, 0), min(lo+24, length)
                p = pose[start:stop].detach().requires_grad_(True)
                t = translation[start:stop].detach().requires_grad_(True)
                vertices, joints = run_smplx_model(p, t, source['betas'], source['gender'],
                    joints_ind=SMPLX_JOINTS_28, smpl_model=objective.model)
                terms = objective.chunk_terms(vertices,joints,start,lo,stop)
                gp, gt = torch.autograd.grad((terms*weights).sum(), (p, t))
                grad_pose[start:stop] += gp; grad_translation[start:stop] += gt
                sums += terms.detach()
        objective.last_terms = sums.detach().cpu().tolist()
        ctx.save_for_backward(grad_pose, grad_translation)
        return (sums*weights).sum()

    @staticmethod
    def backward(ctx, upstream):
        pose, translation = ctx.saved_tensors
        return upstream*pose, upstream*translation, None


class NativeDNOObjective:
    term_names = ('body','anchors','velocity','scene')

    def __init__(self, projector, model, sdf, info, source_hs, edit):
        self.source, self.model, self.edit = projector.source, model.eval().requires_grad_(False), edit
        self.differential = NativeSceneDifferential(projector, model, sdf, info)
        self.body_scale = .05 if edit else .01
        self.scene_scale = max(source_hs, 1.)
        self.last_terms = None
        self.weights = [1.]*len(self.term_names)

    def chunk_terms(self,vertices,joints,start,lo,stop):
        length=len(self.source['joints']);offset=lo-start
        delta=joints-self.source['joints'][start:stop]
        body=delta[offset:].square().sum()/(length*28*3*self.body_scale**2)
        if self.edit:
            anchors=delta[offset:,NATIVE_ANCHORS].square().sum()/(length*6*3*.001**2)
            velocity=((delta[1:]-delta[:-1])*30).square().sum()/((length-1)*28*3*.1**2)
            scene=self.differential.frame_sums(vertices[offset:]).sum()/(length*self.scene_scale)
        else:
            anchors,velocity,scene=(body.new_zeros(()) for _ in range(3))
        return torch.stack((body,anchors,velocity,scene))

    def term_record(self):
        return {name:value*weight for name,value,weight in zip(self.term_names,self.last_terms,self.weights)}

    def __call__(self, pose, translation):
        return _PhysicalLoss.apply(pose, translation, self)


class ConstrainedDNOObjective(NativeDNOObjective):
    term_names = ('body','hand','stance','boundary','velocity','seam_velocity','scene')

    @torch.no_grad()
    def __init__(self,projector,model,sdf,info,source_hs,floor,object_vertices):
        from .surface_edit import native_hand_distances,FEET
        from utils import run_smplx_model,SMPLX_JOINTS_28
        super().__init__(projector,model,sdf,info,source_hs,True)
        source=self.source;length=len(source['pose']);device=source['pose'].device
        self.fixed=projector.fixed
        self.hand_mask=torch.zeros(length,2,device=device,dtype=torch.bool)
        self.reference_chunks={}
        self.source_fk_reference_max_error_m=0.
        for lo in range(0,length,24):
            start,stop=max(lo-1,0),min(lo+24,length)
            _,reference=run_smplx_model(source['pose'][start:stop],source['translation'][start:stop],
                source['betas'],source['gender'],joints_ind=SMPLX_JOINTS_28,smpl_model=self.model)
            self.reference_chunks[start]=reference
            self.source_fk_reference_max_error_m=max(self.source_fk_reference_max_error_m,
                float((reference-source['joints'][start:stop]).abs().max()))
            vertices=(source['object_rotation'][lo:stop]@object_vertices.T).transpose(1,2)+source['object_translation'][lo:stop,None]
            self.hand_mask[lo:stop]=native_hand_distances(source['joints'][lo:stop],vertices)<.05
        self.stance_mask=source['joints'][:,FEET,1]<floor+source['pose'].new_tensor((.08,.08,.04,.04))
        self.boundary_mask=torch.zeros(length,device=device,dtype=torch.bool)
        for seam in projector.seams.tolist():self.boundary_mask[seam-6:seam]=True
        self.seam_mask=torch.zeros(length-1,device=device,dtype=torch.bool)
        self.seam_mask[projector.seams-1]=True
        # Empty sets have zero numerator and contribute zero to the objective.
        self.denominators=dict(hand=max(int(self.hand_mask.sum()),1)*3*.01**2,
            stance=max(int(self.stance_mask.sum()),1)*3*.005**2,
            boundary=max(int(self.boundary_mask.sum()),1)*28*3*.01**2,
            seam_velocity=max(int(self.seam_mask.sum()),1)*28*3*.1**2)

    def chunk_terms(self,vertices,joints,start,lo,stop):
        from .surface_edit import HANDS,FEET
        length=len(self.source['pose']);offset=lo-start
        # Identical source/current FK batches make the zero increment exactly zero.
        # This removes only source FK rounding, never diffusion reconstruction error.
        delta=joints-self.reference_chunks[start];owned=delta[offset:]
        velocity=(delta[1:]-delta[:-1])*30
        body=owned.square().sum()/(length*28*3*.05**2)
        hand=(owned[:,HANDS].square()*self.hand_mask[lo:stop,:,None]).sum()/self.denominators['hand']
        stance=(owned[:,FEET].square()*self.stance_mask[lo:stop,:,None]).sum()/self.denominators['stance']
        boundary=(owned.square()*self.boundary_mask[lo:stop,None,None]).sum()/self.denominators['boundary']
        speed=velocity.square().sum()/((length-1)*28*3*.1**2)
        seam=(velocity.square()*self.seam_mask[start:stop-1,None,None]).sum()/self.denominators['seam_velocity']
        visible=torch.where(self.fixed[lo:stop,None,None],self.source['verts'][lo:stop],vertices[offset:])
        scene=self.differential.frame_sums(visible).sum()/(length*self.scene_scale)
        return torch.stack((body,hand,stance,boundary,speed,seam,scene))


@torch.enable_grad()
def optimize_latent(decoder, objective, initial, view, settings, destination, stage):
    sys.path.insert(0, str(destination['repository']))
    from dno import DNO, DNOOptions
    # The official decorrelation padding and zero-scale perturbation draw noise.
    # Reset both paired edits so those draws correspond at every iteration.
    torch.manual_seed(42)
    traces = []

    def generate(latent):
        return decoder.decode(latent, view, source_history=settings.get('source_history', False)).unsqueeze(0)

    def criterion(value):
        predicted = value[0]
        pose, translation = decoder.pose(predicted)
        physical = objective(pose, translation)
        feature = (predicted[:, 2:]-decoder.clean[:, 2:]).square().mean()
        traces.append(dict(iteration=len(traces), **objective.term_record(), feature=float(feature.detach())))
        return (physical+feature).reshape(1)

    steps = settings['editing_steps'] if objective.edit else settings['reconstruction_steps']
    options = DNOOptions(num_opt_steps=steps, lr=settings['lr'], lr_warm_up_steps=settings['warmup'],
                         decorrelate_scale=settings['decorrelate_scale'], perturb_scale=0.)
    engine = DNO(generate, criterion, initial, options)
    def finite_gradient(gradient):
        assert torch.isfinite(gradient).all(), 'nonfinite gradient through HSI DDIM'
        return gradient
    engine.current_z.register_hook(finite_gradient)
    audit = destination.get('gradient_audit')
    if audit is not None:
        audit(engine.current_z, stage, 0)
    torch.cuda.synchronize(initial.device); began = time.perf_counter()
    for lo in range(0, steps, 50):
        engine(min(50, steps-lo))
        checkpoint_path = destination['path']/f'{stage}-step{engine.step_count:04d}.pt'
        with checkpoint_path.open('xb') as handle:
            torch.save(dict(latent=engine.current_z.detach().cpu(), optimizer=engine.optimizer.state_dict(),
                step=engine.step_count, initial=engine.start_z.cpu(), traces=traces,
                options=vars(options), view=view, cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(initial.device)), handle)
        if audit is not None and engine.step_count in (50, steps):
            audit(engine.current_z, stage, engine.step_count)
        last = engine.hist[-1]
        torch.cuda.synchronize(initial.device)
        peak = torch.cuda.max_memory_allocated(initial.device)/1024**3
        assert peak <= 5, peak
        print(json.dumps(dict(stage=stage, step=engine.step_count, loss=float(last['loss'].mean()),
            gradient_norm=float(last['grad_norm'].mean()), physical=traces[-1],
            seconds=time.perf_counter()-began, peak_memory_gib=peak,
            checkpoint=str(checkpoint_path))), flush=True)
    torch.cuda.synchronize(initial.device)
    with (destination['path']/(stage+'-trace.json')).open('x') as handle:
        json.dump(dict(terms=traces, seconds=time.perf_counter()-began,
            optimizer=[{k:float(row[k][0]) for k in ('step', 'lr', 'loss', 'loss_decorrelate', 'grad_norm')}
                       for row in engine.hist]), handle, indent=2)
    return engine.current_z.detach()


@torch.enable_grad()
def optimize_geometry(projector, objective, settings, destination):
    from dno import warmup_scheduler, cosine_decay_scheduler
    value = torch.zeros(len(projector.translation), 69, device=projector.translation.device, requires_grad=True)
    optimizer = torch.optim.Adam([value], lr=settings['lr'])
    trace = []
    for step in range(settings['editing_steps']):
        optimizer.zero_grad()
        pose, translation = native_increment_pose(projector, value)
        loss = objective(pose, translation)
        loss.backward()
        norm = value.grad.norm()
        trace.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), terms=objective.last_terms))
        if float(norm) == 0:
            trace[-1]['stationary'] = True
            break
        value.grad.div_(norm)
        fraction = warmup_scheduler(step, settings['warmup'])*cosine_decay_scheduler(
            step, settings['editing_steps'], settings['editing_steps'], decay_first=False)
        optimizer.param_groups[0]['lr'] = settings['lr']*fraction
        optimizer.step()
    with (destination/'geometry-trace.json').open('x') as handle: json.dump(trace, handle, indent=2)
    return native_increment_pose(projector, value.detach())


def dno_motion_probe(teacher, projector, model, sdf, info, evaluate, baseline, task, ordinal,
                     floor, length, protocol, dest):
    from .surface_edit import decode_body
    from .body_projection import smooth_body_target
    from .diagnostics import body_readout_measures
    from .continuation_outcomes import write_json
    source = projector.source
    started = time.perf_counter(); before_calls = teacher.calls
    root = Path(teacher.cfg.hsi_body_projection.protocol).resolve().parents[2]
    query, = (root/protocol['query_cache']).glob(f'lanes/*/task-{ordinal:03d}/teacher.pt')
    cache = torch.load(query, map_location=teacher.device, weights_only=False)
    decoder = HSIDDIM(teacher, cache['windows'], source, task, ordinal, protocol['method'])
    settings = protocol['method']
    destination = dict(repository=protocol['dno_repository'], path=dest)
    motions, metrics, audits = {}, {}, {}

    @torch.no_grad()
    def save_motion(name, pose, translation):
        pose, translation = pose.detach().clone(), translation.detach().clone()
        pose[projector.fixed] = source['pose'][projector.fixed]
        translation[projector.fixed] = source['translation'][projector.fixed]
        if name == 'source':
            motion = source
        else:
            motion = dict(source, pose=pose, translation=translation)
            motion['verts'], motion['joints'] = decode_body(motion, model)
            motion['verts'][projector.fixed] = source['verts'][projector.fixed]
            motion['joints'][projector.fixed] = source['joints'][projector.fixed]
        record = evaluate(motion)
        record.update(body_readout_measures(motion, source, floor, length))
        record.update(projector.measures(transforms.axis_angle_to_matrix(pose), translation))
        record.update(native_anchor_max_error_m=float((motion['joints'][:, NATIVE_ANCHORS]-
                    source['joints'][:, NATIVE_ANCHORS]).norm(dim=-1).max()),
            native_body28_mean_displacement_cm=float((motion['joints']-source['joints']).norm(dim=-1).mean()*100),
            initial_final_max_error_m=float((motion['joints'][projector.fixed]-source['joints'][projector.fixed]).abs().max()),
            object_max_error_m=float((motion['object_translation']-source['object_translation']).abs().max()))
        state = {k:v.cpu() for k,v in motion.items() if torch.is_tensor(v) and k != 'verts'}
        with (dest/(name+'.pt')).open('xb') as handle: torch.save(state, handle)
        write_json(dest/(name+'-metrics.json'), record)
        motions[name], metrics[name] = state, record
        print(json.dumps(dict(task=ordinal, stage=name, hs=record['scene_human_penetration_s_mean'],
            contact=record['contact_percent'], body_cm=record['native_body28_mean_displacement_cm'])), flush=True)
        return motion

    @torch.no_grad()
    def final_projection(name, motion):
        rotation, translation = smooth_body_target(source, motion, keep_planar=True)
        save_motion(name+'_smooth', transforms.matrix_to_axis_angle(rotation.double()).float(), translation)
        rotation, translation, audit = projector.solve(rotation, translation)
        states = audit.pop('states')
        with (dest/(name+'-projection-attempts.pt')).open('xb') as handle: torch.save(states, handle)
        pose = transforms.matrix_to_axis_angle(rotation.double()).to(source['pose'])
        pose[projector.fixed] = source['pose'][projector.fixed]
        save_motion(name+'_projected', pose, translation)
        audits[name] = audit
        assert metrics[name+'_projected']['native_anchor_max_error_m'] <= 1e-5

    save_motion('source', source['pose'], source['translation'])
    native_keys = tuple(evaluate(source))
    source_error = max(abs(float(metrics['source'][k])-float(baseline[k])) for k in native_keys)
    assert source_error <= 1e-5, source_error
    identity_pose, identity_translation = decoder.pose(decoder.clean)
    identity_error = max(float((identity_pose-source['pose']).abs().max()),
                         float((identity_translation-source['translation']).abs().max()))
    assert identity_error <= 1e-6, identity_error
    with torch.no_grad():
        latent = decoder.invert()
        with (dest/'inversion.pt').open('xb') as handle: torch.save(latent.cpu(), handle)
        for view in ('correct', 'wrong'):
            save_motion(view+'_inversion', *decoder.pose(decoder.decode(latent, view)))
    reconstruction = NativeDNOObjective(projector, model, sdf, info, baseline['scene_human_penetration_s_mean'], False)
    reconstructed = optimize_latent(decoder, reconstruction, latent, 'correct', settings, destination, 'reconstruction')
    for view in ('correct', 'wrong'):
        with torch.no_grad():
            rebuilt = save_motion(view+'_reconstruction', *decoder.pose(decoder.decode(reconstructed, view)))
        final_projection(view+'_reconstruction', rebuilt)
    editing = NativeDNOObjective(projector, model, sdf, info, baseline['scene_human_penetration_s_mean'], True)
    for view in ('correct', 'wrong'):
        result = optimize_latent(decoder, editing, reconstructed, view, settings, destination, view+'_edit')
        with torch.no_grad():
            edited = save_motion(view+'_raw', *decoder.pose(decoder.decode(result, view)))
        final_projection(view, edited)
    geometry_pose, geometry_translation = optimize_geometry(projector, editing, settings, dest)
    geometry = save_motion('geometry_raw', geometry_pose, geometry_translation)
    final_projection('geometry', geometry)
    torch.cuda.synchronize(teacher.device)
    record = dict(means=metrics, projection=audits, identity_error=identity_error,
        seconds=time.perf_counter()-started, hsi_calls=teacher.calls-before_calls,
        peak_memory_gib=torch.cuda.max_memory_allocated(teacher.device)/1024**3)
    assert record['peak_memory_gib'] <= 5, record['peak_memory_gib']
    return record


def latent_gradient_measures(terms, latent, decorrelation_weight):
    """Separate objective derivatives and their actual summed update direction."""
    gradients = {name: torch.autograd.grad(value, latent, retain_graph=True)[0]
                 for name, value in terms.items()}
    total = gradients['body']+gradients['feature']+decorrelation_weight*gradients['decorrelation']
    vectors = dict(body=gradients['body'], feature=gradients['feature'],
                   reconstruction=gradients['body']+gradients['feature'],
                   decorrelation1000=1000*gradients['decorrelation'], total=total)
    norms = {k: float(v.norm()) for k, v in vectors.items()}
    cosines = {a+'__'+b: float((vectors[a]*vectors[b]).sum()/(vectors[a].norm()*vectors[b].norm()))
               for a, b in [('body', 'feature'), ('reconstruction', 'decorrelation1000'),
                            ('reconstruction', 'total')]}
    return dict(losses={k: float(v.detach()) for k,v in terms.items()}, norms=norms,
                cosines=cosines, regularizer_to_reconstruction_norm=norms['decorrelation1000']/norms['reconstruction'],
                actual_decorrelation_weight=decorrelation_weight,
                reconstruction_projection_on_total=float((vectors['reconstruction']*total).sum()/total.square().sum()))


@torch.enable_grad()
def audit_latent_gradient(decoder, objective, latent, weight):
    from dno import noise_regularize_1d
    with torch.random.fork_rng(devices=[latent.device.index]):
        torch.manual_seed(42)
        value = latent.detach().requires_grad_(True)
        predicted = decoder.decode(value, 'correct')
        pose, translation = decoder.pose(predicted)
        terms = dict(body=objective(pose, translation),
                     feature=(predicted[:, 2:]-decoder.clean[:, 2:]).square().mean(),
                     decorrelation=noise_regularize_1d(value, dim=3).sum())
        measured = latent_gradient_measures(terms, value, weight)
        channel = value.detach().flatten(2)
        measured['latent'] = dict(mean=float(value.mean()), std=float(value.std()),
            rms=float(value.square().mean().sqrt()), max_abs=float(value.abs().max()),
            channel_mean_abs_mean=float(channel.mean(-1).abs().mean()),
            channel_std_mean=float(channel.std(-1).mean()))
        return measured


@torch.no_grad()
def record_dno_prediction(decoder, projector, model, evaluate, floor, length, dest, name, prediction):
    pose,translation=decoder.pose(prediction)
    return record_dno_pose(projector,model,evaluate,floor,length,dest,name,pose,translation)


@torch.no_grad()
def record_dno_pose(projector,model,evaluate,floor,length,dest,name,pose,translation):
    from .surface_edit import decode_body
    from .diagnostics import body_readout_measures
    from .continuation_outcomes import write_json
    source = projector.source
    motion = dict(source, pose=pose, translation=translation)
    motion['verts'], motion['joints'] = decode_body(motion, model)
    motion['verts'][projector.fixed] = source['verts'][projector.fixed]
    motion['joints'][projector.fixed] = source['joints'][projector.fixed]
    record = evaluate(motion)
    record.update(body_readout_measures(motion, source, floor, length))
    record.update(projector.measures(transforms.axis_angle_to_matrix(pose), translation))
    record.update(native_anchor_max_error_m=float((motion['joints'][:, NATIVE_ANCHORS]-source['joints'][:, NATIVE_ANCHORS]).norm(dim=-1).max()),
        native_body28_mean_displacement_cm=float((motion['joints']-source['joints']).norm(dim=-1).mean()*100),
        initial_final_max_error_m=float((motion['joints'][projector.fixed]-source['joints'][projector.fixed]).abs().max()),
        object_max_error_m=0.)
    with (dest/(name+'.pt')).open('xb') as handle:
        torch.save({k:v.cpu() for k,v in motion.items() if torch.is_tensor(v) and k!='verts'}, handle)
    write_json(dest/(name+'-metrics.json'), record)
    return motion, record


def dno_reconstruction_probe(teacher, projector, model, sdf, info, evaluate, baseline, task, ordinal,
                             floor, length, protocol, dest):
    from .continuation_outcomes import write_json
    started = time.perf_counter(); before_calls = teacher.calls
    root = Path(teacher.cfg.hsi_body_projection.protocol).resolve().parents[2]
    query, = (root/protocol['query_cache']).glob(f'lanes/*/task-{ordinal:03d}/teacher.pt')
    previous, = (root/protocol['previous_run']).glob(f'lanes/*/task-{ordinal:03d}')
    cache = torch.load(query, map_location=teacher.device, weights_only=False)
    source, settings = projector.source, protocol['method']
    decoder = HSIDDIM(teacher, cache['windows'], source, task, ordinal, settings)
    sys.path.insert(0, protocol['dno_repository'])
    objective = NativeDNOObjective(projector, model, sdf, info, baseline['scene_human_penetration_s_mean'], False)
    metrics, gradients = dict(source=dict(baseline, native_body28_mean_displacement_cm=0.)), {}

    @torch.no_grad()
    def save_prediction(name, prediction):
        _, record = record_dno_prediction(decoder, projector, model, evaluate, floor, length, dest, name, prediction)
        metrics[name] = record
        print(json.dumps(dict(task=ordinal, stage=name, body_cm=record['native_body28_mean_displacement_cm'],
                              contact=record['contact_percent'])), flush=True)

    identity_pose, identity_translation = decoder.pose(decoder.clean)
    identity_error = max(float((identity_pose-source['pose']).abs().max()),
                         float((identity_translation-source['translation']).abs().max()))
    assert identity_error <= 1e-6
    legacy = torch.load(previous/'inversion.pt', map_location=teacher.device, weights_only=False)
    aligned = decoder.invert()
    with (dest/'aligned-inversion.pt').open('xb') as handle: torch.save(aligned.cpu(), handle)
    endpoint = dict(alpha_first=float(decoder.alpha[0]), alpha_last=float(decoder.alpha[-1]),
        jump_rms=float((legacy-aligned).square().mean().sqrt()),
        jump_max_abs=float((legacy-aligned).abs().max()), aligned_rms=float(aligned.square().mean().sqrt()),
        legacy_rms=float(legacy.square().mean().sqrt()), clean_identity_error=identity_error)
    write_json(dest/'endpoint.json', endpoint)

    for name, value in [('legacy', legacy), ('aligned', aligned)]:
        for mode, kwargs in [('ddim10', {}), ('ddim100', dict(steps=100)),
                             ('source_history10', dict(source_history=True))]:
            with torch.no_grad(): prediction = decoder.decode(value, 'correct', **kwargs)
            save_prediction(name+'_'+mode, prediction)
            endpoint[name+'_'+mode+'_feature_per_window'] = (prediction[:, 2:]-decoder.clean[:, 2:]).square().mean((1,2)).cpu().tolist()

    def audit(value, stage, step):
        weight = 0. if stage.endswith('reg0') else 1000.
        measured = audit_latent_gradient(decoder, objective, value, weight)
        gradients[f'{stage}-{step}'] = measured
        write_json(dest/f'{stage}-gradient{step:04d}.json', measured)
        print(json.dumps(dict(task=ordinal, stage=stage, audit_step=step,
            regularizer_to_reconstruction_norm=measured['regularizer_to_reconstruction_norm'],
            reconstruction_total_cosine=measured['cosines']['reconstruction__total'])), flush=True)

    old = json.loads((previous/'metrics.json').read_text())
    metrics['legacy_reg1000'] = old['means']['correct_reconstruction']
    audit(legacy, 'legacy_reg1000', 0)
    for step in (50, 300):
        saved = torch.load(previous/f'reconstruction-step{step:04d}.pt', map_location=teacher.device, weights_only=False)
        audit(saved['latent'], 'legacy_reg1000', step)
    write_json(dest/'legacy-reference.json', dict(task_directory=str(previous),
        metrics=str(previous/'metrics.json'), final_motion=str(previous/'correct_reconstruction.pt')))
    for name, value, weight in [('legacy_reg0', legacy, 0.), ('aligned_reg1000', aligned, 1000.),
                                ('aligned_reg0', aligned, 0.)]:
        destination = dict(repository=protocol['dno_repository'], path=dest, gradient_audit=audit)
        result = optimize_latent(decoder, objective, value, 'correct', dict(settings, decorrelate_scale=weight), destination, name)
        with torch.no_grad(): predicted = decoder.decode(result, 'correct')
        save_prediction(name, predicted)
    torch.cuda.synchronize(teacher.device)
    record = dict(means=metrics, gradients=gradients, endpoint=endpoint, identity_error=identity_error,
        previous_directory=str(previous), seconds=time.perf_counter()-started,
        hsi_calls=teacher.calls-before_calls, peak_memory_gib=torch.cuda.max_memory_allocated(teacher.device)/1024**3)
    assert record['peak_memory_gib'] <= 5, record['peak_memory_gib']
    return record


@torch.no_grad()
def history_window_measures(decoder, prediction, joints):
    """Owned native blocks include interpolation with the next coarse sample."""
    delta = joints-decoder.source['joints']
    distance = delta.norm(dim=-1)*100
    rows = []
    for i in range(len(prediction)):
        start, stop = (0 if i==0 else 3*(2+14*i)), 3*(16+14*i)
        history_delta = prediction[i:i+1, :2]-decoder.clean[i:i+1, :2]
        history_points = decoder.dataset.denormalize_torch(prediction[i:i+1, :2, :84])-decoder.dataset.denormalize_torch(decoder.clean[i:i+1, :2, :84])
        row = dict(window=i,native_start=start,native_stop=stop,
            body_mean_cm=float(distance[start:stop].mean()),
            body_rms_cm=float(delta[start:stop].square().sum(-1).mean().sqrt()*100),
            unlocked_body_mean_cm=float(distance[max(start,6):min(stop,len(joints)-3)].mean()),
            root_mean_cm=float(distance[start:stop,0].mean()),
            feature_mse=float((prediction[i,2:]-decoder.clean[i,2:]).square().mean()),
            input_history_feature_mse=float(history_delta.square().mean()),
            input_history_joint_mean_cm=float(history_points.reshape(2,28,3).norm(dim=-1).mean()*100))
        if i:
            boundary = decoder.reframe(prediction[i-1:i,-2:],decoder.mats[i-1],decoder.mats[i])
            clean_boundary = decoder.reframe(decoder.clean[i-1:i,-2:],decoder.mats[i-1],decoder.mats[i])
            row.update(boundary_feature_mse=float((boundary-prediction[i:i+1,:2]).square().mean()),
                clean_history_feature_max_error=float((clean_boundary-decoder.clean[i:i+1,:2]).abs().max()))
        rows.append(row)
    return rows, distance


@torch.enable_grad()
def history_fit_gradients(decoder, objective, latent):
    values, derivatives = {}, {}
    for name, source_history in [('generated',False),('source',True)]:
        value = latent.detach().requires_grad_(True)
        prediction = decoder.decode(value,'correct',source_history=source_history)
        pose, translation = decoder.pose(prediction)
        body = objective(pose,translation)
        feature = (prediction[:,2:]-decoder.clean[:,2:]).square().mean()
        gradient, = torch.autograd.grad(body+feature,value)
        blocks = gradient[0,:,0].T.reshape(len(decoder.windows),14,216)
        derivatives[name] = blocks.detach()
        values[name] = dict(body=float(body.detach()),feature=float(feature.detach()),
            norm=float(gradient.norm()),window_norm=blocks.square().sum((1,2)).sqrt().cpu().tolist())
    numerator = (derivatives['generated']*derivatives['source']).sum((1,2))
    denominator = derivatives['generated'].square().sum((1,2)).sqrt()*derivatives['source'].square().sum((1,2)).sqrt()
    values['window_cosine'] = [float(n/d) if float(d)>0 else None for n,d in zip(numerator,denominator)]
    return values, derivatives


def dno_history_probe(teacher, projector, model, sdf, info, evaluate, baseline, task, ordinal,
                       floor, length, protocol, dest):
    from .continuation_outcomes import write_json
    started=time.perf_counter();before_calls=teacher.calls
    root=Path(teacher.cfg.hsi_body_projection.protocol).resolve().parents[2]
    previous,=(root/protocol['previous_run']).glob(f'lanes/*/task-{ordinal:03d}')
    query,=(root/protocol['query_cache']).glob(f'lanes/*/task-{ordinal:03d}/teacher.pt')
    cache=torch.load(query,map_location=teacher.device,weights_only=False)
    decoder=HSIDDIM(teacher,cache['windows'],projector.source,task,ordinal,protocol['method'])
    objective=NativeDNOObjective(projector,model,sdf,info,baseline['scene_human_penetration_s_mean'],False)
    initial=torch.load(previous/'aligned-inversion.pt',map_location=teacher.device,weights_only=False)
    fitted=torch.load(previous/'aligned_reg0-step0300.pt',map_location=teacher.device,weights_only=False)
    assert torch.equal(initial,fitted['initial'])
    old=json.loads((previous/'metrics.json').read_text())
    metrics=dict(source=dict(baseline,native_body28_mean_displacement_cm=0.))
    windows,gradients,paired_first_window={}, {}, {}
    source_pose,source_translation=decoder.pose(decoder.clean)
    identity_error=max(float((source_pose-projector.source['pose']).abs().max()),
                       float((source_translation-projector.source['translation']).abs().max()))
    assert identity_error<=1e-6

    def readout_pair(prefix,value):
        predictions=[]
        for suffix,source_history in [('A',False),('S',True)]:
            name=prefix+suffix
            with torch.no_grad():prediction=decoder.decode(value,'correct',source_history=source_history)
            motion,record=record_dno_prediction(decoder,projector,model,evaluate,floor,length,dest,name,prediction)
            rows,distance=history_window_measures(decoder,prediction,motion['joints'])
            with (dest/(name+'-window-errors.pt')).open('xb') as handle:
                torch.save(dict(prediction=prediction.cpu(),body28_distance_cm=distance.cpu()),handle)
            metrics[name],windows[name]=record,rows;predictions.append(prediction)
            print(json.dumps(dict(task=ordinal,stage=name,body_cm=record['native_body28_mean_displacement_cm'],
                                  contact=record['contact_percent'])),flush=True)
        paired_first_window[prefix]=float((predictions[0][0]-predictions[1][0]).abs().max())
        assert paired_first_window[prefix]==0

    readout_pair('initial_',initial)
    readout_pair('G',fitted['latent'])
    replay_error=max(abs(float(metrics[new][k])-float(old['means'][saved][k]))
        for new,saved in [('initial_A','aligned_ddim10'),('initial_S','aligned_source_history10'),('GA','aligned_reg0')]
        for k in old['means'][saved])
    assert replay_error<=1e-5,replay_error
    gradient,blocks=history_fit_gradients(decoder,objective,fitted['latent'])
    gradients['G']=gradient
    with (dest/'G-history-gradients.pt').open('xb') as handle:torch.save({k:v.cpu() for k,v in blocks.items()},handle)
    write_json(dest/'G-history-gradients.json',gradient)
    reconstructed=optimize_latent(decoder,objective,initial,'correct',protocol['method'],
        dict(repository=protocol['dno_repository'],path=dest),'source_fit')
    readout_pair('S',reconstructed)
    gradient,blocks=history_fit_gradients(decoder,objective,reconstructed)
    gradients['S']=gradient
    with (dest/'S-history-gradients.pt').open('xb') as handle:torch.save({k:v.cpu() for k,v in blocks.items()},handle)
    write_json(dest/'S-history-gradients.json',gradient)
    torch.cuda.synchronize(teacher.device)
    result=dict(means=metrics,window_measures=windows,gradients=gradients,identity_error=identity_error,
        replay_error=replay_error,first_window_decode_error=paired_first_window,previous_directory=str(previous),
        seconds=time.perf_counter()-started,hsi_calls=teacher.calls-before_calls,
        peak_memory_gib=torch.cuda.max_memory_allocated(teacher.device)/1024**3)
    assert result['peak_memory_gib']<=5,result['peak_memory_gib']
    return result


@torch.no_grad()
def constrained_motion_measures(objective,motion,object_vertices):
    from .surface_edit import HANDS,FEET,native_hand_distances
    source=objective.source;delta=motion['joints']-source['joints']
    hand_count=int(objective.hand_mask.sum());foot_count=int(objective.stance_mask.sum())
    boundary_count=int(objective.boundary_mask.sum());retained=0
    for lo in range(0,len(delta),24):
        stop=min(lo+24,len(delta))
        vertices=(motion['object_rotation'][lo:stop]@object_vertices.T).transpose(1,2)+motion['object_translation'][lo:stop,None]
        near=native_hand_distances(motion['joints'][lo:stop],vertices)<.05
        retained+=int((near&objective.hand_mask[lo:stop]).sum())
    return dict(active_hand_mean_drift_cm=float((delta[:,HANDS].norm(dim=-1)*objective.hand_mask).sum()/max(hand_count,1)*100),
        active_foot_mean_drift_cm=float((delta[:,FEET].norm(dim=-1)*objective.stance_mask).sum()/max(foot_count,1)*100),
        prefix_body_mean_drift_cm=float((delta.norm(dim=-1)*objective.boundary_mask[:,None]).sum()/max(boundary_count*28,1)*100),
        source_hand_contact_retention=retained/hand_count if hand_count else 1.,
        source_active_hand_samples=hand_count,source_stance_foot_samples=foot_count,
        source_boundary_frames=boundary_count)


@torch.enable_grad()
def constrained_gradient_audit(decoder,objective,latent,view):
    value=latent.detach().requires_grad_(True)
    predicted=decoder.decode(value,view,source_history=True)
    pose,translation=decoder.pose(predicted)
    weights=list(objective.weights);records={};gradients={}
    for i,name in enumerate(objective.term_names):
        objective.weights=[float(j==i) for j in range(len(weights))]
        loss=objective(pose,translation)
        derivative,=torch.autograd.grad(loss,value,retain_graph=True)
        gradients[name]=derivative.detach()
        records[name]=dict(value=float(loss.detach()),gradient_norm=float(derivative.norm()),weight=weights[i])
    objective.weights=weights
    feature=(predicted[:,2:]-decoder.clean[:,2:]).square().mean()
    derivative,=torch.autograd.grad(feature,value)
    gradients['feature']=derivative.detach()
    records['feature']=dict(value=float(feature.detach()),gradient_norm=float(derivative.norm()),weight=1.)
    physical=sum(weights[i]*gradients[name] for i,name in enumerate(objective.term_names))
    constraints=sum(gradients[name] for name in objective.term_names if name not in ('scene','body'))
    scene=gradients['scene'];norm_product=constraints.norm()*scene.norm()
    return dict(terms=records,total_gradient_norm=float((physical+derivative).norm()),
        constraint_gradient_norm=float(constraints.norm()),scene_gradient_norm=float(scene.norm()),
        constraint_scene_cosine=float((constraints*scene).sum()/norm_product) if float(norm_product)>0 else None),gradients


def dno_constrained_probe(teacher,projector,model,sdf,info,evaluate,baseline,task,ordinal,
                           floor,length,protocol,dest):
    from .continuation_outcomes import write_json
    started=time.perf_counter();before_calls=teacher.calls
    root=Path(teacher.cfg.hsi_body_projection.protocol).resolve().parents[2]
    previous,=(root/protocol['previous_run']).glob(f'lanes/*/task-{ordinal:03d}')
    query,=(root/protocol['query_cache']).glob(f'lanes/*/task-{ordinal:03d}/teacher.pt')
    cache=torch.load(query,map_location=teacher.device,weights_only=False)
    decoder=HSIDDIM(teacher,cache['windows'],projector.source,task,ordinal,protocol['method'])
    object_vertices=teacher.dataset.obj_rest_verts[task['object_name']]
    objective=ConstrainedDNOObjective(projector,model,sdf,info,baseline['scene_human_penetration_s_mean'],floor,object_vertices)
    initial=torch.load(previous/'source_fit-step0300.pt',map_location=teacher.device,weights_only=False)['latent']
    source=projector.source
    source_metrics=dict(baseline,native_body28_mean_displacement_cm=0.,**constrained_motion_measures(objective,source,object_vertices))
    metrics=dict(source=source_metrics);windows={};audits={}
    with (dest/'source.pt').open('xb') as handle:
        torch.save({k:v.cpu() for k,v in source.items() if torch.is_tensor(v) and k!='verts'},handle)
    objective(source['pose'],source['translation'])
    source_terms=objective.term_record()
    source_hs_error=abs(source_terms['scene']*objective.scene_scale-baseline['scene_human_penetration_s_mean'])
    assert source_hs_error<=1e-5,source_hs_error
    write_json(dest/'physical-source.json',dict(terms=source_terms,native_hs_error=source_hs_error,
        source_fk_reference_max_error_m=objective.source_fk_reference_max_error_m,
        active_hand_samples=int(objective.hand_mask.sum()),stance_foot_samples=int(objective.stance_mask.sum()),
        boundary_frames=int(objective.boundary_mask.sum()),scales_m=dict(body=.05,hand=.01,stance=.005,boundary=.01),velocity_scale_m_s=.1))

    def save(name,value,view):
        with torch.no_grad():prediction=decoder.decode(value,view,source_history=True)
        def augmented(motion):return dict(evaluate(motion),**constrained_motion_measures(objective,motion,object_vertices))
        motion,record=record_dno_prediction(decoder,projector,model,augmented,floor,length,dest,name,prediction)
        rows,distance=history_window_measures(decoder,prediction,motion['joints'])
        windows[name]=rows;metrics[name]=record
        with (dest/(name+'-window-errors.pt')).open('xb') as handle:
            torch.save(dict(prediction=prediction.cpu(),body28_distance_cm=distance.cpu()),handle)
        print(json.dumps(dict(task=ordinal,stage=name,hs=record['scene_human_penetration_s_mean'],
            contact=record['contact_percent'],support=record['source_floor_support_fraction'],
            active_hand_cm=record['active_hand_mean_drift_cm'],prefix_cm=record['prefix_body_mean_drift_cm'])),flush=True)

    def audit(name,value,view):
        record,gradients=constrained_gradient_audit(decoder,objective,value,view)
        audits[name]=record
        write_json(dest/(name+'-gradient.json'),record)
        with (dest/(name+'-gradients.pt')).open('xb') as handle:torch.save({k:v.cpu() for k,v in gradients.items()},handle)
        print(json.dumps(dict(task=ordinal,stage=name,gradient_norm=record['total_gradient_norm'],
            constraint_gradient_norm=record['constraint_gradient_norm'],scene_gradient_norm=record['scene_gradient_norm'])),flush=True)

    save('C0',initial,'correct');save('W0',initial,'wrong')
    old=json.loads((previous/'metrics.json').read_text())['means']['SS']
    replay_error=max(abs(float(metrics['C0'][k])-float(old[k])) for k in old)
    assert replay_error<=1e-5,replay_error
    audit('C0',initial,'correct');audit('W0',initial,'wrong')
    for name,view,scene_weight in [('C','correct',1.),('W','wrong',1.),('Q','correct',0.)]:
        objective.weights[-1]=scene_weight
        result=optimize_latent(decoder,objective,initial,view,protocol['method'],
            dict(repository=protocol['dno_repository'],path=dest),name)
        save(name,result,view);audit(name,result,view)
    objective.weights[-1]=1.
    pose,translation=optimize_geometry(projector,objective,protocol['method'],dest)
    def augmented(motion):return dict(evaluate(motion),**constrained_motion_measures(objective,motion,object_vertices))
    _,metrics['G']=record_dno_pose(projector,model,augmented,floor,length,dest,'G',pose,translation)
    torch.cuda.synchronize(teacher.device)
    record=dict(means=metrics,window_measures=windows,gradients=audits,previous_directory=str(previous),
        replay_error=replay_error,source_physical_terms=source_terms,source_hs_error=source_hs_error,
        source_fk_reference_max_error_m=objective.source_fk_reference_max_error_m,
        seconds=time.perf_counter()-started,hsi_calls=teacher.calls-before_calls,
        peak_memory_gib=torch.cuda.max_memory_allocated(teacher.device)/1024**3)
    assert record['peak_memory_gib']<=5,record['peak_memory_gib']
    return record


def summarize_constrained_editing(run_root,task_manifest,device='cuda:0'):
    from .scene_calibration import paired_local_metrics
    from .continuation_outcomes import write_json
    run_root=Path(run_root);tasks=json.loads(Path(task_manifest).read_text())['tasks']
    records=[json.loads(p.read_text()) for p in run_root.glob('lanes/*/task-*/metrics.json')]
    assert len(records)==len(tasks) and {r['task'] for r in records}=={t['canonical_ordinal'] for t in tasks}
    arms=list(records[0]['means']);keys=list(records[0]['means']['C']);scenes=sorted({r['scene'] for r in records})
    by_task={a:{str(r['task']):{k:r['means'][a][k] for k in keys} for r in records} for a in arms}
    by_scene={a:{s:{k:sum(r['means'][a][k] for r in records if r['scene']==s)/sum(r['scene']==s for r in records)
        for k in keys} for s in scenes} for a in arms}
    means={a:{k:sum(r['means'][a][k] for r in records)/len(records) for k in keys} for a in arms}
    pairs=[(a,'source') for a in arms if a!='source']+[('C','W'),('C','Q'),('C','G'),('C','C0'),('W','W0'),('Q','C0')]
    contrasts={a+'__minus__'+b:{unit:paired_local_metrics(values[b],values[a],device)
        for unit,values in [('task',by_task),('scene',by_scene)]} for a,b in pairs}
    changes={unit:{view:{name:{k:values[view][name][k]-values[view+'0'][name][k] for k in keys}
        for name in values[view]} for view in ('C','W')} for unit,values in [('task',by_task),('scene',by_scene)]}
    increment={unit:paired_local_metrics(values['W'],values['C'],device) for unit,values in changes.items()}
    protections={}
    for a in ('C','W','Q','G'):
        protections[a]={}
        for r in records:
            m,b=r['means'][a],r['means']['source']
            protections[a][str(r['task'])]=dict(contact=m['contact_percent']>=b['contact_percent']-.002,
                support=m['source_floor_support_fraction']>=b['source_floor_support_fraction']-.002,
                foot_sliding=m['foot_sliding']<=b['foot_sliding']+.01,hand=m['active_hand_mean_drift_cm']<=1.,
                stance=m['active_foot_mean_drift_cm']<=.5,prefix=m['prefix_body_mean_drift_cm']<=1.,
                seam_velocity=m['correction_seam_speed_mean_cm_s']<=10.,mean_velocity=m['correction_speed_mean_cm_s']<=10.,
                max_velocity=m['correction_speed_max_cm_s']<=30.,root=m['root_max_change_m']<=.1,
                angle=m['rotation_max_change_deg']<=20.,body=m['native_body28_mean_displacement_cm']<=5.,
                fixed=m['object_max_error_m']==0 and m['initial_final_max_error_m']==0)
    hs='scene_human_penetration_s_mean'
    conditions=dict(source_improvement=means['C'][hs]<=.99*means['source'][hs],
        correct_scene=means['C'][hs]<=means['W'][hs]-.005*means['source'][hs],
        scene_increment=increment['task'][hs]['delta']<0,
        scene_objective=means['C'][hs]<means['Q'][hs],geometry_reference=means['C'][hs]<means['G'][hs],
        protection=all(all(v.values()) for v in protections['C'].values()))
    remaining27={a+'__minus__'+b:paired_local_metrics({k:v for k,v in by_task[b].items() if k!='375'},
        {k:v for k,v in by_task[a].items() if k!='375'},device) for a,b in pairs}
    summary=dict(tasks=len(records),scenes=len(scenes),means=means,contrasts=contrasts,edit_increment_contrast=increment,
        protections=protections,protection_pass_counts={a:sum(all(v.values()) for v in rows.values()) for a,rows in protections.items()},
        conditions=conditions,utility=all(conditions.values()),remaining27=remaining27,
        task375={a:by_task[a]['375'] for a in arms},hsi_calls=sum(r['hsi_calls'] for r in records),
        task_seconds_sum=sum(r['seconds'] for r in records),peak_memory_gib=max(r['peak_memory_gib'] for r in records),
        test_set_development=True,timing_comparison_valid=False)
    output=run_root/'analysis';output.mkdir()
    write_json(output/'summary.json',summary);write_json(output/'records.json',records)
    for arm in arms:
        for unit,values in [('task',by_task[arm]),('scene',by_scene[arm])]:write_json(output/f'{arm}-{unit}.json',dict(metrics=values))
    return summary


def summarize_history_fitting(run_root,task_manifest,device='cuda:0'):
    from .scene_calibration import paired_local_metrics
    from .continuation_outcomes import write_json
    run_root=Path(run_root)
    tasks=json.loads(Path(task_manifest).read_text())['tasks']
    records=[json.loads(p.read_text()) for p in run_root.glob('lanes/*/task-*/metrics.json')]
    assert len(records)==len(tasks) and {r['task'] for r in records}=={t['canonical_ordinal'] for t in tasks}
    arms=list(records[0]['means']);keys=list(records[0]['means']['GA'])
    by_task={a:{str(r['task']):{k:r['means'][a][k] for k in keys} for r in records} for a in arms}
    scenes=sorted({r['scene'] for r in records})
    by_scene={a:{s:{k:sum(r['means'][a][k] for r in records if r['scene']==s)/sum(r['scene']==s for r in records)
                  for k in keys} for s in scenes} for a in arms}
    means={a:{k:sum(r['means'][a][k] for r in records)/len(records) for k in keys} for a in arms}
    pairs=[('SS','GA'),('SA','SS'),('GS','GA'),('SA','GA'),('SS','GS'),('initial_S','initial_A')]
    contrasts={a+'__minus__'+b:{unit:paired_local_metrics(values[b],values[a],device)
        for unit,values in [('task',by_task),('scene',by_scene)]} for a,b in pairs}
    changes={unit:{fit:{name:{k:values[fit+'A'][name][k]-values[fit+'S'][name][k] for k in keys}
        for name in values[fit+'A']} for fit in ('G','S')} for unit,values in [('task',by_task),('scene',by_scene)]}
    interaction={unit:paired_local_metrics(values['G'],values['S'],device) for unit,values in changes.items()}
    gates={a:{str(r['task']):dict(body=r['means'][a]['native_body28_mean_displacement_cm']<=1.,
        contact=r['means'][a]['contact_percent']>=r['means']['source']['contact_percent']-.002,
        support=r['means'][a]['source_floor_support_fraction']>=r['means']['source']['source_floor_support_fraction']-.002,
        foot_sliding=r['means'][a]['foot_sliding']<=r['means']['source']['foot_sliding']+.01)
        for r in records} for a in ('GA','GS','SS','SA')}
    remaining27={a+'__minus__'+b:paired_local_metrics({k:v for k,v in by_task[b].items() if k!='375'},
        {k:v for k,v in by_task[a].items() if k!='375'},device) for a,b in pairs}
    summary=dict(tasks=len(records),scenes=len(scenes),means=means,contrasts=contrasts,interaction=interaction,
        reconstruction=gates,pass_counts={a:sum(all(v.values()) for v in rows.values()) for a,rows in gates.items()},
        remaining27=remaining27,task375={a:by_task[a]['375'] for a in arms},
        hsi_calls=sum(r['hsi_calls'] for r in records),task_seconds_sum=sum(r['seconds'] for r in records),
        peak_memory_gib=max(r['peak_memory_gib'] for r in records),test_set_development=True,timing_comparison_valid=False)
    output=run_root/'analysis';output.mkdir()
    write_json(output/'summary.json',summary);write_json(output/'records.json',records)
    for arm in arms:
        for unit,values in [('task',by_task[arm]),('scene',by_scene[arm])]:write_json(output/f'{arm}-{unit}.json',dict(metrics=values))
    return summary


def summarize_reconstruction(run_root, task_manifest, device='cuda:0'):
    from .scene_calibration import paired_local_metrics
    from .continuation_outcomes import write_json
    run_root = Path(run_root)
    tasks = json.loads(Path(task_manifest).read_text())['tasks']
    records = [json.loads(p.read_text()) for p in run_root.glob('lanes/*/task-*/metrics.json')]
    assert len(records)==len(tasks) and {r['task'] for r in records}=={t['canonical_ordinal'] for t in tasks}
    arms = list(records[0]['means'])
    keys = sorted(set.intersection(*(set(r['means'][a]) for r in records for a in arms)))
    by_task = {a:{str(r['task']):{k:r['means'][a][k] for k in keys} for r in records} for a in arms}
    scenes = sorted({r['scene'] for r in records})
    by_scene = {a:{s:{k:sum(r['means'][a][k] for r in records if r['scene']==s)/
        sum(r['scene']==s for r in records) for k in keys} for s in scenes} for a in arms}
    means = {a:{k:sum(r['means'][a][k] for r in records)/len(records) for k in keys} for a in arms}
    pairs = [('legacy_reg0','legacy_reg1000'), ('aligned_reg1000','legacy_reg1000'),
        ('aligned_reg0','legacy_reg0'), ('aligned_reg0','aligned_reg1000'),
        ('aligned_reg0','legacy_reg1000')]
    pairs += [(f'{endpoint}_{variant}',f'{endpoint}_ddim10') for endpoint in ('legacy','aligned')
              for variant in ('ddim100','source_history10')]
    contrasts = {a+'__minus__'+b:{unit:paired_local_metrics(values[b],values[a],device)
        for unit,values in [('task',by_task),('scene',by_scene)]} for a,b in pairs}
    changes = {unit:{end:{name:{k:values[end+'_reg0'][name][k]-values[end+'_reg1000'][name][k]
        for k in keys} for name in values[end+'_reg0']} for end in ('legacy','aligned')}
        for unit,values in [('task',by_task),('scene',by_scene)]}
    interaction = {unit:paired_local_metrics(values['legacy'],values['aligned'],device) for unit,values in changes.items()}
    gates = {a:{str(r['task']):dict(body=r['means'][a]['native_body28_mean_displacement_cm']<=1.,
        contact=r['means'][a]['contact_percent']>=r['means']['source']['contact_percent']-.002,
        support=r['means'][a]['source_floor_support_fraction']>=r['means']['source']['source_floor_support_fraction']-.002,
        foot_sliding=r['means'][a]['foot_sliding']<=r['means']['source']['foot_sliding']+.01)
        for r in records} for a in ('legacy_reg1000','legacy_reg0','aligned_reg1000','aligned_reg0')}
    remaining27 = {a+'__minus__'+b:paired_local_metrics({k:v for k,v in by_task[b].items() if k!='375'},
        {k:v for k,v in by_task[a].items() if k!='375'},device) for a,b in pairs}
    summary = dict(tasks=len(records),scenes=len(scenes),means=means,contrasts=contrasts,interaction=interaction,
        reconstruction=gates,pass_counts={a:sum(all(v.values()) for v in rows.values()) for a,rows in gates.items()},
        remaining27=remaining27,task375={a:by_task[a]['375'] for a in arms},
        hsi_calls=sum(r['hsi_calls'] for r in records),task_seconds_sum=sum(r['seconds'] for r in records),
        peak_memory_gib=max(r['peak_memory_gib'] for r in records),test_set_development=True,
        timing_comparison_valid=False,training_allowed=False)
    output=run_root/'analysis';output.mkdir()
    write_json(output/'summary.json',summary);write_json(output/'records.json',records)
    for arm in arms:
        for unit,values in [('task',by_task[arm]),('scene',by_scene[arm])]:
            write_json(output/f'{arm}-{unit}.json',dict(metrics=values))
    return summary


def summarize_dno(run_root, task_manifest, device='cuda:7'):
    from .scene_calibration import paired_local_metrics
    from .continuation_outcomes import write_json
    run_root = Path(run_root)
    tasks = json.loads(Path(task_manifest).read_text())['tasks']
    records = [json.loads(p.read_text()) for p in run_root.glob('lanes/*/task-*/metrics.json')]
    assert len(records) == len(tasks)
    assert {r['task'] for r in records} == {t['canonical_ordinal'] for t in tasks}
    arms = list(records[0]['means'])
    by_task = {a:{str(r['task']):r['means'][a] for r in records} for a in arms}
    scenes = sorted({r['scene'] for r in records})
    by_scene = {a:{s:{k:sum(r['means'][a][k] for r in records if r['scene']==s)/
        sum(r['scene']==s for r in records) for k in by_task[a][str(records[0]['task'])]}
        for s in scenes} for a in arms}
    means = {a:{k:sum(float(r['means'][a][k]) for r in records)/len(records)
        for k in rkeys} for a in arms for rkeys in [records[0]['means'][a]]}
    pairs = [(a, 'source') for a in arms if a != 'source']
    pairs += [('correct_projected', 'wrong_projected'), ('correct_projected', 'geometry_projected'),
        ('correct_raw', 'correct_reconstruction'), ('wrong_raw', 'wrong_reconstruction'),
        ('correct_reconstruction', 'wrong_reconstruction'), ('correct_projected', 'correct_raw'),
        ('wrong_projected', 'wrong_raw'),
        ('correct_projected', 'correct_reconstruction_projected'),
        ('wrong_projected', 'wrong_reconstruction_projected')]
    contrasts = {a+'__minus__'+b:{unit:paired_local_metrics(data[b], data[a], device)
        for unit, data in [('task', by_task), ('scene', by_scene)]} for a,b in pairs}
    changes = {view:{unit:{name:{k:values[view+'_projected'][name][k]-
        values[view+'_reconstruction_projected'][name][k] for k in values[view+'_projected'][name]}
        for name in values[view+'_projected']} for unit,values in [('task',by_task),('scene',by_scene)]}
        for view in ('correct','wrong')}
    incremental_contrast = {unit:paired_local_metrics(changes['wrong'][unit], changes['correct'][unit], device)
                            for unit in ('task','scene')}
    hs = 'scene_human_penetration_s_mean'; oskey = 'scene_obj_penetration_s_mean'
    source, correct, wrong = (means[a] for a in ('source', 'correct_projected', 'wrong_projected'))
    conditions = dict(improves_source=correct[hs] <= .99*source[hs],
        correct_scene_benefit=correct[hs] <= wrong[hs]-.005*source[hs],
        contact=correct['contact_percent'] >= source['contact_percent']-.002,
        foot_sliding=correct['foot_sliding'] <= source['foot_sliding']+.01,
        object_scene=correct[oskey] <= 1.01*source[oskey],
        completion=correct['completed'] >= source['completed'],
        support=correct['source_floor_support_fraction'] >= source['source_floor_support_fraction']-.002,
        anchors=max(r['means'][a]['native_anchor_max_error_m'] for r in records
                    for a in ('correct_projected', 'wrong_projected', 'geometry_projected')) <= 1e-5)
    reconstruction = {}
    for row in records:
        a, b = row['means']['correct_reconstruction'], row['means']['source']
        reconstruction[str(row['task'])] = dict(body=a['native_body28_mean_displacement_cm'] <= 1.,
            contact=a['contact_percent'] >= b['contact_percent']-.002,
            support=a['source_floor_support_fraction'] >= b['source_floor_support_fraction']-.002,
            foot_sliding=a['foot_sliding'] <= b['foot_sliding']+.01)
    coverage, sensitivity = {}, {}
    for reference in ('source', 'wrong_projected', 'geometry_projected'):
        delta = [r['means']['correct_projected'][hs]-r['means'][reference][hs] for r in records]
        coverage[reference] = dict(improved=sum(d<0 for d in delta), worsened=sum(d>0 for d in delta), equal=sum(d==0 for d in delta))
        sensitivity[reference] = paired_local_metrics(
            {k:v for k,v in by_task[reference].items() if k!='375'},
            {k:v for k,v in by_task['correct_projected'].items() if k!='375'}, device)
    summary = dict(tasks=len(records), scenes=len(scenes), windows=sum(r['windows'] for r in records),
        means=means, contrasts=contrasts, edit_increment_contrast=incremental_contrast,
        conditions=conditions, utility=all(conditions.values()),
        reconstruction=reconstruction, reconstruction_pass_count=sum(all(x.values()) for x in reconstruction.values()),
        coverage=coverage, remaining27=sensitivity, hsi_calls=sum(r['hsi_calls'] for r in records),
        peak_memory_gib=max(r['peak_memory_gib'] for r in records),
        task_seconds_sum=sum(r['seconds'] for r in records), resource_contention=True,
        timing_comparison_valid=False, training_allowed=False, test_set_development=True)
    output = run_root/'analysis'; output.mkdir()
    for arm in arms:
        for unit, values in [('task', by_task[arm]), ('scene', by_scene[arm])]:
            write_json(output/f'{arm}-{unit}.json', dict(metrics=values))
    write_json(output/'summary.json', summary); write_json(output/'records.json', records)
    return summary
