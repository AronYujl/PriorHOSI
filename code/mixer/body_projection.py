"""Native skeletal projection of cached body targets with fixed hand/foot tracks."""
import math
import time

import torch
from pytorch3d import transforms

from .kinematic_composition import _PARENTS_22, _PARENTS_24, _forward_kinematics
from .hsi_motion_target import planar_yaw
from .surface_edit import spline_basis, edit_envelope, yaw_matrix

ANCHORS = (22, 23, 7, 8, 10, 11)
NATIVE_ANCHORS = (24, 26, 7, 8, 10, 11)


def native_rest_offsets(model, betas):
    shaped = model.v_template+torch.einsum('vci,i->vc',model.shapedirs[...,:len(betas)],betas)
    joints = model.J_regressor@shaped
    offsets = torch.cat((joints[:1],joints[1:22]-joints[list(_PARENTS_22[1:])],
                         (joints[25]-joints[20])[None],(joints[40]-joints[21])[None]))
    return offsets


def smooth_body_target(source, full, keep_planar=False):
    original = transforms.axis_angle_to_matrix(source['pose'])
    predicted = transforms.axis_angle_to_matrix(full['pose'])
    if not keep_planar:
        yaw = planar_yaw(predicted[:,0]@original[:,0].transpose(-1,-2))
        predicted[:,0] = yaw_matrix(-yaw)@predicted[:,0]
    angular = transforms.matrix_to_axis_angle(original.transpose(-1,-2)@predicted)
    bound = math.radians(20)
    angular = angular*torch.clamp(bound/angular.norm(dim=-1,keepdim=True),max=1)
    displacement = torch.zeros_like(source['translation'])
    if keep_planar:
        displacement = (full['translation']-source['translation']).clamp(-.1,.1)
    else:
        displacement[:,1] = (full['joints'][:,0,1]-source['joints'][:,0,1]).clamp(-.1,.1)
    raw = torch.cat((displacement,angular.flatten(1)),dim=-1)
    basis = spline_basis(len(raw),raw.device)
    smooth = (basis.double()@(torch.linalg.pinv(basis.double())@raw.double())).to(raw.dtype)
    envelope = edit_envelope(len(raw),raw.device)[:,None]
    shift = smooth[:,:3].clamp(-.1,.1)*envelope
    angle = smooth[:,3:].reshape(-1,22,3)
    angle = angle*torch.clamp(bound/angle.norm(dim=-1,keepdim=True),max=1)*envelope[:,:,None]
    return original@transforms.axis_angle_to_matrix(angle), source['translation']+shift


class NativeBodyProjection:
    position_scale = .05
    angle_scale = math.radians(10)

    def __init__(self, source, offsets):
        self.source = source
        self.offsets = offsets
        self.rotation = transforms.axis_angle_to_matrix(source['pose'])
        self.translation = source['translation']
        self.fixed = edit_envelope(len(self.translation),self.translation.device)==0
        ancestry = torch.zeros(6,22,dtype=torch.bool,device=offsets.device)
        for i,anchor in enumerate(ANCHORS):
            parent = _PARENTS_24[anchor]
            while parent>=0:
                ancestry[i,parent] = True
                parent = _PARENTS_22[parent]
        self.ancestry = ancestry
        _,self.points = self.forward(self.rotation,self.translation)
        self.reference = self.points[:,ANCHORS]
        self.seams = torch.arange(16,len(self.translation)//3,14,device=offsets.device)*3

    def forward(self,rotation,translation):
        offsets = self.offsets[None].expand(len(rotation),-1,-1).clone()
        offsets[:,0] = offsets[:,0]+translation
        return _forward_kinematics(rotation,offsets)

    def jacobian(self,global_rotation,points):
        lever = points[:,ANCHORS,None]-points[:,None,:22]
        axes = global_rotation.transpose(-1,-2)[:,None]
        angular = torch.cross(axes.expand(-1,6,-1,-1,-1),lever[:,:,:,None].expand(-1,-1,-1,3,-1),dim=-1)
        angular = angular*self.ancestry[None,:,:,None,None]*self.angle_scale
        angular = angular.permute(0,1,4,2,3).reshape(len(points),18,66)
        position = torch.eye(3,device=points.device,dtype=points.dtype)[None,None].expand(len(points),6,-1,-1)
        return torch.cat((position.reshape(len(points),18,3)*self.position_scale,angular),-1)

    def inverse(self,jacobian):
        jacobian = jacobian.double()
        return jacobian.transpose(-1,-2)@torch.linalg.pinv(jacobian@jacobian.transpose(-1,-2),hermitian=True,rtol=1e-10)

    def update(self,rotation,translation,direction):
        shift = direction[:,:3]*self.position_scale
        angle = direction[:,3:].reshape(-1,22,3)*self.angle_scale
        ratio = torch.minimum(torch.clamp(.005/shift.norm(dim=-1),max=1),
                              torch.clamp(math.radians(2)/angle.norm(dim=-1).amax(-1),max=1))
        ratio = torch.where(self.fixed,torch.zeros_like(ratio),ratio)
        new_rotation = rotation@transforms.axis_angle_to_matrix(angle*ratio[:,None,None])
        new_translation = translation+shift*ratio[:,None]
        return (torch.where(self.fixed[:,None,None,None],self.rotation,new_rotation),
                torch.where(self.fixed[:,None],self.translation,new_translation))

    def restore(self,rotation,translation,iterations):
        for _ in range(iterations):
            global_rotation,points = self.forward(rotation,translation)
            jacobian = self.jacobian(global_rotation,points)
            error = (self.reference-points[:,ANCHORS]).flatten(1)
            delta = (self.inverse(jacobian)@error.double()[...,None]).squeeze(-1).to(rotation.dtype)
            rotation,translation = self.update(rotation,translation,delta)
        return rotation,translation

    def measures(self,rotation,translation):
        _,points = self.forward(rotation,translation)
        change = points-self.points
        speed = (change[1:]-change[:-1]).norm(dim=-1)*30
        angle = transforms.matrix_to_axis_angle(self.rotation.transpose(-1,-2)@rotation).norm(dim=-1)
        return dict(anchor_max_error_m=float((points[:,ANCHORS]-self.reference).norm(dim=-1).max()),
            root_max_change_m=float((translation-self.translation).norm(dim=-1).max()),
            rotation_max_change_deg=float(angle.max()*180/math.pi),
            correction_speed_max_cm_s=float(speed.max()*100),
            correction_speed_mean_cm_s=float(speed.mean()*100),
            correction_seam_speed_mean_cm_s=float(speed[self.seams-1].mean()*100),
            body_displacement_cm=float(change.norm(dim=-1).mean()*100))

    @torch.no_grad()
    def solve(self,target_rotation,target_translation):
        device = self.translation.device
        torch.cuda.synchronize(device);started=time.perf_counter()
        rotation,translation = self.rotation.clone(),self.translation.clone()
        trace = []
        for iteration in range(80):
            relative = self.rotation.transpose(-1,-2)@rotation
            angular = transforms.matrix_to_axis_angle(rotation.transpose(-1,-2)@target_rotation)/self.angle_scale
            shift = (target_translation-translation)/self.position_scale
            displacement = (translation-self.translation)/self.position_scale
            neighbor = transforms.matrix_to_axis_angle(relative[:-1].transpose(-1,-2)@relative[1:])/self.angle_scale
            angular[:-1] += 9*neighbor; angular[1:] -= 9*neighbor
            difference = displacement[1:]-displacement[:-1]
            shift[:-1] += 9*difference; shift[1:] -= 9*difference
            direction = torch.cat((shift,angular.flatten(1)),-1)/37
            global_rotation,points = self.forward(rotation,translation)
            jacobian = self.jacobian(global_rotation,points)
            normal = (self.inverse(jacobian)@(jacobian.double()@direction.double()[...,None])).squeeze(-1)
            direction = direction-normal.to(direction.dtype)
            rotation,translation = self.update(rotation,translation,direction)
            rotation,translation = self.restore(rotation,translation,5)
            if iteration%10==0 or iteration==79:
                trace.append(dict(iteration=iteration+1,**self.measures(rotation,translation)))
        fitted_rotation,fitted_translation = rotation,translation
        residual = transforms.matrix_to_axis_angle(self.rotation.transpose(-1,-2)@fitted_rotation)
        attempts,states = [],[]
        for amplitude in (1.,.5,.25,.125,.0625,.03125,.015625,0.):
            if amplitude==0:
                rotation,translation = self.rotation,self.translation
            else:
                rotation = self.rotation@transforms.axis_angle_to_matrix(residual*amplitude)
                translation = self.translation+(fitted_translation-self.translation)*amplitude
                rotation,translation = self.restore(rotation,translation,20)
            measures = self.measures(rotation,translation)
            limits = dict(anchor_max_error_m=1e-6,root_max_change_m=.1,rotation_max_change_deg=20,
                correction_speed_max_cm_s=30,correction_speed_mean_cm_s=10,correction_seam_speed_mean_cm_s=10)
            failures = [k for k,limit in limits.items() if measures[k]>limit]
            attempts.append(dict(amplitude=amplitude,failures=failures,**measures))
            states.append(dict(amplitude=amplitude,rotation=rotation.cpu(),translation=translation.cpu()))
            if not failures:break
        torch.cuda.synchronize(device)
        audit = dict(seconds=time.perf_counter()-started,iterations=80,trace=trace,attempts=attempts,
                     amplitude=amplitude,final=measures,states=states)
        return rotation,translation,audit


def tangent_body_direction(projection, direction):
    """The source tangent projector, including its fixed temporal boundary."""
    global_rotation, points = projection.forward(projection.rotation, projection.translation)
    jacobian = projection.jacobian(global_rotation, points).double()
    value = direction.double().clone()
    value[projection.fixed] = 0
    normal = (projection.inverse(jacobian)@(jacobian@value[..., None])).squeeze(-1)
    tangent = value-normal
    return tangent, jacobian


def body_direction_velocity(projection, direction):
    """Analytic velocity of all24 FK points for a normalized right-local tangent."""
    rotation, points = projection.forward(projection.rotation, projection.translation)
    rotation, points = rotation.double(), points.double()
    linear = [direction[:, :3]*projection.position_scale]
    angular = [(rotation[:, 0]@direction[:, 3:6, None]).squeeze(-1)*projection.angle_scale]
    local = direction[:, 3:].reshape(-1, 22, 3)*projection.angle_scale
    for joint in range(1, 24):
        parent = _PARENTS_24[joint]
        linear.append(linear[parent]+torch.cross(angular[parent], points[:, joint]-points[:, parent], dim=-1))
        omega = angular[parent]
        if joint < 22:
            omega = omega+(rotation[:, joint]@local[:, joint, :, None]).squeeze(-1)
        angular.append(omega)
    return torch.stack(linear, dim=1)


def normalize_body_directions(projection, directions):
    """Equal linearized body RMS followed by one common kinematic scale."""
    normalized, audits, limits = {}, {}, []
    for name, direction in directions.items():
        velocity = body_direction_velocity(projection, direction)
        rms = velocity.square().sum(-1).mean().sqrt()
        value = direction*(.001/rms) if float(rms) > 0 else direction.clone()
        motion = body_direction_velocity(projection, value)
        speed = (motion[1:]-motion[:-1]).norm(dim=-1)*30
        ratios = torch.stack((
            (value[:, :3]*projection.position_scale).norm(dim=-1).max()/.005,
            (value[:, 3:].reshape(-1,22,3)*projection.angle_scale).norm(dim=-1).max()/math.radians(2),
            speed.max()/.3, speed.mean()/.1, speed[projection.seams-1].mean()/.1,
        ))
        limits.append(ratios.max())
        normalized[name] = value
        audits[name] = dict(projected_body_rms_m=float(rms), zero_direction=float(rms)==0,
                           unit_limit_ratios=ratios.tolist())
    scale = float(torch.stack(limits).max().clamp_min(1).reciprocal())
    for name in normalized:
        normalized[name] = normalized[name]*scale
        audits[name]['achieved_linear_body_rms_m'] = float(body_direction_velocity(projection,normalized[name]).square().sum(-1).mean().sqrt())
    return normalized, dict(common_scale=scale,directions=audits)


def native_increment_pose(projection, direction, start=0, stop=None):
    """Identity-preserving native pose for a tangent increment, with gradients."""
    source = projection.source
    stop = len(source['pose']) if stop is None else stop
    value = direction*~projection.fixed[start:stop,None]
    rotation = projection.rotation[start:stop].double()
    changed = rotation@transforms.axis_angle_to_matrix(value[:,3:].double().reshape(-1,22,3)*projection.angle_scale)
    difference = transforms.matrix_to_axis_angle(changed)-transforms.matrix_to_axis_angle(rotation)
    pose = source['pose'][start:stop]+difference.to(source['pose'])
    translation = source['translation'][start:stop]+value[:,:3].to(source['translation'])*projection.position_scale
    return pose,translation


class NativeSceneDifferential:
    """Native human-scene penetration and its69-coordinate source derivative."""
    def __init__(self, projection, model, sdf, info):
        self.projection, self.model = projection, model
        device = projection.translation.device
        self.sdf = torch.as_tensor(sdf,device=device,dtype=torch.float32)[None,None]
        self.centroid = torch.as_tensor(info['centroid'],device=device,dtype=torch.float32)
        self.extent = torch.as_tensor(info['extents'],device=device,dtype=torch.float32).max()

    def frame_sums(self, vertices):
        # Preserve the native evaluator's normalization, ordering and reduction.
        normalized = (vertices.float()-self.centroid.reshape(1,1,3))/(self.extent/2)
        values = torch.nn.functional.grid_sample(self.sdf,
            normalized[:,:,[2,1,0]].reshape(1,-1,1,1,3),padding_mode='border',align_corners=True)
        signed = values.reshape(vertices.shape[:2])*self.extent/2
        return torch.minimum(signed,torch.zeros_like(signed)).abs().sum(-1)

    @torch.enable_grad()
    def gradient(self):
        from utils import run_smplx_model, SMPLX_JOINTS_28
        source = self.projection.source
        length = len(source['pose'])
        gradient = torch.zeros(length,69,device=source['pose'].device,dtype=torch.float64)
        total = 0.
        self.model.eval().requires_grad_(False)
        for start in range(0,length,24):
            stop = min(start+24,length)
            value = torch.zeros(stop-start,69,device=gradient.device,dtype=torch.float64,requires_grad=True)
            pose,translation = native_increment_pose(self.projection,value,start,stop)
            vertices,_ = run_smplx_model(pose,translation,source['betas'],source['gender'],
                                        joints_ind=SMPLX_JOINTS_28,smpl_model=self.model)
            vertices = torch.where(self.projection.fixed[start:stop,None,None],source['verts'][start:stop],vertices)
            loss = self.frame_sums(vertices).sum()/length
            gradient[start:stop] = torch.autograd.grad(loss,value)[0]
            total += float(loss.detach())
        assert torch.isfinite(gradient).all()
        return gradient,total
