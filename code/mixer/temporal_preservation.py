"""Same-window HOI dynamics preservation in physical root-local FK coordinates."""
import time

import torch
from pytorch3d import transforms

from .conditional_repair import locked_encode, project_local_pose
from .relation_projection import synchronize
from .scene_evidence import local_armijo


def root_local_features(geometry, state):
    """21 nonroot joints; world root rotation and positions share the same frame."""
    rotation = geometry.world_rotation[:, None] @ state['global_rotation'][..., 0, :, :]
    relative = state['human'][..., 1:22, :] - state['human'][..., :1, :]
    return (rotation.transpose(-1, -2)[..., None, :, :] @ relative[..., None]).squeeze(-1)


def physical_derivatives(features, timestamps, valid_frames, valid_links):
    """Uniform seconds, explicit missing frames/edges; stencils end at t>=2."""
    intervals = timestamps[:, 1:] - timestamps[:, :-1]
    dt = intervals[:, :1]
    if not bool((dt > 0).all()) or not torch.allclose(intervals, dt.expand_as(intervals), atol=1e-9, rtol=1e-6):
        raise ValueError('Temporal requires verified equally spaced timestamps')
    links = valid_links & valid_frames[:, 1:] & valid_frames[:, :-1]
    velocity = (features[:, 2:] - features[:, 1:-1]) / dt[..., None, None]
    acceleration = (features[:, 2:] - 2 * features[:, 1:-1] + features[:, :-2]) / dt[..., None, None].square()
    return velocity, acceleration, links[:, 1:], links[:, 1:] & links[:, :-1]


def masked_square_mean(value, mask):
    count = mask.sum(1)
    per_frame = value.square().flatten(2).mean(2)
    return torch.where(mask, per_frame, torch.zeros_like(per_frame)).sum(1) / count.clamp_min(1)


class TemporalObjective:
    def __init__(self, fitter, incoming_parameters, timestamps, valid_frames,
                 valid_links, velocity_scale_m_per_s, acceleration_scale_m_per_s2,
                 lambda_v=1., lambda_a=1.):
        self.geometry = fitter.geometry
        self.timestamps = timestamps.detach().clone()
        self.valid_frames, self.valid_links = valid_frames.clone(), valid_links.clone()
        self.scales = (velocity_scale_m_per_s, acceleration_scale_m_per_s2)
        self.weights = (lambda_v, lambda_a)
        with torch.no_grad():
            self.center = transforms.axis_angle_to_matrix(
                self.geometry.decode(incoming_parameters)['local_delta']).detach()
            # Raw-source objective anchor was decoded with the source geometry.
            source = fitter.objective.anchor
            self.source_features = root_local_features(self.geometry, source).detach()
            self.reference = self.derivatives(self.source_features)

    def derivatives(self, features):
        return physical_derivatives(features, self.timestamps, self.valid_frames, self.valid_links)

    def terms(self, parameters):
        state = self.geometry.decode(parameters)
        velocity, acceleration, vm, am = self.derivatives(root_local_features(self.geometry, state))
        ev = masked_square_mean(velocity - self.reference[0], vm)
        ea = masked_square_mean(acceleration - self.reference[1], am)
        rotation = transforms.axis_angle_to_matrix(state['local_delta'])
        keep = ((rotation[:, 2:] - self.center[:, 2:]).square().sum((-1, -2))
                / (2 * self.geometry.angle_scale**2)).mean((1, 2))
        return dict(keep=keep, velocity_error_m2_per_s2=ev, acceleration_error_m2_per_s4=ea,
                    velocity_normalized=ev / self.scales[0]**2,
                    acceleration_normalized=ea / self.scales[1]**2)

    def __call__(self, parameters):
        terms = self.terms(parameters)
        return terms['keep'] + self.weights[0]*terms['velocity_normalized'] + self.weights[1]*terms['acceleration_normalized']


class TemporalPreservation:
    def __init__(self, enabled=False, lambda_v=1., lambda_a=1.,
                 velocity_scale_m_per_s=None, acceleration_scale_m_per_s2=None,
                 iterations=20, initial_step=.25, max_backtracks=10):
        self.enabled = enabled and (lambda_v != 0 or lambda_a != 0)
        self.objective_options = dict(lambda_v=lambda_v, lambda_a=lambda_a,
            velocity_scale_m_per_s=velocity_scale_m_per_s,
            acceleration_scale_m_per_s2=acceleration_scale_m_per_s2)
        self.iterations, self.initial_step, self.max_backtracks = iterations, initial_step, max_backtracks
        if self.enabled and (velocity_scale_m_per_s is None or acceleration_scale_m_per_s2 is None
                             or velocity_scale_m_per_s <= 0 or acceleration_scale_m_per_s2 <= 0):
            raise ValueError('Freeze positive physical source scales before enabling Temporal')

    def apply(self, fitter, incoming, parameters, timestamps=None, valid_frames=None, valid_links=None):
        # Bypass precedes FK, metadata access, encoding and all random operations.
        if not self.enabled:
            return incoming, parameters, dict(reason='disabled', changed=False)
        if fitter.foot_guard_mode != 'quality':
            raise ValueError('Both Temporal arms require original-proposal quality guards')
        if timestamps is None or valid_frames is None or valid_links is None:
            return incoming, parameters, dict(reason='missing_time_metadata', changed=False)
        synchronize(incoming)
        started = time.perf_counter()
        if fitter.invalid_proposal or not all(bool(v.all()) for v in fitter.guards(parameters).values()):
            return incoming, parameters, dict(reason='infeasible_incoming', changed=False)
        with torch.enable_grad():
            objective = TemporalObjective(fitter, parameters, timestamps, valid_frames,
                                          valid_links, **self.objective_options)
            counts = [m.sum(1).tolist() for m in objective.reference[2:]]
            if not any(any(c) for c in counts):
                return incoming, parameters, dict(reason='empty_stencils', changed=False, valid_counts=counts)
            before = objective.terms(parameters)
            traces = []
            origin = parameters.detach().clone()
            reason = 'optimized'
            for _ in range(self.iterations):
                guards = []
                def admissible(p):
                    checks = fitter.guards(p)
                    guards.append({k:v.tolist() for k,v in checks.items()})
                    return torch.stack(list(checks.values())).all(0)
                try:
                    parameters, trace = local_armijo(parameters, objective, admissible,
                        initial_step=self.initial_step, max_backtracks=self.max_backtracks,
                        gradient_transform=lambda p,g: project_local_pose(fitter.objective,p,g),
                        projected_slope=True)
                except FloatingPointError:
                    reason = 'nonfinite_objective'
                    parameters = origin
                    break
                for trial, checks in zip(trace['trials'], guards):
                    trial['guards'] = checks
                traces.append(trace)
            after = objective.terms(parameters)
        changed = not torch.equal(parameters, origin)
        output = locked_encode(fitter.geometry, parameters) if changed else incoming
        synchronize(incoming)
        values = lambda terms: {k:v.detach().cpu().tolist() for k,v in terms.items()}
        return output, parameters, dict(reason=reason, changed=changed, valid_counts=counts,
            before=values(before), after=values(after), steps=traces,
            seconds=time.perf_counter()-started,
            references=dict(dynamics='raw_source_same_window', keep='incoming_edited_parameters',
                            protection='original_proposal', contact='raw_source_anchors'))
