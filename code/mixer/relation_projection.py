"""First-order source-anchor protection for the DP contribution alone."""
import time

import torch


RANK_RTOL = 1e-6
RANK_ATOL = 1e-10


def synchronize(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def contact_jacobian(objective, parameters):
    """Six VJPs recover frame-local blocks; the decoder has no temporal coupling.

    Rows are masked source hand vectors in metres. History columns are excluded.
    Tests compare this decomposition with a full-window Jacobian, including at
    nonzero parameters. Inactive rows are zero and contribute no numerical rank.
    """
    with torch.enable_grad():
        point = parameters.detach().requires_grad_(True)
        residual = objective.contact_residual(objective.geometry.decode(point))
        residual = (residual * objective.contact[..., None]).flatten(-2)
        rows = [torch.autograd.grad(residual[..., i].sum(), point, retain_graph=i < 5)[0][:, 2:]
                for i in range(6)]
    return torch.stack(rows, dim=-2).detach(), residual.detach()


@torch.no_grad()
def nullspace_projection(jacobian, direction, rtol=RANK_RTOL, atol=RANK_ATOL):
    """Orthogonal nullspace projection using float64 thin SVD, no damping."""
    if not torch.isfinite(jacobian).all() or not torch.isfinite(direction).all():
        raise FloatingPointError('nonfinite relation projection input')
    matrix, vector = jacobian.double(), direction.double()
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    retained = singular > torch.maximum(singular[..., :1] * rtol, singular.new_tensor(atol))
    coefficients = (vh @ vector.unsqueeze(-1)).squeeze(-1) * retained
    result = (vector - (vh.transpose(-1, -2) @ coefficients.unsqueeze(-1)).squeeze(-1)).to(direction)
    before = (matrix @ vector.unsqueeze(-1)).norm()
    after = (matrix @ result.double().unsqueeze(-1)).norm()
    norm, kept = vector.norm(), result.double().norm()
    if not torch.isfinite(result).all():
        raise FloatingPointError('nonfinite relation projection result')
    scale = matrix.norm() * kept
    return result, dict(rank=retained.sum(-1).cpu().tolist(), rtol=rtol, atol=atol,
        damping=0., solve_dtype='float64', jacobian_dtype=str(jacobian.dtype),
        original_norm=float(norm), projected_norm=float(kept),
        r_keep=float(kept / norm) if norm else None,
        jv_before=float(before), jv_after=float(after),
        normalized_residual=float(after / scale) if scale else 0.,
        direction_dot_projected=float((vector * result.double()).sum()),
        reason='zero_direction' if norm == 0 else ('rank_zero' if not retained.any() else 'projected'))


def project_dp_gradient(objective, parameters, gradient, jacobian=None):
    """Project the unscaled gradient (equivalently its negative descent ray)."""
    synchronize(parameters)
    start = time.perf_counter()
    if jacobian is None:
        jacobian, residual = contact_jacobian(objective, parameters)
    else:
        residual = (objective.contact_residual(objective.geometry.decode(parameters))
                    * objective.contact[..., None]).detach()
    synchronize(parameters)
    jacobian_seconds = time.perf_counter() - start
    start = time.perf_counter()
    future, record = nullspace_projection(jacobian, gradient[:, 2:])
    result = torch.cat((torch.zeros_like(gradient[:, :2]), future), dim=1).detach()
    synchronize(parameters)
    record.update(jacobian_seconds=jacobian_seconds, solve_seconds=time.perf_counter()-start,
        jacobian_shape=list(jacobian.shape), active_anchors=int(objective.contact.sum()),
        active_rows=int(objective.contact.sum()) * 3,
        free_parameter_count=parameters[:, 2:].numel(),
        source_residual_norm=float(residual.double().norm()))
    if not objective.contact.any():
        record['reason'] = 'no_active_contact'
    return result, record


def parameter_dp_proxy(gradient, proposal, origin, coefficient):
    return coefficient * (gradient.detach() * (proposal-origin.detach())).flatten(1).sum(1)
