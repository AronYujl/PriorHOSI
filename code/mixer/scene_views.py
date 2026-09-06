"""Teacher-only temporal observations on the native query lattice."""
import torch

from utils import transform_points


TEACHER_VIEWS = ('legacy_occupied', 'environment_only_temporal',
                 'mismatched_environment_temporal')


@torch.no_grad()
def temporal_environment(common, candidate, context, sampler, shift_local_x_m=0.):
    """Replace only temporal blocks; keep goal, anchor and positions untouched.

    The current visual object-query path is batch1. This helper supports its
    block contract explicitly and never treats an additional batch as time.
    """
    if candidate.shape[0] != 1:
        raise ValueError('teacher temporal views require native visual batch1')
    if sampler.scene_type != 'occ_temp':
        raise ValueError('teacher temporal views require occ_temp')
    dataset = sampler.dataset
    mat = context['mat']
    positions = dataset.denormalize_torch(candidate[..., :84])
    world = transform_points(positions, mat)
    grid = dataset.create_meshgrid(batch_size=1).to(candidate.device)
    frames = sampler._get_temp_frame_indices(sampler.temp_voxel_num)
    target = sampler.mask_ind if sampler.mask_ind != -1 else 0
    blocks, centers, invalid = [], [], []
    for frame in frames:
        query_mat = mat.clone()
        query_mat[:, :3, 3] = world[:, frame, target * 3:target * 3 + 3]
        query_mat[:, 1, 3] = 0
        query_mat[:, :3, 3] += shift_local_x_m * mat[:, :3, 0]
        points = transform_points(grid, query_mat)
        occ = dataset.get_occ_for_points(points, None, context['scene_flag'])
        blocks.append(occ.reshape(-1, *dataset.nb_voxels).float().permute(0, 2, 1, 3))
        centers.append(query_mat[:, :3, 3])
        bounds = dataset.scene_grid_torch.to(points)
        invalid.append(((points < bounds[:3]) | (points >= bounds[3:6])).any(-1).float().mean())
    updated = list(common)
    updated[15] = torch.cat([common[15][:1], *blocks], dim=0)
    return tuple(updated), dict(frames=frames, centers=torch.stack(centers),
                               outside_fraction=torch.stack(invalid),
                               shift_local_x_m=shift_local_x_m)


def teacher_common(common, candidate, context, sampler, view):
    if view == 'legacy_occupied':
        return common
    if view == 'environment_only_temporal':
        return temporal_environment(common, candidate, context, sampler)[0]
    raise ValueError('production teacher view must be legacy_occupied or environment_only_temporal')
