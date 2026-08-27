from functools import partial

import torch
import torch.nn.functional as F

from priors.hsi.body_proxy import proxy_points
from priors.hsi.constants import FLOOR_EXCLUSION_HEIGHT_M, SDF_MARGIN_M

def apply_hand_object_interaction_guidance_loss(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels):
    # human_jnts: BS X T X 24 X 3
    # obj_verts: BS X T X Nv' X 3
    # pred_seq_com_pos: BS X T X 3
    # pred_obj_rot_mat: BS X T X 3 X 3
    # contact_labels: BS X T X 4

    num_seq = human_jnts.shape[0]
    num_steps = human_jnts.shape[1]

    # Contact loss: minimize the distance between palm joints and the nearest object vertices.
    l_palm_idx = 22
    r_palm_idx = 23

    left_palm_jpos = human_jnts[:, :, l_palm_idx, :] # BS X T X 3
    right_palm_jpos = human_jnts[:, :, r_palm_idx, :] # BS X T X 3

    contact_points = torch.cat((left_palm_jpos[:, :, None, :], \
                right_palm_jpos[:, :, None, :]), dim=2) # BS X T X 2 X 3
    bs, seq_len, _, _ = contact_points.shape

    dists = torch.cdist(contact_points.reshape(bs*seq_len, 2, 3)[:, :, :], \
                obj_verts.reshape(bs*seq_len, -1, 3)) # (BS*T) X 2 X N_object
    dists, _ = torch.min(dists, 2) # (BS*T) X 2

    pred_contact_semantic = contact_labels[:, :, -4:-2] # BS X T X 2

    contact_labels = pred_contact_semantic > 0.95

    contact_labels = contact_labels.reshape(bs*seq_len, -1)[:, :2].detach().to(dists.device) # (BS*T) X 2

    zero_target = torch.zeros_like(dists).to(dists.device)
    contact_threshold = 0.02

    loss_contact = F.l1_loss(torch.maximum(dists*contact_labels[:, :2]-contact_threshold, zero_target), \
            zero_target)

    # Temporal consistency loss.
    left_palm_to_obj_com = left_palm_jpos - pred_seq_com_pos.detach() # BS X T X 3
    right_palm_to_obj_com = right_palm_jpos - pred_seq_com_pos.detach()
    relative_left_palm_jpos = torch.matmul(pred_obj_rot_mat.detach().transpose(2, 3), \
                    left_palm_to_obj_com[:, :, :, None]).squeeze(-1) # BS X T X 3
    relative_right_palm_jpos = torch.matmul(pred_obj_rot_mat.detach().transpose(2, 3), \
                    right_palm_to_obj_com[:, :, :, None]).squeeze(-1)

    contact_labels = contact_labels.reshape(num_seq, num_steps, -1) # BS X T X 2

    # Expand dimensions of contact_labels for multiplication
    left_contact_labels_expanded = contact_labels[:, :, 0:1]
    left_contact_mask = left_contact_labels_expanded * left_contact_labels_expanded.transpose(-1, -2)

    right_contact_labels_expanded = contact_labels[:, :, 1:2]
    right_contact_mask = right_contact_labels_expanded * right_contact_labels_expanded.transpose(-1, -2) # BS X T X T

    left_norms = torch.norm(relative_left_palm_jpos, dim=-1, keepdim=True)
    left_normalized = relative_left_palm_jpos / left_norms
    left_similarity = torch.matmul(left_normalized, left_normalized.transpose(-1, -2))

    right_norms = torch.norm(relative_right_palm_jpos, dim=-1, keepdim=True)
    right_normalized = relative_right_palm_jpos / right_norms
    right_similarity = torch.matmul(right_normalized, right_normalized.transpose(-1, -2)) # BS X T X T

    loss_consistency = 1 - torch.mean(left_similarity * left_contact_mask) + \
                1 - torch.mean(right_similarity * right_contact_mask)

    loss = bs * (loss_contact + loss_consistency)

    return loss

def apply_feet_floor_contact_guidance(human_jnts):
    # human_jnts: BS X T X 28 X 3
    left_toe_idx = 10
    right_toe_idx = 11

    l_toe_height = human_jnts[:, :, left_toe_idx, 1:2] # BS X T X 1
    r_toe_height = human_jnts[:, :, right_toe_idx, 1:2] # BS X T X 1
    support_foot_height = torch.minimum(l_toe_height, r_toe_height) # BS X T X 1

    loss_feet_floor_contact = F.mse_loss(support_foot_height, torch.ones_like(support_foot_height)*0.02)

    loss = human_jnts.shape[0] * loss_feet_floor_contact

    return loss

def apply_hoi_guidance_loss(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels, scene_flag, get_nearest_free_voxel):
    # Hand-object contact + temporal consistency, plus feet-floor contact.
    loss_feet_floor_contact = apply_feet_floor_contact_guidance(human_jnts)

    loss_hand_object_interaction = apply_hand_object_interaction_guidance_loss(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels)
    loss = loss_hand_object_interaction * 10 + loss_feet_floor_contact * 500
    return loss

def _apply_hsi_voxel_guidance_loss(human_jnts, scene_flag, get_nearest_free_voxel):
    """The released nearest-free-voxel guidance term.

    This is intentionally the unmasked legacy operation.  The object-guidance
    route and the default-off HSI route retain its exact query and reduction.
    The scene-only mesh route below uses the floor-masked equivalent, whose
    query is the expensive part that can be reduced safely.
    """
    is_penetrating, nearest_free_points = get_nearest_free_voxel(human_jnts, scene_flag)
    loss = F.mse_loss(human_jnts, nearest_free_points) * 20000
    return loss


def _apply_hsi_near_floor_voxel_guidance_loss(
    human_jnts, scene_flag, get_nearest_free_voxel
):
    """Evaluate the legacy near-floor target without querying irrelevant rows.

    The reference implementation queried the complete tensor and then selected
    ``y < FLOOR_EXCLUSION_HEIGHT_M`` for the scene-only split.  Selection is
    detached, so packing only those points leaves both the derivative and the
    full-tensor denominator unchanged.  A separate one-row query also preserves
    mixed-batch scene alignment.
    """
    if human_jnts.ndim != 4 or human_jnts.shape[-1] != 3:
        raise ValueError("human_jnts must have shape [B,T,J,3]")
    batch_size = human_jnts.shape[0]
    scene_flag = torch.as_tensor(scene_flag, device=human_jnts.device).reshape(-1)
    if scene_flag.numel() != batch_size:
        raise ValueError("scene_flag does not match the guidance batch")

    near_floor = (human_jnts[..., 1] < FLOOR_EXCLUSION_HEIGHT_M).detach()
    # Keep a graph edge even when every point is outside the eligible band, so
    # the zero loss has the same zero gradient contract as the queried path.
    # Build it from the masked tensor: a non-eligible NaN is zero in the legacy
    # ``where`` result and must not turn this zero loss into NaN.
    masked_human = torch.where(
        near_floor.unsqueeze(-1), human_jnts, torch.zeros_like(human_jnts)
    )
    squared_delta = masked_human.sum() * 0.0
    for batch_index in range(batch_size):
        selected = human_jnts[batch_index][near_floor[batch_index]]
        if selected.numel() == 0:
            continue
        selected = selected.reshape(1, 1, -1, 3)
        _, nearest_free_points = get_nearest_free_voxel(
            selected, scene_flag[batch_index : batch_index + 1]
        )
        if nearest_free_points.shape != selected.shape:
            raise ValueError(
                "get_nearest_free_voxel returned free-point shape %s, expected %s"
                % (tuple(nearest_free_points.shape), tuple(selected.shape))
            )
        squared_delta = squared_delta + (selected - nearest_free_points).pow(2).sum()
    return 20000.0 * squared_delta / human_jnts.numel()


def _config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _mesh_signed_distance(geometry, points):
    """Query one geometry or one geometry per batch element."""
    if isinstance(geometry, (list, tuple)):
        if len(geometry) != points.shape[0]:
            raise ValueError(
                "per-sample scene geometry count does not match guidance batch: "
                "%d != %d" % (len(geometry), points.shape[0])
            )
        distances = []
        for batch_index, scene_geometry in enumerate(geometry):
            if scene_geometry is None:
                raise ValueError("scene geometry is missing for guidance batch %d" % batch_index)
            distances.append(scene_geometry.signed_distance(points[batch_index:batch_index + 1]))
        return torch.cat(distances, dim=0)
    if geometry is None:
        raise ValueError("mesh-SDF guidance requires a scene geometry")
    return geometry.signed_distance(points)


def _mesh_penetration_loss(geometry, points, above_floor):
    """Query the mesh only for eligible proxy points, preserving full mean scale.

    The floor split is detached by contract.  Boolean indexing therefore only
    changes which SDF samples are requested; the selected coordinates retain
    their normal autograd path.  The numerator is divided by the complete
    ``B*T*N`` proxy-point count, exactly matching ``tensor.mean()`` on the old
    dense query while avoiding work for the below-floor population.
    """
    if isinstance(geometry, (list, tuple)):
        if len(geometry) != points.shape[0]:
            raise ValueError(
                "per-sample scene geometry count does not match guidance batch: "
                "%d != %d" % (len(geometry), points.shape[0])
            )
        geometries = tuple(geometry)
    else:
        if geometry is None:
            raise ValueError("mesh-SDF guidance requires a scene geometry")
        geometries = (geometry,) * points.shape[0]

    # Use a finite masked tensor for the zero edge.  This retains a graph edge
    # without allowing a NaN in an ineligible point to contaminate a zero loss.
    masked_points = torch.where(
        above_floor.unsqueeze(-1), points, torch.zeros_like(points)
    )
    numerator = masked_points.sum() * 0.0
    for batch_index, scene_geometry in enumerate(geometries):
        if scene_geometry is None:
            raise ValueError("scene geometry is missing for guidance batch %d" % batch_index)
        selected = points[batch_index][above_floor[batch_index]]
        if selected.numel() == 0:
            continue
        selected = selected.reshape(1, 1, -1, 3)
        signed_distance = scene_geometry.signed_distance(selected)
        penetration = F.relu(SDF_MARGIN_M - signed_distance)
        numerator = numerator + penetration.pow(2).sum()
    return numerator / points[..., 0].numel()


def apply_hsi_guidance_loss(human_jnts, scene_flag, get_nearest_free_voxel):
    """Apply the released HSI nearest-free-voxel guidance exactly."""
    return _apply_hsi_voxel_guidance_loss(
        human_jnts, scene_flag, get_nearest_free_voxel
    )


def apply_hsi_mesh_guidance_loss(
    human_jnts,
    global_jrot_mat,
    local_jrot_mat,
    *,
    scene_flag,
    get_nearest_free_voxel,
    geometry,
    proxy="area512",
    sdf_weight=0.0,
):
    """Apply scene-only mesh guidance with an explicit keyword-only contract."""
    if get_nearest_free_voxel is None:
        raise ValueError("HSI guidance requires get_nearest_free_voxel")
    sdf_weight = float(sdf_weight)
    if sdf_weight <= 0.0:
        return apply_hsi_guidance_loss(
            human_jnts, scene_flag, get_nearest_free_voxel
        )
    if geometry is None:
        raise ValueError("mesh-SDF guidance requires a scene geometry")

    points = proxy_points(
        human_jnts,
        global_jrot_mat,
        local_jrot_mat,
        proxy=str(proxy),
    )
    above_floor = (points[..., 1] >= FLOOR_EXCLUSION_HEIGHT_M).detach()
    loss_pen = sdf_weight * _mesh_penetration_loss(geometry, points, above_floor)

    # The old code queried all joints, masked the target afterward, and divided
    # by human_jnts.numel().  The helper queries only the same detached mask and
    # retains that full-tensor denominator.
    loss_ground = _apply_hsi_near_floor_voxel_guidance_loss(
        human_jnts, scene_flag, get_nearest_free_voxel
    )
    # Both terms are scalar means.  The mesh weight is calibrated on the
    # batch-size-1 P16 protocol, while the retained voxel term preserves the
    # legacy mean reduction; neither term gets an additional batch factor.
    return loss_ground + loss_pen


def apply_hsi_scene_guidance_loss(
    human_jnts,
    global_jrot_mat,
    local_jrot_mat,
    *,
    scene_flag,
    get_nearest_free_voxel,
    geometry,
    cfg,
):
    """Shared scene-only route used by both diffusion sampler implementations."""
    proxy_name = _config_value(cfg, "hsi_guidance_sdf_proxy", None)
    sdf_weight = _config_value(cfg, "hsi_guidance_sdf_weight", 0.0)
    if proxy_name in (None, False, "", "none") or sdf_weight in (None, False):
        return apply_hsi_guidance_loss(
            human_jnts, scene_flag, get_nearest_free_voxel
        )
    sdf_weight = float(sdf_weight)
    if sdf_weight <= 0.0:
        return apply_hsi_guidance_loss(
            human_jnts, scene_flag, get_nearest_free_voxel
        )
    return apply_hsi_mesh_guidance_loss(
        human_jnts,
        global_jrot_mat,
        local_jrot_mat,
        scene_flag=scene_flag,
        get_nearest_free_voxel=get_nearest_free_voxel,
        geometry=geometry,
        proxy=str(proxy_name),
        sdf_weight=sdf_weight,
    )

def apply_hsi_guidance_fn(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels, scene_flag, get_nearest_free_voxel):
    # Object arguments satisfy the guidance_fn convention and are deliberately unused, as apply_hoi_guidance_loss accepts and ignores scene_flag and get_nearest_free_voxel.
    return apply_hsi_guidance_loss(
        human_jnts,
        scene_flag,
        get_nearest_free_voxel,
    )


def apply_mixed_guidance_fn(
    human_jnts,
    obj_verts,
    pred_seq_com_pos,
    pred_obj_rot_mat,
    contact_labels,
    scene_flag,
    get_nearest_free_voxel,
    *,
    is_object,
):
    """Apply the appropriate guidance terms independently to each batch row."""
    object_mask = torch.as_tensor(
        is_object, device=human_jnts.device, dtype=torch.bool
    ).reshape(-1)
    if object_mask.numel() != human_jnts.shape[0]:
        raise ValueError("is_object does not match the guidance batch")
    scene_flag = torch.as_tensor(scene_flag, device=human_jnts.device).reshape(-1)
    if scene_flag.numel() != human_jnts.shape[0]:
        raise ValueError("scene_flag does not match the guidance batch")

    loss = human_jnts.new_zeros(())
    hsi_mask = torch.logical_not(object_mask)
    if bool(hsi_mask.any()):
        loss = loss + apply_hsi_guidance_loss(
            human_jnts[hsi_mask], scene_flag[hsi_mask], get_nearest_free_voxel
        )
    if bool(object_mask.any()):
        loss = loss + apply_hosi_guidance_loss(
            human_jnts[object_mask],
            obj_verts[object_mask],
            pred_seq_com_pos[object_mask],
            pred_obj_rot_mat[object_mask],
            contact_labels[object_mask],
            scene_flag[object_mask],
            get_nearest_free_voxel,
        )
    return loss

def apply_hosi_guidance_loss(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels, scene_flag, get_nearest_free_voxel):
    bs = human_jnts.shape[0]

    loss_hand_object_interaction = apply_hand_object_interaction_guidance_loss(human_jnts, obj_verts, pred_seq_com_pos, pred_obj_rot_mat, contact_labels)
    loss = loss_hand_object_interaction * 10

    # Floor-object penetration loss (y-up), moved out of apply_hand_object_interaction_guidance_loss.
    # Weight 100 = inner 10 (former loss_floor_object * 10) x outer 10 (loss_hand_object_interaction * 10).
    loss_floor_object = torch.minimum(obj_verts[:, :, :, -2], \
                torch.zeros_like(obj_verts[:, :, :, -2])).abs().mean()
    loss += bs * loss_floor_object * 100

    loss += apply_hsi_guidance_loss(human_jnts, scene_flag, get_nearest_free_voxel)

    is_penetrating, nearest_free_points = get_nearest_free_voxel(obj_verts, scene_flag)
    loss += F.mse_loss(obj_verts, nearest_free_points) * 1000

    return loss

def select_guidance_fn(use_guidance, is_object):
    """Select guidance without sending scene-only inputs through object terms.
    Scene-only LINGO has exactly-zero ``obj_rot_mat_ref``, so the temporal-consistency term in
    ``apply_hand_object_interaction_guidance_loss`` normalizes palm positions relative to a zero object frame and returns NaN.
    Its 0.95 gate was crossed by 447 contact-semantic predictions over one real scene-only LINGO episode.
    """
    if not use_guidance:
        return None
    is_object = torch.as_tensor(is_object, dtype=torch.bool).reshape(-1)
    if not bool(is_object.any()):
        return apply_hsi_guidance_fn
    if bool(is_object.all()):
        return apply_hosi_guidance_loss
    mixed = partial(apply_mixed_guidance_fn, is_object=is_object)
    # The sampler builds object geometry only for object rows.  Preserve the
    # object-side callable as metadata so the P16-GQ SDF-aware path can route
    # the two subsets without invoking this legacy adapter twice.
    mixed.object_guidance_fn = apply_hosi_guidance_loss
    return mixed
