import os
import hashlib
import pickle as pkl
import pickle
import random
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
import trimesh

from utils import *
from constants import *
from priors.hoi.models import load_trained_hoi_prior
from priors.core.representation import transform_object_points_for_next_window
from priors.core.window_codec import WindowStateCodec, project_to_so3


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def synchronize_cuda(device):
    device = torch.device(device)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def convert_to_serializable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def recompute_rollout_bps(codec, references, rest_vertices, sequence_names, chunk_size=8):
    """Recompute each condition from current generated pose without future GT."""
    result = torch.empty(references.shape[0], 1024, 3, device=references.device, dtype=references.dtype)
    by_object = {}
    for row in range(references.shape[0]):
        by_object.setdefault(sequence_names[row].split('_')[1], []).append(row)
    for object_name, rows in by_object.items():
        rest = rest_vertices[object_name]
        for offset in range(0, len(rows), chunk_size):
            selected = rows[offset:offset + chunk_size]
            rotations = references[selected]
            vertices = rest[None].expand(len(selected), -1, -1)
            result[selected] = codec.recompute_bps(vertices, rotations)
    return result
from datasets.infbagel import InfBaGelDataset
from guidance_loss import *
import json
from eval_metrics import *

import pytorch3d.transforms as transforms
from constants import *

def compute_metrics(sampler_body, cfg, points_orig, global_rot_6d, points_gt_orig, obj_trans_pred, obj_trans_gt, obj_rot_mat_pred, obj_rot_mat_gt, start_point_all_gt, start_object_trans_all_gt, end_object_trans_all_gt, xy_points_all_gt, seq_name_dict, obj_rest_verts, rest_human_offsets_all, transl_all, betas_all, gender_all):
    # points_orig: 534 X 7 X T X 84
    # global_rot_6d: 534 X 7 X T X 132
    # points_gt_orig: N X T X 84
    # obj_trans_pred: 534 X 7 X T X 3
    # obj_trans_gt: N X T X 3
    # obj_rot_mat_pred: 534 X 7 X T X 9
    # obj_rot_mat_gt: N X T X 9

    device = cfg.device

    # Per-gender SMPL-X models (batch_size=1), created once and reused across all segments.
    smplx_model_cache = {}

    object_sdf_root = os.path.join(ROOT_DIR, 'data', 'object', 'rest_object_sdf_256_npy_files')
    obj_sdf = {}
    obj_sdf_json = {}
    for file in sorted(os.listdir(object_sdf_root)):
        if not file.endswith('.npy'):
            continue
        obj_name = file.split('.')[0]
        sdf_path = os.path.join(object_sdf_root, file)
        obj_sdf[obj_name] = np.load(sdf_path)
        obj_sdf_json[obj_name] = json.load(open(os.path.join(object_sdf_root, f'{file[:-4]}.json'), 'r'))
    
    with open(os.path.join(SMPL_DIR, 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
        idxs_data = pkl.load(f)
    hand_idxs = np.concatenate([idxs_data['left_hand'], idxs_data['right_hand']]) # 1556

    points_all_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 3*(cfg.dataset.nb_joints -4)).to(device)
    object_trans_all_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 3).to(device)
    object_rot_mat_all_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 9).to(device)

    start_point_all = torch.zeros(0, 1, 3*cfg.dataset.nb_joints).to(device)
    start_object_trans_all = torch.zeros(0, 1, 3).to(device)
    end_object_trans_all = torch.zeros(0, 1, 3).to(device)
    xy_points_all = torch.zeros(0, 1, 3*cfg.dataset.nb_joints).to(device)

    feet_height_list = []
    foot_sliding_list = []

    gt_contact_percent_list = []
    pred_contact_percent_list = []
    contact_acc_list = []
    contact_precision_list = []
    contact_recall_list = []
    contact_f1_list = []

    hand_pen_loss_list = []
    hand_pen_ratio_list = []

    human_pen_loss_list = []
    human_pen_ratio_list = []

    sum_len = 0
    per_sequence_metrics = {}
    for seg_id_true in range(len(seq_name_dict)):
        obj_name = seq_name_dict[seg_id_true].split('_')[1]

        points_orig_seg = points_orig[seg_id_true][:3].reshape(-1, cfg.max_window_size, 3*(cfg.dataset.nb_joints))
        points_orig_seg = points_orig_seg[:, cfg.auto_regre_num:, :]

        global_rot_6d_seg = global_rot_6d[seg_id_true][:3].reshape(-1, cfg.max_window_size, 22*6)
        global_rot_6d_seg = global_rot_6d_seg[:, cfg.auto_regre_num:, :]

        obj_trans_pred_seg = obj_trans_pred[seg_id_true][:3].reshape(-1, cfg.max_window_size, 3)
        obj_trans_pred_seg = obj_trans_pred_seg[:, cfg.auto_regre_num:, :]

        obj_rot_mat_pred_seg = obj_rot_mat_pred[seg_id_true][:3].reshape(-1, cfg.max_window_size, 9)
        obj_rot_mat_pred_seg = obj_rot_mat_pred_seg[:, cfg.auto_regre_num:, :]

        points_gt_orig_seg = points_gt_orig[sum_len:sum_len+3].reshape(-1, cfg.dataset.nb_joints-4, 3) # T * J * 3
        obj_trans_gt_seg = obj_trans_gt[sum_len:sum_len+3].reshape(-1, 3)
        obj_rot_mat_gt_seg = obj_rot_mat_gt[sum_len:sum_len+3].reshape(-1, 3, 3)
        sum_len += 3

        obj_rest_verts_gt_seg = load_object_geometry_w_rest_geo(obj_rot_mat_gt_seg, obj_trans_gt_seg, obj_rest_verts[obj_name])

        start_point_all = torch.cat((start_point_all, points_orig_seg[0:1, 0, :].unsqueeze(1)), dim=0)
        start_object_trans_all = torch.cat((start_object_trans_all, obj_trans_pred_seg[0:1, 0, :].unsqueeze(1)), dim=0)
        end_object_trans_all = torch.cat((end_object_trans_all, obj_trans_pred_seg[-1:, -1, :].unsqueeze(1)), dim=0)
        xy_points_all = torch.cat((xy_points_all, points_orig_seg[:, -2, :].unsqueeze(1)), dim=0)
        
        joints = interpolate_joints(points_orig_seg.reshape(-1, 3*(cfg.dataset.nb_joints)), scale=cfg.interp_s)
        obj_trans, obj_rot_mat = interp_object(obj_trans_pred_seg.reshape(-1, 3).cpu().numpy(), obj_rot_mat_pred_seg.reshape(-1, 9).cpu().numpy(), cfg.interp_s)
        
        joints = joints.reshape(-1, (cfg.max_window_size-2)*cfg.interp_s, 3*(cfg.dataset.nb_joints)) # S * 42 * 84
        
        # FK to get joint positions.
        curr_seq_local_jpos = rest_human_offsets_all[seg_id_true][:cfg.max_window_size*cfg.interp_s-6] # [42, 24, 3]
        curr_seq_local_jpos = curr_seq_local_jpos.repeat(joints.shape[0], 1, 1, 1) # [S, 42, 24, 3]
        curr_seq_local_jpos[:, :, 0, :] = joints.reshape(-1, 42, 28, 3)[:, :, 0, :]
        
        global_jrot_mat_seg = transforms.rotation_6d_to_matrix(global_rot_6d_seg.reshape(-1, 22, 6)) # [S*14, 22, 3, 3]
        local_jrot_mat_seg = sampler_body.dataset.quat_ik_torch(global_jrot_mat_seg.reshape(-1, 22, 3, 3))

        local_jrot_q_seg = transforms.matrix_to_quaternion(local_jrot_mat_seg)
        local_jrot_q_48 = interp_jrot(local_jrot_q_seg, 3).reshape(-1, cfg.max_window_size*cfg.interp_s-6, 22, 4) # [S, 42, 22, 4]
        
        local_jrot_mat_48 = transforms.quaternion_to_matrix(local_jrot_q_48).reshape(-1, 22, 3, 3) # [S*42, 22, 3, 3]

        _, human_jnts_48 = sampler_body.dataset.quat_fk_torch(local_jrot_mat_48, curr_seq_local_jpos.reshape(-1, 24, 3)) # [S*42, 24, 3]
        human_jnts_48 = human_jnts_48.detach()

        # reconstruct human verts and joints
        transl = transl_all[seg_id_true] # 3
        betas = betas_all[seg_id_true] # 16
        gender = gender_all[seg_id_true] # 'male'
        # Everything below this line is y-up.  The released code wrapped this call in yup_to_zup/zup_to_yup because OMOMO's rotation channel held global rotations of a zup_to_yup-rotated template and transl held -zup_to_yup(J0) instead of -J0; datasets/infbagel.py now fixes both at the source (see its "One y-up world, one y-up template" block and datasets/utils.py, "Asset world frame"), so that sandwich is the error rather than the correction and has to be deleted in the same change -- measured on real OMOMO with the real SMPL-X model, keeping it against the fixed source moves the vertices 1.87 m, which does not crash but silently collapses hand_pen/human_pen toward zero because the SDF query leaves the object's box.  The two yup_to_zup uses further down, the CHOIS npz export and the collision inputs, are unrelated genuine frame conversions and stay.
        root_trans = joints.reshape(-1, 28, 3)[:, 0, :] + transl  # y-up pelvis + y-up SMPL-X translation offset
        pose_pred = transforms.matrix_to_axis_angle(local_jrot_mat_48).reshape(-1, 22, 3)  # already y-up locals

        if gender not in smplx_model_cache:
            smplx_model_cache[gender] = create_smplx_model(gender, device, batch_size=1)
        verts, joints = run_smplx_model(pose_pred, root_trans, betas[None].repeat(root_trans.shape[0], 1), gender, joints_ind=None, smpl_model=smplx_model_cache[gender])

        points_all_48 = torch.cat((points_all_48, human_jnts_48.reshape(-1, cfg.max_window_size*cfg.interp_s-6, 3*(cfg.dataset.nb_joints-4))), dim=0)

        obj_trans = obj_trans.reshape(-1, (cfg.max_window_size-2)*cfg.interp_s, 3)
        obj_rot_mat = obj_rot_mat.reshape(-1, (cfg.max_window_size-2)*cfg.interp_s, 9)

        object_trans_all_48 = torch.cat((object_trans_all_48, torch.from_numpy(obj_trans).to(device)), dim=0)
        object_rot_mat_all_48 = torch.cat((object_rot_mat_all_48, torch.from_numpy(obj_rot_mat).to(device)), dim=0)
        
        model_name = cfg.ckpt_path.split('/')[-1]
        if cfg.get('save_chois_eval_npz', False):
            chois_output_dir = os.path.abspath(str(cfg.chois_eval_output_dir))
            chois_ground_truth_dir = os.path.abspath(str(cfg.chois_eval_ground_truth_dir))
            chois_path = os.path.join(chois_output_dir, f"{seq_name_dict[seg_id_true]}.npz")
            chois_gt_path = os.path.join(chois_ground_truth_dir, f"{seq_name_dict[seg_id_true]}.npz")
            if os.path.exists(chois_path):
                raise FileExistsError(f"Refusing to overwrite CHOIS evaluator input: {chois_path}")
            if os.path.exists(chois_gt_path):
                raise FileExistsError(f"Refusing to overwrite CHOIS evaluator GT: {chois_gt_path}")
            np.savez(
                chois_path,
                seq_name=np.asarray(seq_name_dict[seg_id_true]),
                global_jpos=yup_to_zup(human_jnts_48).cpu().numpy(),
            )  # T X 24 X 3, official evaluator Z-up convention
            np.savez(
                chois_gt_path,
                seq_name=np.asarray(seq_name_dict[seg_id_true]),
                global_jpos=yup_to_zup(points_gt_orig_seg).cpu().numpy(),
            )
        
        # Save motion parameters for mesh recovery
        if cfg.save_motion_params:
            motion_params_dir = os.path.join('motion_params', cfg.exp_name, model_name[:-4])
            if not os.path.exists(motion_params_dir):
                os.makedirs(motion_params_dir)

            # Prepare complete motion parameters
            motion_params = {
                'seq_name': seq_name_dict[seg_id_true],
                'human_motion': {
                    'pose_pred': pose_pred.cpu().numpy(),  # [T, 22, 3] - SMPL body pose (axis-angle)
                    'root_trans': root_trans.cpu().numpy(),  # [T, 3] - root translation
                    'betas': betas.cpu().numpy(),  # [16] - SMPL shape parameters
                    'gender': gender  # string - gender info
                },
                'object_motion': {
                    'obj_trans': obj_trans,  # [T, 3] - object translation
                    'obj_rot_mat': obj_rot_mat,  # [T, 3, 3] - object rotation matrices
                    'obj_name': obj_name  # string - object name
                }
            }

            # Save as pickle file for complete data preservation
            with open(os.path.join(motion_params_dir, f"{seq_name_dict[seg_id_true]}_motion_params.pkl"), 'wb') as f:
                pickle.dump(motion_params, f)

        floor_height = determine_floor_height_and_contacts(human_jnts_48.cpu().numpy().reshape(-1, 24, 3))
        foot_sliding = compute_foot_sliding_for_smpl(human_jnts_48.cpu().numpy().reshape(-1, 24, 3), floor_height)
        feet_height_list.append(floor_height)
        foot_sliding_list.append(foot_sliding)

        obj_rest_verts_pred_seg = load_object_geometry_w_rest_geo(torch.from_numpy(obj_rot_mat).reshape(-1, 3, 3).float().to(device), torch.from_numpy(obj_trans).reshape(-1, 3).float().to(device), obj_rest_verts[obj_name])

        gt_contact_percent, pred_contact_percent, contact_acc, contact_precision, contact_recall, contact_f1 = \
            compute_hand_object_interaction(human_jnts_48.reshape(-1, 24, 3), points_gt_orig_seg, obj_rest_verts_pred_seg, obj_rest_verts_gt_seg)
        
        gt_contact_percent_list.append(gt_contact_percent)
        pred_contact_percent_list.append(pred_contact_percent)
        contact_acc_list.append(contact_acc)
        contact_precision_list.append(contact_precision)
        contact_recall_list.append(contact_recall)
        contact_f1_list.append(contact_f1)

        verts = verts.reshape(-1, 10475, 3)
        hand_verts = verts[:, hand_idxs, :]

        obj_trans = torch.from_numpy(obj_trans).reshape(-1, 3).to(device)
        obj_rot_mat = torch.from_numpy(obj_rot_mat).reshape(-1, 3, 3).to(device)

        segment_hand_pen_loss = None
        segment_hand_pen_ratio = None
        segment_human_pen_loss = None
        segment_human_pen_ratio = None
        if obj_name not in ['woodchair', 'whitechair', 'largebox', 'largetable', 'plasticbox', 'trashcan']:   
            hand_pen_loss, hand_pen_ratio = compute_collision(yup_to_zup(hand_verts), obj_sdf[obj_name], obj_sdf_json[obj_name], yup_to_zup_rotation_matrix(obj_rot_mat), yup_to_zup(obj_trans))
            hand_pen_loss_list.append(hand_pen_loss)
            hand_pen_ratio_list.append(hand_pen_ratio)

            human_pen_loss, human_pen_ratio = compute_collision(yup_to_zup(verts), obj_sdf[obj_name], obj_sdf_json[obj_name], yup_to_zup_rotation_matrix(obj_rot_mat), yup_to_zup(obj_trans))
            human_pen_loss_list.append(human_pen_loss)
            human_pen_ratio_list.append(human_pen_ratio)

            segment_hand_pen_loss = float(hand_pen_loss)
            segment_hand_pen_ratio = float(hand_pen_ratio)
            segment_human_pen_loss = float(human_pen_loss * 10475 / 100)
            segment_human_pen_ratio = float(human_pen_ratio)

            # print(f'scene_name: {seq_name_dict[seg_id_true]}, hand_pen_loss: {hand_pen_loss}, hand_pen_ratio: {hand_pen_ratio}, human_pen_loss: {human_pen_loss}, human_pen_ratio: {human_pen_ratio}')

        segment_mpjpe, segment_trans_dist, segment_obj_trans_dist, segment_obj_rot_dist = compute_gt_difference(
            human_jnts_48.reshape(-1, 24, 3),
            points_gt_orig_seg.reshape(-1, 24, 3),
            obj_trans.reshape(-1, 3),
            obj_trans_gt_seg.reshape(-1, 3),
            obj_rot_mat.reshape(-1, 3, 3),
            obj_rot_mat_gt_seg.reshape(-1, 3, 3),
        )
        segment_xy_pred = points_orig_seg[:, -2, :].reshape(-1, 28, 3)[:, 0].clone()
        segment_xy_gt = xy_points_all_gt[sum_len - 3:sum_len].reshape(-1, 28, 3)[:, 0].clone()
        segment_xy_pred[:, 1] = 0.0
        segment_xy_gt[:, 1] = 0.0
        per_sequence_metrics[seq_name_dict[seg_id_true]] = {
            'object_name': obj_name,
            'foot_sliding': float(foot_sliding),
            'contact_precision': float(contact_precision),
            'contact_recall': float(contact_recall),
            'contact_f1': float(contact_f1),
            'mpjpe': float(segment_mpjpe),
            'trans_dist': float(segment_trans_dist),
            'obj_trans_dist': float(segment_obj_trans_dist),
            'obj_rot_dist': float(segment_obj_rot_dist),
            'pelvis_goal_error_cm': float(
                torch.linalg.norm(segment_xy_pred - segment_xy_gt, dim=-1).mean().item() * 100
            ),
            'end_obj_trans_err': float(torch.linalg.norm(
                obj_trans.reshape(-1, 3)[-1]
                - obj_trans_gt_seg.reshape(-1, 3)[-1]
            ).item() * 100),
            'hand_pen_loss_omomo': segment_hand_pen_loss,
            'hand_pen_ratio': segment_hand_pen_ratio,
            'human_pen_loss_infbagel': segment_human_pen_loss,
            'human_pen_ratio': segment_human_pen_ratio,
        }


    hand_pen_loss = np.array(hand_pen_loss_list).mean()
    hand_pen_ratio = np.array(hand_pen_ratio_list).mean()
    human_pen_loss = np.array(human_pen_loss_list).mean()
    human_pen_ratio = np.array(human_pen_ratio_list).mean()

    mpjpe, trans_dist, obj_trans_dist, obj_rot_dist = compute_gt_difference(points_all_48, points_gt_orig, object_trans_all_48, obj_trans_gt, object_rot_mat_all_48, obj_rot_mat_gt)

    start_point_err, start_obj_trans_err, end_obj_trans_err, xy_points_err = compute_condition_matching(start_point_all, start_object_trans_all, end_object_trans_all, xy_points_all, start_point_all_gt, start_object_trans_all_gt, end_object_trans_all_gt, xy_points_all_gt)

    feet_height = np.array(feet_height_list).mean()
    foot_sliding = np.array(foot_sliding_list).mean()

    contact_precision = np.array(contact_precision_list).mean()
    contact_recall = np.array(contact_recall_list).mean()
    contact_f1 = np.array(contact_f1_list).mean()
    contact_percent = np.array(pred_contact_percent_list).mean()
    gt_contact_percent = np.array(gt_contact_percent_list).mean()
    contact_acc = np.array(contact_acc_list).mean()

    metrics = {
        'end_obj_trans_err': end_obj_trans_err,
        'xy_points_err': xy_points_err,
        'feet_height': feet_height,
        'foot_sliding': foot_sliding,
        'contact_acc': contact_acc,
        'contact_precision': contact_precision,
        'contact_recall': contact_recall,
        'contact_f1': contact_f1,
        'contact_percent': contact_percent,
        'gt_contact_percent': gt_contact_percent,
        'mpjpe': mpjpe,
        'trans_dist': trans_dist,
        'obj_trans_dist': obj_trans_dist,
        'obj_rot_dist': obj_rot_dist,
        'hand_pen_loss_omomo': hand_pen_loss,
        'hand_pen_ratio': hand_pen_ratio,
        'human_pen_loss_infbagel': human_pen_loss * 10475 / 100,
        'human_pen_ratio': human_pen_ratio
    }

    return metrics, per_sequence_metrics


def _gt_contact_window(sequence_contact, step, cfg):
    """Slice the ground-truth contact window a rollout step actually covers.

    Consecutive windows overlap by ``auto_regre_num``, so the stride is
    ``max_window_size - auto_regre_num`` (14 at the shipped 16/2 settings), NOT
    ``max_window_size``: window ``s`` spans global frames
    ``[s*stride, s*stride + max_window_size)``.  An off-by-``auto_regre_num``
    slice does not crash -- it yields a plausible but wrong ceiling -- so this
    lives in one place and is pinned by tests/test_hoi_guidance_gt_mask.py.
    Short sequences repeat their last annotated frame so the window stays full
    length and the sampler's shape guard still means what it says.
    """
    stride = cfg.max_window_size - cfg.auto_regre_num
    start = step * stride
    frames = sequence_contact.shape[0]
    if start >= frames:
        return sequence_contact[-1:].repeat(cfg.max_window_size, 1)
    piece = sequence_contact[start:start + cfg.max_window_size]
    if piece.shape[0] < cfg.max_window_size:
        pad = cfg.max_window_size - piece.shape[0]
        piece = torch.cat([piece, piece[-1:].repeat(pad, 1)], dim=0)
    return piece


def _hoi_guidance_uses_ground_truth(cfg):
    """True only when the P6 cell-U ground-truth mask probe is explicitly on.

    The probe is NOT deployable: ground-truth contact does not exist at
    inference time.  It exists only to bound how much of the engagement gap
    guidance could recover given a perfect engagement decision, so it must be
    unreachable unless the sampler is configured for it by hand.
    """
    guidance = getattr(getattr(cfg.sampler, "pelvis", None), "guidance", None)
    if guidance is None or not bool(guidance.get("enabled", False)):
        return False
    return str(guidance.get("contact_mask_source", "predicted")) == "ground_truth"


def sample_step(cfg, mat, fixed_points, sampler, scene_flag, text_clip_embedding, pelvis_goal, scene_goal, object_goal,
                need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rest_verts, obj_vert_normals, seq_name_dict, obj_rot_mat_ref_first_step_batch, human_dict, ground_truth_contact=None):
    batch_size = fixed_points.shape[0]
    object_goal_temp = object_goal.clone()
    pelvis_goal = transform_points(pelvis_goal.reshape(batch_size, 1, 3), torch.inverse(mat)).reshape(batch_size, 1, 3) # convert to local coordinates
    scene_goal = transform_points(scene_goal.reshape(batch_size, 1, 3), torch.inverse(mat)).reshape(batch_size, 1, 3)
    object_goal = transform_points(object_goal.reshape(batch_size, 1, 3), torch.inverse(mat)).reshape(batch_size, 1, 3)
    # print(f'pelvis_goal: {pelvis_goal}', 'scene_goal: ', scene_goal, 'object_goal: ', object_goal, 'pi: ', pi, 'need_pi: ', need_pi, 'need_scene: ', need_scene, 'need_pelvis_dir: ', need_pelvis_dir, 'is_object: ', is_object)

    if not cfg.add_object_voxel:
        object_points = None

    # switch via cfg.sample_type: consistency -> consistency model sampling; diffusion -> diffusion model sampling
    if cfg.sample_type == 'consistency':
        guidance_fn = apply_hoi_guidance_loss
        samples, occs = sampler.cm_sample_loop(fixed_points, mat, scene_flag, text_clip_embedding, pelvis_goal, scene_goal,
                                            object_goal, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref_first_step_batch, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, guidance_fn, cfg.guidance_weight, object_only=True, w=cfg.w)
    else:
        samples, occs = sampler.p_sample_loop(fixed_points, mat, scene_flag, text_clip_embedding, pelvis_goal, scene_goal,
                                            object_goal, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref_first_step_batch, obj_rest_verts, seq_name_dict, object_only=True, ground_truth_contact=ground_truth_contact)

    points_gene = samples[-1]
    
    points = points_gene[:, :, :cfg.dataset.nb_joints*3].reshape(batch_size, cfg.max_window_size, cfg.dataset.nb_joints*3)
    points_orig = transform_points(sampler.dataset.denormalize_torch(points), mat)

    global_rot_6d = points_gene[:, :, 84:216].reshape(batch_size, cfg.max_window_size, 22*6)

    obj_trans = points_gene[:, :, 216:219].reshape(batch_size, cfg.max_window_size, 3)
    obj_rot = points_gene[:, :, 219:228].reshape(batch_size, cfg.max_window_size, 3, 3)
    obj_trans_orig = transform_points(sampler.dataset.denormalize_torch(obj_trans, is_object=True), mat)

    contact_label = points_gene[:, :, 228:232].reshape(batch_size, cfg.max_window_size, 4)
    # l_toe_height = points_orig.reshape(batch_size, cfg.max_window_size, cfg.dataset.nb_joints, 3)[:, :, 10, 1:2] # BS X T X 1
    # r_toe_height = points_orig.reshape(batch_size, cfg.max_window_size, cfg.dataset.nb_joints, 3)[:, :, 11, 1:2] # BS X T X 1
    # support_foot_height = torch.minimum(l_toe_height, r_toe_height) # BS X T X 1
    # end_obj_trans_err = torch.linalg.norm(obj_trans_orig[:, -1, :] - object_goal_temp, dim=-1).mean() * 100
    # import pdb; pdb.set_trace()

    info_dict = {
        'points_orig': points_orig.reshape(batch_size, cfg.max_window_size, 3*cfg.dataset.nb_joints),
        'obj_trans_orig': obj_trans_orig,
        'object_rot_mat': obj_rot.reshape(batch_size, cfg.max_window_size, 9),
        'contact_label': contact_label,
        'global_rot_6d': global_rot_6d,
        # 'pelvis_goal': transform_points(pelvis_goal.unsqueeze(1), mat).reshape(batch_size, 3), # global coordinates
        # 'pi': pi,
        # 'need_pi': need_pi,
        # 'need_scene': need_scene,
        # 'need_pelvis_dir': need_pelvis_dir,
        # 'scene_flag': scene_flag,
        # 'scene_goal': scene_goal,
        # 'occ': occs[-1],
    }

    return info_dict


# aggregate evaluation results
def summarize_metrics(all_metrics):
    metrics_summary = {}

    # compute mean values
    for key in all_metrics[0].keys():
        values = [metrics[key] for metrics in all_metrics if key in metrics]
        if values:
            metrics_summary[key] = np.mean(values)

    return metrics_summary

def get_mat(cfg, points):
    batch_size = points.shape[0]
    pelvis_new = points[:, -cfg.auto_regre_num, :9].cpu().numpy().reshape(batch_size, 3, 3)
    trans_mats = np.repeat(np.eye(4)[np.newaxis, :, :], batch_size, axis=0)
    for ip, pn in enumerate(pelvis_new):
        _, ret_R, ret_t = rigid_transform_3D(np.matrix(pn), rest_pelvis, False)
        ret_t[1] = 0.0
        rot_euler = R.from_matrix(ret_R).as_euler('zxy')
        shift_euler = np.array([0, 0, rot_euler[2]])
        shift_rot_matrix2 = R.from_euler('zxy', shift_euler).as_matrix()
        trans_mats[ip, :3, :3] = shift_rot_matrix2
        trans_mats[ip, :3, 3] = ret_t.reshape(-1)
    mat = torch.from_numpy(trans_mats).to(device=cfg.device, dtype=torch.float32)

    return mat

@hydra.main(version_base=None, config_path="config", config_name="config_sample_infbagel")
def test(cfg: DictConfig) -> None:
    device = cfg.device
    seed_everything(int(cfg.seed))
    synchronize_cuda(device)
    end_to_end_start = time.perf_counter()

    seg_id_dict = pkl.load(open(os.path.join(ROOT_DIR, 'data', 'test', 'seq_id.pkl'), 'rb'))
    seg_id_dict = [0] + list(seg_id_dict.values())
    
    rest_verts_root = os.path.join(ROOT_DIR, 'data', 'object', 'rest_object_geo')
    obj_rest_verts = {}
    obj_vert_normals = {}
    for file in sorted(os.listdir(rest_verts_root)):
        if not file.endswith('.ply'):
            continue
        obj_name = file.split('.')[0]
        rest_obj_path = os.path.join(rest_verts_root, file)
        mesh = trimesh.load_mesh(rest_obj_path)
        rest_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_rest_verts[obj_name] = torch.from_numpy(zup_to_yup(rest_verts)).float().to(device)
        vert_normals = np.asarray(mesh.vertex_normals) # Nv X 3
        obj_vert_normals[obj_name] = torch.from_numpy(zup_to_yup(vert_normals)).float().to(device)

    object_sdf_root = os.path.join(ROOT_DIR, 'data', 'object', 'rest_object_sdf_256_npy_files')
    obj_sdf = {}
    obj_sdf_json = {}
    for file in sorted(os.listdir(object_sdf_root)):
        if not file.endswith('.npy'):
            continue
        obj_name = file.split('.')[0]
        sdf_path = os.path.join(object_sdf_root, file)
        obj_sdf[obj_name] = np.load(sdf_path)
        obj_sdf_json[obj_name] = json.load(open(os.path.join(object_sdf_root, f'{file[:-4]}.json'), 'r'))
        
    dataset_sequence_count = len(seg_id_dict) - 1
    if dataset_sequence_count != int(cfg.hoi_expected_sequences):
        raise ValueError(
            f"Expected {cfg.hoi_expected_sequences} HOI sequences, found {dataset_sequence_count}"
        )
    sequence_limit = cfg.get('hoi_sequence_limit')
    if sequence_limit is not None:
        sequence_limit = int(sequence_limit)
        if sequence_limit <= 0 or sequence_limit > dataset_sequence_count:
            raise ValueError(f"Invalid hoi_sequence_limit: {sequence_limit}")
        seg_id_dict = seg_id_dict[:sequence_limit + 1]
    sample_len = len(seg_id_dict) - 1

    output_dir = os.path.abspath(str(cfg.hoi_output_dir))
    if os.path.exists(output_dir):
        raise FileExistsError(f"Refusing to overwrite HOI evaluation output: {output_dir}")
    os.makedirs(output_dir, exist_ok=False)
    if cfg.get('save_chois_eval_npz', False):
        prediction_dir = os.path.abspath(str(cfg.chois_eval_output_dir))
        ground_truth_dir = os.path.abspath(str(cfg.chois_eval_ground_truth_dir))
        for path in (prediction_dir, ground_truth_dir):
            if os.path.exists(path):
                raise FileExistsError(f"Refusing to overwrite CHOIS export directory: {path}")
            os.makedirs(path, exist_ok=False)

    # only evaluate the single ckpt specified by cfg.ckpt_path (no longer iterate over the checkpoint directory)
    model_name = cfg.ckpt_path.split('/')[-1]
    metrics_filename = f"metrics_{model_name.split('.')[0].split('_')[-1]}.pkl"

    checkpoint_metadata = None
    if cfg.get('expert') == 'hoi':
        if not cfg.ckpt_path:
            raise ValueError('HOIPrior evaluation requires ckpt_path')
        checkpoint_sha256 = sha256_file(str(cfg.ckpt_path))
        if cfg.get('checkpoint_sha256') and str(cfg.checkpoint_sha256) != checkpoint_sha256:
            raise ValueError(
                f'HOIPrior checkpoint hash mismatch: {checkpoint_sha256} != {cfg.checkpoint_sha256}'
            )
        model_body, checkpoint_metadata = load_trained_hoi_prior(
            str(cfg.ckpt_path), torch.device(device),
            weight_variant=str(cfg.checkpoint_weight_variant),
        )
        checkpoint_metadata['sha256'] = checkpoint_sha256
    else:
        model_body = init_model(list(cfg.model.values())[0], device=device, eval=True)
    
    print(OmegaConf.to_yaml(cfg))
    
    # initialize the dataset
    synhsi_dataset = InfBaGelDataset(**cfg.dataset)
    window_codec = WindowStateCodec(
        synhsi_dataset.min_torch, synhsi_dataset.max_torch,
        synhsi_dataset.obj_min_torch, synhsi_dataset.obj_max_torch,
        bps_path=Path(ROOT_DIR) / 'code/bps.pt',
    )

    sampler_body = hydra.utils.instantiate(cfg.sampler.pelvis)
    sampler_body.set_dataset_and_model(synhsi_dataset, model_body)

    # prepare the test metric results
    all_metrics = []

    # store results, compute metrics at the end
    points_all = torch.zeros(sample_len, 0, cfg.max_window_size, 3*cfg.dataset.nb_joints).to(device)
    global_rot_6d_all = torch.zeros(sample_len, 0, cfg.max_window_size, 22*6).to(device)
    object_trans_all = torch.zeros(sample_len, 0, cfg.max_window_size, 3).to(device)
    object_rot_mat_all = torch.zeros(sample_len, 0, cfg.max_window_size, 9).to(device)
    
    rest_human_offsets_all = torch.zeros(0, cfg.max_window_size*cfg.interp_s, 24, 3).to(device)
    transl_all = torch.zeros(0, 3).to(device)
    betas_all = torch.zeros(0, 16).to(device)
    gender_all = []
    
    transl_batch = []
    betas_batch = []

    points_all_gt = torch.zeros(0, cfg.max_window_size, 3*cfg.dataset.nb_joints).to(device)
    points_step_all_gt = torch.zeros(0, cfg.max_window_size, 3*cfg.dataset.nb_joints).to(device)

    object_trans_all_gt = torch.zeros(0, cfg.max_window_size, 3).to(device)
    object_rot_mat_all_gt = torch.zeros(0, cfg.max_window_size, 9).to(device)

    points_all_gt_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 3*cfg.dataset.nb_joints).to(device)
    pose_all_gt = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 22, 3).to(device)

    object_trans_all_gt_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 3).to(device)
    object_rot_mat_all_gt_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 9).to(device)

    points_fk_all_gt_48 = torch.zeros(0, cfg.max_window_size*cfg.interp_s-6, 3*(cfg.dataset.nb_joints-4)).to(device)

    start_point_all_gt = torch.zeros(0, 1, 3*cfg.dataset.nb_joints).to(device)
    start_object_trans_all_gt = torch.zeros(0, 1, 3).to(device)
    end_object_trans_all_gt = torch.zeros(0, 1, 3).to(device)
    xy_points_all_gt = torch.zeros(0, 1, 3*cfg.dataset.nb_joints).to(device)

    mat_batch = []
    fixed_points_batch = []
    scene_flag_batch = []
    text_clip_embedding_batch = []
    pelvis_goal_batch = []
    scene_goal_batch = []
    object_goal_batch = []
    need_scene_batch = []
    need_pelvis_dir_batch = []
    pi_batch = []
    end_pi_batch = []
    seq_length_batch = []
    need_pi_batch = []
    is_loco_batch = []
    is_object_batch = []
    obj_bps_data_first_step_batch = []
    obj_rot_mat_ref_first_step_batch = []
    object_points_batch = []
    first_object_points_batch = []
    first_object_trans_batch = []
    # Per-sequence ground-truth contact at the model frame rate, for the
    # preregistered P6 cell-U upper-bound probe only.  Lengths differ per
    # sequence, so this stays a list and is sliced per rollout window.  Filled
    # ONLY when the sampler is configured with contact_mask_source=ground_truth;
    # on every other configuration it stays empty and neither of its two read
    # sites (the timing warmup and the per-step slice) is reached.
    gt_contact_label_batch = []
    # World-frame human rotations at the keyframe grid, retained for the
    # non-deployable teacher-forcing history diagnostic below.
    global_rot_6d_all_gt = torch.zeros(0, cfg.max_window_size, 22*6).to(device)
    # PER-WINDOW ground-truth contact, one row per (sequence, window), for the
    # teacher-forcing diagnostic ONLY.  This deliberately does NOT reuse
    # gt_contact_label_batch + _gt_contact_window: that pair assumes the stored
    # track spans the whole sequence, but data_dict['contact_label'] is a single
    # 16-frame WINDOW (code/datasets/infbagel.py:628-629), so at step 2 the
    # stride-14 start index 28 exceeds the 16 frames available and the slicer
    # falls into its short-sequence branch, returning window 0's LAST frame
    # repeated 16 times instead of global frames 28-29.  Measured on the full
    # protocol: that flips the contact bits in 128 of 438 sequences at step 2
    # (.claude/scratch/tf_prereg/g4_repaired_all438_v2.json).  Reading each
    # window's own contact channel here is exact by construction, and leaves
    # gt_contact_label_batch untouched so the sealed P6 cell-U path stays
    # bit-identical.
    contact_all_gt = torch.zeros(0, cfg.max_window_size, 4).to(device)

    max_len = 3
    teacher_forcing_history = _normalize_teacher_forcing_history(cfg)

    seq_name_dict = {}
    seg_id_true = 0
    for seg_id in range(0, len(seg_id_dict)-1):
        data_dict = synhsi_dataset.__getitem__(seg_id_dict[seg_id])

        seq_name_dict[seg_id_true] = data_dict['seq_name']
        obj_name = seq_name_dict[seg_id_true].split('_')[1]
        seg_id_true += 1
        
        joints, mat, object_trans, object_rot_mat, scene_flag, text_clip_embedding, pelvis_goal, scene_goal, object_goal, \
        need_scene, need_pelvis_dir, pi, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, object_points = data_dict['joints'], data_dict['mat'], data_dict['object_trans'], data_dict['object_rot_mat'], data_dict['scene_flag'], data_dict['text_clip_embedding'], data_dict['pelvis_goal'], data_dict['scene_goal'], data_dict['object_goal'], data_dict['need_scene'], data_dict['need_pelvis_dir'], data_dict['pi'], data_dict['need_pi'], data_dict['is_loco'], data_dict['is_object'], data_dict['obj_bps_data'], data_dict['obj_rot_mat_ref'], data_dict['object_points']
        
        end_pi = data_dict['end_pi']
        seq_length = data_dict['seg_len']

        contact_label = torch.from_numpy(data_dict['contact_label']).to(device)

        global_rot_6d = data_dict['global_rot_6d'].reshape(1, -1, 22*6).to(device)
        rest_human_offsets = torch.from_numpy(data_dict['rest_human_offsets']).to(device)
        transl = torch.from_numpy(data_dict['transl'])[None].to(device)
        betas = torch.from_numpy(data_dict['betas'])[None].to(device)
        transl_all = torch.cat([transl_all, transl], dim=0)
        
        rest_human_offsets_all = torch.cat([rest_human_offsets_all, rest_human_offsets.unsqueeze(0).repeat(48, 1, 1)[None]], dim=0)
        betas_all = torch.cat([betas_all, betas], dim=0)
        gender_all.append(data_dict['gender'])
        
        transl_batch.append(transl.repeat(1, 16, 1))
        betas_batch.append(betas.repeat(1, 16, 1))

        joints = torch.from_numpy(joints).to(device).reshape(1, -1, cfg.dataset.nb_joints*3)
        mat = torch.from_numpy(mat).to(device).reshape(1, 4, 4)
        object_trans = torch.from_numpy(object_trans).to(device).reshape(1, -1, 3)
        object_rot_mat = torch.from_numpy(object_rot_mat).to(device).reshape(1, -1, 9)
        text_clip_embedding = text_clip_embedding.to(device).unsqueeze(0)
        pelvis_goal = torch.from_numpy(pelvis_goal).to(device).unsqueeze(0)
        scene_goal = torch.from_numpy(scene_goal).to(device).unsqueeze(0)
        object_goal = torch.from_numpy(object_goal).to(device).unsqueeze(0)
        obj_bps_data = obj_bps_data.to(device).unsqueeze(0)
        obj_rot_mat_ref = torch.from_numpy(obj_rot_mat_ref).to(device).reshape(1, 3, 3)
        object_points = torch.from_numpy(object_points).reshape(1, -1, 3).to(device) # 1 X 1024 X 3

        # convert to global coordinates
        pelvis_goal = transform_points(pelvis_goal.unsqueeze(1), mat).reshape(cfg.batch_size, 3)
        scene_goal = transform_points(scene_goal.unsqueeze(1), mat).reshape(cfg.batch_size, 3)
        object_goal = transform_points(object_goal.unsqueeze(1), mat).reshape(cfg.batch_size, 3)
        # print("global pelvis_goal: ", pelvis_goal, "scene_goal: ", scene_goal, "object_goal: ", object_goal)

        need_scene, need_pelvis_dir, pi, need_pi, is_loco, is_object = \
            torch.from_numpy(np.array([need_scene])).to(device), \
            torch.from_numpy(np.array([need_pelvis_dir])).to(device), torch.from_numpy(np.array([pi])).to(device), \
            torch.from_numpy(np.array([need_pi])).to(device), torch.from_numpy(np.array([is_loco])).to(device), \
            torch.from_numpy(np.array([is_object])).to(device)
        end_pi, seq_length = torch.from_numpy(np.array([end_pi])).to(device), torch.from_numpy(np.array([seq_length])).to(device)

        # convert to global coordinates
        points_orig = transform_points(synhsi_dataset.denormalize_torch(joints), mat)
        object_trans_orig = transform_points(synhsi_dataset.denormalize_torch(object_trans, is_object=True),mat)

        points_step_all_gt = torch.cat([points_step_all_gt, points_orig], dim=0)
        
        first_object_points = object_points
        object_points = object_points.squeeze(0) # 1024 X 3
        first_object_trans = object_trans_orig[:, 0, :].reshape(1, 3)
        
        fixed_points = points_orig[:, :cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, cfg.dataset.nb_joints*3)
        fixed_points = sampler_body.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))

        obj_fixed = object_trans_orig[:, :cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, -1)
        obj_fixed = sampler_body.dataset.normalize_torch(transform_points(obj_fixed, torch.inverse(mat)), is_object=True)
        obj_rot_fixed = object_rot_mat[:, :cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, -1)

        fixed_contact_label = contact_label[:cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, -1)

        global_rot_6d_fixed = global_rot_6d[:, :cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, -1)
        # merge human and object data
        fixed_points = torch.cat([fixed_points, global_rot_6d_fixed, obj_fixed, obj_rot_fixed, fixed_contact_label], dim=-1) # 84 + 3 + 9 + 4

        mat_batch.append(mat)
        fixed_points_batch.append(fixed_points)
        scene_flag_batch.append(torch.tensor(scene_flag))
        text_clip_embedding_batch.append(text_clip_embedding)
        # pelvis_goal_batch.append(pelvis_goal)
        scene_goal_batch.append(scene_goal)
        object_goal_batch.append(object_goal)
        need_scene_batch.append(need_scene)
        need_pelvis_dir_batch.append(need_pelvis_dir)
        # pi_batch.append(pi)
        need_pi_batch.append(need_pi)
        is_loco_batch.append(is_loco)
        is_object_batch.append(is_object)
        obj_bps_data_first_step_batch.append(obj_bps_data)
        obj_rot_mat_ref_first_step_batch.append(obj_rot_mat_ref)
        object_points_batch.append(object_points)
        first_object_points_batch.append(first_object_points)
        first_object_trans_batch.append(first_object_trans)
        # Preregistered P6 cell U (docs/plan/PHASE_1B_HOI/05_INFERENCE_GUIDANCE.md,
        # "2026-08-04 ... P6", as amended by its 2026-08-21 corrected-mask
        # section).  NON-DEPLOYABLE upper-bound probe: ground-truth contact does
        # not exist at inference time.  Sliced per rollout window below, never
        # used unless the sampler is explicitly configured with
        # contact_mask_source=ground_truth.
        #
        # This MUST be the whole-sequence track, NOT data_dict['contact_label'].
        # The latter is one 16-frame WINDOW (code/datasets/infbagel.py:626-629)
        # while _gt_contact_window slices at stride
        # max_window_size - auto_regre_num = 14, so a 16-frame track degenerates:
        # step 1 returns frames 14,15 then frame 15 repeated 14 times, and step 2
        # returns frame 15 repeated 16 times.  The sealed 2026-08-05 cell U ran on
        # exactly that, which inflated its recorded engagement fraction to
        # 0.7891457382039574 where the corrected track measures
        # 0.6612442922374430 (13902 of 21024 engaged frames; both reproduced
        # CPU-only by .claude/scratch/cellu_fix/verify_mask_arithmetic.py).
        #
        # Independent of the teacher-forcing contact channel: that path does not
        # use this pair at all, it reads its own per-window accumulator
        # contact_all_gt.  This fix repairs the SOURCE for the guidance mask; that
        # one BYPASSES the source for a 2-frame history.  Neither touches
        # _gt_contact_window, whose stride arithmetic is correct as written.
        if _hoi_guidance_uses_ground_truth(cfg):
            gt_contact_label_batch.append(torch.from_numpy(
                synhsi_dataset.sequence_contact_label(seg_id_dict[seg_id])
            ).to(device))

        pelvis_goal_batch_temp = []
        pi_batch_temp = []
        end_pi_batch_temp = []
        seq_length_batch_temp = []
        
        for step in range(max_len):
            data_dict = synhsi_dataset.__getitem__(seg_id_dict[seg_id] + step)
            joints, mat, object_trans, object_rot_mat, pelvis_goal, pi, obj_rot_mat_ref = data_dict['joints'], data_dict['mat'], data_dict['object_trans'], data_dict['object_rot_mat'], data_dict['pelvis_goal'], data_dict['pi'], data_dict['obj_rot_mat_ref']
            
            end_pi, seq_length = data_dict['end_pi'], data_dict['seg_len']

            joints = torch.from_numpy(joints).to(device).reshape(1, -1, cfg.dataset.nb_joints*3)
            mat = torch.from_numpy(mat).to(device).reshape(1, 4, 4)
            object_trans = torch.from_numpy(object_trans).to(device).reshape(1, -1, 3)
            object_rot_mat = torch.from_numpy(object_rot_mat).to(device).reshape(1, -1, 9)
            obj_rot_mat_ref = torch.from_numpy(obj_rot_mat_ref).to(device).reshape(1, 3, 3)
            pelvis_goal = torch.from_numpy(pelvis_goal).to(device).unsqueeze(0)
            pi = torch.from_numpy(np.array([pi])).to(device)

            end_pi, seq_length = torch.from_numpy(np.array([end_pi])).to(device), torch.from_numpy(np.array([seq_length])).to(device)

            pelvis_goal = transform_points(pelvis_goal.unsqueeze(1), mat).reshape(cfg.batch_size, 3)
            pelvis_goal_batch_temp.append(pelvis_goal)
            
            pi_batch_temp.append(pi)
            end_pi_batch_temp.append(end_pi)
            seq_length_batch_temp.append(seq_length)

            # convert to global coordinates
            points_orig = transform_points(synhsi_dataset.denormalize_torch(joints), mat)
            object_trans_orig = transform_points(synhsi_dataset.denormalize_torch(object_trans, is_object=True),mat)

            joints_gt = torch.from_numpy(data_dict['joints_gt']).to(device).reshape(1, -1, cfg.dataset.nb_joints*3) # 1 X 48 X 3*28
            object_rot_mat_gt = torch.from_numpy(data_dict['object_rot_mat_gt']).to(device).reshape(1, -1, 9) # 1 X 48 X 9
            object_trans_gt = torch.from_numpy(data_dict['object_trans_gt']).to(device).reshape(1, -1, 3) # 1 X 48 X 3

            # compare in global coordinates
            points_all_gt = torch.cat([points_all_gt, points_orig], dim=0)
            object_trans_all_gt = torch.cat([object_trans_all_gt, object_trans_orig], dim=0)
            object_rot_mat_global = (object_rot_mat.reshape(1, cfg.max_window_size, 3, 3) @ obj_rot_mat_ref).reshape(1, cfg.max_window_size, 9)
            object_rot_mat_all_gt = torch.cat([object_rot_mat_all_gt, object_rot_mat_global], dim=0)

            if teacher_forcing_history != 'off':
                contact_all_gt = torch.cat([
                    contact_all_gt,
                    torch.as_tensor(
                        data_dict['contact_label'], dtype=contact_all_gt.dtype, device=device,
                    ).reshape(1, cfg.max_window_size, 4),
                ], dim=0)
                # Gated so the default path gains no tensor op and no memory:
                # nothing reads this accumulator unless the diagnostic is on.
                # The lift mirrors the model path's own world-frame conversion
                # at the end of the rollout body, and is cross-checked by the
                # adjacent-window overlap guard in _teacher_forcing_history --
                # consecutive windows reach the same world rotations through
                # different per-window ``mat`` frames, so agreeing there is
                # evidence this lift is the right one.
                global_rot_6d_window = data_dict['global_rot_6d'].to(device).reshape(
                    1, cfg.max_window_size, 22, 6,
                )
                global_jrot_mat_gt_world = mat[:, None, None, :3, :3] @ transforms.rotation_6d_to_matrix(
                    global_rot_6d_window
                )
                global_rot_6d_all_gt = torch.cat([
                    global_rot_6d_all_gt,
                    transforms.matrix_to_rotation_6d(global_jrot_mat_gt_world).reshape(
                        1, cfg.max_window_size, 22*6,
                    ),
                ], dim=0)

            points_all_gt_48 = torch.cat([points_all_gt_48, joints_gt[:, 6:, :]], dim=0)
            object_trans_all_gt_48 = torch.cat([object_trans_all_gt_48, object_trans_gt[:, 6:, :]], dim=0)
            object_rot_mat_all_gt_48 = torch.cat([object_rot_mat_all_gt_48, object_rot_mat_gt[:, 6:, :]], dim=0)

            global_rot_6d_gt = data_dict['global_rot_6d_gt'].to(device) # [48, 22, 6]
            rest_human_offsets = torch.from_numpy(data_dict['rest_human_offsets']).to(global_rot_6d_gt.device) # [24, 3]

            # FK to get joint positions.
            curr_seq_local_jpos = rest_human_offsets.unsqueeze(0).repeat(global_rot_6d_gt.shape[0], 1, 1) # [48, 24, 3]
            curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # [48, 24, 3]
            curr_seq_local_jpos[:, 0, :] = joints_gt.reshape(-1, 28, 3)[:, 0, :]
            
            global_jrot_mat_gt = mat[None, :, :3, :3] @ transforms.rotation_6d_to_matrix(global_rot_6d_gt) # [48, 22, 3, 3]
            local_jrot_mat_gt = synhsi_dataset.quat_ik_torch(global_jrot_mat_gt.reshape(-1, 22, 3, 3)) # [b*t, 22, 3, 3]

            pose_all_gt = torch.cat([pose_all_gt, transforms.matrix_to_axis_angle(local_jrot_mat_gt[6:]).reshape(1, -1, 22, 3)], dim=0)

            _, human_jnts_gt = synhsi_dataset.quat_fk_torch(local_jrot_mat_gt, curr_seq_local_jpos) # [48, 24, 3]
            points_fk_all_gt_48 = torch.cat([points_fk_all_gt_48, human_jnts_gt.reshape(1, 48, -1)[:, 6:, :]], dim=0)

        xy_points_all_gt = torch.cat([xy_points_all_gt, points_all_gt[-max_len:][:, -2, :].unsqueeze(1)], dim=0)
        start_point_all_gt = torch.cat([start_point_all_gt, points_all_gt[-max_len:][0:1, 0, :].unsqueeze(1)], dim=0)
        start_object_trans_all_gt = torch.cat([start_object_trans_all_gt, object_trans_all_gt[-max_len:][0:1, 0, :].unsqueeze(1)], dim=0)
        end_object_trans_all_gt = torch.cat([end_object_trans_all_gt, object_trans_all_gt[-max_len:][-1:, -1, :].unsqueeze(1)], dim=0)

        # if not os.path.exists(os.path.join('t2m_results_48', 'gt', f"{data_dict['seq_name']}.npz")):
        #     if not os.path.exists(os.path.join('t2m_results_48', 'gt')):
        #         os.makedirs(os.path.join('t2m_results_48', 'gt'))

        #     np.savez(os.path.join('t2m_results_48', 'gt', f"{data_dict['seq_name']}.npz"), seq_name=data_dict['seq_name'], \
        #                 global_jpos=yup_to_zup(points_fk_all_gt_48.reshape(-1, 24, 3)).cpu().numpy()) # T X 24 X 3

        pelvis_goal_batch.append(torch.stack(pelvis_goal_batch_temp))
        
        pi_batch.append(torch.stack(pi_batch_temp))
        end_pi_batch.append(torch.stack(end_pi_batch_temp))
        seq_length_batch.append(torch.stack(seq_length_batch_temp))

    transl_batch = torch.stack(transl_batch)
    betas_batch = torch.stack(betas_batch)

    mat_batch = torch.stack(mat_batch).reshape(-1, 4, 4) # 534 X 4 X 4
    fixed_points_batch = torch.stack(fixed_points_batch).reshape(-1, 2, 232) # 534 X 2 X 232
    scene_flag_batch = torch.stack(scene_flag_batch) # 534 X 1
    text_clip_embedding_batch = torch.stack(text_clip_embedding_batch).reshape(-1, 1, 768) # 534 X 1 X 1 X 768
    pelvis_goal_batch = torch.stack(pelvis_goal_batch) # 534 X 7 X 1 X 3
    scene_goal_batch = torch.stack(scene_goal_batch) # 534 X 1 X 3
    object_goal_batch = torch.stack(object_goal_batch) # 534 X 1 X 3
    need_scene_batch = torch.stack(need_scene_batch).reshape(-1) # 534
    need_pelvis_dir_batch = torch.stack(need_pelvis_dir_batch).reshape(-1) # 534
    pi_batch = torch.stack(pi_batch).reshape(-1, max_len) # 534 X 7
    end_pi_batch = torch.stack(end_pi_batch).reshape(-1, max_len) # 534 X 7
    seq_length_batch = torch.stack(seq_length_batch).reshape(-1, max_len) # 534 X 7
    need_pi_batch = torch.stack(need_pi_batch).reshape(-1) # 534
    is_loco_batch = torch.stack(is_loco_batch).reshape(-1) # 534
    is_object_batch = torch.stack(is_object_batch).reshape(-1) # 534
    obj_bps_data_first_step_batch = torch.stack(obj_bps_data_first_step_batch) # 534 X 1 X 1 X 1024 X 3
    obj_rot_mat_ref_first_step_batch = torch.stack(obj_rot_mat_ref_first_step_batch) # 534 X 1 X 3 X 3
    object_points_batch = torch.stack(object_points_batch) # 534 X 3 X 1024
    first_object_points_batch = torch.stack(first_object_points_batch).reshape(sample_len, 1024, 3) # 534 X 1024 X 3
    first_object_trans_batch = torch.stack(first_object_trans_batch) # 534 X 1 X 3

    batch_size = object_points_batch.shape[0]

    if cfg.get('hoi_timing_warmup', True):
        warmup_human = {
            'rest_human_offsets': rest_human_offsets_all[:1, :cfg.max_window_size],
            'transl': transl_batch[:1],
            'betas': betas_batch[:1],
            'gender': gender_all[:1],
        }
        sample_step(
            cfg, mat_batch[:1], fixed_points_batch[:1], sampler_body,
            scene_flag_batch[:1], text_clip_embedding_batch[:1], pelvis_goal_batch[:1, 0],
            scene_goal_batch[:1], object_goal_batch[:1], need_scene_batch[:1],
            need_pelvis_dir_batch[:1], pi_batch[:1, 0], end_pi_batch[:1, 0],
            seq_length_batch[:1, 0], need_pi_batch[:1], is_loco_batch[:1],
            is_object_batch[:1], obj_bps_data_first_step_batch[:1], object_points_batch[:1],
            obj_rest_verts, obj_vert_normals, {0: seq_name_dict[0]},
            obj_rot_mat_ref_first_step_batch[:1], warmup_human,
            ground_truth_contact=(
                # Discarded timing warmup, but the cell-U guard is deliberately
                # strict: hand it the matching window-0 slice for sequence 0
                # rather than relaxing the guard for an untimed path.
                torch.stack([
                    _gt_contact_window(gt_contact_label_batch[0], 0, cfg)
                ]).to(device)
                if _hoi_guidance_uses_ground_truth(cfg) else None
            ),
        )
        synchronize_cuda(device)
        seed_everything(int(cfg.seed))
        if hasattr(sampler_body, 'reset_sampling_audit'):
            sampler_body.reset_sampling_audit()

    generation_seconds = 0.0
    
    for step in range(0, max_len):
        print(f"step: {step}")
        if step != 0:
            history_joints = points.reshape(batch_size, cfg.max_window_size, 28, 3)[:, -cfg.auto_regre_num:]
            history_human_rotation = transforms.rotation_6d_to_matrix(
                global_rot_6d.reshape(batch_size, cfg.max_window_size, 22, 6)
            )[:, -cfg.auto_regre_num:]
            history_object_translation = obj_trans[:, -cfg.auto_regre_num:]
            history_object_rotation = project_to_so3(
                object_rot_mat_global.reshape(batch_size, cfg.max_window_size, 3, 3)
            )[:, -cfg.auto_regre_num:]
            history_contact = contact_label[:, -cfg.auto_regre_num:]
            if teacher_forcing_history != 'off':
                history_joints, history_human_rotation, history_object_translation, \
                    history_object_rotation, history_contact = _teacher_forcing_history(
                        teacher_forcing_history,
                        step,
                        history_joints,
                        history_human_rotation,
                        history_object_translation,
                        history_object_rotation,
                        history_contact,
                        points_all_gt,
                        global_rot_6d_all_gt,
                        object_trans_all_gt,
                        object_rot_mat_all_gt,
                        contact_all_gt,
                        torch.arange(batch_size, device=device),
                        max_len,
                        cfg,
                    )
            fixed_points_batch, current_frame = window_codec.encode(
                history_joints, history_human_rotation,
                global_object_translation=history_object_translation,
                global_object_rotation=history_object_rotation,
                contact=history_contact,
            )
            mat_batch = torch.eye(4, device=device, dtype=fixed_points_batch.dtype)[None].repeat(batch_size, 1, 1)
            mat_batch[:, :3, :3] = current_frame.world_to_local.transpose(-1, -2)
            mat_batch[:, :3, 3] = current_frame.origin
            obj_rot_mat_ref_first_step_batch = current_frame.object_reference[:, None]
            current_bps = recompute_rollout_bps(
                window_codec, current_frame.object_reference, obj_rest_verts, seq_name_dict,
            )
            obj_bps_data_first_step_batch = current_bps[:, None, None]

        human_dict = {'rest_human_offsets': rest_human_offsets_all[:, :cfg.max_window_size], 'transl': transl_batch, 'betas': betas_batch, 'gender': gender_all}

        # Preregistered P6 cell U: slice the ground-truth contact window this
        # rollout step actually covers.  Consecutive windows overlap by
        # auto_regre_num, so the stride is (max_window_size - auto_regre_num),
        # NOT max_window_size -- window s spans global frames
        # [s*stride, s*stride + max_window_size).  Getting this wrong is the
        # expected failure mode and does not crash, which is why the sampler
        # revalidates the shape before using it.  Short sequences are padded by
        # repeating their last annotated frame so the window stays full length.
        gt_contact_window = None
        if _hoi_guidance_uses_ground_truth(cfg):
            gt_contact_window = torch.stack([
                _gt_contact_window(sequence_contact, step, cfg)
                for sequence_contact in gt_contact_label_batch
            ]).to(device)

        synchronize_cuda(device)
        generation_start = time.perf_counter()
        info_dict = sample_step(cfg, mat_batch, fixed_points_batch, sampler_body, scene_flag_batch, text_clip_embedding_batch, pelvis_goal_batch[:,step], scene_goal_batch, object_goal_batch, need_scene_batch, need_pelvis_dir_batch, pi_batch[:,step], end_pi_batch[:,step], seq_length_batch[:,step], need_pi_batch, is_loco_batch, is_object_batch, obj_bps_data_first_step_batch, object_points_batch, obj_rest_verts, obj_vert_normals, seq_name_dict, obj_rot_mat_ref_first_step_batch, human_dict, ground_truth_contact=gt_contact_window)
        synchronize_cuda(device)
        generation_seconds += time.perf_counter() - generation_start
        
        points = info_dict['points_orig'].clone() # 534 X T X 3*28
        obj_trans = info_dict['obj_trans_orig'].clone() # 534 X T X 3
        obj_rot_mat = info_dict['object_rot_mat'].clone() # 534 X T X 9
        contact_label = info_dict['contact_label'].clone() # 534 X T X 4
        global_rot_6d = info_dict['global_rot_6d'].clone() # 534 X T X 22*6

        object_rot_mat_global = project_to_so3(
            obj_rot_mat.reshape(sample_len, cfg.max_window_size, 3, 3)
        ) @ obj_rot_mat_ref_first_step_batch
        object_rot_mat_global = project_to_so3(object_rot_mat_global).reshape(
            sample_len, cfg.max_window_size, 9,
        )
        points_all = torch.cat([points_all, points.unsqueeze(1)], dim=1)
        object_trans_all = torch.cat([object_trans_all, obj_trans.unsqueeze(1)], dim=1)
        object_rot_mat_all = torch.cat([object_rot_mat_all, object_rot_mat_global.unsqueeze(1)], dim=1)
        
        global_jrot_mat = transforms.rotation_6d_to_matrix(global_rot_6d.reshape(sample_len, cfg.max_window_size, 22, 6))
        global_jrot_mat = mat_batch[:, None, None, :3, :3] @ global_jrot_mat
        global_rot_6d = transforms.matrix_to_rotation_6d(global_jrot_mat).reshape(sample_len, cfg.max_window_size, 22*6)

        global_rot_6d_all = torch.cat([global_rot_6d_all, global_rot_6d.unsqueeze(1)], dim=1) # B * S * T * (22*6)

    metrics, per_sequence_metrics = compute_metrics(sampler_body, cfg, points_all, global_rot_6d_all, points_fk_all_gt_48, object_trans_all, object_trans_all_gt_48, object_rot_mat_all, object_rot_mat_all_gt_48, start_point_all_gt, start_object_trans_all_gt, end_object_trans_all_gt, xy_points_all_gt, seq_name_dict, obj_rest_verts, rest_human_offsets_all, transl_all, betas_all, gender_all)
    
    all_metrics.append(metrics)
    
    # aggregate and save evaluation metrics
    metrics_summary = summarize_metrics(all_metrics)

    generated_frames = sample_len * max_len * (cfg.max_window_size - cfg.auto_regre_num) * cfg.interp_s
    synchronize_cuda(device)
    end_to_end_seconds = time.perf_counter() - end_to_end_start
    per_sequence_path = os.path.abspath(str(cfg.get(
        'per_sequence_metrics_path', os.path.join(output_dir, 'per_sequence_metrics.json')
    )))
    per_sequence_payload = {
        'schema_version': 1,
        'seed': int(cfg.seed),
        'sequence_count': len(per_sequence_metrics),
        'metrics': per_sequence_metrics,
    }
    if teacher_forcing_history != 'off':
        # Stamped ONLY when the non-deployable diagnostic is on, and inserted
        # before 'sequence_count' so the key order stays stable for readers.
        # Stamping it unconditionally would add a fifth top-level key to every
        # future eval and break the byte-for-byte reproducibility invariant this
        # file carries (docs/HOIPRIOR_EVIDENCE_INDEX.md:462 -- three guided
        # Arm-B evals on two hosts reproduce it exactly).  Absence of the key
        # therefore means 'off', which is the default and the only deployable
        # setting.
        per_sequence_payload = {
            'schema_version': 1,
            'seed': int(cfg.seed),
            'teacher_forcing_history': teacher_forcing_history,
            'sequence_count': len(per_sequence_metrics),
            'metrics': per_sequence_metrics,
        }
    with open(per_sequence_path, 'x') as handle:
        json.dump(per_sequence_payload, handle, indent=2, default=convert_to_serializable)

    evaluation_result = {
        'schema_version': 2,
        'model_name': model_name,
        'seed': int(cfg.seed),
        'sample_count': sample_len,
        'dataset_sequence_count': dataset_sequence_count,
        'is_timing_subset': sample_len != dataset_sequence_count,
        'windows_per_sample': max_len,
        'metrics': metrics_summary,
        'checkpoint': checkpoint_metadata,
        'data_contract': {
            'contract_sha256': cfg.get('data_contract_sha256'),
            'audit_sha256': cfg.get('data_audit_sha256'),
            'scene_condition_loaded': bool(cfg.load_scene),
            'short_sequence_windows': 0,
            'text_coverage_rate': 1.0,
        },
        'normalization_audit': (
            sampler_body.audit_dict() if hasattr(sampler_body, 'audit_dict') else None
        ),
        'per_sequence_metrics': {
            'path': per_sequence_path,
            'sha256': sha256_file(per_sequence_path),
            'sequence_count': len(per_sequence_metrics),
        },
        'generation_metrics': {
            'warmup_batches_excluded': 1 if cfg.get('hoi_timing_warmup', True) else 0,
            'generated_frames': generated_frames,
            'generation_seconds': generation_seconds,
            'fps': generated_frames / generation_seconds,
            'end_to_end_seconds': end_to_end_seconds,
            'timing_cuda_synchronized': True,
        },
        'chois_export': {
            'enabled': bool(cfg.get('save_chois_eval_npz', False)),
            'prediction_dir': str(cfg.chois_eval_output_dir),
            'ground_truth_dir': str(cfg.chois_eval_ground_truth_dir),
        },
        'execution_provenance': _execution_provenance(device),
    }
    if teacher_forcing_history != 'off':
        # Same rule as per_sequence_metrics.json above: the non-deployable
        # diagnostic announces itself, and the default path emits the exact key
        # set every sealed HOI eval emits.  Placed first so a reader sees it
        # before any metric.
        evaluation_result = {
            'schema_version': 2,
            'teacher_forcing_history': teacher_forcing_history,
            **{k: v for k, v in evaluation_result.items() if k != 'schema_version'},
        }
    metrics_path = os.path.join(output_dir, 'aggregate_metrics.json')
    with open(metrics_path, 'x') as handle:
        json.dump(evaluation_result, handle, indent=2, default=convert_to_serializable)
    
    # if not os.path.exists(os.path.join('results', cfg.exp_name)):
    #     os.makedirs(os.path.join('results', cfg.exp_name))
    # with open(os.path.join('results', cfg.exp_name, metrics_filename), 'wb') as f:
    #     pkl.dump({'all_metrics': all_metrics, 'summary': metrics_summary}, f)

    # print the evaluation metrics summary
    print(f"\n{metrics_filename} Evaluation Metrics Summary:")
    for key, value in metrics_summary.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nTest completed.")


# --- Execution provenance -----------------------------------------------------
#
# Deliberately placed below ``test()`` rather than beside the other helpers at
# the top of the file.  Twenty tracked sites cite absolute line numbers in this
# module (158, 249, 337-345, 377, 384-389, 407, 415, 502, 563-572, 625), and
# seven of them live in ``experiments/results/*.json`` and
# ``experiments/registry.jsonl`` -- append-only records, two of which are
# hash-verified by ``tools/diagnose_hoi_d2p.py`` and ``tools/diagnose_hoi_d2f.py``
# and none of which may be edited to follow a shifted line.  P11's sealed result
# says "FileNotFoundError at code/test_infbagel_hoi.py:502 for
# ../data/test/seq_id.pkl"; inserting anything above line 502 would silently
# falsify that record.  Mid-file imports match this module's existing style
# (lines 70-76).  ``tests/hoi/test_hoi_evaluation_provenance.py`` pins the
# anchor.  The teacher-forcing normalization and pure history substitution
# helpers also stay in this post-``test()`` region for the same append-only
# line-anchor reason; Python resolves their globals when ``test()`` runs.
import platform
import socket
import subprocess


def _normalize_teacher_forcing_history(cfg):
    """Validate the non-deployable history diagnostic and its opt-in guard.

    YAML 1.1 parses a bare ``off`` as ``False``.  Treat that value, ``None``
    and the literal string ``off`` as the same default, while rejecting every
    other unknown mode instead of silently disabling the diagnostic.
    """
    raw_mode = cfg.get('teacher_forcing_history', 'off')
    if raw_mode is False or raw_mode is None:
        mode = 'off'
    elif isinstance(raw_mode, str):
        mode = raw_mode.lower()
    else:
        mode = repr(raw_mode)
    if mode not in {'off', 'full', 'vertical'}:
        raise ValueError(
            "teacher_forcing_history must be one of off/full/vertical; "
            f"got {raw_mode!r}"
        )
    if mode != 'off' and not bool(cfg.get('hoi_diagnostic_not_a_model_score', False)):
        raise ValueError(
            f"teacher_forcing_history={mode!r} uses ground truth and is not a model score; "
            "set hoi_diagnostic_not_a_model_score: true to enable this non-deployable probe"
        )
    return mode


def _teacher_forcing_row_indices(sequence_indices, step, max_len):
    """Return the accumulated ``seq * max_len + step`` rows for a step."""
    if int(step) < 0 or int(step) >= int(max_len):
        raise ValueError(f"teacher-forcing step {step} is outside max_len={max_len}")
    return sequence_indices * int(max_len) + int(step)


def _validate_teacher_forcing_tensor(name, replacement, model_tensor):
    if replacement.shape != model_tensor.shape or replacement.dtype != model_tensor.dtype:
        raise ValueError(
            f"teacher-forcing {name} shape/dtype mismatch: GT {tuple(replacement.shape)}/"
            f"{replacement.dtype} versus model {tuple(model_tensor.shape)}/{model_tensor.dtype}"
        )


def _validate_teacher_forcing_gt_overlap(
    points_all_gt,
    object_trans_all_gt,
    global_rot_6d_all_gt,
    rows,
    step,
    auto_regre_num,
):
    """Fail closed if adjacent accumulated GT windows do not overlap.

    ``.claude/scratch/tf_prereg/check_overlap_invariant.py`` measured worst
    real-data disagreement of 4.768e-7 m (joints), 3.576e-7 m (object
    translation), and 1.788e-7 for human-rotation 6-D values.  The 1e-5
    absolute tolerance is therefore about 20x the measured float32 round-off.
    """
    if int(step) < 1:
        raise ValueError("teacher-forcing GT overlap validation requires step >= 1")
    previous_rows = rows - 1
    for name, accumulated in (
        ('joints', points_all_gt),
        ('object translation', object_trans_all_gt),
        ('human rotation 6d', global_rot_6d_all_gt),
    ):
        current = accumulated.index_select(0, rows)[:, :auto_regre_num]
        previous = accumulated.index_select(0, previous_rows)[:, -auto_regre_num:]
        max_deviation = float(torch.max(torch.abs(current - previous)).item())
        if not torch.allclose(current, previous, atol=1e-5, rtol=0.0):
            raise ValueError(
                f"teacher-forcing GT overlap failed for {name}: row {rows.detach().cpu().tolist()}, "
                f"step {step}, max deviation {max_deviation:.9g}"
            )


def _teacher_forcing_history(
    mode,
    step,
    history_joints,
    history_human_rotation,
    history_object_translation,
    history_object_rotation,
    history_contact,
    points_all_gt,
    global_rot_6d_all_gt,
    object_trans_all_gt,
    object_rot_mat_all_gt,
    contact_all_gt,
    sequence_indices,
    max_len,
    cfg,
):
    """Substitute the first history frames with accumulated world-frame GT.

    This pure tensor helper intentionally returns the caller's exact model
    tensors for ``off`` and for step zero.  The evaluator only calls it from
    the non-off branch at ``step != 0``; keeping the no-op cases explicit makes
    the default path identity-testable without a GPU.
    """
    if mode not in {'off', 'full', 'vertical'}:
        raise ValueError(f"invalid teacher-forcing mode {mode!r}")
    if mode == 'off':
        return (
            history_joints,
            history_human_rotation,
            history_object_translation,
            history_object_rotation,
            history_contact,
        )
    if int(step) == 0:
        raise ValueError(
            "teacher-forcing history must never be applied at step 0: "
            "step 0 history is already dataset ground truth"
        )

    auto_regre_num = int(cfg.auto_regre_num)
    batch_size = int(history_joints.shape[0])
    if sequence_indices.ndim != 1 or int(sequence_indices.shape[0]) != batch_size:
        raise ValueError("teacher-forcing sequence_indices must have one row per model batch item")
    rows = _teacher_forcing_row_indices(sequence_indices, step, max_len).to(
        device=points_all_gt.device, dtype=torch.long,
    )
    independent_rows = torch.stack([
        points_all_gt[int(sequence) * int(max_len) + int(step)]
        for sequence in sequence_indices.detach().cpu().tolist()
    ])
    vectorized_rows = points_all_gt.index_select(0, rows)
    if not torch.equal(independent_rows, vectorized_rows):
        raise ValueError(
            f"teacher-forcing vectorized GT gather disagrees with independent row gather "
            f"for rows {rows.detach().cpu().tolist()} at step {step}"
        )
    if points_all_gt.ndim != 3 or points_all_gt.shape[2] != 28 * 3:
        raise ValueError(f"teacher-forcing points_all_gt has unexpected shape {tuple(points_all_gt.shape)}")
    if global_rot_6d_all_gt.ndim != 3 or global_rot_6d_all_gt.shape[2] != 22 * 6:
        raise ValueError(
            f"teacher-forcing global_rot_6d_all_gt has unexpected shape {tuple(global_rot_6d_all_gt.shape)}"
        )
    _validate_teacher_forcing_gt_overlap(
        points_all_gt,
        object_trans_all_gt,
        global_rot_6d_all_gt,
        rows,
        step,
        auto_regre_num,
    )
    gt_joints = vectorized_rows.reshape(batch_size, -1, 28, 3)
    gt_joints = gt_joints[:, :auto_regre_num]
    if gt_joints.shape != history_joints.shape:
        raise ValueError(
            f"teacher-forcing joints row {rows.tolist()} does not provide the model history shape "
            f"{tuple(history_joints.shape)}"
        )
    if mode == 'vertical':
        substituted_joints = history_joints.clone()
        substituted_joints[..., 1] = gt_joints[..., 1]
    else:
        substituted_joints = gt_joints

    gt_human_rotation = transforms.rotation_6d_to_matrix(
        global_rot_6d_all_gt.index_select(0, rows).reshape(batch_size, -1, 22, 6)
    )[:, :auto_regre_num]
    gt_object_translation = object_trans_all_gt.index_select(0, rows)[:, :auto_regre_num]
    gt_object_rotation = project_to_so3(
        object_rot_mat_all_gt.index_select(0, rows).reshape(batch_size, -1, 3, 3)
    )[:, :auto_regre_num]
    if contact_all_gt.ndim != 3 or contact_all_gt.shape[1:] != (int(cfg.max_window_size), 4):
        raise ValueError(
            f"teacher-forcing contact_all_gt has unexpected shape {tuple(contact_all_gt.shape)}"
        )
    gt_contact = contact_all_gt.index_select(0, rows)[:, :auto_regre_num]

    for name, replacement, model_tensor in (
        ('joints', substituted_joints, history_joints),
        ('human rotation', gt_human_rotation, history_human_rotation),
        ('object translation', gt_object_translation, history_object_translation),
        ('object rotation', gt_object_rotation, history_object_rotation),
        ('contact', gt_contact, history_contact),
    ):
        _validate_teacher_forcing_tensor(name, replacement, model_tensor)
    return (
        substituted_joints,
        gt_human_rotation,
        gt_object_translation,
        gt_object_rotation,
        gt_contact,
    )


def _safe(callback):
    """``callback()``, or None if it raised for any reason.

    Provenance is metadata about an evaluation that has already finished.
    Losing the evaluation because one metadata read failed would be strictly
    worse than recording that field as null, so every field is read through
    here.
    """
    try:
        return callback()
    except Exception:
        return None


def _subprocess_text(command, cwd=None):
    """Stripped stdout of ``command``, or None if it could not be run.

    The timeout is the point of this wrapper as much as the exception handling.
    A hang is not an exception: an ``nvidia-smi`` wedged on a sick driver would
    block here forever, after ``per_sequence_metrics.json`` has been written but
    before ``aggregate_metrics.json``, and both files open with mode ``'x'`` -- so
    the half-finished output would then also block a clean retry.
    ``TimeoutExpired`` is a ``SubprocessError``, so it lands in the same handler.
    """
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _nvidia_driver_version():
    value = _subprocess_text(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']
    )
    if not value:
        return None
    return value.splitlines()[0].strip()


def _gpu_name(device):
    resolved = torch.device(device)
    if resolved.type != 'cuda' or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(resolved)


def _null_git_provenance():
    return {
        'commit': None,
        'branch': None,
        'dirty': None,
        'dirty_entry_count': None,
        'resolved_at': 'metrics_write',
    }


def _git_provenance(repo):
    """Evaluation-time Git identity, or null fields outside a repository.

    ``dirty`` is defined by ``--porcelain=v1 --untracked-files=all``, matching
    ``tools/experiment.py``'s ``git_state`` and ``train_hoi_prior.py``'s
    ``_git_status_porcelain``, so the three cannot disagree about what "clean"
    means.  The porcelain listing itself is deliberately not stored -- only
    whether it was empty and how many entries it had.

    ``resolved_at`` is recorded because this identity is read when
    ``aggregate_metrics.json`` is written rather than at evaluator start.
    AGENTS.md requires trainers to resolve HEAD once at run start precisely
    because a multi-hour run can outlive its own commit (P10's A10/A01 arms
    started at ``91232ad`` and were recorded as ``5d39ac3``).  A full HOI
    evaluation takes about six minutes, so the same exposure exists here but is
    bounded by six minutes rather than a day; reading it at write time is what
    keeps provenance collection entirely downstream of sampling and metric
    computation, which is the stronger requirement for an evaluator whose
    outputs are compared bitwise against sealed baselines.
    """
    record = _null_git_provenance()
    commit = _subprocess_text(['git', 'rev-parse', 'HEAD'], cwd=repo)
    if commit is None:
        return record
    record['commit'] = commit
    record['branch'] = _subprocess_text(['git', 'branch', '--show-current'], cwd=repo) or None
    status = _subprocess_text(
        ['git', 'status', '--porcelain=v1', '--untracked-files=all'], cwd=repo
    )
    if status is not None:
        entries = status.splitlines()
        record['dirty'] = bool(entries)
        record['dirty_entry_count'] = len(entries)
    return record


def _null_execution_provenance():
    return {
        'hostname': None,
        'device': None,
        'gpu_name': None,
        'nvidia_driver_version': None,
        'torch_version': None,
        'cuda_version': None,
        'cudnn_version': None,
        'python_version': None,
        'git': _null_git_provenance(),
    }


def _execution_provenance(device):
    """Host, accelerator and code identity of the process that wrote the metrics.

    ``aggregate_metrics.json`` recorded no execution environment and no
    evaluation-time commit at all; its only commit was ``checkpoint.git_commit``,
    which is the *training* commit.  The sealed baseline
    ``p1-hoi-p8-eval-w3-guided-s42-20260809`` therefore had no recorded code
    identity: it ran 12:24:25-12:27:43 on 2026-08-09 while the commit holding its
    code, ``5e89644``, landed at 12:43:32, about sixteen minutes later.  Closing
    that gap is what this block is for.

    Numerically inert by construction.  It is called once, after the last metric
    has been computed and after ``per_sequence_metrics.json`` has been written;
    it consumes no RNG and adds no synchronization to the sampling loop.

    It also never raises.  A CPU-only host, an absent ``nvidia-smi`` and a
    non-Git directory each yield null fields rather than an exception, because an
    evaluation must not be lost to a failed metadata read.

    These fields make ``aggregate_metrics.json`` host-dependent by design, which
    is exactly why they are not written into ``per_sequence_metrics.json``: that
    file carries no absolute paths, so its hash is a valid cross-host readout and
    several sealed records pin it
    (``bbcd9e1b550d42bf4ac19f9a55db4b9eebb896a8ddb2d562b5226a11b297f6b2``).
    """
    record = _null_execution_provenance()
    try:
        record['hostname'] = _safe(socket.gethostname)
        record['device'] = _safe(lambda: str(device))
        record['gpu_name'] = _safe(lambda: _gpu_name(device))
        record['nvidia_driver_version'] = _safe(_nvidia_driver_version)
        record['torch_version'] = _safe(lambda: str(torch.__version__))
        record['cuda_version'] = _safe(lambda: torch.version.cuda)
        record['cudnn_version'] = _safe(torch.backends.cudnn.version)
        record['python_version'] = _safe(platform.python_version)
        git = _safe(lambda: _git_provenance(Path(__file__).resolve().parents[1]))
        if git is not None:
            record['git'] = git
    except Exception as error:  # defence in depth; every field is already guarded
        record = _null_execution_provenance()
        record['unavailable'] = str(error)
    return record


if __name__ == '__main__':
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ.setdefault('ROOT_DIR', '../')

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))
    test()
