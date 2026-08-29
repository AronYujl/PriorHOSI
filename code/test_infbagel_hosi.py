import os
import time
import hashlib
import pickle as pkl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
import trimesh
import hydra
import json
import torch.nn.functional as F
import random

from utils import *
from constants import *
from datasets.infbagel import InfBaGelDataset
from astar import get_path
from guidance_loss import *
from eval_metrics import *

import pytorch3d.transforms as transforms
import math

METRIC_NAMES = [
    'feet_height', 'foot_sliding', 'hand_pen_loss_omomo', 'hand_pen_ratio',
    'human_pen_loss_infbagel', 'human_pen_ratio',
    'xy_points_err', 'end_obj_trans_err', 'contact_percent',
    'scene_human_penetration_s_mean',
    'scene_human_penetration_s_max', 'scene_human_penetration_frame_ratio',
    'scene_obj_penetration_s_mean',
    'scene_obj_penetration_s_max', 'scene_obj_penetration_frame_ratio',
]


def seed_everything(seed):
    """Seed every RNG used by sampling and metric vertex subsampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def synchronize_cuda(device):
    """Synchronize CUDA explicitly at timing boundaries; no-op on CPU."""
    device = torch.device(device)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON-serializable Python types"""
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

_HAND_IDXS = None
def _get_hand_idxs():
    """Hand vertex index cache: MANO_SMPLX_vertex_ids.pkl content is fixed; read from disk and concatenated only on first call, reused afterward.
    Consumes no global random numbers; the return value is identical to re-reading every time."""
    global _HAND_IDXS
    if _HAND_IDXS is None:
        with open(os.path.join(SMPL_DIR, 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            idxs_data = pkl.load(f)
        _HAND_IDXS = np.concatenate([idxs_data['left_hand'], idxs_data['right_hand']])  # 1556 hand vertices
    return _HAND_IDXS


def sample_step(cfg, step, mat, fixed_points, sampler, cond, trajectory, pi, end_pi, seq_length, obj_bps_data, object_points, obj_rest_verts, obj_vert_normals, seq_name_dict, obj_rot_mat_ref, human_dict, obj_rot_mat_prefix):
    raw_text = cond['raw_text']
    text_emb = cond['text_emb']
    pelvis_goal = cond['pelvis_goal']
    pelvis_goal = transform_points(pelvis_goal.reshape(1, 1, 3), torch.inverse(mat))
    scene_goal = torch.zeros_like(pelvis_goal)
    scene_goal = transform_points(scene_goal.reshape(1, 1, 3), torch.inverse(mat))
    object_goal = cond['object_goal']
    object_goal = transform_points(object_goal.reshape(1, 1, 3), torch.inverse(mat))

    need_scene = cond['need_scene']
    need_pelvis_dir = cond['need_pelvis_dir']
    is_loco = cond['is_loco']
    need_pi = cond['need_pi']

    is_object = cond['need_object']

    speed_new = None
    if is_loco:
        if not is_object and not bool(cfg.get('hsi_progress_fix', False)):
            pi = torch.zeros((cfg.batch_size, ), dtype=torch.long).to(cfg.device)

        curr_loc = mat[0, :3, 3].cpu().numpy()
        curr_loc = np.array([curr_loc[0], curr_loc[2]]).reshape(1, 2)

        dist = np.linalg.norm(curr_loc - trajectory, axis=1)
        min_idx = np.argmin(dist)
        # 2026-08-20: the 0.8 m lookahead is an ARC LENGTH, and was never exercised
        # before episode set v2 (v1 clamped on 0/2271 windows).  Config-gated so the
        # released value stays the default; see config_sample_infbagel_lingo_hsi.yaml.
        lookahead_m = float(cfg.get('hsi_lookahead_m', 0.8))
        base_step = math.ceil(trajectory.shape[0] / np.sum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)) * lookahead_m)
        pelvis_goal = torch.tensor([trajectory[min(min_idx+base_step, len(trajectory)-1)][0], 0,
                                    trajectory[min(min_idx+base_step, len(trajectory)-1)][1]]).reshape(1, 1, 3).to(cfg.device).float()

        pelvis_goal = transform_points(pelvis_goal, torch.inverse(mat))

    if not cfg.use_pi:
        need_pi = torch.zeros((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)
        pi = torch.zeros((cfg.batch_size, ), dtype=torch.long).to(cfg.device)

    print(f'pelvis_goal: {pelvis_goal}', 'object_goal: ', object_goal, 'seq_len: ', seq_length, 'pi: ', pi,  'raw_text: ', raw_text, 'speed: ', speed_new)

    scene_flag = sampler.dataset.scene_dict[cond['scene_name']]
    scene_flag = torch.tensor([scene_flag]*cfg.batch_size).to(cfg.device)

    pelvis_goal = pelvis_goal.reshape(cfg.batch_size, 3)
    scene_goal = scene_goal.reshape(cfg.batch_size, 3)
    object_goal = object_goal.reshape(cfg.batch_size, 3)

    if not cfg.add_object_voxel:
        object_points = None

    human_dict['rest_human_offsets'] = human_dict['rest_human_offsets'][None, None, :].repeat(1, cfg.max_window_size, 1, 1)

    guidance_fn = select_guidance_fn(cfg.use_guidance, is_object)

    if cfg.sample_type == 'consistency':
        samples, occs = sampler.cm_sample_loop(fixed_points, mat, scene_flag, text_emb, pelvis_goal, scene_goal, \
                                            object_goal, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, guidance_fn, cfg.guidance_weight, object_only=False, w=cfg.w, obj_rot_mat_prefix=obj_rot_mat_prefix)
    elif cfg.sample_type == 'diffusion':
        samples, occs = sampler.p_sample_loop(fixed_points, mat, scene_flag, text_emb, pelvis_goal, scene_goal, \
                                            object_goal, need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object, obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, obj_vert_normals, seq_name_dict, human_dict, guidance_fn, cfg.guidance_weight, object_only=False, obj_rot_mat_prefix=obj_rot_mat_prefix)

    points_gene = samples[-1]

    points = points_gene[:, :, :cfg.dataset.nb_joints*3].reshape(cfg.batch_size, cfg.max_window_size, cfg.dataset.nb_joints*3)
    points_orig = transform_points(sampler.dataset.denormalize_torch(points), mat)

    global_rot_6d = points_gene[:, :, 84:216].reshape(cfg.batch_size, cfg.max_window_size, 22*6)

    obj_trans = points_gene[:, :, 216:219].reshape(cfg.batch_size, cfg.max_window_size, 3)
    obj_rot = points_gene[:, :, 219:228].reshape(cfg.batch_size, cfg.max_window_size, 3, 3)
    obj_trans_orig = transform_points(sampler.dataset.denormalize_torch(obj_trans, is_object=True), mat)

    contact_label = points_gene[:, :, 228:232].reshape(cfg.batch_size, cfg.max_window_size, 4)

    global_jrot_mat = transforms.rotation_6d_to_matrix(global_rot_6d.reshape(cfg.batch_size, cfg.max_window_size, 22, 6))
    # Unchanged by the 2026-08-18 frame correction, and correct in either frame:
    # `mat[:3, :3]` is the window frame's local->world rotation (the inverse of
    # the heading shift the dataset applied), so this undoes exactly the shift the
    # channel carries.  It is the same rigid transform whatever world frame the
    # channel lives in.  `get_mat` is likewise unchanged: it reads only the joint
    # channel, fits it to the y-up `constants.rest_pelvis`, and takes the same
    # extrinsic-'zxy' y-euler the dataset does -- so after the correction the two
    # headings agree to 0.017 deg max (they disagreed by 85 deg p50 before).
    global_jrot_mat = mat[:, None, None, :3, :3] @ global_jrot_mat
    global_rot_6d = transforms.matrix_to_rotation_6d(global_jrot_mat).reshape(cfg.batch_size, cfg.max_window_size, 22*6)

    info_dict = {
        'points_orig': points_orig.reshape(cfg.batch_size, cfg.max_window_size, 3*cfg.dataset.nb_joints),
        'obj_trans_orig': obj_trans_orig,
        'object_rot_mat': obj_rot.reshape(cfg.batch_size, cfg.max_window_size, 9),
        'contact_label': contact_label,
        'global_rot_6d': global_rot_6d,
    }

    return info_dict

def get_mat(cfg, points, t):
    batch_size = points.shape[0]
    pelvis_new = points[:, t, :9].cpu().numpy().reshape(batch_size, 3, 3)
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

def get_guidance_from_json(cfg, test_item, max_episode=10):
    """Build guidance conditions from JSON test data"""
    cond = {}

    cond['scene_name'] = test_item['scene_name']

    cond['pelvis_goal'] = torch.tensor(test_item['pelvis_goal']).float().to(cfg.device)
    cond['object_goal'] = torch.tensor(test_item['object_goal']).float().to(cfg.device)

    cond['need_scene'] = torch.ones((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)
    cond['need_pelvis_dir'] = torch.ones((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)
    cond['need_object'] = torch.ones((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)
    cond['is_loco'] = torch.ones((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)
    cond['need_pi'] = torch.ones((cfg.batch_size, ), dtype=torch.bool).to(cfg.device)

    cond['start_location'] = torch.tensor(test_item['start_location']).float().to(cfg.device)
    cond['episode_num'] = max_episode

    return cond



def sha256_file(path, chunk_size=1 << 20):
    """Streamed SHA256 of a file; same helper and same spelling as the HOI evaluator."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _subsample_seed(seed, scene_name, test_idx):
    """A per-episode seed for the object-vertex subsample, independent of run order.

    Keyed on the episode's own identity (scene name and index within the scene) rather
    than on a counter, so an episode draws the same subsample whether it is evaluated
    alone, in a shard, or in the full 469-episode sweep.
    """
    digest = hashlib.sha256(f'{seed}|{scene_name}|{test_idx}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big', signed=False) & ((1 << 63) - 1)


def load_scene_sdf_data(scene_sdf_root):
    """Load SDF data and meta info for all scenes"""
    scene_sdf = {}
    scene_sdf_json = {}

    for file in sorted(os.listdir(scene_sdf_root)):
        if not file.endswith('.npy'):
            continue
        scene_name = file.split('.')[0]
        sdf_path = os.path.join(scene_sdf_root, file)
        scene_sdf[scene_name] = np.load(sdf_path)

        json_path = os.path.join(scene_sdf_root, f'{file[:-4]}_info.json')
        if os.path.exists(json_path):
            scene_sdf_json[scene_name] = json.load(open(json_path, 'r'))

    return scene_sdf, scene_sdf_json

def compute_scene_sdf_penetration(human_verts, scene_name, scene_sdf, scene_sdf_json):
    """
    Compute scene-SDF penetration metrics for human vertices.
    Inputs:
        human_verts: vertex sequence [T, N, 3] (Y-up coordinate system)
        scene_name: scene name
        scene_sdf: scene SDF data dict
        scene_sdf_json: scene SDF meta-info dict
    Outputs:
        penetration_percent: average percentage of penetrating points
        penetration_mean: average penetration depth
        penetration_max: maximum penetration depth
        penetration_frame_ratio: ratio of penetrating frames
    """
    sdf_volume = scene_sdf[scene_name]  # [256, 256, 256]
    sdf_info = scene_sdf_json[scene_name]
    centroid = np.array(sdf_info['centroid'])  # [3]
    extents = np.array(sdf_info['extents'])    # [3]

    sdf_volume = torch.from_numpy(sdf_volume).float()
    human_verts = human_verts.float() if torch.is_tensor(human_verts) else torch.from_numpy(human_verts).float()
    device = human_verts.device
    sdf_volume = sdf_volume.to(device)

    T, N = human_verts.shape[:2]

    centroid = torch.tensor(centroid).to(device).float()
    extents = torch.tensor(extents).to(device).float()

    vertices = human_verts.reshape(1, -1, 3)

    # Normalize to [-1, 1]
    vertices_normalized = (vertices - centroid.reshape(1, 1, 3)) / (extents.reshape(1, 1, 3).max() / 2.0)

    sdf_grids = sdf_volume.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256, 256]

    # grid_sample expects coordinate order [z, y, x]
    sdf_values = F.grid_sample(
        sdf_grids,
        vertices_normalized[:, :, [2, 1, 0]].view(1, T * N, 1, 1, 3),
        padding_mode='border',
        align_corners=True
    ).reshape(T, N)

    sdf_values = sdf_values * extents.max() / 2.

    penetration_masks = (sdf_values < 0)  # [T, N]
    penetration_percent = penetration_masks.float().mean().item()

    negative_distances = torch.minimum(sdf_values, torch.zeros_like(sdf_values))
    penetration_sum_per_frame = negative_distances.abs().sum(dim=-1)  # [T]
    penetration_s_mean = penetration_sum_per_frame.mean().item()

    penetration_s_max = penetration_sum_per_frame.max().item()

    penetrating_frames = (penetration_masks.sum(dim=-1) > 0)  # [T]
    penetration_frame_ratio = penetrating_frames.float().mean().item()

    return penetration_percent, penetration_s_mean, penetration_s_max, penetration_frame_ratio

def compute_metrics_for_sample(points_all, obj_trans, obj_rot, test_item,
                              obj_rest_verts, obj_sdf, obj_sdf_json, synhsi_dataset,
                              verts, joints, transformed_obj_verts, obj_name, scene_sdf, scene_sdf_json, human_faces,
                              subsample_seed=0):
    """Compute evaluation metrics for a single sample

    ``subsample_seed`` seeds the object-vertex subsample below from a dedicated
    generator so this function draws nothing from the global RNG.  See the comment
    at the subsample for the measurement that motivates it.
    """
    metrics = {}

    # Hand vertex index cache: read from disk only on first call, reused afterward
    hand_idxs = _get_hand_idxs()

    T = points_all.shape[0]

    # Foot height metric
    floor_height = determine_floor_height_and_contacts(joints.detach().cpu().numpy())
    metrics['feet_height'] = floor_height.item() * 100

    # Foot sliding metric
    sliding = compute_foot_sliding_for_smpl(joints.detach().cpu().numpy(), floor_height)
    metrics['foot_sliding'] = sliding

    # Hand-object penetration metric using hand-vertex SDF collision detection
    hand_verts = verts[:, hand_idxs, :]
    obj_trans_reshaped = obj_trans.reshape(-1, 3)
    obj_rot_mat = obj_rot.reshape(-1, 3, 3)

    hand_pen_loss_omomo, hand_pen_ratio = compute_collision(
        yup_to_zup(hand_verts), obj_sdf[obj_name], obj_sdf_json[obj_name],
        yup_to_zup_rotation_matrix(obj_rot_mat), yup_to_zup(obj_trans_reshaped)
    )
    metrics['hand_pen_loss_omomo'] = hand_pen_loss_omomo
    metrics['hand_pen_ratio'] = hand_pen_ratio

    human_pen_loss_infbagel, human_pen_ratio = compute_collision(
        yup_to_zup(verts), obj_sdf[obj_name], obj_sdf_json[obj_name],
        yup_to_zup_rotation_matrix(obj_rot_mat), yup_to_zup(obj_trans_reshaped)
    )
    metrics['human_pen_loss_infbagel'] = human_pen_loss_infbagel * 10475 / 100
    metrics['human_pen_ratio'] = human_pen_ratio

    # Goal-reaching metric
    final_pelvis = joints[-1, 0, :]  # pelvis position of the last frame (joint 0)
    final_pelvis[1] = 0
    final_obj_trans = obj_trans[-1]

    target_pelvis = torch.tensor(test_item['pelvis_goal']).float().to(joints.device)
    target_obj = torch.tensor(test_item['object_goal']).float().to(joints.device)

    pelvis_error = torch.norm(final_pelvis - target_pelvis).item() * 100
    obj_error = torch.norm(final_obj_trans - target_obj).item() * 100

    metrics['xy_points_err'] = pelvis_error
    metrics['end_obj_trans_err'] = obj_error

    # Contact percentage using hand joints
    contact_threshold = 0.05

    lhand_idx = 24  # left-hand joint index
    rhand_idx = 26  # right-hand joint index

    lhand_jnt = joints[:, lhand_idx, :]  # [T, 3]
    rhand_jnt = joints[:, rhand_idx, :]  # [T, 3]

    lhand_jnt2obj_dist = torch.norm(lhand_jnt[:, None, :] - transformed_obj_verts, dim=-1)  # T X N
    rhand_jnt2obj_dist = torch.norm(rhand_jnt[:, None, :] - transformed_obj_verts, dim=-1)  # T X N

    lhand_jnt2obj_dist_min = lhand_jnt2obj_dist.min(dim=1)[0]  # T
    rhand_jnt2obj_dist_min = rhand_jnt2obj_dist.min(dim=1)[0]  # T

    lhand_contact = (lhand_jnt2obj_dist_min < contact_threshold)
    rhand_contact = (rhand_jnt2obj_dist_min < contact_threshold)

    contact_frames = (lhand_contact | rhand_contact).sum().item()
    metrics['contact_percent'] = contact_frames / T if T > 0 else 0.0

    # Scene-human penetration
    scene_name = f"{test_item['scene_name']}_sdf"
    _, penetration_s_mean, penetration_s_max, penetration_frame_ratio = compute_scene_sdf_penetration(
        verts, scene_name, scene_sdf, scene_sdf_json
    )
    metrics['scene_human_penetration_s_mean'] = penetration_s_mean
    metrics['scene_human_penetration_s_max'] = penetration_s_max
    metrics['scene_human_penetration_frame_ratio'] = penetration_frame_ratio

    # Downsample object vertices to 10475 points (same sampling across all time frames).
    #
    # 2026-08-29: drawn from a DEDICATED generator, not the global RNG.  Every one of
    # the 13 objects exceeds 10475 vertices (min clothesstand 17996, max suitcase
    # 38353), so this branch fires on all 469 episodes, and the evaluator interleaves
    # sample_step (which draws the diffusion noise) with this call.  Drawing here from
    # the global RNG therefore shifted the noise stream of every LATER episode:
    # measured, episode 0 is unchanged and episodes 1..3 differ from the no-draw run by
    # max abs 5.71 / 6.17 / 5.92.  That made a single episode unreproducible in
    # isolation and a sharded run unequal to a serial one.  Seeding per episode from
    # (seed, scene_name, test_idx) also makes the subsample identical across episodes
    # of the same object instead of a fresh Monte-Carlo draw each time -- two
    # consecutive global draws shared only 6076/10475 indices.
    T, Nv = transformed_obj_verts.shape[:2]
    if Nv > 10475:
        subsample_generator = torch.Generator(device='cpu')
        subsample_generator.manual_seed(subsample_seed)
        indices = torch.randperm(Nv, generator=subsample_generator).to(
            transformed_obj_verts.device
        )[:10475]
        transformed_obj_verts_sampled = transformed_obj_verts[:, indices, :]
    else:
        transformed_obj_verts_sampled = transformed_obj_verts

    # Scene-object penetration
    _, obj_penetration_s_mean, obj_penetration_s_max, obj_penetration_frame_ratio = compute_scene_sdf_penetration(
        transformed_obj_verts_sampled, scene_name, scene_sdf, scene_sdf_json
    )
    metrics['scene_obj_penetration_s_mean'] = obj_penetration_s_mean
    metrics['scene_obj_penetration_s_max'] = obj_penetration_s_max
    metrics['scene_obj_penetration_frame_ratio'] = obj_penetration_frame_ratio

    return metrics

@hydra.main(version_base=None, config_path="config", config_name="config_sample_infbagel")
def main(cfg: DictConfig) -> None:
    cfg.vis = True
    device = cfg.device
    seed_everything(int(cfg.seed))

    # Per-gender SMPL-X models (batch_size=1), created once and reused across all sequences.
    smplx_model_cache = {}

    # Load object geometry data
    rest_verts_root = os.path.join(ROOT_DIR, 'data', 'object', 'rest_object_geo')
    obj_rest_verts = {}
    obj_vert_normals = {}
    obj_faces = {}
    for file in sorted(os.listdir(rest_verts_root)):
        if not file.endswith('.ply'):
            continue
        obj_name = file.split('.')[0]
        rest_obj_path = os.path.join(rest_verts_root, file)
        mesh = trimesh.load_mesh(rest_obj_path)
        rest_verts = np.asarray(mesh.vertices)
        obj_rest_verts[obj_name] = torch.from_numpy(zup_to_yup(rest_verts)).float().to(device)
        vert_normals = np.asarray(mesh.vertex_normals)
        obj_vert_normals[obj_name] = torch.from_numpy(zup_to_yup(vert_normals)).float().to(device)
        obj_faces[obj_name] = torch.from_numpy(np.asarray(mesh.faces)).to(device)

    # Load object SDF data
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

    # Load scene SDF data
    scene_sdf_root = os.path.join(ROOT_DIR, 'data', 'hosi_test', 'Scene_sdf')
    scene_sdf, scene_sdf_json = load_scene_sdf_data(scene_sdf_root)

    model_name = os.path.splitext(cfg.ckpt_path.split('/')[-1])[0]
    base_output_dir = os.path.abspath(str(cfg.hosi_output_dir))
    if cfg.save_results_json:
        if os.path.exists(base_output_dir):
            raise FileExistsError(f"Refusing to overwrite HOSI evaluation output: {base_output_dir}")
        os.makedirs(base_output_dir, exist_ok=False)

    # 2026-08-29: expert dispatch.  `expert: hoi` evaluates HOIPrior -- a Phase 1B
    # checkpoint behind priors.hoi.diffusion.HOIPriorSampler -- on this benchmark.
    # It is a different model class with a different checkpoint schema, so it
    # cannot go through init_model: load_trained_hoi_prior rejects a released
    # InfBaGel state dict outright, which is the rule that the released
    # checkpoint must never initialize an expert.
    #
    # This produces the G==0 anchor of the preregistered gating operator.  Note
    # what it is NOT: the model is scene-blind, but `get_path` (code/astar.py)
    # plans on the scene occupancy and the evaluator feeds a point on that path
    # in as `pelvis_goal` at every window, so the anchor is a scene-blind model
    # under scene-aware waypoint supervision.
    checkpoint_metadata = None
    expert = cfg.get('expert')
    if expert == 'hoi':
        # Imported inside the branch, not at module scope: this module is also
        # imported by code/test_infbagel_lingo_hsi.py, and the default and HSI
        # paths keep their existing import graph.
        from priors.hoi.models import load_trained_hoi_prior
        if not cfg.ckpt_path:
            raise ValueError('HOIPrior evaluation requires ckpt_path')
        checkpoint_sha256 = sha256_file(str(cfg.ckpt_path))
        if cfg.get('checkpoint_sha256') and str(cfg.checkpoint_sha256) != checkpoint_sha256:
            raise ValueError(
                f'HOIPrior checkpoint hash mismatch: {checkpoint_sha256} != {cfg.checkpoint_sha256}'
            )
        model_body, checkpoint_metadata = load_trained_hoi_prior(
            str(cfg.ckpt_path), torch.device(device),
            weight_variant=str(cfg.get('checkpoint_weight_variant', 'online')),
        )
        checkpoint_metadata['sha256'] = checkpoint_sha256
        if cfg.sample_type != 'diffusion':
            raise ValueError(
                'HOIPrior has no consistency-model sampler (it was never '
                f'distilled); sample_type must be diffusion, got {cfg.sample_type!r}'
            )
    elif expert is not None:
        raise ValueError(f'unknown expert for HOSI evaluation: {expert!r}')
    else:
        model_body = init_model(cfg.model.infbagel, device=device, eval=True)

    json_data_dir = os.path.join(ROOT_DIR, 'data', 'hosi_test', 'data')
    scene_files = sorted(f for f in os.listdir(json_data_dir) if f.endswith('.json'))
    # A deliberate scene subset, for cost probes only.  hosi_expected_episodes
    # must be set to null alongside it, or the aggregate guard refuses the run.
    scene_limit = cfg.get('hosi_scene_limit', None)
    if scene_limit is not None:
        scene_files = scene_files[:int(scene_limit)]

    all_scenes_metrics = []
    gen_time_list = []
    fps_list = []
    frames_list = []
    episode_time_list = []
    timing_warmup_sequences = int(cfg.get('timing_warmup_sequences', 5))
    timing_sequence_index = 0

    skipped_scenes = 0

    # [Speedup] Dataset is built only once; later scenes only update scene-related data (see set_test_scene)
    synhsi_dataset = None
    sampler_body = None

    for scene_file in tqdm(scene_files, desc="Processing scenes"):
        scene_name = scene_file.split('.')[0]

        print(f"=== Processing scene: {scene_name} ===")

        with open(os.path.join(json_data_dir, scene_file), 'r') as f:
            test_data = json.load(f)

        print(f"Scene {scene_name}: {len(test_data)} test items")

        # [Speedup] First scene builds the full dataset; later scenes only recompute scene-related data
        cfg.dataset.test_scene_name = scene_name
        if synhsi_dataset is None:
            synhsi_dataset = InfBaGelDataset(**cfg.dataset)
        else:
            synhsi_dataset.set_test_scene(scene_name)
        sampler_body = hydra.utils.instantiate(cfg.sampler.pelvis)
        sampler_body.set_dataset_and_model(synhsi_dataset, model_body)

        scene_metrics = []

        for test_idx, test_item in enumerate(tqdm(test_data, desc=f"Processing {scene_name}")):
            synchronize_cuda(cfg.device)
            _episode_start_time = time.perf_counter()
            print(f"\n=== Test item {test_idx + 1}/{len(test_data)} ===")
            print(f"Scene: {test_item['scene_name']}, Object: {test_item['object_name']}")

            cond = get_guidance_from_json(cfg, test_item)

            data_idx = test_item['data_idx']
            data_dict = synhsi_dataset.__getitem__(data_idx)
            cond['raw_text'] = synhsi_dataset.text[data_idx][0]

            print(f"Text: {cond['raw_text']}, start location: {test_item['start_location']}, "
                  f"pelvis goal: {test_item['pelvis_goal']}, object goal: {test_item['object_goal']}")

            seq_name_dict = {0: data_dict['seq_name']}

            joints = data_dict['joints']
            mat = data_dict['mat']
            object_trans = data_dict['object_trans']
            object_rot_mat = data_dict['object_rot_mat']
            scene_flag = data_dict['scene_flag']
            text_clip_embedding = data_dict['text_clip_embedding']
            obj_bps_data = data_dict['obj_bps_data']
            obj_rot_mat_ref = data_dict['obj_rot_mat_ref']
            object_points = data_dict['object_points']
            end_pi = data_dict['end_pi']
            seq_length = data_dict['seg_len']
            contact_label = torch.from_numpy(data_dict['contact_label']).reshape(1, -1, 4).to(device)
            global_rot_6d = data_dict['global_rot_6d'].reshape(1, -1, 22*6).to(device)
            rest_human_offsets = torch.from_numpy(data_dict['rest_human_offsets']).to(device)
            betas = torch.from_numpy(data_dict['betas']).to(device)
            transl = torch.from_numpy(data_dict['transl']).to(device)
            gender = data_dict['gender']

            joints = torch.from_numpy(joints).to(device).reshape(1, -1, cfg.dataset.nb_joints*3)
            mat = torch.from_numpy(mat).to(device).reshape(1, 4, 4)
            object_trans = torch.from_numpy(object_trans).to(device).reshape(1, -1, 3)
            object_rot_mat = torch.from_numpy(object_rot_mat).to(device).reshape(1, -1, 9)
            text_clip_embedding = text_clip_embedding.to(device).unsqueeze(0)
            obj_bps_data = obj_bps_data.to(device).unsqueeze(0)
            obj_rot_mat_ref = torch.from_numpy(obj_rot_mat_ref).to(device).reshape(1, 3, 3)
            object_points = torch.from_numpy(object_points).reshape(1, -1, 3).to(device)

            cond['text_emb'] = text_clip_embedding

            mat_T = mat[0, :3, :3].T

            points_orig = sampler_body.dataset.denormalize_torch(joints)
            object_trans_orig = sampler_body.dataset.denormalize_torch(object_trans, is_object=True)

            theta = np.arctan2(-cond['pelvis_goal'].cpu().numpy()[2]+cond['start_location'].cpu().numpy()[2],
                                cond['pelvis_goal'].cpu().numpy()[0]-cond['start_location'].cpu().numpy()[0]) + np.pi/2
            rot_matrix = R.from_euler('y', theta).as_matrix()
            MAT = torch.from_numpy(rot_matrix).to(device).float()

            points_orig = points_orig.reshape(cfg.batch_size, cfg.max_window_size, cfg.dataset.nb_joints, 3) @ MAT.t()
            points_orig = points_orig.reshape(cfg.batch_size, cfg.max_window_size, cfg.dataset.nb_joints*3)

            object_trans_orig = object_trans_orig.reshape(cfg.batch_size, cfg.max_window_size, 3) @ MAT.t()
            object_points = object_points.reshape(cfg.batch_size, -1, 3) @ MAT.t()
            object_rot_mat = object_rot_mat.reshape(cfg.batch_size, cfg.max_window_size, 3, 3)

            translation_shift = points_orig[:, [0], :3] - cond['start_location']
            translation_shift[0, 0, 1] = 0.
            points_orig = points_orig.reshape(cfg.batch_size, -1, cfg.dataset.nb_joints, 3)
            points_orig[:, :, :] -= translation_shift
            points_orig = points_orig.reshape(cfg.batch_size, -1, 3*cfg.dataset.nb_joints)
            object_trans_orig[:, :, :] -= translation_shift
            object_points = object_points - translation_shift

            global_jrot_mat = transforms.rotation_6d_to_matrix(global_rot_6d.reshape(-1, 22, 6))
            global_jrot_mat = MAT @ global_jrot_mat
            global_rot_6d = transforms.matrix_to_rotation_6d(global_jrot_mat)

            if cond['is_loco'].any():
                start_loc = cond['start_location'].cpu().numpy()[[0, 2]]
                end_loc = cond['pelvis_goal'].cpu().numpy()[[0, 2]]
                trajectory = get_path(start_loc, end_loc, sampler_body.dataset)
                seg_len = math.ceil(np.sum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)) / 0.8) + 1
            else:
                trajectory = None

            points_all = []
            global_rot_6d_all = []
            object_trans_all = []
            object_rot_mat_all_rel = []
            object_rot_mat_all = []

            _seq_gen_time = 0

            for step in tqdm(range(seg_len), desc="Sampling windows"):
                print(f"  Window {step + 1}/{seg_len}")

                if step == 0:
                    obj_name = seq_name_dict[0].split('_')[1]
                    pred_obj_rot_mat_seg = (MAT @ mat_T @ object_rot_mat[:, 0, :].reshape(1, 3, 3) @ obj_rot_mat_ref).reshape(-1, 3, 3)
                    pred_seq_com_pos_seg = object_trans_orig[:, 0, :].reshape(-1, 3)
                    obj_rest_verts_seg = load_object_geometry_w_rest_geo(pred_obj_rot_mat_seg, pred_seq_com_pos_seg, obj_rest_verts[obj_name])
                    obj_rest_verts_seg = obj_rest_verts_seg.reshape(1, -1, 3)
                    # 2026-08-29: dedicated generator, seeded per (episode, window).
                    # These 1024 points are the object's footprint in the scene
                    # occupancy grid (get_occ_for_points), i.e. MODEL CONDITIONING and
                    # not a metric, and the draw is repeated at every window.  Taking
                    # them from the global RNG made the conditioning depend on run
                    # order, so the same episode conditioned differently when
                    # evaluated alone, in a shard, or in the full sweep.  The
                    # distribution is unchanged -- still a fresh uniform 1024-subset
                    # per window -- only its dependence on call order is removed.
                    object_points_generator = torch.Generator(device='cpu')
                    object_points_generator.manual_seed(
                        _subsample_seed(int(cfg.seed), f'{scene_name}|objpts', test_idx * 10000 + step)
                    )
                    indices = torch.randperm(
                        obj_rest_verts_seg.shape[1], generator=object_points_generator
                    ).to(obj_rest_verts_seg.device)[:1024]
                    object_points = obj_rest_verts_seg[:, indices, :].reshape(1, 1024, 3)

                    mat = get_mat(cfg, points_orig, 0)

                    global_rot_6d = global_rot_6d.reshape(1, cfg.max_window_size, 22, 6)
                    init_global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d[:, 0, 0, :]).reshape(1, 3, 3)
                    init_global_orient = transforms.matrix_to_axis_angle(init_global_rot_mat).cpu().numpy()
                    init_global_orient_euler = R.from_rotvec(init_global_orient).as_euler('zxy')
                    shift_euler = np.zeros_like(init_global_orient_euler)
                    shift_euler[:, 2] = -init_global_orient_euler[:, 2]
                    shift_rot_matrix = R.from_euler('zxy', shift_euler).as_matrix()

                    global_jrot_mat = transforms.rotation_6d_to_matrix(global_rot_6d)
                    global_jrot_mat = torch.from_numpy(shift_rot_matrix).float()[:, None, None].to(device) @ global_jrot_mat

                    mat[:, :3, :3] = torch.from_numpy(np.linalg.inv(shift_rot_matrix)).float().to(device)
                    init_joints = points_orig.reshape(cfg.batch_size, cfg.max_window_size, -1, 3)[:, 0, 0, :].float()
                    mat[:, 0, 3] = init_joints[:, 0]
                    mat[:, 2, 3] = init_joints[:, 2]

                    fixed_points = points_orig[:, :cfg.auto_regre_num, :].reshape(cfg.batch_size, cfg.auto_regre_num, cfg.dataset.nb_joints*3)
                    fixed_points = sampler_body.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))

                    obj_fixed = object_trans_orig[:, :cfg.auto_regre_num].reshape(cfg.batch_size, cfg.auto_regre_num, -1)
                    obj_fixed = sampler_body.dataset.normalize_torch(transform_points(obj_fixed, torch.inverse(mat)), is_object=True)

                    obj_rot_fixed = object_rot_mat[:, :cfg.auto_regre_num].reshape(cfg.batch_size, cfg.auto_regre_num, -1)

                    contact_fixed = contact_label[:, :cfg.auto_regre_num].reshape(cfg.batch_size, cfg.auto_regre_num, -1)

                    global_rot_6d = transforms.matrix_to_rotation_6d(global_jrot_mat).reshape(cfg.batch_size, cfg.max_window_size, 22*6)
                    global_rot_6d_fixed = global_rot_6d[:, :cfg.auto_regre_num].reshape(cfg.batch_size, cfg.auto_regre_num, -1)

                    fixed_points = torch.cat([fixed_points, global_rot_6d_fixed, obj_fixed, obj_rot_fixed, contact_fixed], dim=-1)

                    pi = data_dict['pi']
                else:
                    obj_name = seq_name_dict[0].split('_')[1]
                    pred_obj_rot_mat_seg = (MAT @ mat_T @ object_rot_mat[:, -cfg.auto_regre_num, :].reshape(1, 3, 3) @ obj_rot_mat_ref).reshape(-1, 3, 3)
                    pred_seq_com_pos_seg = obj_trans[:, -cfg.auto_regre_num, :].reshape(-1, 3)
                    obj_rest_verts_seg = load_object_geometry_w_rest_geo(pred_obj_rot_mat_seg, pred_seq_com_pos_seg, obj_rest_verts[obj_name])
                    obj_rest_verts_seg = obj_rest_verts_seg.reshape(1, -1, 3)
                    # Same as the step==0 branch above, same seed scheme, same reason.
                    object_points_generator = torch.Generator(device='cpu')
                    object_points_generator.manual_seed(
                        _subsample_seed(int(cfg.seed), f'{scene_name}|objpts', test_idx * 10000 + step)
                    )
                    indices = torch.randperm(
                        obj_rest_verts_seg.shape[1], generator=object_points_generator
                    ).to(obj_rest_verts_seg.device)[:1024]
                    object_points = obj_rest_verts_seg[:, indices, :].reshape(1, 1024, 3)

                    mat = get_mat(cfg, points, -cfg.auto_regre_num)
                    global_rot_6d = global_rot_6d.reshape(1, cfg.max_window_size, 22, 6)

                    init_global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d[:, -cfg.auto_regre_num, 0, :]).reshape(1, 3, 3)
                    init_global_orient = transforms.matrix_to_axis_angle(init_global_rot_mat).cpu().numpy()
                    init_global_orient_euler = R.from_rotvec(init_global_orient).as_euler('zxy')
                    shift_euler = np.zeros_like(init_global_orient_euler)
                    shift_euler[:, 2] = -init_global_orient_euler[:, 2]
                    shift_rot_matrix = R.from_euler('zxy', shift_euler).as_matrix()

                    global_jrot_mat = transforms.rotation_6d_to_matrix(global_rot_6d)
                    global_jrot_mat = torch.from_numpy(shift_rot_matrix).float()[:, None, None].to(device) @ global_jrot_mat

                    mat[:, :3, :3] = torch.from_numpy(np.linalg.inv(shift_rot_matrix)).float().to(device)
                    init_joints = points.reshape(cfg.batch_size, cfg.max_window_size, -1, 3)[:, -cfg.auto_regre_num, 0, :].float()
                    mat[:, 0, 3] = init_joints[:, 0]
                    mat[:, 2, 3] = init_joints[:, 2]

                    fixed_points = points[:, -cfg.auto_regre_num:, :].reshape(cfg.batch_size, cfg.auto_regre_num, cfg.dataset.nb_joints*3)
                    fixed_points = sampler_body.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))

                    obj_fixed = obj_trans[:, -cfg.auto_regre_num:].reshape(cfg.batch_size, cfg.auto_regre_num, -1)
                    obj_fixed = sampler_body.dataset.normalize_torch(transform_points(obj_fixed, torch.inverse(mat)), is_object=True)

                    obj_rot_fixed = object_rot_mat[:, -cfg.auto_regre_num:].reshape(cfg.batch_size, cfg.auto_regre_num, -1)

                    global_rot_6d = transforms.matrix_to_rotation_6d(global_jrot_mat).reshape(cfg.batch_size, cfg.max_window_size, 22*6)
                    global_rot_6d_fixed = global_rot_6d[:, -cfg.auto_regre_num:].reshape(cfg.batch_size, cfg.auto_regre_num, -1)

                    fixed_contact_label = contact_label[:, -cfg.auto_regre_num:].reshape(cfg.batch_size, cfg.auto_regre_num, -1)

                    fixed_points = torch.cat([fixed_points, global_rot_6d_fixed, obj_fixed, obj_rot_fixed, fixed_contact_label], dim=-1)

                phase = 0
                speed_inter = 3
                pi = torch.tensor([int((step + phase) * (cfg.max_window_size - cfg.auto_regre_num) * speed_inter)]).to(device=cfg.device, dtype=torch.long)
                end_pi = pi + torch.tensor([int(cfg.max_window_size * speed_inter)]).to(device=cfg.device, dtype=torch.long)

                assume_seg_len = seg_len
                seq_length = torch.tensor([int(assume_seg_len * (cfg.max_window_size - cfg.auto_regre_num) * speed_inter + 6)]).to(device=cfg.device, dtype=torch.long)

                human_dict = {
                    'rest_human_offsets': rest_human_offsets,
                    'betas': betas,
                    'transl': transl,
                    'gender': gender
                }

                synchronize_cuda(cfg.device)
                _seq_start_time = time.perf_counter()
                info_dict = sample_step(cfg, step, mat, fixed_points, sampler_body, cond, trajectory, pi, end_pi, seq_length, obj_bps_data, object_points, obj_rest_verts, obj_vert_normals, seq_name_dict, obj_rot_mat_ref, human_dict, MAT @ mat_T)
                synchronize_cuda(cfg.device)
                _seq_end_time = time.perf_counter()
                _seq_gen_time += _seq_end_time - _seq_start_time

                points = info_dict['points_orig'].clone()
                obj_trans = info_dict['obj_trans_orig'].clone()
                object_rot_mat = info_dict['object_rot_mat'].clone()
                contact_label = info_dict['contact_label'].clone()
                global_rot_6d = info_dict['global_rot_6d'].clone()

                object_rot_mat_global = (MAT @ mat_T @ object_rot_mat.reshape(cfg.max_window_size, 3, 3) @ obj_rot_mat_ref).reshape(object_rot_mat.shape)

                if step == seg_len - 1:
                    points_all.append(points.cpu().numpy())
                    object_trans_all.append(obj_trans.cpu().numpy())
                    object_rot_mat_all.append(object_rot_mat_global.cpu().numpy())
                    object_rot_mat_all_rel.append(object_rot_mat.cpu().numpy())
                    global_rot_6d_all.append(global_rot_6d.cpu().numpy())
                else:
                    points_all.append(points.cpu().numpy()[:, :-cfg.auto_regre_num])
                    object_trans_all.append(obj_trans.cpu().numpy()[:, :-cfg.auto_regre_num])
                    object_rot_mat_all.append(object_rot_mat_global.cpu().numpy()[:, :-cfg.auto_regre_num])
                    object_rot_mat_all_rel.append(object_rot_mat.cpu().numpy()[:, :-cfg.auto_regre_num])
                    global_rot_6d_all.append(global_rot_6d.cpu().numpy()[:, :-cfg.auto_regre_num])


            points_all = torch.from_numpy(np.concatenate(points_all, axis=1)).reshape(cfg.batch_size, -1, cfg.dataset.nb_joints, 3)
            object_trans_all = np.concatenate(object_trans_all, axis=1).reshape(-1, 3)
            object_rot_mat_all = np.concatenate(object_rot_mat_all, axis=1).reshape(-1, 9)
            global_rot_6d_all = torch.from_numpy(np.concatenate(global_rot_6d_all, axis=1)).reshape(-1, 22, 6)

            obj_trans, obj_rot_mat = interp_object(object_trans_all, object_rot_mat_all, cfg.interp_s)
            obj_trans, obj_rot_mat = torch.from_numpy(obj_trans).to(device).float(), torch.from_numpy(obj_rot_mat).to(device).reshape(-1, 3, 3).float()

            points_all = interpolate_joints(points_all.reshape(-1, 3*(cfg.dataset.nb_joints)), scale=cfg.interp_s)

            global_rot_mat_all = transforms.rotation_6d_to_matrix(global_rot_6d_all.reshape(-1, 22, 6))
            local_jrot_mat_all = sampler_body.dataset.quat_ik_torch(global_rot_mat_all.reshape(-1, 22, 3, 3))
            local_rot_q_all = transforms.matrix_to_quaternion(local_jrot_mat_all)
            local_rot_q_all = interp_jrot(local_rot_q_all, cfg.interp_s).reshape(-1, 22, 4)
            local_rot_mat_all = transforms.quaternion_to_matrix(local_rot_q_all).reshape(-1, 22, 3, 3)

            # No frame transform on this path.  The released code ran SMPL-X in
            # the z-up world -- yup_to_zup on the translation and the pose, then
            # zup_to_yup on the vertices and joints -- because OMOMO's rotation
            # channel carried global rotations of a zup_to_yup-rotated template
            # and `transl - pelvis` was `-zup_to_yup(J0)` rather than `-J0`.  Both
            # of those are fixed at the source now (datasets/infbagel.py applies a
            # world change of basis to the root only, restores the y-up rest
            # template, and restores the y-up SMPL-X translation), so the decoded
            # locals and the translation are already what SMPL-X's own y-up
            # template expects and the sandwich would reintroduce the error.
            root_trans = (points_all.reshape(-1, 28, 3)[:, 0, :].to(device) + transl)
            pose_pred = transforms.matrix_to_axis_angle(local_rot_mat_all).reshape(-1, 22, 3).to(device)

            if gender not in smplx_model_cache:
                smplx_model_cache[gender] = create_smplx_model(gender, device, batch_size=1)
            human_verts, joints = run_smplx_model(pose_pred, root_trans, betas[None].repeat(root_trans.shape[0], 1), gender, joints_ind=SMPLX_JOINTS_28, smpl_model=smplx_model_cache[gender])
            human_faces = smplx_model_cache[gender].faces

            rest_verts = obj_rest_verts[obj_name][None].repeat(obj_rot_mat.shape[0], 1, 1)
            transformed_obj_verts = obj_rot_mat.bmm(rest_verts.transpose(1, 2)) + obj_trans[:, :, None]
            transformed_obj_verts = transformed_obj_verts.transpose(1, 2)  # T X Nv X 3

            _num_frames = int(joints.shape[0])
            _seq_fps = _num_frames / _seq_gen_time if _seq_gen_time > 0 else 0.0
            synchronize_cuda(cfg.device)
            _episode_time = time.perf_counter() - _episode_start_time
            is_warmup = timing_sequence_index < timing_warmup_sequences
            timing_sequence_index += 1
            if not is_warmup:
                gen_time_list.append(_seq_gen_time)
                fps_list.append(_seq_fps)
                frames_list.append(_num_frames)
                episode_time_list.append(_episode_time)
            print(
                f"Sequence generation time: {_seq_gen_time:.4f}s, frames: {_num_frames}, "
                f"FPS: {_seq_fps:.3f}, end-to-end: {_episode_time:.4f}s, warmup: {is_warmup}"
            )

            print("Computing evaluation metrics...")
            metrics = compute_metrics_for_sample(
                points_all, obj_trans, obj_rot_mat,
                test_item, obj_rest_verts, obj_sdf, obj_sdf_json, synhsi_dataset,
                human_verts, joints, transformed_obj_verts, obj_name,
                scene_sdf, scene_sdf_json, human_faces,
                subsample_seed=_subsample_seed(int(cfg.seed), scene_name, test_idx),
            )

            completed = (metrics['xy_points_err'] < 10.0 and metrics['end_obj_trans_err'] < 10.0)
            metrics['completed'] = completed
            metrics['scene_name'] = test_item['scene_name']
            metrics['object_name'] = test_item['object_name']
            metrics['test_idx'] = test_idx

            scene_metrics.append(metrics)

            print(f"  feet_height: {metrics['feet_height']:.2f}")
            print(f"  foot_sliding: {metrics['foot_sliding']:.2f}")
            print(f"  hand_pen_loss_omomo: {metrics['hand_pen_loss_omomo']:.2f}")
            print(f"  hand_pen_ratio: {metrics['hand_pen_ratio']:.2f}")
            print(f"  human_pen_loss_infbagel: {metrics['human_pen_loss_infbagel']:.2f}")
            print(f"  human_pen_ratio: {metrics['human_pen_ratio']:.2f}")
            print(f"  xy_points_err: {metrics['xy_points_err']:.2f}")
            print(f"  end_obj_trans_err: {metrics['end_obj_trans_err']:.2f}")
            print(f"  contact_percent: {metrics['contact_percent']:.2f}")
            print(f"  scene_human_penetration_s_mean: {metrics['scene_human_penetration_s_mean']:.2f}")
            print(f"  scene_human_penetration_s_max: {metrics['scene_human_penetration_s_max']:.2f}")
            print(f"  scene_human_penetration_frame_ratio: {metrics['scene_human_penetration_frame_ratio']:.3f}")
            print(f"  scene_obj_penetration_s_mean: {metrics['scene_obj_penetration_s_mean']:.2f}")
            print(f"  scene_obj_penetration_s_max: {metrics['scene_obj_penetration_s_max']:.2f}")
            print(f"  scene_obj_penetration_frame_ratio: {metrics['scene_obj_penetration_frame_ratio']:.3f}")
            print(f"  completed: {completed}")

        if scene_metrics:
            all_scenes_metrics.extend(scene_metrics)
        else:
            skipped_scenes += 1

    # 2026-08-29: expected-episode guard, matching hoi_expected_sequences on the HOI
    # path.  Without it a partial sweep emitted a complete-looking aggregate over
    # however many episodes finished -- and `skipped_scenes` was initialized and
    # printed but never incremented, so it reported 0 whatever happened.  The
    # benchmark is 67 scenes x 7 objects = 469 episodes; set
    # hosi_expected_episodes: null to evaluate a deliberate subset.
    expected_episodes = cfg.get('hosi_expected_episodes', 469)
    if expected_episodes is not None and len(all_scenes_metrics) != int(expected_episodes):
        raise ValueError(
            f"HOSI evaluation produced {len(all_scenes_metrics)} episodes, expected "
            f"{int(expected_episodes)} ({len(scene_files)} scene files, "
            f"{skipped_scenes} scenes contributed nothing). Refusing to report an "
            f"aggregate over an incomplete sweep; set hosi_expected_episodes=null to "
            f"evaluate a subset deliberately."
        )

    if all_scenes_metrics:
        print(f"\n=== Overall Evaluation Results ===")

        statistics = {}
        for metric_name in METRIC_NAMES:
            values = [m[metric_name] for m in all_scenes_metrics if metric_name in m and m[metric_name] is not None]
            if values:
                statistics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }

        completed_count = sum(1 for m in all_scenes_metrics if m.get('completed', False))
        statistics['completion_rate'] = completed_count / len(all_scenes_metrics)
        statistics['total_samples'] = len(all_scenes_metrics)
        statistics['completed_samples'] = completed_count

        generation_metrics = None
        if gen_time_list and fps_list and frames_list:
            generation_metrics = {
                'aits': float(np.mean(gen_time_list)),
                'avg_fps': float(np.mean(fps_list)),
                'aggregate_fps': float(np.sum(frames_list) / np.sum(gen_time_list)),
                'total_generation_seconds': float(np.sum(gen_time_list)),
                'timed_sequence_count': len(gen_time_list),
                'avg_frames_per_seq': float(np.mean(frames_list)),
                'avg_end_to_end_episode_seconds': float(np.mean(episode_time_list)),
                'llm_planning_seconds': None,
                'warmup_sequences_excluded': timing_warmup_sequences,
                'timing_cuda_synchronized': True,
            }

        # 2026-08-29: record the guidance state IN THE PAYLOAD.  A HOI run id that
        # says "guided" is not evidence that it was: --eval-tag does not set
        # guidance, and on this path `use_guidance` selects only whether the
        # EVALUATOR passes a guidance_fn, which is always false for the HOI expert
        # while HOIPrior's own preregistered guidance is configured on the sampler.
        # The only trustworthy record is what the sampler reports about itself.
        sampler_audit = None
        if sampler_body is not None and hasattr(sampler_body, 'audit_dict'):
            sampler_audit = sampler_body.audit_dict()

        evaluation_results = {
            'model_name': model_name,
            'seed': int(cfg.seed),
            'expert': expert,
            'sample_type': str(cfg.sample_type),
            'evaluator_guidance_fn': bool(cfg.use_guidance),
            'checkpoint': checkpoint_metadata,
            'sampler_audit': sampler_audit,
            'scene_order': scene_files,
            'individual_metrics': all_scenes_metrics,
            'statistics': statistics,
            'summary': {
                'total_evaluated': len(all_scenes_metrics),
                'completion_rate': statistics['completion_rate'],
                'key_metrics': {
                    'avg_feet_height': statistics.get('feet_height', {}).get('mean', 0),
                    'avg_foot_sliding': statistics.get('foot_sliding', {}).get('mean', 0),
                    'avg_hand_pen_loss_omomo': statistics.get('hand_pen_loss_omomo', {}).get('mean', 0),
                    'avg_hand_pen_ratio': statistics.get('hand_pen_ratio', {}).get('mean', 0),
                    'avg_human_pen_loss_infbagel': statistics.get('human_pen_loss_infbagel', {}).get('mean', 0),
                    'avg_human_pen_ratio': statistics.get('human_pen_ratio', {}).get('mean', 0),
                    'avg_pelvis_error': statistics.get('xy_points_err', {}).get('mean', 0),
                    'avg_object_error': statistics.get('end_obj_trans_err', {}).get('mean', 0),
                    'avg_contact_percent': statistics.get('contact_percent', {}).get('mean', 0),
                    'avg_scene_human_penetration_s_mean': statistics.get('scene_human_penetration_s_mean', {}).get('mean', 0),
                    'avg_scene_human_penetration_s_max': statistics.get('scene_human_penetration_s_max', {}).get('mean', 0),
                    'avg_scene_human_penetration_frame_ratio': statistics.get('scene_human_penetration_frame_ratio', {}).get('mean', 0),
                    'avg_scene_obj_penetration_s_mean': statistics.get('scene_obj_penetration_s_mean', {}).get('mean', 0),
                    'avg_scene_obj_penetration_s_max': statistics.get('scene_obj_penetration_s_max', {}).get('mean', 0),
                    'avg_scene_obj_penetration_frame_ratio': statistics.get('scene_obj_penetration_frame_ratio', {}).get('mean', 0)
                }
            }
        }

        if generation_metrics is not None:
            evaluation_results['summary']['generation_metrics'] = generation_metrics

        if cfg.save_results_json:
            overall_eval_path = os.path.join(base_output_dir, 'overall_evaluation_summary.json')
            with open(overall_eval_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=convert_to_serializable)
            aggregate_eval_path = os.path.join(base_output_dir, 'aggregate_metrics.json')
            aggregate_results = {
                'model_name': evaluation_results['model_name'],
                'seed': evaluation_results['seed'],
                'expert': evaluation_results['expert'],
                'sample_type': evaluation_results['sample_type'],
                'evaluator_guidance_fn': evaluation_results['evaluator_guidance_fn'],
                'checkpoint': evaluation_results['checkpoint'],
                'sampler_audit': evaluation_results['sampler_audit'],
                'statistics': evaluation_results['statistics'],
                'summary': evaluation_results['summary'],
            }
            with open(aggregate_eval_path, 'x') as f:
                json.dump(aggregate_results, f, indent=2, default=convert_to_serializable)
            print(f"\nAll results saved to: {base_output_dir}")

        print(f"Total evaluated: {len(all_scenes_metrics)} samples")
        print(f"Skipped scenes: {skipped_scenes}")
        print(f"Processed scenes: {len(scene_files) - skipped_scenes}")
        print(f"Task completion rate: {statistics['completion_rate']:.2%}")

        summary = evaluation_results['summary']['key_metrics']
        print(f"\nKey Metrics (Average):")
        print(f"  Feet Height: {summary['avg_feet_height']:.2f}")
        print(f"  Foot Sliding: {summary['avg_foot_sliding']:.2f}")
        print(f"  Hand Penetration Loss: {summary['avg_hand_pen_loss_omomo']:.2f}")
        print(f"  Hand Penetration Ratio: {summary['avg_hand_pen_ratio']:.3f}")
        print(f"  Human Penetration Loss: {summary['avg_human_pen_loss_infbagel']:.2f}")
        print(f"  Human Penetration Ratio: {summary['avg_human_pen_ratio']:.3f}")
        print(f"  Pelvis Position Error: {summary['avg_pelvis_error']:.2f}")
        print(f"  Object Position Error: {summary['avg_object_error']:.2f}")
        print(f"  Contact Percentage: {summary['avg_contact_percent']:.2f}")
        print(f"  Scene-Human Penetration Mean: {summary['avg_scene_human_penetration_s_mean']:.2f}")
        print(f"  Scene-Human Penetration Max: {summary['avg_scene_human_penetration_s_max']:.2f}")
        print(f"  Scene-Human Penetration Frame Ratio: {summary['avg_scene_human_penetration_frame_ratio']:.3f}")
        print(f"  Scene-Object Penetration Mean: {summary['avg_scene_obj_penetration_s_mean']:.2f}")
        print(f"  Scene-Object Penetration Max: {summary['avg_scene_obj_penetration_s_max']:.2f}")
        print(f"  Scene-Object Penetration Frame Ratio: {summary['avg_scene_obj_penetration_frame_ratio']:.3f}")

        if gen_time_list and fps_list and frames_list:
            print("\n=== Generation Latency and Rate Statistics (all sequences) ===")
            print(f"AITS (avg inference time per sequence): {generation_metrics['aits']:.4f}s")
            print(f"FPS (avg frame rate): {generation_metrics['avg_fps']:.3f}")
            print(f"Avg frames per sequence: {generation_metrics['avg_frames_per_seq']:.1f}")
            print(f"End-to-end episode latency: {generation_metrics['avg_end_to_end_episode_seconds']:.4f}s")
            print("LLM planning latency: N/A (Atomic-HOSI baseline has no LLM planner)")
    else:
        print("No test items successfully processed!")

if __name__ == "__main__":
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['ROOT_DIR'] = '../'

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))
    main()
