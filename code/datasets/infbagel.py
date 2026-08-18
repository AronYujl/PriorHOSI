import os
from collections import OrderedDict

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
import pickle as pkl
import trimesh
import random
from datasets.utils import (get_occupancy_from_npy, zup_to_yup, yup_to_zup,
                            get_smpl_parents, resolve_asset_world_up,
                            rest_offsets_to_yup, world_up_correction,
                            apply_world_correction_to_axis_angle)

from bps_torch.bps import bps_torch
import pytorch3d.transforms as transforms
from scipy.ndimage import distance_transform_edt


def _compute_single_occ_ref(occ):
    dist_transform = distance_transform_edt(occ, return_distances=True, return_indices=True)
    indices = np.array(dist_transform[1])  # [3, W, H, D]

    w, h, d = occ.shape
    x, y, z = np.meshgrid(np.arange(w), np.arange(h), np.arange(d), indexing='ij')
    coords = np.stack([x, y, z], axis=0)  # [3, W, H, D]

    displacements = indices - coords  # [3, W, H, D]
    return np.transpose(displacements, (1, 2, 3, 0))


class LazyOccRef:
    def __init__(self, occ, capacity=4):
        self.occ = occ
        self.capacity = capacity
        self.cache = OrderedDict()

    def __len__(self):
        return self.occ.shape[0]

    def __getitem__(self, scene_id):
        if isinstance(scene_id, int):
            return self._get(scene_id)
        return torch.stack([self._get(sid) for sid in scene_id.tolist()])

    def _get(self, scene_id):
        if scene_id in self.cache:
            self.cache.move_to_end(scene_id)
            return self.cache[scene_id]

        displacements = _compute_single_occ_ref(self.occ[scene_id].cpu().numpy())
        occ_ref = torch.from_numpy(displacements).to(device=self.occ.device, dtype=torch.int16)

        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[scene_id] = occ_ref
        return occ_ref


class InfBaGelDataset(Dataset):
    def __init__(self, folder, device, mesh_grid, batch_size, step, nb_voxels, train=True,
                 load_scene=True, load_language=True, load_pelvis_goal=False, load_scene_goal=False,
                 load_object_goal=False, load_object_payload=True,
                 use_random_frame_bps=False, use_object_keypoints=False,
                 max_window_size=16,
                 use_pi=True,
                 vis=True,
                 start_type='stand',
                 test_scene_name=None,
                 asset_world_up='auto',
                 **kwargs):

        self.folder = folder
        self.device = device
        self.train = train
        self.load_scene = load_scene
        self.load_language = load_language
        self.load_pelvis_goal = load_pelvis_goal
        self.load_scene_goal = load_scene_goal
        self.load_object_goal = load_object_goal
        # Per-sample object tensors: the ~19 GB of object_points / cano BPS /
        # contact labels that only __getitem__ reads.  A corpus that draws no
        # sample from this dataset (lingo_only) still needs its normalization
        # scalars and its scene table, but never these.
        self.load_object_payload = load_object_payload
        self.use_pi = use_pi
        self.vis = vis
        self.start_type = start_type
        self.test_scene_name = test_scene_name
        self.max_window_size = max_window_size

        self.rest_object_geo_folder = os.path.join(folder, 'rest_object_geo')
        # Loaded verbatim.  The world-frame normalization happens once, below,
        # after rest_human_offsets_aligned is available -- it is the array that
        # decides the frame, and normalizing the rotations without it is how the
        # two frames got mixed in the first place.
        self.global_orient = np.load(os.path.join(folder, 'human_orient.npy'))

        # human_pose holds parent-relative local rotations.  They are properties
        # of the SMPL template, not of the world frame, so no world change of
        # basis applies to them and none is performed.  Verified: forward
        # kinematics from the untouched locals plus the corrected root plus the
        # y-up rest template reproduces human_joints_aligned.npy to 4e-7 m on
        # LINGO and 8e-7 m on OMOMO, against 0.56 m for every other combination.
        self.human_pose = np.load(os.path.join(folder, 'human_pose.npy'))

        self.transl = np.load(os.path.join(folder, 'transl_aligned.npy'))
        
        betas_path = os.path.join(folder, 'betas.npy')
        self.betas = np.load(betas_path)

        gender_path = os.path.join(folder, 'gender.pkl')
        with open(gender_path, 'rb') as f:
            self.gender = pkl.load(f)
        
        self.joints = np.load(os.path.join(folder, 'human_joints_aligned.npy'))
        self.ori_sequence_start_idx = np.load(os.path.join(folder, 'start_idx.npy')).astype(np.int64)
        self.ori_sequence_end_idx = np.load(os.path.join(folder, 'end_idx.npy')).astype(np.int64)

        self.use_random_frame_bps = use_random_frame_bps
        self.use_object_keypoints = use_object_keypoints

        self.use_pen_loss = kwargs.get('use_pen_loss', False)

        self.parents_22 = get_smpl_parents(use_joints24=False) # 22
        self.parents_24 = get_smpl_parents(use_joints24=True) # 24
        if self.load_object_goal:
            # load object data
            # Object sequence ids are required even for the scene-free HOIPrior
            # evaluator.  Loading this metadata must not imply loading Scene*.
            with open(os.path.join(folder, 'scene_name.pkl'), 'rb') as f:
                self.scene_name = pkl.load(f)
            self.object_rot_mat = np.load(os.path.join(folder, 'object_rot_mat.npy'))
            self.object_trans = np.load(os.path.join(folder, 'object_trans.npy'))
            if self.load_object_payload and os.path.exists(os.path.join(folder, 'object_points.npy')):
                self.object_points = np.load(os.path.join(folder, 'object_points.npy'))
            else:
                self.object_points = None
            if self.use_object_keypoints:
                pass
                # self.transformed_obj_verts = np.load(os.path.join(folder, 'transformed_obj_verts.npy'))
                # self.rest_pose_obj_nn_pts = np.load(os.path.join(folder, 'rest_pose_obj_nn_pts.npy'))
            with open(os.path.join(folder, 'object_name.pkl'), 'rb') as f:
                self.object_name = pkl.load(f)
            # self.ori_w_idx = np.load(os.path.join(folder, 'ori_w_idx.npy'))
            if True: # self.train:
                self.dest_obj_bps_npy_folder = os.path.join(folder, 'cano_object_bps_npy_files_joints24_120')
            else:
                self.dest_obj_bps_npy_folder = os.path.join(folder, 'cano_object_bps_npy_files_for_test_joints24_120')
            self.rest_object_geo_folder = os.path.join(folder, 'rest_object_geo')

            self.obj_rest_verts = {}
            self.obj_vert_normals = {}
            for file in os.listdir(self.rest_object_geo_folder):
                if not file.endswith('.ply'):
                    continue
                obj_name = file.split('.')[0]
                rest_obj_path = os.path.join(self.rest_object_geo_folder, file)
                mesh = trimesh.load_mesh(rest_obj_path)
                rest_verts = np.asarray(mesh.vertices) # Nv X 3
                self.obj_rest_verts[obj_name] = torch.from_numpy(zup_to_yup(rest_verts)).float()
                vert_normals = np.asarray(mesh.vertex_normals) # Nv X 3
                self.obj_vert_normals[obj_name] = torch.from_numpy(zup_to_yup(vert_normals)).float()

        else:
            self.object_rot_mat = None
            self.object_trans = None
            self.object_points = None
            self.object_name = None
        
        self.rest_human_offsets = np.load(os.path.join(folder, 'rest_human_offsets_aligned.npy'))

        # ------------------------------------------------------------------
        # One y-up world, one y-up template.
        #
        # joints / scene occupancy / pelvis and scene goals are y-up in every
        # corpus, but human_orient is y-up only in LINGO (data/dataset) and z-up
        # in OMOMO (data/train, data/test), and rest_human_offsets_aligned.npy
        # ships with zup_to_yup already applied to the y-up SMPL template.  The
        # released code reconciled that with zup_to_yup(human_orient) +
        # zup_to_yup(human_pose), which is a *conjugation*: it happens to cancel
        # against the conjugated template on OMOMO and does not cancel on LINGO,
        # leaving the LINGO rotation channel 90 deg about +x away from its own
        # joint channel and every FK against rest_human_offsets wrong by 0.56 m.
        #
        # resolve_asset_world_up decides the frame functionally -- FK from the
        # candidate correction must reproduce human_joints_aligned.npy -- and
        # raises rather than guessing.  The two hypotheses are ~1e-7 m against
        # ~1.2 m apart, so there is no grey zone.
        self.asset_world_up, self.asset_frame_probe_errors = resolve_asset_world_up(
            self.global_orient, self.human_pose, self.joints, self.rest_human_offsets,
            self.ori_sequence_start_idx, self.parents_22, requested=asset_world_up,
        )
        self.rest_human_offsets = rest_offsets_to_yup(self.rest_human_offsets)
        if self.asset_world_up != 'y':
            self.global_orient = apply_world_correction_to_axis_angle(
                self.global_orient, world_up_correction(self.asset_world_up),
            )
            # transl_aligned carries the same baked-in conjugation the template
            # does: transl - pelvis is -zup_to_yup(J0) rather than -J0.  Restore
            # the y-up SMPL-X translation so `pelvis + (transl - pelvis)` is the
            # value SMPL-X's own y-up template expects, and the caller needs no
            # yup_to_zup/zup_to_yup sandwich around the forward pass.
            pelvis = np.asarray(self.joints[:, 0], dtype=np.float64)
            self.transl = pelvis - yup_to_zup(pelvis - np.asarray(self.transl, dtype=np.float64))

        if self.load_language:
            if self.max_window_size == 16:
                language_motion_dict_filename = 'language_motion_dict__inter_and_loco__16.pkl'

            with open(os.path.join(self.folder, 'language_motion_dict', language_motion_dict_filename), 'rb') as f:
                language_motion_dict = pkl.load(f)
            self.end_range = language_motion_dict['end_range']
            self.text = language_motion_dict['text']

            self.clip_features = np.load(os.path.join(self.folder, 'clip_features.npy'))
            with open(os.path.join(self.folder, 'text2features_idx.pkl'), 'rb') as f:
                self.text2features_idx = pkl.load(f)

            self.need_scene = language_motion_dict['need_scene']
            self.need_pelvis_dir = language_motion_dict['need_pelvis_dir']
            self.pi = language_motion_dict['pi']
            self.need_pi = language_motion_dict['need_pi']
            self.left_hand_inter_frame = language_motion_dict['left_hand_inter_frame']
            self.right_hand_inter_frame = language_motion_dict['right_hand_inter_frame']

            self.start_ind = language_motion_dict['start_idx']
            self.end_ind = language_motion_dict['end_idx']
            
            if self.load_object_goal:
                self.need_object = language_motion_dict['need_object']
            
            self.ori_sequence_idx = language_motion_dict['ori_sequence_idx']

        self.step = step
        self.batch_size = batch_size

        if self.load_scene:
            self.mesh_grid = mesh_grid
            self.nb_voxels = nb_voxels
            self.scene_occ = []
            self.scene_occ_ref = []
            self.scene_dict = {}
            if not hasattr(self, 'scene_name'):
                with open(os.path.join(folder, 'scene_name.pkl'), 'rb') as f:
                    self.scene_name = pkl.load(f) # list of scene names
            if not self.vis:
                self.scene_folder = os.path.join(folder, 'Scene')
                scene_file_list = sorted(os.listdir(self.scene_folder))
            else:
                self.scene_folder = os.path.join(folder, 'Scene_vis')
                scene_file_list = sorted(os.listdir(self.scene_folder))
                scene_file_list = [file for file in scene_file_list if file.split('.')[0] == self.test_scene_name]

            for sid, file in enumerate(scene_file_list):
                # print(f"{sid} Loading Scene Mesh {file}")
                if 'occ' not in file:
                    scene_occ = np.load(os.path.join(self.scene_folder, file))
                    scene_occ = torch.from_numpy(scene_occ).to(device=device, dtype=bool)
                else:
                    scene_occ = np.load(os.path.join(self.scene_folder, file))
                
                self.scene_occ.append(scene_occ)
                self.scene_dict[file[:-4]] = sid
            if not self.vis and self.load_object_goal: # todo: can be optimized
                self.scene_occ = get_occupancy_from_npy(self.scene_occ)
                self.scene_occ = torch.from_numpy(self.scene_occ).to(device=self.device, dtype=bool)
                with open(os.path.join(folder, 'scene_name2file.pkl'), 'rb') as f:
                    self.scene_name2file = pkl.load(f)
            else:
                self.scene_occ = torch.stack(self.scene_occ)
            
            if self.vis:
                self.scene_occ_ref = self.compute_occ_ref(self.scene_occ)
            else:
                self.scene_occ_ref = LazyOccRef(self.scene_occ)

            if not self.vis:
                self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
                self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)
            else:
                if 'demo' not in self.test_scene_name:
                    self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
                    self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)
                else:
                    self.scene_grid_np = np.array([-4, 0, -6, 4, 2, 6, 400, 100, 600])
                    self.scene_grid_torch = torch.tensor([-4, 0, -6, 4, 2, 6, 400, 100, 600]).to(device)

            # self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
            # self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)

            self.batch_id = torch.linspace(0, batch_size - 1, batch_size).tile((nb_voxels[0]*nb_voxels[1]*nb_voxels[2], 1)).T \
                .reshape(-1, 1).to(device=device, dtype=torch.long)

            if self.load_object_goal:
                self.batch_id_obj = torch.linspace(0, batch_size - 1, batch_size).tile((1024, 1)).T \
                    .reshape(-1, 1).to(device=device, dtype=torch.long)

        if self.max_window_size == 16:
            norm = np.load(os.path.join(folder, 'norm.npy'))

        self.min = norm[0].astype(np.float32)
        self.max = norm[1].astype(np.float32)
        self.min_torch = torch.tensor(self.min).to(device)
        self.max_torch = torch.tensor(self.max).to(device)

        if self.load_object_goal:
            self.obj_min = norm[2].astype(np.float32)
            self.obj_max = norm[3].astype(np.float32)
            self.obj_min_torch = torch.tensor(self.obj_min).to(device)
            self.obj_max_torch = torch.tensor(self.obj_max).to(device)
            self.obj_bps_data = None
            self.bps_dict = {}
            self.rest_bps_data = {}
            self.rest_obj_verts = {}

            if self.load_object_payload:
                start_idx = 0
                bps_file_list = sorted(os.listdir(self.dest_obj_bps_npy_folder))
                # Two passes so the destination is allocated exactly once.  The
                # previous list-then-cat held the whole payload twice at its
                # peak (~9.4 GB each).  Pass 1 reads only the .npy headers via
                # mmap, so the offsets and dtype are known before any data is
                # materialized; pass 2 copies each file into its final slice.
                # Concatenation order, offsets and values are unchanged.
                shapes = []
                for file in bps_file_list:
                    header = np.load(os.path.join(self.dest_obj_bps_npy_folder, file), mmap_mode='r')
                    shapes.append(header.shape)
                    bps_dtype = header.dtype
                    self.bps_dict[file[:-4]] = (start_idx, start_idx + header.shape[0])
                    start_idx += header.shape[0]
                    del header

                self.obj_bps_data = torch.empty(
                    (start_idx,) + tuple(shapes[0][1:]),
                    dtype=torch.from_numpy(np.empty(0, dtype=bps_dtype)).dtype,
                )
                offset = 0
                for file, shape in zip(bps_file_list, shapes):
                    obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, file)
                    obj_bps_data = np.load(obj_bps_npy_path)  # T X 1024 X 3
                    self.obj_bps_data[offset:offset + shape[0]] = torch.from_numpy(obj_bps_data)
                    offset += shape[0]
                    del obj_bps_data

            if self.use_object_keypoints and self.load_object_payload:
                self.bps_torch = bps_torch()
                self.obj_bps = zup_to_yup(torch.load('./bps.pt')['obj'])
                for object_file in os.listdir(self.rest_object_geo_folder):
                    if object_file.endswith('.npy'):
                        object_name = object_file.split('.')[0]
                        self.rest_bps_data[object_name] = zup_to_yup(np.load(os.path.join(self.rest_object_geo_folder, object_file)))

        if self.load_object_goal:
            self.contact_label = {}
            if self.load_object_payload:
                contact_label_folder = os.path.join(folder, 'contact_label_npy_files')
                for file in os.listdir(contact_label_folder):
                    self.contact_label[file[:-4]] = np.load(os.path.join(contact_label_folder, file))

    def set_test_scene(self, test_scene_name):
        """[accel] Switch test scene: recompute only scene-related data (scene_occ / scene_dict /
        scene_occ_ref / scene_grid), reusing all scene-independent loaded data (human/obj_bps/
        contact_label/clip etc.), avoiding repeated reads of large files when rebuilding the whole dataset.

        Logic is identical to the scene-related part of the self.load_scene branch in __init__; consumes no
        global RNG (torch/numpy/python), so the effect on results is zero, bit-for-bit identical."""
        self.test_scene_name = test_scene_name
        if not self.load_scene:
            return

        folder = self.folder
        device = self.device

        self.scene_occ = []
        self.scene_occ_ref = []
        self.scene_dict = {}
        with open(os.path.join(folder, 'scene_name.pkl'), 'rb') as f:
            self.scene_name = pkl.load(f)  # list of scene names
        if not self.vis:
            self.scene_folder = os.path.join(folder, 'Scene')
            scene_file_list = sorted(os.listdir(self.scene_folder))
        else:
            self.scene_folder = os.path.join(folder, 'Scene_vis')
            scene_file_list = sorted(os.listdir(self.scene_folder))
            scene_file_list = [file for file in scene_file_list if file.split('.')[0] == self.test_scene_name]

        for sid, file in enumerate(scene_file_list):
            if 'occ' not in file:
                scene_occ = np.load(os.path.join(self.scene_folder, file))
                scene_occ = torch.from_numpy(scene_occ).to(device=device, dtype=bool)
            else:
                scene_occ = np.load(os.path.join(self.scene_folder, file))

            self.scene_occ.append(scene_occ)
            self.scene_dict[file[:-4]] = sid
        if not self.vis and self.load_object_goal:
            self.scene_occ = get_occupancy_from_npy(self.scene_occ)
            self.scene_occ = torch.from_numpy(self.scene_occ).to(device=self.device, dtype=bool)
            with open(os.path.join(folder, 'scene_name2file.pkl'), 'rb') as f:
                self.scene_name2file = pkl.load(f)
        else:
            self.scene_occ = torch.stack(self.scene_occ)

        if self.vis:
            self.scene_occ_ref = self.compute_occ_ref(self.scene_occ)
        else:
            self.scene_occ_ref = LazyOccRef(self.scene_occ)

        if not self.vis:
            self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
            self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)
        else:
            if 'demo' not in self.test_scene_name:
                self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
                self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)
            else:
                self.scene_grid_np = np.array([-4, 0, -6, 4, 2, 6, 400, 100, 600])
                self.scene_grid_torch = torch.tensor([-4, 0, -6, 4, 2, 6, 400, 100, 600]).to(device)

    def __getitem__(self, idx):
        if self.load_language:
            start_idx = int(self.start_ind[idx])
            end_idx = int(self.end_ind[idx])
            assert end_idx - start_idx == self.max_window_size * 3

            pelvis_goal = np.zeros((3, )).astype(np.float32)
            scene_goal = np.zeros((3, )).astype(np.float32)
            object_goal = np.zeros((3, )).astype(np.float32)
            is_loco = False
            is_object = False

            text = self.text[idx][0]
            text_clip_embedding = self.clip_features[self.text2features_idx[text]]  # (1, 768)
            text_clip_embedding = torch.from_numpy(text_clip_embedding).float().reshape(1, -1)
            text_clip_embedding = text_clip_embedding / torch.norm(text_clip_embedding, dim=1, keepdim=True)
            
            left_hand_inter_frame = self.left_hand_inter_frame[idx]
            right_hand_inter_frame = self.right_hand_inter_frame[idx]
            if self.load_object_goal:
                is_object = self.need_object[idx]

            origin_sequence_idx = self.ori_sequence_idx[idx] 

            if left_hand_inter_frame != -1:
                scene_goal = self.joints[left_hand_inter_frame, 24].copy()  # left hand index1
            elif right_hand_inter_frame != -1:
                scene_goal = self.joints[right_hand_inter_frame, 26].copy()  # right hand index1

            seq_len = self.ori_sequence_end_idx[origin_sequence_idx] - self.ori_sequence_start_idx[origin_sequence_idx]
            need_scene = self.need_scene[idx]
            need_pelvis_dir = self.need_pelvis_dir[idx]
            pi = self.pi[idx]
            need_pi = self.need_pi[idx]
            if need_pi and self.train:
                pi = pi + np.random.randint(-5, 5)
                pi = max(pi, 0)
            if not need_pi:
                pi = np.random.randint(0, seq_len - self.max_window_size * self.step)
                

            if need_pelvis_dir:
                if 'sit down' in text or 'lie down' in text:
                    # pelvis_goal = self.joints[int(self.end_range[idx]), 0].copy()
                    scene_goal = self.joints[int(self.end_range[idx]), 0].copy() # use hand goal to locate end pelvis goal
                    pelvis_goal = self.joints[end_idx-3, 0].copy() # align with omomo
                else:
                    pelvis_goal = self.joints[end_idx-3, 0].copy()
                    is_loco = True
                pelvis_goal[1] = 0.

        joints = self.joints[start_idx: end_idx: self.step]
        init_joints = np.array([joints[0, 0, 0], 0., joints[0, 0, 2]]) # human's local frame
        joints = joints - init_joints
        pelvis_goal = pelvis_goal - init_joints
        scene_goal = scene_goal - init_joints

        if is_object:
            object_goal = self.object_trans[int(self.end_range[idx])-4].copy() - init_joints # human's local frame
            assert int(self.end_range[idx]) == int(self.ori_sequence_end_idx[origin_sequence_idx])
            # object_goal[1] = 0. (3-dim position represent final object position)
            if self.scene_name[origin_sequence_idx] in self.bps_dict:
                bps_start_idx, bps_end_idx = self.bps_dict[self.scene_name[origin_sequence_idx]]
                obj_bps_data = self.obj_bps_data[bps_start_idx:bps_end_idx]
                assert obj_bps_data.shape[0] == self.ori_sequence_end_idx[origin_sequence_idx] - self.ori_sequence_start_idx[origin_sequence_idx]
                if self.use_random_frame_bps:
                    random_sampled_t_idx = random.sample(list(range(obj_bps_data.shape[0])), 1)[0]
                else: # use the first frame of this window for object bps
                    random_sampled_t_idx = start_idx - self.ori_sequence_start_idx[origin_sequence_idx] 
                obj_bps_data = obj_bps_data[random_sampled_t_idx:random_sampled_t_idx+1] # 1 X 1024 X 3
                obj_bps_data = zup_to_yup(obj_bps_data)
                # bps_set = self.obj_bps + self.object_trans[self.ori_sequence_start_idx[origin_sequence_idx]+random_sampled_t_idx][None,None,:] # 1X1024X3
                # lhand_point = self.joints[self.ori_sequence_start_idx[origin_sequence_idx]+random_sampled_t_idx][24,:] # 3
                # rhand_point = self.joints[self.ori_sequence_start_idx[origin_sequence_idx]+random_sampled_t_idx][26,:] # 3
                # lhand_delta = torch.from_numpy(lhand_point[None, None, :]) - bps_set
                # rhand_delta = torch.from_numpy(rhand_point[None, None, :]) - bps_set
                # obj_bps_data = torch.cat([obj_bps_data, lhand_delta, rhand_delta], axis=-1) # 1 X 1024 X 9
        else:
            # print("obj_bps_npy not found: ", self.scene_name[origin_sequence_idx])
            obj_bps_data = torch.zeros((1, 1024, 3), dtype=torch.float32)

        # transform object goal to human's local frame
        if is_object:
            object_name = self.object_name[origin_sequence_idx]
            
            object_rot_mat = self.object_rot_mat[start_idx: end_idx: self.step] # human-relative rotation matrix
            if self.use_random_frame_bps:
                object_rot_mat_ref = self.object_rot_mat[self.ori_sequence_start_idx[origin_sequence_idx]: self.ori_sequence_end_idx[origin_sequence_idx]][random_sampled_t_idx]
            else:
                object_rot_mat_ref = object_rot_mat[0]
            object_rot_mat_orig = object_rot_mat.copy()
            object_rot_mat = self.prep_rel_obj_rot_mat_w_reference_mat(object_rot_mat, object_rot_mat_ref)
            
            object_trans = self.object_trans[start_idx: end_idx: self.step]
            object_trans_orig = object_trans.copy()
            object_trans = object_trans - init_joints
        else:
            object_name = "none"
            object_rot_mat = np.zeros((joints.shape[0], 3, 3))
            object_rot_mat_ref = object_rot_mat[0]
            object_trans = np.zeros((joints.shape[0], 3))
        
        if is_object and self.use_object_keypoints:
            rest_obj_bps_data = self.rest_bps_data[self.object_name[origin_sequence_idx]]
            nn_pts_on_mesh = self.obj_bps + torch.from_numpy(rest_obj_bps_data).float().to(self.obj_bps.device) # 1 X 1024 X 3 
            nn_pts_on_mesh = nn_pts_on_mesh.squeeze(0) # 1024 X 3 
            
            # random sample 100 points used for training
            # sampled_vidxs = random.sample(list(range(1024)), 100) 
            # sampled_nn_pts_on_mesh = nn_pts_on_mesh[sampled_vidxs] # 100 X 3 
            # rest_pose_obj_nn_pts = sampled_nn_pts_on_mesh.clone()
            rest_pose_obj_nn_pts = self.obj_rest_verts[object_name]
            indices = torch.randperm(rest_pose_obj_nn_pts.shape[0])[:100]
            rest_pose_obj_nn_pts = rest_pose_obj_nn_pts[indices] # 100 X 3
            sampled_nn_pts_on_mesh = rest_pose_obj_nn_pts.clone() # 100 X 3
            rest_pose_obj_normals = self.obj_vert_normals[object_name][indices] # 100 X 3

            # compute nn points for each frame
            object_rot_mat_orig = torch.from_numpy(object_rot_mat_orig).to(sampled_nn_pts_on_mesh.device) # T X 3 X 3
            object_trans_orig = torch.from_numpy(object_trans_orig).to(sampled_nn_pts_on_mesh.device)
            sampled_nn_pts_on_mesh = sampled_nn_pts_on_mesh[None].repeat(object_rot_mat_orig.shape[0], 1, 1) # T X 100 X 3
            transformed_obj_verts = object_rot_mat_orig.bmm(sampled_nn_pts_on_mesh.transpose(2, 1)) + \
                object_trans_orig.unsqueeze(2) # T X 3 X 100, in global frame
            transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X 100 X 3
            # transformed_obj_verts = self.transformed_obj_verts[start_idx: end_idx: self.step]
            # rest_pose_obj_nn_pts = self.rest_pose_obj_nn_pts[origin_sequence_idx]
        else:
            transformed_obj_verts = torch.zeros((object_rot_mat.shape[0], 100, 3))
            rest_pose_obj_nn_pts = torch.zeros((100, 3))
            rest_pose_obj_normals = torch.zeros((100, 3))
        
        # rest_verts, obj_mesh_faces, transformed_obj_verts = \
        #             self.load_rest_pose_object_geometry_and_transform(
        #                 object_name, object_rot_mat, object_trans)

        global_orient = self.global_orient[start_idx: end_idx]
        init_global_orient = global_orient[0]
        # Window heading canonicalization.  scipy's lowercase 'zxy' is
        # EXTRINSIC, so R_root = Ry(c) @ Rx(b) @ Rz(a) for angles [a, b, c] and
        # index 2 is the outermost rotation -- the one about world +y.  Removing
        # it is therefore a heading removal in the strict sense: the shift axis
        # is exactly +-y, and the root's uprightness (R[1,1], the angle between
        # the body up axis and world up) is algebraically invariant under it.
        #
        # This arithmetic is the released code's, unchanged.  What changed is its
        # input: before the world-frame normalization above, global_orient was
        # conjugated, its vertical was +-z, and index 2 read a rotation about a
        # horizontal axis -- so the shift removed 3.6 deg (p50) of a heading that
        # is uniform on the circle, i.e. it canonicalized nothing.
        #
        # pytorch3d's matrix_to_euler_angles(R, 'ZXY') is INTRINSIC and its index
        # 2 is the innermost y rotation, which is a body-frame rotation and not a
        # heading.  priors/core/window_codec.py uses 'YXZ'[..., 0] instead, which
        # is the same quantity as the line below; see that file and gate E in
        # tests/hsi/test_representation_frame.py.
        init_global_orient_euler = R.from_rotvec(init_global_orient).as_euler('zxy')
        shift_euler = np.array([0, 0, -init_global_orient_euler[2]])
        shift_rot_matrix = R.from_euler('zxy', shift_euler).as_matrix()

        global_orient = torch.from_numpy(global_orient).reshape(-1, 1, 3) # T X 3 X 3
        human_pose = torch.from_numpy(self.human_pose[start_idx: end_idx]).reshape(-1, 21, 3) # T X 21 X 3

        local_rot_aa = torch.cat([global_orient, human_pose], dim=1) # T X 22 X 3
        local_rot_mat = transforms.axis_angle_to_matrix(local_rot_aa)
        global_rot_mat = self.local2global_pose(local_rot_mat) # T X 22 X 3 X 3

        global_rot_mat = torch.from_numpy(shift_rot_matrix).float()[None, None] @ global_rot_mat.float()

        global_rot_6d = transforms.matrix_to_rotation_6d(global_rot_mat) # T X 22 X 6

        mat = np.eye(4)
        mat[:3, :3] = np.linalg.inv(shift_rot_matrix.T).T
        mat[:3, 3] = init_joints
        mat = mat.astype(np.float32)

        joints = joints @ shift_rot_matrix.T
        pelvis_goal = pelvis_goal @ shift_rot_matrix.T
        scene_goal = scene_goal @ shift_rot_matrix.T
        if is_object:
            object_trans = object_trans @ shift_rot_matrix.T
            # object_rot_mat = mat[:3, :3] @ object_rot_mat
            object_goal = object_goal @ shift_rot_matrix.T

        joints = self.normalize(joints)
        joints = joints.astype(np.float32).reshape((joints.shape[0], -1))

        if is_object:
            object_trans = self.normalize(object_trans, is_object=True)

        if not self.vis and self.load_scene:
            if self.load_object_goal:
                scene_flag = self.scene_dict[f'occ_{self.scene_name2file[self.scene_name[origin_sequence_idx]]}']
            else:
                scene_flag = self.scene_dict[self.scene_name[start_idx]]
        else:
            scene_flag = 0

        if not self.use_pi:
            pi = 0
            need_pi = False

        if is_object and self.object_points is not None:
            object_points = self.object_points[start_idx]
        else:
            object_points = np.zeros((1024, 3))

        if is_object and self.load_object_goal:
            contact_label = self.contact_label[self.scene_name[origin_sequence_idx]]
            contact_label = contact_label[start_idx - self.ori_sequence_start_idx[origin_sequence_idx]: end_idx - self.ori_sequence_start_idx[origin_sequence_idx]: self.step]
        else:
            contact_label = np.zeros((len(joints), 4))

        if self.train:
            transl = self.transl[start_idx] - self.joints[start_idx][0]
        else:
            transl = self.transl[origin_sequence_idx]

        info = {
            'joints': joints.astype(np.float32),
            'global_rot_6d': global_rot_6d[::self.step],
            'mat': mat.astype(np.float32),
            'object_trans': object_trans.astype(np.float32),
            'object_rot_mat': object_rot_mat.astype(np.float32),
            'scene_flag': scene_flag,
            'text_clip_embedding': text_clip_embedding,
            'pelvis_goal': pelvis_goal.astype(np.float32),
            'scene_goal': scene_goal.astype(np.float32),
            'object_goal': object_goal.astype(np.float32),
            'need_scene': need_scene,
            'need_pelvis_dir': need_pelvis_dir,
            'pi': pi,
            'need_pi': need_pi,
            'is_loco': is_loco,
            'is_object': is_object,
            'obj_bps_data': obj_bps_data,
            'obj_rot_mat_ref': object_rot_mat_ref.astype(np.float32),
            'rest_pose_obj_nn_pts': rest_pose_obj_nn_pts,
            'rest_pose_obj_normals': rest_pose_obj_normals,
            'transformed_obj_verts': transformed_obj_verts,
            'object_points': object_points,
            'seq_name': self.scene_name[origin_sequence_idx],
            'contact_label': contact_label.astype(np.float32),
            'joints_gt': self.joints[start_idx: end_idx],
            'global_rot_6d_gt': global_rot_6d,
            'object_trans_gt': self.object_trans[start_idx: end_idx] if self.object_trans is not None else np.zeros((48, 3)),
            'object_rot_mat_gt': self.object_rot_mat[start_idx: end_idx] if self.object_rot_mat is not None else np.zeros((48, 3, 3)),
            'rest_human_offsets': self.rest_human_offsets[origin_sequence_idx].astype(np.float32),
            'transl': transl,
            'betas': self.betas[origin_sequence_idx],
            'gender': self.gender[origin_sequence_idx],
            'seg_len': seq_len,
            'end_pi': min(pi + self.max_window_size * self.step, self.ori_sequence_end_idx[origin_sequence_idx] - self.ori_sequence_start_idx[origin_sequence_idx]),
            'object_name': object_name
        }
        
        return info

    def get_pene_occ_count(self, points, scene_flag):
        occ = (self.scene_occ[scene_flag]).to(dtype=torch.int8).clone().to(dtype=torch.int8)

        T, N = points.shape[0], points.shape[1]
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel = torch.div((points - self.scene_grid_torch[:3]), voxel_size) # [T * N, 3]
        voxel = voxel.to(dtype=torch.long)
        # voxel = rearrange(voxel, 'b p c -> (b p) c')
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        voxel[torch.logical_not(in_bound)] = 0
        voxel = voxel.reshape(T, N, -1)
        
        t_idx = torch.arange(T, device=occ.device).unsqueeze(1).expand(T, N)
        # Find all positions with value 1 and set them to 3

        if not self.vis:
            mask = (occ[t_idx, voxel[..., 0], voxel[..., 1], voxel[..., 2]] == 1)
            occ[t_idx[mask], voxel[..., 0][mask], voxel[..., 1][mask], voxel[..., 2][mask]] = 3
            # Count the number of 3s in occ (number of penetrating occ)
            pene_count = torch.sum(occ == 3, dim=(1, 2, 3)).cpu().numpy()
        else:
            mask = (occ[voxel[..., 0], voxel[..., 1], voxel[..., 2]] == 1)
            occ[voxel[..., 0][mask], voxel[..., 1][mask], voxel[..., 2][mask]] = 3
            # Count the number of 3s in occ (number of penetrating occ)
            pene_count = torch.sum(occ == 3, dim=(1, 2)).cpu().numpy()
        
        # mask = (occ[0, voxel[..., 0], voxel[..., 1], voxel[..., 2]] == 1)
        # occ[0, voxel[..., 0][mask], voxel[..., 1][mask], voxel[..., 2][mask]] = 3
        # pene_count = torch.sum(occ == 3).cpu().numpy()

        return pene_count

    def add_object_points(self, points, occ):
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel = torch.div((points - self.scene_grid_torch[:3]), voxel_size)
        voxel = voxel.to(dtype=torch.long)
        # voxel = rearrange(voxel, 'b p c -> (b p) c')
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        voxel[torch.logical_not(in_bound)] = 0
        if self.train:
            voxel = torch.cat([self.batch_id_obj, voxel], dim=-1)
        # voxel = voxel[in_bound]
        if self.train:
            occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]] = 2 # 2 represents object (todo: object index?)
        else:
            if self.vis:
                occ[0, voxel[:, 0], voxel[:, 1], voxel[:, 2]] = 2
            else:
                voxel = torch.cat([self.batch_id_obj, voxel], dim=-1)
                occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]] = 2
                # occ = occ.unsqueeze(0)
                # occ[0, voxel[:, 0], voxel[:, 1], voxel[:, 2]] = 2

    def get_occ_for_points(self, points, obj_points, scene_flag):
        batch_size = points.shape[0]
        seq_len = points.shape[1]
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:]) # 0.02
        voxel = torch.div((points - self.scene_grid_torch[:3]), voxel_size)
        voxel = voxel.to(dtype=torch.long)
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        voxel[torch.logical_not(in_bound)] = 0

        self.batch_id = torch.linspace(0, batch_size - 1, batch_size).tile((self.nb_voxels[0]*self.nb_voxels[1]*self.nb_voxels[2], 1)).T \
            .reshape(-1, 1).to(device=points.device, dtype=torch.long)
        
        if self.train:
            voxel = torch.cat([self.batch_id, voxel], dim=1)

        occ = (self.scene_occ[scene_flag]).to(dtype=torch.int8)

        if self.load_object_goal:
            self.batch_id_obj = torch.linspace(0, batch_size - 1, batch_size).tile((1024, 1)).T \
                .reshape(-1, 1).to(device=points.device, dtype=torch.long)

        if obj_points is not None:
            self.add_object_points(obj_points, occ)

        if self.train:
            occ_for_points = occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]]
        else:
            if self.vis:
                occ_for_points = occ[0, voxel[:, 0], voxel[:, 1], voxel[:, 2]]
            else:
                voxel = torch.cat([self.batch_id, voxel], dim=1)
                occ_for_points = occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]]
                # occ = occ.unsqueeze(0)
                # occ_for_points = occ[0, voxel[:, 0], voxel[:, 1], voxel[:, 2]]
        occ_for_points[torch.logical_not(in_bound)] = 1 # 1 represents occupied
        occ_for_points = occ_for_points.reshape(batch_size, seq_len, -1)
        
        return occ_for_points

    def _get_nearest_free_voxel_direct(self, points, scene_flag):
        """Query occupancy and references without materializing full scene batches."""
        original_shape = points.shape[:-1]
        batch_size = points.shape[0]
        seq_len = points.shape[1]
        N = points.shape[2]

        points_flat = points.reshape(-1, 3)

        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel_indices = torch.div(points_flat - self.scene_grid_torch[:3], voxel_size).long()

        valid_mask = torch.all((voxel_indices >= 0) & (voxel_indices < self.scene_grid_torch[6:] - 0), dim=-1)
        voxel_indices[torch.logical_not(valid_mask)] = 0
        voxel_indices = voxel_indices.reshape(batch_size, seq_len*N, 3)

        scene_flag = scene_flag.reshape(-1)
        b_idx = torch.arange(batch_size, device=self.scene_occ.device).unsqueeze(1).expand(batch_size, seq_len*N)

        is_penetrating = (self.scene_occ[scene_flag[b_idx], voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]] == 1)
        valid_mask = valid_mask.reshape(is_penetrating.shape)

        nearest_free_points = points_flat.clone().reshape(batch_size, seq_len*N, 3)

        # For penetrating points, get the displacement and compute the safe position
        penetrating_mask = valid_mask & is_penetrating
        if penetrating_mask.any():
            pen_indices = voxel_indices[penetrating_mask]
            penetrating_scene_flags = scene_flag[b_idx[penetrating_mask]]
            displacements = torch.empty_like(pen_indices, dtype=torch.int16)
            for scene_id in torch.unique(penetrating_scene_flags).tolist():
                scene_mask = penetrating_scene_flags == scene_id
                scene_pen_indices = pen_indices[scene_mask]
                occ_ref = self.scene_occ_ref[int(scene_id)]
                displacements[scene_mask] = occ_ref[scene_pen_indices[:, 0], scene_pen_indices[:, 1], scene_pen_indices[:, 2]]
            # Compute the safe position
            nearest_free_points[penetrating_mask] = (pen_indices + displacements) * voxel_size + self.scene_grid_torch[:3]

        return is_penetrating.reshape(original_shape), nearest_free_points.reshape(*original_shape, 3)

    def _get_nearest_free_voxel_materialized(self, points, scene_flag):
        with torch.no_grad():
            occ = self.scene_occ[scene_flag]
            occ_ref = self.scene_occ_ref[scene_flag]

        original_shape = points.shape[:-1]
        batch_size = points.shape[0]
        seq_len = points.shape[1]
        N = points.shape[2]

        points_flat = points.reshape(-1, 3)
        
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel_indices = torch.div(points_flat - self.scene_grid_torch[:3], voxel_size).long()

        valid_mask = torch.all((voxel_indices >= 0) & (voxel_indices < self.scene_grid_torch[6:] - 0), dim=-1)
        voxel_indices[torch.logical_not(valid_mask)] = 0
        voxel_indices = voxel_indices.reshape(batch_size, seq_len*N, 3)

        b_idx = torch.arange(batch_size, device=occ.device).unsqueeze(1).expand(batch_size, seq_len*N)

        is_penetrating = (occ[b_idx, voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]] == 1)
        valid_mask = valid_mask.reshape(is_penetrating.shape)

        nearest_free_points = points_flat.clone().reshape(batch_size, seq_len*N, 3)

        # For penetrating points, get the displacement and compute the safe position
        penetrating_mask = valid_mask & is_penetrating
        if penetrating_mask.any():
            pen_indices = voxel_indices[penetrating_mask]
            # Get the displacement vector directly from the occ_ref tensor
            displacements = occ_ref[b_idx[penetrating_mask], pen_indices[:, 0], pen_indices[:, 1], pen_indices[:, 2]]
            # Compute the safe position
            nearest_free_points[penetrating_mask] = (pen_indices + displacements) * voxel_size + self.scene_grid_torch[:3]
            # import pdb; pdb.set_trace()
        
        return is_penetrating.reshape(original_shape), nearest_free_points.reshape(*original_shape, 3)

    def get_nearest_free_voxel(self, points, scene_flag):
        return self._get_nearest_free_voxel_direct(points, scene_flag)

    def create_meshgrid(self, batch_size=1):
        bbox = self.mesh_grid
        size = (self.nb_voxels[0], self.nb_voxels[1], self.nb_voxels[2])
        x = torch.linspace(bbox[0], bbox[1], size[0])
        y = torch.linspace(bbox[2], bbox[3], size[1])
        z = torch.linspace(bbox[4], bbox[5], size[2])
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        grid = grid.repeat(batch_size, 1, 1)

        return grid

    def __len__(self):
        return len(self.start_ind)

    def normalize(self, data, is_object=False):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        if is_object:
            data = -1. + 2. * (data - self.obj_min) / (self.obj_max - self.obj_min)
        else:
            data = -1. + 2. * (data - self.min) / (self.max - self.min)
        data = data.reshape(shape_orig)

        return data

    def normalize_torch(self, data, is_object=False):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        if is_object:
            data = -1. + 2. * (data - self.obj_min_torch) / (self.obj_max_torch - self.obj_min_torch)
        else:
            data = -1. + 2. * (data - self.min_torch) / (self.max_torch - self.min_torch)
        data = data.reshape(shape_orig)

        return data

    def denormalize(self, data, is_object=False):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        if is_object:
            data = (data + 1.) * (self.obj_max - self.obj_min) / 2. + self.obj_min
        else:
            data = (data + 1.) * (self.max - self.min) / 2. + self.min
        data = data.reshape(shape_orig)

        return data
    
    def denormalize_torch(self, data, is_object=False):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        if is_object:
            data = (data + 1.) * (self.obj_max_torch - self.obj_min_torch) / 2. + self.obj_min_torch
        else:
            data = (data + 1.) * (self.max_torch - self.min_torch) / 2. + self.min_torch
        data = data.reshape(shape_orig)

        return data

    # load rest pose object geometry from omomo dataset
    def load_rest_pose_object_geometry_and_transform(self, object_name, obj_rot, obj_com_pos):
        rest_obj_path = os.path.join(self.rest_object_geo_folder, object_name+".ply")
        
        mesh = trimesh.load_mesh(rest_obj_path)
        rest_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3
    
        rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
        transformed_obj_verts = obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

        return rest_verts, obj_mesh_faces, transformed_obj_verts
    
    def prep_rel_obj_rot_mat_w_reference_mat(self, obj_rot_mat, ref_rot_mat):
        # obj_rot_mat: T X 3 X 3 / BS X T X 3 X 3 
        # ref_rot_mat: BS X 1 X 3 X 3/ 1 X 3 X 3 
        obj_rot_mat = torch.tensor(obj_rot_mat)
        ref_rot_mat = torch.tensor(ref_rot_mat)
        if obj_rot_mat.dim() == 4:
            timesteps = obj_rot_mat.shape[1]

            init_obj_rot_mat = ref_rot_mat.repeat(1, timesteps, 1, 1) # BS X T X 3 X 3
            rel_rot_mat = torch.matmul(obj_rot_mat, init_obj_rot_mat.transpose(2, 3)) # BS X T X 3 X 3
        else:
            timesteps = obj_rot_mat.shape[0]

            # Compute relative rotation matrix with respect to the first frame's object geometry. 
            init_obj_rot_mat = ref_rot_mat.repeat(timesteps, 1, 1) # T X 3 X 3
            # R_rel = R_obj @ R_ref^T
            rel_rot_mat = torch.matmul(obj_rot_mat, init_obj_rot_mat.transpose(1, 2)) # T X 3 X 3
        return rel_rot_mat.cpu().numpy()
    
    def local2global_pose(self, local_pose):
        # local_pose: T X J X 3 X 3 
        kintree = self.parents_22 

        bs = local_pose.shape[0]

        local_pose = local_pose.view(bs, -1, 3, 3)

        global_pose = local_pose.clone()

        for jId in range(len(kintree)):
            parent_id = kintree[jId]
            if parent_id >= 0:
                global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

        return global_pose # T X J X 3 X 3 

    def quat_ik_torch(self, grot_mat):
        # grot: T X J X 3 X 3 
        parents = self.parents_22 

        grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

        res = torch.cat(
                [
                    grot[..., :1, :],
                    transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                    grot[..., 1:, :]),
                ],
                dim=-2) # T X J X 4 

        res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

        return res_mat
    
    def quat_fk_torch(self, lrot_mat, lpos, use_joints24=True):
        # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
        # lpos: N X J/(J+2) X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
        if use_joints24:
            parents = self.parents_24
        else:
            parents = self.parents_22 

        lrot = transforms.matrix_to_quaternion(lrot_mat)

        gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
        for i in range(1, len(parents)):
            gp.append(
                transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
            )
            if i < lrot.shape[-2]:
                gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

        res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

        return res

    def compute_occ_ref(self, occ):
        """Compute the reference position from each occupied voxel to the nearest free voxel in the scene

        Args:
            occ (torch.Tensor): scene occupancy grid of shape [B, W, H, D], 1 means occupied, 0 means free

        Returns:
            torch.Tensor: tensor of shape [B, W, H, D, 3], storing the displacement vector from each voxel to the nearest free voxel
        """
        # Compute the distance field to the nearest free voxel
        device = occ.device
        occ = occ.cpu().numpy()

        # Process each batch separately
        batch_size = occ.shape[0]
        batch_displacements = []

        for b in range(batch_size):
            batch_displacements.append(_compute_single_occ_ref(occ[b]))

        # Stack the results of all batches [B, W, H, D, 3]
        batch_displacements = np.stack(batch_displacements, axis=0)

        # Convert to a torch tensor
        occ_ref = torch.from_numpy(batch_displacements).to(device=device, dtype=torch.int16)

        return occ_ref
