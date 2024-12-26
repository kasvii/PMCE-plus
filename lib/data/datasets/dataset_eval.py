from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import torch
import joblib
import numpy as np
from collections import defaultdict

from configs import constants as _C
from .._dataset import BaseDataset
from ...utils import transforms
from ...utils import data_utils as d_utils
from ...utils.kp_utils import convert_kps, root_centering

FPS = 30
class EvalDataset(BaseDataset):
    def __init__(self, cfg, data, split, backbone):
        super(EvalDataset, self).__init__(cfg, False)
        
        self.prefix = ''
        self.data = data
        parsed_data_path = os.path.join(_C.PATHS.PARSED_DATA, f'{data}_vm_{split}_{backbone}.pth')
        self.labels_ori = joblib.load(parsed_data_path)
        print('self.labels_ori[kp2d].shape', len(self.labels_ori['kp2d'][-2]))
        print('self.labels_ori[gt_marker].shape', len(self.labels_ori['gt_marker'][-2]))
        print('self.labels_ori[pred_marker].shape', len(self.labels_ori['pred_marker'][-2]))
        print('self.labels_ori[flipped_pred_marker].shape', len(self.labels_ori['flipped_pred_marker'][-2]))
        self.labels = defaultdict(list)
        for key, val in self.labels_ori.items():
            if key == 'vid' or key == 'gender':
                for idx in range(len(val)):
                    self.labels[key].extend([val[idx]] * len(self.labels_ori['pose'][idx]))
            elif 'init' in key:
                for idx in range(len(val)):
                    self.labels[key].append(val[idx][:1].repeat(len(self.labels_ori['pose'][idx]), 1, 1))
            else:
                for idx in range(len(val)):
                    self.labels[key].append(val[idx])
        for key in self.labels.keys():
            if key != 'vid' and key != 'gender':
                self.labels[key] = torch.cat(self.labels[key])
        self.prepare_video_batch()

        print('self.labels[kp2d].shape', len(self.labels['kp2d']))
        print('self.labels[gt_marker].shape', len(self.labels['gt_marker']))
        print('self.labels[pred_marker].shape', len(self.labels['pred_marker']))
        print('self.labels[flipped_pred_marker].shape', len(self.labels['flipped_pred_marker']))

    def load_data(self, index, flip=False):
        if flip:
            self.prefix = 'flipped_'
        else:
            self.prefix = ''
        
        target = self.__getitem__(index)
        for key, val in target.items():
            if isinstance(val, torch.Tensor):
                target[key] = val.unsqueeze(0)
        return target

    def __getitem__(self, index):
        target = {}
        target = self.get_data(index)
        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)

        return target

    def __len__(self):
        return len(self.video_indices)

    def prepare_labels(self, index, target):
        start_index, end_index = self.video_indices[index]
        
        # Ground truth SMPL parameters
        pose = self.labels['pose'][start_index:end_index+1].clone().float()
        target['pose'] = transforms.axis_angle_to_matrix(pose.reshape(-1, 24, 3))
        target['betas'] = self.labels['betas'][start_index:end_index+1].clone()
        if self.data == '3dpw':
            target['gender'] = self.labels['gender'][start_index]
        
        # Sequence information
        target['res'] = self.labels['res'][start_index]
        target['vid'] = self.labels['vid'][start_index]
        target['frame_id'] = self.labels['frame_id'][start_index:end_index+1][1:]
        vis = [True]
        if self.data == 'h36m':
            next_frame = 2
        else:
            next_frame = 1
        for num in range(1, len(target['frame_id'])):
            if target['frame_id'][num] - target['frame_id'][num - 1] == next_frame:
                vis.append(True)
            else:
                vis.append(False)
        target['vis'] = vis
        
        # Camera information
        self.get_naive_intrinsics(target['res'])
        target['cam_intrinsics'] = self.cam_intrinsics
        R = self.labels['cam_poses'][start_index:end_index+1][:, :3, :3].clone()
        if 'emdb' in self.data.lower():
            # Use groundtruth camera angular velocity.
            # Can be updated with SLAM results if you have it.
            cam_angvel = transforms.matrix_to_rotation_6d(R[:-1] @ R[1:].transpose(-1, -2))
            cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel)) * FPS
            target['R'] = R
        else:
            cam_angvel = torch.zeros((len(target['pose']) - 1, 6))
        target['cam_angvel'] = cam_angvel
        
        # GT 3. gt marker
        target['gt_marker'] = self.labels['gt_marker'][start_index:end_index+1] / 1000
        
        # GT joints 2D & 3D
        if self.data == 'h36m' or self.data == 'mpii3d':
            gt_kp3d = torch.zeros((self.n_frames, self.n_joints + 14, 3))
            gt_kp3d[:, self.n_joints:] = convert_kps(self.labels['joints3D'][start_index:end_index+1], 'spin', 'common')
            gt_kp2d = torch.zeros((self.n_frames - 1, self.n_joints + 14, 2))
            gt_kp2d[:, self.n_joints:] = convert_kps(self.labels['joints2D'][start_index+1:end_index+1, ..., :2], 'spin', 'common')
            
            target['kp3d'] = gt_kp3d
            target['full_kp2d'] = gt_kp2d
            target['weak_kp2d'] = torch.zeros_like(gt_kp2d)
        
        return target
    
    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        scale = torch.stack([torch.tensor(1.0), h / w], dim=-1)
        return X / w * 2 - scale

    def prepare_inputs(self, index, target):
        start_index, end_index = self.video_indices[index]
        for key in ['features', 'bbox']:
            data = self.labels[self.prefix + key][start_index:end_index+1][1:]
            target[key] = data
        
        bbox = self.labels[self.prefix + 'bbox'][start_index:end_index+1][..., [0, 1, -1]].clone().float()
        bbox[:, 2] = bbox[:, 2] / 200
        
        # Normalize keypoints
        kp2d, bbox = self.keypoints_normalizer(
            self.labels[self.prefix + 'kp2d'][start_index:end_index+1][..., :2].clone().float(), 
            target['res'], target['cam_intrinsics'], 224, 224, bbox)
        
        # full img kp2d
        kp2d = self.labels[self.prefix + 'kp2d'][start_index:end_index+1][..., :2].clone().float()
        res = self.labels['res'][start_index:end_index+1][0].clone()
        kp2d = self.normalize_screen_coordinates(kp2d, w=res[0], h=res[1])
        kp2d = kp2d.reshape(kp2d.shape[0], -1)
        
        target['kp2d'] = kp2d
        target['bbox'] = bbox[1:]
        
        # Masking out low confident keypoints
        mask = self.labels[self.prefix + 'kp2d'][start_index:end_index+1][..., -1] < 0.3
        target['input_kp2d'] = self.labels['kp2d'][start_index:end_index+1][1:]
        target['input_kp2d'][mask[1:]] *= 0
        target['mask'] = mask[1:]
        
        # marker3d
        pred_marker = self.labels[self.prefix + 'pred_marker'][start_index:end_index+1]
        pred_marker[..., :3] = pred_marker[..., :3] / 1000
        target['pred_marker'] = pred_marker

        return target

    def prepare_initialization(self, index, target):
        start_index, end_index = self.video_indices[index]
        # Initial frame per-frame estimation
        target['init_kp3d'] = root_centering(self.labels[self.prefix + 'init_kp3d'][start_index:end_index+1][:1, :self.n_joints]).reshape(1, -1)
        target['init_pose'] = transforms.axis_angle_to_matrix(self.labels[self.prefix + 'init_pose'][start_index:end_index+1][:1]).cpu()
        pose_root = target['pose'][:, 0].clone()
        target['init_root'] = transforms.matrix_to_rotation_6d(pose_root)
        
        return target
        
    def get_data(self, index):
        target = {}
        target = self.prepare_labels(index, target)
        target = self.prepare_inputs(index, target)
        target = self.prepare_initialization(index, target)
        
        return target