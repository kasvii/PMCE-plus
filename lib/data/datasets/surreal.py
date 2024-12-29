from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import torch
import joblib
import numpy as np

from configs import constants as _C
from .._dataset import BaseDataset
from smplx import SMPL
from .._dataset import BaseDataset
from ..utils.augmentor import *
from ...utils import data_utils as d_utils
from ...utils import transforms
from ...models import build_body_model
from ...utils.kp_utils import convert_kps, root_centering

class SURREAL(BaseDataset):
    def __init__(self, cfg, dset='train'):
        super(SURREAL, self).__init__(cfg, dset=='train')
        parsed_data_path = os.path.join(_C.PATHS.PARSED_DATA, f'surreal_vm_{dset}_backbone.pth')
        parsed_data_path = parsed_data_path.replace('backbone', cfg.MODEL.BACKBONE.lower())

        self.has_3d = True
        self.has_traj = True
        self.has_smpl = True
        self.has_verts = True

        # Among 31 joints format, 14 common joints are avaialable
        self.mask = torch.zeros(_C.KEYPOINTS.NUM_JOINTS + 14)
        self.mask[-14:] = 1
        
        self.smpl_gender = {
            0: SMPL(_C.BMODEL.FLDR, gender='male', num_betas=10),
            1: SMPL(_C.BMODEL.FLDR, gender='female', num_betas=10),
            2: SMPL(_C.BMODEL.FLDR, gender='neutral', num_betas=10),
        }
        
        self.epoch = 0
        self.labels = joblib.load(parsed_data_path)
        if cfg.DEBUG:
            for key in self.labels.keys():
                self.labels[key] = self.labels[key][0:458]
        self.n_frames = cfg.DATASET.SEQLEN + 1

        self.smpl = build_body_model('cpu', self.n_frames)
        self.SMPLAugmentor = SMPLAugmentor(cfg, False)
        self.VideoAugmentor = VideoAugmentor(cfg)

    @property
    def __name__(self, ):
        return 'SURREAL'
    
    def __len__(self):
        return len(self.labels['kp2d'])

    def compute_3d_keypoints(self, index):
        return self.labels['joints3D'][index]
    
    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        scale = torch.stack([torch.tensor(1.0), h / w], dim=-1)
        return X / w * 2 - scale

    def __getitem__(self, index):
        return self.get_single_sequence(index)
    
    def get_inputs(self, index, target, vis_thr=0.6):
        # 2D keypoints detection
        kp2d = self.labels['kp2d'][index:index+1][..., :2].clone().repeat(self.n_frames, 1, 1)
        bbox = self.labels['bbox'][index:index+1][..., [0, 1, -1]].clone().repeat(self.n_frames, 1)
        bbox[:, 2] = bbox[:, 2] / 200
        kp2d, bbox = self.keypoints_normalizer(kp2d, target['res'], self.cam_intrinsics, 224, 224, bbox)    
        
        # full img kp2d
        kp2d = self.labels['kp2d'][index:index+1][..., :2].clone().repeat(self.n_frames, 1, 1)
        res = self.labels['res'][index].clone()
        kp2d = self.normalize_screen_coordinates(kp2d, w=res[0], h=res[1])
        kp2d = kp2d.reshape(kp2d.shape[0], -1) # [82, 34]
        
        target['bbox'] = bbox[1:]
        target['kp2d'] = kp2d
        target['mask'] = (self.labels['kp2d'][index:index+1][..., -1] < vis_thr).repeat(self.n_frames - 1, 1)
        
        # Image features
        target['features'] = self.labels['features'][index:index+1].clone().repeat(self.n_frames - 1, 1)
        
        # marker3d
        pred_marker = self.labels['pred_marker'][index:index+1].repeat(self.n_frames, 1, 1)
        pred_marker[..., :3] = pred_marker[..., :3] / 1000
        target['pred_marker'] = pred_marker
        
        return target
    
    def get_labels(self, index, target):
        # SMPL parameters
        # NOTE: We use NeuralAnnot labels for Human36m and MPII3D only for the 0th frame input.
        #       We do not supervise the network on SMPL parameters.
        target['pose'] = transforms.axis_angle_to_matrix(
            self.labels['pose'][index:index+1].clone().repeat(self.n_frames, 1).reshape(-1, 24, 3))
        target['betas'] = self.labels['betas'][index:index+1].clone().repeat(self.n_frames, 1)        # No t
        
        # Apply SMPL augmentor (y-axis rotation and initial frame noise)
        target = self.SMPLAugmentor(target)
    
        # 3D and 2D keypoints
        gt_kp3d = self.labels['joints3D'][index:index+1].clone().repeat(self.n_frames, 1, 1)
        gt_kp2d = self.labels['joints2D'][index:index+1, ..., :2].clone().repeat(self.n_frames - 1, 1, 1)
        gt_kp3d = root_centering(gt_kp3d.clone()) # Center the root joint to the pelvis.
            
        conf = self.mask.repeat(self.n_frames, 1).unsqueeze(-1)        
        gt_kp2d = torch.cat((gt_kp2d, conf[1:]), dim=-1)
        gt_kp3d = torch.cat((gt_kp3d, conf), dim=-1)
        target['kp3d'] = gt_kp3d
        target['full_kp2d'] = gt_kp2d
        target['weak_kp2d'] = torch.zeros_like(gt_kp2d)
        target['contact'] = torch.ones((self.n_frames - 1, 4)) * (-1)
            
        if self.has_verts:
            # SMPL vertices available for 3DPW
            with torch.no_grad():
                gender = 2
                output = self.smpl_gender[gender](
                    body_pose=target['pose'][1:, 1:],
                    global_orient=target['pose'][1:, :1],
                    betas=target['betas'][1:],
                    pose2rot=False,
                )
                target['verts'] = output.vertices.clone()
        else:
            # No SMPL vertices available
            target['verts'] = torch.zeros((self.n_frames - 1, 6890, 3)).float()
            
        # GT 3. gt marker
        target['gt_marker'] = self.labels['gt_marker'][index:index+1].clone().repeat(self.n_frames, 1, 1) / 1000
            
        return target
    
    def get_camera_info(self, index, target):
        # start_index, end_index = self.video_indices[index]
        
        # Intrinsics
        target['res'] = self.labels['res'][index].clone()
        self.get_naive_intrinsics(target['res'])
        target['cam_intrinsics'] = self.cam_intrinsics.clone()
        
        # Extrinsics pose
        R = self.labels['cam_poses'][index:index+1, :3, :3].clone().repeat(self.n_frames, 1, 1).float()
        yaw = transforms.axis_angle_to_matrix(torch.tensor([[0, 2 * np.pi * np.random.uniform(), 0]])).float()
        if self.__name__ == 'Human36M':
            # Map Z-up to Y-down coordinate
            zup2ydown = transforms.axis_angle_to_matrix(torch.tensor([[-np.pi/2, 0, 0]])).float()
            zup2ydown = torch.matmul(yaw, zup2ydown)
            R = torch.matmul(R, zup2ydown)
        elif self.__name__ == 'MPII3D':
            # Map Y-up to Y-down coordinate
            yup2ydown = transforms.axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float()
            yup2ydown = torch.matmul(yaw, yup2ydown)
            R = torch.matmul(R, yup2ydown)
            
        return target

    def get_single_sequence(self, index):
        # Universal target
        target = {'has_full_screen': torch.tensor(True),
                  'has_smpl': torch.tensor(self.has_smpl),
                  'has_traj': torch.tensor(self.has_traj),
                  'has_verts': torch.tensor(self.has_verts),
                  'transl': torch.zeros((self.n_frames, 3)),
                  
                  # Null camera motion
                  'R': torch.eye(3).repeat(self.n_frames, 1, 1),
                  'cam_angvel': torch.zeros((self.n_frames - 1, 6)),
                  
                  # Null root orientation and velocity
                  'pose_root': torch.zeros((self.n_frames, 6)),
                  'vel_root': torch.zeros((self.n_frames - 1, 3)),
                  'init_root': torch.zeros((1, 6)),
                  }
        
        self.get_camera_info(index, target)
        self.get_inputs(index, target)
        self.get_labels(index, target)
        
        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)
        
        return target