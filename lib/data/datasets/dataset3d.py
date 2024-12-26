from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import joblib
import numpy as np

from .._dataset import BaseDataset
from ..utils.augmentor import *
from ...utils import data_utils as d_utils
from ...utils import transforms
from ...models import build_body_model
from ...utils.kp_utils import convert_kps, root_centering

class Dataset3D(BaseDataset):
    def __init__(self, cfg, fname, training):
        super(Dataset3D, self).__init__(cfg, training)

        self.epoch = 0
        self.labels = joblib.load(fname)
        if cfg.DEBUG:
            for key in self.labels.keys():
                self.labels[key] = self.labels[key][0:458]
        self.n_frames = cfg.DATASET.SEQLEN + 1

        if self.training:
            self.prepare_video_batch()

        self.smpl = build_body_model('cpu', self.n_frames)
        self.SMPLAugmentor = SMPLAugmentor(cfg, False)
        self.VideoAugmentor = VideoAugmentor(cfg)
    
    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        scale = torch.stack([torch.tensor(1.0), h / w], dim=-1)
        return X / w * 2 - scale

    def __getitem__(self, index):
        return self.get_single_sequence(index)
    
    def get_inputs(self, index, target, vis_thr=0.6):
        start_index, end_index = self.video_indices[index]
        
        # 2D keypoints detection
        kp2d = self.labels['kp2d'][start_index:end_index+1][..., :2].clone()
        bbox = self.labels['bbox'][start_index:end_index+1][..., [0, 1, -1]].clone()
        bbox[:, 2] = bbox[:, 2] / 200
        kp2d, bbox = self.keypoints_normalizer(kp2d, target['res'], self.cam_intrinsics, 224, 224, bbox)    
        
        # full img kp2d
        kp2d = self.labels['kp2d'][start_index:end_index+1][..., :2].clone()
        res = self.labels['res'][start_index:end_index+1][0].clone()
        kp2d = self.normalize_screen_coordinates(kp2d, w=res[0], h=res[1])
        kp2d = kp2d.reshape(kp2d.shape[0], -1) # [82, 34]
        
        target['bbox'] = bbox[1:]
        target['kp2d'] = kp2d # [82, 37] = [init + 81, 2d + bbox]
        target['mask'] = self.labels['kp2d'][start_index+1:end_index+1][..., -1] < vis_thr
        
        # Image features
        target['features'] = self.labels['features'][start_index+1:end_index+1].clone()
        
        # marker3d
        pred_marker = self.labels['pred_marker'][start_index:end_index+1]
        pred_marker[..., :3] = pred_marker[..., :3] / 1000
        target['pred_marker'] = pred_marker
        
        return target
    
    def get_labels(self, index, target):
        start_index, end_index = self.video_indices[index]
        
        # SMPL parameters
        # NOTE: We use NeuralAnnot labels for Human36m and MPII3D only for the 0th frame input.
        #       We do not supervise the network on SMPL parameters.
        target['pose'] = transforms.axis_angle_to_matrix(
            self.labels['pose'][start_index:end_index+1].clone().reshape(-1, 24, 3))
        target['betas'] = self.labels['betas'][start_index:end_index+1].clone()        # No t
        
        # Apply SMPL augmentor (y-axis rotation and initial frame noise)
        target = self.SMPLAugmentor(target)
    
        # 3D and 2D keypoints
        if self.__name__ == 'ThreeDPW': # 3DPW has SMPL labels
            gt_kp3d = self.labels['joints3D'][start_index:end_index+1].clone()
            gt_kp2d = self.labels['joints2D'][start_index+1:end_index+1, ..., :2].clone()
            gt_kp3d = root_centering(gt_kp3d.clone()) # Center the root joint to the pelvis.
            
        else: # Human36m and MPII do not have SMPL labels
            gt_kp3d = torch.zeros((self.n_frames, self.n_joints + 14, 3))
            gt_kp3d[:, self.n_joints:] = convert_kps(self.labels['joints3D'][start_index:end_index+1], 'spin', 'common')
            gt_kp2d = torch.zeros((self.n_frames - 1, self.n_joints + 14, 2))
            gt_kp2d[:, self.n_joints:] = convert_kps(self.labels['joints2D'][start_index+1:end_index+1, ..., :2], 'spin', 'common')
        
        conf = self.mask.repeat(self.n_frames, 1).unsqueeze(-1)        
        gt_kp2d = torch.cat((gt_kp2d, conf[1:]), dim=-1)
        gt_kp3d = torch.cat((gt_kp3d, conf), dim=-1)
        target['kp3d'] = gt_kp3d
        target['full_kp2d'] = gt_kp2d
        target['weak_kp2d'] = torch.zeros_like(gt_kp2d)
        
        if self.__name__ != 'ThreeDPW': # 3DPW does not contain world-coordinate motion
            # Foot ground contact labels for Human36M and MPII3D
            target['contact'] = self.labels['stationaries'][start_index+1:end_index+1].clone()
        else:
            # No foot ground contact label available for 3DPW
            target['contact'] = torch.ones((self.n_frames - 1, 4)) * (-1)
            
        if self.has_verts:
            # SMPL vertices available for 3DPW
            with torch.no_grad():
                start_index, end_index = self.video_indices[index]
                if self.__name__ == 'ThreeDPW':
                    gender = self.labels['gender'][start_index].item()
                else:
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
        target['gt_marker'] = self.labels['gt_marker'][start_index:end_index+1] / 1000
            
        return target
    
    def get_init_frame(self, target):
        # Prepare initial frame
        output = self.smpl.get_output(
            body_pose=target['init_pose'][:, 1:],
            global_orient=target['init_pose'][:, :1],
            betas=target['betas'][:1],
            pose2rot=False
        )
        target['init_kp3d'] = root_centering(output.joints[:1, :self.n_joints]).reshape(1, -1)   
        
        return target
    
    def get_camera_info(self, index, target):
        start_index, end_index = self.video_indices[index]
        
        # Intrinsics
        target['res'] = self.labels['res'][start_index:end_index+1][0].clone()
        self.get_naive_intrinsics(target['res'])
        target['cam_intrinsics'] = self.cam_intrinsics.clone()
        
        # Extrinsics pose
        R = self.labels['cam_poses'][start_index:end_index+1, :3, :3].clone().float()
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
        self.get_init_frame(target)
        
        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)
        
        return target