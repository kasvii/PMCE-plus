from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import joblib
import pickle
import numpy as np
import scipy.stats as stats
from lib.utils import transforms
import scipy.sparse as ssp
from smplx.lbs import vertices2joints

from configs import constants as _C

from ..utils.augmentor import *
from .._dataset import BaseDataset
from ...models import build_body_model
from ...utils import data_utils as d_utils
from ...utils.kp_utils import root_centering

def compute_contact_label(feet, thr=1e-2, alpha=5):
    vel = torch.zeros_like(feet[..., 0])
    label = torch.zeros_like(feet[..., 0])
    
    vel[1:-1] = (feet[2:] - feet[:-2]).norm(dim=-1) / 2.0
    vel[0] = vel[1].clone()
    vel[-1] = vel[-2].clone()
    
    label = 1 / (1 + torch.exp(alpha * (thr ** -1) * (vel - thr)))
    return label


class AMASSDataset(BaseDataset):
    def __init__(self, cfg):
        label_pth = _C.PATHS.AMASS_LABEL
        super(AMASSDataset, self).__init__(cfg, training=True)

        self.supervise_pose = cfg.TRAIN.STAGE == 'stage1'
        self.labels = joblib.load(label_pth)
        if cfg.DEBUG:
            for key in self.labels.keys():
                self.labels[key] = self.labels[key][0:458]
        self.SequenceAugmentor = SequenceAugmentor(cfg.DATASET.SEQLEN + 1)

        # Load augmentators
        self.VideoAugmentor = VideoAugmentor(cfg)
        self.SMPLAugmentor = SMPLAugmentor(cfg)
        self.d_img_feature = _C.IMG_FEAT_DIM[cfg.MODEL.BACKBONE]
        
        self.n_frames = int(cfg.DATASET.SEQLEN * self.SequenceAugmentor.l_factor) + 1
        self.smpl = build_body_model('cpu', self.n_frames)
        self.prepare_video_batch()
        
        # Naive assumption of image intrinsics
        self.img_w, self.img_h = 1000, 1000
        self.get_naive_intrinsics((self.img_w, self.img_h))
        
        self.CameraAugmentor = CameraAugmentor(cfg.DATASET.SEQLEN + 1, self.img_w, self.img_h, self.focal_length)
        
        # load markers
        self.human36_joint_num = 17
        vm_B = ssp.load_npz(_C.BMODEL.MARKER_REGRESSOR_B).A.astype(float)
        self.vm_B = torch.tensor(vm_B, dtype=torch.float32).T
        joints_regressor_h36m = np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
        self.joints_regressor_h36m = torch.tensor(joints_regressor_h36m, dtype=torch.float32)

        self.vm_joint_num = self.joints_regressor_h36m.shape[0] + self.vm_B.shape[0]
        
        self.selected_indices = [i for i in range(6890)]
        with open(_C.BMODEL.SMPL_INDICES, 'rb') as f:
            smpl_indices = pickle.load(f)
        for body_part in smpl_indices.keys():
            body_part_indices = list(smpl_indices[body_part].numpy())
            if body_part in _C.BMODEL.IGNORE_PART:
                for idx in body_part_indices:
                    self.selected_indices.remove(idx)
         
    @property
    def __name__(self, ):
        return 'AMASS'
    
    def generate_syn_error(self, batch, vertex_num):
        mean = 0.
        std = 0.05
        size = (batch, vertex_num, 3)
        noise = np.random.normal(loc=mean, scale=std, size=size)
        pdf_values = stats.norm.pdf(noise, mean, std)
        # cdf_values = stats.norm.cdf(noise, mean, std)
        highest_value = 1 / (std * np.sqrt(2 * np.pi))
        pdf_values = pdf_values / highest_value
        pdf_values = np.mean(pdf_values, axis=-1, keepdims=True)

        return noise, pdf_values
    
    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        scale = torch.stack([torch.tensor(1.0), torch.tensor(h / w)], dim=-1)
        return X / w * 2 - scale
    
    def get_input(self, target):
        gt_kp3d = target['kp3d']
        gt_marker = target['gt_marker']
        noise, confidence = self.generate_syn_error(gt_marker.shape[0], self.vm_joint_num)
        noise, confidence = torch.from_numpy(noise).float(), torch.from_numpy(confidence).float()
        inpt_marker = gt_marker + noise
        inpt_marker = torch.cat([inpt_marker, confidence], dim=-1)
        inpt_kp3d = self.VideoAugmentor(gt_kp3d[:, :self.n_joints, :-1].clone())
        kp2d_proj = perspective_projection(inpt_kp3d, self.cam_intrinsics)
        kp2d = kp2d_proj.clone()
        mask = self.VideoAugmentor.get_mask()
        kp2d_proj, bbox = self.keypoints_normalizer(kp2d_proj, target['res'], self.cam_intrinsics, 224, 224)    
        
        # full img kp2d
        kp2d = self.normalize_screen_coordinates(kp2d, w=self.img_w, h=self.img_h)
        kp2d = kp2d.reshape(kp2d.shape[0], -1)
        
        target['bbox'] = bbox[1:]
        target['kp2d'] = kp2d
        target['pred_marker'] = inpt_marker
        target['mask'] = mask[1:]
        target['features'] = torch.zeros((self.SMPLAugmentor.n_frames, self.d_img_feature)).float()
        
        return target
    
    def get_groundtruth(self, target):
        # GT 1. Joints
        gt_kp3d = target['kp3d']
        gt_kp2d = perspective_projection(gt_kp3d, self.cam_intrinsics)
        target['kp3d'] = torch.cat((gt_kp3d, torch.ones_like(gt_kp3d[..., :1]) * float(self.supervise_pose)), dim=-1)
        target['full_kp2d'] = torch.cat((gt_kp2d, torch.ones_like(gt_kp2d[..., :1]) * float(self.supervise_pose)), dim=-1)[1:]
        target['weak_kp2d'] = torch.zeros_like(target['full_kp2d'])
        target['init_kp3d'] = root_centering(gt_kp3d[:1, :self.n_joints].clone()).reshape(1, -1)
        target['verts'] = torch.zeros((self.SMPLAugmentor.n_frames, 6890, 3)).float()
        
        # GT 2. Root pose
        vel_world = (target['transl'][1:] - target['transl'][:-1])
        pose_root = target['pose_root'].clone()
        vel_root = (pose_root[:-1].transpose(-1, -2) @ vel_world.unsqueeze(-1)).squeeze(-1)
        target['vel_root'] = vel_root.clone()
        target['pose_root'] = transforms.matrix_to_rotation_6d(pose_root)
        target['init_root'] = target['pose_root'][:1].clone()
        
        # GT 3. Foot contact
        contact = compute_contact_label(target['feet'])
        if 'tread' in target['vid']:
            target['contact'] = torch.ones_like(contact) * (-1)
        else:
            target['contact'] = contact
        
        return target
    
    def forward_smpl(self, target):
        output = self.smpl.get_output(
            body_pose=torch.cat((target['init_pose'][:, 1:], target['pose'][1:, 1:])),
            global_orient=torch.cat((target['init_pose'][:, :1], target['pose'][1:, :1])),
            betas=target['betas'],
            pose2rot=False)
        
        target['transl'] = target['transl'] - output.offset
        target['transl'] = target['transl'] - target['transl'][0]
        target['kp3d'] = output.joints
        gt_marker = vertices2joints(self.vm_B, output.vertices[:, self.selected_indices])
        gt_joint_h36m = vertices2joints(self.joints_regressor_h36m, output.vertices)
        gt_marker = torch.cat([gt_joint_h36m, gt_marker], dim=-2)
        target['gt_marker'] = gt_marker
        target['feet'] = output.feet[1:] + target['transl'][1:].unsqueeze(-2)
        
        return target
    
    def augment_data(self, target):
        # Augmentation 1. SMPL params augmentation
        target = self.SMPLAugmentor(target)
        
        # Augmentation 2. Sequence speed augmentation
        target = self.SequenceAugmentor(target)

        # Get world-coordinate SMPL
        target = self.forward_smpl(target)
        
        # Augmentation 3. Virtual camera generation
        target = self.CameraAugmentor(target)
        
        return target
    
    def load_amass(self, index, target):
        start_index, end_index = self.video_indices[index]
        
        # Load AMASS labels
        pose = torch.from_numpy(self.labels['pose'][start_index:end_index+1].copy())
        pose = transforms.axis_angle_to_matrix(pose.reshape(-1, 24, 3))
        transl = torch.from_numpy(self.labels['transl'][start_index:end_index+1].copy())
        betas = torch.from_numpy(self.labels['betas'][start_index:end_index+1].copy())
        
        # Stack GT
        target.update({'vid': self.labels['vid'][start_index], 
                  'pose': pose, 
                  'transl': transl, 
                  'betas': betas})

        return target

    def get_single_sequence(self, index):
        target = {'res': torch.tensor([self.img_w, self.img_h]).float(),
                  'cam_intrinsics': self.cam_intrinsics.clone(),
                  'has_full_screen': torch.tensor(True),
                  'has_smpl': torch.tensor(self.supervise_pose),
                  'has_traj': torch.tensor(True),
                  'has_verts': torch.tensor(False),}
        
        target = self.load_amass(index, target)
        target = self.augment_data(target)
        target = self.get_groundtruth(target)
        target = self.get_input(target)
        
        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)

        return target
    

def perspective_projection(points, cam_intrinsics, rotation=None, translation=None):
    K = cam_intrinsics
    if rotation is not None:
        points = torch.matmul(rotation, points.transpose(1, 2)).transpose(1, 2)
    if translation is not None:
        points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points.float())
    return projected_points[:, :, :-1]