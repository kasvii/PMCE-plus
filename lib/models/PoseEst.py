from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import torch
from torch import nn
import numpy as np

from configs import constants as _C
from lib.models.layers import MotionEncoder

class Network(nn.Module):
    def __init__(self, 
                 smpl,
                 seqlen,
                 pose_checkpoint_path,
                 pose_dr=0.1,
                 pose_embed=256,
                 pose_layers=5,
                 img_feat=2048,
                 **kwargs
                 ):
        super().__init__()
        
        n_joints = _C.KEYPOINTS.NUM_JOINTS
        self.smpl = smpl
        in_dim = n_joints * 2 + 3
        d_context = pose_embed + n_joints * 3
        
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, n_joints, 2))        
        
        # Module 1. Motion Encoder
        self.motion_encoder = MotionEncoder(num_frames=seqlen,
                                                 num_joints=n_joints,
                                                 embed_dim=pose_embed,
                                                 img_dim=img_feat,
                                                 depth=pose_layers) 
        
        if os.path.isfile(pose_checkpoint_path):
            print(f"Loading pretrained posenet from {pose_checkpoint_path}")
            checkpoint = torch.load(pose_checkpoint_path, map_location='cuda')
            ignore_keys = ['smpl.body_pose', 'smpl.betas', 'smpl.global_orient', 'smpl.J_regressor_extra', 'smpl.J_regressor_eval']
            model_state_dict = {k: v for k, v in checkpoint['model'].items() if k not in ignore_keys}
            self.load_state_dict(model_state_dict, strict=False)
            print(f"=> loaded checkpoint '{pose_checkpoint_path}' ")
        else:
            print(f"=> Warning! no checkpoint found at '{pose_checkpoint_path}'.")
            
    def preprocess(self, x, mask):
        self.b, self.f = x.shape[:2]
        
        # Treat masked keypoints
        mask_embedding = mask.unsqueeze(-1) * self.mask_embedding
        _mask = mask.unsqueeze(-1).repeat(1, 1, 1, 2).reshape(self.b, self.f, -1)
        _mask_embedding = mask_embedding.reshape(self.b, self.f, -1)
        x[_mask] = 0.0
        x = x + _mask_embedding
        return x
        
    def forward(self, x, img_features=None, mask=None, **kwargs):
        
        if mask is not None:
            x = self.preprocess(x, mask)
        
        # Stage 1. Estimate 3D Pose Sequence
        pred_kp3d, joint_feats = self.motion_encoder(x, img_features)
        
        output = {
            'kp3d_nn': pred_kp3d,
            'joint_feats': joint_feats,
        }
        
        return output