from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch import nn
import numpy as np

from configs import constants as _C
from lib.models.layers import MotionDecoder
from lib.utils.transforms import axis_angle_to_matrix


class Network(nn.Module):
    def __init__(self, 
                 smpl,
                 seqlen,
                 pose_embed=256,
                 mesh_dr=0.1,
                 mesh_embed=512,
                 mesh_layers=3,
                 img_feat=2048,
                 **kwargs
                 ):
        super().__init__()
        
        n_joints = _C.KEYPOINTS.NUM_JOINTS
        n_marker = _C.KEYPOINTS.NUM_MARKER
        self.smpl = smpl
        
        # Module 4. Motion Decoder
        self.motion_decoder = MotionDecoder(num_frames=seqlen,
                                                 num_joint=n_joints,
                                                 num_marker=n_marker,
                                                 in_dim=pose_embed,
                                                 embed_dim=mesh_embed,
                                                 img_dim=img_feat,
                                                 hidden_dim=mesh_embed*2,
                                                 depth=mesh_layers)
        
        self.proj_marker = nn.Linear(3, pose_embed)
    
    def forward_smpl(self, **kwargs):
        self.output = self.smpl(self.pred_pose, 
                                self.pred_shape,
                                cam=self.pred_cam,
                                return_full_pose=not self.training,
                                **kwargs,
                                )
        
        # Return output
        output = {'contact': self.pred_contact,
                  'pose': self.pred_pose, 
                  'betas': self.pred_shape, 
                  'cam': self.pred_cam,
                  'poses_root_cam': self.output.global_orient,
                  'kp3d_nn': self.pred_kp3d,
                  'kp3d_nn_2': self.pred_kp3d_st2,
                  'marker_nn': self.pred_marker,
                  'marker_nn_2': self.pred_marker_st2,
                  'verts_cam': self.output.vertices}
        
        if self.training:
            output.update({
                'kp3d': self.output.joints,
                'full_kp2d': self.output.full_joints2d,
                'weak_kp2d': self.output.weak_joints2d,
            })
        else:
            output.update({
                'poses_body': self.output.body_pose,
                'trans_cam': self.output.full_cam})
        
        return output        
        
    def forward(self, joint_feats, marker_feats, img_features=None, 
                cam_intrinsics=None, bbox=None, res=None, **kwargs):
        B, T = joint_feats.shape[:2] # [64, 81, 17, 515]
        # TODO add marker #
        marker_feats = marker_feats[..., :3]
        pred_marker = marker_feats.reshape(B, T, 81, 3) # [8, 82, 81, 4]
        marker_feats = self.proj_marker(marker_feats)
        marker_feats = torch.cat((marker_feats, pred_marker), dim=-1)
            
        # Stage 4. Decode SMPL motion
        pred_pose, pred_shape, pred_cam, pred_contact, pred_kp3d_st2, pred_marker_st2 = self.motion_decoder(joint_feats, marker_feats, img_features)
        # --------- #
        
        # --------- Register predictions --------- #
        self.pred_kp3d = joint_feats[..., -3:].reshape(B, T, 17, 3)
        self.pred_kp3d_st2 = pred_kp3d_st2
        self.pred_marker = pred_marker
        self.pred_marker_st2 = pred_marker_st2
        
        self.pred_pose = pred_pose
        self.pred_shape = pred_shape
        self.pred_cam = pred_cam
        self.pred_contact = pred_contact
        # --------- #
        
        # --------- Build SMPL --------- #
        output = self.forward_smpl(cam_intrinsics=cam_intrinsics, bbox=bbox, res=res)
        # --------- #
        
        return output