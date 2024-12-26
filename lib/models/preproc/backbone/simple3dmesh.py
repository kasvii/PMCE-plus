from os import path as osp
import numpy as np
np.set_printoptions(suppress=True)
import time
import torch
import torch.nn as nn
# from torch.nn import functional as F

from lib.models.preproc.backbone.marker_config import cfg
from . import simple3dpose


class Simple3DMesh(nn.Module):
    def __init__(self, flip_pairs=None):
        super(Simple3DMesh, self).__init__()

        self.joint_num = cfg.dataset.num_joints

        self.simple3dpose = simple3dpose.get_model(flip_pairs=flip_pairs)

    def forward(self, x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item=None, flip_output=False, flip_mask=None, is_train=True):
        """Forward pass
        Inputs:
            x: image, size = (B, 3, 224, 224)
        Returns:
            pred_xyz_jts: camera 3d pose (joints + virtual markers), size = (B, J+K, 3)
            confidence: confidence score for each body point in 3d pose, size = (B, J+K), for loss_{conf}
            pred_uvd_jts_flat: uvd 3d pose (joints + virtual markers), size = (B, (J+K)*3), for loss_{pose}
            mesh3d: non-parametric 3d coordinates of mesh vertices, size = (B, V, 3), for loss_{mesh}
        """
        # batch_size = x.shape[0]
        # 3D pose estimation, get confidence from 3D heatmaps
        pred_xyz_jts, confidence, pred_uvd_jts_flat, pred_root_xy_img = self.simple3dpose(x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item, flip_output, flip_mask)       # (B, J+K, 3), (B, J+K)

        confidence_ret = confidence.clone()
        pred_xyz_jts_ret = pred_xyz_jts.clone()
    
        # detach pose3d to mesh for faster convergence
        pred_xyz_jts = pred_xyz_jts.detach()
        confidence = confidence.detach()
        
        adaptive_A = None
        mesh3d = None

        return pred_xyz_jts_ret, pred_uvd_jts_flat, adaptive_A, confidence_ret, mesh3d, None, pred_root_xy_img


def get_model(flip_pairs):
    model = Simple3DMesh(flip_pairs=flip_pairs)
    return model