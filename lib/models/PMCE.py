from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch import nn
import numpy as np

from configs import constants as _C
from lib.models import PoseEst, CoevoDecoder

class Network(nn.Module):
    def __init__(self, 
                 smpl,
                 seqlen,
                 pose_checkpoint_path,
                 **kwargs
                 ):
        super().__init__()
        
        self.pose_lifter = PoseEst.Network(smpl, seqlen, pose_checkpoint_path, **kwargs)
        self.mesh_regressor = CoevoDecoder.Network(smpl, seqlen, **kwargs)
        
    def forward(self, x, marker, img_features=None, mask=None,
                cam_intrinsics=None, bbox=None, res=None, **kwargs):
        pose_output = self.pose_lifter(x, img_features, mask)
        joint_feats = pose_output['joint_feats']
        output = self.mesh_regressor(joint_feats, marker, img_features, cam_intrinsics, bbox, res)
        
        return output