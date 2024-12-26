from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

from ..utils.normalizer import Normalizer
from ...models import build_body_model
from ...utils import transforms
from ...utils.kp_utils import root_centering
from ...utils.imutils import compute_cam_intrinsics
import sys
import numpy as np
from skimage.util.shape import view_as_windows


KEYPOINTS_THR = 0.3

def convert_dpvo_to_cam_angvel(traj, fps):
    """Function to convert DPVO trajectory output to camera angular velocity"""
    
    # 0 ~ 3: translation, 3 ~ 7: Quaternion
    quat = traj[:, 3:]
    
    # Convert (x,y,z,q) to (q,x,y,z)
    quat = quat[:, [3, 0, 1, 2]]
    
    # Quat is camera to world transformation. Convert it to world to camera
    world2cam = transforms.quaternion_to_matrix(torch.from_numpy(quat)).float()
    R = world2cam.mT
    
    # Compute the rotational changes over time.
    cam_angvel = transforms.matrix_to_axis_angle(R[:-1] @ R[1:].transpose(-1, -2))
    
    # Convert matrix to 6D representation
    cam_angvel = transforms.matrix_to_rotation_6d(transforms.axis_angle_to_matrix(cam_angvel))
    
    # Normalize 6D angular velocity
    cam_angvel = cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel) # Normalize
    cam_angvel = cam_angvel * fps
    cam_angvel = torch.cat((cam_angvel, cam_angvel[:1]), dim=0)
    return cam_angvel


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, tracking_results, width, height, fps):
        
        self.tracking_results = tracking_results
        self.width = width
        self.height = height
        self.fps = fps
        self.res = torch.tensor([width, height]).float()
        self.intrinsics = compute_cam_intrinsics(self.res)
        self.n_frames = cfg.DATASET.SEQLEN
        
        self.device = cfg.DEVICE.lower()
        
        self.smpl = build_body_model('cpu')
        self.keypoints_normalizer = Normalizer(cfg)
        
        self._to = lambda x: x.unsqueeze(0).to(self.device)
        
        self.prepare_video_batch()
        
    def prepare_video_batch(self):
        self.video_indices = []
        if 'frame_id' not in self.tracking_results.keys():
            return
        indexes = np.arange(len(self.tracking_results['frame_id']))
        if indexes.shape[0] < self.n_frames: return
        chunks = view_as_windows(
            indexes, (self.n_frames), step=self.n_frames
        )
        start_finish = chunks[:, (0, -1)].tolist()
        self.video_indices += start_finish  
        
    def __len__(self):
        return len(self.video_indices)

    def load_data(self, index, flip=False):
        if flip:
            self.prefix = 'flipped_'
        else:
            self.prefix = ''
        
        return self.__getitem__(index)
    
    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        scale = torch.stack([torch.tensor(1.0), h / w], dim=-1)
        return X / w * 2 - scale
    
    def __getitem__(self, index):
        if index >= len(self): return
        start_index, end_index = self.video_indices[index]
        
        # Process 2D keypoints
        kp2d = torch.from_numpy(self.tracking_results[self.prefix + 'keypoints'][start_index:end_index+1][..., :2]).float()
        mask = torch.from_numpy(self.tracking_results[self.prefix + 'keypoints'][start_index:end_index+1][..., -1] < KEYPOINTS_THR)
        bbox = torch.from_numpy(self.tracking_results[self.prefix + 'bbox'][start_index:end_index+1]).float()
        _, bbox = self.keypoints_normalizer(
            kp2d.clone(), self.res, self.intrinsics, 224, 224, bbox
        )
        
        kp2d = self.normalize_screen_coordinates(kp2d, w=self.res[0], h=self.res[1])
        kp2d = kp2d.reshape(kp2d.shape[0], -1)
        
        # Process image features
        features = self.tracking_results[self.prefix + 'features'][start_index:end_index+1]
        
        marker = torch.from_numpy(self.tracking_results[self.prefix + 'marker'][start_index:end_index+1]).float()
        marker[..., :3] = marker[..., :3] / 1000
        
        return (
            self._to(kp2d),                                 # 2d keypoints
            self._to(marker),                               # 3d keypoints
            self._to(features),                             # image features
            self._to(mask),                                 # keypoints mask
            self.tracking_results['frame_id'][start_index:end_index+1],              # frame indices
            {'cam_intrinsics': self._to(self.intrinsics),   # other keyword arguments
             'bbox': self._to(bbox),
             'res': self._to(self.res)},
            )