from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import functional as F

from configs import constants as _C
from lib.utils import transforms
from lib.utils.kp_utils import root_centering

class WHAMLoss(nn.Module):
    def __init__(
            self,
            cfg=None,
            device=None,
    ):
        super(WHAMLoss, self).__init__()
        
        self.cfg = cfg
        self.n_joints = _C.KEYPOINTS.NUM_JOINTS
        self.criterion = nn.MSELoss()
        self.criterion_noreduce = nn.MSELoss(reduction='none')
        
        self.pose_loss_weight = cfg.LOSS.POSE_LOSS_WEIGHT
        self.shape_loss_weight = cfg.LOSS.SHAPE_LOSS_WEIGHT
        self.keypoint_2d_loss_weight = cfg.LOSS.JOINT2D_LOSS_WEIGHT
        self.keypoint_3d_loss_weight = cfg.LOSS.JOINT3D_LOSS_WEIGHT
        self.vel_3d_loss_weight = cfg.LOSS.VEL_LOSS_WEIGHT
        self.marker_3d_loss_weight = cfg.LOSS.MARKER3D_LOSS_WEIGHT
        self.cascaded_loss_weight = cfg.LOSS.CASCADED_LOSS_WEIGHT
        self.vertices_loss_weight = cfg.LOSS.VERTS3D_LOSS_WEIGHT
        self.contact_loss_weight = cfg.LOSS.CONTACT_LOSS_WEIGHT
        self.root_vel_loss_weight = cfg.LOSS.ROOT_VEL_LOSS_WEIGHT
        self.root_pose_loss_weight = cfg.LOSS.ROOT_POSE_LOSS_WEIGHT
        self.sliding_loss_weight = cfg.LOSS.SLIDING_LOSS_WEIGHT
        self.camera_loss_weight = cfg.LOSS.CAMERA_LOSS_WEIGHT
        self.loss_weight = cfg.LOSS.LOSS_WEIGHT
        
        kp_weights = [
            0.5, 0.5, 0.5, 0.5, 0.5, # Face
            1.5, 1.5, 4, 4, 4, 4,    # Arms
            1.5, 1.5, 4, 4, 4, 4,    # Legs
            4, 4, 1.5, 1.5, 4, 4,   # Legs
            4, 4, 1.5, 1.5, 4, 4,   # Arms
            0.5, 0.5            # Head
        ]
        
        theta_weights = [
            0.1, 1.0, 1.0, 1.0, 1.0,        # pelvis, lhip, rhip, spine1, lknee
            1.0, 1.0, 1.0, 1.0, 1.0,        # rknn, spine2, lankle, rankle, spin3
            0.1, 0.1,                       # Foot
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # neck, lisldr, risldr, head, losldr, rosldr,
            1.0, 1.0, 1.0, 1.0,             # lelbow, relbow, lwrist, rwrist
            0.1, 0.1,                       # Hand
        ]
        self.theta_weights = torch.tensor([[theta_weights]]).float().to(device)
        self.theta_weights /= self.theta_weights.mean()
        self.kp_weights = torch.tensor([kp_weights]).float().to(device)
        
        self.epoch = -1
        self.step()

    def step(self):
        self.epoch += 1
        self.skip_camera_loss = self.epoch < self.cfg.LOSS.CAMERA_LOSS_SKIP_EPOCH

    def forward(self, pred, gt):
        
        loss = 0.0
        b, f = gt['kp3d'].shape[:2]
        
        # <======= Predictions and Groundtruths
        pred_kp3d_nn = pred['kp3d_nn']     # [64, 81, 17, 3]
        gt_kp3d = root_centering(gt['kp3d'])
        bbox = gt['bbox']
        # =======>
        
        # Compute 3D keypoint loss
        loss_keypoints_3d_nn = keypoint_3d_loss(
            pred_kp3d_nn,
            gt_kp3d[:, :, :self.n_joints],
            self.kp_weights[:, :self.n_joints],
            criterion=self.criterion_noreduce,
        )
        
        # Compute 3D keypoint velocity loss
        loss_vel_3d_nn = vel_3d_loss(
            pred_kp3d_nn,
            gt_kp3d[:, :, :self.n_joints],
            self.kp_weights[:, :self.n_joints],
            criterion=self.criterion_noreduce,
        )
        
        loss_keypoints_3d_nn *= self.keypoint_3d_loss_weight
        loss_vel_3d_nn *= self.vel_3d_loss_weight
        
        
        loss_dict = {
            '3d_nn': loss_keypoints_3d_nn * self.loss_weight,
            'vel': loss_vel_3d_nn * self.loss_weight,
        }
        
        loss = sum(loss for loss in loss_dict.values())
        
        return loss, loss_dict


class WHAMLoss_stage1(nn.Module):
    def __init__(
            self,
            cfg=None,
            device=None,
    ):
        super(WHAMLoss_stage1, self).__init__()
        
        self.cfg = cfg
        self.n_joints = _C.KEYPOINTS.NUM_JOINTS
        self.criterion = nn.MSELoss()
        self.criterion_noreduce = nn.MSELoss(reduction='none')
        
        self.keypoint_3d_loss_weight = cfg.LOSS.JOINT3D_LOSS_WEIGHT
        self.vel_3d_loss_weight = cfg.LOSS.VEL_LOSS_WEIGHT
        self.accel_3d_loss_weight = cfg.LOSS.ACCEL_LOSS_WEIGHT
        self.loss_weight = cfg.LOSS.LOSS_WEIGHT
        
        kp_weights = [
            0.5, 0.5, 0.5, 0.5, 0.5, # Face
            1.5, 1.5, 4, 4, 4, 4,    # Arms
            1.5, 1.5, 4, 4, 4, 4,    # Legs
            4, 4, 1.5, 1.5, 4, 4,   # Legs
            4, 4, 1.5, 1.5, 4, 4,   # Arms
            0.5, 0.5            # Head
        ]
        
        self.kp_weights = torch.tensor([kp_weights]).float().to(device)
        
        self.epoch = -1
        self.step()

    def step(self):
        self.epoch += 1
        self.skip_camera_loss = self.epoch < self.cfg.LOSS.CAMERA_LOSS_SKIP_EPOCH

    def forward(self, pred, gt):
        
        loss = 0.0
        b, f = gt['kp3d'].shape[:2]
        
        # <======= Predictions and Groundtruths
        pred_kp3d_nn = pred['kp3d_nn']     # [64, 81, 17, 3]
        
        gt_kp3d = root_centering(gt['kp3d'])
        bbox = gt['bbox']
        # =======>

        # Compute 3D keypoint loss
        loss_keypoints_3d_nn = keypoint_3d_loss(
            pred_kp3d_nn,
            gt_kp3d[:, :, :self.n_joints],
            self.kp_weights[:, :self.n_joints],
            criterion=self.criterion_noreduce,
        )
        
        # Compute 3D keypoint accel loss
        loss_accel_3d_nn = accel_3d_loss(
            pred_kp3d_nn,
            gt_kp3d[:, :, :self.n_joints],
            self.kp_weights[:, :self.n_joints],
            criterion=self.criterion_noreduce,
        )
        
        loss_keypoints_3d_nn *= self.keypoint_3d_loss_weight
        loss_accel_3d_nn *= self.accel_3d_loss_weight
        
        loss_dict = {
            '3d_nn': loss_keypoints_3d_nn * self.loss_weight,
            # 'vel': loss_vel_3d_nn * self.loss_weight,
            'accel': loss_accel_3d_nn * self.loss_weight,
        }
        
        loss = sum(loss for loss in loss_dict.values())
        
        return loss, loss_dict

class WHAMLoss_stage2(nn.Module):
    def __init__(
            self,
            cfg=None,
            device=None,
    ):
        super(WHAMLoss_stage2, self).__init__()
        
        self.cfg = cfg
        self.n_joints = _C.KEYPOINTS.NUM_JOINTS
        self.n_marker = _C.KEYPOINTS.NUM_MARKER
        self.criterion = nn.MSELoss()
        self.criterion_noreduce = nn.MSELoss(reduction='none')
        
        self.pose_loss_weight = cfg.LOSS.POSE_LOSS_WEIGHT
        self.shape_loss_weight = cfg.LOSS.SHAPE_LOSS_WEIGHT
        self.keypoint_2d_loss_weight = cfg.LOSS.JOINT2D_LOSS_WEIGHT
        self.keypoint_3d_loss_weight = cfg.LOSS.JOINT3D_LOSS_WEIGHT
        self.marker_3d_loss_weight = cfg.LOSS.MARKER3D_LOSS_WEIGHT
        self.cascaded_loss_weight = cfg.LOSS.CASCADED_LOSS_WEIGHT
        self.vertices_loss_weight = cfg.LOSS.VERTS3D_LOSS_WEIGHT
        self.contact_loss_weight = cfg.LOSS.CONTACT_LOSS_WEIGHT
        self.root_vel_loss_weight = cfg.LOSS.ROOT_VEL_LOSS_WEIGHT
        self.root_pose_loss_weight = cfg.LOSS.ROOT_POSE_LOSS_WEIGHT
        self.sliding_loss_weight = cfg.LOSS.SLIDING_LOSS_WEIGHT
        self.camera_loss_weight = cfg.LOSS.CAMERA_LOSS_WEIGHT
        self.loss_weight = cfg.LOSS.LOSS_WEIGHT
        self.accel_3d_loss_weight = cfg.LOSS.ACCEL_LOSS_WEIGHT
        
        
        kp_weights = [
            0.5, 0.5, 0.5, 0.5, 0.5, # Face
            1.5, 1.5, 4, 4, 4, 4,    # Arms
            1.5, 1.5, 4, 4, 4, 4,    # Legs
            4, 4, 1.5, 1.5, 4, 4,   # Legs
            4, 4, 1.5, 1.5, 4, 4,   # Arms
            0.5, 0.5            # Head
        ]
        
        marker_weights = [1.0] * self.n_marker
        
        theta_weights = [
            0.1, 1.0, 1.0, 1.0, 1.0,        # pelvis, lhip, rhip, spine1, lknee
            1.0, 1.0, 1.0, 1.0, 1.0,        # rknn, spine2, lankle, rankle, spin3
            0.1, 0.1,                       # Foot
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # neck, lisldr, risldr, head, losldr, rosldr,
            1.0, 1.0, 1.0, 1.0,             # lelbow, relbow, lwrist, rwrist
            0.1, 0.1,                       # Hand
        ]
        self.theta_weights = torch.tensor([[theta_weights]]).float().to(device)
        self.theta_weights /= self.theta_weights.mean()
        self.kp_weights = torch.tensor([kp_weights]).float().to(device)
        self.marker_weights = torch.tensor([marker_weights]).float().to(device)
        
        self.epoch = -1
        self.step()

    def step(self):
        self.epoch += 1
        self.skip_camera_loss = self.epoch < self.cfg.LOSS.CAMERA_LOSS_SKIP_EPOCH

    def forward(self, pred, gt):
        
        loss = 0.0
        b, f = gt['kp3d'].shape[:2]
        
        # <======= Predictions and Groundtruths
        pred_betas = pred['betas']                    # [64, 81, 10]
        pred_pose = pred['pose'].reshape(b, f, -1, 6) # [64, 81, 24, 6]
        pred_kp3d_nn = pred['kp3d_nn_2']   # [64, 81, 17, 3]
        pred_marker_nn = pred['marker_nn_2']   # [64, 81, 17, 3]
        
        pred_kp3d_smpl = root_centering(pred['kp3d'].reshape(b, f, -1, 3)) # [64, 81, 31, 3]
        pred_full_kp2d = pred['full_kp2d'] # [64, 81, 31, 2]
        pred_weak_kp2d = pred['weak_kp2d'] # [64, 81, 31, 2]
        pred_contact = pred['contact']     # [64, 81, 4]
        
        gt_betas = gt['betas']
        gt_pose = gt['pose']
        gt_kp3d = root_centering(gt['kp3d'])
        gt_marker = gt['gt_marker']
        gt_full_kp2d = gt['full_kp2d']
        gt_weak_kp2d = gt['weak_kp2d']
        gt_contact = gt['contact']
        bbox = gt['bbox']
        # =======>
        
        loss_keypoints_full = full_projected_keypoint_loss(
            pred_full_kp2d, 
            gt_full_kp2d, 
            bbox, 
            self.kp_weights, 
            criterion=self.criterion_noreduce, 
        )
        
        loss_keypoints_weak = weak_projected_keypoint_loss(
            pred_weak_kp2d,
            gt_weak_kp2d,
            self.kp_weights,
            criterion=self.criterion_noreduce
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d_nn = keypoint_3d_loss(
            pred_kp3d_nn,
            gt_kp3d[:, :, :self.n_joints],
            self.kp_weights[:, :self.n_joints],
            criterion=self.criterion_noreduce,
        )
        
        loss_marker_3d_nn = keypoint_3d_loss(
            pred_marker_nn,
            gt_marker,
            self.marker_weights,
            criterion=self.criterion_noreduce,
        )
        
        loss_keypoints_3d_smpl = keypoint_3d_loss(
            pred_kp3d_smpl,
            gt_kp3d,
            self.kp_weights,
            criterion=self.criterion_noreduce,
        )
        
        # Compute 3D keypoint accel loss
        loss_accel_3d_nn = accel_3d_loss(
            pred_kp3d_nn,
            gt_kp3d[:, :, :self.n_joints],
            self.kp_weights[:, :self.n_joints],
            criterion=self.criterion_noreduce,
        )
        
        loss_cascaded = keypoint_3d_loss(
            pred_kp3d_nn,
            torch.cat((pred_kp3d_smpl[:, :, :self.n_joints], gt_kp3d[:, :, :self.n_joints, -1:]), dim=-1),
            self.kp_weights[:, :self.n_joints] * 0.5,
            criterion=self.criterion_noreduce,
        )
        
        loss_vertices = vertices_loss(
            pred['verts_cam'],
            gt['verts'],
            gt['has_verts'],
            criterion=self.criterion_noreduce,
        )
                
        # Compute loss on SMPL parameters
        smpl_mask = gt['has_smpl']
        loss_regr_pose, loss_regr_betas = smpl_losses(
            pred_pose,
            pred_betas,
            gt_pose,
            gt_betas,
            self.theta_weights,
            smpl_mask,
            criterion=self.criterion_noreduce
        )
        
        # Compute loss on foot contact
        loss_contact = contact_loss(
            pred_contact,
            gt_contact,
            self.criterion_noreduce
        )
                        
        loss_keypoints = loss_keypoints_full + loss_keypoints_weak
        loss_keypoints *= self.keypoint_2d_loss_weight
        loss_keypoints_3d_smpl *= self.keypoint_3d_loss_weight
        loss_keypoints_3d_nn *= self.keypoint_3d_loss_weight
        loss_marker_3d_nn *= self.marker_3d_loss_weight
        loss_cascaded *= self.cascaded_loss_weight
        loss_vertices *= self.vertices_loss_weight
        loss_contact *= self.contact_loss_weight
        loss_accel_3d_nn *= self.accel_3d_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.shape_loss_weight
        
        loss_dict = {
            'pose': loss_regr_pose * self.loss_weight,
            'betas': loss_regr_betas * self.loss_weight,
            '2d': loss_keypoints * self.loss_weight,
            '3d': loss_keypoints_3d_smpl * self.loss_weight,
            '3d_nn': loss_keypoints_3d_nn * self.loss_weight,
            '3dmarker_nn': loss_marker_3d_nn * self.loss_weight,
            'casc': loss_cascaded * self.loss_weight,
            'v3d': loss_vertices * self.loss_weight,
            'contact': loss_contact * self.loss_weight,
            'accel_nn': loss_accel_3d_nn * self.loss_weight,
        }
        
        loss = sum(loss for loss in loss_dict.values())
        
        return loss, loss_dict

def root_loss(
    pred_vel_root,
    pred_pose_root,
    gt_vel_root,
    gt_pose_root,
    stationary,
    criterion
):
    
    mask_r = (gt_pose_root != 0.0).all(dim=-1).all(dim=-1)
    mask_v = (gt_vel_root != 0.0).all(dim=-1).all(dim=-1)
    mask_s = (stationary != -1).any(dim=1).any(dim=1)
    mask_v = mask_v * mask_s
    
    if mask_r.any():
        loss_r = criterion(pred_pose_root, gt_pose_root)[mask_r].mean()
    else:
        loss_r = torch.FloatTensor(1).fill_(0.).to(gt_pose_root.device)[0]
    
    if mask_v.any():
        loss_v = 0
        T = gt_vel_root.shape[0]
        ws_list = [1, 3, 9, 27]
        for ws in ws_list:
            tmp_v = 0
            for m in range(T//ws):
                cumulative_v = torch.sum(pred_vel_root[:, m:(m+1)*ws] - gt_vel_root[:, m:(m+1)*ws], dim=1)
                tmp_v += torch.norm(cumulative_v, dim=-1)
            loss_v += tmp_v
        loss_v = loss_v[mask_v].mean()
    else:
        loss_v = torch.FloatTensor(1).fill_(0.).to(gt_vel_root.device)[0]

    return loss_v, loss_r


def contact_loss(
        pred_stationary,
        gt_stationary,
        criterion,
):
    
    mask = gt_stationary != -1
    if mask.any():
        loss = criterion(pred_stationary, gt_stationary)[mask].mean()
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(gt_stationary.device)[0]
    return loss



def full_projected_keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        bbox,
        weight,
        criterion,
):
    
    scale = bbox[..., 2:] * 200.
    conf = gt_keypoints_2d[..., -1]
    
    if (conf > 0).any():
        loss = torch.mean(
            weight * (conf * torch.norm(pred_keypoints_2d - gt_keypoints_2d[..., :2], dim=-1)
        ) / scale, dim=1).mean() * conf.mean()
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(gt_keypoints_2d.device)[0]
    return loss


def weak_projected_keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        weight,
        criterion,
):
    
    conf = gt_keypoints_2d[..., -1]
    if (conf > 0).any():
        loss = torch.mean(
            weight * (conf * torch.norm(pred_keypoints_2d - gt_keypoints_2d[..., :2], dim=-1)
        ), dim=1).mean() * conf.mean() * 5
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(gt_keypoints_2d.device)[0]
    return loss


def keypoint_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        weight,
        criterion,
):
    
    conf = gt_keypoints_3d[..., -1]
    if (conf > 0).any():
        if weight.shape[-2] > 17:
            pred_keypoints_3d[..., -14:] = pred_keypoints_3d[..., -14:] - pred_keypoints_3d[..., -14:].mean(dim=-2, keepdims=True)
            gt_keypoints_3d[..., -14:] = gt_keypoints_3d[..., -14:] - gt_keypoints_3d[..., -14:].mean(dim=-2, keepdims=True)
        
        loss = torch.mean(
            weight * (conf * torch.norm(pred_keypoints_3d - gt_keypoints_3d[..., :3], dim=-1)
        ), dim=1).mean() * conf.mean()
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(gt_keypoints_3d.device)[0]
    return loss

def vel_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        weight,
        criterion,
):
    
    conf = gt_keypoints_3d[..., -1]
    if (conf > 0).any():
        if weight.shape[-2] > 17:
            pred_keypoints_3d[..., -14:] = pred_keypoints_3d[..., -14:] - pred_keypoints_3d[..., -14:].mean(dim=-2, keepdims=True)
            gt_keypoints_3d[..., -14:] = gt_keypoints_3d[..., -14:] - gt_keypoints_3d[..., -14:].mean(dim=-2, keepdims=True)
        
        vel_pred = pred_keypoints_3d[:, 1:] - pred_keypoints_3d[:, :-1]
        vel_gt = gt_keypoints_3d[:, 1:] - gt_keypoints_3d[:, :-1]
        
        loss = torch.mean(
            weight * (conf[:, :-1] * torch.norm(vel_pred - vel_gt[..., :3], dim=-1)
        ), dim=1).mean() * conf.mean()
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(gt_keypoints_3d.device)[0]
    return loss

def accel_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        weight,
        criterion,
):
    
    conf = gt_keypoints_3d[..., -1]
    if (conf > 0).any():
        if weight.shape[-2] > 17:
            pred_keypoints_3d[..., -14:] = pred_keypoints_3d[..., -14:] - pred_keypoints_3d[..., -14:].mean(dim=-2, keepdims=True)
            gt_keypoints_3d[..., -14:] = gt_keypoints_3d[..., -14:] - gt_keypoints_3d[..., -14:].mean(dim=-2, keepdims=True)
        
        accel_pred = pred_keypoints_3d[:, :-2] - 2 * pred_keypoints_3d[:, 1:-1] + pred_keypoints_3d[:, 2:]
        accel_gt = gt_keypoints_3d[:, :-2] - 2 * gt_keypoints_3d[:, 1:-1] + gt_keypoints_3d[:, 2:]
        
        loss = torch.mean(
            weight * (conf[:, :-2] * torch.norm(accel_pred - accel_gt[..., :3], dim=-1)
        ), dim=1).mean() * conf.mean()
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(gt_keypoints_3d.device)[0]
    return loss


def vertices_loss(
        pred_verts,
        gt_verts,
        mask,
        criterion,
):
    
    if mask.sum() > 0:
        # Align
        pred_verts = pred_verts.view_as(gt_verts)
        pred_verts = pred_verts - pred_verts.mean(-2, True)
        gt_verts = gt_verts - gt_verts.mean(-2, True)
        
        loss = torch.mean(
            (torch.norm(pred_verts - gt_verts, p=1, dim=-1)[mask]
        ), dim=1).mean() * mask.float().mean()
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(gt_verts.device)[0]
    return loss


def smpl_losses(
        pred_pose,
        pred_betas,
        gt_pose,
        gt_betas,
        weight,
        mask,
        criterion,
):

    if mask.any().item():
        loss_regr_pose = torch.mean(
            weight * torch.square(pred_pose - gt_pose)[mask].mean(-1)
        ) * mask.float().mean()
        loss_regr_betas = F.mse_loss(pred_betas, gt_betas, reduction='none')[mask].mean() * mask.float().mean()
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(gt_pose.device)[0]
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(gt_pose.device)[0]

    return loss_regr_pose, loss_regr_betas


def camera_loss(
    pred_cam_r,
    gt_cam_r,
    cam_angvel,
    mask,
    criterion,
    skip
):
    if mask.any() and not skip:
        # Camera pose loss in 6D representation
        loss_r = criterion(pred_cam_r, gt_cam_r)[mask].mean()
        
        # Reconstruct camera angular velocity and compute reconstruction loss
        pred_R = transforms.rotation_6d_to_matrix(pred_cam_r)
        cam_angvel_from_R = transforms.matrix_to_rotation_6d(pred_R[:, :-1] @ pred_R[:, 1:].transpose(-1, -2))
        cam_angvel_from_R = (cam_angvel_from_R - torch.tensor([[[1, 0, 0, 0, 1, 0]]]).to(cam_angvel)) * 30
        loss_a = criterion(cam_angvel, cam_angvel_from_R)[mask].mean()
        
        loss = loss_r + loss_a
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(gt_cam_r.device)[0]
        
    return loss


def sliding_loss(
    foot_position,
    contact_prob,
):
    """ Compute foot skate loss when foot is assumed to be on contact with ground
    
    foot_position: 3D foot (heel and toe) position, torch.Tensor (B, F, 4, 3)
    contact_prob: contact probability of foot (heel and toe), torch.Tensor (B, F, 4)
    """
    
    contact_mask = (contact_prob > 0.5).detach().float()
    foot_velocity = foot_position[:, 1:] - foot_position[:, :-1]
    loss = (torch.norm(foot_velocity, dim=-1) * contact_mask[:, 1:]).mean()
    return loss
