import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
import subprocess
import glob
from collections import defaultdict
import imageio
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from termcolor import colored
from torchvision import transforms
from torch.utils.data import DataLoader

# detection module
from virtualpose.core.config import config as det_cfg
from virtualpose.core.config import update_config as det_update_config
from virtualpose.utils.transforms import inverse_affine_transform_pts_cuda
from virtualpose.utils.utils import load_backbone_validate
import virtualpose.models as det_models
import virtualpose.dataset as det_dataset

from lib.data.datasets.demo_dataset import DemoDataset
from lib.models.preproc.backbone.marker_config import cfg
from lib.utils.aug_utils import flip_img

from .backbone import simple3dmesh

def output2original_scale(meta, output, vis=False):
    img_paths, trans_batch = meta['image'], meta['trans']
    bbox_batch, depth_batch, roots_2d = output['bboxes'], output['depths'], output['roots_2d']

    scale = torch.tensor((det_cfg.NETWORK.IMAGE_SIZE[0] / det_cfg.NETWORK.HEATMAP_SIZE[0], \
                        det_cfg.NETWORK.IMAGE_SIZE[1] / det_cfg.NETWORK.HEATMAP_SIZE[1]), \
                        device=bbox_batch.device, dtype=torch.float32)
    
    det_results = []
    valid_frame_idx = []
    max_person = 0
    for i, img_path in enumerate(img_paths):
        if vis:
            img = cv2.imread(img_path)
        
        frame_id = int(img_path.split('/')[-1][:-4])-1
        trans = trans_batch[i].to(bbox_batch[i].device).float()

        n_person = 0
        for bbox, depth, root_2d in zip(bbox_batch[i], depth_batch[i], roots_2d[i]):
            if torch.all(bbox == 0):
                break
            bbox = (bbox.view(-1, 2) * scale[None, [1, 0]]).view(-1)
            root_2d *= scale[[1, 0]]
            bbox_origin = inverse_affine_transform_pts_cuda(bbox.view(-1, 2), trans).reshape(-1)
            roots_2d_origin = inverse_affine_transform_pts_cuda(root_2d.view(-1, 2), trans).reshape(-1)

            # frame_id, x_min, y_min, x_max, y_max, pixel_root_x, pixel_root_y, depth
            det_results.append([frame_id] + bbox_origin.cpu().numpy().tolist() + roots_2d_origin.cpu().numpy().tolist() + depth.cpu().numpy().tolist())

            if vis:
                img = cv2.putText(img, '%.2fmm'%depth, (int(bbox_origin[0]), int(bbox_origin[1] - 5)),\
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                img = cv2.rectangle(img, (int(bbox_origin[0]), int(bbox_origin[1])), (int(bbox_origin[2]), int(bbox_origin[3])), \
                    (255, 0, 0), 1)
                img = cv2.circle(img, (int(roots_2d_origin[0]), int(roots_2d_origin[1])), 5, (0, 0, 255), -1)
            n_person += 1

        if vis:
            cv2.imwrite(f'{cfg.vis_dir}/origin_det_{i}.jpg', img)
        max_person = max(n_person, max_person)
        if n_person:
            valid_frame_idx.append(frame_id)
    return det_results, max_person, valid_frame_idx

def detect_all_persons(img_dir):
    # prepare detection model
    virtualpose_name = 'third-party/VirtualPose' 
    det_update_config(f'{virtualpose_name}/configs/images/images_inference.yaml')

    det_model = eval('det_models.multi_person_posenet.get_multi_person_pose_net')(det_cfg, is_train=False)
    with torch.no_grad():
        det_model = torch.nn.DataParallel(det_model.cuda())

    pretrained_file = osp.join(f'{virtualpose_name}', det_cfg.NETWORK.PRETRAINED)
    state_dict = torch.load(pretrained_file)
    new_state_dict = {k:v for k, v in state_dict.items() if 'backbone.pose_branch.' not in k}
    det_model.module.load_state_dict(new_state_dict, strict = False)
    pretrained_file = osp.join(f'{virtualpose_name}', det_cfg.NETWORK.PRETRAINED_BACKBONE)
    det_model = load_backbone_validate(det_model, pretrained_file)

    # prepare detection dataset
    infer_dataset = det_dataset.images(
        det_cfg, img_dir, focal_length=1700, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
        ]))

    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=det_cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
    
    det_model.eval()

    max_person = 0
    detection_all = []
    valid_frame_idx_all = []
    with torch.no_grad():
        for _, (inputs, targets_2d, weights_2d, targets_3d, meta, input_AGR) in enumerate(tqdm(infer_loader, dynamic_ncols=True)):
            _, _, output, _, _ = det_model(views=inputs, meta=meta, targets_2d=targets_2d,
                                                            weights_2d=weights_2d, targets_3d=targets_3d, input_AGR=input_AGR)
            det_results, n_person, valid_frame_idx = output2original_scale(meta, output)
            detection_all += det_results
            valid_frame_idx_all += valid_frame_idx
            max_person = max(n_person, max_person)

    # list to array
    detection_all = np.array(detection_all)  # (N*T, 8)
    return detection_all, max_person, valid_frame_idx_all

def load_checkpoint(load_path, master=True):
    try:
        print(f"Fetch model weight from {load_path}")
        checkpoint = torch.load(load_path, map_location='cuda')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint %s exists!\n"%(load_path), e)

class Simple3DMeshInferencer:
    def __init__(self, args, load_path='', writer=None, img_path_list=[], detection_all=[], max_person=-1, fps=-1):
        self.args = args
        # prepare inference dataset
        demo_dataset = DemoDataset(img_path_list, detection_all)
        self.demo_dataset = demo_dataset
        self.detection_all = detection_all
        self.img_path_list = img_path_list
        self.max_person = max_person
        self.fps = fps
        self.demo_dataloader = DataLoader(self.demo_dataset, batch_size=min(args.batch_size, len(detection_all)), num_workers=8)

        # prepare inference model
        self.model = simple3dmesh.get_model(demo_dataset.flip_pairs)

        # load weight 
        if load_path != '':
            print('==> Loading checkpoint')
            checkpoint = load_checkpoint(load_path, master=True)

            if 'model_state_dict' in checkpoint.keys():
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            try:
                if cfg.model.name == 'simple3dmesh' and cfg.model.simple3dmesh.noise_reduce:
                    self.model.simple3dmesh.load_state_dict(state_dict, strict=True)
                else:
                    self.model.load_state_dict(state_dict, strict=False)
                print(colored(f'Successfully load checkpoint from {load_path}.', 'green'))
            except:
                print(colored(f'Failed to load checkpoint in {load_path}', 'red'))
        if self.model:
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model)

        # initialize others
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.J_regressor = torch.Tensor(self.demo_dataset.joint_regressor).cuda()
        self.vm_B = torch.Tensor(self.demo_dataset.vm_B).cuda()
        self.selected_indices = self.demo_dataset.selected_indices

    def infer(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        results = defaultdict(list)
        with torch.no_grad():
            for i, meta in enumerate(tqdm(self.demo_dataloader, dynamic_ncols=True)):
                for k, _ in meta.items():
                    meta[k] = meta[k].cuda()

                imgs = meta['img'].cuda()
                inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
                pose_root = meta['root_cam'].cuda()
                depth_factor = meta['depth_factor'].cuda()

                pred_xyz_jts_ret, _, _, confidence, _, _, _ = self.model(imgs, inv_trans, intrinsic_param, pose_root, depth_factor, flip_item=None, flip_mask=None)
                pred_xyz_jts_ret_conf = torch.cat([pred_xyz_jts_ret, confidence.unsqueeze(-1)], dim=-1)
                
                # flip_test
                if isinstance(imgs, list):
                    imgs_flip = [flip_img(img.clone()) for img in imgs]
                else:
                    imgs_flip = flip_img(imgs.clone())
                    
                pred_xyz_jts_ret_flip, _, _, confidence_flip, _, _, _ = self.model(imgs_flip, inv_trans, intrinsic_param, pose_root, depth_factor, flip_item=None, flip_mask=None)
                pred_xyz_jts_ret_flip_conf = torch.cat([pred_xyz_jts_ret_flip, confidence_flip.unsqueeze(-1)], dim=-1)
                
                results['frame_id'].append(meta['img_idx'].detach().cpu().numpy())
                results['bbox'].append(meta['bbox'].detach().cpu().numpy())
                results['marker'].append(pred_xyz_jts_ret_conf.detach().cpu().numpy())
                results['flipped_marker'].append(pred_xyz_jts_ret_flip_conf.detach().cpu().numpy())

        for term in results.keys():
            results[term] = np.concatenate(results[term])
        return results