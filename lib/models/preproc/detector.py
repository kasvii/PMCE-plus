from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal
from progress.bar import Bar

from ultralytics import YOLO
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    get_track_id,
    vis_pose_result,
)

from lib.models.preproc.backbone.utils import draw_bbox

ROOT_DIR = osp.abspath(f"{__file__}/../../../../")
VIT_DIR = osp.join(ROOT_DIR, "third-party/ViTPose")

VIS_THRESH = 0.3
BBOX_CONF = 0.5
TRACKING_THR = 0.1
MINIMUM_FRMAES = 30
MINIMUM_JOINTS = 6

class DetectionModel(object):
    def __init__(self, device):
        
        # ViTPose
        pose_model_cfg = osp.join(VIT_DIR, 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py')
        pose_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'vitpose-h-multi-coco.pth')
        self.pose_model = init_pose_model(pose_model_cfg, pose_model_ckpt, device=device.lower())
        
        # YOLO
        bbox_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'yolov8x.pt')
        self.bbox_model = YOLO(bbox_model_ckpt)
        
        self.device = device
        self.initialize_tracking()
        
    def initialize_tracking(self, ):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': [],
            'marker': [],
            'flipped_marker': []
        }
        
    def xyxy_to_cxcys(self, bbox, s_factor=1.05):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
        return np.array([[cx, cy, scale]])
        
    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        X = self.tracking_results['keypoints'].copy()
        mask = X[..., -1] > VIS_THRESH

        bbox = np.zeros((len(X), 3))
        for i, (kp, m) in enumerate(zip(X, mask)):
            bb = [kp[m, 0].min(), kp[m, 1].min(),
                  kp[m, 0].max(), kp[m, 1].max()]
            cx, cy = [(bb[2]+bb[0])/2, (bb[3]+bb[1])/2]
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            s = np.stack((bb_w, bb_h)).max()
            bb = np.array((cx, cy, s))
            bbox[i] = bb
        
        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results['bbox'] = bbox
    
    def track(self, img, fps, length):
        
        # bbox detection
        bboxes = self.bbox_model.predict(
            img, device=self.device, classes=0, conf=BBOX_CONF, save=False, verbose=False
        )[0].boxes.xyxy.detach().cpu().numpy()
        bboxes = [{'bbox': bbox} for bbox in bboxes]
        
        # keypoints detection
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results=bboxes,
            format='xyxy',
            return_heatmap=False,
            outputs=None)
        
        # person identification
        pose_results, self.next_id = get_track_id(
            pose_results,
            self.pose_results_last,
            self.next_id,
            use_oks=False,
            tracking_thr=TRACKING_THR,
            use_one_euro=True,
            fps=fps)
        
        for pose_result in pose_results:
            n_valid = (pose_result['keypoints'][:, -1] > VIS_THRESH).sum()
            if n_valid < MINIMUM_JOINTS: continue
            
            _id = pose_result['track_id']
            xyxy = pose_result['bbox']
            bbox = xyxy
            
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            self.tracking_results['keypoints'].append(pose_result['keypoints'])
        
        self.frame_id += 1
        self.pose_results_last = pose_results
        
    def bbox_iou(self, bbox1, bbox2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, x2, y2 = bbox1
        x1_p, y1_p, x2_p, y2_p = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
        
        # Calculate the (x, y)-coordinates of the intersection rectangle
        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)
        
        # Calculate the area of intersection rectangle
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        
        # Calculate the area of both bounding boxes
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x2_p - x1_p) * (y2_p - y1_p)
        
        # Calculate the IoU
        iou = inter_area / float(bbox1_area + bbox2_area - inter_area)
        
        return iou
    
    def merge_marker(self, marker_results, iou_threshold=0.05):
        frame_num = max(np.max(self.tracking_results['frame_id']), np.max(marker_results['frame_id']))
        for frame_id in range(frame_num):
            tracking_idxs = np.where(self.tracking_results['frame_id'] == frame_id)[0]
            marker_idxs = np.where(marker_results['frame_id'] == frame_id)[0]
            
            for marker_idx in marker_idxs:
                marker_bbox_xywh = marker_results['bbox'][marker_idx]
                best_iou = 0
                best_idx = -1
                for tracking_idx in tracking_idxs:
                    tracking_bbox_xyxy = self.tracking_results['bbox'][tracking_idx]
                    iou = self.bbox_iou(tracking_bbox_xyxy, marker_bbox_xywh)
                    
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_idx = tracking_idx
                
                if best_idx != -1:
                    self.tracking_results['marker'][best_idx] = np.array(marker_results['marker'][marker_idx])
                    self.tracking_results['flipped_marker'][best_idx] = np.array(marker_results['flipped_marker'][marker_idx])
        
    def process(self, fps, marker_results):
        for key in ['id', 'frame_id', 'keypoints']:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        self.tracking_results['marker'] = np.zeros((len(self.tracking_results['id']), 81, 4))
        self.tracking_results['flipped_marker'] = np.zeros((len(self.tracking_results['id']), 81, 4))
        
        self.merge_marker(marker_results)
        self.compute_bboxes_from_keypoints()
            
        output = defaultdict(lambda: defaultdict(list))
        ids = np.unique(self.tracking_results['id'])
        for _id in ids:
            idxs = np.where(self.tracking_results['id'] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]
        
        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                del output[_id]
                continue
            
            kernel = int(int(fps/2) / 2) * 2 + 1
            smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
            output[_id]['bbox'] = smoothed_bbox
        
        return output