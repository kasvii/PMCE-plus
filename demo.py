import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
from lib.models.preproc.marker_extractor import Simple3DMeshInferencer, detect_all_persons
from lib.models.preproc.backbone.marker_config import cfg as marker_cfg

def run(cfg,
        video,
        output_pth,
        network,
        calib=None,
        save_pkl=False,
        visualize=False):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    image_dir = osp.join(output_pth, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    img_cnt = 0
    
    # Preprocess
    with torch.no_grad():
        # if not (osp.exists(osp.join(output_pth, 'tracking_results.pth'))):
        if True:
            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            
            print('Preprocess: 2D detection ...')
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break
                
                # 2D detection and tracking
                detector.track(img, fps, length)
                cv2.imwrite(osp.join(image_dir, f'{img_cnt:06d}.png'), img)
                img_cnt += 1
                
            # get all image paths
            img_path_list = glob(osp.join(image_dir, '*.png'))
            detection_all, max_person, valid_frame_idx_all = detect_all_persons(image_dir)
            # ############ prepare virtual marker model ############
            # create the model instance and load checkpoint
            load_path_test = marker_cfg.test.weight_path
            assert marker_cfg.model.name == 'simple3dmesh', 'check marker_cfg of the model name'
            marker_extractor = Simple3DMeshInferencer(args, load_path=load_path_test, img_path_list=img_path_list, detection_all=detection_all, max_person=max_person, fps=fps)
            # ############ inference virtual marker model ############
            print(f"===> Start marker extracting...")
            marker_results = marker_extractor.infer(epoch=0)
            
            tracking_results = detector.process(fps, marker_results)
            
            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            
            # Save the processed data
            joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
            logger.info(f'Save processed data at {output_pth}')
        
        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
            logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
    
    # run pmce++
    results = defaultdict(dict)
    
    for _id in range(len(tracking_results.keys())):
        # Build dataset
        dataset = CustomDataset(cfg, tracking_results[_id], width, height, fps)
        results[_id]['pose'], results[_id]['trans'], results[_id]['betas'], results[_id]['verts'], results[_id]['frame_ids'] = [], [], [], [], []
    
        n_seqs = len(dataset)
        for seq in range(n_seqs):
            with torch.no_grad():
                if cfg.FLIP_EVAL:
                    # Forward pass with flipped input
                    flipped_batch = dataset.load_data(seq, True)
                    x, marker, features, mask, frame_id, kwargs = flipped_batch
                    flipped_pred = network(x, marker, features, mask=mask, **kwargs)
                    
                    # Forward pass with normal input
                    batch = dataset.load_data(seq)
                    x, marker, features, mask, frame_id, kwargs = batch
                    pred = network(x, marker, features, mask=mask, **kwargs)
                    
                    # Merge two predictions
                    flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                    pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                    flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                    avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                    avg_pose = avg_pose.reshape(-1, 144)
                    
                    # Refine trajectory with merged prediction
                    network.mesh_regressor.pred_pose = avg_pose.view_as(network.mesh_regressor.pred_pose)
                    network.mesh_regressor.pred_shape = avg_shape.view_as(network.mesh_regressor.pred_shape)
                    pred = network.mesh_regressor.forward_smpl(**kwargs)
                
                else:
                    # data
                    batch = dataset.load_data(seq)
                    x, marker, features, mask, cam_angvel, frame_id, kwargs = batch
                    
                    # inference
                    pred = network(x, marker, features, mask=mask, **kwargs)
            
            # ========= Store results ========= #
            pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
            pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
            pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
            pred_trans = (pred['trans_cam'] - network.mesh_regressor.output.offset).cpu().numpy()
            
            results[_id]['pose'].append(pred_pose)
            results[_id]['trans'].append(pred_trans)
            results[_id]['betas'].append(pred['betas'].cpu().squeeze(0).numpy())
            results[_id]['verts'].append((pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy())
            results[_id]['frame_ids'].append(frame_id)
        
        if len(results[_id]['pose']) != 0:
            for key in results[_id].keys():
                results[_id][key] = np.concatenate(results[_id][key], axis=0)
    
    if save_pkl:
        joblib.dump(results, osp.join(output_pth, "pmce_plus_output.pkl"))
     
    # Visualize
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        with torch.no_grad():
            run_vis_on_demo(cfg, video, results, output_pth, network.mesh_regressor.smpl)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, 
                        default='examples/demo.mp4', 
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='output/demo', 
                        help='output folder to write results')
    
    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')
    
    parser.add_argument('--save_pkl', action='store_true',
                        help='Save output as pkl file')
    
    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')
    
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size')
    
    parser.add_argument('--marker_path', type=str, default=None, 
                        help='Camera calibration file path')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Load network ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Output folder
    sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(cfg, 
        args.video, 
        output_pth, 
        network, 
        args.calib, 
        save_pkl=args.save_pkl,
        visualize=args.visualize)
        
    print()
    logger.info('Done !')