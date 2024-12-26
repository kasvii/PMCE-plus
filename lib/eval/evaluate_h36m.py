import os
import time
import os.path as osp
from glob import glob
from collections import defaultdict

import torch
import imageio.v2 as imageio
import numpy as np
from smplx import SMPL
from loguru import logger
from progress.bar import Bar

from configs import constants as _C
from configs.config import parse_args
from lib.data.dataloader import setup_eval_dataloader
from lib.models import build_network, build_body_model
from lib.eval.eval_utils import (
    compute_error_accel,
    batch_align_by_pelvis,
    batch_align_by_pelvis_coco,
    batch_align_by_pelvis_h36m,
    batch_compute_similarity_transform_torch,
)
from lib.utils import transforms
from lib.utils.utils import prepare_output_dir
from lib.utils.utils import prepare_batch
from lib.utils.imutils import avg_preds
from lib.utils.kp_utils import root_centering

try:
    from lib.vis.renderer import Renderer
    _render = True
except:
    print("PyTorch3D is not properly installed! Cannot render the SMPL mesh")
    _render = False


m2mm = 1e3
@torch.no_grad()
def main(cfg, args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Dataloaders ========= #
    eval_loader = setup_eval_dataloader(cfg, 'h36m', 'test', cfg.MODEL.BACKBONE)
    logger.info(f'Dataset loaded')
    
    # ========= Load network ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Build SMPL models with each gender
    smpl = {k: SMPL(_C.BMODEL.FLDR, gender=k).to(cfg.DEVICE) for k in ['male', 'female', 'neutral']}
    
    # Load vertices -> joints regression matrix to evaluate
    J_regressor_eval = torch.from_numpy(
        np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
    )[_C.KEYPOINTS.H36M_TO_J14, :].unsqueeze(0).float().to(cfg.DEVICE)
    if cfg.TRAIN.STAGE == 'stage1':
        pelvis_idxs = [11, 12] # coco
    else:
        pelvis_idxs = [2, 3]   # common
    
    accumulator = defaultdict(list)
    bar = Bar('Inference', fill='#', max=len(eval_loader))
    with torch.no_grad():
        for i in range(len(eval_loader)):
            # Original batch
            batch = eval_loader.dataset.load_data(i, False)
            x, marker, inits, features, kwargs, gt = prepare_batch(batch, cfg.DEVICE) # , cfg.TRAIN.STAGE=='stage2'
            
            if cfg.FLIP_EVAL and cfg.TRAIN.STAGE=='stage2':
                flipped_batch = eval_loader.dataset.load_data(i, True)
                f_x, f_marker, f_inits, f_features, f_kwargs, _ = prepare_batch(flipped_batch, cfg.DEVICE) # , cfg.TRAIN.STAGE=='stage2'
            
                # Forward pass with flipped input
                flipped_pred = network(f_x, f_marker, f_features, **f_kwargs)
            
            # Forward pass with normal input
            if cfg.TRAIN.STAGE=='stage2':
                pred = network(x, marker, features, **kwargs) # 81 frame
            else:
                pred = network(x, features, **kwargs)
            
            if cfg.FLIP_EVAL and cfg.TRAIN.STAGE=='stage2':
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
            
            if cfg.TRAIN.STAGE == 'stage1':
                pred_j3d = pred['kp3d_nn'].squeeze(0).cpu()
            else:
                # <======= Build predicted SMPL
                pred_output = smpl['neutral'](body_pose=pred['poses_body'], 
                                            global_orient=pred['poses_root_cam'], 
                                            betas=pred['betas'].squeeze(0), 
                                            pose2rot=False)
                pred_verts = pred_output.vertices.cpu()
                pred_j3d = torch.matmul(J_regressor_eval, pred_output.vertices).cpu()
                # =======>
            
            # <======= Build groundtruth SMPL
            if cfg.TRAIN.STAGE == 'stage1':
                target_j3d = gt['kp3d']
                target_j3d = root_centering(target_j3d)
                target_j3d = target_j3d[:, -pred_j3d.shape[-2]:, :].cpu()
            else:
                target_j3d = gt['kp3d'].squeeze(0)
                target_j3d = root_centering(target_j3d)
                target_j3d = target_j3d[:, -pred_j3d.shape[-2]:, :].cpu()
            # =======>
            
            # <======= Compute performance of the current sequence
            if cfg.TRAIN.STAGE == 'stage1':
                pred_j3d, target_j3d = batch_align_by_pelvis_coco(
                        pred_j3d, target_j3d, pelvis_idxs
                )
            else:
                pred_j3d, target_j3d = batch_align_by_pelvis_h36m(
                    [pred_j3d, target_j3d], pelvis_idxs
                )
            S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
            pa_mpjpe = torch.sqrt(((S1_hat - target_j3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * m2mm
            mpjpe = torch.sqrt(((pred_j3d - target_j3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * m2mm
            accel = np.zeros((len(pred_j3d,)))
            accel[1:-1] = compute_error_accel(joints_pred=pred_j3d, joints_gt=target_j3d)
            accel = accel * (25 ** 2)       # per frame^s to per s^2
            
            summary_string = f'{batch["vid"]} | PA-MPJPE: {pa_mpjpe.mean():.1f}   MPJPE: {mpjpe.mean():.1f}  ACCEL: {accel.mean():.1f}'
            bar.suffix = summary_string
            bar.next()
            
            # <======= Accumulate the results over entire sequences
            accumulator['pa_mpjpe'].append(pa_mpjpe)
            accumulator['mpjpe'].append(mpjpe)
            accumulator['accel'].append(accel)
            # =======>
            
            # <======= (Optional) Render the prediction
            if not (_render and args.render):
                # Skip if PyTorch3D is not installed or rendering argument is not parsed.
                continue
            
            # Save path
            viz_pth = osp.join('output', 'visualization_h36m')
            os.makedirs(viz_pth, exist_ok=True)
            
            # Build Renderer
            width, height = batch['cam_intrinsics'][0][0, :2, -1].numpy() * 2
            focal_length = batch['cam_intrinsics'][0][0, 0, 0].item()
            renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl['neutral'].faces)
            
            # Get images and writer
            frame_list = batch['frame_id'][0].numpy()
            imname_list = sorted(glob(osp.join(_C.PATHS.THREEDPW_PTH, 'imageFiles', batch['vid'][:-2], '*.jpg')))
            writer = imageio.get_writer(osp.join(viz_pth, batch['vid'] + '.mp4'), 
                                        mode='I', format='FFMPEG', fps=30, macro_block_size=1)
            
            # Skip the invalid frames
            for i, frame in enumerate(frame_list):
                image = imageio.imread(imname_list[frame])
                vertices = pred['verts_cam'][i] + pred['trans_cam'][[i]]
                image = renderer.render_mesh(vertices, image)
                writer.append_data(image)
            writer.close()
            # =======>
            
    for k, v in accumulator.items():
        accumulator[k] = np.concatenate(v).mean()

    print('')
    log_str = 'Evaluation on H36M, '
    log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in accumulator.items()])
    logger.info(log_str)
    
if __name__ == '__main__':
    cfg, cfg_file, args = parse_args(test=True)
    cfg = prepare_output_dir(cfg, cfg_file)
    
    main(cfg, args)
