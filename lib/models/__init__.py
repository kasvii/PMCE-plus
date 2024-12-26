import os, sys
import yaml
import torch
from loguru import logger

from configs import constants as _C
from .smpl import SMPL


def build_body_model(device, batch_size=1, gender='neutral', **kwargs):
    sys.stdout = open(os.devnull, 'w')
    body_model = SMPL(
        model_path=_C.BMODEL.FLDR,
        gender=gender,
        batch_size=batch_size,
        create_transl=False).to(device)
    sys.stdout = sys.__stdout__
    return body_model


def build_network(cfg, smpl):
    if cfg.TRAIN.STAGE == 'stage2':
    # from .wham import Network
        from .PMCE import Network
    else:
        from .PoseEst import Network
    
    with open(cfg.MODEL_CONFIG, 'r') as f:
        model_config = yaml.safe_load(f)
    model_config.update({'d_feat': _C.IMG_FEAT_DIM[cfg.MODEL.BACKBONE]})
    
    network = Network(smpl, cfg.DATASET.SEQLEN, cfg.TRAIN.POSE_CHECKPOINT, **model_config).to(cfg.DEVICE)
    
    # Load Checkpoint
    if os.path.isfile(cfg.TRAIN.CHECKPOINT):
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT)
        ignore_keys = ['smpl.body_pose', 'smpl.betas', 'smpl.global_orient', 
                       'smpl.J_regressor_extra', 'smpl.J_regressor_eval',
                       'pose_lifter.smpl.body_pose', 'pose_lifter.smpl.betas', 'pose_lifter.smpl.global_orient', 
                       'pose_lifter.smpl.J_regressor_extra', 'pose_lifter.smpl.J_regressor_eval',
                       'mesh_regressor.smpl.body_pose', 'mesh_regressor.smpl.betas', 'mesh_regressor.smpl.global_orient', 'mesh_regressor.smpl.J_regressor_extra', 'mesh_regressor.smpl.J_regressor_eval', 
                       'module.smpl.body_pose', 'module.smpl.betas', 'module.smpl.global_orient', 
                       'module.smpl.J_regressor_extra', 'module.smpl.J_regressor_eval',
                       'module.pose_lifter.smpl.body_pose', 'module.pose_lifter.smpl.betas', 'module.pose_lifter.smpl.global_orient', 
                       'module.pose_lifter.smpl.J_regressor_extra', 'module.pose_lifter.smpl.J_regressor_eval',
                       'module.mesh_regressor.smpl.body_pose', 'module.mesh_regressor.smpl.betas', 'module.mesh_regressor.smpl.global_orient', 'module.mesh_regressor.smpl.J_regressor_extra', 'module.mesh_regressor.smpl.J_regressor_eval']
        
        model_state_dict = {k[7:] if k.startswith('module') else k: v for k, v in checkpoint['model'].items() if k not in ignore_keys}
        network.load_state_dict(model_state_dict, strict=False)
        logger.info(f"=> loaded checkpoint '{cfg.TRAIN.CHECKPOINT}' ")
    else:
        logger.info(f"=> Warning! no checkpoint found at '{cfg.TRAIN.CHECKPOINT}'.")
        
    return network