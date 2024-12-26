import argparse
from yacs.config import CfgNode as CN
import datetime

# Configuration variable
cfg = CN()

cfg.TITLE = 'default'
cfg.OUTPUT_DIR = 'results'
# cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.DEBUG = False
cfg.EVAL = False
cfg.RESUME = False
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 5
cfg.SEED_VALUE = -1
cfg.SUMMARY_ITER = 50
cfg.MODEL_CONFIG = ''
cfg.FLIP_EVAL = False
CST = datetime.timezone(datetime.timedelta(hours=8))
save_folder = 'exp_' + str(datetime.datetime.now(tz=CST))[5:-16]
save_folder = save_folder.replace(" ", "_")
save_folder = save_folder.replace(":", "_")
cfg.EXP_NAME = save_folder

cfg.TRAIN = CN()
cfg.TRAIN.STAGE = 'stage1'
cfg.TRAIN.DATASET_EVAL = '3dpw'
cfg.TRAIN.CHECKPOINT = ''
cfg.TRAIN.POSE_CHECKPOINT = ''
cfg.TRAIN.BATCH_SIZE = 64
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 999
cfg.TRAIN.OPTIM = 'Adam'
cfg.TRAIN.LR = 3e-4
cfg.TRAIN.LR_FINETUNE = 5e-5
cfg.TRAIN.LR_PATIENCE = 5
cfg.TRAIN.LR_DECAY_RATIO = 0.1
cfg.TRAIN.WD = 0.0
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.MILESTONES = [50, 70]

cfg.DATASET = CN()
cfg.DATASET.SEQLEN = 81
cfg.DATASET.RATIO = [1.0, 0, 0, 0, 0]

cfg.MODEL = CN()
cfg.MODEL.BACKBONE = 'vit'
cfg.MODEL.VM_PATH = './dataset/marker'

cfg.LOSS = CN()
cfg.LOSS.SHAPE_LOSS_WEIGHT = 0.001
cfg.LOSS.JOINT2D_LOSS_WEIGHT = 5.
cfg.LOSS.JOINT3D_LOSS_WEIGHT = 5.
cfg.LOSS.VEL_LOSS_WEIGHT = 0.01
cfg.LOSS.ACCEL_LOSS_WEIGHT = 0.01
cfg.LOSS.MARKER3D_LOSS_WEIGHT = 0.005
cfg.LOSS.VERTS3D_LOSS_WEIGHT = 1.
cfg.LOSS.POSE_LOSS_WEIGHT = 1.
cfg.LOSS.CASCADED_LOSS_WEIGHT = 0.0
cfg.LOSS.CONTACT_LOSS_WEIGHT = 0.04
cfg.LOSS.ROOT_VEL_LOSS_WEIGHT = 0.001
cfg.LOSS.ROOT_POSE_LOSS_WEIGHT = 0.4
cfg.LOSS.SLIDING_LOSS_WEIGHT = 0.5
cfg.LOSS.CAMERA_LOSS_WEIGHT = 0.04
cfg.LOSS.LOSS_WEIGHT = 60.
cfg.LOSS.CAMERA_LOSS_SKIP_EPOCH = 5


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def get_cfg(args, test):
    """
    Define configuration.
    """
    import os
    
    cfg = get_cfg_defaults()
    if os.path.exists(args.cfg):
        cfg.merge_from_file(args.cfg)
    
    cfg.merge_from_list(args.opts)
    if test:
        cfg.merge_from_list(['EVAL', True])

    return cfg.clone()


def bool_arg(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def parse_args(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='./configs/debug.yaml', help='cfg file path')
    parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
    parser.add_argument(
        "--eval-set", type=str, default='3dpw', help="Evaluation dataset")
    parser.add_argument(
        "--eval-split", type=str, default='test', help="Evaluation data split")
    parser.add_argument('--render', default=False, type=bool_arg,
                        help='Render SMPL meshes after the evaluation')
    parser.add_argument('--save-results', default=False, type=bool_arg,
                        help='Save SMPL parameters after the evaluation')
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="Modify config options using the command-line")
    
    args = parser.parse_args()
    print(args, end='\n\n')
    cfg_file = args.cfg
    cfg = get_cfg(args, test)

    return cfg, cfg_file, args