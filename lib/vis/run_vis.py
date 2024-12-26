import os
import os.path as osp

import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar
from lib.vis.renderer import Renderer

def run_vis_on_demo(cfg, video, results, output_pth, smpl, vis_global=False):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # create renderer with cliff focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
        osp.join(output_pth, f'output_{video.split("/")[-1]}'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1
    )
    print('Rendering results ...')
    rendered_img_output_pth = osp.join(output_pth, 'rendered_img')
    if not osp.exists(rendered_img_output_pth):
        os.makedirs(rendered_img_output_pth)
    
    frame_i = 0
    # run rendering
    while (cap.isOpened()):
        flag, org_img = cap.read()
        if not flag: break
        img = org_img[..., ::-1].copy()
        
        # render onto the input video
        renderer.create_camera(default_R, default_T)
        for _id, val in results.items():
            # render onto the image
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            img = renderer.render_mesh(torch.from_numpy(val['verts'][frame_i2]).to(cfg.DEVICE), img)
            cv2.imwrite(os.path.join(rendered_img_output_pth, f"{str(frame_i).zfill(4)}.png"), img[..., ::-1])
            
        writer.append_data(img)
        frame_i += 1
    writer.close()