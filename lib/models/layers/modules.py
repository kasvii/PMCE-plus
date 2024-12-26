from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
from torch import nn
from functools import partial
from configs import constants as _C
from .utils import rollout_global_motion, load_checkpoint
from lib.utils.transforms import axis_angle_to_matrix
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp, Attention

class Integrator(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel=1024):
        super().__init__()
        
        self.layer1 = nn.Linear(in_channel, hid_channel)
        self.relu1 = nn.GELU()
        self.dr1 = nn.Dropout(0.1)
        
        self.layer2 = nn.Linear(hid_channel, hid_channel)
        self.relu2 = nn.GELU()
        self.dr2 = nn.Dropout(0.1)
        
        self.layer3 = nn.Linear(hid_channel, out_channel)
        
        
    def forward(self, x, feat):
        res = x
        mask = (feat != 0).all(dim=-1).all(dim=-1)
        
        out = torch.cat((x, feat), dim=-1)
        out = self.layer1(out)
        out = self.relu1(out)
        out = self.dr1(out)
        
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.dr2(out)
        
        out = self.layer3(out)
        out[mask] = out[mask] + res[mask]
        
        return out

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class MotionEncoder(nn.Module):
    def __init__(self, num_frames=16, num_joints=17, embed_dim=256, img_dim=2048, depth=5, pretrained=False,      
                 num_heads=8, mlp_ratio=0.5, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()

        in_dim = 2
        out_dim = 3    
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.joint_embed = nn.Linear(in_dim, embed_dim)
        self.imgfeat_embed = nn.Linear(img_dim, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_joints = num_joints

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth

        self.SpatialBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TemporalBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_s = norm_layer(embed_dim)
        self.norm_t = norm_layer(embed_dim)

        self.regression = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        self.fusion = torch.nn.Conv2d(in_channels=num_frames, out_channels=num_frames, kernel_size=1)
        self.linear = nn.Linear(num_joints * out_dim, embed_dim)
        
        if pretrained:
            self._load_pretrained_model()

    def _load_pretrained_model(self):
        print("Loading pretrained posenet...")
        checkpoint = load_checkpoint(load_dir=cfg.MODEL.posenet_path, pick_best=True)
        self.load_state_dict(checkpoint['model_state_dict'])
        
    def SpaTemHead(self, x, img=None):
        b, t, j, c = x.shape
        x = rearrange(x, 'b t j c  -> (b j) t c')
        x = self.joint_embed(x)
        x += self.temporal_pos_embed
        x = self.pos_drop(x)
        
        x = rearrange(x, '(b j) t c -> (b t) j c', j=j)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        # add img feat
        if img is not None:
            x = x + rearrange(self.imgfeat_embed(img), 'b t c  -> (b t) 1 c')
        
        return x

    def forward(self, x, img):
        b, t = x.shape[:2]
        # bbox = x[..., -3:] # b, t, 3
        x = x.reshape(b, t, self.num_joints, -1) # b, t, j, 2
        b, t, j, c = x.shape
        # [b t j c]
        x = self.SpaTemHead(x, img) # bj t c
        
        for i in range(self.depth):
            SpaAtten = self.SpatialBlocks[i]
            TemAtten = self.TemporalBlocks[i]
            x = rearrange(x, '(b t) j c -> (b j) t c', t=t)
            x = TemAtten(x)
            x = self.norm_t(x)
            x = rearrange(x, '(b j) t c -> (b t) j c', j=j)
            x = SpaAtten(x)
            x = self.norm_s(x)

        x_feat = rearrange(x, '(b t) j c -> b t j c', t=t)
        x_kp3d = self.fusion(x_feat)
        x_kp3d = self.regression(x_kp3d)
        # x_feat = self.linear(x.detach().reshape(b, t, -1))
        x_feat = torch.cat((x_feat, x_kp3d), dim=-1)

        return x_kp3d, x_feat
    
class AdaLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(AdaLayerNorm, self).__init__()
        self.mlp_gamma = nn.Linear(2048, num_features)
        self.mlp_beta = nn.Linear(2048, num_features)
        self.eps = eps

    def forward(self, x, img_feat):
        size = x.size()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        gamma = self.mlp_gamma(img_feat).view(size[0], 1, -1).expand(size)
        beta = self.mlp_beta(img_feat).view(size[0], 1, -1).expand(size)
        return gamma * (x - mean) / (std + self.eps) + beta
    
class Block_AdaLN(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=AdaLayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, img_feat):
        x = x + self.drop_path(self.attn(self.norm1(x, img_feat)))
        x = x + self.drop_path(self.mlp(self.norm2(x, img_feat)))
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, v_dim, kv_num, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.kv_num = kv_num
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(v_dim, v_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):

        B, N, C = xq.shape
        v_dim = xv.shape[-1]
        q = self.wq(xq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,N1,C] -> [B,N1,H,(C/H)] -> [B,H,N1,(C/H)]
        k = self.wk(xk).reshape(B, self.kv_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,N2,C] -> [B,N2,H,(C/H)] -> [B,H,N2,(C/H)]
        v = self.wv(xv).reshape(B, self.kv_num, self.num_heads, v_dim // self.num_heads).permute(0, 2, 1, 3)  # [B,N2,C] -> [B,N2,H,(C/H)] -> [B,H,N2,(C/H)]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N1,(C/H)] @ [B,H,(C/H),N2] -> [B,H,N1,N2]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, v_dim)   # [B,H,N1,N2] @ [B,H,N2,(C/H)] -> [B,H,N1,(C/H)] -> [B,N1,H,(C/H)] -> [B,N1,C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, kv_num, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.2, 
                 attn_drop=0.2, drop_path=0.2, act_layer=nn.GELU, norm_layer=AdaLayerNorm, has_mlp=True):
        super().__init__()
        self.normq = norm_layer(q_dim)
        self.normk = norm_layer(k_dim)
        self.normv = norm_layer(v_dim)
        self.kv_num = kv_num
        self.attn = CrossAttention(q_dim, v_dim, kv_num = kv_num, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(q_dim)
            mlp_hidden_dim = int(q_dim * mlp_ratio)
            self.mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xq, xk, xv, img_feat):
        xq = xq + self.drop_path(self.attn(self.normq(xq, img_feat), self.normk(xk, img_feat), self.normv(xv, img_feat)))
        if self.has_mlp:
            xq = xq + self.drop_path(self.mlp(self.norm2(xq, img_feat)))

        return xq
    
class CoevoBlock(nn.Module):
    def __init__(self, num_joint, num_vertx, joint_dim=64, vertx_dim=64, num_frames=None):
        super(CoevoBlock, self).__init__()

        self.num_joint = num_joint
        self.num_vertx = num_vertx
        joint_num_heads = 8
        vertx_num_heads = 2
        mlp_ratio = 4.
        drop = 0.
        attn_drop = 0.
        drop_path = 0.2
        qkv_bias = True
        qk_scale = None

        self.j_Q_embed = nn.Parameter(torch.randn(1, self.num_joint, joint_dim))
        self.v_Q_embed = nn.Parameter(torch.randn(1, self.num_vertx, vertx_dim))

        self.proj_v2j_dim = nn.Linear(vertx_dim, joint_dim)
        self.proj_j2v_dim = nn.Linear(joint_dim, vertx_dim)
        self.v2j_K_embed = nn.Parameter(torch.randn(1, self.num_vertx, joint_dim))
        self.j2v_K_embed = nn.Parameter(torch.randn(1, self.num_joint, vertx_dim))

        self.joint_SA_FFN = Block_AdaLN(dim=joint_dim, num_heads=joint_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                  drop=drop, attn_drop=attn_drop, drop_path=drop_path)
        self.vertx_SA_FFN = Block_AdaLN(dim=vertx_dim, num_heads=vertx_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                  drop=drop, attn_drop=attn_drop, drop_path=drop_path)

        self.joint_CA_FFN = CrossAttentionBlock(q_dim=joint_dim, k_dim=joint_dim, v_dim=vertx_dim, kv_num = num_vertx, num_heads=joint_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                drop=drop, attn_drop=attn_drop, drop_path=drop_path, has_mlp=True)
        self.vertx_CA_FFN = CrossAttentionBlock(q_dim=vertx_dim, k_dim=vertx_dim, v_dim=joint_dim, kv_num = num_joint, num_heads=vertx_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                drop=drop, attn_drop=attn_drop, drop_path=drop_path, has_mlp=True)
        
        self.num_frames = num_frames
        self.joint_SAT_FFN = Block(dim=joint_dim, num_heads=joint_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                   qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path)
        self.vertx_SAT_FFN = Block(dim=vertx_dim, num_heads=vertx_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path)

    def forward(self, joint_feat, vertx_feat, img_feat):

        # CA + FFN
        joint_feat, vertx_feat = self.joint_CA_FFN(joint_feat + self.j_Q_embed, self.proj_v2j_dim(vertx_feat) + self.v2j_K_embed, vertx_feat, img_feat), \
                                 self.vertx_CA_FFN(vertx_feat + self.v_Q_embed, self.proj_j2v_dim(joint_feat) + self.j2v_K_embed, joint_feat, img_feat)

        # SA + FFN
        joint_feat, vertx_feat = self.joint_SA_FFN(joint_feat, img_feat), self.vertx_SA_FFN(vertx_feat, img_feat) # [B,17,64], [B,431,64]
        
        # SAT + FFN
        joint_feat, vertx_feat = rearrange(joint_feat, '(b t) j c -> (b j) t c', t=self.num_frames), rearrange(vertx_feat, '(b t) m c -> (b m) t c', t=self.num_frames)
        joint_feat, vertx_feat = self.joint_SAT_FFN(joint_feat), self.vertx_SAT_FFN(vertx_feat)
        
        joint_feat, vertx_feat = rearrange(joint_feat, '(b j) t c -> (b t) j c', j=self.num_joint), rearrange(vertx_feat, '(b m) t c -> (b t) m c', m=self.num_vertx)

        return joint_feat, vertx_feat

class Regressor(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dims):
        super().__init__()
        self.n_outs = len(out_dims)

        for i, out_dim in enumerate(out_dims):
            setattr(self, 'declayer%d'%i, nn.Linear(in_dim, out_dim))
            nn.init.xavier_uniform_(getattr(self, 'declayer%d'%i).weight, gain=0.01)

    def forward(self, x):
        preds = []
        for j in range(self.n_outs):
            out = getattr(self, 'declayer%d'%j)(x)
            preds.append(out)

        return preds

class MotionDecoder(nn.Module):
    def __init__(self, num_frames, num_joint, num_marker, in_dim=256, embed_dim=512, img_dim=2048, hidden_dim=1024, depth=2, norm_layer=None):
        super(MotionDecoder, self).__init__()
        
        self.depth = depth
        self.n_pose = 24
        self.num_joint = num_joint
        self.num_marker = num_marker
        joint_dim = embed_dim
        marker_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.gru = nn.GRU(
            input_size=img_dim,
            hidden_size=1024,
            bidirectional=True,
            num_layers=2
        )
        
        self.joint_proj = nn.Linear(in_dim + 3, joint_dim)
        self.marker_proj = nn.Linear(in_dim + 3, marker_dim)
        self.joint_pos_embed = nn.Parameter(torch.randn(1, self.num_joint, joint_dim))
        self.marker_pos_embed = nn.Parameter(torch.randn(1, self.num_marker, marker_dim))
        
        self.CoevoBlocks = nn.ModuleList([
            CoevoBlock(num_joint=num_joint, num_vertx=num_marker, joint_dim=joint_dim, vertx_dim=marker_dim, num_frames=num_frames)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        
        self.proj_joint_feat2coor = nn.Linear(joint_dim, 3)
        self.proj_vertx_feat2coor = nn.Linear(marker_dim, 3)
        
        self.linear1 = nn.Linear((num_joint + num_marker) * embed_dim, hidden_dim)
        self.relu1 = nn.GELU()
        
        self.integrator = Integrator(hidden_dim + img_dim * 2, hidden_dim)
        
        self.regressor = Regressor(
            hidden_dim, embed_dim, [self.n_pose *6, 10, 3, 4]
        )
        
    def forward(self, joint_feats, marker_feats, img_feats):
        B, T, J, Cj = joint_feats.shape
        M, Cm = marker_feats.shape[-2:]
        # temporal image feature
        img_feats, _ = self.gru(img_feats.permute(1,0,2))
        img_feats = img_feats.permute(1,0,2)
        
        joints = joint_feats[..., -3:].reshape(B, T, self.num_joint, 3)
        markers = marker_feats[..., -3:].reshape(B, T, self.num_marker, 3)
        
        joint_feats, marker_feats = joint_feats.reshape(-1, J, Cj), marker_feats.reshape(-1, M, Cm)
        joint_feats, marker_feats = self.joint_proj(joint_feats), self.marker_proj(marker_feats) # [B,17,3] -> [B,17,64], [B,431,3] -> [B,431,64]
        # pos_embed
        joint_feats, marker_feats = joint_feats + self.joint_pos_embed, marker_feats + self.marker_pos_embed
        
        for idx in range(self.depth):
            joint_feats, marker_feats = self.CoevoBlocks[idx](joint_feats, marker_feats, img_feats)
            joint_feats = self.norm(joint_feats)   # [B, T, J, C]
            marker_feats = self.norm(marker_feats) # [B, T, M, C]
         
        # joint coordinate
        joints, markers = self.proj_joint_feat2coor(joint_feats).reshape(B, T, J, 3), self.proj_vertx_feat2coor(marker_feats).reshape(B, T, M, 3) # [B, T, J, 3], [B, T, M, 3]
        
        motion_feats = torch.cat((joint_feats, marker_feats), dim=-2) # [B, T, J+M, C]
        motion_feats = self.linear1(motion_feats.reshape(B, T, -1))   # [B, T, hidden]
        motion_feats = self.relu1(motion_feats)
        motion_feats = self.integrator(motion_feats, img_feats)       # [B, T, C]
        
        (pred_pose, pred_shape, pred_cam, pred_contact) = self.regressor(motion_feats)
        
        return pred_pose, pred_shape, pred_cam, pred_contact, joints, markers