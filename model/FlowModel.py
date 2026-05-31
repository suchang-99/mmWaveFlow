import torch
import torch.nn.functional as F

from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn

from torchdiffeq import odeint
import matplotlib.pyplot as plt
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from typing import Final, Optional, Type


class AttnPool(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=8, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim_in) * 0.02)
        self.mha = nn.MultiheadAttention(dim_in, num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, dim_out))
        #self.mlp=nn.Linear(dim_in, dim_out)

    def forward(self, x, mask=None):
        # x: (B, N, C)
        B, N, C = x.shape
        q = self.query.expand(B, 1, C)
        y, _ = self.mha(q, x, x, key_padding_mask=mask)
        return self.out(y.squeeze(1)).unsqueeze(1)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class PositionalEmbedding(nn.Module):
    def __init__(self, in_dims=3, embed_dim=384):
        super(PositionalEmbedding, self).__init__()
        self.in_dims = in_dims
        self.embed_dim = embed_dim

        self.freq_bands = embed_dim // (in_dims * 2)

        self.freqs = (2 ** torch.linspace(0, self.freq_bands - 1, self.freq_bands))/2

        self.remain = embed_dim - self.freq_bands * in_dims * 2

    def forward(self, x):
        """
        x: [B,2, 3]
        return: [B, 2, 384]
        """
        _,point_num,_=x.shape
        x = x.unsqueeze(-1) * self.freqs.to(x.device)   # [B, 3, freq_bands]

        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        out = torch.cat([x_sin, x_cos], dim=-1)  # [B, 3, 2*freq_bands]
        out = out.reshape(x.shape[0],point_num, -1)        # [B, 384]
        out = F.pad(out, (0, self.remain), 'constant', 0)
        return out


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa* self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp* self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, latent_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, latent_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class ConditionAlignment(nn.Module):
    def __init__(self, latent_dim=256, nhead=8, latent_num=16, hidden_size=512, dropout=0.3, mlp_ratio=4.0):
        super().__init__()
        self.latent_num = latent_num
        self.latent_dim = latent_dim//2
        self.raw_dim=latent_dim
        # shared queries: (1, K, D)
        self.queries = nn.Parameter(torch.randn(1, latent_num, self.latent_dim) * 0.02)
        self.compress = nn.Sequential(
            nn.LayerNorm(self.raw_dim),
            nn.Linear(self.raw_dim, self.raw_dim),
            nn.GELU(),
            nn.Linear(self.raw_dim, self.latent_dim),
        )
        # pre-norm for stability
        self.q_norm = nn.LayerNorm(self.latent_dim)
        self.x_norm = nn.LayerNorm(self.latent_dim)

        self.cross = nn.MultiheadAttention(self.latent_dim, nhead, dropout=0, batch_first=True)

        # FFN on slots, output hidden_size
        mid = int(self.latent_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, hidden_size),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, latent_dim)
        return: (B, K, hidden_size)  # K=latent_num
        """
        B = x.size(0)
        q = self.queries.expand(B, -1, -1)              # (B,K,D)

        # cross-attn with pre-norm + residual
        x=self.compress(x)
        attn_out, _ = self.cross(self.q_norm(q), self.x_norm(x), self.x_norm(x))
        s = q + attn_out                                # (B,K,D)
        return self.mlp(s)


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        latent_num=32,
        latent_dim=128,
        is_condition=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size= hidden_size
        self.latent_dim=latent_dim
        self.is_condition=is_condition

        self.x_embedder=nn.Linear(latent_dim,hidden_size)

        self.c_embedder=ConditionAlignment(latent_dim=latent_dim,hidden_size=hidden_size,latent_num=latent_num, dropout=self.is_condition)

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.num_patches = latent_num
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=True)
        self.attenpool=AttnPool(latent_dim,hidden_size)

        block_kwargs={}
        block_kwargs['attn_drop'] = 0
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,**block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.latent_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # print(self.pos_embed.shape)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size,self.num_patches)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        #nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t,condition=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x=self.x_embedder(x)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)   # (N, D)
        c =  self.c_embedder(condition)+t.unsqueeze(1)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        return x

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    #pos = pos.reshape(-1)  # (M,)
    pos = np.arange(pos)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PointFlowMatching(Module):

    def __init__(self,latent_num,latent_dim, net_depth=30,is_condition=0.3):
        super().__init__()

        self.latent_num=latent_num
        self.latent_dim=latent_dim
        self.net=DiT(depth=net_depth,hidden_size=512,latent_num=self.latent_num,latent_dim=self.latent_dim,is_condition=is_condition)
        self.uncondi_token = nn.Parameter(torch.randn( self.latent_num, self.latent_dim) * 0.02)

        self.clip_values = (-1., 1.)
        self.clip_flow_values = (-3., 3)
        self.odeint_kwargs = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        )
        self.pos3d_emb=PositionalEmbedding(3,self.latent_dim)
        self.is_global_emb=False



    def get_loss(self, x_0, y_0,mean_x=0,mean_mm=0, context=None,t=None,is_zero_t=False):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, point_num,point_dim = x_0.size()
        device = x_0.device
        alpha = beta = 0.5  # 越小越贴近两端；例如 0.3 比 0.5 更尖
        dist = torch.distributions.Beta(torch.tensor(alpha, device=device),
                                torch.tensor(beta,  device=device))
        times = dist.sample((batch_size,))
        times[:20]=0
        times[-20:]=1

        t=times.unsqueeze(1).unsqueeze(2)
        noise_scale = t * (1 - t)
        eps=torch.randn_like(x_0)
        tmp_noise = eps * noise_scale*0.1

        sample_z=t * y_0 + (1. - t) * x_0
        noised =sample_z +tmp_noise

        flow=y_0-x_0

        flow=flow.detach()

        #null_indicator = torch.rand(x_0.size(0), device=x_0.device) > 0.5
        null_indicator=torch.rand(x_0.size(0), device=x_0.device) < (0.3 + 0.4 *times)
        tcondition=x_0.clone()

        tcondition[null_indicator]=y_0[null_indicator]

        pred_flow = self.net(noised, times,tcondition)

        loss=F.mse_loss(flow.view(-1, point_dim), pred_flow.view(-1, point_dim), reduction='mean')

        maeloss=F.l1_loss(flow.view(-1, point_dim), pred_flow.view(-1, point_dim), reduction='mean')

        return loss,maeloss


    def sample(self, sampled_data, mean_pos=0, steps=2, point_dim=3, flexibility=0.0, ret_traj=False):
        times = torch.linspace(0., 1., steps, device=sampled_data.device)
        times[0] += 0.01
        z = sampled_data.clone()
        condi = z.clone()  # +self.direction_emb(torch.zeros(z.size(0), dtype=torch.long, device=z.device)).unsqueeze(1)
        dt = (times[-1] - times[-2]).unsqueeze(0)
        for t in times[:-1]:
            v1 = self.net(z, t.unsqueeze(0), condition=condi)
            z = z + v1 * dt

        return z

    def reverse_sample(self, sampled_data, mean_pos=0,steps=2, point_dim=3, flexibility=0.0, ret_traj=False):

        times = torch.linspace(1., 0., steps, device=sampled_data.device)
        times[0]-=0.01
        z=sampled_data.clone()
        dt = (times[-1] - times[-2]).unsqueeze(0)
        condi = z.clone()
        for t in times[:-1]:
            v = self.net(z, t.unsqueeze(0), condition=condi)
            z = z + v*dt
        return z

