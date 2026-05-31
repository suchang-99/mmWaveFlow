import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PyTorchEMD.emd import earth_mover_distance as EMD
import utils.universal_utils as uni_utils
import utils.model_utils as model_utils
from model.FlowModel import PointFlowMatching

import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import soft_label_patch_contrastive_loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()




class _CrossSelfBlock(nn.Module):
    """ Pre-LN: Cross-Attn(Q=slots←K/V=x) → Self-Attn(on slots) → FFN(on slots) """
    def __init__(self, d_model, nhead, dropout, mlp_ratio):
        super().__init__()
        self.ln_q1 = nn.LayerNorm(d_model)
        self.ln_x  = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop  = nn.Dropout(dropout)

        self.ln_q2 = nn.LayerNorm(d_model)
        self.selfa = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, x):
        # Cross-Attn: Q=q, K/V=x
        qn = self.ln_q1(q)
        xn = self.ln_x(x)
        out, _ = self.cross(qn, xn, xn, need_weights=False)
        q = q + self.drop(out)

        # Self-Attn on slots
        qn = self.ln_q2(q)
        out, _ = self.selfa(qn, qn, qn, need_weights=False)
        q = q + self.drop(out)

        # FFN
        q = q + self.ffn(q)
        return q

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8, scale_init=2.0):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d) * scale_init)
    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.g


class CrossAttnEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, n_query=16, n_layers=2, dropout=0.2, mlp_ratio=4.0):
        super().__init__()
        self.d_model = d_model
        self.n_query = n_query

        self.queries = nn.Parameter(torch.randn(n_query, d_model) * 0.02)
        blocks = []
        for _ in range(n_layers):
            blocks.append(_CrossSelfBlock(d_model, nhead, dropout, mlp_ratio))
        self.blocks = nn.ModuleList(blocks)
        self.readout_norm = nn.LayerNorm(d_model)
        self.prenorm = RMSNorm(d_model)

        self.mean_pre=nn.Linear(d_model,d_model)
        self.log_pre=nn.Linear(d_model,d_model)
        tlayer = nn.TransformerEncoderLayer(d_model= d_model, nhead=8, batch_first=True,dropout=dropout)
        self.self_atten = nn.TransformerEncoder(tlayer, num_layers=3)
        #self.proj=nn.Linear(32,16)

    def forward(self, x):
        """
        x: (B, N, D)
        returns:
          slots: (B, K, D)
          z:     (B, D)
        """
        B, N, D = x.shape
        assert D == self.d_model
        x=self.prenorm(x)

        q = self.queries.unsqueeze(0).repeat(B, 1, 1)  # (B,K,D)
        for blk in self.blocks:
            q = blk(q, x)
       
        slots = q  # (B,K,D)
        slots=self.self_atten(slots)
        mean=self.mean_pre(slots)
        log_var=self.log_pre(slots)

        return mean,torch.clamp(log_var,min=-0.5,max=5)#slots


class mWGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        latent_dim=cfg["core"]["latent_dim"]
        latent_num=cfg["core"]["latent_num"]

        from model.Shape2VecSet import myEncoder
        self.encoder=myEncoder(
            depth=12,
            dim=256,
            queries_dim=256,
            num_inputs=512,
            num_latents=32,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
            drop_rate=self.cfg["dataset"]["opt"]["drop_rate"],
        )
        self.decoder=self.encoder

        self.mmencoder=myEncoder(
            depth=12,
            dim=256,
            queries_dim=256,
            num_inputs=self.cfg["dataset"]["opt"]["pc_num"],
            num_latents=32,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
            drop_rate=self.cfg["dataset"]["opt"]["drop_rate"],
        )
        self.mmdecoder=self.mmencoder

        self.cross_attention=CrossAttnEncoder(d_model=latent_dim,nhead=8,n_query=latent_num,n_layers=1)

        self.token_projection=nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2,bias=False),
            nn.LayerNorm( latent_dim//2),
            nn.LeakyReLU(),
            nn.Linear( latent_dim//2,  latent_dim//2,bias=False),
        )
        #self.init_weights()

        if(cfg['core']['is_gen']):
            self.flow=PointFlowMatching(latent_num,latent_dim,net_depth=self.cfg['core']['ini_flow_depth'])



    def init_weights(self):
        for module in [self.encoder, self.mmencoder, self.cross_attention]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    if m.weight is not None:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)

        # learnable queries
        if hasattr(self.cross_attention, 'queries'):
            nn.init.normal_(self.cross_attention.queries, mean=0, std=0.02)


    def prepare_for_train_flow(self):
        latent_num=self.cfg["core"]["latent_num"]
        latent_dim=self.cfg["core"]["latent_dim"]
        self.flow=PointFlowMatching(latent_num,latent_dim,net_depth=self.cfg['core']['final_flow_depth'],is_condition=0).to(self.cfg["core"]["device"])

    def get_z_loss(self,z,z_sigma):
        tmp_var = torch.exp(z_sigma)
        return torch.mean(torch.mean(0.5 * torch.mean(torch.pow(z, 2)
                         + tmp_var - 1.0 - z_sigma,
                         dim=[1, 2])))


    def forward(self,data,mmdata,mean_dense=0,mean_mm=0,epoch=0):

        z_mu=self.encoder.encode(data)
        mmz_mu=self.mmencoder.encode(mmdata)

        mmz_mu, mmz_sigma = self.cross_attention(mmz_mu)
        z_mu, z_sigma = self.cross_attention(z_mu)


        if(self.cfg["core"]["is_reparam"]):
            mask = (torch.rand(z_mu.shape[0], device=z_mu.device) < 0.30)
            z = model_utils.reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
            z[mask]=z_mu[mask]

            mm_z = model_utils.reparameterize_gaussian(mean=mmz_mu, logvar=mmz_sigma)
            mm_z[mask]=mmz_mu[mask]
        else:
            z=z_mu
            mm_z=mmz_mu

        z_loss=self.get_z_loss(z_mu,z_sigma)
        mm_z_loss=self.get_z_loss(mmz_mu,mmz_sigma)


        pc_recons,global_pos=self.decoder(z,is_return_offset=True)

        tmp_mask = torch.sum(torch.abs(data), dim=-1) > 0
        dist1, dist2 = uni_utils.my_chamfer_distance(data, pc_recons, first_dim=1, is_return_dist=True)
        pc_recons_loss = torch.mean(dist1[tmp_mask]) + torch.mean(dist2)

        mm_recons,mm_global_pos=self.mmdecoder(mm_z,is_return_offset=True)

        tmp_mask = torch.sum(torch.abs(mmdata), dim=-1) > 0
        dist1, dist2 = uni_utils.my_chamfer_distance(mmdata, mm_recons, first_dim=1, is_return_dist=True)
        mm_recons_loss = torch.mean(dist1[tmp_mask]) + torch.mean(dist2)

        z_loss= 0.5*z_loss + 0.5*mm_z_loss


        flow_z=z
        flow_mmz=mm_z

        if (self.cfg['core']['is_joint']):
            z_recons_loss,mae_loss=self.flow.get_loss(flow_z,flow_mmz,mean_dense,mean_mm )
        else:
            z_recons_loss,mae_loss=self.flow.get_loss(flow_z.clone().detach(),flow_mmz.clone().detach(),mean_dense,mean_mm)


        proj_mm_z=self.token_projection(mm_z)
        proj_z=self.token_projection(z)

        mcl=soft_label_patch_contrastive_loss(proj_z,proj_mm_z,temperature=self.cfg["dataset"]["opt"]["cl_temperature"])*0.1
        if(self.cfg["dataset"]["opt"]["enhance_cl"]):
            mcl+=torch.nn.functional.mse_loss(z_mu,mmz_mu,reduction="mean")+torch.nn.functional.l1_loss(z_mu,mmz_mu.detach(),reduction="mean")
        else:
            mcl+=torch.nn.functional.l1_loss(z_mu,mmz_mu.detach(),reduction="mean")


        emdl=torch.mean(EMD(data, pc_recons, False))
        mmemd=torch.mean(EMD(mmdata, mm_recons, False))
        
        loss={
            "z_kl":z_loss,
            "z_flow":z_recons_loss,
            "mae_z":mae_loss,
            "pc_re":pc_recons_loss,
            "mm_re":mm_recons_loss,
            "em":emdl,
            "mmem":mmemd,
            "mcl":mcl,
            "reg": torch.mean(torch.abs(z)+torch.abs(mm_z))/2,
        }

        return loss


    def sample(self,dense_data,mmdata,mean_dense,mean_mm,batch_idx):

        mmz_mu=self.mmencoder.encode(mmdata)
        dense_z = self.encoder.encode(dense_data)

        mmz_mu, mmz_sigma = self.cross_attention(mmz_mu)
        dense_z_mean,z_sigma=self.cross_attention(dense_z)

        if(self.cfg["core"]["is_reparam"]):
            mask = (torch.rand(mmz_mu.shape[0], device=mmz_mu.device) < 0.30)
            mm_z = model_utils.reparameterize_gaussian(mean=mmz_mu, logvar=mmz_sigma)
            mm_z[mask]=mmz_mu[mask]
            dense_z = model_utils.reparameterize_gaussian(mean=dense_z_mean, logvar=z_sigma)
            dense_z[mask]=dense_z_mean[mask]
        else:
            mm_z=mmz_mu
            dense_z=dense_z_mean

        samples_dense_z = self.flow.reverse_sample(mm_z,mean_mm)
        sampled_dense_pc=self.decoder(samples_dense_z)

        tmp_mask = torch.sum(torch.abs(dense_data), dim=-1) > 0
        dist1, dist2 = uni_utils.my_chamfer_distance(dense_data, sampled_dense_pc, first_dim=1, is_return_dist=True)
        m2d_recons_loss = torch.mean(dist1[tmp_mask]) + torch.mean(dist2)

        samples_mm_z = self.flow.sample(dense_z,mean_dense)
        sample_mm_pc=self.mmdecoder(samples_mm_z)

        tmp_mask = torch.sum(torch.abs(mmdata), dim=-1) > 0
        dist1, dist2 = uni_utils.my_chamfer_distance(mmdata, sample_mm_pc, first_dim=1, is_return_dist=True)

        d2m_recons_loss = torch.mean(dist1[tmp_mask]) + torch.mean(dist2)

        emdl=torch.mean(EMD(dense_data, sampled_dense_pc, False))
        mmemd=torch.mean(EMD(mmdata, sample_mm_pc, False))

        sample_result={
            "cd":m2d_recons_loss,
            "d2m_cd":d2m_recons_loss,
            "em":emdl,
            "mmem":mmemd,
            "results":sampled_dense_pc,
            "mmre":sample_mm_pc,
        }
        return sample_result


    def shared_parameters(self):
        params = list(self.encoder.parameters()) +  \
                 list(self.mmencoder.parameters()) +  \
                 list(self.cross_attention.parameters())
        return params







