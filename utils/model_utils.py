import torch
import numpy as np
import torch.nn.functional as F

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def soft_label_patch_contrastive_loss(z_lidar, z_radar, temperature=0.1, bidirectional=True, intra_sample_softness=0.2):
    """
    z_lidar, z_radar: (B, P, D)
    intra_sample_softness: 同一样本内不同 patch 之间的正样本权重（0~1之间，推荐 0.05~0.2）
    """
    B, P, D = z_lidar.shape
    N = B * P

    # 归一化
    z_lidar = F.normalize(z_lidar, p=2, dim=-1)
    z_radar = F.normalize(z_radar, p=2, dim=-1)

    lidar_patches = z_lidar.reshape(N, D)  # (N, D)
    radar_patches = z_radar.reshape(N, D)  # (N, D)

    # 相似度矩阵 (N, N)
    logits = (lidar_patches @ radar_patches.t()) / temperature

    # 构建软标签矩阵 Y (N, N)
    # 先得到每个 patch 对应的 batch 索引
    batch_idx = torch.arange(B, device=z_lidar.device).repeat_interleave(P)  # (N,)

    # 方式1：对角线（完全正样本）
    Y = torch.eye(N, device=z_lidar.device)  # (N, N)

    # 方式2：同一样本内不同 patch，设为 intra_sample_softness
    same_batch_mask = (batch_idx[:, None] == batch_idx[None, :])  # (N, N)
    # 排除对角线
    off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=z_lidar.device)
    same_batch_off_diag = same_batch_mask & off_diag_mask

    Y[same_batch_off_diag] = intra_sample_softness

    # 归一化每一行，使得正样本权重之和为 1
    Y_normalized = Y / Y.sum(dim=1, keepdim=True)  # (N, N)

    # 手动计算 softmax 交叉熵（软标签版本）
    log_softmax = F.log_softmax(logits, dim=1)  # (N, N)
    loss_i2r = - (Y_normalized * log_softmax).sum(dim=1).mean()

    if bidirectional:
        # Radar -> LiDAR：对称操作
        logits_t = logits.t()
        Y_t = Y.t()  # 对称的软标签也需要转置
        Y_t_normalized = Y_t / Y_t.sum(dim=1, keepdim=True)
        log_softmax_t = F.log_softmax(logits_t, dim=1)
        loss_r2i = - (Y_t_normalized * log_softmax_t).sum(dim=1).mean()
        loss = 0.5 * (loss_i2r + loss_r2i)
    else:
        loss = loss_i2r

    return loss