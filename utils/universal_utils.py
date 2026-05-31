
import torch.nn as nn
import torch.nn.functional as F
from time import time
import torch
import numpy as np

import time
import os



def my_chamfer_distance(x, y, first_dim=1, scale=1, is_return_dist=False,factor=2):
    distance_matrix = torch.cdist(x, y)**2
    #rint(distance_matrix.shape)

    # print(distance_matrix[:10,:10])
    av_dist1 = torch.min(distance_matrix, first_dim+1)[0]
    av_dist2 = torch.min(distance_matrix, first_dim)[0]

    if (is_return_dist):
        return av_dist1, av_dist2
    return scale * torch.mean(av_dist1) + (1 - scale) * torch.mean( av_dist2)


def mmfi_coordinate_transform(data):
    original_y = data[..., 0].clone()
    #print(data.shape)
    # 交换Y和Z轴，并对新Z轴（原Y轴）取反
    data[..., 0] = data[..., 2]  # Y <- Z



    data[..., 2] = -data[..., 1]  # Z <- -Y
    data[..., 1] = -original_y

    # X轴保持不变
    return data


def get_all_file(file_dir, is_dir=False):
    for root, dirs, files in os.walk(file_dir):
        # print("root", root)  # 当前目录路径
        # print("dirs", dirs)  # 当前路径下所有子目录
        # print("files", files)  # 当前路径下所有非目录子文件
        if (is_dir):
            return dirs
        return files


def load_bin_data(bin_file,pc_num=3):
    with open(bin_file, 'rb') as f:
        raw_data = f.read()
        data_tmp = np.frombuffer(raw_data, dtype=np.float64)
        data_tmp = data_tmp.reshape(-1, pc_num)
    return data_tmp


def normal_data(data):
    tmp_mask = torch.sum(torch.abs(data), dim=-1) > 0
    mean_pos = torch.sum(data, dim=1) / torch.sum(tmp_mask, dim=1).unsqueeze(1)
    mean_pos = mean_pos.unsqueeze(1)

    data -= mean_pos
    data[~tmp_mask] = 0
    return data,mean_pos


def rotate_data_batch(data,dim,angles):
    """
        Rotate a batch of point clouds around the Z-axis without using loops.

        Parameters:
        - point_clouds: Tensor of shape (N, num_points, 3) representing the point clouds.
        - angles: Tensor of shape (N,) representing the rotation angles in degrees for each point cloud.

        Returns:
        - rotated_point_clouds: Tensor of shape (N, num_points, 3) with the rotated point clouds.
        """
    # Convert angles to radians

    angles_rad = angles * (torch.pi / 180.0)

    # Calculate cosine and sine for all angles
    cos_angles = torch.cos(angles_rad)
    sin_angles = torch.sin(angles_rad)

    # Create a tensor for the rotation matrix components
    # Shape: (N, 2, 2) where N is the number of point clouds
    zero_tensor=torch.zeros(cos_angles.shape).to(data.device)
    one_tensor=torch.ones(cos_angles.shape).to(data.device)
    if dim == 0:
        rotation_matrices = torch.stack([
            one_tensor, zero_tensor, zero_tensor,
            zero_tensor, cos_angles, -sin_angles,
            zero_tensor, sin_angles, cos_angles
        ], dim=1).reshape(-1, 3, 3)

    elif dim == 1:
        rotation_matrices = torch.stack([
            cos_angles, zero_tensor, sin_angles,
            zero_tensor, one_tensor, zero_tensor,
            -sin_angles, zero_tensor, cos_angles
        ], dim=1).reshape(-1, 3, 3)
    elif dim == 2:
        rotation_matrices = torch.stack([
            cos_angles, -sin_angles, zero_tensor,
            sin_angles, cos_angles, zero_tensor,
            zero_tensor, zero_tensor, one_tensor
        ], dim=1).reshape(-1, 3, 3)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")


    rotation_matrices=rotation_matrices.to(data.device)
    rotated_point_clouds = torch.bmm(rotation_matrices, data.permute(0,2,1))
    return rotated_point_clouds.permute(0,2,1)