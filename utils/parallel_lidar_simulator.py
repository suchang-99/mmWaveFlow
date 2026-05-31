'''
Written by Chang Su
'''
import numpy as np

import torch
import torch.nn as nn
from model.smpl.pytorch.smpl import SMPL
def torch_calc_laser_direction(mesh_r, mesh_theta, mesh_phi):
    laser_direction = torch.cat(((torch.cos(mesh_theta) * torch.cos(mesh_phi)).unsqueeze(1),
                                 (torch.cos(mesh_theta) * torch.sin(mesh_phi)).unsqueeze(1),
                                 torch.sin(mesh_theta).unsqueeze(1)), axis=1)
    return laser_direction

class LidarSimulator(nn.Module):
    def __init__(self):
        super().__init__()

        self.faces = torch.Tensor(SMPL().faces).long()

        self.predefined_theta_resolution = torch.tensor(np.radians(20 / 64)).cuda()
        self.predefined_phi_resolution=torch.tensor(np.radians(360/2650)).cuda()


    def set_faces(self,faces):
        self.faces=torch.Tensor(faces.astype(np.int16)).type(torch.int32)


    def new_torch_bacth_ray_triangle_intersection(self, ray_origins, ray_directions, triangle_vertices_list):

        epsilon = 1e-6

        vertex0, vertex1, vertex2 = triangle_vertices_list[:,:, 0], triangle_vertices_list[:,:, 1], triangle_vertices_list[
                                                                                                :,:, 2]
        edge1 = vertex1 - vertex0
        edge2 = vertex2 - vertex0

        h = torch.cross(ray_directions.unsqueeze(1).repeat(1, edge2.shape[1], 1),
                        edge2, dim=-1)

        a = torch.einsum('ijk,ijk->ij', h, edge1)

        a_abs = torch.abs(a)

        a_mask = a_abs < epsilon
        a[a_mask] = 1.0  # to avoid division by zero for parallel rays

        f = 1.0 / a

        s = ray_origins.unsqueeze(1) - vertex0

        u = f * torch.einsum('ijk,ijk->ij', h, s)

        mask = torch.logical_or(u < 0.0, u > 1.0)

        f[mask] = 0.0  # set f to zero for rays where u is not within [0, 1]

        q = torch.cross(s, edge1)

        v = f * torch.einsum('ijk,ijk->ij', ray_directions.unsqueeze(1), q)

        mask = torch.logical_or(v < 0.0, u + v > 1.0)

        f[mask] = 0.0  # set f to zero for rays where v is not within [0, 1]

        t = f * torch.einsum('ijk,ijk->ij', edge2, q)

        mask = t > epsilon

        t[~mask] = 0.0  # set t to zero for rays where t is not greater than epsilon

        t[t<0.1] = np.inf
        t = torch.min(t, dim=1)

        t = t.values

        intersections = ray_origins + t.unsqueeze(1) * ray_directions

        return intersections,t==np.inf


    def calc_spherical_coordinate_3dim(self, data):
        mesh_r = torch.sqrt(torch.sum(data ** 2, dim=2))
        mesh_phi = torch.arctan2(data[:, :, 1], data[:, :, 0])
        mesh_theta = torch.arcsin(data[:, :, 2] / mesh_r)
        return mesh_r, mesh_theta, mesh_phi

    def calc_spherical_coordinate_4dim(self, data):
        mesh_r = torch.sqrt(torch.sum(data ** 2, dim=3))
        mesh_phi = torch.arctan2(data[:, :,:, 1], data[:, :, :,0])
        mesh_theta = torch.arcsin(data[:, :, :,2] / mesh_r)
        return mesh_r, mesh_theta, mesh_phi

    def all_zero(self,pc_num):
        final_result = torch.zeros((torch.sum(pc_num), 3)).cuda()
        return final_result

    def complete_all_missing_point(self,mesh_mean,pc_num):
        final_result = mesh_mean.repeat_interleave(pc_num, dim=0)
        return final_result


    def get_selected_laser_direction(self,mesh):


        mesh_r, mesh_theta, mesh_phi = self.calc_spherical_coordinate_3dim(mesh)

        phi_gap = ((mesh_phi)/self.predefined_phi_resolution).type(torch.int32)
        theta_gap=((mesh_theta)/self.predefined_theta_resolution).type(torch.int32)
        data = torch.cat((phi_gap.unsqueeze(2), theta_gap.unsqueeze(2)), dim=2)


        unique_data=torch.zeros((0,2)).cuda()
        tmp_pc_num=[]
        for item in data:
            tmp_data=torch.unique(item, dim=0)
            if(tmp_data.shape[0]>1024):
                idx = torch.linspace(0, tmp_data.shape[0] - 1, steps=1024, device=tmp_data.device).long()
                tmp_data = tmp_data[idx]
            unique_data=torch.cat((unique_data,tmp_data),dim=0)
            tmp_pc_num.append(tmp_data.shape[0])
        pc_num = torch.tensor(tmp_pc_num, dtype=torch.int32).cuda()

        return unique_data[:,0]*self.predefined_phi_resolution,unique_data[:,1]*self.predefined_theta_resolution,pc_num




    def get_gen_mesh_pc(self, lidar_pos, source_mesh):

        mean_mesh = torch.mean(source_mesh, dim=1)
        mesh = source_mesh - lidar_pos.unsqueeze(1).cuda()

        tri_mesh = mesh[:, self.faces]

        laser_phi,laser_theta, pc_num = self.get_selected_laser_direction(mesh)

        mesh_r, mesh_theta, mesh_phi = self.calc_spherical_coordinate_4dim(tri_mesh)

        data = torch.cat((mesh_phi.unsqueeze(3), mesh_theta.unsqueeze(3)), dim=3)

        mesh_info = torch.cat(
            (torch.max(data[:, :, :, 0], dim=2).values.unsqueeze(2),
             torch.min(data[:, :, :, 0], dim=2).values.unsqueeze(2),
             torch.max(data[:, :, :, 1], dim=2).values.unsqueeze(2),
             torch.min(data[:, :, :, 1], dim=2).values.unsqueeze(2)),
            dim=2)
        mesh_info = torch.repeat_interleave(mesh_info, pc_num, dim=0)


        mesh_info[:, :, :2] = mesh_info[:, :, :2] - laser_phi.unsqueeze(1).unsqueeze(2)
        mesh_info[:, :, 2:] = mesh_info[:, :, 2:] - laser_theta.unsqueeze(1).unsqueeze(2)

        x_result = torch.logical_and(mesh_info[:, :, 0] > 0, mesh_info[:, :, 1] < 0)
        y_result = torch.logical_and(mesh_info[:, :, 2] > 0, mesh_info[:, :, 3] < 0)
        result = torch.logical_and(x_result, y_result)

        selected_index = torch.where(result == True)

        laser_index, mesh_num = torch.unique_consecutive(selected_index[0], return_counts=True, return_inverse=False)

        if (mesh_num.shape[0] == 0):
            return self.complete_all_missing_point(mean_mesh, pc_num)
        selected_mesh_matrix = torch.zeros((laser_index.shape[0], max(mesh_num)), dtype=torch.long)

        selected_mesh_index = selected_index[1].split(mesh_num.tolist())
        for i in range(0, laser_index.shape[0]):
            selected_mesh_matrix[i, :len(selected_mesh_index[i])] = selected_mesh_index[i]


        selected_laser = torch_calc_laser_direction(None,mesh_theta= laser_theta,mesh_phi=laser_phi)[laser_index]

        selected_faces = self.faces[selected_mesh_matrix]
        boundaries = torch.cumsum(pc_num, dim=0) - 1
        the_current_mesh = torch.bucketize(laser_index, boundaries)

        mesh_index, every_mesh_num = torch.unique_consecutive(the_current_mesh, return_counts=True,
                                                              return_inverse=False)

        begin_point = torch.repeat_interleave(lidar_pos[mesh_index], every_mesh_num, dim=0)

        source_mesh = source_mesh[mesh_index]
        every_mesh_num_sum = torch.cumsum(every_mesh_num, dim=0)
        all_mesh = [source_mesh[0][selected_faces[:every_mesh_num_sum[0]]]]
        for i in range(1, mesh_index.shape[0]):
            all_mesh.append(source_mesh[i][selected_faces[every_mesh_num_sum[i - 1]:every_mesh_num_sum[i]]])

        selected_mesh = torch.cat(all_mesh, dim=0)
        intersection, mask = self.new_torch_bacth_ray_triangle_intersection(begin_point,
                                                                            selected_laser.type(torch.float32),
                                                                            selected_mesh)
        intersection[mask] = 0

        return intersection.split(every_mesh_num.tolist())

    def forward(self, lidar_pos, source_mesh,pc_num,laser_direction):
        gen_pc=self.get_gen_mesh_pc(lidar_pos, source_mesh)
        return gen_pc
