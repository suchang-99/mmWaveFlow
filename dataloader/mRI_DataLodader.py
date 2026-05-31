import time
from operator import index

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import pathlib
import numpy as np
import json
import pandas as pd
#import cv2
import os
import torch
from torch.optim.lr_scheduler import StepLR
from utils.parallel_lidar_simulator import LidarSimulator
import math

from utils.prior_utils import MaxMixturePrior
from model.smpl.pytorch.smpl import SMPL

def fuse_data(datalist, idx,fuse_num=5):
    assert fuse_num%2==1
    single_num=fuse_num//2
    # [print(i) for i in range(max(0, idx - single_num), min(len(datalist), idx + single_num+1))]
    pcs = [datalist[i] for i in range(max(0, idx-single_num), min(len(datalist), idx+single_num+1))]
    return np.vstack(pcs)

def coordinate_transform(data):
    original_y = data[..., 1].clone()

    data[..., 1] = data[..., 2]  # Y <- Z
    data[..., 2] = -original_y  # Z <- -Y

    # X轴保持不变
    return data


class FitttingModel():
    """Implementation of SMPLify, use surface."""

    def __init__(self):
        self.pose_prior = MaxMixturePrior(prior_folder="data/preprocess_data/",
                                          num_gaussians=8,
                                          dtype=torch.float32).cuda()
        self.SMPL_TO_COCO17 = [
                            0,            # 0  nose
                            12, 16,       # 1,2  lEye, rEye  → 用头部顶点代替
                            18, 19,       # 3,4  lEar, rEar  → 用头部顶点代替
                            9, 2,         # 5,6  lShoulder, rShoulder
                            11, 4,        # 7,8  lElbow, rElbow
                            13, 6,        # 9,10  lWrist, rWrist
                            3, 1,         # 11,12  lHip, rHip
                            5, 0,         # 13,14  lKnee, rKnee
                            7, 2          # 15,16  lAnkle, rAnkle
                        ]
        self.SMPL_HEAD_VID = [411, 296, 440]
        self.SMPLModel=SMPL()
        self.device="cuda:0"
        self.corr_idx=[
            16,17,18,19,20,21,1,2,4,5,7,8]
        self.coco_regressor=torch.tensor(np.load("data/preprocess_data/J_regressor_coco.npy"),device=self.device).unsqueeze(0)


    def fit_posetomesh(self, pose,iter_num=1000):
        pose = torch.tensor(pose).cuda()
        ini_rot = torch.nn.Parameter(
            torch.tensor([0., np.pi, 0.], device=self.device)  # 先创建叶子
            .expand(pose.shape[0], -1)  # 形状匹配
            .clone().detach()  # 断开图，重新变成叶子
        )

        ini_pose_params = torch.nn.Parameter(torch.zeros(pose.shape[0], 69, device=self.device))

        betas = torch.nn.Parameter(torch.zeros(pose.shape[0], 10, device=self.device))

        trans_param = torch.nn.Parameter(
            torch.mean(pose[:, 11:13], dim=1).clone().detach()
        )

        optimizer = torch.optim.AdamW([ini_rot, ini_pose_params, betas, trans_param], lr=1e-2)
        scheduler_1 = StepLR(optimizer, step_size=1000, gamma=0.1)
        criterion = torch.nn.MSELoss(reduction='none')
        tmp_coco_regressor=self.coco_regressor.expand(pose.shape[0],-1,-1).to(torch.float32)
        for step in range(iter_num):
            optimizer.zero_grad()
            pose_params = torch.cat([ini_rot, ini_pose_params], dim=1)
            pred_vertices, pred_j = self.SMPLModel(pose_params, betas, trans_param)
            result=torch.bmm(tmp_coco_regressor,pred_vertices)
            joints3d=result
            loss = torch.mean(torch.nn.functional.pairwise_distance(joints3d, pose))+torch.mean(self.pose_prior(ini_pose_params,betas))*1e-5+torch.mean(torch.abs(betas))*1e-4
            loss.backward()
            optimizer.step()
            scheduler_1.step()
        return pose_params,betas,trans_param,pred_vertices,loss.item(),torch.mean(torch.nn.functional.pairwise_distance(joints3d, pose),-1)

def rodrigues(axis: torch.Tensor, theta: float):
    assert axis.shape == (3,)
    axis = axis / axis.norm()          # 归一化
    cos = math.cos(theta)
    sin = math.sin(theta)
    ux, uy, uz = axis
    cross_mat = torch.tensor([[  0, -uz,  uy],
                              [ uz,   0, -ux],
                              [-uy,  ux,   0]], dtype=axis.dtype, device=axis.device)
    R = cos * torch.eye(3, dtype=axis.dtype, device=axis.device) + \
        sin * cross_mat + \
        (1 - cos) * torch.ger(axis, axis)
    return R


def rotate_points(points: torch.Tensor, axis: torch.Tensor, angle_deg: float):
    theta = math.radians(angle_deg)
    R = rodrigues(axis, theta)          # (3, 3)
    return torch.einsum('bnj,ij->bni', points, R)

def _ensure_bx3(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 3:
        raise ValueError(f"last dim must be 3, got {x.shape}")
    return x

def smpl_to_zup(points: torch.Tensor, method: str = "rx90") -> torch.Tensor:

    p = _ensure_bx3(points)
    dtype = p.dtype
    device = p.device


    # method == "swap_yz_flipx":
    # M = FlipX @ SwapYZ
    # SwapYZ = [[1,0,0],[0,0,1],[0,1,0]]
    # FlipX  = diag(-1,1,1)
    M = torch.tensor([[-1., 0., 0.],
                      [ 0., 0., 1.],
                      [ 0., 1., 0.]], dtype=dtype, device=device)

    return torch.matmul(p, M.T)

def fitting_all_data(valid_pose,fit_model):
    all_prepose=torch.zeros(0,72)
    all_prebeta=torch.zeros(0,10)
    all_pretrans=torch.zeros(0,3)
    all_joint_loss=torch.zeros(0)
    gap_num=200
    lidar_pos=torch.zeros(gap_num,3).cuda()
    lidar_pos[:,1]-=5
    for i in range(0,valid_pose.shape[0],gap_num):
        prepose,prebeta,pretrans,pred_mesh,loss_item,joint_loss=fit_model.fit_posetomesh(valid_pose[i:i+gap_num],iter_num=3000)
        torch.cuda.empty_cache()
        # pred_mesh=smpl_to_zup(pred_mesh)
        all_prepose=torch.cat([all_prepose,prepose.detach().cpu()],0)
        all_prebeta=torch.cat([all_prebeta,prebeta.detach().cpu()],0)
        all_pretrans=torch.cat([all_pretrans,pretrans.detach().cpu()],0)
        all_joint_loss=torch.cat([all_joint_loss,joint_loss.detach().cpu()],0)

    save_dict={
        "prepose":all_prepose.cpu().numpy(),
        "prebeta":all_prebeta.cpu().numpy(),
        "pretrans":all_pretrans.cpu().numpy(),
        "joint_loss":all_joint_loss.cpu().numpy(),
    }
    return save_dict



class mRIDataset(Dataset):

    def __init__(self, cfg, mode):

        self.cfg = cfg
        self.mode = mode

        if(not os.path.exists(cfg.data_path)):
            self.generate_data(self.mode)

        test_subject=[17,18,19,20]
        all_data_dict=np.load(cfg.data_path,allow_pickle=True).item()

        if(mode=="train"):
            for i in range(1,21):
                if(i in test_subject):
                    continue
                subject_data=all_data_dict[i]
                if(not hasattr(self,"data_list")):
                    self.data_list=subject_data
                else:
                    self.data_list+=subject_data
        else:
            self.data_list=[]
            for i in test_subject:
                subject_data=all_data_dict[i]
                self.data_list+=subject_data

        self.len=len(self.data_list)
        print("data length of "+mode+":"+str(self.len))

    def generate_partial_pc(self,data_dict, simulator, valid_pose, valid_frames, subject_name):
        gap_num = 15
        lidar_pos = torch.zeros(1, 3).cuda()
        lidar_pos[:, 1] -= 5
        smplModel = SMPL().cuda()
        for i in range(0, valid_pose.shape[0], gap_num):
            pred_mesh = smplModel(torch.tensor(data_dict["prepose"][i:i + gap_num]).cuda(),
                                  torch.tensor(data_dict["prebeta"][i:i + gap_num]).cuda(),
                                  torch.tensor(data_dict["pretrans"][i:i + gap_num]).cuda())[0]
            pred_mesh = smpl_to_zup(pred_mesh)
            sim_pc_result = simulator.get_gen_mesh_pc(lidar_pos.expand(pred_mesh.shape[0], -1), pred_mesh)
            for tmp_index in range(0, len(sim_pc_result)):
                file_path = self.cfg.raw_data_path+"processed_data/" + subject_name + "/lidar/frame" + str(
                    valid_frames[0] + i + tmp_index) + ".pkl"
                if (not os.path.exists(file_path)):
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                save_data = sim_pc_result[tmp_index].cpu().detach().numpy()
                valid_mask = np.sum(np.abs(save_data), axis=1) > 1e-5
                save_data = save_data[valid_mask]
                if (save_data.shape[0] > 512):
                    selected_idx = np.linspace(0, save_data.shape[0] - 1, 512).astype(np.int32)
                    save_data = save_data[selected_idx]
                else:
                    save_data = np.pad(save_data, ((0, 512 - save_data.shape[0]), (0, 0)), 'constant',
                                       constant_values=0)
                # print(save_data.shape)

                save_data = torch.tensor(save_data)
                with open(file_path, 'wb') as f:
                    pickle.dump(save_data, f)


    def generate_data(self,mode):

        # split_path = r"data\mRI\dataset_release\model\mmWave\mmWave_protocol1_datasplit2.pkl"
        # with open(split_path, 'rb') as f:
        #     split = pickle.load(f)
        # file = pickle.load(open('.../file_path/XXX.cpl', 'rb'))
        fit_model=FitttingModel()
        simulator=LidarSimulator()

        the_paired_data_dict={}
        for i in range(1,21):
            tmp_data_list=[]
            subject_name="subject"+str(i)
            pose_path=self.cfg.raw_data_path+"dataset_release/aligned_data/pose_labels/"+subject_name+"_all_labels.cpl"
            mmwave_data=self.cfg.raw_data_path+"dataset_release/aligned_data/radar/singleframe/"+subject_name+".csv"

            pose_label = pickle.load(open(pose_path, 'rb'))

            valid_frames=pose_label["radar_avail_frames"]

            all_gt_pose=pose_label["refined_gt_kps"].transpose(0,2,1)  #(num_frames,17,3)
            valid_pose=all_gt_pose[valid_frames[0]:valid_frames[1]]

            processed_data=fitting_all_data(valid_pose,fit_model)

            #processed_data=np.load("data/mRI/processed_data/new_"+subject_name+"_pre.npy",allow_pickle=True).item()

            loss_item=processed_data["joint_loss"]

            self.generate_partial_pc(processed_data, simulator, valid_pose, valid_frames, subject_name)

            df=pd.read_csv(mmwave_data)
            grouped_data = dict(tuple(df.groupby('Frame #')))
            corr_path_list=[]
            mmwave_pc_list=[]
            gt_pose_list=[]
            loss_item_list=[]
            for frames_idx in list(grouped_data.keys()):
                mmwave_data=grouped_data[frames_idx]

                mmwave_pc=mmwave_data[['X', 'Y', 'Z']].to_numpy()
                gt_corr_idx=int(mmwave_data["Camera Frame"].to_numpy()[0])
                if(gt_corr_idx>=valid_frames[1]):
                    break
                gt_pose=all_gt_pose[gt_corr_idx]
                gt_pose_list.append(gt_pose)
                loss_item_list.append(loss_item[gt_corr_idx - valid_frames[0]])
                corr_lidar_pc_path=self.cfg.raw_data_path+"processed_data/"+subject_name+"/lidar/frame"+str(gt_corr_idx)+".pkl"
                radar_pc_path=self.cfg.raw_data_path+"processed_data/"+subject_name+"/mmwave/frame"+str(frames_idx)+".pkl"
                corr_path_list.append((radar_pc_path,corr_lidar_pc_path))
                mmwave_pc_list.append(mmwave_pc)

            the_paired_data_dict[i]=tmp_data_list
            print(len(tmp_data_list))

            for index in range(len(corr_path_list)):
                mmwave_data=torch.tensor(fuse_data(mmwave_pc_list,index,fuse_num=7))
                # source_mmwave_data=mmwave_data.clone()
                tmp_dist=torch.cdist(mmwave_data,mmwave_data.mean(0).unsqueeze(0)).reshape(-1)
                mmwave_data=mmwave_data[tmp_dist<1.3]

                if(mmwave_data.shape[0]<10):
                    continue

                tmp_dist=torch.cdist(mmwave_data,mmwave_data)
                tmp_dist[tmp_dist==0]=100

                min_dist=torch.min(tmp_dist,dim=1).values
                mmwave_data=mmwave_data[min_dist<0.5]

                if(mmwave_data.shape[0]<10):
                    continue

                radar_pc_path=corr_path_list[index][0]
                data_dict = {
                    "mmwave_data_dir": radar_pc_path,
                    "data_dir": corr_path_list[index][1],
                    "gt_pose": gt_pose_list[index],
                    "loss_item": loss_item_list[index],
                }
                tmp_data_list.append(data_dict)

                if(not os.path.exists(radar_pc_path)):
                    os.makedirs(os.path.dirname(radar_pc_path),exist_ok=True)

                with open(radar_pc_path, 'wb') as f:
                    pickle.dump(mmwave_data, f)

        np.save(self.cfg.raw_data_path+"all_data_list.npy",the_paired_data_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        single_data = self.data_list[index]

        data_file = single_data['data_dir'].replace(".bin",".pkl")

        with open(data_file, 'rb') as f:
            try:
                pc_data= pickle.load(f, encoding='bytes')
            except Exception as e:
                print(f)

        tmp_num=torch.sum(torch.sum(torch.abs(pc_data),dim=-1)>0)
        pc_data=pc_data[:tmp_num].repeat(int(np.ceil(512/tmp_num.item())),1)[:512]
        pc_data=pc_data.to(torch.float32)

        single_data['index']=index

        with open(single_data['mmwave_data_dir'], 'rb') as f:
            try:
                mmwave_data= pickle.load(f, encoding='bytes') #data/MMFi_Dataset/E02/S20/A09/lidar/frame234.pkl
            except Exception as e:
                print(f)
                print(e)
        tmp_num=torch.sum(torch.sum(torch.abs(mmwave_data),dim=-1)>0)
        mmwave_data=mmwave_data[:tmp_num].repeat(int(np.ceil(512/tmp_num.item())),1)[:512]
        return {
            "point_cloud": pc_data.to(torch.float32),
            "mmwave_point_cloud": mmwave_data.to(torch.float32),
            "idx":index,
        }




