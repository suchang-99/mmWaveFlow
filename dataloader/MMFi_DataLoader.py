import time
from operator import index

import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import pickle
import pathlib
import numpy as np
import json

#import cv2
# import utils.wod_utils as wod_utils
import utils.universal_utils as uni_utils
import os
import sys
import torch



class MMFiDataset(Dataset):

    def __init__(self, cfg, mode):

        self.cfg = cfg
        self.mode = mode
        fliternum=50
        valid_index = np.load(self.cfg.data_path+ mode + "_mmwave_num.npy", allow_pickle=True)

        data_path = self.cfg.data_path + self.mode+"_list.npy"
        if (not os.path.exists(data_path)):
            self.generate_data(mode)

        if(mode=="train"):
            self.data_list=np.load(data_path,allow_pickle=True)
            self.data_list=self.data_list[valid_index>fliternum]
        else:
            self.data_list=np.load(data_path,allow_pickle=True)
            self.data_list = self.data_list[valid_index > fliternum]

        self.len=len(self.data_list)
        print("data length of "+mode+":"+str(self.len))


    def generate_data(self,mode):
        if(mode=="train"):
            environment_list=["E01","E02","E03"]
        else:
            environment_list=["E04"]


        suffix = "data/MMFi_Dataset/"
        modality="lidar"
        self.data_list=[]
        mmwave_num_list=[]
        for environment in environment_list:
            env_path=suffix+environment+"/"

            subject_list=uni_utils.get_all_file(env_path,is_dir=True)

            for subject in subject_list:
                subject_path=env_path+subject+"/"
                action_list=uni_utils.get_all_file(subject_path,is_dir=True)
                print(subject,len(action_list))
                for action in action_list:
                    action_path=subject_path+action+"/"+modality+"/"
                    frame_list=uni_utils.get_all_file(action_path,is_dir=False)
                    ground_truth_path=subject_path+action+"/ground_truth.npy"
                    ground_truth_data=np.load(ground_truth_path,allow_pickle=True)
                    ground_truth_data=uni_utils.mmfi_coordinate_transform(torch.tensor(ground_truth_data))
                    for tmp_index,frame in enumerate(frame_list):
                        frame_path=action_path+frame

                        pc_data=torch.tensor(uni_utils.load_bin_data(frame_path))
                        new_pc=self.fliter_pc(pc_data,ground_truth_data[tmp_index])
                        if(new_pc.shape[0]<50):
                            print(frame_path)

                        if(new_pc.shape[0]>512):
                            save_points=new_pc[:512]
                        else:
                            try:
                                save_points = torch.nn.functional.pad(new_pc, (0, 0, 0, 512 - new_pc.shape[0]))
                            except Exception as e:
                                print(new_pc.shape)
                                print(frame_path)

                        #save_frame_path=frame_path.replace("old_path/data","data").replace(".bin",".pkl")

                        save_frame_path = frame_path.replace(".bin", ".pkl")
                        if(not os.path.exists(save_frame_path)):
                            os.makedirs(os.path.dirname(save_frame_path),exist_ok=True)
                        with open(save_frame_path, 'wb') as f:
                            pickle.dump(save_points, f)

                        mmwave_frame_path=frame_path.replace("lidar/","mmwave/")
                        saved_mmwave_path,mmwave_num=self.merge_mmwave_data(mmwave_frame_path,action_path,frame_list,tmp_index,ground_truth_data[tmp_index])
                        mmwave_num_list.append(mmwave_num)
                        self.data_list.append({"data_dir":save_frame_path,
                                               "mmwave_data_dir":saved_mmwave_path,})

        np.save(self.cfg.data_path+mode+"_list.npy",self.data_list)
        mmwave_num_array = np.array(mmwave_num_list)
        np.save(self.cfg.data_path + mode + "_mmwave_num.npy", mmwave_num_array)

    def merge_mmwave_data(self,mmwave_frame_path,action_path,frame_list,tmp_index,joint):

        mmwave_pc_data = torch.tensor(uni_utils.load_bin_data(mmwave_frame_path,pc_num=5))[:,:3]
        if(tmp_index!=0):
            t_pc=torch.tensor(uni_utils.load_bin_data((action_path+frame_list[tmp_index-1]).replace("lidar/","mmwave/"),pc_num=5))[:,:3]
            mmwave_pc_data=torch.cat([mmwave_pc_data,t_pc],dim=0)
        if(tmp_index!=296):
            t_pc=torch.tensor(uni_utils.load_bin_data((action_path+frame_list[tmp_index+1]).replace("lidar/","mmwave/"),pc_num=5))[:,:3]
            mmwave_pc_data=torch.cat([mmwave_pc_data,t_pc],dim=0)
        if(tmp_index>1):
            t_pc = torch.tensor(
                uni_utils.load_bin_data((action_path + frame_list[tmp_index - 2]).replace("lidar/", "mmwave/"),pc_num=5))[:, :3]
            mmwave_pc_data = torch.cat([mmwave_pc_data, t_pc], dim=0)
        if(tmp_index<295):
            t_pc = torch.tensor(
                uni_utils.load_bin_data((action_path + frame_list[tmp_index + 2]).replace("lidar/", "mmwave/"),pc_num=5))[:, :3]
            mmwave_pc_data = torch.cat([mmwave_pc_data, t_pc], dim=0)

        mmwave_pc_data=mmwave_pc_data[torch.logical_and(mmwave_pc_data[:,2]>-1.2,mmwave_pc_data[:,2]<1.5)]
        tmpdist = torch.min(torch.cdist(mmwave_pc_data.to(torch.float32)[:,:2], joint[:,:2]), -1)[0]
        new_mmwave_pc_data=mmwave_pc_data[tmpdist<0.8]
        if(new_mmwave_pc_data.shape[0]>tmpdist.shape[0]/2):
            mmwave_pc_data=new_mmwave_pc_data

        mean_mmwave_data= torch.mean(mmwave_pc_data,dim=0,keepdim=True)
        tmpdist = torch.min(torch.cdist(mmwave_pc_data.to(torch.float32), mean_mmwave_data.to(torch.float32)),-1)[0]
        mmwave_pc_data = mmwave_pc_data[tmpdist < 1.2]

        save_frame_path = mmwave_frame_path.replace(".bin", ".pkl")
        real_num = mmwave_pc_data.shape[0]
        if real_num >= 512:
            pad_points = mmwave_pc_data[:512]
            real_num = 512
        else:
            pad_points = torch.nn.functional.pad(mmwave_pc_data, (0, 0, 0, 512 - real_num))

        if (not os.path.exists(save_frame_path)):
            os.makedirs(os.path.dirname(save_frame_path), exist_ok=True)
        with open(save_frame_path, 'wb') as f:
            pickle.dump(pad_points, f)

        return save_frame_path, real_num

    def fliter_pc(self,source_pc,coords,threshold=0.3):

        min_threshold = torch.tensor(0.4)
        max_threshold=torch.tensor(0.2)

        min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
        min_y, max_y = coords[:, 1].min(), coords[:, 1].max()

        t_mask = ((source_pc[:, 0] > (min_x - min_threshold)) & (source_pc[:, 0] < (max_x + max_threshold)) &
                  (source_pc[:, 1] > (min_y - min_threshold)) & (source_pc[:, 1] < (max_y + max_threshold)))

        pc = source_pc[t_mask]

        if(True):
            min_z=coords[:, 2].min()
            max_z=pc[:,2].max()-0.05
            new_coords=coords.clone()
            new_coords[:,2]=new_coords[:,2]-(new_coords[:,2].max()-max_z)

            pc_y_diff = np.diff(pc[:, 1])
            the_index = pc_y_diff > 0.1
            split_indices = np.where(the_index)[0] + 1
            split_point_clouds = np.split(pc, split_indices)

            final_pc=torch.zeros((0,3))
            for index, item in enumerate(split_point_clouds):
                if not (len(item) > 15 and item[0][2] < min_z+0.1):
                    final_pc=torch.cat((final_pc,item),dim=0)
            tmp_mask=final_pc[:,2]<(min_z+0.1)

            selected_pc=final_pc[tmp_mask]
            ankle_pc=new_coords[[2,3,5,6]]
            tmpdist=torch.min(torch.cdist(selected_pc.to(torch.float32)[:,:2],ankle_pc[:,:2]),-1)[0]
            tmp_mask=~tmp_mask
            tmp_mask[tmp_mask==False]=tmpdist<0.15

            final_pc[~tmp_mask]=0

        return final_pc

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

        if(tmp_num<1):
             mmwave_data[0]=pc_data[0]
             tmp_num=torch.tensor(1)

        gap_num=np.floor(512%tmp_num.item())
        padding_data=mmwave_data[np.linspace(0,tmp_num-1,gap_num.astype(np.int32)).astype(np.int32)]

        mmwave_data=torch.cat([mmwave_data[:tmp_num].repeat(int(np.floor(512/tmp_num.item())),1),padding_data],dim=0)

        return {
            "point_cloud": pc_data.to(torch.float32),
            "mmwave_point_cloud": mmwave_data.to(torch.float32),
            "idx":index,
        }




