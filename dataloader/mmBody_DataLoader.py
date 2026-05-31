import os
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
import torch
import pickle

class mmBodySequenceLoader(object):
    def __init__(self, seq_path: str, skip_head: int = 0, skip_tail: int = 0, resource=['radar']) -> None:
        self.seq_path = seq_path
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.resource = resource
        # load transformation matrix
        with open(os.path.join(seq_path, 'calib.txt'), "rt") as f:
            calib = eval(f.readline())
        self.calib = {
            'image': calib['kinect_master'],
            'depth': calib['kinect_master'],
        }

    def __len__(self):
        return len(os.listdir(os.path.join(self.seq_path, 'bounding_box/master'))) - self.skip_head - self.skip_tail

    def __getitem__(self, idx: int):
        result = {}
        if 'radar' in self.resource:
            # print(os.path.join(
            #     self.seq_path, 'radar', 'frame_{}.npy'.format(idx + self.skip_head)))
            result['radar'] = np.load(os.path.join(
                self.seq_path, 'radar', 'frame_{}.npy'.format(idx + self.skip_head)))
        if 'image' in self.resource:
            result['image'] = cv2.imread(os.path.join(
                self.seq_path, 'image', 'master', 'frame_{}.png'.format(idx + self.skip_head)))
        if 'depth' in self.resource:
            result['depth'] = np.load(os.path.join(
                self.seq_path, 'depth_pcl', 'master', 'frame_{}.npy'.format(idx + self.skip_head)))
        result['mesh'] = np.load(os.path.join(
            self.seq_path, 'mesh', 'frame_{}.npz'.format(idx + self.skip_head)))

        return result


class mmBody(Dataset):
    def __init__(self, cfg, train=True, **kwargs):
        self.data_path = cfg.data_path
        self.train = train
        if self.train:
            self.mode = "train"
        else:
            self.mode = "test"
        
        # 其他参数初始化...
        self.clip_step = kwargs.get('clip_step', 1)
        self.clip_frames = kwargs.get('clip_frames', 1)
        self.clip_range = self.clip_frames * self.clip_step
        self.output_dim = kwargs.get('output_dim', 151)
        self.skip_head = kwargs.get('skip_head', 0)
        self.skip_tail = kwargs.get('skip_tail', 0)
        
        # 修改：支持多个测试场景
        self.test_scenes = kwargs.get('test_scenes', ['lab1','lab2', 'furnished','poor_lighting','rain'])  # 改为列表'lab1', 'lab2', 'furnished','poor_lighting','rain']
        self.input_data = kwargs.get('input_data', ['radar', 'depth'])
        self.num_points = kwargs.get('num_points', 512)
        
        if train:
            seq_index = kwargs.get('seq_idxes', 20)
        else:
            seq_index = kwargs.get('seq_idxes', 2)  # 每个场景的sequence数量
        
        self.seq_idxes = seq_index
        self.features = kwargs.get('feat_dim', 3)
        self.init_index_map()
        print(self.index_map[-1])

    def init_index_map(self):
        # init the index map for each frame
        self.index_map = [0, ]
        self.seq_paths = []
        
        if self.train:
            seq_dirs = ['sequence_{}'.format(i) for i in range(self.seq_idxes)]
            self.seq_paths = [os.path.join(self.data_path, "train", p) for p in seq_dirs]
        else:
            # 遍历所有测试场景
            for scene in self.test_scenes:
                seq_dirs = ['sequence_{}'.format(i) for i in range(self.seq_idxes)]
                scene_paths = [os.path.join(self.data_path, "test", scene, p) for p in seq_dirs]
                self.seq_paths.extend(scene_paths)

        print('Data path: ', self.seq_paths)
        print(f'Total sequences: {len(self.seq_paths)}')

        self.seq_loaders = {}
        for path in self.seq_paths:
            seq_loader = mmBodySequenceLoader(path, self.skip_head, self.skip_tail, resource=self.input_data)
            self.seq_loaders.update({path: seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))


    def global_to_seq_index(self, global_idx: int):
        for seq_idx in range(len(self.index_map) - 1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx + 1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                return seq_idx, frame_idx

    def pad_data(self, data, return_choices=False):
        # pad point cloud with the fixed num of points
        if data.shape[0] > self.num_points:
            r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
        else:
            repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
            r = np.random.choice(data.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
        if return_choices:
            return data[r, :], r
        return data[r, :]

    def filter_pcl(self, bounding_pcl: np.ndarray, target_pcl: np.ndarray, bound: float = 0.2, offset: float = 0):
        """
        Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
        """
        upper_bound = bounding_pcl[:, :3].max(axis=0) + bound
        lower_bound = bounding_pcl[:, :3].min(axis=0) - bound
        lower_bound[2] += offset

        mask_x = (target_pcl[:, 0] >= lower_bound[0]) & (
                target_pcl[:, 0] <= upper_bound[0])
        mask_y = (target_pcl[:, 1] >= lower_bound[1]) & (
                target_pcl[:, 1] <= upper_bound[1])
        mask_z = (target_pcl[:, 2] >= lower_bound[2]) & (
                target_pcl[:, 2] <= upper_bound[2])
        index = mask_x & mask_y & mask_z
        return target_pcl[index]

    def load_data(self, seq_loader, idx):
        frame = seq_loader[idx]
        radar_pcl = frame['radar']
        depth_pcl = frame['depth']


        radar_pcl[:, 3:] /= np.array([5e-38, 5., 150.])

        mesh_pose = frame['mesh']['pose']
        mesh_shape = frame['mesh']['shape']
        mesh_joint = frame['mesh']['joints'][:22]

        arbe_data = self.filter_pcl(mesh_joint, radar_pcl, 0.5)

        if arbe_data.shape[0] == 0:
            # remove bad frame
            return None, None

        bbox_center = ((mesh_joint.max(axis=0) + mesh_joint.min(axis=0)) / 2)[:3]
        arbe_data[:, :3] -= bbox_center

        # padding
        arbe_data = self.pad_data(arbe_data)


        mesh_pose[:3] -= bbox_center
        # label = np.concatenate((mesh_pose, mesh_shape), axis=0)

        depth_pcl=self.pad_data(depth_pcl)
        depth_pcl[:,:3]-=bbox_center

        return torch.tensor(arbe_data)[:,:3],torch.tensor(depth_pcl)[:,:3]

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        #print(seq_idx,frame_idx)
        if(self.mode=="train"):
            mmwave_path=f"../mmWaveGen/data/mmbody/process_{self.mode}/sequence_{seq_idx}/radar/{frame_idx}.npy"
            depth_path=f"../mmWaveGen/data/mmbody/process_{self.mode}/sequence_{seq_idx}/depth/{frame_idx}.npy"

            try:
                with open(mmwave_path, 'rb') as f:
                    try:
                        mmwave_data = pickle.load(f, encoding='bytes')  # data/MMFi_Dataset/E02/S20/A09/lidar/frame234.pkl
                    except Exception as e:
                  
                        print(f)

                with open(depth_path, 'rb') as f:
                    try:
                        depth_data = pickle.load(f, encoding='bytes')  # data/MMFi_Dataset/E02/S20/A09/lidar/frame234.pkl
                    except Exception as e:
                        print(f)

            except Exception as e:
                 pass
        else:
            seq_loader = self.seq_loaders[seq_path]
            mmwave_data, depth_data = self.load_data(seq_loader, frame_idx)

        if (False and self.mode=="test" ):
            dist_matrix = torch.cdist(depth_data,depth_data)
            dist_matrix[dist_matrix==0]=float('inf')
            min_distances = torch.min(dist_matrix, dim=1)[0]
            removed_mask = min_distances > 0.1
            fliter_num=torch.sum(removed_mask)
            depth_data[removed_mask]= depth_data[torch.linspace(0, 512-fliter_num - 1, fliter_num).to(torch.long)]

        return {
            "point_cloud": depth_data.to(torch.float32),
            "mmwave_point_cloud": mmwave_data.to(torch.float32),
            "idx":idx,
        }
