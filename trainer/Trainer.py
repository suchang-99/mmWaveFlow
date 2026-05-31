import time
from itertools import islice
import torch
import numpy as np

import utils.universal_utils as uni_utils
import time
import sys
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import os
import pickle

class Trainer:
    def __init__(self, cfg,
                 model,
                 optimizer,
                 LossModel,
                 weight_method=None,
                 ema_decay=0.9999,
                 use_ema=False,
                 ema_start_epoch=31,
                 ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = cfg["core"]["device"]
        self.lossModel = LossModel
        self.weight_method = weight_method


    def train(self, epoch, data_loader):
        self.model.train()
        batch_len = len(data_loader)
        
        for batch_idx, data in enumerate(data_loader):
            dense_x = data['point_cloud'].cuda()
            mm_x = data['mmwave_point_cloud'].cuda()
            
            if self.cfg["dataset"]["opt"]["is_rotate"]:
                rotate_angle = (torch.rand((dense_x.shape[0], 1)).cuda() * 180) - 90
                rotate_angle[:rotate_angle.shape[0]//2] = 0
                dense_x = uni_utils.rotate_data_batch(dense_x, dim=2, angles=rotate_angle).contiguous()
                mm_x = uni_utils.rotate_data_batch(mm_x, dim=2, angles=rotate_angle).contiguous()
            
            mm_x, mean_mm = uni_utils.normal_data(mm_x)
            dense_x, mean_dense = uni_utils.normal_data(dense_x)


            loss_dict = self.model(dense_x, mm_x, mean_dense, mean_mm)
            loss = self.lossModel.calc_loss(epoch, batch_idx, batch_len, loss_dict)
            
            self.optimizer.zero_grad()
            if self.cfg["core"]["is_pcgrad"]:
                self.optimizer.pc_backward(loss)
            else:
                loss.backward()

            orig_grad_norm = clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

        self.lossModel.visual_epoch_info_and_save(epoch)

    def test(self, epoch, data_loader):
        self.model.eval()
        batch_len = len(data_loader)
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                dense_x = data['point_cloud'].cuda()
                mm_x = data['mmwave_point_cloud'].cuda()

                mm_x, mean_mm = uni_utils.normal_data(mm_x)
                dense_x, mean_dense = uni_utils.normal_data(dense_x)

                sample_result = self.model.sample(dense_x, mm_x, mean_dense, mean_mm, batch_idx=batch_idx)
                del sample_result["results"]
                self.lossModel.visualize_loss(epoch, batch_idx, batch_len, sample_result["cd"], sample_result, is_test=True)

            self.lossModel.visual_epoch_info_and_save(epoch, self.model, is_test=True)
