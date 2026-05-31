


import hydra

from omegaconf import DictConfig, OmegaConf
# from rectified_flow_pytorch import RectifiedFlow, Unet, Trainer
import platform
from dataloader.get_dataset import get_dataset
import time
import numpy as np
import torch
import random
import utils.visual_utils as visual_utils
from model.mWGModel import mWGModel
from collections import defaultdict

from calflops import calculate_flops
from loss.loss import LossModel

from trainer.Trainer import Trainer
from utils.pcgrad import PCGrad
import os, shutil, datetime, tempfile



def backup_files(file_list, backup_dir='./exp_backups'):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dst_root = os.path.join(backup_dir, f'code_{ts}')
    os.makedirs(dst_root, exist_ok=True)

    for f in file_list:
        name = os.path.basename(f)          # 只取文件名/目录名
        dst = os.path.join(dst_root, name)
        if os.path.isfile(f):
            shutil.copy2(f, dst)
        else:
            shutil.copytree(f, dst)
    print(f'[Backup] 已复制 → {dst_root}')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def train(cfg: DictConfig) -> None:
    print(cfg["core"])
    setup_seed(7)

    train_dataloader,train_dataset,test_dataloader,test_dataset=get_dataset(cfg)

    my_net=mWGModel(cfg).cuda()

    my_lossModel=LossModel(cfg)

   
    if(cfg["core"]["is_pcgrad"]):
        optimizer = PCGrad(torch.optim.AdamW(my_net.parameters(),
                                             lr=1e-4, # 2e-4 with cos sc
                                             ))
    else:
        optimizer = torch.optim.AdamW(my_net.parameters(),
                                     lr=1e-4,
                                     )



    myTrainer=Trainer(cfg=cfg,
                      model=my_net,
                      optimizer=optimizer,
                      LossModel=my_lossModel,
                      )

    for epoch in range(0,300):

        myTrainer.train(epoch,train_dataloader)

        if(epoch%1==0):
            myTrainer.test(epoch,test_dataloader)

        if(epoch==cfg['dataset']['opt']['joint_train_epoch']):

            myTrainer.model.prepare_for_train_flow()

            for param in my_net.encoder.parameters():
                param.requires_grad = False
            for param in my_net.mmencoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW([
               # {'params': my_net.encoder.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
              #  {'params': my_net.decoder.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
              #  {'params': my_net.mmencoder.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
             #   {'params': my_net.mmdecoder.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
                {'params': my_net.cross_attention.parameters(), 'lr': cfg['dataset']['opt']['two_stage_lr']*0.1, 'weight_decay': 0.01},
                {'params': my_net.flow.parameters(), 'lr': cfg['dataset']['opt']['two_stage_lr'], 'weight_decay': 0.01},
            ], lr=1e-4)
                
            myTrainer.model=my_net
            myTrainer.optimizer=optimizer
            myTrainer.cfg["core"]["is_joint"]=False
            myTrainer.cfg["core"]["is_pcgrad"]=False


        if(epoch==0):
            backup_files([
                'train.py',
                'model/mWGModel.py',
                'model/FlowModel.py',
                'configs/config.yaml',
                'trainer/Trainer.py',
                'loss/loss.py'
            ])







@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    if(platform.system()=="Windows"):
        cfg.core.is_debug = True
    #print(OmegaConf.to_yaml(cfg)) # 打印配置
    train(cfg)


if __name__ == "__main__":
    my_app()