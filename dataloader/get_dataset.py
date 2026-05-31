import hydra
import torch
from dataloader.MMFi_DataLoader import MMFiDataset
from dataloader.mmBody_DataLoader import mmBody
from dataloader.mRI_DataLodader import mRIDataset

def get_dataset(cfg):

    if(cfg["dataset"]["dataname"]=="mmfi"):
        train_dataset=MMFiDataset(cfg["dataset"]["opt"],mode="train")
        test_dataset=MMFiDataset(cfg["dataset"]["opt"],mode="test")
    elif(cfg["dataset"]["dataname"]=="mmbody"):
        train_dataset=mmBody(cfg["dataset"]["opt"],train=True)
        test_dataset=mmBody(cfg["dataset"]["opt"],train=False)
    elif(cfg["dataset"]["dataname"]=="mri"):
        train_dataset=mRIDataset(cfg["dataset"]["opt"],mode="train")
        test_dataset=mRIDataset(cfg["dataset"]["opt"],mode="test")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg["dataloader"]["train"]["batch_size"],
                                               shuffle=cfg["dataloader"]["train"]["shuffle"],
                                               num_workers=cfg["dataloader"]["train"]["num_workers"])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg["dataloader"]["test"]["batch_size"],
                                              shuffle=cfg["dataloader"]["test"]["shuffle"],
                                              num_workers=cfg["dataloader"]["test"]["num_workers"])
    return train_loader,train_dataset,test_loader,test_dataset

