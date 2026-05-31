
import torch
import numpy as np

import sys
from collections import defaultdict
import os

class LossModel():

    def __init__(self,cfg):
        self.cfg=cfg
        self.recoder=defaultdict(list)
        self.max_error=100
        self.save_path = ""
        self.epoch_save_path = ""
        self.key_weight={
            "z_kl":cfg['dataset']['opt']['z_kl'], # 0.03 0.01
            "z_flow":cfg['dataset']['opt']['z_flow'],
            "pc_re":cfg['dataset']['opt']['pc_re'],
            "mm_re":cfg['dataset']['opt']['mm_re'], #120
            "em":cfg['dataset']['opt']['em'],
            "mmem":cfg['dataset']['opt']['mmem'],  #10 5 4
            "mcl":cfg['dataset']['opt']['mcl'],
            "reg":cfg['dataset']['opt']['reg'],
        }
        print(self.key_weight)

    def visualize_loss(self, epoch, batch_idx, total_batch, loss, all_loss_dict,is_test=False):

        if(is_test):
            output_str = '\rT:%d:[%d / %d], ' % (epoch, batch_idx + 1, total_batch)
            output_str += f'loss: %.4f' % loss
        else:
            output_str = '\r%d:[%d / %d], ' % (epoch, batch_idx + 1, total_batch)
            output_str += f'loss: %.4f' % loss


        for loss_name in all_loss_dict.keys():
            output_str += ', '
            try:
                output_str += f'{loss_name}: %.3f' % all_loss_dict[loss_name]

                self.recoder[loss_name].append(all_loss_dict[loss_name].cpu().detach().numpy())
            except Exception as e:
                continue
        sys.stdout.write(output_str)
        sys.stdout.flush()

    def save_model(self,mean_kp,epoch,model):
        now_mode = "gen"
        if (mean_kp < self.max_error):
            self.max_error = mean_kp
            try:
                os.remove(self.save_path)
            except Exception:
                pass

            self.save_path = "ckpt/" +now_mode+"_"+self.cfg["dataset"]["dataname"]+"_"+ str(epoch) + "_" + str(round(mean_kp, 3)) + ".pkl"
            torch.save(model, self.save_path)
            print("Save model")
        if (epoch % 10 == 0):
            try:
                os.remove(self.epoch_save_path)
            except Exception:
                pass
            self.epoch_save_path = "ckpt/"+str(epoch)+now_mode+"_"+self.cfg["dataset"]["dataname"]+"_"+ str(epoch) + "_" + str(round(mean_kp, 3)) + ".pkl"
            torch.save(model, self.epoch_save_path)

    def visual_epoch_info_and_save(self,epoch,model=None,is_test=False):

        output_str = f'\nSummary '
        for loss_name in self.recoder.keys():
            output_str += ', '
            try:
                output_str += f'{loss_name}: %.5f' % np.mean(self.recoder[loss_name])
                # self.recoder[loss_name]=[]
            except Exception as e:
                continue
        output_str+="\n"
        sys.stdout.write(output_str)
        sys.stdout.flush()

        if(is_test):
            mean_metric=np.mean(self.recoder["cd"])
            self.save_model(mean_metric,epoch,model)
        self.recoder = defaultdict(list)


    def calc_loss(self,epoch,batch_idx,total_batch,input_dict,is_test=False):

        bp_loss =(input_dict["z_kl"] *self.key_weight["z_kl"]+
                  input_dict["z_flow"]*self.key_weight["z_flow"] +   #1
                  input_dict["pc_re"]*self.key_weight["pc_re"] +  #10
                  input_dict["mm_re"]*self.key_weight["mm_re"]+ #50
                  input_dict["em"]*self.key_weight["em"]+        #1
                  input_dict["mmem"]*self.key_weight["mmem"] + #10
                  input_dict["mcl"]*self.key_weight["mcl"]+    #1
                  input_dict['reg']*self.key_weight["reg"] #1e-4
                  )

        self.recoder["loss"].append(bp_loss.cpu().detach().numpy())
        self.visualize_loss(epoch,batch_idx,total_batch,bp_loss,input_dict,is_test)

        if(self.cfg["core"]["is_pcgrad"]):
            pcgrad_loss=[input_dict["z_kl"]*self.key_weight["z_kl"] ,
                     input_dict["z_flow"]*self.key_weight["z_flow"] ,
                     input_dict["pc_re"]*self.key_weight["pc_re"] ,
                     input_dict["mm_re"]*self.key_weight["mm_re"], #120
                     input_dict["em"]*self.key_weight["em"],
                     input_dict["mmem"]*self.key_weight["mmem"] , #10
                     input_dict["mcl"]*self.key_weight["mcl"],
                     input_dict['reg']*self.key_weight["reg"]]
            return pcgrad_loss

        return bp_loss



