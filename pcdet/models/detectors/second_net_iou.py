import torch
from .detector3d_template import Detector3DTemplate
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
import pdb
import time
class SECONDNetIoU(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        
        dense_head_cfg = model_cfg['DENSE_HEAD']

        if dense_head_cfg.get('ONLY_OPTIM_GROUP', None):
            self.only_optim_group = dense_head_cfg['ONLY_OPTIM_GROUP']
        else:
            self.only_optim_group = False

        if self.training == False:
            self.only_optim_group = False
    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        module_cnt = 0
        for cur_module in self.module_list:
            
            #if self.training and self.only_optim_group and module_cnt == len(self.module_list)-1:
            #    continue
            batch_dict = cur_module(batch_dict)
            module_cnt += 1
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if 'num_group' in batch_dict:
                
                pred_dicts, recall_dicts = self.post_processing_multicriterion_group(batch_dict)
                #pred_dicts, recall_dicts = self.post_processing_multicriterion(batch_dict)
                
            else:
                pred_dicts, recall_dicts = self.post_processing_multicriterion(batch_dict)
          
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rpn + loss_rcnn
        
        
        return loss, tb_dict, disp_dict
