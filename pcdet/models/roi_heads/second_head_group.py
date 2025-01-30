import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, loss_utils
import pdb
import time
class SECONDHeadGroup(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = self.model_cfg.ROI_GRID_POOL.IN_CHANNEL * GRID_SIZE * GRID_SIZE

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        shared_fc_list2 = []
        pre_channel = self.model_cfg.ROI_GRID_POOL.IN_CHANNEL * GRID_SIZE * GRID_SIZE
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list2.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list2.append(nn.Dropout(self.model_cfg.DP_RATIO))

        if model_cfg.get('MULTI_RPN', None):
            self.multi_rpn = True
        else:
            self.multi_rpn = False
        
        if self.multi_rpn:
            self.group_models = nn.ModuleDict()
            self.group_models['shared_fc_layer'] = nn.Sequential(*shared_fc_list2)
            self.group_models['iou_layer'] = self.make_fc_layers(
                    input_channels=pre_channel, output_channels=1, fc_list=self.model_cfg.IOU_FC
                )
        if model_cfg.get('USE_GROUP_UNTIL_EPOCH', None):
            self.group_epoch = model_cfg['USE_GROUP_UNTIL_EPOCH']
        else:
            self.group_epoch = 10000

        if model_cfg.get('ONLY_OPTIM_GROUP', None):
            self.only_optim_group = model_cfg['ONLY_OPTIM_GROUP']
        else:
            self.only_optim_group = False

        if model_cfg.get('ONLY_OPTIM_ORI', None):
            self.only_optim_ori = model_cfg['ONLY_OPTIM_ORI']
        else:
            self.only_optim_ori = False
        
        if self.training == False:
            self.only_optim_group = False
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.iou_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=1, fc_list=self.model_cfg.IOU_FC
        )
        self.init_weights(weight_init='xavier')
        
    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                spatial_features_2d: (B, C, H, W)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'].detach()
        spatial_features_2d = batch_dict['spatial_features_2d'].detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)

        dataset_cfg = batch_dict['dataset_cfg']
        min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
        min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
        voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
        voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
        down_sample_ratio = self.model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO

        pooled_features_list = []
        #torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            # Map global boxes coordinates to feature map coordinates
            x1 = (rois[b_id, :, 0] - rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            x2 = (rois[b_id, :, 0] + rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            y1 = (rois[b_id, :, 1] - rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            y2 = (rois[b_id, :, 1] + rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)

            angle, _ = common_utils.check_numpy_to_torch(rois[b_id, :, 6])

            cosa = torch.cos(angle)
            sina = torch.sin(angle)

            theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            grid = nn.functional.affine_grid(
                theta,
                torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size)),
                align_corners=True
            )

            pooled_features = nn.functional.grid_sample(
                spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width),
                grid, align_corners=True
            )

            pooled_features_list.append(pooled_features)

        torch.backends.cudnn.enabled = True
        pooled_features = torch.cat(pooled_features_list, dim=0)

        return pooled_features

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        
        if 'pseudo_collection' in batch_dict:
            
            if batch_dict['cur_epoch'] <= self.group_epoch:
                self.only_optim_ori = False
            else:
                self.only_optim_ori = True #We do not use group after certain epoch for pseudo label collection
        else:# In case it's training or testing
            if 'st' in batch_dict: #self training
                if batch_dict['cur_epoch'] <= self.group_epoch and batch_dict['source'] == False: 
                    self.only_optim_ori = False
                else:
                    self.only_optim_ori = True
                
            else: #source training or testing
                if self.training: 
                    self.only_optim_ori = False
                else:
                    self.only_optim_ori = True 

        num_group = batch_dict['num_group']
        
        for group_idx in range(num_group+1):
            
            if group_idx == 0:
                if self.training and self.only_optim_group :
                    self.forward_ret_dict = {}
                    continue
                new_dict = {
                    'batch_size': batch_dict['batch_size'],
                    'batch_box_preds': batch_dict['batch_box_preds'],
                    'batch_cls_preds': batch_dict['batch_cls_preds'],
                    'spatial_features_2d': batch_dict['spatial_features_2d'],
                    'dataset_cfg': batch_dict['dataset_cfg'],
                    'gt_boxes': batch_dict['gt_boxes']
                }
                
            else:
                if self.only_optim_ori:
                    continue
                new_dict = {
                        'batch_size': batch_dict['batch_size'],
                        'batch_box_preds': batch_dict['batch_box_preds{0}'.format(group_idx)],
                        'batch_cls_preds': batch_dict['batch_cls_preds{0}'.format(group_idx)],
                        'spatial_features_2d': batch_dict['spatial_features_2d{0}'.format(group_idx)],
                        'dataset_cfg': batch_dict['dataset_cfg'],
                        
                    }
                if self.training:
                    new_dict['gt_boxes'] = batch_dict['gt_boxes{0}'.format(group_idx)]
            
            
            targets_dict = self.proposal_layer(
                new_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )
            
            if self.training:
                targets_dict = self.assign_targets(new_dict)
                new_dict['rois'] = targets_dict['rois']
                new_dict['roi_labels'] = targets_dict['roi_labels']
            
            # RoI aware pooling
            pooled_features = self.roi_grid_pool(new_dict)  # (BxN, C, 7, 7)
            batch_size_rcnn = pooled_features.shape[0]
            
            if self.multi_rpn and group_idx > 0:
                
                shared_features = self.group_models['shared_fc_layer'](pooled_features.view(batch_size_rcnn, -1, 1))
                rcnn_iou = self.group_models['iou_layer'](shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*N, 1)
            else:
                
                shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
                rcnn_iou = self.iou_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*N, 1)
            
            if not self.training:
                if group_idx == 0:
                    batch_dict['batch_cls_preds'] = rcnn_iou.view(new_dict['batch_size'], -1, rcnn_iou.shape[-1])
                    batch_dict['batch_box_preds'] = new_dict['rois']
                    batch_dict['roi_scores'] = new_dict['roi_scores']
                    batch_dict['roi_labels'] = new_dict['roi_labels']
                    batch_dict['rois'] = new_dict['rois']
                    batch_dict['cls_preds_normalized'] = False
                else:
                    
                    batch_dict['batch_cls_preds{0}'.format(group_idx)] = rcnn_iou.view(new_dict['batch_size'], -1, rcnn_iou.shape[-1])
                    batch_dict['batch_box_preds{0}'.format(group_idx)] = new_dict['rois']
                    batch_dict['roi_scores{0}'.format(group_idx)] = new_dict['roi_scores']
                    batch_dict['roi_labels{0}'.format(group_idx)] = new_dict['roi_labels']
                    batch_dict['rois{0}'.format(group_idx)] = new_dict['rois']
            else:
                if group_idx == 0:
                    targets_dict['rcnn_iou'] = rcnn_iou
                    self.forward_ret_dict = targets_dict
                else:
                    targets_dict['rcnn_iou'] = rcnn_iou
                    self.forward_ret_dict['targets_dict{0}'.format(group_idx)] = targets_dict
        if self.training:
            self.forward_ret_dict['group'] = num_group
        
        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        if self.only_optim_group == False:
            rcnn_loss_cls, cls_tb_dict = self.get_box_iou_layer_loss(self.forward_ret_dict, 0)
            rcnn_loss += rcnn_loss_cls
            tb_dict.update(cls_tb_dict)

        if self.only_optim_ori == False and 'group' in self.forward_ret_dict:

            for idx_group in range(self.forward_ret_dict['group']):
                
                rcnn_loss_cls, cls_tb_dict = self.get_box_iou_layer_loss(self.forward_ret_dict['targets_dict{0}'.format(idx_group+1)], idx_group + 1)
                rcnn_loss += rcnn_loss_cls
                tb_dict.update(cls_tb_dict)
               
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        
        return rcnn_loss, tb_dict

    def get_box_iou_layer_loss(self, forward_ret_dict, idx):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_iou = forward_ret_dict['rcnn_iou']
        rcnn_iou_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_iou_flat = rcnn_iou.view(-1)
        if loss_cfgs.IOU_LOSS == 'BinaryCrossEntropy':
            batch_loss_iou = nn.functional.binary_cross_entropy_with_logits(
                rcnn_iou_flat,
                rcnn_iou_labels.float(), reduction='none'
            )
        elif loss_cfgs.IOU_LOSS == 'L2':
            batch_loss_iou = nn.functional.mse_loss(rcnn_iou_flat, rcnn_iou_labels, reduction='none')
        elif loss_cfgs.IOU_LOSS == 'smoothL1':
            diff = rcnn_iou_flat - rcnn_iou_labels
            batch_loss_iou = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(diff, 1.0 / 9.0)
        elif loss_cfgs.IOU_LOSS == 'focalbce':
            batch_loss_iou = loss_utils.sigmoid_focal_cls_loss(rcnn_iou_flat, rcnn_iou_labels)
        else:
            raise NotImplementedError

        iou_valid_mask = (rcnn_iou_labels >= 0).float()
        rcnn_loss_iou = (batch_loss_iou * iou_valid_mask).sum() / torch.clamp(iou_valid_mask.sum(), min=1.0)

        rcnn_loss_iou = rcnn_loss_iou * loss_cfgs.LOSS_WEIGHTS['rcnn_iou_weight']
        tb_dict = {'rcnn_loss_iou{0}'.format(idx): rcnn_loss_iou.item()}
        return rcnn_loss_iou, tb_dict


    def get_box_iou_layer_loss_group(self, forward_ret_dict, idx):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_iou = forward_ret_dict['rcnn_iou{0}'.format(idx)]
        rcnn_iou_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_iou_flat = rcnn_iou.view(-1)
        if loss_cfgs.IOU_LOSS == 'BinaryCrossEntropy':
            batch_loss_iou = nn.functional.binary_cross_entropy_with_logits(
                rcnn_iou_flat,
                rcnn_iou_labels.float(), reduction='none'
            )
        elif loss_cfgs.IOU_LOSS == 'L2':
            batch_loss_iou = nn.functional.mse_loss(rcnn_iou_flat, rcnn_iou_labels, reduction='none')
        elif loss_cfgs.IOU_LOSS == 'smoothL1':
            diff = rcnn_iou_flat - rcnn_iou_labels
            batch_loss_iou = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(diff, 1.0 / 9.0)
        elif loss_cfgs.IOU_LOSS == 'focalbce':
            batch_loss_iou = loss_utils.sigmoid_focal_cls_loss(rcnn_iou_flat, rcnn_iou_labels)
        else:
            raise NotImplementedError

        iou_valid_mask = (rcnn_iou_labels >= 0).float()
        rcnn_loss_iou = (batch_loss_iou * iou_valid_mask).sum() / torch.clamp(iou_valid_mask.sum(), min=1.0)

        rcnn_loss_iou = rcnn_loss_iou * loss_cfgs.LOSS_WEIGHTS['rcnn_iou_weight']
        tb_dict = {'rcnn_loss_iou': rcnn_loss_iou.item()}
        return rcnn_loss_iou, tb_dict