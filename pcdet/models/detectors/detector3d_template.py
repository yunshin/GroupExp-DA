import torch
import os
import torch.nn as nn
from .. import backbones_3d, backbones_2d, dense_heads, roi_heads
from ..backbones_3d import vfe, pfe
from ..backbones_2d import map_to_bev
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.iou3d_nms import iou3d_nms_utils
from pcdet.config import cfg
import pdb

class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            cls_preds = batch_dict['batch_cls_preds'][batch_mask]

            src_cls_preds = cls_preds
            src_box_preds = box_preds
            assert cls_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels', False) else label_preds + 1

                
                
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.sum() > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds, cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois, cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict
    def load_params_from_file_train(self, filename, logger, to_cpu=False, state_name='model_state'):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint[state_name]

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])
        
        mapping = {
            'dense_head.models.cls0.weight': 'dense_head.conv_cls.weight',
            'dense_head.models.cls0.bias': 'dense_head.conv_cls.bias',
            'dense_head.models.box0.weight': 'dense_head.conv_box.weight',
            'dense_head.models.box0.bias': 'dense_head.conv_box.bias',
            'dense_head.models.dir0.weight': 'dense_head.conv_dir_cls.weight',
            'dense_head.models.dir0.bias': 'dense_head.conv_dir_cls.bias',

        }

        update_model_state = {}
        new_val_dict = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            
            elif key in mapping:
                new_val_dict[mapping[key]] = val
            else:
                print(key)  

        for key in new_val_dict:

            model_state_disk[key] = new_val_dict[key]
       
        
        if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None): ##Strict loading
            self.load_state_dict(model_state_disk, strict=False)
        else:
            state_dict = self.state_dict()
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)

            for key in state_dict:
                if key not in update_model_state:
                    logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
       
        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))


   
    def load_params_from_file_group(self, filename, logger, to_cpu=False, state_name='model_state'):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint[state_name]

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}

        dense_head_state_group = {}
        for key, val in model_state_disk.items():
            

            if 'dense_head' in key:
                dense_head_state_group[key] = val
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()

        for key, val in state_dict.items():
            
            if 'dense_head.conv_cls' in key:
                
                if '.weight' in key:
                    update_model_state[key] = dense_head_state_group['dense_head.conv_cls.weight']
                elif '.bias' in key:
                    update_model_state[key] = dense_head_state_group['dense_head.conv_cls.bias']

            elif 'dense_head.conv_box' in key:
                
                if '.weight' in key:
                    update_model_state[key] = dense_head_state_group['dense_head.conv_box.weight']
                elif '.bias' in key:
                    update_model_state[key] = dense_head_state_group['dense_head.conv_box.bias']

            elif 'dense_head.conv_dir_cls' in key:
                
                if '.weight' in key:
                    update_model_state[key] = dense_head_state_group['dense_head.conv_dir_cls.weight']
                elif '.bias' in key:
                    update_model_state[key] = dense_head_state_group['dense_head.conv_dir_cls.bias']

        
        if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
            #self.load_state_dict(model_state_disk)

            state_dict = self.state_dict()
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)

            for key in state_dict:
                if key not in update_model_state:
                    logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

            
        else:
            state_dict = self.state_dict()
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)

            for key in state_dict:
                if key not in update_model_state:
                    logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))


        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_from_file(self, filename, logger, to_cpu=False, state_name='model_state'):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint[state_name]

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        
        if 'dense_head.models.cls0.weight' in self.state_dict():
            use_group = True
        else:
            use_group= False

        if use_group and 'dense_head.group_models.cls.weight' in self.state_dict():
            use_multi_rpn = True
        else:
            use_multi_rpn = False
        
        if use_group and 'dense_head.group_models.attn.layers.0.key_proj.weight' in model_state_disk.keys():
            load_from_group = True # this means its the start of the training
        else:
            load_from_group = False
        if load_from_group and 'dense_head.group_models.cls.weight' in model_state_disk.keys():
            exist_multi = True
        else:
            exist_multi = False
        
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            
            if load_from_group == False and use_group and 'roi_head' in key:


                
                if 'shared_fc_layer' in key:
                    
                    key_name = key[24:]
                    update_model_state['roi_head.group_models.shared_fc_layer' + key_name] = val
                    #model_state_disk['roi_head.group_models.shared_fc_layer' + key_name] = val
                    print('roi_head.group_models.shared_fc_layer' + key_name)
                if 'iou_layers' in key:
                    
                    key_name = key[19:]
                    update_model_state['roi_head.group_models.iou_layer' + key_name] = val
                    #model_state_disk['roi_head.group_models.iou_layer' + key_name] = val
            if load_from_group == False and use_group and 'dense_head' in key:

                if 'conv_cls.weight' in key:
                    
                    update_model_state['dense_head.models.cls0.weight'] = val
                    #model_state_disk['dense_head.models.cls0.weight'] = val
                elif 'conv_cls.bias' in key:
                    update_model_state['dense_head.models.cls0.bias'] = val
                    #model_state_disk['dense_head.models.cls0.bias'] = val

                if 'conv_box.weight' in key:
                    update_model_state['dense_head.models.box0.weight'] = val
                    #model_state_disk['dense_head.models.box0.weight'] = val
                elif 'conv_box.bias' in key:
                    update_model_state['dense_head.models.box0.bias'] = val
                    #model_state_disk['dense_head.models.box0.bias'] = val

                if 'conv_dir_cls.weight' in key:
                    update_model_state['dense_head.models.dir0.weight'] = val
                    #model_state_disk['dense_head.models.dir0.weight'] = val
                elif 'conv_dir_cls.bias' in key:
                    update_model_state['dense_head.models.dir0.bias'] = val
                    #model_state_disk['dense_head.models.dir0.bias'] = val
                if use_multi_rpn:
                    if 'conv_cls.weight' in key:

                        update_model_state['dense_head.group_models.cls.weight'] = val
                        #model_state_disk['dense_head.group_models.cls.weight'] = val
                    elif 'conv_cls.bias' in key:
                        update_model_state['dense_head.group_models.cls.bias'] = val
                        #model_state_disk['dense_head.group_models.cls.bias'] = val

                    if 'conv_box.weight' in key:
                        update_model_state['dense_head.group_models.box.weight'] = val
                        #model_state_disk['dense_head.group_models.box.weight'] = val
                    elif 'conv_box.bias' in key:
                        update_model_state['dense_head.group_models.box.bias'] = val
                        #model_state_disk['dense_head.group_models.box.bias'] = val

                    if 'conv_dir_cls.weight' in key:
                        update_model_state['dense_head.group_models.dir.weight'] = val
                        #model_state_disk['dense_head.group_models.dir.weight'] = val
                    elif 'conv_dir_cls.bias' in key:
                        update_model_state['dense_head.group_models.dir.bias'] = val
                        #model_state_disk['dense_head.group_models.dir.bias'] = val
            if load_from_group and use_group and exist_multi == False and use_multi_rpn:
                
                if 'cls0.weight' in key:
                    
                    update_model_state['dense_head.group_models.cls.weight'] = val
                    #model_state_disk['dense_head.group_models.cls.weight'] = val
                elif 'cls0.bias' in key:
                    update_model_state['dense_head.group_models.cls.bias'] = val
                    #model_state_disk['dense_head.group_models.cls.bias'] = val  
                

                if 'box0.weight' in key:
                    update_model_state['dense_head.group_models.box.weight'] = val
                    #model_state_disk['dense_head.group_models.box.weight'] = val
                elif 'box0.bias' in key:
                    update_model_state['dense_head.group_models.box.bias'] = val
                    #model_state_disk['dense_head.group_models.box.bias'] = val

                if 'dir0.weight' in key:
                    update_model_state['dense_head.group_models.dir.weight'] = val
                    #model_state_disk['dense_head.group_models.dir.weight'] = val
                elif 'dir0.bias' in key:
                    update_model_state['dense_head.group_models.dir.bias'] = val
                    #model_state_disk['dense_head.group_models.dir.bias'] = val
        if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
            
            for key, val in update_model_state.items():
                model_state_disk[key] = val
            self.load_state_dict(model_state_disk, strict=False)
        else:
            
            state_dict = self.state_dict()
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict, strict=False)

            for key in state_dict:
                if key not in update_model_state:
                    logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
    def load_params_from_file_test(self, filename, logger, to_cpu=False, state_name='model_state'):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint[state_name]

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        
       
        
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
           
        if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
            
            for key, val in update_model_state.items():
                model_state_disk[key] = val
            self.load_state_dict(model_state_disk, strict=True)
        else:
            
            state_dict = self.state_dict()
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict, strict=True)

            for key in state_dict:
                if key not in update_model_state:
                    logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))   
    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                except:
                    print(' Optimizer state dict not loaded')
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    def post_processing_multicriterion(self, batch_dict):
        """
        For 
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
           
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            iou_preds = batch_dict['batch_cls_preds'][batch_mask]
            cls_preds = batch_dict['roi_scores'][batch_mask]

            if isinstance(cls_preds, list):
                cls_preds = torch.cat(cls_preds).squeeze()
            else:
                cls_preds = cls_preds.squeeze()

            src_iou_preds = iou_preds
            src_box_preds = box_preds
            src_cls_preds = cls_preds
            assert iou_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                iou_preds = torch.sigmoid(iou_preds)
                cls_preds = torch.sigmoid(cls_preds)

            # TODO
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                iou_preds, label_preds = torch.max(iou_preds, dim=-1)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels',
                                                                                False) else label_preds + 1
                if isinstance(label_preds, list):
                    label_preds = torch.cat(label_preds, dim=0)

                if post_process_cfg.NMS_CONFIG.get('SCORE_WEIGHTS', None):
                    weight_iou = post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.iou
                    weight_cls = post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.cls

                if post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) == 'iou' or \
                        post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) is None:
                    nms_scores = iou_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'cls':
                    nms_scores = cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'hybrid_iou_cls':
                    assert weight_iou + weight_cls == 1
                    nms_scores = weight_iou * iou_preds + \
                                 weight_cls * cls_preds
                else:
                    raise NotImplementedError

                selected, selected_scores = class_agnostic_nms(
                    box_scores=nms_scores, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    raise NotImplementedError

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
                'pred_cls_scores': cls_preds[selected],
                'pred_iou_scores': iou_preds[selected]
            }

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict



    def post_processing_multicriterion_group(self, batch_dict):
        """
        For 
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:

        """
        num_group = batch_dict['num_group']

        
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
           

            
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            iou_preds = batch_dict['batch_cls_preds'][batch_mask]
            cls_preds = batch_dict['roi_scores'][batch_mask]

            for group_idx in range(num_group):
                
                box_preds_group = batch_dict['batch_box_preds{0}'.format(group_idx+1)][batch_mask]
                iou_preds_group = batch_dict['batch_cls_preds{0}'.format(group_idx+1)][batch_mask]
                cls_preds_group = batch_dict['roi_scores{0}'.format(group_idx+1)][batch_mask]

                box_preds = torch.cat([box_preds, box_preds_group],0)
                iou_preds = torch.cat([iou_preds, iou_preds_group],0)
                cls_preds = torch.cat([cls_preds, cls_preds_group],0)
            
            if 'test_only_baseline' in batch_dict:
                
                box_preds = box_preds[:100]
                iou_preds = iou_preds[:100]
                cls_preds = cls_preds[:100]
                
            if 'test_all' in batch_dict:
                box_preds = box_preds[:]
                iou_preds = iou_preds[:]
                cls_preds = cls_preds[:]
            if 'test_only_group' in batch_dict:
                box_preds = box_preds[100:]
                iou_preds = iou_preds[100:]
                cls_preds = cls_preds[100:]
            
            
            if 'pseudo_collection' not in batch_dict:
                
                box_preds = box_preds[:100]
                iou_preds = iou_preds[:100]
                cls_preds = cls_preds[:100]
            
            if isinstance(cls_preds, list):
                cls_preds = torch.cat(cls_preds).squeeze()
            else:
                cls_preds = cls_preds.squeeze()

            src_iou_preds = iou_preds
            src_box_preds = box_preds
            src_cls_preds = cls_preds
            assert iou_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                iou_preds = torch.sigmoid(iou_preds)
                cls_preds = torch.sigmoid(cls_preds)

            # TODO
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
               
                iou_preds, label_preds = torch.max(iou_preds, dim=-1)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels', False) else label_preds + 1

                if batch_dict.get('has_class_labels', False):
                    pdb.set_trace()
                '''
                for group_idx in range(num_group):

                    label_preds_group = batch_dict['roi_labels{0}'.format(group_idx+1)][index] if batch_dict.get('has_class_labels', False) else label_preds + 1
                    label_preds = torch.cat([label_preds, label_preds_group])

                pdb.set_trace()
                '''
                if isinstance(label_preds, list):
                    label_preds = torch.cat(label_preds, dim=0)

                if post_process_cfg.NMS_CONFIG.get('SCORE_WEIGHTS', None):
                    weight_iou = post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.iou
                    weight_cls = post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.cls

                if post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) == 'iou' or \
                        post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) is None:
                    nms_scores = iou_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'cls':
                    nms_scores = cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'hybrid_iou_cls':
                    assert weight_iou + weight_cls == 1
                    nms_scores = weight_iou * iou_preds + \
                                 weight_cls * cls_preds
                else:
                    raise NotImplementedError
                
                group_indicator = torch.zeros((num_group+1)*100).long().cuda()
                for group_idx in range(num_group):
                    group_indicator[(group_idx+1)*100:(group_idx+2)*100] = group_idx + 1

    
                selected, selected_scores = class_agnostic_nms(
                    box_scores=nms_scores, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
                      
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    raise NotImplementedError
                
                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
                group_indicator = group_indicator[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
                'pred_cls_scores': cls_preds[selected],
                'pred_iou_scores': iou_preds[selected],
                'group_indicator': group_indicator
            }

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict