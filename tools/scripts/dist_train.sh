#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file ./cfgs/da-nuscenes-kitti_models/secondiou_st3d/group_exp/fixed_group_pretrain_subset.yaml --pretrained_model ../secondiou_sn_car_pretrain.pth

