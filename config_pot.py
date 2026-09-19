"""AXIA config for the ISPRS Potsdam dataset (RGB + DSM, 6 classes).

Note: mIoU follows the ISPRS convention and averages over the first 5
classes, excluding clutter (``C.num_eval_classes = 5``).
"""
import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 3407

"""GPU Device Config"""
C.gpu_id = 0
C.gpu_devices = str(C.gpu_id)

C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))

"""Dataset Path"""
C.dataset_name = 'Potsdam'
# TODO: point this to your local copy of the dataset.
C.dataset_path = './datasets/Potsdam'
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = False          
C.x_root_folder = osp.join(C.dataset_path, 'DSM')
C.x_format = '.png'
C.x_is_single_channel = True
C.train_source = osp.join(C.dataset_path, 'train.txt')
C.eval_source = osp.join(C.dataset_path, 'test.txt')
C.num_train_imgs = 7200
C.num_eval_imgs = 2400
C.num_classes = 6
C.class_names = ['impervious', 'building', 'low_vegetation', 'tree', 'car', 'clutter']
C.num_eval_classes = 5  

# Potsdam/Vaihingen standard color-palette labels (BGR; index == class id).
# Corresponding RGB: white = impervious, blue = building, cyan = low vegetation,
# green = tree, yellow = car, red = clutter/background.
C.gt_color_palette = [
    (255, 255, 255),  # 0: impervious surfaces
    (255,   0,   0),  # 1: building
    (255, 255,   0),  # 2: low vegetation
    (  0, 255,   0),  # 3: tree
    (  0, 255, 255),  # 4: car
    (  0,   0, 255),  # 5: clutter/background
]

"""Image Config"""
C.background = 255
C.image_height = 300
C.image_width = 300
C.rgb_norm_mean = np.array([0.485, 0.456, 0.406])
C.rgb_norm_std = np.array([0.229, 0.224, 0.225])
# DSM statistics from the training split (single channel, replicated to 3).
C.x_norm_mean = np.array([0.185, 0.185, 0.185])
C.x_norm_std = np.array([0.222, 0.222, 0.222])

"""Network Settings"""
C.backbone = 'dual_encoder'
# TODO: download the ImageNet-pretrained Swin checkpoints (see README).
C.pretrained_model = './pretrained/swin_large_patch4_window7_224_22k.pth'
C.sar_pretrained_model = './pretrained/swin_tiny_patch4_window7_224.pth'
C.decoder = 'SARD'
C.decoder_channels = 512
C.aux_head = 'FCNHead'          # None | 'FCNHead' | 'AFDAH'
C.aux_head_embed_dim = 192      # AFDAH internal dim (only used when aux_head='AFDAH')
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 5e-5
C.adapter_lr = 1e-4
C.decoder_lr = 2e-4
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.grad_clip = 1.0
C.batch_size = 8
C.nepochs = 200
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 4
C.train_scale_array = [0.75, 1, 1.25, 1.5]
C.warm_up_epoch = 10

"""Eval Config"""
C.eval_flip = False
C.eval_batch_size = 12

"""Validation Strategy Config"""
C.val_early_epochs = 10
C.val_early_frequency = 5
C.val_late_frequency = 1

"""Resume Config"""
C.resume_from = None

"""Path Config"""
C.log_dir = osp.abspath('log_' + C.dataset_name)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))
