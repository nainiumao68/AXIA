"""AXIA config for the CART dataset (RGB + Thermal, 10 classes)."""
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
C.dataset_name = 'CART'
# TODO: point this to your local copy of the dataset.
C.dataset_path = './datasets/CART'
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.png'
C.x_root_folder = osp.join(C.dataset_path, 'T')
C.x_format = '.png'
C.x_is_single_channel = True
C.gt_root_folder = osp.join(C.dataset_path, 'Labels')
C.gt_format = '.png'
C.gt_transform = True
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.num_train_imgs = 1731
C.num_eval_imgs = 279
C.num_classes = 10
C.class_names = [
    'Water', 'Bare Ground', 'Rocky Terrain', 'Shrubs', 'Sky',
    'Trees', 'Devel Struct', 'Road', 'Vehicles', 'Person'
]

# CART label mapping (raw label value -> training class index).
C.label_mapping = [
    (9, 0),    # Water         (41.54%)
    (2, 1),    # Bare Ground   (18.42%)
    (3, 2),    # Rocky Terrain (17.28%)
    (6, 3),    # Shrubs         (6.56%)
    (8, 4),    # Sky            (6.32%)
    (7, 5),    # Trees          (4.91%)
    (4, 6),    # Devel Struct   (3.45%)
    (5, 7),    # Road           (1.41%)
    (10, 8),   # Vehicles       (0.05%)
    (11, 9),   # Person         (0.04%)
    (0, 255),  # Don't Care -> ignore
    (1, 255),  # Noise -> ignore
]

"""Image Config"""
C.background = 255
C.image_height = 600
C.image_width = 960
C.rgb_norm_mean = np.array([0.485, 0.456, 0.406])
C.rgb_norm_std = np.array([0.229, 0.224, 0.225])
# Thermal statistics from the training split (single channel, replicated to 3).
C.x_norm_mean = np.array([0.428, 0.428, 0.428])
C.x_norm_std = np.array([0.201, 0.201, 0.201])

"""Network Settings"""
C.backbone = 'dual_encoder'
# TODO: download the ImageNet-pretrained Swin checkpoints (see README).
C.pretrained_model = './pretrained/swin_large_patch4_window7_224_22k.pth'
C.sar_pretrained_model = './pretrained/swin_tiny_patch4_window7_224.pth'
C.decoder = 'SARD'
C.decoder_channels = 448
C.aux_head = 'FCNHead'          # None | 'FCNHead' | 'AFDAH'
C.aux_head_embed_dim = 128      # AFDAH internal dim (only used when aux_head='AFDAH')
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 2e-5
C.adapter_lr = 1e-4
C.decoder_lr = 3e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.grad_clip = 0.8
C.batch_size = 2
C.nepochs = 300
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0
C.train_scale_array = [0.75, 1, 1.25]
C.warm_up_epoch = 20

"""Eval Config"""
C.eval_flip = False
C.eval_batch_size = 2

"""Validation Strategy Config"""
C.val_early_epochs = 20
C.val_early_frequency = 5
C.val_late_frequency = 1

"""Resume Config"""
C.resume_from = None

"""Path Config"""
C.log_dir = osp.abspath('log_' + C.dataset_name)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))
