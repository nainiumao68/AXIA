"""AXIA config for the YESeg-OPT-SAR dataset (RGB + SAR, 8 classes)."""
import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 12345

"""GPU Device Config"""
C.gpu_id = 0
C.gpu_devices = str(C.gpu_id)

C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))

"""Dataset Path"""
C.dataset_name = 'YESeg'
# TODO: point this to your local copy of the dataset.
C.dataset_path = './datasets/YESeg-OPT-SAR'
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'Label_normalized')
C.gt_format = '.png'
C.gt_transform = True
C.x_root_folder = osp.join(C.dataset_path, 'SAR')
C.x_format = '.png'
C.x_is_single_channel = True
C.train_source = osp.join(C.dataset_path, 'train.txt')
C.eval_source = osp.join(C.dataset_path, 'test.txt')
C.num_train_imgs = 1784
C.num_eval_imgs = 447
C.num_classes = 8
C.class_names = ['background', 'bareground', 'lowvegetation', 'trees',
                 'houses', 'water', 'roads', 'others']

"""Image Config"""
C.background = 255
C.image_height = 256
C.image_width = 256
C.rgb_norm_mean = np.array([0.485, 0.456, 0.406])
C.rgb_norm_std = np.array([0.229, 0.224, 0.225])
C.x_norm_mean = np.array([0.1612, 0.1612, 0.1612])
C.x_norm_std = np.array([0.1335, 0.1335, 0.1335])

"""Network Settings"""
C.backbone = 'dual_encoder'
# TODO: download the ImageNet-pretrained Swin checkpoints (see README).
C.pretrained_model = './pretrained/swin_large_patch4_window7_224_22k.pth'
C.sar_pretrained_model = './pretrained/swin_tiny_patch4_window7_224.pth'
C.decoder = 'SARD'
C.decoder_channels = 328
C.aux_head = 'FCNHead'          # None | 'FCNHead' | 'AFDAH'
C.aux_head_embed_dim = 128      # AFDAH internal dim (only used when aux_head='AFDAH')
C.optimizer = 'AdamW'

"""Train Config (grouped learning rates)"""
C.lr = 6e-5
C.adapter_lr = 3e-4
C.decoder_lr = 1e-4
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.grad_clip = 1.0
C.batch_size = 8
C.nepochs = 300
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0
C.train_scale_array = [0.75, 1.0, 1.25, 1.50]
C.warm_up_epoch = 10

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
