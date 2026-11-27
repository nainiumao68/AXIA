import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 12345

C.gpu_id = 0
C.gpu_devices = str(C.gpu_id)

C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))

C.dataset_name = 'PIE-RGB-SAR'
C.dataset_path = './datasets/PIE-RGB-SAR'
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.tif'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.tif'
C.gt_transform = True
C.x_root_folder = osp.join(C.dataset_path, 'SAR')
C.x_format = '.tif'
C.x_is_single_channel = True
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "val.txt")
C.num_train_imgs = 2433
C.num_eval_imgs = 2432
C.num_classes = 6
C.class_names = ['background', 'building', 'road', 'water', 'forest', 'farmland']

C.background = 255
C.image_height = 256
C.image_width = 256
C.rgb_norm_mean = np.array([83.405, 89.751, 73.422]) / 255.0
C.rgb_norm_std = np.array([24.829, 26.953, 29.48]) / 255.0
C.x_norm_mean = np.full(3, 84.869 / 255.0)
C.x_norm_std = np.full(3, 57.769 / 255.0)

C.backbone = 'dual_encoder'
C.pretrained_model = './pretrained/swin_large_patch4_window7_224_22k.pth'
C.sar_pretrained_model = './pretrained/swin_tiny_patch4_window7_224.pth'
C.decoder = 'SARD'
C.decoder_channels = 328
C.decoder_dropout = 0.05
C.aux_head = 'FCNHead'
C.optimizer = 'AdamW'

C.lr = 6e-5
C.adapter_lr = 3e-4
C.decoder_lr = 1e-4
C.lr_power = 0.9
C.weight_decay = 0.01
C.grad_clip = 1.0
C.batch_size = 8
C.nepochs = 300
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0
C.train_scale_array = [0.75, 1, 1.25]
C.warm_up_epoch = 20

C.eval_flip = False
C.eval_batch_size = 12

C.val_frequency = 5

C.resume_from = None

C.log_dir = osp.abspath('log_' + C.dataset_name)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))
