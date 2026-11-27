import os.path as osp
import time
from collections import OrderedDict

import torch

from .logger import get_logger
from utils.pyt_utils import load_model, link_file, ensure_dir

logger = get_logger()


class State(object):
    def __init__(self):
        self.epoch = 1
        self.iteration = 0
        self.dataloader = None
        self.model = None
        self.optimizer = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            assert k in ['epoch', 'iteration', 'dataloader', 'model', 'optimizer']
            setattr(self, k, v)


class Engine(object):
    def __init__(self):
        logger.info("PyTorch Version {}".format(torch.__version__))
        self.state = State()
        self.devices = [0]
        self.distributed = False

        self.continue_state_object = None

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration

    def save_checkpoint(self, path):
        logger.info("Saving checkpoint to file {}".format(path))
        t_start = time.time()

        ensure_dir(osp.dirname(path))

        new_state_dict = OrderedDict()
        for k, v in self.state.model.state_dict().items():
            new_state_dict[k[7:] if k.startswith('module.') else k] = v

        state_dict = {
            'model': new_state_dict,
            'optimizer': self.state.optimizer.state_dict(),
            'epoch': self.state.epoch,
            'iteration': self.state.iteration,
        }

        t_iobegin = time.time()
        torch.save(state_dict, path)
        t_end = time.time()
        logger.info("Save checkpoint to file {}, Time usage:\n\tprepare checkpoint: {}, IO: {}".format(
            path, t_iobegin - t_start, t_end - t_iobegin))

    def link_tb(self, source, target):
        ensure_dir(source)
        ensure_dir(target)
        link_file(source, target)

    def restore_checkpoint(self):
        t_start = time.time()
        tmp = torch.load(self.continue_state_object, map_location='cpu')
        t_ioend = time.time()

        self.state.model = load_model(self.state.model, tmp['model'], strict=False)

        try:
            self.state.optimizer.load_state_dict(tmp['optimizer'])
            logger.info("成功加载优化器状态")
        except Exception as e:
            logger.warning(f"加载优化器状态失败: {e}")

        self.state.epoch = tmp['epoch'] + 1 if 'epoch' in tmp else 1
        self.state.iteration = tmp['iteration'] if 'iteration' in tmp else 0
        logger.info(f"从第 {self.state.epoch} 轮、迭代 {self.state.iteration} 继续训练")

        t_end = time.time()
        logger.info("Load checkpoint from file {}, Time usage:\n\tIO: {}, restore checkpoint: {}".format(
            self.continue_state_object, t_ioend - t_start, t_end - t_ioend))

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning("A exception occurred during Engine initialization, "
                           "give up running process")
            return False
