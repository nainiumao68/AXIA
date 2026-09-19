"""Training engine: state registry, checkpoint save/restore, TensorBoard linking.

Single-GPU only; distributed training is not supported.
"""
import os
import os.path as osp
import time
import argparse

import torch

from .logger import get_logger
from utils.pyt_utils import load_model, extant_file, link_file, ensure_dir

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
            assert k in ['epoch', 'iteration', 'dataloader', 'model',
                         'optimizer']
            setattr(self, k, v)


class Engine(object):
    def __init__(self, custom_parser=None):
        logger.info(
            "PyTorch Version {}".format(torch.__version__))
        self.state = State()
        self.devices = [0]
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()

        # Set later by the training script if a resume is requested.
        self.continue_state_object = None

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        p.add_argument('--local_rank', default=0, type=int,
                       help='process rank on node')
        p.add_argument('-p', '--port', type=str,
                       default='16005',
                       dest="port",
                       help='port for init_process_group')

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration

    def save_checkpoint(self, path):
        logger.info("Saving checkpoint to file {}".format(path))
        t_start = time.time()

        ensure_dir(os.path.dirname(path))

        state_dict = {}

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.state.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            new_state_dict[key] = v
        state_dict['model'] = new_state_dict
        state_dict['optimizer'] = self.state.optimizer.state_dict()
        state_dict['epoch'] = self.state.epoch
        state_dict['iteration'] = self.state.iteration

        t_iobegin = time.time()
        torch.save(state_dict, path)
        del state_dict
        del new_state_dict
        t_end = time.time()
        logger.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare checkpoint: {}, IO: {}".format(
                path, t_iobegin - t_start, t_end - t_iobegin))

    def link_tb(self, source, target):
        ensure_dir(source)
        ensure_dir(target)
        link_file(source, target)

    def restore_checkpoint(self):
        t_start = time.time()
        try:
            tmp = torch.load(self.continue_state_object, map_location=torch.device('cpu'))
            t_ioend = time.time()

            # Non-strict restore tolerates architecture drift in old checkpoints.
            self.state.model = load_model(self.state.model, tmp['model'], is_restore=True, strict=False)

            try:
                self.state.optimizer.load_state_dict(tmp['optimizer'])
                logger.info("Optimizer state loaded")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")
                logger.info("Continuing with a fresh optimizer state")

            if 'epoch' in tmp:
                self.state.epoch = tmp['epoch'] + 1
                logger.info(f"Resuming from epoch {self.state.epoch}")
            else:
                logger.warning("No epoch info in checkpoint; starting from epoch 1")
                self.state.epoch = 1

            if 'iteration' in tmp:
                self.state.iteration = tmp['iteration']
                logger.info(f"Resuming from iteration {self.state.iteration}")
            else:
                logger.warning("No iteration info in checkpoint; starting from iteration 0")
                self.state.iteration = 0

            del tmp
            t_end = time.time()
            logger.info(
                "Load checkpoint from file {}, "
                "Time usage:\n\tIO: {}, restore checkpoint: {}".format(
                    self.continue_state_object, t_ioend - t_start, t_end - t_ioend))
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Training from scratch instead")
            self.state.epoch = 1
            self.state.iteration = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "An exception occurred during Engine initialization, "
                "giving up running process")
            return False
