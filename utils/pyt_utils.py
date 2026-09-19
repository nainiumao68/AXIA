"""Shared PyTorch / filesystem utilities."""
import argparse
import importlib.util
import logging
import os
import random
import sys
import time
from collections import OrderedDict

import torch

# engine.logger.get_logger() configures the root logger; modules simply share it.
logger = logging.getLogger()


def load_config(config_path):
    """Import a python config file and return its ``config`` object."""
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    module_name = f"user_cfg_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config file: {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, 'config'):
        raise AttributeError(f"Config file {config_path} must define a `config` object")
    return module.config


def load_model(model, model_file, is_restore=False, strict=False):
    t_start = time.time()

    if model_file is None:
        return model

    if isinstance(model_file, str):
        try:
            state_dict = torch.load(model_file)
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            elif 'module' in state_dict.keys():
                state_dict = state_dict['module']
        except Exception as e:
            logger.error(f"Error loading model file {model_file}: {e}")
            return model
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    own_state_dict = model.state_dict()

    ckpt_keys = set(state_dict.keys())
    own_keys = set(own_state_dict.keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if missing_keys:
        logger.warning(f"Missing keys when loading model ({len(missing_keys)}): "
                       f"{', '.join(list(missing_keys)[:5])}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading model ({len(unexpected_keys)}): "
                       f"{', '.join(list(unexpected_keys)[:5])}...")

    try:
        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Model loaded with strict={strict}")
    except RuntimeError as e:
        logger.warning(f"Strict loading failed: {e}")

        if strict:
            logger.info("Retrying with strict=False ...")
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.info("Non-strict loading succeeded")
            except Exception as e2:
                logger.error(f"Non-strict loading also failed: {e2}")
                return model
        else:
            return model

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model


def extant_file(x):
    """'Type' for argparse - checks that the file exists without opening it."""
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        if os.name == 'nt':
            os.system('rd /s /q "{}"'.format(target))
        else:
            os.system('rm -rf {}'.format(target))

    if os.name == 'nt':
        # Windows has no unprivileged symlinks; copy instead.
        import shutil
        if os.path.isdir(src):
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
    else:
        os.system('ln -s {} {}'.format(src, target))


def ensure_dir(path):
    if not os.path.isdir(path):
        try:
            sleeptime = random.randint(0, 3)
            time.sleep(sleeptime)
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
