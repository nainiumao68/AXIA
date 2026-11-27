import os
import shutil
import argparse
import logging
import importlib.util
from collections import OrderedDict

logger = logging.getLogger()


def load_config(config_path):
    path = os.path.abspath(config_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('axia_config', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load config: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def load_model(model, model_file, strict=False):
    if model_file is None:
        return model
    import torch

    state = torch.load(model_file, map_location='cpu') if isinstance(model_file, str) else model_file
    while isinstance(state, dict):
        wrapper = next((key for key in ('model', 'state_dict', 'module')
                        if key in state and isinstance(state[key], dict)), None)
        if wrapper is None:
            break
        state = state[wrapper]
    own_keys = set(model.state_dict())
    state = OrderedDict((key[7:] if key.startswith('module.') else key, value)
                        for key, value in state.items())
    if own_keys and all(key.startswith('module.') for key in own_keys):
        state = OrderedDict(('module.' + key, value) for key, value in state.items())
    if not own_keys.intersection(state):
        raise RuntimeError('Checkpoint has no parameter keys matching this model')
    result = model.load_state_dict(state, strict=strict)
    if result.missing_keys or result.unexpected_keys:
        logger.warning(f'Checkpoint keys: {len(result.missing_keys)} missing, '
                       f'{len(result.unexpected_keys)} unexpected')
    return model


def extant_file(x):
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src, target):
    src = os.path.abspath(src)
    target = os.path.abspath(target)
    if src == target:
        raise ValueError('Source and target must differ')
    if os.path.islink(target):
        os.unlink(target)
    if os.name == 'nt':
        if os.path.isdir(src):
            shutil.copytree(src, target, dirs_exist_ok=True)
        else:
            shutil.copy2(src, target)
    else:
        if os.path.isdir(target):
            os.rmdir(target)
        elif os.path.exists(target):
            os.unlink(target)
        os.symlink(src, target, target_is_directory=os.path.isdir(src))


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
