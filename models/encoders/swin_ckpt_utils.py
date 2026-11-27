import torch


def load_checkpoint_cpu(path):
    return torch.load(path, map_location='cpu')


def unwrap_swin_checkpoint(raw, branch=None):
    while isinstance(raw, dict):
        wrapper = next((key for key in ('state_dict', 'model', 'module')
                        if key in raw and isinstance(raw[key], dict)), None)
        if wrapper is None:
            break
        raw = raw[wrapper]
    if not isinstance(raw, dict):
        raise TypeError('Expected a checkpoint state dictionary')
    state = {(key[7:] if key.startswith('module.') else key): value
             for key, value in raw.items()}
    suffix = 'patch_embed.proj.weight'
    prefixes = [key[:-len(suffix)] for key in state if key.endswith(suffix)]
    if not prefixes:
        raise ValueError('No Swin patch embedding found in checkpoint')
    if branch is not None:
        candidates = [prefix for prefix in prefixes
                      if prefix.rstrip('.') == branch.rstrip('.')]
        if len(candidates) != 1:
            raise ValueError(f'Unknown or ambiguous Swin branch: {branch}')
        prefix = candidates[0]
    else:
        prefix = min(prefixes, key=lambda p: (state[p + suffix].shape[1], p))
    return {key[len(prefix):]: value for key, value in state.items()
            if key.startswith(prefix)}, prefix


def adapt_patch_embed_in_chans(weight, in_chans):
    if in_chans <= 0:
        raise ValueError('in_chans must be positive')
    if weight.shape[1] == in_chans:
        return weight
    return weight.sum(dim=1, keepdim=True).repeat(1, in_chans, 1, 1) / in_chans
