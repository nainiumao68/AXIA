"""Exponential Moving Average of model weights.

Usage:

    ema = EMAModel(model, decay=0.9998)
    for batch in loader:
        ...
        optimizer.step()
        ema.update(model)            # sync the EMA after each parameter update

    # validation:
    miou = validate(ema.ema_model)

    # saving:
    torch.save({'model': ema.state_dict()}, path)

Notes:
- Buffers (e.g. BN running stats) are copied directly rather than
  interpolated, following the YOLOv5 / timm convention; integer buffers such
  as ``num_batches_tracked`` are copied as well.
"""
import copy

import torch
import torch.nn as nn


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        assert 0.0 < decay < 1.0, f"decay must be in (0,1), got {decay}"
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if k not in msd:
                continue
            source = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(source.to(v.dtype), alpha=1.0 - self.decay)
            else:
                v.copy_(source)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)

    def to(self, device):
        self.ema_model.to(device)
        return self
