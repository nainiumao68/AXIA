"""Learning-rate schedules with linear warmup."""
import math
from abc import ABCMeta, abstractmethod


class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter):
        pass


class PolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_iter):
        return self.start_lr * (
                (1 - float(cur_iter) / self.total_iters) ** self.lr_power)


class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        else:
            return self.start_lr * (
                    (1 - float(cur_iter) / self.total_iters) ** self.lr_power)


class WarmUpCosineLR(BaseLR):
    """Linear warmup followed by cosine annealing down to
    ``start_lr * min_lr_ratio``."""

    def __init__(self, start_lr, total_iters, warmup_steps, min_lr_ratio=0.01):
        self.start_lr = start_lr
        self.total_iters = total_iters + 0.0
        self.warmup_steps = max(1, warmup_steps)
        self.min_lr = start_lr * min_lr_ratio

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        progress = (cur_iter - self.warmup_steps) / max(
            1.0, (self.total_iters - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.start_lr - self.min_lr) * cos_factor
