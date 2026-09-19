"""AXIA training script.

Usage:
    python train.py --cfg config_pie.py [--gpu-id 0]
"""
import os
import os.path as osp
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from engine.engine import Engine
from engine.logger import get_logger
from utils.lr_policy import WarmUpPolyLR, WarmUpCosineLR
from utils.ema import EMAModel
from utils.pyt_utils import ensure_dir, link_file, load_config
from utils.validation import ValidationHelper
from dataloader.loader import get_train_loader
from models.builder_AXIA import EncoderDecoder as segmodel

from tensorboardX import SummaryWriter


# --------------- freeze / param utilities ---------------

def verify_freeze_state(model, logger):
    """Log the parameter freeze status per module group."""
    categories = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            cat = 'Frozen (RGB Swin-Large / X stages 3-4)'
        elif 'dga' in name or 'ssa' in name:
            cat = 'Adapters (DGA+SSA)'
        elif 'backbone.rgb_proj' in name or 'backbone.sar_proj' in name:
            cat = 'Channel projections'
        elif 'backbone.aux_' in name:
            cat = 'X Swin-Tiny (trainable stages)'
        elif 'backbone.swin_' in name:
            cat = 'RGB Swin backbone (trainable)'
        elif 'backbone.FFMs' in name:
            cat = 'Stage fusion (LSGF)'
        elif 'decode_head' in name:
            cat = 'Decoder'
        elif 'aux_head' in name:
            cat = 'Auxiliary head'
        else:
            cat = 'Other trainable'
        categories.setdefault(cat, [0, 0])
        categories[cat][0] += p.numel()
        categories[cat][1] += 1

    logger.info('=' * 60)
    logger.info('Parameter freeze status:')
    for cat, (count, n_tensors) in sorted(categories.items()):
        logger.info(f'  {cat}: {count:>12,} params ({n_tensors} tensors)')
    logger.info('=' * 60)


def set_frozen_bn_eval(model):
    """Keep BN layers inside fully frozen submodules in eval mode."""
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()


def _collect_decay_nodecay(params_iter, norm_layer):
    """Split parameters into decay / no-decay groups (trainable only)."""
    decay, no_decay = [], []
    seen = set()

    for name, m in params_iter:
        if isinstance(m, nn.Linear):
            if m.weight.requires_grad and id(m.weight) not in seen:
                decay.append(m.weight); seen.add(id(m.weight))
            if m.bias is not None and m.bias.requires_grad and id(m.bias) not in seen:
                no_decay.append(m.bias); seen.add(id(m.bias))
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                            nn.ConvTranspose2d, nn.ConvTranspose3d)):
            if m.weight.requires_grad and id(m.weight) not in seen:
                decay.append(m.weight); seen.add(id(m.weight))
            if m.bias is not None and m.bias.requires_grad and id(m.bias) not in seen:
                no_decay.append(m.bias); seen.add(id(m.bias))
        elif isinstance(m, norm_layer) or isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                    nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None and m.weight.requires_grad and id(m.weight) not in seen:
                no_decay.append(m.weight); seen.add(id(m.weight))
            if m.bias is not None and m.bias.requires_grad and id(m.bias) not in seen:
                no_decay.append(m.bias); seen.add(id(m.bias))

    return decay, no_decay, seen


def _is_adapter_param(name):
    return 'dga' in name or 'ssa' in name


def build_param_groups(model, base_lr, adapter_lr, decoder_lr, norm_layer, logger):
    """Grouped learning rates:
    - DGA/SSA adapters use ``adapter_lr``;
    - decode_head / aux_head use ``decoder_lr`` (freshly initialized modules
      usually want a larger LR);
    - everything else uses ``base_lr``.
    """
    adapter_modules = []
    decoder_modules = []
    other_modules = []

    for full_name, module in model.named_modules():
        if _is_adapter_param(full_name):
            adapter_modules.append((full_name, module))
        elif (full_name.startswith('decode_head')
              or full_name.startswith('aux_head')):
            decoder_modules.append((full_name, module))
        else:
            other_modules.append((full_name, module))

    adapter_decay, adapter_no_decay, adapter_seen = _collect_decay_nodecay(
        adapter_modules, norm_layer)
    decoder_decay, decoder_no_decay, decoder_seen = _collect_decay_nodecay(
        decoder_modules, norm_layer)
    other_decay, other_no_decay, other_seen = _collect_decay_nodecay(
        other_modules, norm_layer)

    all_seen = adapter_seen | decoder_seen | other_seen
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in all_seen:
            continue
        is_adapter = _is_adapter_param(name)
        is_decoder = (name.startswith('decode_head')
                      or name.startswith('aux_head'))
        if is_adapter:
            target_no_decay = adapter_no_decay
            target_decay = adapter_decay
        elif is_decoder:
            target_no_decay = decoder_no_decay
            target_decay = decoder_decay
        else:
            target_no_decay = other_no_decay
            target_decay = other_decay
        if ('bias' in name or 'gamma' in name or 'scale' in name
                or 'alpha' in name or 'beta' in name):
            target_no_decay.append(p)
        else:
            target_decay.append(p)

    params_list = [
        dict(params=adapter_decay,    lr=adapter_lr, weight_decay=None),
        dict(params=adapter_no_decay, lr=adapter_lr, weight_decay=0.),
        dict(params=decoder_decay,    lr=decoder_lr, weight_decay=None),
        dict(params=decoder_no_decay, lr=decoder_lr, weight_decay=0.),
        dict(params=other_decay,      lr=base_lr,    weight_decay=None),
        dict(params=other_no_decay,   lr=base_lr,    weight_decay=0.),
    ]
    for g in params_list:
        if g['weight_decay'] is None:
            del g['weight_decay']

    adapter_total = sum(p.numel() for p in adapter_decay + adapter_no_decay)
    decoder_total = sum(p.numel() for p in decoder_decay + decoder_no_decay)
    other_total = sum(p.numel() for p in other_decay + other_no_decay)
    logger.info(f'[optimizer] Adapter (DGA+SSA) params: {adapter_total:,} (lr={adapter_lr})')
    logger.info(f'[optimizer] Decoder params: {decoder_total:,} (lr={decoder_lr})')
    logger.info(f'[optimizer] Other trainable params: {other_total:,} (lr={base_lr})')

    return params_list


# --------------- main ---------------

def main():
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='path to the config file')
    parser.add_argument('--gpu-id', type=int, default=None)
    parser.add_argument('--gpu-devices', type=str, default=None)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        # The engine-level `-c/--continue` flag resumes from a checkpoint.
        if args.continue_fpath:
            engine.continue_state_object = args.continue_fpath

        config = load_config(args.cfg)

        gpu_id_from_cfg = getattr(config, 'gpu_id', 0)
        gpu_devices_from_cfg = getattr(config, 'gpu_devices', str(gpu_id_from_cfg))
        if args.gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
            logger.info(f"Using GPU from command line: {args.gpu_id}")
        elif args.gpu_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
            logger.info(f"Using GPU devices from command line: {args.gpu_devices}")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices_from_cfg
            logger.info(f"Using GPU devices from config: {gpu_devices_from_cfg}")

        cudnn.benchmark = True
        seed = getattr(config, 'seed', 12345)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        train_loader = get_train_loader(config)

        ensure_dir(config.log_dir)
        ensure_dir(config.checkpoint_dir)
        logger.info(f"Log dir: {config.log_dir}")
        logger.info(f"Checkpoint dir: {config.checkpoint_dir}")

        logger.info(f"Dataset: {config.dataset_name}")
        logger.info(f"Train samples: {config.num_train_imgs}")
        logger.info(f"Val samples: {config.num_eval_imgs}")
        logger.info(f"Classes: {config.num_classes}")
        if hasattr(config, 'class_names'):
            logger.info(f"Class names: {config.class_names}")

        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

        ce_kwargs = dict(reduction='mean', ignore_index=config.background)
        class_weight = getattr(config, 'class_loss_weight', None)
        if class_weight is not None:
            ce_kwargs['weight'] = torch.tensor(class_weight, dtype=torch.float32)
            logger.info(f'Class weights: {class_weight}')
        label_smoothing = float(getattr(config, 'label_smoothing', 0.0))
        if label_smoothing > 0:
            ce_kwargs['label_smoothing'] = label_smoothing
            logger.info(f'Label smoothing: {label_smoothing}')
        criterion = nn.CrossEntropyLoss(**ce_kwargs)
        BatchNorm2d = nn.BatchNorm2d

        # The builder loads pretrained backbone weights and freezes stages internally.
        model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

        verify_freeze_state(model, logger)

        if hasattr(criterion, 'cuda'):
            try:
                criterion = criterion.cuda()
            except Exception:
                pass

        validation_helper = ValidationHelper(config, model, engine.devices)

        # Grouped learning rates.
        base_lr = config.lr
        adapter_lr = getattr(config, 'adapter_lr', base_lr)
        decoder_lr = getattr(config, 'decoder_lr', base_lr)
        logger.info(f'Learning rates: base={base_lr}, adapter={adapter_lr}, decoder={decoder_lr}')

        params_list = build_param_groups(model, base_lr, adapter_lr, decoder_lr, BatchNorm2d, logger)

        if hasattr(criterion, 'parameters'):
            criterion_params = [p for p in criterion.parameters() if p.requires_grad]
            if criterion_params:
                params_list.append({
                    'params': criterion_params,
                    'lr': base_lr * 5.0,
                    'weight_decay': 0.0
                })

        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                params_list, lr=base_lr,
                betas=(0.9, 0.999), eps=1e-8,
                weight_decay=config.weight_decay)
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(
                params_list, lr=base_lr,
                momentum=config.momentum, dampening=0,
                weight_decay=config.weight_decay, nesterov=True)
        else:
            raise NotImplementedError(f'Unsupported optimizer: {config.optimizer}')

        total_iteration = config.nepochs * config.niters_per_epoch
        warmup_iters = config.niters_per_epoch * config.warm_up_epoch
        lr_schedule = str(getattr(config, 'lr_schedule', 'poly')).lower()
        if lr_schedule == 'cosine':
            min_lr_ratio = float(getattr(config, 'lr_min_ratio', 0.01))
            lr_policy = WarmUpCosineLR(base_lr, total_iteration, warmup_iters,
                                       min_lr_ratio=min_lr_ratio)
            logger.info(
                f'LR schedule: WarmUpCosine (peak={base_lr}, '
                f'min_ratio={min_lr_ratio}, warmup_iters={warmup_iters}, '
                f'total_iters={total_iteration})')
        else:
            lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration,
                                     warmup_iters)
            logger.info(
                f'LR schedule: WarmUpPoly (peak={base_lr}, power={config.lr_power}, '
                f'warmup_iters={warmup_iters}, total_iters={total_iteration})')

        # Per-group initial LRs are scaled proportionally by the schedule.
        initial_lrs = [pg['lr'] for pg in optimizer.param_groups]

        logger.info('............. AXIA training .............')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            cur = torch.cuda.current_device()
            logger.info(f"GPU: {torch.cuda.get_device_name(cur)}  "
                        f"Memory: {torch.cuda.get_device_properties(cur).total_memory / 1024**3:.1f} GB")

        model.to(device)
        engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)

        # EMA (created after the model is on CUDA, before any resume).
        use_ema = bool(getattr(config, 'use_ema', False))
        ema_decay = float(getattr(config, 'ema_decay', 0.9998))
        ema = None
        if use_ema:
            ema = EMAModel(model, decay=ema_decay)
            ema.to(device)
            logger.info(f'EMA enabled: decay={ema_decay}')
        else:
            logger.info('EMA disabled')

        if getattr(config, 'resume_from', None) is not None:
            engine.continue_state_object = config.resume_from
            engine.restore_checkpoint()
            logger.info(f"Resumed from checkpoint, epoch {engine.state.epoch}")
        elif engine.continue_state_object:
            engine.restore_checkpoint()
            logger.info(f"Resumed from checkpoint, epoch {engine.state.epoch}")

        optimizer.zero_grad()

        logger.info('Start training ...')

        early_epochs = getattr(config, 'val_early_epochs', 10)
        early_freq = getattr(config, 'val_early_frequency', 5)
        late_freq = getattr(config, 'val_late_frequency', 1)
        logger.info(f'Validation policy: every {early_freq} epochs for the first '
                    f'{early_epochs} epochs, then every {late_freq} epoch(s)')

        for epoch in range(engine.state.epoch, config.nepochs + 1):
            model.train()
            set_frozen_bn_eval(model)

            logger.info(f"Epoch {epoch}...")
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

            dataloader = iter(train_loader)
            sum_loss = 0

            for idx in pbar:
                engine.update_iteration(epoch, idx)
                minibatch = next(dataloader)
                imgs = minibatch['data'].cuda(non_blocking=True)
                gts = minibatch['label'].cuda(non_blocking=True)
                modal_xs = minibatch['modal_x'].cuda(non_blocking=True)

                loss = model(imgs, modal_xs, gts)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(f"NaN/Inf loss: {loss.item()}")
                    continue

                optimizer.zero_grad()
                loss.backward()

                if getattr(config, 'grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        config.grad_clip)

                optimizer.step()

                if ema is not None:
                    ema.update(model)

                current_idx = (epoch - 1) * config.niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)
                scale = lr / base_lr if base_lr > 0 else 1.0
                for pg, init_lr in zip(optimizer.param_groups, initial_lrs):
                    pg['lr'] = init_lr * scale

                sum_loss += loss.item()
                cur_lr = optimizer.param_groups[2]['lr'] if len(optimizer.param_groups) > 2 else lr
                print_str = (f'Epoch {epoch}/{config.nepochs} '
                             f'Iter {idx+1}/{config.niters_per_epoch}: '
                             f'lr={cur_lr:.4e} loss={loss.item():.4f} '
                             f'avg={sum_loss/(idx+1):.4f}')
                del loss
                pbar.set_description(print_str, refresh=False)

            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

            should_validate = False
            if epoch <= early_epochs:
                should_validate = (epoch % max(1, early_freq) == 0) or (epoch == config.nepochs)
            else:
                should_validate = (epoch % max(1, late_freq) == 0) or (epoch == config.nepochs)

            if should_validate:
                logger.info(f"Epoch {epoch}: validating ...")
                # With EMA enabled, validate the EMA weights (the raw-model mIoU
                # is still computed implicitly on the next non-EMA run).
                if ema is not None:
                    miou, is_best = validation_helper.validate(epoch, ema.ema_model)
                    tb.add_scalar('val_miou_ema', miou, epoch)
                    logger.info(f'[EMA] Epoch {epoch} val_miou={miou:.4f}')
                else:
                    miou, is_best = validation_helper.validate(epoch, model)
                    tb.add_scalar('val_miou', miou, epoch)

                if is_best:
                    logger.info(f"Best model mIoU: {miou:.4f}")
                    ensure_dir(config.checkpoint_dir)
                    best_path = osp.join(config.checkpoint_dir, 'best_model.pth')
                    # With EMA, save the EMA weights (consistent with validation).
                    if ema is not None:
                        from collections import OrderedDict
                        new_sd = OrderedDict()
                        for k, v in ema.ema_model.state_dict().items():
                            key = k[7:] if k.startswith('module.') else k
                            new_sd[key] = v
                        torch.save({'model': new_sd, 'epoch': epoch,
                                    'iteration': engine.state.iteration,
                                    'ema_decay': ema_decay}, best_path)
                        logger.info(f'[EMA] Saved EMA weights -> {best_path}')
                    else:
                        engine.save_checkpoint(best_path)
                    link_file(best_path, osp.join(config.checkpoint_dir, 'epoch-last.pth'))
                    with open(osp.join(config.checkpoint_dir, 'best_model_info.txt'), 'w') as f:
                        f.write(f"Epoch: {epoch}\nmIoU: {miou:.4f}\n")

        logger.info('Training finished!')


if __name__ == '__main__':
    main()
