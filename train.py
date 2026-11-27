import os
import os.path as osp
import sys
import time
import argparse

from utils.pyt_utils import ensure_dir, extant_file, link_file, load_config


def verify_freeze_state(model, logger):
    categories = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            cat = 'FROZEN'
        elif 'dga' in name or 'ssa' in name:
            cat = 'Adapters (DGA+SSA)'
        elif 'backbone.rgb_proj' in name:
            cat = 'RGB channel projection'
        elif 'backbone.aux_' in name:
            cat = 'Swin-Tiny X encoder'
        elif 'backbone.swin_' in name:
            cat = 'RGB Swin backbone (trainable)'
        elif 'backbone.FFM' in name or 'backbone.FFMs' in name:
            cat = 'FFMs (fusion)'
        elif 'decode_head' in name or 'aux_head' in name:
            cat = 'Decoder'
        else:
            cat = 'Other trainable'
        categories.setdefault(cat, [0, 0])
        categories[cat][0] += p.numel()
        categories[cat][1] += 1

    logger.info('=' * 60)
    logger.info('Parameter freeze status:')
    for cat, (count, n_params) in sorted(categories.items()):
        logger.info(f'  {cat}: {count:>12,} params ({n_params} tensors)')
    logger.info('=' * 60)


def set_frozen_bn_eval(model):
    import torch.nn as nn

    for _, m in model.backbone.named_modules():
        is_frozen = not any(p.requires_grad for p in m.parameters(recurse=False)
                            if p is not None)
        if is_frozen and isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def _is_adapter(name):
    return 'dga' in name or 'ssa' in name


def _is_decoder(name):
    return name.startswith('decode_head') or name.startswith('aux_head')


def build_param_groups(model, base_lr, adapter_lr, decoder_lr, norm_layer, logger):
    import torch.nn as nn

    norm_params = set()
    for mod_name, m in model.named_modules():
        if isinstance(m, (norm_layer, nn.BatchNorm1d, nn.BatchNorm2d,
                          nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            for p_name, _ in m.named_parameters(recurse=False):
                norm_params.add(f'{mod_name}.{p_name}' if mod_name else p_name)

    grouped = {
        'Adapter (DGA+SSA)': (adapter_lr, [], []),
        'Decoder': (decoder_lr, [], []),
        'Other trainable': (base_lr, [], []),
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_adapter(name):
            key = 'Adapter (DGA+SSA)'
        elif _is_decoder(name):
            key = 'Decoder'
        else:
            key = 'Other trainable'
        no_decay = (name in norm_params or name.endswith('bias')
                    or any(t in name for t in ('gamma', 'gammax', 'scale', 'alpha', 'beta')))
        grouped[key][2 if no_decay else 1].append(p)

    params_list = []
    for label, (lr, decay, no_decay) in grouped.items():
        params_list.append(dict(params=decay, lr=lr))
        params_list.append(dict(params=no_decay, lr=lr, weight_decay=0.))
        logger.info(f'[optimizer] {label}: {sum(p.numel() for p in decay + no_decay):,} '
                    f'params (lr={lr})')

    return params_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='配置文件路径')
    parser.add_argument('--gpu-id', type=int, default=None)
    parser.add_argument('--gpu-devices', type=str, default=None)
    parser.add_argument('-c', '--continue', type=extant_file, metavar='FILE',
                        dest='continue_fpath', help='resume training from a checkpoint')
    args = parser.parse_args()
    config = load_config(args.cfg)

    if args.gpu_id is not None:
        visible_devices = str(args.gpu_id)
    elif args.gpu_devices is not None:
        visible_devices = args.gpu_devices
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    else:
        visible_devices = getattr(config, 'gpu_devices', str(getattr(config, 'gpu_id', 0)))
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    run_training(config, args.continue_fpath)


def run_training(config, continue_fpath):
    import torch
    import torch.nn as nn
    import torch.backends.cudnn as cudnn
    from tensorboardX import SummaryWriter
    from tqdm import tqdm

    from dataloader.unified_loader import get_train_loader
    from engine.engine import Engine
    from engine.logger import get_logger
    from models.builder_AXIA import EncoderDecoder as segmodel
    from utils.lr_policy import WarmUpPolyLR
    from utils.validation import ValidationHelper

    logger = get_logger()
    with Engine() as engine:
        engine.continue_state_object = continue_fpath
        logger.info(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(device)
            logger.info(f"GPU: {props.name}  显存: {props.total_memory / 1024**3:.1f} GB")
        cudnn.benchmark = True
        seed = getattr(config, 'seed', 12345)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)

        train_loader, _ = get_train_loader(engine, config)

        ensure_dir(config.log_dir)
        ensure_dir(config.checkpoint_dir)
        logger.info(f"日志目录: {config.log_dir}")
        logger.info(f"检查点目录: {config.checkpoint_dir}")
        logger.info(f"数据集: {config.dataset_name}")
        logger.info(f"训练样本数: {config.num_train_imgs}")
        logger.info(f"验证样本数: {config.num_eval_imgs}")
        logger.info(f"类别数: {config.num_classes}")
        if hasattr(config, 'class_names'):
            logger.info(f"类别名称: {config.class_names}")

        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, config.tb_dir + '/tb')

        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        BatchNorm2d = nn.BatchNorm2d

        model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

        verify_freeze_state(model, logger)

        validation_helper = ValidationHelper(config, model, [device])

        base_lr = config.lr
        adapter_lr = getattr(config, 'adapter_lr', base_lr)
        decoder_lr = getattr(config, 'decoder_lr', base_lr)
        logger.info(f'学习率: base_lr={base_lr}, adapter_lr={adapter_lr}, decoder_lr={decoder_lr}')

        params_list = build_param_groups(model, base_lr, adapter_lr, decoder_lr, BatchNorm2d, logger)

        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                params_list, lr=base_lr, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=config.weight_decay)
        else:
            raise NotImplementedError(f'不支持的优化器: {config.optimizer}')

        total_iteration = config.nepochs * config.niters_per_epoch
        warmup_iters = config.niters_per_epoch * config.warm_up_epoch
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, warmup_iters)
        logger.info(f'LR schedule: WarmUpPoly (peak={base_lr}, power={config.lr_power}, '
                    f'warmup_iters={warmup_iters}, total_iters={total_iteration})')

        initial_lrs = [pg['lr'] for pg in optimizer.param_groups]

        logger.info('.............双编码器训练.............')
        model.to(device)
        engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)

        if getattr(config, 'resume_from', None) is not None:
            engine.continue_state_object = config.resume_from
            engine.restore_checkpoint()
            logger.info(f"从检查点继续，epoch {engine.state.epoch}")
        elif engine.continue_state_object:
            engine.restore_checkpoint()
            logger.info(f"从检查点继续，epoch {engine.state.epoch}")

        logger.info('开始训练 ...')

        val_frequency = config.val_frequency
        if not isinstance(val_frequency, int) or isinstance(val_frequency, bool) or val_frequency <= 0:
            raise ValueError('val_frequency must be a positive integer')
        logger.info(f'验证策略: 全程每{val_frequency}轮验证一次')

        for epoch in range(engine.state.epoch, config.nepochs + 1):
            model.train()
            set_frozen_bn_eval(model)

            logger.info(f"Epoch {epoch}...")
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

            dataloader = iter(train_loader)
            sum_loss = 0

            for idx in pbar:
                try:
                    engine.update_iteration(epoch, idx)
                    minibatch = next(dataloader)
                    imgs = minibatch['data'].to(device, non_blocking=True)
                    gts = minibatch['label'].to(device, non_blocking=True)
                    modal_xs = minibatch['modal_x'].to(device, non_blocking=True)

                    loss = model(imgs, modal_xs, gts)
                except Exception as e:
                    if getattr(config, 'dataset_name', '').upper() != 'CART':
                        logger.error(f"批次 {idx+1} 出错: {e}")
                    continue

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    if getattr(config, 'dataset_name', '').upper() != 'CART':
                        logger.warning(f"NaN/Inf loss: {loss.item()}")
                    continue

                optimizer.zero_grad()
                loss.backward()

                if getattr(config, 'grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        config.grad_clip)

                optimizer.step()

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

            if epoch % val_frequency == 0:
                logger.info(f"Epoch {epoch}: 验证中...")
                miou, is_best = validation_helper.validate(epoch, model)
                tb.add_scalar('val_miou', miou, epoch)

                if is_best:
                    logger.info(f"最佳模型 mIoU: {miou:.4f}")
                    ensure_dir(config.checkpoint_dir)
                    best_path = osp.join(config.checkpoint_dir, 'best_model.pth')
                    engine.save_checkpoint(best_path)
                    link_file(best_path, osp.join(config.checkpoint_dir, 'epoch-last.pth'))
                    with open(osp.join(config.checkpoint_dir, 'best_model_info.txt'), 'w') as f:
                        f.write(f"Epoch: {epoch}\nmIoU: {miou:.4f}\n")

        tb.close()
        logger.info('训练完成!')


if __name__ == '__main__':
    main()
