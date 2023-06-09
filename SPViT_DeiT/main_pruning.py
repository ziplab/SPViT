# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Modifications copyright (c) 2023 Zhuang AI Group, Haoyu He

import random
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils import create_scheduler, create_2optimizers, NativeScaler

from datasets import build_dataset
from engine import train_one_epoch_pruning, evaluate_pruning
from losses import DistillationLoss
from samplers import RASampler

import models_pruning
import utils
from params import args
from logger import logger

from tensorboardX import SummaryWriter
from collections import OrderedDict


torch.backends.cuda.matmul.allow_tf32 = False


class Custom_scaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def main():
    utils.init_distributed_mode(args)
    if utils.get_rank() != 0:
        logger.disabled = True
    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    torch.backends.cudnn.deterministic = True
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Load searched architectures
    if args.assigned_indicators or args.searching_model:
        # Must have both msa_indicators and ffn_indicators
        assert (args.assigned_indicators and args.searching_model)

        logger.info("Fine-tuning from searched architecture...")

        logger.info(f"MSA indicators: {args.assigned_indicators}")
        logger.info(f"FFN indicator path: {args.searching_model}")

        ffn_indicators = []
        names = []

        # Get the FFN indicators
        if args.searching_model:
            searching_checkpoint = torch.load(args.searching_model, map_location='cpu')

            for name in searching_checkpoint['model'].keys():
                if 'assigned_indicator_index' in name:
                    ffn_indicators.append(searching_checkpoint['model'][name])
                    names.append(name)

    else:
        ffn_indicators = None
        logger.info("Searching...")

    logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        att_layer=args.att_layer,
        ffn_layer=args.ffn_layer,
        loss_lambda=args.loss_lambda,
        target_flops=args.target_flops,
        theta=args.theta,
        msa_indicators=args.assigned_indicators,
        ffn_indicators=ffn_indicators
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: ' + str(n_parameters))

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer1, optimizer2 = create_2optimizers(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler1, _ = create_scheduler(args, optimizer1, args.epochs, args.warmup_epochs, args.min_lr)

    # The architecture optimizer settings, architecture learning rate doesn't decay
    # and the minimum lr for architectures is the same as base_lr
    lr_scheduler2, _ = create_scheduler(args, optimizer2, args.epochs, 0, args.arc_lr)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
            # loss_lambda=args.loss_lambda
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)

    if args.output_dir and utils.is_main_process():
        writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
            checkpoint_dict = checkpoint['model']
        elif 'deit_base_patch16_224-b5f2ef4d.pth' in args.resume or \
                'deit_small_patch16_224-cd65a155.pth' in args.resume or \
                'deit_tiny_patch16_224-a1311bcf.pth' in args.resume:
            # Separate qkv into qk and v in pre-trained Swin models
            # Hard-coded swin model names

            checkpoint = torch.load(args.resume, map_location='cpu')
            old_dict = checkpoint['model']
            new_dict = OrderedDict()
            for name in old_dict.keys():
                if 'attn.qkv.' in name:
                    new_dict[name.replace('qkv', 'qk')] = old_dict[name][:old_dict[name].shape[0] // 3 * 2]
                    new_dict[name.replace('qkv', 'v')] = old_dict[name][-old_dict[name].shape[0] // 3:]
                else:
                    new_dict[name] = old_dict[name]

            checkpoint_dict = new_dict
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            checkpoint_dict = checkpoint['model']

        if args.assigned_indicators and args.searching_model:

            if not args.auto_resume and not args.eval:
                # When start fine-tuning, prune ffn layers
                checkpoint_dict = utils.prune_ffn(checkpoint_dict, ffn_indicators)

            msg = model_without_ddp.load_state_dict(checkpoint_dict, strict=False)

        else:
            msg = model_without_ddp.load_state_dict(checkpoint_dict, strict=False)

        print('loading from ', args.resume)
        print(msg)

        if not args.eval and 'optimizer1' in checkpoint and 'lr_scheduler1' in checkpoint and 'epoch' in checkpoint:
            optimizer1.load_state_dict(checkpoint['optimizer1'])
            lr_scheduler1.load_state_dict(checkpoint['lr_scheduler1'])
            optimizer2.load_state_dict(checkpoint['optimizer2'])
            lr_scheduler2.load_state_dict(checkpoint['lr_scheduler2'])
            args.start_epoch = checkpoint['epoch'] + 1
            # if args.model_ema:
            #     utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats, msa_indicators_list, msa_thresholds_list, ffn_indicators_list, flops, kernel_thresholds = evaluate_pruning(data_loader_val, model, device)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    if args.throughput:
        throughput(data_loader_val, model, logger)
        return

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if hasattr(model.module, 'set_epoch'):
            model.module.set_epoch(epoch)

        train_stats, str_threshold_indicators = train_one_epoch_pruning(
            model, criterion, data_loader_train,
            optimizer1, optimizer2, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            set_training_mode=args.finetune == '', logger=logger  # keep in eval mode during finetuning
        )

        lr_scheduler1.step(epoch)
        lr_scheduler2.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'last_checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer1': optimizer1.state_dict(),
                    'lr_scheduler1': lr_scheduler1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    'lr_scheduler2': lr_scheduler2.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats, msa_indicators_list, msa_thresholds_list, ffn_indicators_list, flops, kernel_thresholds = evaluate_pruning(data_loader_val, model, device)

        # Some visualizations for gate parameters
        if args.output_dir and utils.is_main_process() and args.assigned_indicators:

            for i, block in enumerate(kernel_thresholds):
                block_thresholds_dict = {}
                for k, _ in enumerate(block):
                    block_thresholds_dict['{}_{}'.format(i, k)] = kernel_thresholds[i][k]
                writer.add_scalars('train/thresholds_{}'.format('block_' + str(i)), block_thresholds_dict, epoch)

        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if max_accuracy < test_stats["acc1"] and args.assigned_indicators:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'lr_scheduler1': lr_scheduler1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'lr_scheduler2': lr_scheduler2.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            },  os.path.join(args.output_dir, 'best_checkpoint.pth'))

        # Frequency for saving searched FFN indicators
        if (epoch + 1) % 10 == 0 and epoch > 0 and not args.assigned_indicators:
            utils.save_ffn_indicators(model_without_ddp, epoch, logger, args.output_dir)

        max_accuracy = max(max_accuracy, test_stats["acc1"])
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'msa_indicators_list': msa_indicators_list,
                     'msa_thresholds_list': msa_thresholds_list,
                     'ffn_indicators_list': ffn_indicators_list,
                     'flops: ': flops}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # We set the maximum searching epoch to 20
        if epoch >= 20 and not args.assigned_indicators:
            return

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
