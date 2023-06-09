# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Modifications copyright (c) 2021 Zhuang AI Group, Haoyu He

import os
import random
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_our_optimizer_2params
from logger import create_logger
from utils import load_checkpoint_pruning, get_grad_norm, auto_resume_helper, \
    reduce_tensor, save_best_checkpoint_pruning, save_last_checkpoint_pruning, \
    save_ffn_indicators
from tensorboardX import SummaryWriter
from timm.models import create_model
from utils import DistillationLoss

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    # Load searched architectures
    if config.EXTRA.assigned_indicators or config.EXTRA.searching_model:
        # Must have both msa_indicators and ffn_indicators
        assert (config.EXTRA.assigned_indicators and config.EXTRA.searching_model)

        logger.info("Fine-tuning from searched architecture...")

        logger.info(f"MSA indicators: {config.EXTRA.assigned_indicators}")
        logger.info(f"FFN indicator path: {config.EXTRA.searching_model}")

        ffn_indicators = []
        names = []

        # Get the FFN indicators
        if config.EXTRA.searching_model:
            searching_checkpoint = torch.load(config.EXTRA.searching_model, map_location='cpu')

            for name in searching_checkpoint['model'].keys():
                if 'assigned_indicator_index' in name:
                    ffn_indicators.append(searching_checkpoint['model'][name])
                    names.append(name)

    else:
        ffn_indicators = None
        logger.info("Searching...")

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, ffn_indicators)
    model.cuda()
    logger.info(str(model))

    optimizer1, optimizer2 = build_our_optimizer_2params(config, model)

    if config.AMP_OPT_LEVEL != "O0":
        model, [optimizer1, optimizer2] = amp.initialize(model, [optimizer1, optimizer2], opt_level=config.AMP_OPT_LEVEL)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
                                                      broadcast_buffers=False, find_unused_parameters=False)

    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    lr_scheduler1 = build_scheduler(config, optimizer1, len(data_loader_train))
    lr_scheduler2 = build_scheduler(config, optimizer2, len(data_loader_train), num_steps=int(config.EXTRA.arc_decay * len(data_loader_train)),
                                    warmup_steps=int(config.EXTRA.arc_warmup * len(data_loader_train)), min_lr=config.EXTRA.arc_min_lr)

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if config.EXTRA.distillation_type != 'none':
        assert config.EXTRA.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {config.EXTRA.teacher_model}")
        teacher_model = create_model(
            config.EXTRA.teacher_model,
            pretrained=False,
            num_classes=config.MODEL.NUM_CLASSES,
            global_pool='avg',
        )
        if config.EXTRA.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.EXTRA.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(config.EXTRA.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to('cuda')
        teacher_model.eval()

    criterion = DistillationLoss(
        criterion, teacher_model, config.EXTRA.distillation_type, config.EXTRA.distillation_alpha, config.EXTRA.distillation_tau
    )

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint_pruning(config, model_without_ddp, optimizer1, optimizer2, lr_scheduler1,
                                                      lr_scheduler2, logger, ffn_indicators,
                                                      auto_resume=bool(resume_file))

        acc1, acc5, loss = validate(config, data_loader_val, model, -1)

        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        if hasattr(model.module, 'set_epoch'):
            model.module.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer1, optimizer2, epoch, mixup_fn, lr_scheduler1, lr_scheduler2)
        if dist.get_rank() == 0 and (epoch % 1 == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_last_checkpoint_pruning(config, epoch, model_without_ddp, max_accuracy, optimizer1, optimizer2, lr_scheduler1, lr_scheduler2, logger)

            # Save FFN indicators
            if epoch % 1 == 0 and not config.EXTRA.assigned_indicators and epoch!=0:
                save_ffn_indicators(config, epoch, model_without_ddp, logger)

        acc1, acc5, loss = validate(config, data_loader_val, model, epoch)

        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if dist.get_rank() == 0 and max_accuracy < acc1 and config.EXTRA.assigned_indicators:
            save_best_checkpoint_pruning(config, epoch, model_without_ddp, max_accuracy, optimizer1, optimizer2, lr_scheduler1, lr_scheduler2, logger)
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        # End searching
        if epoch >= 20 and not config.EXTRA.assigned_indicators:
            return

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer1, optimizer2, epoch, mixup_fn, lr_scheduler1, lr_scheduler2):
    model.train()
    optimizer1.zero_grad()
    optimizer2.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    bop_loss_meter = AverageMeter()

    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs, msa_indicator_list, msa_threshold_list, ffn_indicator_list, ffn_threshold_list = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            cls_loss = criterion(samples, outputs, targets)

            if config.EXTRA.assigned_indicators:
                loss = cls_loss / config.TRAIN.ACCUMULATION_STEPS
                bop_loss = torch.zeros_like(cls_loss)
            else:
                bop_loss = model.module.calculate_bops_loss()
                loss = (bop_loss + cls_loss) / config.TRAIN.ACCUMULATION_STEPS

            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, [optimizer1, optimizer2]) as scaled_loss:
                    scaled_loss.backward()

                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer1), config.TRAIN.CLIP_GRAD)
                    _ = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer2), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer1))
                    _ = get_grad_norm(amp.master_params(optimizer2))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer1), config.TRAIN.CLIP_GRAD)
                    _ = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer2), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer1))
                    _ = get_grad_norm(amp.master_params(optimizer2))

            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:

                optimizer1.step()
                optimizer1.zero_grad()

                optimizer2.step()
                optimizer2.zero_grad()

                lr_scheduler1.step_update(epoch * num_steps + idx)
                lr_scheduler2.step_update(epoch * num_steps + idx)
        else:
            cls_loss = criterion(samples, outputs, targets)
            if config.EXTRA.assigned_indicators:
                bop_loss = torch.zeros_like(cls_loss)
                loss = cls_loss
            else:
                bop_loss = model.module.calculate_bops_loss()
                loss = cls_loss + bop_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, [optimizer1, optimizer2]) as scaled_loss:
                    scaled_loss.backward()

                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer1), config.TRAIN.CLIP_GRAD)
                    _ = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer2), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer1))
                    _ = get_grad_norm(amp.master_params(optimizer2))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer1), config.TRAIN.CLIP_GRAD)
                    _ = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer2), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer1))
                    _ = get_grad_norm(amp.master_params(optimizer2))

            optimizer1.step()
            optimizer2.step()

            lr_scheduler1.step_update(epoch * num_steps + idx)
            lr_scheduler2.step_update(epoch * num_steps + idx)

        # break

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        cls_loss_meter.update(cls_loss.item(), targets.size(0))
        bop_loss_meter.update(bop_loss.item(), targets.size(0))

        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer1.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'cls_loss {cls_loss_meter.val:.4f} ({cls_loss_meter.avg:.4f})\t'
                f'bop_loss {bop_loss_meter.val:.4f} ({bop_loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            if not config.EXTRA.assigned_indicators:

                str_ffn_indicators = str(
                    [[sum([i.item() for i in block_list]) for block_list in layer_list] for layer_list in
                     ffn_indicator_list])
                logger.info('ffn_indicators: ' + str_ffn_indicators)

                str_msa_thresholds = str(
                    [[["{:.3f}".format(i.item()) for i in block_list] for block_list in layer_list] for layer_list in msa_threshold_list])

                logger.info('msa_thresholds: ' + str_msa_thresholds)

    if dist.get_rank() == 0:
        writer.add_scalar('train/loss', loss.item(), epoch)
        writer.add_scalar('train/cls_loss', cls_loss_meter.val, epoch)
        writer.add_scalar('train/bop_loss', bop_loss_meter.val, epoch)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, epoch):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    bop_loss_meter = AverageMeter()

    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output, msa_indicator_list, msa_threshold_list, ffn_indicator_list, ffn_threshold_list = model(images)

        # measure accuracy and record loss
        cls_loss = criterion(output, target)

        if config.EXTRA.assigned_indicators:
            bop_loss = torch.zeros_like(cls_loss)
            loss = cls_loss
        else:
            bop_loss = model.module.calculate_bops_loss()
            loss = cls_loss + bop_loss

        # break

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        cls_loss = reduce_tensor(cls_loss)
        bop_loss = reduce_tensor(bop_loss)

        loss_meter.update(loss.item(), target.size(0))
        cls_loss_meter.update(cls_loss.item(), target.size(0))
        bop_loss_meter.update(bop_loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'cls_loss {cls_loss_meter.val:.4f} ({cls_loss_meter.avg:.4f})\t'
                f'bop_loss {bop_loss_meter.val:.4f} ({bop_loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

            if not config.EXTRA.assigned_indicators:
                str_msa_indicators = str(
                    [[[i.item() for i in block_list] for block_list in layer_list] for layer_list in
                     msa_indicator_list])
                logger.info('msa_indicators: ' + str_msa_indicators)

                str_ffn_indicators = str(
                    [[sum([i.item() for i in block_list]) for block_list in layer_list] for layer_list in
                     ffn_indicator_list])
                logger.info('ffn_indicators: ' + str_ffn_indicators)

                str_msa_thresholds = str(
                    [[["{:.3f}".format(i.item()) for i in block_list] for block_list in layer_list] for layer_list in msa_threshold_list])

                logger.info('msa_thresholds: ' + str_msa_thresholds)

    if not config.EXTRA.assigned_indicators:
        logger.info('Flops: ' + str(model.module.flops()[0].item() / 1e9))

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    if dist.get_rank() == 0:
        writer.add_scalar('val/acc1', acc1_meter.avg, epoch)
        writer.add_scalar('val/acc5', acc5_meter.avg, epoch)
        writer.add_scalar('val/loss', loss_meter.avg, epoch)

        if not config.EXTRA.assigned_indicators:

            msa_thresholds = [[[i.item() for i in block_list] for block_list in layer_list] for layer_list in msa_threshold_list]

            for i, layer in enumerate(msa_thresholds):
                for j, block in enumerate(layer):
                    block_thresholds_dict = {}
                    for k, _ in enumerate(block):
                        block_thresholds_dict['{}_{}_{}'.format(i, j, k)] = msa_thresholds[i][j][k]
                    writer.add_scalars('train/msa_thresholds_{}_{}'.format('layer_' + str(i), 'block_' + str(j)),
                                       block_thresholds_dict, epoch)

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


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


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_arc_lr = config.EXTRA.architecture_lr * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_arc_min_lr = config.EXTRA.arc_min_lr * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_arc_lr = linear_scaled_arc_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_arc_min_lr = linear_scaled_arc_min_lr * config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.EXTRA.architecture_lr = linear_scaled_arc_lr
    config.EXTRA.arc_min_lr = linear_scaled_arc_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(config.OUTPUT, 'tensorboard'))

        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
