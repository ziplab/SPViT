# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Modifications copyright (c) 2021 Zhuang AI Group, Haoyu He

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():

            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        # loss.backward()
        # for name, param in model.named_parameters():
        #     if 'thresholds' in name:
        #         print(name, param.grad)

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_pruning(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer1: torch.optim.Optimizer, optimizer2: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, logger=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():

            outputs, msa_indicators_list, msa_thresholds_list, ffn_indicators_list = model(samples)
            loss_cls = criterion(samples, outputs, targets)

            if not model.module.assigned_indicators:
                loss_bop = model.module.calculate_bops_loss()
                loss = loss_cls + loss_bop
            else:
                loss_bop = torch.zeros(1).to(loss_cls.device)
                loss = loss_cls

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order1 = hasattr(optimizer1, 'is_second_order') and optimizer1.is_second_order

        if not model.module.assigned_indicators:
            loss_scaler(loss, optimizer1, optimizer2, clip_grad=max_norm, create_graph=is_second_order1, model=model)
        else:

            # Not using architecture optimizer during fine-tuning
            loss_scaler(loss, optimizer1, None, clip_grad=max_norm, create_graph=is_second_order1, model=model)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(loss_bop=loss_bop.item())
        metric_logger.update(lr=optimizer1.param_groups[0]["lr"])

        str_msa_thresholds = ''
        if not model.module.assigned_indicators and utils.get_rank() == 0:
            str_msa_thresholds = str(
                [["{:.3f}".format(i.item()) for i in blocks] for blocks in msa_thresholds_list])

            logger.info(str_msa_thresholds)

            str_ffn_indicators = str(
                [i.item() for i in ffn_indicators_list])

            logger.info(str_ffn_indicators)

        # break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, str_msa_thresholds


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # metric_logger.log_indicator(indicators)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_pruning(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, msa_indicators_list, msa_thresholds_list, ffn_indicators_list = model(images)
            loss = criterion(output, target)

        # metric_logger.log_indicator(indicators)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        # break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    str_msa_indicators = ''
    str_msa_thresholds = ''
    str_ffn_indicators = ''
    str_flops = ''
    msa_thresholds = []

    # If searching, print some stuff
    if not model.module.assigned_indicators and utils.get_rank() == 0:
        str_msa_indicators = str(
            [[i.item() for i in blocks] for blocks in msa_indicators_list])

        print('str_msa_indicators: ', str_msa_indicators)

        str_msa_thresholds = str(
            [["{:.3f}".format(i.item()) for i in blocks] for blocks in msa_thresholds_list])

        print('str_msa_thresholds: ', str_msa_thresholds)

        str_ffn_indicators = str(
            [i.item() for i in ffn_indicators_list])

        print('str_ffn_indicators: ', str_ffn_indicators)

        str_flops = str("{:.3f}".format(model.module.flops()[0].item() / 1e9))
        print('flops: ', str_flops)

        msa_thresholds = [[i.item() for i in blocks] for blocks in msa_thresholds_list]

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, str_msa_indicators, str_msa_thresholds,\
           str_ffn_indicators, str_flops, msa_thresholds
