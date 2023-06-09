# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Modifications copyright (c) 2021 Zhuang AI Group, Haoyu He

import os
import torch
import torch.distributed as dist

from torch.utils.data import Dataset, Sampler, SubsetRandomSampler, DistributedSampler
from typing import Iterator, List, Optional, Union
from operator import itemgetter
from collections import OrderedDict
import torch.nn.functional as F

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        # config.TRAIN.START_EPOCH = 0
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def save_last_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_latest.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def save_best_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def prune_ffn(state_dict, ffn_indicators, depths):
    checkpoint_dict = state_dict

    idx = 0
    for i in range(len(depths)):

        for j in range(depths[i]):
            assigned_indicator_index = ffn_indicators[idx].nonzero().squeeze(-1)

            in_dim = checkpoint_dict[f'layers.{i}.blocks.{j}.mlp.fc1.weight'].shape[1]
            checkpoint_dict[f'layers.{i}.blocks.{j}.mlp.fc1.weight'] = torch.gather(checkpoint_dict[f'layers.{i}.blocks.{j}.mlp.fc1.weight'], 0,
                                             assigned_indicator_index.unsqueeze(-1).expand(-1, in_dim))
            checkpoint_dict[f'layers.{i}.blocks.{j}.mlp.fc1.bias'] = torch.gather(checkpoint_dict[f'layers.{i}.blocks.{j}.mlp.fc1.bias'], 0,
                                             assigned_indicator_index)
            checkpoint_dict[f'layers.{i}.blocks.{j}.mlp.fc2.weight'] = torch.gather(checkpoint_dict[f'layers.{i}.blocks.{j}.mlp.fc2.weight'], 1,
                                             assigned_indicator_index.unsqueeze(0).expand(in_dim, -1))

            idx += 1

    return checkpoint_dict


def load_checkpoint_pruning(config, model, optimizer1, optimizer2, lr_scheduler1, lr_scheduler2, logger, ffn_indicators, auto_resume=False):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
        checkpoint_dict = checkpoint['model']

    elif 'swin_base_patch4_window7_224.pth' in config.MODEL.RESUME or \
            'swin_small_patch4_window7_224.pth' in config.MODEL.RESUME or \
            'swin_tiny_patch4_window7_224.pth' in config.MODEL.RESUME:
        # Separate qkv into qk and v in pre-trained Swin models
        # Hard-coded swin model names

        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
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
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        checkpoint_dict = checkpoint['model']

    if config.EXTRA.assigned_indicators:
        if not auto_resume and not config.EVAL_MODE:
            logger.info('auto-resuming')
            checkpoint_dict = prune_ffn(checkpoint_dict, ffn_indicators, config.MODEL.SWIN.DEPTHS)

        msg = model.load_state_dict(checkpoint_dict, strict=False)
    else:
        msg = model.load_state_dict(checkpoint_dict, strict=False)

    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer1' in checkpoint \
            and 'lr_scheduler1' in checkpoint and 'epoch' in checkpoint:
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])

        lr_scheduler1.load_state_dict(checkpoint['lr_scheduler1'])
        lr_scheduler2.load_state_dict(checkpoint['lr_scheduler2'])

        config.defrost()
        # config.TRAIN.START_EPOCH = 0
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint_pruning(config, epoch, model, max_accuracy, optimizer1, optimizer2, lr_scheduler1, lr_scheduler2, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer1': optimizer1.state_dict(),
                  'optimizer2': optimizer2.state_dict(),
                  'lr_scheduler1': lr_scheduler1.state_dict(),
                  'lr_scheduler2': lr_scheduler2.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def save_ffn_indicators(config, epoch, model, logger):
    new_dict = OrderedDict()

    old_dict = model.state_dict()
    for name in old_dict.keys():
        if 'assigned_indicator_index' in name:
            new_dict[name] = old_dict[name]

    save_state = {'model': new_dict}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'search_{epoch}epoch.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def save_best_checkpoint_pruning(config, epoch, model, max_accuracy, optimizer1, optimizer2, lr_scheduler1, lr_scheduler2, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer1': optimizer1.state_dict(),
                  'optimizer2': optimizer2.state_dict(),
                  'lr_scheduler1': lr_scheduler1.state_dict(),
                  'lr_scheduler2': lr_scheduler2.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def save_last_checkpoint_pruning(config, epoch, model, max_accuracy, optimizer1, optimizer2, lr_scheduler1, lr_scheduler2, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer1': optimizer1.state_dict(),
                  'optimizer2': optimizer2.state_dict(),
                  'lr_scheduler1': lr_scheduler1.state_dict(),
                  'lr_scheduler2': lr_scheduler2.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_latest.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def output(vis, dirname, fname):
    filepath = os.path.join(dirname, fname)
    print("Saving to {} ...".format(filepath))
    vis.save(filepath)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        # outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_dist = outputs
        else:
            outputs_dist = outputs

        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        # if outputs_kd is None:
        #     raise ValueError("When knowledge distillation is enabled, the model is "
        #                      "expected to return a Tuple[Tensor, Tensor] with the output of the "
        #                      "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if not isinstance(teacher_outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            teacher_outputs, _, _ = teacher_outputs

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_dist / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_dist.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_dist, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
