# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Modifications copyright (c) 2021 Zhuang AI Group, Haoyu He

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
from logger import logger
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.plateau_lr import PlateauLRScheduler
from torch import optim as optim
from timm.optim.lookahead import Lookahead
from collections import OrderedDict

try:
    from apex.optimizers import FusedAdam
    has_apex = True
except ImportError:
    has_apex = False


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

    def log_indicator(self, info):
        logger.info('indicators: ' + str(info))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        logger.info('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    logger.info('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url))
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def create_scheduler(args, optimizer, epochs, warmup_epochs, min_lr):
    num_epochs = epochs

    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(args, 'lr_cycle_mul', 1.),
            lr_min=args.min_lr,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=warmup_epochs,
            cycle_limit=getattr(args, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(args, 'lr_cycle_mul', 1.),
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=getattr(args, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
    elif args.sched == 'plateau':
        mode = 'min' if 'loss' in getattr(args, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=args.decay_rate,
            patience_t=args.patience_epochs,
            lr_min=args.min_lr,
            mode=mode,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cooldown_t=0,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )

    return lr_scheduler, num_epochs


def add_weight_decay_2ops(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    diff_lr = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'thresholds' in name:
            diff_lr.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}], [
        {'params': diff_lr, 'weight_decay': 0.}]


def create_2optimizers(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters1, parameters2 = add_weight_decay_2ops(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args1 = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args1['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args1['betas'] = args.opt_betas

    opt_args2 = dict(lr=args.arc_lr, weight_decay=0.)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args2['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args2['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'adamw':
        optimizer1 = optim.AdamW(parameters1, **opt_args1)
        optimizer2 = optim.AdamW(parameters2, **opt_args2)

    elif opt_lower == 'fusedadamw':
        optimizer1 = FusedAdam(parameters1, adam_w_mode=True, **opt_args1)
        optimizer2 = FusedAdam(parameters2, adam_w_mode=True, **opt_args2)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer1, optimizer2


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer1, optimizer2, clip_grad=None, parameters=None, create_graph=False, model=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        self._scaler.step(optimizer1)

        if optimizer2:
            self._scaler.step(optimizer2)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def prune_ffn(checkpoint_dict, ffn_indicators):

    depth = 12

    for i in range(depth):
        assigned_indicator_index = ffn_indicators[i].nonzero().squeeze(-1)

        in_dim = checkpoint_dict[f'blocks.{i}.mlp.fc1.weight'].shape[1]
        checkpoint_dict[f'blocks.{i}.mlp.fc1.weight'] = torch.gather(checkpoint_dict[f'blocks.{i}.mlp.fc1.weight'], 0,
                                         assigned_indicator_index.unsqueeze(-1).expand(-1, in_dim))
        checkpoint_dict[f'blocks.{i}.mlp.fc1.bias'] = torch.gather(checkpoint_dict[f'blocks.{i}.mlp.fc1.bias'], 0,
                                         assigned_indicator_index)
        checkpoint_dict[f'blocks.{i}.mlp.fc2.weight'] = torch.gather(checkpoint_dict[f'blocks.{i}.mlp.fc2.weight'], 1,
                                         assigned_indicator_index.unsqueeze(0).expand(in_dim, -1))

        i += 1
    return checkpoint_dict


def save_ffn_indicators(model, epoch, logger, output_dir):
    new_dict = OrderedDict()

    old_dict = model.state_dict()
    for name in old_dict.keys():
        if 'assigned_indicator_index' in name:
            new_dict[name] = old_dict[name]

    save_path = os.path.join(output_dir, f'search_{epoch}epoch.pth')
    logger.info(f"{save_path} saving......")
    save_on_master({
        'model': model.state_dict()
    }, save_path)
    logger.info(f"{save_path} saved !!!")
