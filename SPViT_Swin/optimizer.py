# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Modifications copyright (c) 2021 Zhuang AI Group, Haoyu He

from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords, config.EXTRA.small_weight_decay_num)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def build_our_optimizer_2params(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters1, parameters2 = set_weight_decay_and_lr_2parameters(model, skip, skip_keywords, config.TRAIN.BASE_LR, config.EXTRA.architecture_lr)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer1 = None
    optimizer2 = None

    if opt_lower == 'sgd':
        optimizer1 = optim.SGD(parameters1, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer1 = optim.AdamW(parameters1, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

        optimizer2 = optim.AdamW(parameters2, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer1, optimizer2


def build_our_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay_and_lr(model, skip, skip_keywords, config.TRAIN.BASE_LR, config.EXTRA.architecture_lr)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()

    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':

        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=(), small_decay_num=0.0):
    has_decay = []
    no_decay = []
    small_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (len(param.shape) == 1 and 'thresholds' not in name) or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        elif 'thresholds' in name:
            small_decay.append(param)
        else:
            has_decay.append(param)

    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.},
            {'params': small_decay, 'weight_decay': small_decay_num}]


def set_weight_decay_and_lr_2parameters(model, skip_list=(), skip_keywords=(), regular_lr=None, diff_lr=None):

    has_decay = []
    no_decay = []
    has_diff_lr = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'thresholds' in name:
            has_diff_lr.append(param)
        else:
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                no_decay.append(param)
                # print(f"{name} has no weight decay")
            else:
                has_decay.append(param)

    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}], \
           [{'params': has_diff_lr, 'lr': diff_lr, 'weight_decay': 0.}]


def set_weight_decay_and_lr(model, skip_list=(), skip_keywords=(), regular_lr=None, diff_lr=None):

    has_decay = []
    no_decay = []
    has_diff_lr = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'thresholds' in name:
            has_diff_lr.append(param)
        else:
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                no_decay.append(param)
                # print(f"{name} has no weight decay")
            else:
                has_decay.append(param)

    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.},
            {'params': has_diff_lr, 'lr': diff_lr, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
