# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils.utils as utils
import time


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, logger=None, target_flops=3.0, warm_up=False, if_train_arch=False
                    ):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    logger.info_freq = 10
    compression_rate_print_freq = 100

    warm_up_epoch = 1

    if warm_up and epoch < warm_up_epoch:  # for stable training and better performance
        lamb = 20
    else:
        lamb = 20

    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, logger.info_freq, header, logger)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=False):
            outputs, flops = model(samples)

            if not if_train_arch:
                loss_cls = criterion(samples, outputs, targets)
            else:
                loss_cls = criterion(outputs, targets)
            loss_flops = ((flops / 1e9) - target_flops) ** 2
            loss = lamb * loss_flops + loss_cls
            loss_cls_value = loss_cls.item()
            loss_flops_value = loss_flops.item()

        if not math.isfinite(loss_cls_value):
            logger.info("Loss is {}, stopping training".format(loss_cls_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        if if_train_arch:
            if hasattr(model, 'module'):
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.module.arch_parameters(), create_graph=is_second_order)
            else:
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.arch_parameters(), create_graph=is_second_order)
        else:
            if hasattr(model, 'module'):
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.module.parameters(), create_graph=is_second_order)
            else:
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()

        if data_iter_step % compression_rate_print_freq == 0:
            if hasattr(model, 'module'):  # for DDP 
                prune_kept_num = model.module.get_kept_num()
            else:
                prune_kept_num = model.get_kept_num()
            logger.info(f'prune kept number:{prune_kept_num}')

        metric_logger.update(loss_cls=loss_cls_value)
        metric_logger.update(loss_flops=loss_flops_value)
        metric_logger.update(flops=flops / 1e9)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats:{metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_cls(model: torch.nn.Module, criterion,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer, if_distill: bool,
                        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                        mixup_fn: Optional[Mixup] = None,
                        set_training_mode=True, logger=None, if_train_arch=False, model_ema: Optional[ModelEma] = None
                        ):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    logger.info_freq = 10

    # set accumulation_steps
    accumulation_steps = 256 // 128
    print("train_with_distill:", if_distill)
    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, logger.info_freq, header, logger)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(samples)
            if not if_train_arch:
                if if_distill:
                    loss_cls = criterion(samples, outputs, targets)
                else:
                    loss_cls = criterion(outputs, targets)
            else:
                loss_cls = criterion(outputs, targets)
            loss = loss_cls / accumulation_steps
            loss_cls_value = loss_cls.item()

        if not math.isfinite(loss_cls_value):
            logger.info("Loss is {}, stopping training".format(loss_cls_value))
            sys.exit(1)

        # optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        if if_train_arch:
            if hasattr(model, 'module'):
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.module.arch_parameters(), create_graph=is_second_order)
            else:
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.arch_parameters(), create_graph=is_second_order)
        else:
            if hasattr(model, 'module'):
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.module.parameters(), create_graph=is_second_order)
            else:
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order)

        if (data_iter_step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()
        metric_logger.update(loss_cls=loss_cls_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats:{metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, logger=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header, logger):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            output, flops, layer_flops = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        torch.cuda.synchronize()

        batch_size = images.shape[0]
        metric_logger.update(flops=flops / 1e9)
        metric_logger.update(layer_flops=layer_flops / 1e9)
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if hasattr(model, 'module'):  # for DDP 
        prune_kept_num = model.module.get_kept_num()
    else:
        prune_kept_num = model.get_kept_num()
    logger.info(f'prune kept number:{prune_kept_num}')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    logger.info(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} flops {flops.global_avg:.3f} layer_flops {layer_flops.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, flops=metric_logger.flops,
                layer_flops=metric_logger.layer_flops))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_cls(data_loader, model, device, logger=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    merge_time_total = []
    eval_start_time = time.perf_counter()

    for images, target in metric_logger.log_every(data_loader, 10, header, logger):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            output, flops, layer_flops = model(images)
            loss = criterion(output, target)

        acc1, acc3, acc5 = accuracy(output, target, topk=(1, 3, 5))
        torch.cuda.synchronize()

        batch_size = images.shape[0]
        metric_logger.update(flops=flops / 1e9)
        metric_logger.update(layer_flops=layer_flops / 1e9)
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    logger.info(
        '* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} flops {flops.global_avg:.3f} layer_flops {layer_flops.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top3=metric_logger.acc3, top5=metric_logger.acc5, losses=metric_logger.loss,
                flops=metric_logger.flops, layer_flops=metric_logger.layer_flops))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_vit(data_loader, model, device, logger=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header, logger):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        torch.cuda.synchronize()

        batch_size = images.shape[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_teacher(data_loader, model, device, logger=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header, logger):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            output, _ = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        torch.cuda.synchronize()

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_vedio(data_loader, model, device, logger=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header, logger):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        torch.cuda.synchronize()

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
