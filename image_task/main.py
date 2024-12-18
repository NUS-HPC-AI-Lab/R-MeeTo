import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler, get_state_dict, ModelEma
import timm

from utils.datasets import build_dataset
from utils.engine import evaluate_teacher, evaluate_cls, train_one_epoch_cls, evaluate_vit
from utils.engine import evaluate_vedio
from utils.samplers import RASampler
import utils.utils as utils
import shutil
import warnings
from utils.utils import MultiEpochsDataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler

from utils.losses import DistillDiffPruningLoss_dynamic

from utils.calc_flops import throughput, get_flops
from fvcore.nn import FlopCountAnalysis

from R_MeeTo import RMeeTo_Mamba, RMeeTo_Mamba_Video, RMeeToVisionTransformer, R_MeeTo_Mamba_shuffle

warnings.filterwarnings('ignore')
import random


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def get_args_parser():
    parser = argparse.ArgumentParser('Diffrate training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--multi-reso', default=False, action='store_true', help='')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # mea
    parser.add_argument('--model_ema_decay', type=float, default=0.99996, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--model_ema', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='weight decay (default: 0.00)')
    # git
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--clip_grad', type=float, default=0.0, help='gradient clipping')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'SUBIMNET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./log/temp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--autoresume', action='store_true', help='auto resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--port', default="15662", type=str,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--target_flops', type=float, default=3.0)
    parser.add_argument('--granularity', type=int, default=4,
                        help='the token number gap between each compression rate candidate')
    parser.add_argument('--load_compression_rate', action='store_true',
                        help='eval by exiting compression rate in compression_rate.json')
    parser.add_argument('--warmup_compression_rate', action='store_true', default=False,
                        help='inactive computational constraint in first epoch')

    # distill
    parser.add_argument('--distill', default=False, help='distill')
    # throughput
    parser.add_argument('--throughput', default=False, help='throughput')
    # eval
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # merge para
    # merge method
    parser.add_argument('--merge_method', default='max', help='merge method')
    # merge_interval
    parser.add_argument('--merge_interval', default=2, type=int, help='merge interval')
    # if pruning
    parser.add_argument('--if_pruning', action='store_true', help='if pruning')
    # prune num for every layer
    parser.add_argument('--num_prune', default=0, help='num prune')
    # feature importance metric
    parser.add_argument('--metric', default='X', help='importance metric')
    # distance
    parser.add_argument('--distance', default='cosine', help='distance')
    # if order
    parser.add_argument('--if_order', action='store_true', help='if order')
    # if random
    parser.add_argument('--if_random', action='store_true', help='if random')
    # pretrained_pth
    parser.add_argument('--model_pth', default='./pretrained', help='model path')
    # if_merge_odd
    parser.add_argument('--if_merge_odd', action='store_true', help='if merge odd')
    # merge mode
    parser.add_argument('--merge_mode', default='sum', help='merge mode')
    # shuffle
    parser.add_argument('--if_shuffle', action='store_true', help='shuffle')
    parser.add_argument('--shuffle_rate', default=0.0, type=float, help='shuffle rate')
    # choose
    parser.add_argument('--choose', default='max', help='choose')
    # merge compare
    parser.add_argument('--compare', type=int, default=11, help='compare')
    # subset
    parser.add_argument('--data_ratio', type=float, default=1.0, help='data ratio')
    parser.add_argument('--data_seed', type=int, default=0, help='data seed')

    return parser


def main(args):
    utils.init_distributed_mode(args)

    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir, dist_rank=utils.get_rank())
    logger.info(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

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
                logger.info(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # leveraging MultiEpochsDataLoader for faster data loading
    data_loader_train = MultiEpochsDataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,

    )

    data_loader_val = MultiEpochsDataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
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

    logger.info(f"Creating model: {args.model}")

    print('if_merge_odd:', args.if_merge_odd)
    merge_para = {
        'num_prune': int(args.num_prune),
        'distance': args.distance,
        'metric': args.metric,
        'merge_interval': args.merge_interval,
        'if_prune': args.if_pruning,
        'if_order': args.if_order,
        'class_token': True,
        'if_merge': True,
        'if_merge_odd': args.if_merge_odd,
        'merge_mode': args.merge_mode,
        'choose': args.choose,
        'compare': args.compare,
    }

    if 'shuffle' in args.model:
        merge_para['if_shuffle'] = args.if_shuffle
        merge_para['shuffle_rate'] = args.shuffle_rate

    print("Merge para:", merge_para)

    model = create_model(
        args.model,
        pretrained=True,
        model_pth=args.model_pth,
        merge_para=merge_para
    )

    model_f = create_model(
        args.model,
        pretrained=True,
        model_pth=args.model_pth,
        merge_para=merge_para
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:  # remove head
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
        model.load_state_dict(checkpoint_model, strict=False)

    # model ema
    model.to(device)
    model_ema = None
    print("If: model_ema", args.model_ema)
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        checkpoint = torch.load(f'{args.model_pth}/vim_b_midclstok_81p9acc.pth', map_location='cpu')
        utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    loss_scaler = utils.NativeScalerWithGradNormCount()
    lr_scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=args.min_lr, decay_rate=args.decay_rate)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.autoresume and os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("start_epoch", args.start_epoch)
            args.epoch = args.epochs - args.start_epoch
            print("epoch", args.epoch)
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.distill:
        if 'shuffle' in args.model:
            teacher_name = args.model.replace('_shuffle', '') + '_teacher'
        else:
            teacher_name = args.model + '_teacher'

        teacher_model = create_model(
            teacher_name,
            pretrained=True,
            model_pth=args.model_pth
        )

        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema_teacher = ModelEma(
                teacher_model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')
            utils._load_checkpoint_for_ema(model_ema_teacher, checkpoint['model_ema'])
            model_ema_teacher.ema.to(device)
            print("Test ema_teacher model")
            test_stats = evaluate_teacher(data_loader_val, model_ema_teacher.ema, device, logger=logger)
        teacher_model = teacher_model.to(device)

        # test_stats = evaluate_teacher(data_loader_val, teacher_model, device,logger=logger)

        criterion = DistillDiffPruningLoss_dynamic(
            teacher_model, criterion, clf_weight=1.0, mse_token=True, distill_weight=0.5
        )

    if args.eval:
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            checkpoint['model'] = {('module.' + k): v for k, v in checkpoint['model'].items()}
            model.load_state_dict(checkpoint['model'])

        # add ema load
        if args.model_ema:
            utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

        if args.throughput:
            images = torch.randn(128, 3, args.input_size, args.input_size).to(device)
            teahcer_name = args.model + '_teacher'
            teacher_model_f = create_model(
                teahcer_name,
                pretrained=True,
                model_pth=args.model_pth
            )

            teacher_model_f = teacher_model_f.to(device)
            model_f = model_f.to(device)
            # get flops
            # get_flops(model=teacher_model_f,show_detail=True)
            get_flops(model=model_f, show_detail=True)
            print("throughput for teacher model")
            throughput(images, teacher_model_f)
            print("throughput for student model")
            throughput(images, model_f)
            return
        model.to(device)
        if 'deit' not in args.model:
            if args.model_ema:
                test_stats_ema = evaluate_cls(data_loader_val, model_ema.ema, device, logger)
            else:
                test_stats = evaluate_cls(data_loader_val, model, device, logger)
        else:
            test_stats = evaluate_vit(data_loader_val, model, device, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

        # re-training
    logger.info(f"Start training for {args.epochs - args.start_epoch} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.model_ema:
            train_stats = train_one_epoch_cls(
                model, criterion, data_loader_train,
                optimizer, args.distill,
                device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
                logger=logger,
            )
        else:
            train_stats = train_one_epoch_cls(
                model,
                criterion, data_loader_train,
                optimizer, args.distill,
                device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
                logger=logger,
                model_ema=model_ema
            )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                if not args.model_ema:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'model_ema': get_state_dict(model_ema),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        if not args.model_ema:
            # if 'video' in args.model:
            #     test_stats = evaluate_vedio(data_loader_val, model, device, logger=logger)
            # else:
            test_stats = evaluate_cls(data_loader_val, model, device, logger=logger)
        else:
            # test_stats = evaluate_cls(data_loader_val, model, device,logger=logger)
            test_stats = evaluate_cls(data_loader_val, model_ema.ema, device, logger=logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if utils.is_main_process() and max_accuracy < test_stats['acc1']:
            shutil.copyfile(checkpoint_path, f'{args.output_dir}/model_best.pth')
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Finetune time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
