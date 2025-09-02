# Path: train.py
# This version is modified to be fully compatible with CPU-only execution.

import warnings
warnings.filterwarnings('ignore')
import argparse, time, yaml, os, logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import torch, torch.nn as nn, torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import ImageDataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, resume_checkpoint
from timm.models.layers import convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import NativeScaler

from data.myloader import create_loader
import pyramid_vig
import vig

# Note: Apex has been removed from this version as it's GPU-specific
has_native_amp = hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast')

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE', help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Add all the arguments... (This part is long and unchanged, so I will list them concisely)
# --- Dataset / Model Parameters
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL', help='Name of model to train')
parser.add_argument('--pretrained', action='store_true', default=False, help='Start with pretrained version')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH', help='Initialize model from this checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Resume full model state')
parser.add_argument('--no-resume-opt', action='store_true', default=False, help='prevent resume of optimizer state')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N', help='number of label classes')
parser.add_argument('--gp', default=None, type=str, metavar='POOL', help='Global pool type')
parser.add_argument('--img-size', type=int, default=None, metavar='N', help='Image patch size')
parser.add_argument('--crop-pct', default=None, type=float, metavar='N', help='Input image center crop percent')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN', help='Override mean pixel value')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD', help='Override std deviation')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME', help='Image resize interpolation type')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N', help='validation batch size multiplier')

# --- Optimizer Parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER', help='Optimizer')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON', help='Optimizer Epsilon')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='Momentum')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm')
parser.add_argument('--clip-mode', type=str, default='norm', help='Gradient clipping mode')

# --- Learning Rate Schedule Parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER', help='LR scheduler')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT', help='learning rate cycle len multiplier')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N', help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR', help='warmup learning rate')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N', help='manual epoch number')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N', help='epochs to warmup LR')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate')

# --- Augmentation & Regularization Parameters
parser.add_argument('--no-aug', action='store_true', default=False, help='Disable all training augmentation')
parser.add_argument('--repeated-aug', action='store_true', default=False)
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT', help='Random resize scale')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO', help='Random resize aspect ratio')
parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip probability')
parser.add_argument('--vflip', type=float, default=0., help='Vertical flip probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor')
parser.add_argument('--aa', type=str, default=None, metavar='NAME', help='Use AutoAugment policy')
parser.add_argument('--aug-splits', type=int, default=0, help='Number of augmentation splits')
parser.add_argument('--jsd', action='store_true', default=False, help='Enable Jensen-Shannon Divergence')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT', help='Random erase prob')
parser.add_argument('--remode', type=str, default='const', help='Random erase mode')
parser.add_argument('--recount', type=int, default=1, help='Random erase count')
parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first split')
parser.add_argument('--mixup', type=float, default=0.0, help='mixup alpha')
parser.add_argument('--cutmix', type=float, default=0.0, help='cutmix alpha')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio')
parser.add_argument('--mixup-prob', type=float, default=1.0, help='mixup/cutmix probability')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='mixup/cutmix switch probability')
parser.add_argument('--mixup-mode', type=str, default='batch', help='mixup/cutmix mode')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N', help='turn off mixup after this epoch')
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing')
parser.add_argument('--train-interpolation', type=str, default='random', help='Training interpolation')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT', help='Drop path rate')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT', help='Drop block rate')

# --- Batch Norm Parameters
parser.add_argument('--bn-tf', action='store_true', default=False, help='Use Tensorflow BatchNorm defaults')
parser.add_argument('--bn-momentum', type=float, default=None, help='BatchNorm momentum override')
parser.add_argument('--bn-eps', type=float, default=None, help='BatchNorm epsilon override')
parser.add_argument('--sync-bn', action='store_true', help='Enable synchronized BatchNorm')
parser.add_argument('--dist-bn', type=str, default='', help='Distribute BatchNorm stats')
parser.add_argument('--split-bn', action='store_true', help='Enable separate BN layers per augmentation split')

# --- Model EMA
parser.add_argument('--model-ema', action='store_true', default=False, help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='Force ema to be tracked on CPU')
parser.add_argument('--model-ema-decay', type=float, default=0.9998, help='model ema decay')

# --- Misc
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=50, metavar='N', help='log interval')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N', help='recovery checkpoint interval')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N', help='data loading workers')
parser.add_argument('--save-images', action='store_true', default=False, help='save images of input batches')
parser.add_argument('--amp', action='store_true', default=False, help='use mixed precision training')
parser.add_argument('--channels-last', action='store_true', default=False, help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False, help='Pin CPU memory in DataLoader')
parser.add_argument('--no-prefetcher', action='store_true', default=False, help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH', help='path to output folder')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC', help='Best metric')
parser.add_argument('--tta', type=int, default=0, metavar='N', help='Test time augmentation')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False, help='use multi-epochs-loader')
parser.add_argument("--pretrain_path", default=None, type=str)
parser.add_argument("--evaluate", action='store_true', default=False, help='evaluate model')

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    setup_default_logging()
    args, args_text = _parse_args()
    args.prefetcher = not args.no_prefetcher
    
    # --- MODIFICATION FOR CPU/GPU ---
    # Automatically select device
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1

    # Setup distributed training if applicable and CUDA is available
    if args.distributed:
        assert torch.cuda.is_available(), "Distributed training requires CUDA."
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(f'Training in distributed mode with {args.world_size} GPUs.')
    else:
        args.world_size = 1
        args.rank = 0
        _logger.info(f'Training with a single process on {args.device}.')
    # --- END OF MODIFICATION ---

    torch.manual_seed(args.seed + args.rank)

    model = create_model(
        args.model, pretrained=args.pretrained, num_classes=args.num_classes, drop_rate=args.drop,
        drop_path_rate=args.drop_path, drop_block_rate=args.drop_block, global_pool=args.gp,
        bn_momentum=args.bn_momentum, bn_eps=args.bn_eps, checkpoint_path=args.initial_checkpoint
    )
    if args.pretrain_path:
        state_dict = torch.load(args.pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        _logger.info(f'Loaded pretrain weights from {args.pretrain_path}')
        
    if args.rank == 0:
        _logger.info(f'Model {args.model} created, param count: {sum(p.numel() for p in model.parameters())}')
        
    data_config = resolve_data_config(vars(args), model=model, verbose=args.rank == 0)

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
        
    model.to(args.device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        
    optimizer = create_optimizer_v2(model, opt=args.opt, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    amp_autocast = suppress
    loss_scaler = None
    # AMP is only for CUDA
    if args.amp and args.device == 'cuda':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        _logger.info('Using native Torch AMP. Training in mixed precision.')

    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume, optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler, log_info=args.rank == 0
        )

    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None,
            resume=args.resume
        )
        
    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = NativeDDP(model, device_ids=[args.local_rank])

    lr_scheduler, num_epochs = create_scheduler_v2(
    optimizer,
    sched=args.sched,
    num_epochs=args.epochs,
    min_lr=args.min_lr,
    warmup_lr=args.warmup_lr,
    warmup_epochs=args.warmup_epochs,      # Changed from warmup_t
    cooldown_epochs=args.cooldown_epochs,  # Changed from cooldown_t
    patience_epochs=args.patience_epochs,  # Changed from patience_t
    decay_epochs=args.decay_epochs,        # Changed from decay_t
    decay_rate=args.decay_rate,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.rank == 0:
        _logger.info(f'Scheduled epochs: {num_epochs}')
        
    train_dir = os.path.join(args.data, 'train')
    eval_dir = os.path.join(args.data, 'val')
    if not os.path.exists(eval_dir): eval_dir = os.path.join(args.data, 'validation')
    
    dataset_train = ImageDataset(train_dir)
    dataset_eval = ImageDataset(eval_dir)

    collate_fn, mixup_fn = None, None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes
        )
        if args.prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    train_interpolation = args.train_interpolation if args.no_aug else data_config['interpolation']
    
    loader_train = create_loader(
        dataset_train, input_size=data_config['input_size'], batch_size=args.batch_size, is_training=True,
        use_prefetcher=args.prefetcher, no_aug=args.no_aug, re_prob=args.reprob, re_mode=args.remode,
        re_count=args.recount, re_split=args.resplit, scale=args.scale, ratio=args.ratio,
        hflip=args.hflip, vflip=args.vflip, color_jitter=args.color_jitter, auto_augment=args.aa,
        num_aug_splits=num_aug_splits, interpolation=train_interpolation, mean=data_config['mean'],
        std=data_config['std'], num_workers=args.workers, distributed=args.distributed,
        collate_fn=collate_fn, pin_memory=args.pin_mem, use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    loader_eval = create_loader(
        dataset_eval, input_size=data_config['input_size'], batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False, use_prefetcher=args.prefetcher, interpolation=data_config['interpolation'],
        mean=data_config['mean'], std=data_config['std'], num_workers=args.workers,
        distributed=args.distributed, crop_pct=data_config['crop_pct'], pin_memory=args.pin_mem
    )
    
    if args.jsd: train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).to(args.device)
    elif mixup_active: train_loss_fn = SoftTargetCrossEntropy().to(args.device)
    elif args.smoothing: train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(args.device)
    else: train_loss_fn = nn.CrossEntropyLoss().to(args.device)
    validate_loss_fn = nn.CrossEntropyLoss().to(args.device)

    if args.evaluate:
        validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
        return

    output_dir = ''
    if args.rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), args.model, str(data_config['input_size'][-1])])
        output_dir = get_outdir(output_base, 'train', exp_name)
    
    saver = None
    if args.rank == 0:
        decreasing = True if args.eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=5
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    best_metric, best_epoch = None, None
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed: loader_train.sampler.set_epoch(epoch)
            
            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler, saver, amp_autocast, loss_scaler, model_ema, mixup_fn
            )
            
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
                
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema.module, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)'
                )
                eval_metrics = ema_eval_metrics
                
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[args.eval_metric])

            if output_dir and args.rank == 0:
                update_summary(epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'), write_header=best_metric is None)

            if saver is not None:
                save_metric = eval_metrics[args.eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt: pass
    if best_metric is not None and args.rank == 0:
        _logger.info(f'*** Best metric: {best_metric} (epoch {best_epoch})')

def train_epoch(epoch, model, loader, optimizer, loss_fn, args, lr_scheduler, saver, amp_autocast, loss_scaler, model_ema, mixup_fn):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if hasattr(loader, 'mixup_enabled') and loader.mixup_enabled: loader.mixup_enabled = False
        if mixup_fn is not None: mixup_fn.mixup_enabled = False

    batch_time_m, data_time_m, losses_m = AverageMeter(), AverageMeter(), AverageMeter()
    model.train()
    end = time.time()
    last_idx = len(loader) - 1
    
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        input, target = input.to(args.device), target.to(args.device)
        if mixup_fn is not None: input, target = mixup_fn(input, target)
        if args.channels_last: input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        if not args.distributed: losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad, model=model)
        else:
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        
        if args.device == 'cuda': torch.cuda.synchronize()
        if model_ema is not None: model_ema.update(model)
        
        batch_time_m.update(time.time() - end)
        
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  LR: {lr:.3e}  Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch, batch_idx, len(loader), 100. * batch_idx / last_idx, loss=losses_m,
                        batch_time=batch_time_m, rate=input.size(0) * args.world_size / batch_time_m.val, lr=lr, data_time=data_time_m
                    )
                )

        if saver is not None and args.recovery_interval and (last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)
            
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=epoch * len(loader) + batch_idx)
            
        end = time.time()
        
    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m, losses_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input, target = input.to(args.device), target.to(args.device)
            if args.channels_last: input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)): output = output[0]

            if args.tta > 1:
                output = output.unfold(0, args.tta, args.tta).mean(dim=2)
                target = target[0:target.size(0):args.tta]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss, acc1, acc5 = reduce_tensor(loss.data, args.world_size), reduce_tensor(acc1, args.world_size), reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
                
            if args.device == 'cuda': torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m, loss=losses_m, top1=top1_m, top5=top5_m
                    )
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    return metrics

if __name__ == '__main__':
    main()