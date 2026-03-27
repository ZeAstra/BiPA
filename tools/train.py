# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append(sys.path[0] + '/..')
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo
from torch.optim import Adam
from mmengine.hooks import Hook


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve training speed.
    setup_cache_size_limit_of_dynamo()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # Enable AMP
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # Enable automatic LR scaling
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and 'enable' in cfg.auto_scale_lr and 'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Missing auto_scale_lr configurations.')

    # Resume or auto-resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # Build runner
    runner = Runner.from_cfg(cfg) if 'runner_type' not in cfg else RUNNERS.build(cfg)

    # Add no_mask_embed parameters to optimizer
    add_no_mask_embed_to_optimizer(runner)

    # Register hook
    # runner.register_hook(CustomHook())

    # Start training
    runner.train()


def add_no_mask_embed_to_optimizer(runner):
    no_mask_embed_params = []

    # Find no_mask_embed params in model
    for name, param in runner.model.named_parameters():
        if 'roi_head.mask_head.no_mask_embed' in name:
            no_mask_embed_params.append(param)

    if not no_mask_embed_params:
        runner.logger.warning("no_mask_embed not found in model.")
        return

    # Get optimizer param groups
    param_groups = getattr(runner.optim_wrapper.optimizer, 'param_groups', None)

    if param_groups is None:
        runner.logger.error("No param_groups found in optimizer.")
        return

    # Ensure no duplicate no_mask_embed params
    already_in_optimizer = any(
        any(id(param) == id(p) for p in pg['params']) for pg in param_groups for param in no_mask_embed_params
    )

    if not already_in_optimizer:
        runner.logger.info("Adding no_mask_embed to optimizer param groups.")
        runner.optim_wrapper.optimizer.add_param_group({
            'params': no_mask_embed_params,
            'lr': 10 * runner.optim_wrapper.optimizer.defaults['lr']
        })
    else:
        runner.logger.info("no_mask_embed already in optimizer param groups.")


# class CustomHook(Hook):
#     def after_train_epoch(self, runner):
#         add_no_mask_embed_to_optimizer(runner)


if __name__ == '__main__':
    main()