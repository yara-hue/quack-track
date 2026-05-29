#!/usr/bin/env python
import os
import sys
import yaml
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.siamrpn import LightSiamRPN
from src.trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('quack-track')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info('Building model...')
    model = LightSiamRPN(
        backbone_cfg=cfg['model']['backbone'],
        fpn_cfg=cfg['model']['fpn'],
        head_cfg=cfg['model']['head'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total params: {total_params:,} | Trainable: {trainable_params:,}')

    trainer = Trainer(model, cfg, device=args.device)

    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.start_epoch = checkpoint['epoch'] + 1
        trainer.best_loss = checkpoint.get('loss', float('inf'))
        logger.info(f'Resumed at epoch {checkpoint["epoch"]+1} (best loss: {trainer.best_loss:.4f})')

    trainer.train(checkpoint_dir=args.checkpoint_dir)


if __name__ == '__main__':
    main()
