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
    trainer.train(checkpoint_dir=args.checkpoint_dir)


if __name__ == '__main__':
    main()
