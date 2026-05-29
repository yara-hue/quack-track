#!/usr/bin/env python
import os
import sys
import yaml
import json
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.siamrpn import LightSiamRPN
from src.evaluate import evaluate_uav20l

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('quack-track')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/eval_results.json')
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

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "?")}')

    logger.info('Evaluating on UAV20L...')
    results = evaluate_uav20l(
        model,
        data_root=cfg['data']['uav20l_root'],
        img_root=cfg['data'].get('uav123_root'),
        device=args.device,
        template_sz=cfg['eval']['template_sz'],
        search_sz=cfg['eval']['search_sz'],
    )

    logger.info(f'Mean Precision (20px): {results["mean_precision"]:.3f}')
    logger.info(f'Mean Success (IoU=0.5): {results["mean_success"]:.3f}')
    logger.info(f'AUC: {results["auc"]:.3f}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'mean_precision': float(results['mean_precision']),
            'mean_success': float(results['mean_success']),
            'auc': float(results['auc']),
            'seq_results': {
                k: {
                    'mean_error': float(v['mean_error']),
                    'mean_overlap': float(v['mean_overlap']),
                }
                for k, v in results['seq_results'].items()
            }
        }, f, indent=2)
    logger.info(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
