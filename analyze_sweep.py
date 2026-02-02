"""
Analyze sweep results and extract best hyperparameters.

Reads wandb logs or checkpoint directories to find best configs.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any


def find_best_configs(sweep_dir: str, metric: str = 'val/retrieval_acc_top01', top_k: int = 5):
    """Find top-k configs based on a metric."""
    sweep_path = Path(sweep_dir)
    
    # Look for wandb logs or checkpoint directories
    configs = []
    
    for run_dir in sweep_path.iterdir():
        if not run_dir.is_dir() or 'sweep-' not in run_dir.name:
            continue
        
        # Try to extract config from directory name
        # Format: sweep-XXX-lrXe-Y-clipZ.Z-energyW.W-bszN
        parts = run_dir.name.split('-')
        config = {}
        for part in parts:
            if part.startswith('lr'):
                config['lr'] = float(part[2:].replace('e', 'e'))
            elif part.startswith('clip'):
                config['clip_loss_weight'] = float(part[4:])
            elif part.startswith('energy'):
                config['energy_loss_weight'] = float(part[6:])
            elif part.startswith('bsz'):
                config['batch_size'] = int(part[3:])
        
        # Look for best checkpoint or metrics
        checkpoint_dir = run_dir / 'checkpoints'
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('*.ckpt'))
            if checkpoints:
                config['checkpoint'] = str(checkpoints[-1])  # Use last checkpoint
                configs.append(config)
    
    return configs[:top_k]


def main():
    parser = argparse.ArgumentParser(description='Analyze sweep results')
    parser.add_argument('--sweep_dir', type=str, default='./runs/sweep',
                        help='Directory containing sweep runs')
    parser.add_argument('--results_file', type=str, default='./runs/sweep/sweep_results.json',
                        help='JSON file with sweep results')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top configs to show')
    
    args = parser.parse_args()
    
    # Try to load JSON results
    if os.path.exists(args.results_file):
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        
        print(f"\n{'='*80}")
        print("SWEEP ANALYSIS")
        print(f"{'='*80}")
        print(f"Total configs: {len(results)}")
        
        successful = [r for r in results if r.get('status') == 'success']
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(results) - len(successful)}")
        
        if successful:
            print(f"\nTop {args.top_k} configurations (by run order):")
            for i, r in enumerate(successful[:args.top_k], 1):
                print(f"\n{i}. {r['run_name']}")
                print(f"   Config: {r['config']}")
    else:
        print(f"Results file not found: {args.results_file}")
        print("Please run sweep_train.py first or check wandb logs manually.")


if __name__ == '__main__':
    main()
