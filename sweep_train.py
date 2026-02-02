"""
Hyperparameter tuning for GLIM training.

Based on GLIM paper (https://arxiv.org/html/2505.17099v1):
- Learning rate: 1e-5 (paper), but can explore around this
- Batch size: 64 (paper), but memory constraints may require smaller
- Loss weights: clip_loss_weight=0.5, energy_loss_weight=0.3
- Weight decay: 0 (paper)

This script runs a grid search over key hyperparameters and tracks:
- BLEU@MTV (primary generation metric)
- Retrieval accuracy (representation quality)
- Zero-shot classification (semantic decoding)
- ETES (if energy is enabled)

Usage:
    python sweep_train.py \
      --data_path ./data/tmp/zuco_eeg_label_8variants.df \
      --gpus 0 \
      --max_epochs 50 \
      --use_energy --use_gated_attention
"""

import argparse
import os
import subprocess
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class HPSearchConfig:
    """Single hyperparameter configuration."""
    lr: float
    clip_loss_weight: float
    energy_loss_weight: float
    batch_size: int
    # Optional: can add more params like weight_decay, etc.


def parse_args():
    parser = argparse.ArgumentParser(description='GLIM Training Hyperparameter Sweep')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file')
    
    # Hardware
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU IDs (comma-separated)')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Training control
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Max epochs per config (reduced for tuning)')
    parser.add_argument('--full_val_interval', type=int, default=10,
                        help='Full validation every N epochs')
    
    # Model config (fixed across sweep)
    parser.add_argument('--text_model', type=str, default='google/flan-t5-large')
    parser.add_argument('--use_gated_attention', action='store_true')
    parser.add_argument('--use_energy', action='store_true')
    parser.add_argument('--generation_strategy', type=str, default='beam',
                        choices=['beam', 'nucleus', 'greedy', 'energy'])
    
    # Hyperparameter grids
    parser.add_argument('--lr_grid', nargs='+', type=float,
                        default=[5e-6, 1e-5, 2e-5, 5e-5],
                        help='Learning rate values to try')
    parser.add_argument('--clip_weight_grid', nargs='+', type=float,
                        default=[0.3, 0.5, 0.7],
                        help='CLIP loss weight values (λ in paper)')
    parser.add_argument('--energy_weight_grid', nargs='+', type=float,
                        default=[0.1, 0.3, 0.5],
                        help='Energy loss weight values (only if --use_energy)')
    parser.add_argument('--batch_size_grid', nargs='+', type=int,
                        default=[48, 64, 72],
                        help='Batch sizes to try')
    
    # Sweep control
    parser.add_argument('--limit_configs', type=int, default=0,
                        help='If >0, only run first N configs (for testing)')
    parser.add_argument('--output_dir', type=str, default='./runs/sweep',
                        help='Directory to save sweep results')
    
    # WandB
    parser.add_argument('--project_name', type=str, default='glim-sweep')
    parser.add_argument('--offline', action='store_true',
                        help='Run WandB offline')
    
    return parser.parse_args()


def build_config_grid(args) -> List[HPSearchConfig]:
    """Build grid of hyperparameter configurations."""
    configs = []
    
    for lr in args.lr_grid:
        for clip_w in args.clip_weight_grid:
            energy_weights = args.energy_weight_grid if args.use_energy else [0.0]
            for energy_w in energy_weights:
                for bsz in args.batch_size_grid:
                    configs.append(HPSearchConfig(
                        lr=lr,
                        clip_loss_weight=clip_w,
                        energy_loss_weight=energy_w,
                        batch_size=bsz,
                    ))
    
    if args.limit_configs > 0:
        configs = configs[:args.limit_configs]
    
    return configs


def run_training(config: HPSearchConfig, args, config_idx: int) -> Dict[str, Any]:
    """Run training for one config and return best metrics."""
    
    # Build run name
    run_name = f"sweep-{config_idx:03d}-lr{config.lr:.0e}-clip{config.clip_loss_weight:.1f}"
    if args.use_energy:
        run_name += f"-energy{config.energy_loss_weight:.1f}"
    run_name += f"-bsz{config.batch_size}"
    
    # Build command
    cmd = [
        "python", "train_cli.py",
        "--data_path", args.data_path,
        "--gpus", args.gpus,
        "--num_workers", str(args.num_workers),
        "--max_epochs", str(args.max_epochs),
        "--batch_size", str(config.batch_size),
        "--val_batch_size", "24",
        "--lr", str(config.lr),
        "--text_model", args.text_model,
        "--generation_strategy", args.generation_strategy,
        "--full_val_interval", str(args.full_val_interval),
        "--project_name", args.project_name,
        "--run_name", run_name,
        "--log_dir", args.output_dir,
    ]
    
    if args.use_gated_attention:
        cmd.append("--use_gated_attention")
        cmd.extend(["--gating_type", "elementwise"])
    
    if args.use_energy:
        cmd.append("--use_energy")
        cmd.extend(["--energy_loss_weight", str(config.energy_loss_weight)])
    
    if args.offline:
        cmd.append("--offline")
    
    # Add loss weight arguments
    cmd.extend(["--clip_loss_weight", str(config.clip_loss_weight)])
    cmd.extend(["--commitment_loss_weight", "0.0"])  # Fixed per paper
    
    print(f"\n{'='*80}")
    print(f"CONFIG {config_idx}/{len(build_config_grid(args))}: {run_name}")
    print(f"{'='*80}")
    print(f"Hyperparameters:")
    print(f"  Learning rate: {config.lr}")
    print(f"  CLIP loss weight: {config.clip_loss_weight}")
    print(f"  Energy loss weight: {config.energy_loss_weight}")
    print(f"  Batch size: {config.batch_size}")
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        
        # Parse output to extract metrics (simplified - in practice, read from wandb or checkpoint)
        return {
            'config': asdict(config),
            'run_name': run_name,
            'status': 'success',
            'stdout': result.stdout[-2000:],  # Last 2000 chars
        }
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return {
            'config': asdict(config),
            'run_name': run_name,
            'status': 'failed',
            'error': str(e),
            'stderr': e.stderr[-1000:] if e.stderr else '',
        }


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build config grid
    configs = build_config_grid(args)
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SWEEP")
    print(f"{'='*80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Learning rates: {args.lr_grid}")
    print(f"CLIP weights: {args.clip_weight_grid}")
    if args.use_energy:
        print(f"Energy weights: {args.energy_weight_grid}")
    print(f"Batch sizes: {args.batch_size_grid}")
    print(f"{'='*80}\n")
    
    # Run each config
    results = []
    for idx, config in enumerate(configs, 1):
        result = run_training(config, args, idx)
        results.append(result)
        
        # Save intermediate results
        results_file = os.path.join(args.output_dir, 'sweep_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Successful runs: {len(successful)}/{len(results)}")
    print(f"Failed runs: {len(failed)}/{len(results)}")
    
    if failed:
        print("\nFailed configurations:")
        for r in failed:
            print(f"  {r['run_name']}: {r.get('error', 'Unknown error')}")
    
    print(f"\nResults saved to: {os.path.join(args.output_dir, 'sweep_results.json')}")
    print("\nNext steps:")
    print("1. Check wandb logs for detailed metrics")
    print("2. Evaluate best checkpoints with run_eval.py")
    print("3. Compare metrics: BLEU@MTV, retrieval acc, zero-shot cls, ETES")


if __name__ == '__main__':
    main()
