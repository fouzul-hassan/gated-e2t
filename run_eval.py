"""
GLIM Evaluation Script

Run evaluation metrics (including new ETES energy metric) on a trained checkpoint.
No retraining required.

Usage:
    python run_eval.py --checkpoint_path "path/to/checkpoint.ckpt" \
                       --data_path "./data/tmp/zuco_eeg_label_8variants.df" \
                       --use_energy --use_gated

Arguments:
    --checkpoint_path: Path to the .ckpt file
    --data_path: Path to the .df data file
    --use_energy: Enable energy-based evaluation (ETES)
    --use_gated: Enable gated attention config (must match training)
    --gpus: GPU IDs to use (default: 0)
"""
import os
import argparse
import torch
import warnings
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from model.glim import GLIM
from data.datamodule import GLIMDataModule

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='GLIM Evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    
    # Configuration (must match training)
    parser.add_argument('--use_gated', action='store_true', help='Use if model trained with gated attention')
    parser.add_argument('--use_energy', action='store_true', help='Use if model trained with energy components')
    parser.add_argument('--generation_strategy', type=str, default='beam', help='Generation strategy for eval')
    
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs')
    parser.add_argument('--bsz_test', type=int, default=24, help='Batch size')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    devices = [int(x) for x in args.gpus.split(',')]
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    
    # Logger
    logger = WandbLogger(
        project='glim',
        name='eval-run',
        save_dir='./runs/eval',
        offline=True  # Offline by default for eval
    )
    
    # Data
    print(f"Loading data from {args.data_path}...")
    dm = GLIMDataModule(
        data_path=args.data_path,
        eval_noise_input=False,
        bsz_test=args.bsz_test,
        num_workers=4
    )
    
    # Model
    print(f"Loading model from {args.checkpoint_path}...")
    
    # We need to manually override strict loading if args are different
    # But load_from_checkpoint usually handles hparams if saved correctly.
    # We pass overrides just in case.
    
    model = GLIM.load_from_checkpoint(
        args.checkpoint_path,
        map_location=f"cuda:{devices[0]}",
        strict=False,
        
        # Override config for evaluation
        generation_strategy=args.generation_strategy,
        use_etes_eval=args.use_energy, # Ensure ETES is on if energy is requested
        use_energy_loss=False,         # No loss calculation needed for eval
    )
    
    # Trainer
    trainer = L.Trainer(
        accelerator='gpu',
        devices=devices,
        logger=logger,
        precision='bf16-mixed',
    )
    
    # Run Test
    print("Starting evaluation...")
    trainer.test(model, datamodule=dm)
    print("Evaluation complete! Check logs for metrics.")

if __name__ == '__main__':
    main()
