"""
GLIM Training Script with Command-Line Arguments

Supports:
- Data path via --data_path
- Resume training via --resume_from
- GPU selection via --gpus
- All enhancement features (gated attention, nucleus sampling, energy)

Examples:
    python train_cli.py --data_path ./data/zuco.df
    python train_cli.py --data_path ./data/zuco.df --resume_from ./runs/checkpoint.ckpt
    python train_cli.py --data_path ./data/zuco.df --gpus 0,1 --use_energy
"""
import os
import argparse
import torch
import warnings
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model.glim import GLIM
from data.datamodule import GLIMDataModule

warnings.filterwarnings("ignore", ".*when logging on epoch level in distributed.*")


def parse_args():
    parser = argparse.ArgumentParser(description='GLIM Training with Enhancements')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the preprocessed data file (.df or .pkl)')
    
    # Training control
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=72,
                        help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=24,
                        help='Validation batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # Hardware
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU IDs to use (comma-separated, e.g., "0,1,2")')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model configuration
    parser.add_argument('--text_model', type=str, default='google/flan-t5-large',
                        choices=['google/flan-t5-large', 'google/flan-t5-xl'],
                        help='Text model to use')
    
    # Enhancement features
    parser.add_argument('--use_gated_attention', action='store_true',
                        help='Enable gated attention in EEG encoder')
    parser.add_argument('--gating_type', type=str, default='elementwise',
                        choices=['elementwise', 'headwise'],
                        help='Type of gating mechanism')
    parser.add_argument('--generation_strategy', type=str, default='beam',
                        choices=['beam', 'nucleus', 'greedy', 'energy'],
                        help='Text generation strategy')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling threshold')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    
    # Energy-based features
    parser.add_argument('--use_energy', action='store_true',
                        help='Enable all energy-based features')
    parser.add_argument('--energy_loss_weight', type=float, default=0.3,
                        help='Weight for energy contrastive loss')
    
    # Logging
    parser.add_argument('--project_name', type=str, default='glim',
                        help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='WandB run name (auto-generated if not provided)')
    parser.add_argument('--offline', action='store_true',
                        help='Run WandB in offline mode')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='Directory for logs and checkpoints')
    
    # Validation
    parser.add_argument('--full_val_interval', type=int, default=10,
                        help='Full validation every N epochs')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Setup
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    
    # Create log directory
    group_name = args.run_name or f'glim-{args.generation_strategy}'
    if args.use_gated_attention:
        group_name += '-gated'
    if args.use_energy:
        group_name += '-energy'
    
    log_dir = os.path.join(args.log_dir, group_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Logger
    logger = WandbLogger(
        project=args.project_name,
        name=group_name,
        save_dir=log_dir,
        offline=args.offline,
    )
    
    # Callbacks - Multiple checkpoint saving strategies
    callbacks = [
        # Save every N epochs (all interval checkpoints)
        ModelCheckpoint(
            monitor='epoch',
            dirpath=os.path.join(log_dir, 'checkpoints'),
            save_top_k=-1,
            every_n_epochs=args.full_val_interval,
            filename='interval-epoch={epoch:03d}',
        ),
        # Save top 3 best models based on validation retrieval accuracy
        ModelCheckpoint(
            monitor='val/retrieval_acc_top01',
            mode='max',
            dirpath=os.path.join(log_dir, 'checkpoints'),
            save_top_k=3,
            filename='best-epoch={epoch:03d}-acc={val/retrieval_acc_top01:.4f}',
        ),
        # Always save the last checkpoint
        ModelCheckpoint(
            dirpath=os.path.join(log_dir, 'checkpoints'),
            save_last=True,
            filename='last',
        ),
    ]
    
    # Trainer
    trainer = L.Trainer(
        accelerator='gpu',
        devices=gpu_ids,
        logger=logger,
        max_epochs=args.max_epochs,
        precision='bf16-mixed',
        enable_checkpointing=True,
        callbacks=callbacks,
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
    )
    
    # Data Module
    dm = GLIMDataModule(
        data_path=args.data_path,
        eval_noise_input=False,
        bsz_train=args.batch_size,
        bsz_val=args.val_batch_size,
        num_workers=args.num_workers,
    )
    
    # Model
    model = GLIM(
        # EEG Encoder
        input_eeg_len=1280,
        hidden_eeg_len=96,
        input_text_len=96,
        tgt_text_len=64,
        input_dim=128,
        hidden_dim=256,
        embed_dim=1024,
        text_model_id=args.text_model,
        prompt_nums=(3, 3, 31),
        prompt_dropout_probs=(0.0, 1.0, 1.0),
        n_in_blocks=6,
        n_out_blocks=6,
        
        # Gated Attention
        use_gated_attention=args.use_gated_attention,
        gating_type=args.gating_type,
        
        # Generation Strategy
        generation_strategy=args.generation_strategy,
        top_p=args.top_p,
        temperature=args.temperature,
        
        # Energy-based Features
        use_energy_loss=args.use_energy,
        energy_loss_weight=args.energy_loss_weight,
        use_etes_eval=args.use_energy,
        
        # Standard config
        clip_loss_weight=0.5,
        lr=args.lr,
        bsz_train=args.batch_size,
        bsz_val=args.val_batch_size,
        full_val_interval=args.full_val_interval,
    )
    
    # Train (with optional resume)
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=args.resume_from,  # None = start fresh, path = resume
    )


if __name__ == '__main__':
    main()
