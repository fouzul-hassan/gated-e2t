"""
Training script for GLIM-compatible EEG encoder pretraining.

Usage:
    python run_pretrain.py --data_path ../data/tmp/zuco_eeg_128ch_1280len.df --epochs 100
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path for GLIM imports
sys.path.insert(0, '..')

from Dataset.zuco_dataset import load_zuco, ZuCoDataset
from pretrain_glim_encoder import GLIMEncoderPretrainer, count_parameters

# For linear probe evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='GLIM Encoder Pretraining')
    
    # Data
    parser.add_argument('--data_path', type=str, 
                        default='../data/tmp/zuco_eeg_128ch_1280len.df',
                        help='Path to ZuCo EEG dataset')
    parser.add_argument('--output_dir', type=str, default='Results/GLIM_Pretrain',
                        help='Output directory for checkpoints and logs')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    
    # Model
    parser.add_argument('--n_blocks', type=int, default=6, help='Number of encoder blocks')
    parser.add_argument('--emb_size', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Masking ratio')
    parser.add_argument('--momentum', type=float, default=0.99, help='EMA momentum')
    parser.add_argument('--use_gated_attention', action='store_true', help='Use gated attention')
    
    # System
    parser.add_argument('--gpu', type=int, default=0, help='GPU index, -1 for CPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--linear_probe_interval', type=int, default=5, 
                        help='Run linear probe every N epochs')
    
    return parser.parse_args()


def make_representation(model, loader, device):
    """Extract representations from frozen encoder."""
    model.eval()
    all_repr = []
    all_labels = []
    
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            repr = model(x)  # (B, emb_size)
            all_repr.append(repr.cpu())
            all_labels.append(y)
    
    return torch.cat(all_repr, dim=0), torch.cat(all_labels, dim=0)


def run_linear_probe(model, train_loader, test_loader, device):
    """Evaluate encoder with linear probe classification."""
    train_repr, train_labels = make_representation(model, train_loader, device)
    test_repr, test_labels = make_representation(model, test_loader, device)
    
    # Fit logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_repr.numpy(), train_labels.numpy())
    
    # Predict
    pred = clf.predict(test_repr.numpy())
    acc = accuracy_score(test_labels.numpy(), pred)
    
    return acc


def train_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.copy_weight()  # Ensure target starts same as context
    model.train()
    
    total_loss = 0
    loss_components = {'align': 0, 'std': 0, 'cov': 0}
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for x, _, _ in pbar:
        x = x.to(device)
        
        # Forward
        rep_mask, rep_pred, _, _ = model.pretrain_forward(x)
        loss, loss_dict = model.compute_loss(rep_mask, rep_pred)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # EMA update
        model.momentum_update()
        
        # Logging
        total_loss += loss_dict['total']
        for k in loss_components:
            loss_components[k] += loss_dict[k]
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'align': f"{loss_dict['align']:.4f}",
        })
    
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    data = load_zuco(args.data_path, seed=args.seed)
    
    # Create datasets
    train_dataset = ZuCoDataset(data['All_train_data'], data['All_train_label'], args.patch_size)
    test_dataset = ZuCoDataset(data['test_data'], data['test_label'], args.patch_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create model
    model = GLIMEncoderPretrainer(
        in_len=data['max_len'],  # 1280
        in_dim=data['num_channels'],  # 128
        emb_size=args.emb_size,
        n_blocks=args.n_blocks,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        momentum=args.momentum,
        use_gated_attention=args.use_gated_attention,
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Optimizer (only context encoder + predictor)
    params_to_optimize = list(model.context_encoder.parameters()) + \
                         list(model.predictor.parameters()) + \
                         list(model.patch_embed.parameters()) + \
                         [model.mask_token]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Training loop
    best_acc = 0
    logger.info("Starting pretraining...")
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        avg_loss, loss_components = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Log to tensorboard
        writer.add_scalar('loss/total', avg_loss, epoch)
        for k, v in loss_components.items():
            writer.add_scalar(f'loss/{k}', v, epoch)
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}, align={loss_components['align']:.4f}, "
                   f"std={loss_components['std']:.4f}, cov={loss_components['cov']:.4f}, "
                   f"time={epoch_time:.1f}s")
        
        # Linear probe evaluation
        if epoch % args.linear_probe_interval == 0:
            acc = run_linear_probe(model, train_loader, test_loader, device)
            writer.add_scalar('accuracy/linear_probe', acc, epoch)
            logger.info(f"  Linear probe accuracy: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                # Save best model
                save_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': model.get_encoder_state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': acc,
                    'args': vars(args),
                }, save_path)
                logger.info(f"  Saved best model (acc={acc:.4f})")
    
    # Save final model
    save_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.get_encoder_state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }, save_path)
    
    logger.info(f"Training complete. Best linear probe accuracy: {best_acc:.4f}")
    writer.close()


if __name__ == '__main__':
    main()
