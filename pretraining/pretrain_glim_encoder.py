"""
GLIM-Compatible EEG Encoder Pretraining Module.

Uses GLIM's EncoderBlock architecture with EEG2Rep's self-supervised objective:
- Semantic Subsequence Preserving (SSP) masking
- VICReg loss (alignment + variance + covariance)
- EMA target encoder
"""
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# Import GLIM's EncoderBlock
import sys
sys.path.insert(0, '..')
from model.modules import EncoderBlock, get_1d_sincos_pos_embed_from_grid


class GLIMEncoderPretrainer(nn.Module):
    """
    Wraps GLIM's EncoderBlocks for self-supervised pretraining.
    
    Architecture:
    - Input embedding (patchify EEG)
    - Context encoder (6x EncoderBlock from GLIM)
    - Target encoder (EMA copy, no gradients)
    - Cross-attention predictor
    
    Loss: VICReg (alignment + variance + covariance)
    """
    def __init__(self,
                 in_len: int = 1280,
                 in_dim: int = 128,
                 emb_size: int = 128,
                 n_blocks: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 patch_size: int = 8,
                 mask_ratio: float = 0.5,
                 momentum: float = 0.99,
                 use_gated_attention: bool = False,
                 predictor_layers: int = 2):
        super().__init__()
        
        self.in_len = in_len
        self.in_dim = in_dim
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.momentum = momentum
        
        # Number of patches after patchifying
        self.num_patches = in_len // patch_size  # 1280 / 8 = 160 patches
        
        # Input embedding (patchify + project)
        self.patch_embed = nn.Sequential(
            nn.Linear(in_dim * patch_size, emb_size),
            nn.LayerNorm(emb_size),
            nn.GELU(),
        )
        
        # Positional encoding
        pos_embed = get_1d_sincos_pos_embed_from_grid(emb_size, np.arange(self.num_patches))
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), 
                                       requires_grad=False)
        
        # Context encoder (GLIM's EncoderBlocks - no prompt injection for pretraining)
        self.context_encoder = nn.ModuleList([
            EncoderBlock(emb_size, self.num_patches, 
                        inject_prompt=False,  # No prompts during pretraining
                        temporal_modulate=False,
                        is_causal=False,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        use_gated_attention=use_gated_attention)
            for _ in range(n_blocks)
        ])
        
        # Target encoder (EMA copy)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(emb_size))
        
        # Predictor (cross-attention from mask tokens to context output)
        self.predictor = nn.ModuleList([
            CrossAttnPredictorBlock(emb_size, num_heads, dropout)
            for _ in range(predictor_layers)
        ])
        
        # Output projection
        self.predictor_norm = nn.LayerNorm(emb_size)
        
        # Global average pooling for VICReg
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def copy_weight(self):
        """Copy context encoder weights to target encoder."""
        with torch.no_grad():
            for param_ctx, param_tgt in zip(self.context_encoder.parameters(), 
                                            self.target_encoder.parameters()):
                param_tgt.data = param_ctx.data
    
    def momentum_update(self):
        """EMA update of target encoder."""
        with torch.no_grad():
            for param_ctx, param_tgt in zip(self.context_encoder.parameters(), 
                                            self.target_encoder.parameters()):
                param_tgt.data = self.momentum * param_tgt.data + (1 - self.momentum) * param_ctx.data
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert EEG to patches.
        
        Args:
            x: (B, T, C) where T=1280, C=128
            
        Returns:
            patches: (B, num_patches, emb_size)
        """
        B, T, C = x.shape
        # Reshape to (B, num_patches, patch_size * C)
        x = x.view(B, self.num_patches, self.patch_size * C)
        # Project to embedding dimension
        patches = self.patch_embed(x)  # (B, num_patches, emb_size)
        return patches
    
    def encode_context(self, patches: torch.Tensor) -> torch.Tensor:
        """Encode patches with context encoder."""
        x = patches
        for block in self.context_encoder:
            x = block(x, mask=None, p=None)
        return x
    
    def encode_target(self, patches: torch.Tensor) -> torch.Tensor:
        """Encode patches with target encoder (no gradients)."""
        with torch.no_grad():
            x = patches
            for block in self.target_encoder:
                x = block(x, mask=None, p=None)
        return x
    
    def pretrain_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, 
                                                          torch.Tensor, torch.Tensor]:
        """
        Forward pass for pretraining.
        
        Args:
            x: (B, T, C) EEG input, T=1280, C=128
            
        Returns:
            rep_mask: target representations for masked positions
            rep_mask_prediction: predicted representations for masked positions
            rep_context: full context encoder output (for monitoring)
            rep_target: full target encoder output (for monitoring)
        """
        B = x.shape[0]
        
        # Patchify and add positional encoding
        patches = self.patchify(x)  # (B, num_patches, emb_size)
        patches = patches + self.pos_embed
        
        # SSP Masking: select contiguous chunks as visible
        indices = np.arange(self.num_patches)
        visible_chunks = semantic_subsequence_preserving(indices, chunk_count=2, 
                                                         target_percentage=self.mask_ratio)
        v_index = np.ravel(visible_chunks)
        m_index = np.setdiff1d(indices, v_index)
        
        # Visible patches for context encoder
        visible = patches[:, v_index, :]  # (B, num_visible, emb_size)
        
        # Encode visible with context encoder
        rep_context = self.encode_context(visible)  # (B, num_visible, emb_size)
        
        # Encode all patches with target encoder (no grad)
        rep_target = self.encode_target(patches)  # (B, num_patches, emb_size)
        rep_mask_target = rep_target[:, m_index, :]  # (B, num_masked, emb_size)
        
        # Create mask tokens with positional encoding
        mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, len(m_index), -1)
        mask_tokens = mask_tokens + self.pos_embed[:, m_index, :]
        
        # Predict masked representations via cross-attention
        rep_mask_prediction = mask_tokens
        for pred_block in self.predictor:
            rep_mask_prediction = pred_block(rep_mask_prediction, rep_context)
        rep_mask_prediction = self.predictor_norm(rep_mask_prediction)
        
        return rep_mask_target, rep_mask_prediction, rep_context, rep_target
    
    def compute_loss(self, rep_mask: torch.Tensor, 
                     rep_mask_prediction: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute VICReg loss.
        
        Args:
            rep_mask: target representations (B, num_masked, emb_size)
            rep_mask_prediction: predicted representations (B, num_masked, emb_size)
            
        Returns:
            total_loss: combined VICReg loss
            loss_dict: individual loss components for logging
        """
        # Alignment loss (L2/MSE)
        align_loss = F.mse_loss(rep_mask_prediction, rep_mask)
        
        # Variance and covariance on prediction (prevent collapse)
        # Global average pool over sequence dimension
        y = self.gap(rep_mask_prediction.transpose(1, 2)).squeeze(-1)  # (B, emb_size)
        y = y - y.mean(dim=0)  # Center
        
        # Variance loss: push std > 1
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_y))
        
        # Covariance loss: decorrelate features
        batch_size = y.shape[0]
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_y).pow_(2).sum() / y.shape[-1]
        
        total_loss = align_loss + std_loss + cov_loss
        
        loss_dict = {
            'align': align_loss.item(),
            'std': std_loss.item(),
            'cov': cov_loss.item(),
            'total': total_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normal forward for evaluation (encode all patches).
        
        Args:
            x: (B, T, C) EEG input
            
        Returns:
            representations: (B, emb_size) pooled features
        """
        patches = self.patchify(x)
        patches = patches + self.pos_embed
        out = self.encode_context(patches)
        # Global average pool
        out = self.gap(out.transpose(1, 2)).squeeze(-1)  # (B, emb_size)
        return out
    
    def get_encoder_state_dict(self) -> dict:
        """Get state dict for context encoder (for transfer to GLIM)."""
        return {f'in_blocks.{i}.{key}': val 
                for i, block in enumerate(self.context_encoder) 
                for key, val in block.state_dict().items()}


class CrossAttnPredictorBlock(nn.Module):
    """Cross-attention block for predicting masked representations."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: query (mask tokens), (B, num_masked, dim)
            kv: key-value (context output), (B, num_visible, dim)
        """
        q = q + self.cross_attn(self.norm1(q), kv, kv, need_weights=False)[0]
        q = q + self.mlp(self.norm2(q))
        return q


def semantic_subsequence_preserving(indices: np.ndarray, chunk_count: int = 2, 
                                     target_percentage: float = 0.5) -> List[np.ndarray]:
    """
    Semantic Subsequence Preserving (SSP) masking from EEG2Rep.
    Selects contiguous chunks as visible to preserve temporal semantics.
    
    Args:
        indices: array of patch indices
        chunk_count: number of contiguous chunks to select
        target_percentage: proportion of patches to keep visible
        
    Returns:
        List of selected chunk indices
    """
    total = len(indices)
    target_total = int(total * target_percentage)
    chunk_size = target_total // chunk_count
    
    # Randomly select starting points
    start_points = []
    for _ in range(chunk_count):
        start = random.randint(0, total - chunk_size)
        start_points.append(start)
    
    # Select chunks
    chunks = [indices[start:start + chunk_size] for start in start_points]
    return chunks


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Extract off-diagonal elements from a square matrix."""
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    model = GLIMEncoderPretrainer()
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward
    x = torch.randn(4, 1280, 128)
    rep_mask, rep_pred, rep_ctx, rep_tgt = model.pretrain_forward(x)
    print(f"rep_mask shape: {rep_mask.shape}")
    print(f"rep_pred shape: {rep_pred.shape}")
    
    loss, loss_dict = model.compute_loss(rep_mask, rep_pred)
    print(f"Losses: {loss_dict}")
