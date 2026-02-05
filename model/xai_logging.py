"""
XAI Logging Utilities for GLIM

Provides visualization and logging functions for explainable AI metrics:
- Gate activation statistics and histograms
- Attention weight heatmaps
- WandB integration for logging

Based on "Gated Attention for Large Language Models" (NeurIPS 2025 Best Paper)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
from typing import Optional, Dict, List, Any
import io
from PIL import Image


def compute_gate_entropy(gate_values: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute entropy of gate activations to measure gating diversity.
    
    Higher entropy = more diverse gating patterns (good)
    Lower entropy = more uniform/degenerate gating (potentially problematic)
    
    Args:
        gate_values: Tensor of gate activations (any shape, will be flattened)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized entropy in [0, 1] range
    """
    # Flatten and move to CPU
    gates = gate_values.detach().float().flatten().cpu()
    
    # Create histogram (50 bins between 0 and 1)
    hist, _ = np.histogram(gates.numpy(), bins=50, range=(0, 1), density=True)
    hist = hist / (hist.sum() + eps)  # Normalize to probability distribution
    
    # Compute entropy
    entropy = -np.sum(hist * np.log(hist + eps))
    
    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(50)
    normalized_entropy = entropy / max_entropy
    
    return float(normalized_entropy)


def create_attention_heatmap(
    attn_weights: torch.Tensor,
    query_labels: Optional[List[str]] = None,
    key_labels: Optional[List[str]] = None,
    title: str = "Cross-Attention Weights",
    figsize: tuple = (10, 8),
    cmap: str = "Blues"
) -> Image.Image:
    """
    Create a heatmap visualization of attention weights.
    
    Args:
        attn_weights: Attention weights of shape (query_len, key_len)
        query_labels: Labels for query positions (e.g., EEG tokens)
        key_labels: Labels for key positions (e.g., text tokens)
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        
    Returns:
        PIL Image of the heatmap
    """
    # Move to CPU and convert to numpy
    weights = attn_weights.detach().float().cpu().numpy()
    
    # Limit size for visualization
    max_dim = 64
    if weights.shape[0] > max_dim:
        weights = weights[:max_dim, :]
    if weights.shape[1] > max_dim:
        weights = weights[:, :max_dim]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(weights, aspect='auto', cmap=cmap)
    ax.set_xlabel("Key Position (Text)", fontsize=12)
    ax.set_ylabel("Query Position (EEG)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=10)
    
    # Add tick labels if provided
    if key_labels and len(key_labels) <= 32:
        ax.set_xticks(range(len(key_labels)))
        ax.set_xticklabels(key_labels, rotation=45, ha='right', fontsize=8)
    if query_labels and len(query_labels) <= 32:
        ax.set_yticks(range(len(query_labels)))
        ax.set_yticklabels(query_labels, fontsize=8)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    
    return img


def create_gate_histogram(
    gate_values: torch.Tensor,
    layer_name: str = "All Layers",
    figsize: tuple = (8, 5)
) -> Image.Image:
    """
    Create a histogram of gate activation values.
    
    Args:
        gate_values: Gate activations (any shape, will be flattened)
        layer_name: Name for the layer being visualized
        figsize: Figure size
        
    Returns:
        PIL Image of the histogram
    """
    gates = gate_values.detach().float().flatten().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    n, bins, patches = ax.hist(gates, bins=50, range=(0, 1), 
                                color='steelblue', edgecolor='white', alpha=0.8)
    
    # Add statistics
    mean_val = np.mean(gates)
    std_val = np.std(gates)
    sparsity = np.mean(gates < 0.1) * 100
    
    # Add vertical lines for mean
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.3f}')
    ax.axvline(0.1, color='orange', linestyle=':', linewidth=2, 
               label=f'Sparsity threshold (10%)')
    
    # Add text box with statistics
    stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nSparsity: {sparsity:.1f}%'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel("Gate Activation Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Gate Activation Distribution: {layer_name}", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    
    return img


def create_per_layer_stats_table(layer_stats: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Create a WandB table of per-layer gate statistics.
    
    Args:
        layer_stats: List of dicts with 'name', 'gate_mean', 'gate_std', 'gate_sparsity'
        
    Returns:
        Dictionary ready for WandB table logging
    """
    columns = ["Layer", "Gate Mean", "Gate Std", "Gate Sparsity (%)"]
    data = []
    
    for stat in layer_stats:
        data.append([
            stat.get('name', 'Unknown'),
            round(stat.get('gate_mean', 0.0), 4),
            round(stat.get('gate_std', 0.0), 4),
            round(stat.get('gate_sparsity', 0.0) * 100, 2)
        ])
    
    return {"columns": columns, "data": data}


def log_xai_to_wandb(
    logger,
    gate_stats: Dict[str, Any],
    attn_weights_dict: Optional[Dict[int, torch.Tensor]] = None,
    sample_texts: Optional[List[str]] = None,
    prefix: str = "val",
    current_epoch: int = 0,
    max_attention_samples: int = 4
) -> None:
    """
    Log comprehensive XAI metrics and visualizations to WandB.
    
    Args:
        logger: WandB logger (from Lightning)
        gate_stats: Dictionary from EEGEncoder.get_gate_stats()
        attn_weights_dict: Optional dict of {layer_idx: attention_weights}
        sample_texts: Optional list of input text strings for labeling
        prefix: Logging prefix (e.g., 'val', 'test', 'full_val')
        current_epoch: Current training epoch
        max_attention_samples: Maximum number of attention heatmaps to log
    """
    if logger is None:
        return
    
    try:
        import wandb
        run = logger.experiment
    except (ImportError, AttributeError):
        return
    
    log_dict = {}
    
    # 1. Log aggregate gate statistics as scalars
    aggregate = gate_stats.get('aggregate', gate_stats)
    log_dict[f'{prefix}/xai_gate_mean'] = aggregate.get('gate_mean', 0.0)
    log_dict[f'{prefix}/xai_gate_std'] = aggregate.get('gate_std', 0.0)
    log_dict[f'{prefix}/xai_gate_sparsity'] = aggregate.get('gate_sparsity', 0.0)
    
    if 'gate_entropy' in gate_stats:
        log_dict[f'{prefix}/xai_gate_entropy'] = gate_stats['gate_entropy']
    
    # 2. Log per-layer statistics as table
    per_layer = gate_stats.get('per_layer', [])
    if per_layer:
        table_data = create_per_layer_stats_table(per_layer)
        log_dict[f'{prefix}/xai_layer_stats'] = wandb.Table(
            columns=table_data['columns'],
            data=table_data['data']
        )
    
    # 3. Log gate histogram
    if 'all_gate_values' in gate_stats:
        hist_img = create_gate_histogram(
            gate_stats['all_gate_values'],
            layer_name=f"Epoch {current_epoch}"
        )
        log_dict[f'{prefix}/xai_gate_histogram'] = wandb.Image(hist_img)
    
    # 4. Log attention heatmaps (if available)
    if attn_weights_dict:
        attention_images = []
        for layer_idx, attn_w in list(attn_weights_dict.items())[:max_attention_samples]:
            if attn_w is not None and attn_w.numel() > 0:
                # Take first sample from batch if batched
                if attn_w.dim() > 2:
                    attn_w = attn_w[0]
                
                heatmap = create_attention_heatmap(
                    attn_w,
                    title=f"Decoder Block {layer_idx} - Cross-Attention"
                )
                attention_images.append(wandb.Image(
                    heatmap,
                    caption=f"Layer {layer_idx}"
                ))
        
        if attention_images:
            log_dict[f'{prefix}/xai_attention_heatmaps'] = attention_images
    
    # Log everything
    run.log(log_dict)


def collect_gate_values(encoder) -> torch.Tensor:
    """
    Collect all gate values from a GatedAttention-enabled encoder.
    
    Args:
        encoder: EEGEncoder with gated attention layers
        
    Returns:
        Concatenated tensor of all gate values
    """
    all_gates = []
    
    # Collect from encoder blocks
    for block in encoder.in_blocks:
        if hasattr(block, 'attn') and hasattr(block.attn, 'gate_proj'):
            if hasattr(block.attn, '_last_gate_values'):
                all_gates.append(block.attn._last_gate_values.flatten())
    
    # Collect from decoder blocks
    for block in encoder.out_blocks:
        for attn_name in ['self_attn', 'cross_attn']:
            attn = getattr(block, attn_name, None)
            if attn and hasattr(attn, 'gate_proj'):
                if hasattr(attn, '_last_gate_values'):
                    all_gates.append(attn._last_gate_values.flatten())
    
    if all_gates:
        return torch.cat(all_gates)
    return torch.tensor([])
