"""
Generate publication-quality architecture diagrams:
  1. Signal-JEPA Pretraining 
  2. EEG-to-Text Fine-tuning Pipeline
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Colour palette
# ═══════════════════════════════════════════════════════════════
C = {
    'eeg':       '#1565C0',   # deep blue
    'encoder':   '#2E7D32',   # green
    'target':    '#558B2F',   # olive green
    'predictor': '#F57F17',   # amber
    'loss':      '#C62828',   # red
    'queries':   '#6A1B9A',   # purple
    'aligner':   '#00838F',   # teal
    'decoder':   '#E65100',   # deep orange
    'text':      '#AD1457',   # pink
    'prompt':    '#4527A0',   # indigo
    'gate':      '#FF6F00',   # orange
    'frozen':    '#78909C',   # blue grey
    'bg':        '#FAFAFA',
    'arrow':     '#37474F',
}

def box(ax, xy, w, h, label, color, fontsize=10, sublabel=None, 
        dashed=False, alpha=0.15, text_color=None):
    """Draw a rounded box with label."""
    ls = '--' if dashed else '-'
    lw = 1.5 if not dashed else 1.5
    rect = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.15",
                          facecolor=color, alpha=alpha,
                          edgecolor=color, linewidth=lw, linestyle=ls)
    ax.add_patch(rect)
    tc = text_color or color
    ax.text(xy[0] + w/2, xy[1] + h/2 + (0.12 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold',
            color=tc,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    if sublabel:
        ax.text(xy[0] + w/2, xy[1] + h/2 - 0.2, sublabel,
                ha='center', va='center', fontsize=7.5, color=tc, alpha=0.75,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

def arrow(ax, start, end, color='#37474F', style='->', lw=1.8, connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=connectionstyle))

def label_arrow(ax, start, end, text, color='#37474F', fontsize=8, offset=(0, 0.12)):
    """Draw arrow with a label above it."""
    arrow(ax, start, end, color)
    mid = ((start[0]+end[0])/2 + offset[0], (start[1]+end[1])/2 + offset[1])
    ax.text(mid[0], mid[1], text, ha='center', va='center', fontsize=fontsize,
            color=color, style='italic',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Signal-JEPA Pretraining
# ═══════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(16, 9), facecolor=C['bg'])
ax1.set_xlim(-0.5, 15.5)
ax1.set_ylim(-1, 9)
ax1.axis('off')
ax1.set_facecolor(C['bg'])

# Title
ax1.text(8, 8.4, 'Stage 1: Signal-JEPA Pretraining', fontsize=20, fontweight='bold',
         ha='center', va='center', color='#212121')
ax1.text(8, 7.9, 'Self-supervised representation learning for EEG signals', fontsize=12,
         ha='center', va='center', color='#757575')

# ── Input EEG ──
box(ax1, (0.5, 5.5), 3, 1.2, 'Raw EEG Input', C['eeg'], fontsize=12,
    sublabel='(1280 × 128) — time × channels')

# ── Patch Embedding ──
box(ax1, (0.5, 3.5), 3, 1.2, 'Patch Embedding', C['eeg'], fontsize=11,
    sublabel='patch_size=8 → 160 patches')
arrow(ax1, (2, 5.5), (2, 4.7), C['eeg'])

# ── SSP Masking ──
box(ax1, (4.5, 3.5), 2.5, 1.2, 'SSP Masking', C['gate'], fontsize=10,
    sublabel='Temporal-aware')
arrow(ax1, (3.5, 4.1), (4.5, 4.1), C['gate'])

# ── Visible patches → Context Encoder ──
box(ax1, (0, 1.2), 3, 1.5, 'Context Encoder', C['encoder'], fontsize=12,
    sublabel='6× Transformer Blocks\n128 dim, 8 heads')
ax1.text(1.5, 0.5, 'Visible patches', fontsize=8, ha='center', color=C['encoder'], style='italic')
arrow(ax1, (1.5, 3.5), (1.5, 2.7), C['encoder'])

# ── Masked patches → Target Encoder ──
box(ax1, (5, 1.2), 3, 1.5, 'Target Encoder', C['target'], fontsize=12,
    sublabel='EMA copy (τ=0.99)\nNO gradients', dashed=True)
ax1.text(6.5, 0.5, 'Masked patches', fontsize=8, ha='center', color=C['target'], style='italic')
arrow(ax1, (6.5, 3.5), (6.5, 2.7), C['target'])

# ── EMA arrow from context to target ──
ax1.annotate('', xy=(5, 2.0), xytext=(3, 2.0),
             arrowprops=dict(arrowstyle='->', color=C['frozen'], lw=1.5,
                             connectionstyle="arc3,rad=-0.3", linestyle='--'))
ax1.text(4, 1.3, 'EMA\nupdate', ha='center', va='center', fontsize=7.5,
         color=C['frozen'], style='italic')

# ── Predictor ──
box(ax1, (9.5, 3.5), 3, 1.5, 'Predictor', C['predictor'], fontsize=12,
    sublabel='2× Cross-Attention Blocks\nPredicts masked representations')

# Arrows: Context → Predictor
label_arrow(ax1, (3, 2.0), (9.5, 4.0), 'Context\nrepresentations', C['encoder'],
            offset=(0, 0.25))

# Arrows: Target → Loss
arrow(ax1, (8, 2.0), (11, 3.5), C['target'])

# ── VICReg Loss ──
box(ax1, (9.5, 1.2), 3, 1.5, 'VICReg Loss', C['loss'], fontsize=12,
    sublabel='Alignment + Variance\n+ Covariance')

# Predictor → Loss
arrow(ax1, (11, 3.5), (11, 2.7), C['loss'])

# ── Side annotations ──
# Key insight box
insight_text = (
    "Key Insight:\n"
    "Predicts in representation space,\n"
    "NOT raw EEG reconstruction.\n"
    "→ Learns abstract brain features"
)
ax1.text(14, 6.5, insight_text, fontsize=9, ha='center', va='center',
         color='#37474F', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#81C784', alpha=0.8))

# ── What is learned ──
ax1.text(14, 1.8, '✓ No collapse (var=0.99)\n✓ Loss: 0.85 → 0.356\n✓ 3.5 hours, 1 GPU',
         fontsize=9, ha='center', va='center', color='#2E7D32', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#81C784', alpha=0.8))

# ── Legend ──
legend_items = [
    mpatches.Patch(facecolor=C['encoder'], alpha=0.3, edgecolor=C['encoder'], label='Trainable'),
    mpatches.Patch(facecolor=C['target'], alpha=0.15, edgecolor=C['target'], linestyle='--', label='Frozen (EMA)'),
]
ax1.legend(handles=legend_items, loc='lower left', fontsize=10, frameon=True, fancybox=True)

fig1.tight_layout()
fig1.savefig('eda_figures/architecture_01_jepa_pretraining.png', dpi=200, bbox_inches='tight',
             facecolor=C['bg'])
print('Saved: architecture_01_jepa_pretraining.png')
plt.close(fig1)


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: EEG-to-Text Fine-tuning Architecture
# ═══════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(18, 10), facecolor=C['bg'])
ax2.set_xlim(-0.5, 17.5)
ax2.set_ylim(-1.5, 10)
ax2.axis('off')
ax2.set_facecolor(C['bg'])

# Title
ax2.text(9, 9.3, 'Stage 2: EEG-to-Text Architecture', fontsize=20, fontweight='bold',
         ha='center', va='center', color='#212121')
ax2.text(9, 8.8, 'Gated attention EEG encoder with cross-modal alignment and text generation', fontsize=12,
         ha='center', va='center', color='#757575')

# ══════════ LEFT: EEG Path ══════════

# ── Raw EEG Input ──
box(ax2, (0, 6.5), 3, 1.2, 'Raw EEG', C['eeg'], fontsize=13,
    sublabel='(n, 1280, 128)')

# ── Prompt Embedder ──
box(ax2, (4, 7.2), 2.8, 0.9, 'Prompt Embedder', C['prompt'], fontsize=10,
    sublabel='Task + Dataset + Subject')
arrow(ax2, (6.8, 7.65), (7.5, 6.8), C['prompt'])

# ── Positional Embedding ──
ax2.text(1.5, 5.9, '+ Sinusoidal\n  Pos. Embed', fontsize=8, ha='center',
         color=C['eeg'], style='italic')
arrow(ax2, (1.5, 6.5), (1.5, 5.3), C['eeg'])

# ── EEG Encoder (Backbone) ──
box(ax2, (0, 3.8), 3, 1.5, 'EEG Encoder', C['encoder'], fontsize=13,
    sublabel='6× Encoder Blocks\nGated Self-Attention + adaLN')
# JEPA badge
ax2.text(3.3, 5.1, '★ JEPA\n  Pretrained', fontsize=8, ha='center',
         color='#E65100', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0', edgecolor='#FF9800'))

# ── Projection ──
box(ax2, (0.3, 2.5), 2.4, 0.7, 'Projection', C['encoder'], fontsize=9,
    sublabel='128 → 256 dim')
arrow(ax2, (1.5, 3.8), (1.5, 3.2), C['encoder'])

# ── Learned Queries ──
box(ax2, (4, 1.5), 2.5, 0.9, 'Learned Queries', C['queries'], fontsize=10,
    sublabel='92 tokens × 256 dim')

# ── Q-Merger (Decoder blocks) ──
box(ax2, (0, 0), 3, 1.8, 'Q-Merger', C['queries'], fontsize=13,
    sublabel='6× Decoder Blocks\nGated Self-Attn + Cross-Attn')
arrow(ax2, (1.5, 2.5), (1.5, 1.8), C['encoder'])
arrow(ax2, (4, 1.9), (3, 1.2), C['queries'])

# Label: 96 tokens output
ax2.text(1.5, -0.5, 'EEG embeddings\n(n, 96, 256)', fontsize=9, ha='center',
         color=C['queries'], fontweight='bold')

# ══════════ MIDDLE: Alignment ══════════

# ── Aligner ──
box(ax2, (5, 3.8), 3, 1.5, 'Cross-Modal\nAligner', C['aligner'], fontsize=12,
    sublabel='CLIP-style contrastive\nalignment (EEG ↔ Text)')

# EEG → Aligner
arrow(ax2, (3, 4.5), (5, 4.5), C['encoder'])

# ── Text Encoder Input ──
box(ax2, (5, 6.5), 3, 1.2, 'Text Encoder', C['frozen'], fontsize=12,
    sublabel='BART-Large (frozen)\nEncode input text', dashed=True)
arrow(ax2, (6.5, 6.5), (6.5, 5.3), C['frozen'])

ax2.text(6.5, 5.6, 'Text embeddings', fontsize=8, ha='center', color=C['frozen'], style='italic')

# ══════════ RIGHT: Text Generation ══════════

# ── Decoder ──
box(ax2, (10, 3.2), 3.5, 2.5, 'Text Decoder', C['decoder'], fontsize=14,
    sublabel='BART-Large (frozen)\nAutoregressive generation\nwith cross-attention to\nEEG embeddings', dashed=True)

# EEG embeddings → Decoder (from Q-Merger)
arrow(ax2, (3, 0.9), (10, 4.0), C['queries'])
ax2.text(6.5, 2.8, 'EEG sequence\nembeddings', fontsize=8, ha='center',
         color=C['queries'], style='italic')

# ── Generated Text ──
box(ax2, (10, 0.5), 3.5, 1.2, 'Generated Text', C['text'], fontsize=13,
    sublabel='"The movie presents a compelling..."')
arrow(ax2, (11.75, 3.2), (11.75, 1.7), C['decoder'])

# ══════════ FAR RIGHT: Losses ══════════

box(ax2, (14.5, 5.5), 2.5, 2.5, 'Training\nLosses', C['loss'], fontsize=11,
    sublabel='L_CLS (contrastive)\nL_LM (language model)\nL_commit (VQ)')
arrow(ax2, (13.5, 4.5), (14.5, 6.0), C['loss'])
arrow(ax2, (13.5, 1.1), (14.5, 5.5), C['loss'])

# ══════════ Gated Attention callout ══════════
gate_text = (
    "Gated Attention\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "gate = σ(W_g · query)\n"
    "out = gate ⊙ Attn(Q,K,V)\n\n"
    "Learns to suppress\n"
    "noisy EEG channels\n"
    "& time segments"
)
ax2.text(15.5, 2.5, gate_text, fontsize=8.5, ha='center', va='center',
         color=C['gate'], fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E1', edgecolor=C['gate'], alpha=0.8))

# ══════════ Weight Init annotations ══════════
ax2.text(8.5, 8.1, '█ Trainable    ░ Frozen (pretrained)', fontsize=10,
         ha='center', color='#546E7A',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#B0BEC5'))

fig2.tight_layout()
fig2.savefig('eda_figures/architecture_02_eeg_to_text.png', dpi=200, bbox_inches='tight',
             facecolor=C['bg'])
print('Saved: architecture_02_eeg_to_text.png')
plt.close(fig2)

print('Done ✅')
