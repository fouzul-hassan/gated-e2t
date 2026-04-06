"""
Generate pretraining plots for GLIM_Pretrain5 from TensorBoard event files.
Matches the style of the baseline plots in pretrain_plots_baseline/.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict

matplotlib.use('Agg')

# Try to load TensorBoard events
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Installing tensorboard...")
    os.system("pip install tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------- Configuration ----------
TB_DIR = r"C:\MSc Files\MSc Project\E2T-w-VJEPA\gated-glim\GLIM\pretraining\Results\GLIM_Pretrain5\tensorboard"
OUTPUT_DIR = r"C:\MSc Files\MSc Project\E2T-w-VJEPA\gated-glim\GLIM\pretraining\pretrain_plots_pretrain5"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load TensorBoard data ----------
print(f"Loading TensorBoard data from: {TB_DIR}")

ea = EventAccumulator(TB_DIR)
ea.Reload()

# Print available tags
print("\nAvailable scalar tags:")
tags = ea.Tags().get('scalars', [])
for tag in tags:
    print(f"  - {tag}")

if not tags:
    print("\nERROR: No scalar tags found in the TensorBoard event files!")
    print("Available tag types:", ea.Tags())
    exit(1)

# Extract data into dict
data = {}
for tag in tags:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    data[tag] = {'steps': np.array(steps), 'values': np.array(values)}
    print(f"  {tag}: {len(events)} data points, range [{min(values):.4f}, {max(values):.4f}]")

# ---------- Plot style ----------
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': '#f8f8f8',
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'total': '#2196F3',
    'align': '#FF5722',
    'std': '#4CAF50',
    'cov': '#9C27B0',
    'probe': '#FF9800',
}

# ---------- Plot 1: Total Loss Curve ----------
if 'loss/total' in data:
    fig, ax = plt.subplots(figsize=(12, 6))
    d = data['loss/total']
    ax.plot(d['steps'], d['values'], color=COLORS['total'], linewidth=2, label='Total Loss')
    
    # Add smoothed line
    if len(d['values']) > 10:
        window = max(5, len(d['values']) // 20)
        smoothed = np.convolve(d['values'], np.ones(window)/window, mode='valid')
        smooth_steps = d['steps'][window-1:]
        ax.plot(smooth_steps, smoothed, color='darkblue', linewidth=2.5, alpha=0.8, 
                linestyle='--', label=f'Smoothed (window={window})')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('JEPA Pretraining v2 — Training Loss', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    
    # Annotate start and end
    ax.annotate(f'Start: {d["values"][0]:.3f}', xy=(d['steps'][0], d['values'][0]),
                fontsize=11, color='gray', ha='left')
    ax.annotate(f'End: {d["values"][-1]:.3f}', xy=(d['steps'][-1], d['values'][-1]),
                fontsize=11, color='gray', ha='right')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '01_loss_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")

# ---------- Plot 2: Loss Components ----------
component_tags = ['loss/align', 'loss/std', 'loss/cov']
available_components = [t for t in component_tags if t in data]

if available_components:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for tag in available_components:
        d = data[tag]
        name = tag.split('/')[1]
        color = COLORS.get(name, 'gray')
        ax.plot(d['steps'], d['values'], color=color, linewidth=1.5, alpha=0.7, label=f'{name} (raw)')
        
        # Smoothed
        if len(d['values']) > 10:
            window = max(5, len(d['values']) // 20)
            smoothed = np.convolve(d['values'], np.ones(window)/window, mode='valid')
            smooth_steps = d['steps'][window-1:]
            ax.plot(smooth_steps, smoothed, color=color, linewidth=2.5, 
                    label=f'{name} (smoothed)')
    
    # Also plot total if available
    if 'loss/total' in data:
        d = data['loss/total']
        ax.plot(d['steps'], d['values'], color=COLORS['total'], linewidth=1.5, 
                alpha=0.5, linestyle=':', label='total')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('JEPA Pretraining v2 — Loss Components (VICReg)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '02_loss_components.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ---------- Plot 3: Linear Probe Accuracy ----------
if 'accuracy/linear_probe' in data:
    fig, ax = plt.subplots(figsize=(12, 6))
    d = data['accuracy/linear_probe']
    
    ax.plot(d['steps'], d['values'] * 100, color=COLORS['probe'], linewidth=2.5, 
            marker='o', markersize=6, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=COLORS['probe'])
    
    # Highlight best
    best_idx = np.argmax(d['values'])
    best_epoch = d['steps'][best_idx]
    best_acc = d['values'][best_idx] * 100
    
    ax.axhline(y=best_acc, color=COLORS['probe'], linestyle='--', alpha=0.4)
    ax.annotate(f'Best: {best_acc:.1f}% (epoch {best_epoch})', 
                xy=(best_epoch, best_acc),
                xytext=(best_epoch + 10, best_acc + 2),
                fontsize=12, fontweight='bold', color='#E65100',
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('JEPA Pretraining v2 — Linear Probe Accuracy', fontsize=16, fontweight='bold')
    
    # Add reference line for chance (binary = 50%)
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.5, label='Chance (50%)')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '03_linear_probe_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ---------- Plot 4: Combined Overview ----------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4a: Total loss
if 'loss/total' in data:
    ax = axes[0, 0]
    d = data['loss/total']
    ax.plot(d['steps'], d['values'], color=COLORS['total'], linewidth=1.5, alpha=0.7)
    if len(d['values']) > 10:
        window = max(5, len(d['values']) // 20)
        smoothed = np.convolve(d['values'], np.ones(window)/window, mode='valid')
        ax.plot(d['steps'][window-1:], smoothed, color='darkblue', linewidth=2.5)
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

# 4b: Alignment loss
if 'loss/align' in data:
    ax = axes[0, 1]
    d = data['loss/align']
    ax.plot(d['steps'], d['values'], color=COLORS['align'], linewidth=1.5, alpha=0.7)
    if len(d['values']) > 10:
        window = max(5, len(d['values']) // 20)
        smoothed = np.convolve(d['values'], np.ones(window)/window, mode='valid')
        ax.plot(d['steps'][window-1:], smoothed, color='darkred', linewidth=2.5)
    ax.set_title('Alignment Loss (Prediction)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

# 4c: Std + Cov losses
ax = axes[1, 0]
for tag_name, color in [('loss/std', COLORS['std']), ('loss/cov', COLORS['cov'])]:
    if tag_name in data:
        d = data[tag_name]
        label = tag_name.split('/')[1]
        ax.plot(d['steps'], d['values'], color=color, linewidth=1.5, alpha=0.7, label=f'{label} (raw)')
        if len(d['values']) > 10:
            window = max(5, len(d['values']) // 20)
            smoothed = np.convolve(d['values'], np.ones(window)/window, mode='valid')
            ax.plot(d['steps'][window-1:], smoothed, color=color, linewidth=2.5, label=f'{label} (smoothed)')
ax.set_title('Regularization Losses (VICReg)', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=10)

# 4d: Linear probe
if 'accuracy/linear_probe' in data:
    ax = axes[1, 1]
    d = data['accuracy/linear_probe']
    ax.plot(d['steps'], d['values'] * 100, color=COLORS['probe'], linewidth=2.5,
            marker='o', markersize=5, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=COLORS['probe'])
    best_idx = np.argmax(d['values'])
    ax.axhline(y=d['values'][best_idx] * 100, color=COLORS['probe'], linestyle='--', alpha=0.4)
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.5)
    ax.set_title(f'Linear Probe Accuracy (Best: {d["values"][best_idx]*100:.1f}%)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
else:
    axes[1, 1].text(0.5, 0.5, 'No linear probe data', ha='center', va='center', fontsize=14)

fig.suptitle('JEPA Pretraining v2 — Training Overview', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '04_training_overview.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

# ---------- Plot 5: Loss convergence analysis ----------
if 'loss/total' in data:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    d = data['loss/total']
    
    # 5a: First N epochs (zoom in on early training)
    ax = axes[0]
    n_early = min(50, len(d['values']))
    ax.plot(d['steps'][:n_early], d['values'][:n_early], color=COLORS['total'], linewidth=2)
    ax.set_title(f'Early Training (Epochs 1–{d["steps"][n_early-1]})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.fill_between(d['steps'][:n_early], d['values'][:n_early], alpha=0.1, color=COLORS['total'])
    
    # 5b: Loss rate of change
    ax = axes[1]
    if len(d['values']) > 1:
        loss_diff = np.diff(d['values'])
        ax.bar(d['steps'][1:], loss_diff, color=[COLORS['std'] if v < 0 else COLORS['align'] for v in loss_diff],
               alpha=0.7, width=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title('Loss Change per Epoch (Δloss)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Δ Loss')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '05_convergence_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ---------- Summary Statistics ----------
print("\n" + "="*60)
print("JEPA Pretraining v2 — Training Summary")
print("="*60)

if 'loss/total' in data:
    d = data['loss/total']
    print(f"\nTotal Loss:")
    print(f"  Start:  {d['values'][0]:.4f} (epoch {d['steps'][0]})")
    print(f"  End:    {d['values'][-1]:.4f} (epoch {d['steps'][-1]})")
    print(f"  Min:    {d['values'].min():.4f} (epoch {d['steps'][np.argmin(d['values'])]})")
    print(f"  Max:    {d['values'].max():.4f} (epoch {d['steps'][np.argmax(d['values'])]})")

for component in ['loss/align', 'loss/std', 'loss/cov']:
    if component in data:
        d = data[component]
        name = component.split('/')[1]
        print(f"\n{name.upper()} Loss:")
        print(f"  Start:  {d['values'][0]:.4f}")
        print(f"  End:    {d['values'][-1]:.4f}")
        print(f"  Min:    {d['values'].min():.4f}")

if 'accuracy/linear_probe' in data:
    d = data['accuracy/linear_probe']
    print(f"\nLinear Probe Accuracy:")
    print(f"  Best:   {d['values'].max()*100:.1f}% (epoch {d['steps'][np.argmax(d['values'])]})")
    print(f"  Final:  {d['values'][-1]*100:.1f}% (epoch {d['steps'][-1]})")

print(f"\nTotal epochs: {max(d['steps'][-1] for d in data.values())}")
print(f"\nAll plots saved to: {OUTPUT_DIR}")
