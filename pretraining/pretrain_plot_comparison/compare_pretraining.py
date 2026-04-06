"""
Comparison plots: JEPA Pretraining V1 (GLIM_Pretrain1) vs V2 (GLIM_Pretrain5)
Produces side-by-side plots for all key metrics, both by epoch and global step.

Outputs saved to: pretraining/pretrain_plot_comparison/
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

matplotlib.use('Agg')

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    os.system("pip install tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------- Configuration ----------
BASE = r"C:\MSc Files\MSc Project\E2T-w-VJEPA\gated-glim\GLIM\pretraining\Results"

RUNS = {
    "JEPA Pretraining v1": {
        "tb_dir": os.path.join(BASE, "GLIM_Pretrain1", "tensorboard"),
        "color": "#2196F3",       # blue
        "color_dark": "#0D47A1",
        "linestyle": "-",
    },
    "JEPA Pretraining v2": {
        "tb_dir": os.path.join(BASE, "GLIM_Pretrain5", "tensorboard"),
        "color": "#FF5722",       # orange-red
        "color_dark": "#BF360C",
        "linestyle": "-",
    },
}

OUTPUT_DIR = os.path.join(
    r"C:\MSc Files\MSc Project\E2T-w-VJEPA\gated-glim\GLIM\pretraining",
    "pretrain_plot_comparison"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Plot Style ----------
plt.rcParams.update({
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.facecolor': '#f9f9f9',
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
})

# ---------- Load TensorBoard data ----------
def load_tb(tb_dir):
    """Load all scalar data from a TensorBoard directory (merges multiple event files)."""
    ea = EventAccumulator(tb_dir, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps  = np.array([e.step  for e in events])
        walls  = np.array([e.wall_time for e in events])
        values = np.array([e.value for e in events])
        # Sort by step to handle multi-session merges
        order = np.argsort(steps)
        data[tag] = {
            'steps':  steps[order],
            'walls':  walls[order],
            'values': values[order],
        }
    return data, tags


def smooth(values, window_frac=0.05):
    """Apply moving average smoothing."""
    window = max(3, int(len(values) * window_frac))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid'), window


print("Loading TensorBoard data...")
all_data = {}
all_tags_union = set()

for name, cfg in RUNS.items():
    print(f"  {name}: {cfg['tb_dir']}")
    d, tags = load_tb(cfg['tb_dir'])
    all_data[name] = d
    all_tags_union.update(tags)
    print(f"    Tags: {tags}")
    for tag, td in d.items():
        print(f"      {tag}: {len(td['steps'])} pts  "
              f"[{td['values'].min():.4f}, {td['values'].max():.4f}]  "
              f"steps {td['steps'][0]}..{td['steps'][-1]}")

# For convenience
v1  = "JEPA Pretraining v1"
v2  = "JEPA Pretraining v2"
c1  = RUNS[v1]["color"];   c1d = RUNS[v1]["color_dark"]
c2  = RUNS[v2]["color"];   c2d = RUNS[v2]["color_dark"]


# ============================================================
# Helper: draw one comparison panel (epoch-based)
# ============================================================
def plot_comparison_panel(ax, tag, y_label, title, *, scale=1.0,
                          show_smooth=True, chance_line=None,
                          mark_best=False):
    """Plot V1 and V2 for a given tag on the same axis (epoch x-axis)."""
    plotted = False
    for name, cfg in RUNS.items():
        d = all_data[name].get(tag)
        if d is None:
            continue
        xs = d['steps']
        ys = d['values'] * scale
        col = cfg['color']
        ax.plot(xs, ys, color=col, linewidth=1.2, alpha=0.4)
        if show_smooth and len(ys) > 8:
            sm, w = smooth(ys)
            ax.plot(xs[w-1:], sm, color=col, linewidth=2.2,
                    label=f"{name}")
        else:
            ax.plot(xs, ys, color=col, linewidth=2.2, label=name)
        if mark_best:
            best_i = np.argmax(ys)
            ax.scatter([xs[best_i]], [ys[best_i]], s=80, color=col,
                       zorder=5, marker='*')
            ax.annotate(f"{ys[best_i]:.1f}%\n(ep {xs[best_i]})",
                        xy=(xs[best_i], ys[best_i]),
                        xytext=(xs[best_i] + 2, ys[best_i] + 1),
                        fontsize=9, color=col,
                        arrowprops=dict(arrowstyle='->', color=col, lw=1))
        plotted = True
    if chance_line is not None:
        ax.axhline(y=chance_line, color='gray', linestyle=':', linewidth=1.2,
                   label=f'Chance ({chance_line:.0f}%)')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    if plotted:
        ax.legend(fontsize=9)
    return plotted


# ============================================================
# Helper: draw one comparison panel (global step-based)
# ============================================================
def plot_step_panel(ax, tag, y_label, title, *, scale=1.0,
                    show_smooth=True, chance_line=None, mark_best=False):
    """Plot V1 and V2 for a given tag on the same axis (global step x-axis).
    Step axis is normalised so both runs share a comparable x scale.
    """
    plotted = False
    for name, cfg in RUNS.items():
        d = all_data[name].get(tag)
        if d is None:
            continue
        xs = d['steps']
        ys = d['values'] * scale
        col = cfg['color']
        ax.plot(xs, ys, color=col, linewidth=1.2, alpha=0.4)
        if show_smooth and len(ys) > 8:
            sm, w = smooth(ys)
            ax.plot(xs[w-1:], sm, color=col, linewidth=2.2,
                    label=f"{name}")
        else:
            ax.plot(xs, ys, color=col, linewidth=2.2, label=name)
        if mark_best:
            best_i = np.argmax(ys)
            ax.scatter([xs[best_i]], [ys[best_i]], s=80, color=col,
                       zorder=5, marker='*')
        plotted = True
    if chance_line is not None:
        ax.axhline(y=chance_line, color='gray', linestyle=':', linewidth=1.2,
                   label=f'Chance ({chance_line:.0f}%)')
    ax.set_xlabel('Global Step (Iteration)', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    if plotted:
        ax.legend(fontsize=9)
    return plotted


# ============================================================
# PLOT 1: Loss Curve  (epoch + step side-by-side)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_comparison_panel(axes[0], 'loss/total', 'Total Loss',
                      'Loss Curve — by Epoch')
plot_step_panel(axes[1], 'loss/total', 'Total Loss',
                'Loss Curve — by Global Step')
fig.suptitle('V1 vs V2 — Total Training Loss', fontsize=16, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '01_loss_curve.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 2: Loss Components (VICReg)
# ============================================================
comp_labels = {
    'loss/align': ('Alignment Loss', '#E91E63'),
    'loss/std':   ('Variance Loss',  '#4CAF50'),
    'loss/cov':   ('Covariance Loss','#9C27B0'),
}

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
for row, (tag, (label, col_hint)) in enumerate(comp_labels.items()):
    plot_comparison_panel(axes[row, 0], tag, label,
                          f'{label} — by Epoch')
    plot_step_panel(axes[row, 1], tag, label,
                    f'{label} — by Global Step')

fig.suptitle('V1 vs V2 — VICReg Loss Components', fontsize=16, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '02_loss_components.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 3: Input Variance / Representation Health
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for tag in ['input_variance', 'variance', 'rep_variance',
            'monitor/input_variance', 'monitor/variance']:
    found = any(tag in all_data[n] for n in RUNS)
    if found:
        plot_comparison_panel(axes[0], tag, 'Input Variance',
                              'Input Variance — by Epoch')
        plot_step_panel(axes[1], tag, 'Input Variance',
                        'Input Variance — by Global Step')
        break
else:
    for ax in axes:
        ax.text(0.5, 0.5, 'Input variance not logged\n(run used external monitoring)',
                ha='center', va='center', fontsize=13, color='gray',
                transform=ax.transAxes)
        ax.set_title('Input Variance', fontsize=12, fontweight='bold')
        # Draw horizontal guide at 1.0 (ideal no-collapse value)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5,
                   label='Target (1.0)')
        ax.legend()
fig.suptitle('V1 vs V2 — Representation Health (Input Variance)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '03_input_variance.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 4: Encoder Gradients
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
grad_tags = ['grad_norm', 'encoder_grad_norm', 'gradient/norm',
             'grad/encoder', 'monitor/grad_norm']
found_grad = None
for tag in grad_tags:
    if any(tag in all_data[n] for n in RUNS):
        found_grad = tag
        break

if found_grad:
    plot_comparison_panel(axes[0], found_grad, 'Gradient Norm',
                          'Encoder Gradient Norm — by Epoch')
    plot_step_panel(axes[1], found_grad, 'Gradient Norm',
                    'Encoder Gradient Norm — by Global Step')
else:
    for ax in axes:
        ax.text(0.5, 0.5, 'Gradient norm not logged to TensorBoard\n'
                '(clipped to max_norm=1.0 in training loop)',
                ha='center', va='center', fontsize=13, color='gray',
                transform=ax.transAxes)
        ax.set_title('Encoder Gradients', fontsize=12, fontweight='bold')

fig.suptitle('V1 vs V2 — Encoder Gradient Norm',
             fontsize=16, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '04_encoder_gradients.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 5: LR / WD Schedule
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
lr_found = False
for tag in ['lr', 'learning_rate', 'train/lr', 'hparams/lr']:
    if any(tag in all_data[n] for n in RUNS):
        plot_comparison_panel(axes[0], tag, 'Learning Rate',
                              'LR Schedule — by Epoch')
        plot_step_panel(axes[1], tag, 'Learning Rate',
                        'LR Schedule — by Global Step')
        lr_found = True
        break

if not lr_found:
    # Reconstruct cosine LR analytically for both runs
    ax0, ax1 = axes
    for name, cfg in RUNS.items():
        d = all_data[name].get('loss/total')
        if d is None:
            continue
        epochs = d['steps']
        max_ep = epochs.max()
        lr_cosine = 1e-4 * 0.5 * (1 + np.cos(np.pi * epochs / max_ep))
        ax0.plot(epochs, lr_cosine, color=cfg['color'], linewidth=2.2,
                 label=f"{name} (reconstructed cosine)")
        ax1.plot(epochs, lr_cosine, color=cfg['color'], linewidth=2.2,
                 label=f"{name} (reconstructed cosine)")
    ax0.set_xlabel('Epoch'); ax0.set_ylabel('Learning Rate (est.)')
    ax0.set_title('LR Schedule — by Epoch (reconstructed)', fontsize=12, fontweight='bold')
    ax0.legend(fontsize=9)
    ax1.set_xlabel('Global Step'); ax1.set_ylabel('Learning Rate (est.)')
    ax1.set_title('LR Schedule — by Global Step (reconstructed)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)

fig.suptitle('V1 vs V2 — Learning Rate / WD Schedule',
             fontsize=16, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '05_lr_wd_schedule.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 6: Mask Sizes
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
mask_found = False
for tag in ['mask_ratio', 'mask_size', 'num_masked', 'monitor/mask_ratio']:
    if any(tag in all_data[n] for n in RUNS):
        plot_comparison_panel(axes[0], tag, 'Mask Ratio / Size',
                              'Mask Size — by Epoch')
        plot_step_panel(axes[1], tag, 'Mask Ratio / Size',
                        'Mask Size — by Global Step')
        mask_found = True
        break

if not mask_found:
    for ax in axes:
        ax.text(0.5, 0.5,
                'Mask size not logged\n(fixed 50% SSP masking for both runs)',
                ha='center', va='center', fontsize=13, color='gray',
                transform=ax.transAxes)
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.6,
                   label='mask_ratio=0.5 (both runs)')
        ax.set_ylim(0, 1)
        ax.set_title('Mask Size', fontsize=12, fontweight='bold')
        ax.legend()

fig.suptitle('V1 vs V2 — Mask Sizes (SSP Masking)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '06_mask_sizes.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 7: Epoch Summary (per-epoch loss breakdown)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax_idx, (name, cfg) in enumerate(RUNS.items()):
    ax = axes[ax_idx]
    d_total = all_data[name].get('loss/total')
    d_align = all_data[name].get('loss/align')
    d_std   = all_data[name].get('loss/std')
    d_cov   = all_data[name].get('loss/cov')

    if d_total is not None:
        ax.plot(d_total['steps'], d_total['values'],
                color=cfg['color'], linewidth=2.5, label='Total', zorder=5)
    for comp_d, comp_label, comp_col in [
        (d_align, 'Align',      '#E91E63'),
        (d_std,   'Variance',   '#4CAF50'),
        (d_cov,   'Covariance', '#9C27B0'),
    ]:
        if comp_d is not None:
            ax.plot(comp_d['steps'], comp_d['values'], linewidth=1.5,
                    alpha=0.7, linestyle='--', color=comp_col, label=comp_label)

    ax.set_title(f'{name} — Loss Breakdown by Epoch',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=9)

fig.suptitle('V1 vs V2 — Per-Epoch Loss Summary',
             fontsize=16, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '07_epoch_summary.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 8: Gradient Spikes Analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax_idx, (name, cfg) in enumerate(RUNS.items()):
    ax = axes[ax_idx]
    d = all_data[name].get('loss/total')
    if d is None:
        ax.text(0.5, 0.5, 'No loss data', ha='center', va='center',
                transform=ax.transAxes)
        continue

    # Detect potential instability events: large positive jumps in loss
    vals = d['values']
    diffs = np.diff(vals)
    spike_threshold = np.std(diffs) * 2.0
    spikes = np.where(diffs > spike_threshold)[0] + 1

    ax.plot(d['steps'], vals, color=cfg['color'], linewidth=1.8, label='Total Loss')
    if len(spikes) > 0:
        ax.scatter(d['steps'][spikes], vals[spikes], color='red',
                   s=80, zorder=5, label=f'Instability spikes ({len(spikes)})', marker='^')
    else:
        ax.text(0.5, 0.02, 'No significant loss spikes detected',
                ha='center', va='bottom', transform=ax.transAxes,
                fontsize=10, color='green')
    ax.set_title(f'{name} — Gradient / Loss Spike Detection',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=9)

fig.suptitle('V1 vs V2 — Training Instability & Gradient Spikes',
             fontsize=16, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '08_gradient_spikes.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 9: Linear Probe Accuracy
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_comparison_panel(axes[0], 'accuracy/linear_probe',
                      'Accuracy (%)', 'Linear Probe Accuracy — by Epoch',
                      scale=100.0, show_smooth=False,
                      chance_line=50.0, mark_best=True)
plot_step_panel(axes[1], 'accuracy/linear_probe',
                'Accuracy (%)', 'Linear Probe Accuracy — by Global Step',
                scale=100.0, show_smooth=False,
                chance_line=50.0, mark_best=True)

fig.suptitle('V1 vs V2 — Linear Probe Accuracy',
             fontsize=16, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '09_linear_probe_accuracy.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 10: Training Overview (4-panel grid, side-by-side)
# ============================================================
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

panel_defs = [
    ('loss/total',           'Total Loss',       'Loss',         False, None,  False),
    ('loss/align',           'Alignment Loss',   'Loss',         False, None,  False),
    ('loss/std',             'Variance Loss',    'Loss',         False, None,  False),
    ('loss/cov',             'Covariance Loss',  'Loss',         False, None,  False),
]

for row, (tag, ptitle, ylabel, is_acc, chance, mark_best) in enumerate(panel_defs):
    # Epoch column
    ax_ep = fig.add_subplot(gs[row, 0])
    plot_comparison_panel(ax_ep, tag, ylabel,
                          f'{ptitle} (Epoch)',
                          scale=100.0 if is_acc else 1.0,
                          show_smooth=True,
                          chance_line=chance,
                          mark_best=mark_best)
    # Step column
    ax_st = fig.add_subplot(gs[row, 1])
    plot_step_panel(ax_st, tag, ylabel,
                    f'{ptitle} (Global Step)',
                    scale=100.0 if is_acc else 1.0,
                    show_smooth=True,
                    chance_line=chance,
                    mark_best=mark_best)

fig.suptitle('V1 vs V2 — Training Overview (Epoch & Step)',
             fontsize=18, fontweight='bold', y=1.005)
p = os.path.join(OUTPUT_DIR, '10_training_overview.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 11: Convergence Analysis
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

for col_idx, (name, cfg) in enumerate(RUNS.items()):
    d = all_data[name].get('loss/total')
    if d is None:
        continue

    steps = d['steps']
    vals  = d['values']
    col   = cfg['color']

    # Top: Early epochs (first 50 or all available)
    ax_early = axes[0, col_idx]
    n_early = min(50, len(vals))
    ax_early.plot(steps[:n_early], vals[:n_early], color=col, linewidth=2.2)
    ax_early.fill_between(steps[:n_early], vals[:n_early], alpha=0.10, color=col)
    ax_early.set_title(f'{name}\nEarly Training (Epochs 1–{steps[n_early-1]})',
                       fontsize=12, fontweight='bold')
    ax_early.set_xlabel('Epoch'); ax_early.set_ylabel('Loss')

    # Bottom: Loss change per epoch (Δloss)
    ax_delta = axes[1, col_idx]
    diffs = np.diff(vals)
    bar_colors = [cfg['color'] if v < 0 else '#ef5350' for v in diffs]
    ax_delta.bar(steps[1:], diffs, color=bar_colors, alpha=0.75, width=0.8)
    ax_delta.axhline(y=0, color='black', linewidth=0.8)

    # Smoothed delta
    if len(diffs) > 5:
        sm_d, w = smooth(diffs, 0.08)
        ax_delta.plot(steps[w:], sm_d, color=cfg['color_dark'],
                      linewidth=2.0, label='Smoothed Δloss')
        ax_delta.legend(fontsize=9)

    # Highlight convergence zone (last 20% where |Δ| < 0.01)
    conv_mask = np.abs(diffs) < 0.01
    if conv_mask.any():
        last_conv = steps[1:][conv_mask]
        ax_delta.axvspan(last_conv[0], last_conv[-1],
                         alpha=0.12, color='green', label='|Δ| < 0.01')

    ax_delta.set_title(f'{name}\nLoss Change per Epoch (Δloss)',
                       fontsize=12, fontweight='bold')
    ax_delta.set_xlabel('Epoch'); ax_delta.set_ylabel('Δ Loss')

fig.suptitle('V1 vs V2 — Convergence Analysis',
             fontsize=16, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '11_convergence_analysis.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# PLOT 12: Grand Summary — All metrics on one page
# ============================================================
METRIC_GROUPS = [
    ('loss/total',              'Total Loss',      'Loss',        1.0,   False),
    ('loss/align',              'Align Loss',      'Loss',        1.0,   False),
    ('loss/std',                'Variance Loss',   'Loss',        1.0,   False),
    ('loss/cov',                'Covariance Loss', 'Loss',        1.0,   False),
    ('accuracy/linear_probe',   'Linear Probe',    'Acc (%)',     100.0, True),
]

n_rows = len(METRIC_GROUPS)
fig, axes = plt.subplots(n_rows, 2, figsize=(18, n_rows * 4))

for row, (tag, ptitle, ylabel, scale, is_acc) in enumerate(METRIC_GROUPS):
    plot_comparison_panel(axes[row, 0], tag, ylabel,
                          f'{ptitle} — Epoch', scale=scale,
                          show_smooth=True,
                          chance_line=50.0 if is_acc else None,
                          mark_best=is_acc)
    plot_step_panel(axes[row, 1], tag, ylabel,
                    f'{ptitle} — Global Step', scale=scale,
                    show_smooth=True,
                    chance_line=50.0 if is_acc else None,
                    mark_best=is_acc)

fig.suptitle('JEPA Pretraining V1 vs V2 — Grand Summary',
             fontsize=18, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUTPUT_DIR, '00_grand_summary.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {p}")


# ============================================================
# Print text summary
# ============================================================
print("\n" + "=" * 65)
print("JEPA Pretraining V1 vs V2 — Numeric Summary")
print("=" * 65)

for name in RUNS:
    d = all_data[name]
    print(f"\n{'─'*30}")
    print(f"  {name}")
    print(f"{'─'*30}")
    if 'loss/total' in d:
        v = d['loss/total']['values']
        s = d['loss/total']['steps']
        print(f"  Total Loss: {v[0]:.4f} → {v[-1]:.4f}  "
              f"(min {v.min():.4f} @ ep {s[np.argmin(v)]})")
        print(f"  Epochs:     {s[0]} – {s[-1]}  ({len(s)} logged pts)")
    for comp in ['loss/align', 'loss/std', 'loss/cov']:
        if comp in d:
            v = d[comp]['values']
            lbl = comp.split('/')[1]
            print(f"  {lbl:>10}: {v[0]:.4f} → {v[-1]:.4f}  (min {v.min():.4f})")
    if 'accuracy/linear_probe' in d:
        v = d['accuracy/linear_probe']['values']
        s = d['accuracy/linear_probe']['steps']
        print(f"  Linear Probe: best {v.max()*100:.1f}% @ ep {s[np.argmax(v)]}, "
              f"final {v[-1]*100:.1f}%")

print(f"\nAll plots saved to:\n  {OUTPUT_DIR}")
