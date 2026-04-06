"""
Plot loss/std and loss/cov (VICReg regularization losses) for V1 vs V2.
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = r"C:\MSc Files\MSc Project\E2T-w-VJEPA\gated-glim\GLIM\pretraining\Results"
OUT  = r"C:\MSc Files\MSc Project\E2T-w-VJEPA\gated-glim\GLIM\pretraining\pretrain_plot_comparison"

RUNS = {
    "V1 (Pretrain1)": {"tb": os.path.join(BASE, "GLIM_Pretrain1", "tensorboard"), "color": "#2196F3"},
    "V2 (Pretrain5)": {"tb": os.path.join(BASE, "GLIM_Pretrain5", "tensorboard"), "color": "#FF5722"},
}

def load(tb_dir):
    ea = EventAccumulator(tb_dir, size_guidance={'scalars': 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get('scalars', []):
        evs = ea.Scalars(tag)
        steps = np.array([e.step  for e in evs])
        vals  = np.array([e.value for e in evs])
        order = np.argsort(steps)
        out[tag] = (steps[order], vals[order])
    return out

def smooth(v, frac=0.08):
    w = max(3, int(len(v) * frac))
    return np.convolve(v, np.ones(w)/w, mode='valid'), w

data = {name: load(cfg["tb"]) for name, cfg in RUNS.items()}

plt.rcParams.update({
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.facecolor': '#f9f9f9',
    'figure.facecolor': 'white', 'font.size': 12,
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for tag, ax, title in [
    ('loss/std', axes[0], 'Variance Loss (loss/std)  — VICReg reg'),
    ('loss/cov', axes[1], 'Covariance Loss (loss/cov) — VICReg reg'),
]:
    for name, cfg in RUNS.items():
        if tag not in data[name]:
            ax.text(0.5, 0.5, f'{tag} not found\nfor {name}',
                    ha='center', va='center', transform=ax.transAxes)
            continue
        steps, vals = data[name][tag]
        col = cfg["color"]
        ax.plot(steps, vals, color=col, linewidth=1.2, alpha=0.35)
        if len(vals) > 8:
            sm, w = smooth(vals)
            ax.plot(steps[w-1:], sm, color=col, linewidth=2.4, label=name)
        else:
            ax.plot(steps, vals, color=col, linewidth=2.4, label=name)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

fig.suptitle('V1 vs V2 — Regularization Losses (VICReg std + cov)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(OUT, 'reg_losses_v1_vs_v2.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")

# Also print numeric summary
print("\n--- Numeric Summary ---")
for tag in ['loss/std', 'loss/cov']:
    print(f"\n{tag}:")
    for name in RUNS:
        if tag in data[name]:
            steps, vals = data[name][tag]
            print(f"  {name}: {vals[0]:.4f} → {vals[-1]:.4f}  "
                  f"(min {vals.min():.4f}  max {vals.max():.4f}  epochs {steps[0]}–{steps[-1]})")
