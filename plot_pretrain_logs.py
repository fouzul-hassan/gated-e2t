"""
Plot training metrics from the JEPA pretraining notebook (e2t-jepapretrain.ipynb).
Parses the notebook output cells for INFO log lines and generates visualizations.

Usage:
    python plot_pretrain_logs.py
"""
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
from pathlib import Path

# ── 1. Parse notebook outputs ───────────────────────────────────────────────
def parse_notebook(notebook_path: str) -> list[dict]:
    """Extract all INFO log lines from notebook output cells."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    records = []
    # Regex for the main metric line
    # Example: [14,   990] loss: 0.425 | p0.425 r0.393 | input_var: 0.990 0.990 | masks: [222.0, 187.6] [wd: 4.25e-02] [lr: 1.00e-04] ...
    metric_re = re.compile(
        r"\[(\d+),\s*(\d+)\]\s+loss:\s+([\d.]+)\s+\|\s+p([\d.]+)\s+r([\d.]+)\s+\|\s+"
        r"input_var:\s+([\d.]+)\s+([\d.]+)\s+\|\s+"
        r"masks:\s+\[([\d.]+),\s+([\d.]+)\]\s+"
        r"\[wd:\s+([\d.e+-]+)\]\s+"
        r"\[lr:\s+([\d.e+-]+)\]"
    )
    # Regex for encoder gradient line
    # Example: [14,   990] enc_grad: f/l[5.30e-02 1.99e-02] mn/mx(5.79e-03, 2.73e-01) 3.76e-01
    grad_re = re.compile(
        r"\[(\d+),\s*(\d+)\]\s+enc_grad:\s+f/l\[([\d.e+-]+)\s+([\d.e+-]+)\]\s+"
        r"mn/mx\(([\d.e+-]+),\s+([\d.e+-]+)\)\s+([\d.e+-]+)"
    )

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            lines = []
            if "text" in output:
                lines = output["text"] if isinstance(output["text"], list) else [output["text"]]
            elif "data" in output and "text/plain" in output["data"]:
                txt = output["data"]["text/plain"]
                lines = txt if isinstance(txt, list) else [txt]

            for line in lines:
                m = metric_re.search(line)
                if m:
                    epoch, step = int(m.group(1)), int(m.group(2))
                    rec = {
                        "epoch": epoch, "step": step,
                        "loss": float(m.group(3)),
                        "p": float(m.group(4)),
                        "r": float(m.group(5)),
                        "input_var_ctx": float(m.group(6)),
                        "input_var_tgt": float(m.group(7)),
                        "mask_ctx": float(m.group(8)),
                        "mask_tgt": float(m.group(9)),
                        "wd": float(m.group(10)),
                        "lr": float(m.group(11)),
                    }
                    records.append(rec)

                gm = grad_re.search(line)
                if gm:
                    epoch, step = int(gm.group(1)), int(gm.group(2))
                    # Find matching record and add grad info
                    for r in reversed(records):
                        if r["epoch"] == epoch and r["step"] == step:
                            r["grad_first"] = float(gm.group(3))
                            r["grad_last"] = float(gm.group(4))
                            r["grad_min"] = float(gm.group(5))
                            r["grad_max"] = float(gm.group(6))
                            r["grad_total"] = float(gm.group(7))
                            break

    # Also parse epoch summary lines: "Epoch 17: loss=0.3959, time=249.7s"
    epoch_summary_re = re.compile(r"Epoch\s+(\d+):\s+loss=([\d.]+),\s+time=([\d.]+)s")
    epoch_summaries = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            lines = []
            if "text" in output:
                lines = output["text"] if isinstance(output["text"], list) else [output["text"]]
            for line in lines:
                em = epoch_summary_re.search(line)
                if em:
                    epoch_summaries.append({
                        "epoch": int(em.group(1)),
                        "loss": float(em.group(2)),
                        "time": float(em.group(3)),
                    })

    return records, epoch_summaries


def compute_global_step(records: list[dict], steps_per_epoch: int = 5523) -> list[dict]:
    """Add a global_step field for continuous x-axis plotting."""
    for r in records:
        r["global_step"] = r["epoch"] * steps_per_epoch + r["step"]
    return records


# ── 2. Plotting functions ───────────────────────────────────────────────────
def setup_style():
    """Set up a clean, publication-quality plot style."""
    plt.rcParams.update({
        "figure.facecolor": "#f8f9fa",
        "axes.facecolor": "#ffffff",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 120,
    })


def plot_loss_curve(records, epoch_summaries, save_dir):
    """Plot 1: Overall loss curve (global step + epoch-end markers)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    steps = [r["global_step"] for r in records]
    losses = [r["loss"] for r in records]
    
    # Subsample for cleaner plotting (every 10th point)
    n = max(1, len(steps) // 500)
    ax.plot(steps[::n], losses[::n], color="#4a90d9", linewidth=0.8, alpha=0.7, label="Step loss")
    
    # Epoch-end markers
    if epoch_summaries:
        ep_x = [es["epoch"] * 5523 + 5523 for es in epoch_summaries]
        ep_y = [es["loss"] for es in epoch_summaries]
        ax.scatter(ep_x, ep_y, color="#e74c3c", s=60, zorder=5, label="Epoch-end loss", edgecolors="white", linewidths=1.2)
        for es in epoch_summaries:
            ax.annotate(f'E{es["epoch"]}: {es["loss"]:.4f}',
                       (es["epoch"] * 5523 + 5523, es["loss"]),
                       textcoords="offset points", xytext=(8, 8), fontsize=8, color="#e74c3c")
    
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.set_title("JEPA Pretraining Loss Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_dir / "01_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 01_loss_curve.png")


def plot_loss_components(records, save_dir):
    """Plot 2: Prediction vs Regularization loss components."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    steps = [r["global_step"] for r in records]
    n = max(1, len(steps) // 500)
    
    ax.plot(steps[::n], [r["p"] for r in records][::n], color="#3498db", linewidth=0.8, alpha=0.7, label="Prediction (p)")
    ax.plot(steps[::n], [r["r"] for r in records][::n], color="#e67e22", linewidth=0.8, alpha=0.7, label="Regularization (r)")
    
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss Component")
    ax.set_title("Loss Components: Prediction vs Regularization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "02_loss_components.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 02_loss_components.png")


def plot_input_variance(records, save_dir):
    """Plot 3: Input variance (collapse detection)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    
    steps = [r["global_step"] for r in records]
    n = max(1, len(steps) // 500)
    
    ax.plot(steps[::n], [r["input_var_ctx"] for r in records][::n], color="#27ae60", linewidth=1.0, label="Context var")
    ax.plot(steps[::n], [r["input_var_tgt"] for r in records][::n], color="#8e44ad", linewidth=1.0, alpha=0.7, label="Target var", linestyle="--")
    
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="Collapse threshold")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Input Variance")
    ax.set_title("Input Variance (Collapse Monitor) — Should Stay Near 1.0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "03_input_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 03_input_variance.png")


def plot_encoder_gradients(records, save_dir):
    """Plot 4: Encoder gradient norms over training."""
    grad_records = [r for r in records if "grad_max" in r]
    if not grad_records:
        print("  ⚠️  No gradient data found, skipping gradient plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    steps = [r["global_step"] for r in grad_records]
    n = max(1, len(steps) // 500)
    
    # Top: first/last layer grad norms
    ax = axes[0]
    ax.plot(steps[::n], [r["grad_first"] for r in grad_records][::n], color="#3498db", linewidth=0.7, alpha=0.7, label="First layer")
    ax.plot(steps[::n], [r["grad_last"] for r in grad_records][::n], color="#e67e22", linewidth=0.7, alpha=0.7, label="Last layer")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Encoder Gradient Norms (First / Last Layer)")
    ax.legend()
    ax.set_yscale("log")
    
    # Bottom: max gradient and total gradient norm
    ax = axes[1]
    ax.plot(steps[::n], [r["grad_max"] for r in grad_records][::n], color="#e74c3c", linewidth=0.7, alpha=0.7, label="Max grad")
    ax.plot(steps[::n], [r["grad_total"] for r in grad_records][::n], color="#9b59b6", linewidth=0.7, alpha=0.5, label="Total norm")
    ax.axhline(y=1.0, color="red", linestyle=":", alpha=0.4, label="Clip threshold (1.0)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Max & Total Encoder Gradient Norm (Spike Detection)")
    ax.legend()
    ax.set_yscale("log")
    
    fig.tight_layout()
    fig.savefig(save_dir / "04_encoder_gradients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 04_encoder_gradients.png")


def plot_lr_and_wd(records, save_dir):
    """Plot 5: Learning rate and weight decay schedules."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    steps = [r["global_step"] for r in records]
    n = max(1, len(steps) // 500)
    
    axes[0].plot(steps[::n], [r["lr"] for r in records][::n], color="#2c3e50", linewidth=1.2)
    axes[0].set_xlabel("Global Step")
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_title("Learning Rate Schedule")
    axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
    
    axes[1].plot(steps[::n], [r["wd"] for r in records][::n], color="#16a085", linewidth=1.2)
    axes[1].set_xlabel("Global Step")
    axes[1].set_ylabel("Weight Decay")
    axes[1].set_title("Weight Decay Schedule")
    
    fig.tight_layout()
    fig.savefig(save_dir / "05_lr_wd_schedule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 05_lr_wd_schedule.png")


def plot_mask_sizes(records, save_dir):
    """Plot 6: Context and target mask sizes over training."""
    fig, ax = plt.subplots(figsize=(14, 4))
    
    steps = [r["global_step"] for r in records]
    n = max(1, len(steps) // 500)
    
    ax.plot(steps[::n], [r["mask_ctx"] for r in records][::n], color="#2980b9", linewidth=0.8, alpha=0.7, label="Context mask size")
    ax.plot(steps[::n], [r["mask_tgt"] for r in records][::n], color="#c0392b", linewidth=0.8, alpha=0.7, label="Target mask size")
    
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Mask Size (tokens)")
    ax.set_title("Masking Strategy: Context vs Target Mask Sizes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "06_mask_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 06_mask_sizes.png")


def plot_epoch_summary(epoch_summaries, save_dir):
    """Plot 7: Per-epoch loss bar chart and epoch training time."""
    if not epoch_summaries:
        print("  ⚠️  No epoch summary data, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = [es["epoch"] for es in epoch_summaries]
    losses = [es["loss"] for es in epoch_summaries]
    times = [es["time"] for es in epoch_summaries]
    
    # Loss per epoch
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(epochs)))
    axes[0].bar(epochs, losses, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Epoch-End Loss")
    axes[0].set_title("Loss per Epoch")
    for i, (e, l) in enumerate(zip(epochs, losses)):
        axes[0].text(e, l + 0.001, f"{l:.4f}", ha="center", va="bottom", fontsize=8, rotation=45)
    
    # Time per epoch
    axes[1].bar(epochs, times, color="#3498db", edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Time (seconds)")
    axes[1].set_title("Training Time per Epoch")
    for i, (e, t) in enumerate(zip(epochs, times)):
        axes[1].text(e, t + 1, f"{t:.0f}s", ha="center", va="bottom", fontsize=8)
    
    fig.tight_layout()
    fig.savefig(save_dir / "07_epoch_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 07_epoch_summary.png")


def plot_gradient_spike_histogram(records, save_dir):
    """Plot 8: Histogram of max gradient values to show spike distribution."""
    grad_records = [r for r in records if "grad_max" in r]
    if not grad_records:
        return

    max_grads = [r["grad_max"] for r in grad_records]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Full distribution
    axes[0].hist(max_grads, bins=80, color="#3498db", alpha=0.8, edgecolor="white", linewidth=0.3)
    axes[0].axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="Spike threshold (1.0)")
    axes[0].set_xlabel("Max Gradient Norm")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Max Gradient Norms")
    axes[0].legend()
    
    # Spike percentage over time (rolling)
    window = 100
    spikes = [1 if g > 0.5 else 0 for g in max_grads]
    if len(spikes) > window:
        rolling_pct = [sum(spikes[max(0,i-window):i]) / min(i, window) * 100 for i in range(1, len(spikes)+1)]
        steps = [r["global_step"] for r in grad_records]
        n = max(1, len(steps) // 500)
        axes[1].plot(steps[::n], rolling_pct[::n], color="#e74c3c", linewidth=1.0)
        axes[1].set_xlabel("Global Step")
        axes[1].set_ylabel("Spike % (rolling 100 steps)")
        axes[1].set_title("Gradient Spike Frequency (>0.5 threshold)")
    
    fig.tight_layout()
    fig.savefig(save_dir / "08_gradient_spikes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 08_gradient_spikes.png")


# ── 3. Main ─────────────────────────────────────────────────────────────────
def main():
    notebook_path = Path(__file__).parent / "e2t-jepapretrain.ipynb"
    save_dir = Path(__file__).parent / "pretrain_plots"
    save_dir.mkdir(exist_ok=True)
    
    print(f"📖 Parsing notebook: {notebook_path}")
    records, epoch_summaries = parse_notebook(str(notebook_path))
    print(f"   Found {len(records)} metric records, {len(epoch_summaries)} epoch summaries")
    
    if not records:
        print("❌ No records found! Check that the notebook has output cells.")
        return
    
    records = compute_global_step(records)
    
    # Print quick stats
    epochs_seen = sorted(set(r["epoch"] for r in records))
    print(f"   Epochs: {min(epochs_seen)} → {max(epochs_seen)} ({len(epochs_seen)} epochs logged)")
    print(f"   Loss range: {min(r['loss'] for r in records):.4f} → {max(r['loss'] for r in records):.4f}")
    
    print(f"\n📊 Generating plots to: {save_dir}")
    setup_style()
    
    plot_loss_curve(records, epoch_summaries, save_dir)
    plot_loss_components(records, save_dir)
    plot_input_variance(records, save_dir)
    plot_encoder_gradients(records, save_dir)
    plot_lr_and_wd(records, save_dir)
    plot_mask_sizes(records, save_dir)
    plot_epoch_summary(epoch_summaries, save_dir)
    plot_gradient_spike_histogram(records, save_dir)
    
    print(f"\n✅ All plots saved to: {save_dir}")
    print("   Open the folder to view the PNG files.")


if __name__ == "__main__":
    main()
