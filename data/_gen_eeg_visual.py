"""Generate a visual diagram explaining how EEG data is structured and fed to the encoder."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Load one real sample ──
df = pd.read_pickle('data/tmp/zuco_eeg_label_8variants.df')
row = df.iloc[0]
eeg = row['eeg']       # (1280, 128)
mask = row['mask']      # (1280,)
actual_len = int(mask.sum())  # 855

# ═══════════════════════════════════════════════════════════════
# FIGURE: EEG Input Structure
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14), facecolor='white')
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1.2], width_ratios=[3, 1], 
                       hspace=0.35, wspace=0.25)

# ── Panel A: EEG Heatmap with annotations ──
ax_main = fig.add_subplot(gs[0, 0])

# Show the full matrix transposed so channels = Y, time = X
im = ax_main.imshow(eeg[:, :104].T, aspect='auto', cmap='RdBu_r',
                    vmin=-8, vmax=8, interpolation='none',
                    extent=[0, 1280, 104, 0])

# Mark the padding boundary
ax_main.axvline(x=actual_len, color='#FF6600', linewidth=2.5, linestyle='--')
ax_main.text(actual_len + 5, 52, f'← Padding starts\n   (t={actual_len})', 
             color='#FF6600', fontsize=11, fontweight='bold', va='center')

# Draw region boxes
# Valid signal region
rect_valid = mpatches.FancyBboxPatch((0, 0), actual_len, 104, 
                                      boxstyle="round,pad=0", linewidth=2.5, 
                                      edgecolor='#00CC00', facecolor='none')
ax_main.add_patch(rect_valid)

# Padded time region
rect_pad_time = mpatches.FancyBboxPatch((actual_len, 0), 1280-actual_len, 104,
                                         boxstyle="round,pad=0", linewidth=2, 
                                         edgecolor='gray', facecolor='gray', alpha=0.15)
ax_main.add_patch(rect_pad_time)
ax_main.text((actual_len + 1280)/2, 52, 'ZERO-PADDED\n(time)', 
             ha='center', va='center', fontsize=12, color='gray', fontweight='bold', alpha=0.7)

ax_main.set_xlabel('Time Points (1280 total at 128 Hz → 10 seconds max)', fontsize=13, fontweight='bold')
ax_main.set_ylabel('EEG Channels (104 active electrodes)', fontsize=13, fontweight='bold')
ax_main.set_title('EEG Input Matrix — Each Cell = Voltage (μV) at one electrode at one time point',
                  fontsize=14, fontweight='bold', pad=15)

# Secondary x-axis for seconds
ax_sec = ax_main.twiny()
ax_sec.set_xlim(0, 1280/128)
ax_sec.set_xlabel('Time (seconds)', fontsize=11, color='#555')

cb = plt.colorbar(im, ax=ax_main, shrink=0.8, pad=0.02)
cb.set_label('Voltage (μV)', fontsize=11)

# ── Panel B: Single channel time series ──
ax_ts = fig.add_subplot(gs[1, 0])
ch = 5  # Pick one channel
signal = eeg[:actual_len, ch]
time_axis = np.arange(actual_len) / 128  # seconds

ax_ts.plot(time_axis, signal, color='#1976D2', linewidth=0.6, alpha=0.9)
ax_ts.fill_between(time_axis, signal, alpha=0.15, color='#1976D2')
ax_ts.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax_ts.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax_ts.set_ylabel('μV', fontsize=12, fontweight='bold')
ax_ts.set_title(f'Single Channel Time Series (Channel {ch} / E{ch+1})', 
                fontsize=12, fontweight='bold')
ax_ts.set_xlim(0, actual_len/128)

# Add annotation showing what one "cell" is
ax_ts.annotate('Each point = one cell\nin the matrix above',
               xy=(1.0, signal[128]), xytext=(2.5, signal.max() * 0.8),
               fontsize=10, fontweight='bold', color='#D32F2F',
               arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5))

# ── Panel C: Zoomed-in matrix view ──
ax_zoom = fig.add_subplot(gs[0, 1])

# Show a small 8x8 grid of actual values
n_t, n_c = 8, 8
cell_data = eeg[:n_t, :n_c]

ax_zoom.imshow(cell_data.T, aspect='auto', cmap='RdBu_r', vmin=-4, vmax=4,
               interpolation='none')

# Overlay text values
for t in range(n_t):
    for c in range(n_c):
        val = cell_data[t, c]
        color = 'white' if abs(val) > 2 else 'black'
        ax_zoom.text(t, c, f'{val:.1f}', ha='center', va='center', fontsize=7.5,
                    fontweight='bold', color=color)

ax_zoom.set_xticks(range(n_t))
ax_zoom.set_xticklabels([f't={i}\n({i/128*1000:.0f}ms)' for i in range(n_t)], fontsize=8)
ax_zoom.set_yticks(range(n_c))
ax_zoom.set_yticklabels([f'Ch{i}\n(E{i+1})' for i in range(n_c)], fontsize=8)
ax_zoom.set_title('Zoomed: First 8×8 Values (μV)', fontsize=12, fontweight='bold', pad=10)
ax_zoom.set_xlabel('Time Points', fontsize=10, fontweight='bold')
ax_zoom.set_ylabel('Channels (Electrodes)', fontsize=10, fontweight='bold')

# Add grid
for t in range(n_t + 1):
    ax_zoom.axvline(t - 0.5, color='white', linewidth=0.5)
for c in range(n_c + 1):
    ax_zoom.axhline(c - 0.5, color='white', linewidth=0.5)

# ── Panel D: Explanation text ──
ax_text = fig.add_subplot(gs[1, 1])
ax_text.axis('off')

explanation = (
    "WHAT EACH AXIS MEANS\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "ROWS (1280) = Time Points\n"
    "  • Sampled at 128 Hz\n"
    "  • Each row = 7.8ms apart\n"
    "  • Padded to 1280 (=10s max)\n\n"
    "COLUMNS (128) = Electrodes\n"
    "  • Physical sensors on scalp\n"
    "  • Ch 0–103: active electrodes\n"
    "  • Ch 104–127: zero-padded\n\n"
    "CELL VALUE = Voltage (μV)\n"
    "  • Raw electrical potential\n"
    "  • Range: approx [-30, +42]\n"
    "  • NOT frequency — frequency\n"
    "    bands are derived via FFT"
)
ax_text.text(0.05, 0.95, explanation, transform=ax_text.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#CCCCCC'))

# ── Legend ──
legend_elements = [
    mpatches.Patch(facecolor='none', edgecolor='#00CC00', linewidth=2, label='Valid EEG Signal'),
    mpatches.Patch(facecolor='gray', alpha=0.3, label='Zero-Padded Region'),
    plt.Line2D([0], [0], color='#FF6600', linewidth=2, linestyle='--', label='Padding Boundary'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11, 
           frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 0.99))

# ── Sample info ──
fig.text(0.02, 0.01, 
         f'Sample 0: Subject={row.subject} | Dataset={row.dataset} | Task={row.task} | '
         f'Text="{str(row["input text"])[:60]}..." | Sentiment={row["sentiment label"]}',
         fontsize=9, color='#666', style='italic')

plt.savefig('eda_figures/eeg_input_structure.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print('Saved to eda_figures/eeg_input_structure.png')
plt.close()
