"""
EEG Visualisation helpers for the GLIM demo.

Two plots:
  1. Butterfly plot   — all 128 channels overlaid as time-series waveforms
  2. Topographic maps — spatial amplitude snapshots at each word-aligned window
                        (requires MNE; gracefully degrades to a scatter fallback)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless rendering for Gradio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import Normalize

# ─────────────────────────────────────────────────────────────────────────────
# Approximate 128-channel layout (BioSemi-128 polar coordinates)
# (angle in degrees from anterior, radius normalised 0→1 = centre→ear)
# For a proper demo, replace with exact montage from your cap file.
# BioSemi 128 approximate positions loaded from MNE if available, else fallback grid.
# ─────────────────────────────────────────────────────────────────────────────

def _get_channel_positions(n_channels: int = 128):
    """Return (x, y) positions for n_channels on a unit-radius head circle."""
    try:
        import mne
        montage = mne.channels.make_standard_montage('biosemi128')
        pos3d   = np.array([montage.get_positions()['ch_pos'][ch]
                             for ch in montage.ch_names[:n_channels]])
        x, y    = pos3d[:, 0], pos3d[:, 1]
        # Normalise to [-1, 1]
        x  = x / np.abs(x).max()
        y  = y / np.abs(y).max()
        return x[:n_channels], y[:n_channels]
    except Exception:
        # Fallback: evenly spaced on a circle
        angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
        r      = np.linspace(0.2, 0.95, n_channels // 4)
        radii  = np.tile(r, 4)[:n_channels]
        return radii * np.cos(angles), radii * np.sin(angles)


# Pre-compute once
_CH_X, _CH_Y = _get_channel_positions(128)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Butterfly Plot
# ─────────────────────────────────────────────────────────────────────────────

def butterfly_plot(eeg: np.ndarray, mask: np.ndarray,
                   title: str = "EEG Signal — Butterfly Plot",
                   max_channels: int = 128,
                   alpha: float = 0.45):
    """
    Plot all channels as overlaid time-series waveforms (butterfly plot) interactively.
    """
    import plotly.graph_objects as go

    valid_len = int(mask.sum())
    eeg = eeg[:valid_len]
    
    T, C = eeg.shape
    C    = min(C, max_channels)
    t    = np.arange(T)

    fig = go.Figure()

    # Pre-render traces using Scattergl (WebGL) for high-performance large data
    for ch in range(C):
        if ch < len(_CH_X):
            x = _CH_X[ch]
            if   x < -0.1: color = f'rgba(51, 114, 204, {alpha})'   # blue
            elif x >  0.1: color = f'rgba(216, 64, 64, {alpha})'    # red
            else:          color = f'rgba(128, 128, 128, {alpha})'  # grey
        else:
            color = f'rgba(128, 128, 128, {alpha})'
            
        fig.add_trace(go.Scattergl(
            x=t, y=eeg[:, ch],
            mode='lines',
            line=dict(color=color, width=1),
            name=f'Channel {ch+1}',
            showlegend=False,
            hoverinfo='name+y'
        ))

    # Add custom manual legend entries
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgb(51, 114, 204)', width=2), name='Left Hemisphere'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgb(216, 64, 64)', width=2), name='Right Hemisphere'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgb(128, 128, 128)', width=2), name='Midline'))

    fig.update_layout(
        title=title,
        title_font_color="white",
        plot_bgcolor="#0f0f1a",
        paper_bgcolor="#0f0f1a",
        font=dict(color="#aaa"),
        xaxis=dict(title="Time (samples)", showgrid=False, zeroline=False, linecolor="#444"),
        yaxis=dict(title="Amplitude (µV)", showgrid=False, zeroline=False, linecolor="#444"),
        legend=dict(x=1.02, y=1, bgcolor="rgba(26,26,42,0.8)", font=dict(color="white"), bordercolor="#444", borderwidth=1),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
        hovermode="x unified"
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Topographic Word Snapshots
# ─────────────────────────────────────────────────────────────────────────────

def _head_outline(ax, radius=1.0):
    """Draw a head outline, ears, and nose on ax."""
    # Head circle
    head = Circle((0, 0), radius, fill=False, edgecolor='#888', linewidth=1.5)
    ax.add_patch(head)
    # Nose
    nose_x = [-.07, 0, .07]
    nose_y = [radius - 0.02, radius + 0.12, radius - 0.02]
    ax.plot(nose_x, nose_y, color='#888', linewidth=1.5)
    # Left ear
    ax.plot([-radius, -radius - 0.08, -radius - 0.08, -radius],
            [-0.07, -0.02, 0.07, 0.12], color='#888', linewidth=1.5)
    # Right ear
    ax.plot([radius, radius + 0.08, radius + 0.08, radius],
            [-0.07, -0.02, 0.07, 0.12], color='#888', linewidth=1.5)


def topomap_frame(eeg: np.ndarray, mask: np.ndarray,
                  window_start: int, window_len: int = 50,
                  word_label: str = '',
                  figsize=(3.5, 3.5)) -> plt.Figure:
    """
    One topographic snapshot: mean amplitude across a time window.

    Args:
        eeg:          (T, C) float32
        window_start: start sample index
        window_len:   number of samples to average
        word_label:   word displayed under the head
    """
    end   = min(window_start + window_len, eeg.shape[0])
    frame = eeg[window_start:end].mean(axis=0)   # (C,)

    C  = min(len(frame), len(_CH_X))
    xi = _CH_X[:C]
    yi = _CH_Y[:C]
    zi = frame[:C]

    # Interpolate onto a regular grid
    from scipy.interpolate import griddata
    grid_x, grid_y = np.mgrid[-1.05:1.05:200j, -1.05:1.05:200j]
    try:
        grid_z = griddata((xi, yi), zi, (grid_x, grid_y), method='cubic')
    except Exception:
        grid_z = griddata((xi, yi), zi, (grid_x, grid_y), method='nearest')

    # Mask outside head circle
    outside = grid_x ** 2 + grid_y ** 2 > 1.0
    grid_z[outside] = np.nan

    vmax = np.nanmax(np.abs(grid_z)) or 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize, facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    ax.set_aspect('equal')
    ax.axis('off')

    ax.contourf(grid_x, grid_y, grid_z, levels=60,
                cmap='RdBu_r', norm=norm, zorder=1)

    # Channel dots
    ax.scatter(xi, yi, c='black', s=6, zorder=3, linewidths=0)

    _head_outline(ax, radius=1.0)

    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
        ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6, colors='#aaa')
    cbar.set_label('µV', fontsize=7, color='#aaa')

    ax.set_title(f'"{word_label}"', color='white', fontsize=9, pad=4)
    plt.tight_layout()
    return fig


def topomap_sequence(eeg: np.ndarray, mask: np.ndarray,
                     words: list[str],
                     samples_per_word: int = 50,
                     figsize_each=(3.0, 3.0)) -> list[plt.Figure]:
    """
    Generate one topomap figure per word.

    Args:
        words: list of word strings in the sentence
        samples_per_word: EEG samples per word window (≈50 ms at 500 Hz)
    """
    valid_len = int(mask.sum())
    n_words   = min(len(words), valid_len // samples_per_word)
    figs = []
    for i, word in enumerate(words[:n_words]):
        start = i * samples_per_word
        fig   = topomap_frame(eeg, mask, window_start=start,
                              window_len=samples_per_word,
                              word_label=word, figsize=figsize_each)
        figs.append(fig)
    return figs


def topomap_grid(eeg: np.ndarray, mask: np.ndarray,
                 words: list[str], samples_per_word: int = 50,
                 cols: int = 5, figsize_each=(2.8, 2.8)) -> plt.Figure:
    """
    All word topomap snapshots in a single grid figure.
    Convenient for the pre-computed (static) demo mode.
    """
    valid_len = int(mask.sum())
    n_words   = min(len(words), valid_len // samples_per_word, 15)  # cap at 15
    cols      = min(cols, n_words)
    rows      = (n_words + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(figsize_each[0] * cols, figsize_each[1] * rows),
                              facecolor='#0f0f1a')
    axes = np.array(axes).flatten()

    for i in range(len(axes)):
        ax = axes[i]
        ax.set_facecolor('#0f0f1a')
        if i >= n_words:
            ax.axis('off')
            continue

        word  = words[i]
        start = i * samples_per_word
        end   = min(start + samples_per_word, eeg.shape[0])
        frame = eeg[start:end].mean(axis=0)
        C     = min(len(frame), len(_CH_X))
        xi, yi, zi = _CH_X[:C], _CH_Y[:C], frame[:C]

        from scipy.interpolate import griddata
        gx, gy = np.mgrid[-1.05:1.05:100j, -1.05:1.05:100j]
        try:
            gz = griddata((xi, yi), zi, (gx, gy), method='cubic')
        except Exception:
            gz = griddata((xi, yi), zi, (gx, gy), method='nearest')
        gz[gx ** 2 + gy ** 2 > 1.0] = np.nan

        vmax = np.nanmax(np.abs(gz)) or 1.0
        ax.contourf(gx, gy, gz, levels=30, cmap='RdBu_r',
                    norm=Normalize(-vmax, vmax), zorder=1)
        ax.scatter(xi, yi, c='black', s=4, zorder=3, linewidths=0)
        _head_outline(ax, radius=1.0)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'"{word}"', color='white', fontsize=8, pad=2)

    fig.suptitle('Topographic Word Snapshots', color='white', fontsize=11, y=1.01)
    plt.tight_layout()
    return fig
