"""
EEG Visualisation helpers for the GLIM demo.

Plots:
    1. Butterfly plot             — all 128 channels overlaid as time-series waveforms
    2. EEG feature space          — word-window embeddings projected with t-SNE/PCA
    3. Time-frequency spectrograms — raw vs attention-weighted EEG summaries
    4. Attention heatmaps         — model cross-attention when available, saliency otherwise
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


def _valid_eeg(eeg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid_len = int(mask.sum())
    return eeg[:valid_len] if valid_len > 0 else eeg[:1]


def _normalise_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def _word_windows(eeg: np.ndarray, words: list[str], samples_per_word: int) -> list[tuple[str, np.ndarray]]:
    windows = []
    for idx, word in enumerate(words):
        start = idx * samples_per_word
        end = min(start + samples_per_word, eeg.shape[0])
        if end <= start:
            break
        windows.append((word, eeg[start:end]))
    return windows


def eeg_feature_space_plot(eeg: np.ndarray, mask: np.ndarray, words: list[str],
                           samples_per_word: int = 50,
                           title: str = "EEG Feature Space"):
    """Project word windows into 2D using t-SNE when possible, otherwise PCA."""
    import matplotlib.pyplot as plt

    valid_eeg = _valid_eeg(eeg, mask)
    windows = _word_windows(valid_eeg, words, samples_per_word)

    if not windows:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0f0f1a')
        ax.set_facecolor('#0f0f1a')
        ax.text(0.5, 0.5, 'No word windows available', ha='center', va='center', color='white')
        ax.axis('off')
        return fig

    features = []
    labels = []
    for idx, (word, window) in enumerate(windows):
        # Concatenate mean, std, and a coarse energy summary for a compact descriptor.
        mean_feat = window.mean(axis=0)
        std_feat = window.std(axis=0)
        energy_feat = np.array([
            float(np.mean(np.abs(window))),
            float(np.sqrt(np.mean(window ** 2)))
        ], dtype=np.float32)
        features.append(np.concatenate([mean_feat, std_feat, energy_feat], axis=0))
        labels.append(f"{idx + 1}: {word}")

    features = np.asarray(features, dtype=np.float32)
    n_samples = features.shape[0]
    if n_samples == 1:
        coords = np.zeros((1, 2), dtype=np.float32)
        method_name = 'single-point'
    else:
        coords = None
        method_name = 'PCA'
        try:
            from sklearn.decomposition import PCA
            if n_samples >= 3:
                coords = PCA(n_components=2).fit_transform(features)
                method_name = 'PCA'
        except Exception:
            coords = None

        if coords is None:
            try:
                from sklearn.manifold import TSNE
                perplexity = max(1, min(5, n_samples - 1))
                coords = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto').fit_transform(features)
                method_name = 't-SNE'
            except Exception:
                centered = features - features.mean(axis=0, keepdims=True)
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                coords = centered @ vh[:2].T
                method_name = 'PCA-fallback'

    palette = plt.cm.viridis(np.linspace(0.1, 0.9, n_samples))
    fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    for idx, (x_coord, y_coord) in enumerate(coords):
        ax.scatter(x_coord, y_coord, s=90, color=palette[idx], edgecolor='white', linewidth=0.7)
        ax.text(x_coord + 0.02, y_coord + 0.02, labels[idx], fontsize=8, color='white')

    ax.set_title(f"{title} ({method_name})", color='white', fontsize=12)
    ax.set_xlabel('Component 1', color='#ddd')
    ax.set_ylabel('Component 2', color='#ddd')
    ax.grid(True, color='#334', alpha=0.25)
    ax.tick_params(colors='#ddd')
    plt.tight_layout()
    return fig


def eeg_spectrogram_plot(eeg: np.ndarray, mask: np.ndarray,
                         attention_profile: np.ndarray | None = None,
                         fs: int = 128,
                         title: str = "Time-Frequency Spectrograms"):
    """Show raw and attention-weighted spectrograms for the EEG signal."""
    from scipy.signal import spectrogram
    import matplotlib.pyplot as plt

    valid_eeg = _valid_eeg(eeg, mask)
    signal = valid_eeg.mean(axis=1)

    freqs, times, spec = spectrogram(signal, fs=fs, nperseg=min(64, max(16, len(signal) // 4)),
                                     noverlap=min(48, max(8, len(signal) // 8)), scaling='density', mode='magnitude')
    spec_db = 10 * np.log10(spec + 1e-8)

    if attention_profile is None or len(attention_profile) == 0:
        attention_profile = _normalise_vector(np.abs(signal))
    else:
        attention_profile = _normalise_vector(attention_profile)

    if len(times) > 1:
        attn_resampled = np.interp(times, np.linspace(times.min(), times.max(), len(attention_profile)), attention_profile)
    else:
        attn_resampled = np.array([float(np.mean(attention_profile))], dtype=np.float32)

    weighted_spec = spec_db * (0.55 + 0.45 * attn_resampled[np.newaxis, :])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), facecolor='#0f0f1a', sharey=True)
    for ax in axes:
        ax.set_facecolor('#0f0f1a')

    im0 = axes[0].pcolormesh(times, freqs, spec_db, shading='auto', cmap='magma')
    axes[0].set_title('Input EEG spectrogram', color='white')
    axes[0].set_xlabel('Time (s)', color='#ddd')
    axes[0].set_ylabel('Frequency (Hz)', color='#ddd')

    ax0b = axes[0].twinx()
    ax0b.plot(times, attn_resampled, color='#7dd3fc', linewidth=1.5, alpha=0.95)
    ax0b.set_ylabel('Attention / saliency', color='#7dd3fc')
    ax0b.set_ylim(0, 1)
    ax0b.tick_params(colors='#7dd3fc')

    im1 = axes[1].pcolormesh(times, freqs, weighted_spec, shading='auto', cmap='magma')
    axes[1].set_title('Attention-weighted regions', color='white')
    axes[1].set_xlabel('Time (s)', color='#ddd')

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label('Power (dB)', color='#ddd')
    cbar.ax.tick_params(colors='#ddd')

    fig.suptitle(title, color='white', fontsize=12)
    for ax in axes:
        ax.tick_params(colors='#ddd')
    plt.tight_layout()
    return fig


def attention_heatmap(attention_matrix: np.ndarray | None,
                      query_labels: list[str] | None = None,
                      key_labels: list[str] | None = None,
                      title: str = "Attention Heatmap"):
    """Render attention weights if available, otherwise show a saliency proxy."""
    import matplotlib.pyplot as plt

    if attention_matrix is None:
        fig, ax = plt.subplots(figsize=(8, 2.6), facecolor='#0f0f1a')
        ax.set_facecolor('#0f0f1a')
        ax.text(0.5, 0.5, 'No attention matrix available', ha='center', va='center', color='white')
        ax.axis('off')
        return fig

    weights = np.asarray(attention_matrix, dtype=np.float32)
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if weights.size == 0:
        fig, ax = plt.subplots(figsize=(8, 2.6), facecolor='#0f0f1a')
        ax.set_facecolor('#0f0f1a')
        ax.text(0.5, 0.5, 'Empty attention matrix', ha='center', va='center', color='white')
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    im = ax.imshow(weights, aspect='auto', cmap='Blues', interpolation='nearest')
    ax.set_xlabel('EEG time index', color='#ddd')
    ax.set_ylabel('Query position', color='#ddd')
    ax.set_title(title, color='white')
    if key_labels and len(key_labels) <= 24:
        ax.set_xticks(np.linspace(0, weights.shape[1] - 1, len(key_labels)).astype(int))
        ax.set_xticklabels(key_labels, rotation=45, ha='right', fontsize=8, color='#ddd')
    if query_labels and len(query_labels) <= 24:
        ax.set_yticks(np.linspace(0, weights.shape[0] - 1, len(query_labels)).astype(int))
        ax.set_yticklabels(query_labels, fontsize=8, color='#ddd')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention weight', color='#ddd')
    cbar.ax.tick_params(colors='#ddd')
    ax.tick_params(colors='#ddd')
    plt.tight_layout()
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
