"""
GLIM EEG-to-Text Demo  (Gradio)

Two modes in one app:
  • Static  — instant, pre-computed results (no GPU needed at runtime)
  • Live    — real-time inference on your GPU with JSON disk-cache

Launch:
    cd demo
    pip install gradio scipy mne                   # one-time
    python app.py                                  # opens http://localhost:7860
    python app.py --share                          # + public Gradio link
"""
import sys, os, glob, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from visualise import butterfly_plot, topomap_grid
from inference import (discover_all_checkpoints, run_inference,
                       SENTIMENT_LABELS, RELATION_LABELS,
                       CORPUS_DISPLAY, PARADIGM_DISPLAY,
                       VARIANT_KEYS, compute_all_text_metrics)

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
DEMO_DF_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tmp',
                             'zuco_eeg_to_text_demo.df')
CACHE_DIR    = os.path.join(os.path.dirname(__file__), 'cache')
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# Load demo dataset once
# ─────────────────────────────────────────────────────────────────────────────
print("Loading demo dataset …")
demo_df   = pd.read_pickle(DEMO_DF_PATH)
CKPT_MAP  = discover_all_checkpoints()       # e.g. {'v1': '…/epoch=199.ckpt', 'v2': '…'}
print(f"  {len(demo_df)} samples  |  checkpoints: {list(CKPT_MAP.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Dropdown labels
# ─────────────────────────────────────────────────────────────────────────────
def sample_label(idx: int, row) -> str:
    task  = str(row.get('task', '?'))
    subj  = str(row.get('subject', '?'))
    bleu  = row.get('bleu1_raw', 0.0)
    sent  = str(row.get('sentiment label', 'nan'))[:3].upper()
    rel   = str(row.get('relation label', 'nan'))
    tag   = f"sent={sent}" if sent not in ('NAN', '') else f"rel={rel[:6]}"
    return f"#{idx+1:02d} — {subj} | {'NR' if task != 'task3' else 'TSR'} | BLEU={bleu:.2f} | {tag}"

SAMPLE_LABELS = [sample_label(i, demo_df.iloc[i]) for i in range(len(demo_df))]


# ─────────────────────────────────────────────────────────────────────────────
# Load static pre-computed cache (if exists) for each version
# ─────────────────────────────────────────────────────────────────────────────
def load_static_cache(version: str, n: int) -> list[dict | None]:
    """Return list of pre-computed result dicts (or None if not cached)."""
    cache = []
    for i in range(n):
        path = os.path.join(CACHE_DIR, f"{version}_sample_{i}.json")
        if os.path.exists(path):
            with open(path) as f:
                cache.append(json.load(f))
        else:
            cache.append(None)
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_generation_table(tm: dict) -> pd.DataFrame:
    """Turn text_metrics dict into a displayable DataFrame."""
    rows = []
    for n in [1, 2, 3, 4]:
        rows.append({'Metric': f'BLEU-{n}',
                     '@MTV (multi-target)': f"{tm.get(f'bleu{n}_mtv', 0):.4f}",
                     '@RAW (vs input)':     f"{tm.get(f'bleu{n}_raw', 0):.4f}"})
    for stat, label in [('fmeasure', 'F'), ('precision', 'P'), ('recall', 'R')]:
        rows.append({'Metric': f'ROUGE-1 {label}',
                     '@MTV (multi-target)': f"{tm.get(f'rouge1_{stat}_mtv', 0):.4f}",
                     '@RAW (vs input)':     f"{tm.get(f'rouge1_{stat}_raw', 0):.4f}"})
    rows.append({'Metric': 'WER',
                 '@MTV (multi-target)': '—',
                 '@RAW (vs input)':     f"{tm.get('wer', 0):.4f}"})
    return pd.DataFrame(rows)


def render_cls_table(result: dict) -> pd.DataFrame:
    """Turn classification results into a displayable DataFrame."""
    rows = []

    def top1(labels, probs):
        idx = int(np.argmax(probs))
        p   = probs[idx]
        return labels[idx], f"{p:.3f}"

    # Sentiment
    l, p = top1(result['sentiment_labels'], result['sentiment_probs'])
    prob_str = "  ".join(f"{lb}:{v:.2f}" for lb, v in
                          zip(result['sentiment_labels'], result['sentiment_probs']))
    rows.append({'Task': 'Sentiment (ACC-1)', 'Prediction': l,
                 'Confidence': p, 'All probs': prob_str})

    # Relation top-1 and top-3
    rel_probs  = result['relation_probs']
    rel_labels = result['relation_labels']
    top3_idx   = np.argsort(rel_probs)[::-1][:3]
    top3_str   = ", ".join(f"{rel_labels[i]}({rel_probs[i]:.2f})" for i in top3_idx)
    l, p = top1(rel_labels, rel_probs)
    rows.append({'Task': 'Relation (ACC-1)', 'Prediction': l,
                 'Confidence': p, 'All probs': top3_str})
    rows.append({'Task': 'Relation (top-3)', 'Prediction': top3_str,
                 'Confidence': '—', 'All probs': '—'})

    # Corpus
    l, p = top1(result['corpus_labels'], result['corpus_probs'])
    prob_str = "  ".join(f"{lb}:{v:.2f}" for lb, v in
                          zip(result['corpus_labels'], result['corpus_probs']))
    rows.append({'Task': 'Corpus (ACC-1)', 'Prediction': l,
                 'Confidence': p, 'All probs': prob_str})

    # Reading paradigm (zero-shot, task-blind)
    l, p = top1(result['paradigm_labels'], result['paradigm_probs'])
    prob_str = "  ".join(f"{lb}:{v:.2f}" for lb, v in
                          zip(result['paradigm_labels'], result['paradigm_probs']))
    rows.append({'Task': 'Reading Paradigm NR/TSR (zero-shot)', 'Prediction': l,
                 'Confidence': p, 'All probs': prob_str})

    return pd.DataFrame(rows)


def eeg_numpy(row) -> tuple[np.ndarray, np.ndarray]:
    eeg = np.array(row['eeg'], dtype=np.float32)
    msk = np.array(row['mask'], dtype=np.uint8)
    if eeg.shape[0] == 128:   # (C,T) → (T,C)
        eeg = eeg.T
    return eeg, msk

def compute_pseudo_snr_db(eeg: np.ndarray, mask: np.ndarray) -> float:
    """Compute Peak-to-RMS ratio (Crest Factor) as a proxy for SNR in dB."""
    valid_len = int(mask.sum())
    if valid_len == 0: return 0.0
    valid_eeg = eeg[:valid_len]
    peak = np.max(np.abs(valid_eeg))
    rms = np.sqrt(np.mean(valid_eeg**2))
    return 20 * np.log10(peak / rms) if rms > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Core callback
# ─────────────────────────────────────────────────────────────────────────────

def on_select(sample_choice: str, version: str, mode: str):
    """Called every time a user picks a sample or changes settings."""
    idx = SAMPLE_LABELS.index(sample_choice)
    row = demo_df.iloc[idx]

    eeg, mask = eeg_numpy(row)
    words     = str(row.get('input text', '')).split()
    task_gt   = 'NR' if str(row.get('task', 'task1')) != 'task3' else 'TSR'

    # ── EEG plots ──
    fig_butterfly = butterfly_plot(
        eeg, mask,
        title=f"Butterfly Plot — {row.get('subject', '?')} | {task_gt}")
    fig_topo = topomap_grid(eeg, mask, words, samples_per_word=50, cols=5)

    # ── Text & Stats ──
    raw_input = str(row.get('input text', ''))
    snr_db    = compute_pseudo_snr_db(eeg, mask)
    gt_text   = f"Task: {task_gt} | Subject: {row.get('subject','?')} | Dataset: {row.get('dataset','?')} | SNR (Peak/RMS): {snr_db:.1f} dB"

    # ── Get result (static or live) ──
    if mode == '⚡ Live Inference' and version in CKPT_MAP:
        result = run_inference(idx, row, CKPT_MAP[version], version, DEVICE)
        gen_text = result['gen_text']
        status   = f"✅ {'From cache' if result.get('_from_cache') else 'Fresh inference'} — {version}"
    else:
        # Static: use pre-stored gen text + pre-computed text metrics
        gen_text = str(row.get('gen text', '(Not precomputed — switch to Live)'))
        # Try to load per-sample cache written by create_demo_df.py
        cache_path = os.path.join(CACHE_DIR, f"{version}_sample_{idx}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                result = json.load(f)
            status = f"📂 Pre-computed — {version}"
        else:
            # Fallback: compute text metrics offline (no model needed)
            variants = [str(row.get(k, '')) for k in VARIANT_KEYS]
            tm = (row['text_metrics']
                  if 'text_metrics' in row and isinstance(row['text_metrics'], dict)
                  else compute_all_text_metrics(gen_text, raw_input, variants))
            result = {
                'gen_text':         gen_text,
                'text_metrics':     tm,
                'sentiment_probs':  [0.33, 0.33, 0.34],  # placeholder
                'sentiment_labels': SENTIMENT_LABELS,
                'relation_probs':   [1/9]*9,
                'relation_labels':  RELATION_LABELS,
                'corpus_probs':     [0.5, 0.5],
                'corpus_labels':    CORPUS_DISPLAY,
                'paradigm_probs':   [0.5, 0.5],
                'paradigm_labels':  PARADIGM_DISPLAY,
            }
            status = "⚠️ Text metrics only (no model cache — run create_demo_df.py --checkpoint v1 v2)"

    gen_metrics_df = render_generation_table(result['text_metrics'])
    cls_df         = render_cls_table(result)

    return (fig_butterfly, fig_topo,
            gt_text, raw_input, gen_text,
            gen_metrics_df, cls_df,
            status)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #0d0d1a !important; color: #e0e0f0 !important; }
.gr-button-primary      { background: linear-gradient(135deg,#6e40c9,#3b82f6) !important; }
h1, h2, h3             { color: #a0c4ff !important; }
.gr-tab-label.selected  { color: #6e40c9 !important; border-bottom: 2px solid #6e40c9 !important; }
"""

def build_ui():
    with gr.Blocks(css=CSS, title="LEXI EEG-to-Text Demo") as demo:

        gr.Markdown("""
# 🧠 LEXI - EEG-to-Text Generation
**EEG→Text generation · Zero-shot classification · Topographic visualisation**
""")

        with gr.Row():
            sample_dd  = gr.Dropdown(SAMPLE_LABELS, label="📋 Select Sample",
                                     value=SAMPLE_LABELS[0], scale=4)
            version_dd = gr.Dropdown(list(CKPT_MAP.keys()) or ['v1', 'v2'],
                                     label="🏷️ Checkpoint", value=list(CKPT_MAP.keys())[0]
                                     if CKPT_MAP else 'v1', scale=1)
            mode_dd    = gr.Dropdown(['📂 Static (pre-computed)', '⚡ Live Inference'],
                                     label="⚙️ Mode", value='📂 Static (pre-computed)', scale=1)

        status_box = gr.Textbox(label="Status", interactive=False, max_lines=1, visible=False)

        # ── Row 1: EEG Plots ──
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("🦋 Butterfly Plot"):
                        butterfly_out = gr.Plot(label="All 128 channels (blue=left, red=right)")
                    with gr.Tab("🗺️ Topographic Word Snapshots"):
                        topo_out = gr.Plot(label="Spatial amplitude per word window")

        # ── Row 2: Text Information ──
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Text")
                meta_box  = gr.Textbox(label="Sample Info",   interactive=False, lines=1)
            with gr.Column():
                gr.Markdown("### ")
                input_box = gr.Textbox(label="✏️ Input Text (what subject read)",
                                       interactive=False, lines=3)
            with gr.Column():
                gr.Markdown("### ")
                gen_box   = gr.Textbox(label="🤖 Generated Text (from EEG)",
                                       interactive=False, lines=3)

        # ── Row 2: Metrics ──
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Generation Metrics (BLEU · ROUGE · WER)")
                gen_table = gr.DataFrame(
                    headers=['Metric', '@MTV (multi-target)', '@RAW (vs input)'],
                    interactive=False)

            with gr.Column():
                gr.Markdown("### 🎯 Classification Metrics")
                cls_table = gr.DataFrame(
                    headers=['Task', 'Prediction', 'Confidence', 'All probs'],
                    interactive=False)

        # ── Trigger ──
        inputs  = [sample_dd, version_dd, mode_dd]
        outputs = [butterfly_out, topo_out,
                   meta_box, input_box, gen_box,
                   gen_table, cls_table, status_box]

        for trigger in [sample_dd, version_dd, mode_dd]:
            trigger.change(fn=on_select, inputs=inputs, outputs=outputs)

        # Load first sample on startup
        demo.load(fn=on_select,
                  inputs=[gr.Textbox(value=SAMPLE_LABELS[0], visible=False),
                          gr.Textbox(value=list(CKPT_MAP.keys())[0] if CKPT_MAP else 'v1', visible=False),
                          gr.Textbox(value='📂 Static (pre-computed)', visible=False)],
                  outputs=outputs)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true',
                        help='Create public Gradio share link')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(share=args.share, server_port=args.port,
              server_name='0.0.0.0', show_error=True)
