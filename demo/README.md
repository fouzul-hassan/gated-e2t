# GLIM Demo

## Files

| File | Purpose |
|---|---|
| `create_demo_df.py` | Select best 5–10 test samples, join EEG, compute metrics |
| `visualise.py` | Butterfly plot + topographic word-snapshot figures |
| `inference.py` | Auto-discover checkpoints, load model, disk-cache results |
| `app.py` | Gradio web app (static + live modes) |
| `cache/` | JSON cache for live inference (auto-created) |

---

## Setup (one-time)

```bash
pip install gradio scipy mne joblib
```

---

## Step 1 — Create the demo dataset

```bash
cd demo

# Text metrics only (no GPU needed):
python create_demo_df.py

# Text + classification + ETES metrics pre-computed for v1 and v2:
python create_demo_df.py --checkpoint v1 v2
```

This will:
- Select 10 high-BLEU samples from `data/tmp/glim_gen_results.pkl`
- Join with `zuco_eeg_label_8variants.df` to get EEG arrays
- Pre-compute all BLEU/ROUGE/WER metrics
- Pre-compute ETES alignment scores when a checkpoint is provided
- Cache attention summaries for the demo heatmap when a checkpoint is provided
- Save `data/tmp/zuco_eeg_to_text_demo.df`
- (If `--checkpoint` given) Run GLIM inference and cache results in `demo/cache/`

---

## Step 2 — Launch the demo

```bash
# Local browser (http://localhost:7860):
python app.py

# With public share link:
python app.py --share

# Custom port:
python app.py --port 8080
```

---

## Demo Modes

| Mode | Speed | GPU needed? |
|---|---|---|
| 📂 Static (pre-computed) | Instant | ❌ |
| ⚡ Live Inference | ~3s first time, cached after | ✅ RTX 4050 OK |

**Caching**: live inference results auto-saved to `demo/cache/v1_sample_N.json`.
Subsequent clicks on the same sample are instant (< 0.1s).

---

## Checkpoints Auto-Discovery

The app scans `./runs/v1/*.ckpt`, `./runs/v2/*.ckpt`, etc. and picks the most recently modified file. No manual path needed.

---

## Demo Validation Script

Run the standalone validation script before submission to check individual FR and NFR cases. Each run writes organized JSON and Markdown logs under `demo/test/logs/`.

```bash
cd demo
python prototype_test.py --fr1
python prototype_test.py --fr2
python prototype_test.py --fr3
python prototype_test.py --fr4
python prototype_test.py --fr5
python prototype_test.py --nfr1
python prototype_test.py --nfr2
python prototype_test.py --nfr3
python prototype_test.py --nfr4

# Run everything in one pass:
python prototype_test.py --all

# Optional checkpoint selection for live smoke checks:
python prototype_test.py --fr2 --version v1
```

The script validates:
- FR1: selectable EEG-to-text sample loading
- FR2: static and live mode switching
- FR3: single-visualization layout with butterfly default
- FR4: generated text, input text, generation metrics, ETES metrics, and classification metrics
- FR5: switching among butterfly, feature space, and spectrogram views
- NFR1: responsive cached loading in static mode
- NFR2: queued live inference with progress updates
- NFR3: GPU live inference plus CPU static usability
- NFR4: readable single-panel browser layout

---

## Metrics Shown

**Generation** (per sample, 2 columns: @MTV / @RAW):
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- ROUGE-1 F, ROUGE-1 Precision, ROUGE-1 Recall
- WER

**EEG-aware alignment**:
- ETES alignment, ETES total, ETES reference, ETES gap

**EEG visual diagnostics**:
- EEG Feature Space Visualization
- Time-Frequency Spectrograms
- Attention Heatmap

**Classification** (zero-shot CLIP-like):
- Sentiment (neg/neu/pos) — top-1 + probability bar
- Relation (9 types) — top-1 + top-3 + probabilities
- Corpus (movie/biography) — top-1
- Reading Paradigm NR/TSR — zero-shot, task-blind (`<UNK>` prompt)
