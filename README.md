# GRAPE-GLIM: Gated Representation EEG-to-Language Interface Model

This repository implements **GRAPE-GLIM**, an MSc research pipeline for decoding natural language from raw EEG signals recorded during natural reading. The system extends the original GLIM (Grounded Language-Image Model) baseline with two primary contributions: a **Signal-JEPA self-supervised pretraining stage** for the EEG encoder, and **gated cross-attention** within the transformer backbone.

The pipeline consists of three sequential stages:

1. Self-supervised pretraining of the EEG encoder using a JEPA-style masked prediction objective (Signal-JEPA)
2. Linear probing of the frozen pretrained encoder to verify representation quality
3. End-to-end fine-tuning of the full GLIM model on the EEG-to-text task, optionally initialised from the pretrained encoder

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Stage 1: Signal-JEPA Pretraining](#stage-1-signal-jepa-pretraining)
  - [Architecture](#pretraining-architecture)
  - [Running the Pretrainer](#running-the-pretrainer)
  - [Pretraining Configuration Reference](#pretraining-configuration-reference)
  - [Monitoring and Checkpoints](#monitoring-and-checkpoints)
- [Stage 2: Linear Probe Evaluation](#stage-2-linear-probe-evaluation)
  - [Single-Task Probe (Subject ID)](#single-task-probe)
  - [Multi-Task Probe](#multi-task-probe)
  - [Inspecting the Pretrained Checkpoint](#inspecting-the-pretrained-checkpoint)
- [Stage 3: Fine-Tuning GLIM](#stage-3-fine-tuning-glim)
  - [Option A: Training from Scratch (Baseline)](#option-a-training-from-scratch-baseline)
  - [Option B: Fine-Tuning with JEPA Encoder (GRAPE-GLIM)](#option-b-fine-tuning-with-jepa-encoder-grape-glim)
  - [Option C: Gated Attention with Nucleus Sampling](#option-c-gated-attention-with-nucleus-sampling)
  - [Option D: BART Language Model Backbone](#option-d-bart-language-model-backbone)
  - [Option E: Small Flan-T5 Variant](#option-e-small-flan-t5-variant)
  - [Fine-Tuning Configuration Reference](#fine-tuning-configuration-reference)
  - [Multi-GPU Distributed Training](#multi-gpu-distributed-training)
- [Model Architecture](#model-architecture)
  - [EEG Encoder (GLIMEncoderPretrainer / EEGEncoder)](#eeg-encoder)
  - [Prompt Embedder](#prompt-embedder)
  - [Cross-Modal Aligner](#cross-modal-aligner)
  - [Gated Attention](#gated-attention)
  - [Language Model Decoder](#language-model-decoder)
- [Evaluation and Testing](#evaluation-and-testing)
- [Gradio Demo](#gradio-demo)
- [Experiment Scripts Reference](#experiment-scripts-reference)
- [Runs and Checkpoint Layout](#runs-and-checkpoint-layout)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

**Task.** Given a segment of 128-channel EEG recorded while a participant silently reads a sentence, decode the semantic content of that sentence as natural language text.

**Dataset.** ZuCo 1 and ZuCo 2 (Hollenstein et al., 2018 / 2019). Recordings from up to 30 subjects reading sentences under two paradigms: Normal Reading (NR) and Task-Specific Reading (TSR). The processed dataset contains approximately 17,000 EEG-text pairs split across train, validation, and test phases.

**Key design decisions:**

| Design choice | This work | Original GLIM |
|---|---|---|
| EEG encoder initialisation | Signal-JEPA pretrained weights | Random initialisation |
| Attention mechanism | Optional gated attention (elementwise or headwise sigmoid gate) | Standard scaled dot-product attention |
| Text decoding | Beam search, nucleus sampling, or greedy | Beam search only |
| Language model | Frozen Flan-T5-Large (or Flan-T5-Small / BART-Large-CNN) | Frozen Flan-T5-Large |
| Alignment loss | Symmetric CLIP + optional commitment loss | Symmetric CLIP |

---

## Repository Layout

```
GLIM/
|-- model/
|   |-- glim.py                     # Full GRAPE-GLIM LightningModule
|   |-- modules.py                  # EEGEncoder, Aligner, PromptEmbedder, EncoderBlock
|   |-- energy.py                   # ETESEvaluator, EnergyContrastiveLoss, EnergyGuidedGenerator
|   `-- xai_logging.py              # WandB XAI attention logging helpers
|
|-- pretraining/
|   |-- pretrain_glim_encoder.py    # GLIMEncoderPretrainer model definition
|   |-- run_pretrain.py             # Training script for Signal-JEPA pretraining
|   |-- main.py                     # EEG2Rep-style CLI pretraining (alternative)
|   |-- load_pretrained.py          # Weight transfer: pretrained encoder -> GLIM
|   |-- evaluate_pretrained.py      # Single-task linear probe + t-SNE visualisation
|   |-- evaluate_multitask_probe.py # Multi-task probe (subject, sentiment, relation, paradigm)
|   |-- evaluate_benchmarking_probe.py     # Benchmarking probe pipeline
|   |-- evaluate_benchmarking_probe_svm.py # SVM-based benchmarking probes
|   |-- Dataset/                    # ZuCo memory-mapped dataset loaders for pretraining
|   |-- Models/                     # Architecture modules used by EEG2Rep-style runner
|   `-- Results/                    # Saved pretrain checkpoints and probe outputs
|
|-- data/
|   |-- datamodule.py               # GLIMDataModule, ZuCoDataset, GLIMSampler
|   |-- __STEP1_text_extract_revise.ipynb   # Raw text extraction from ZuCo XML
|   |-- __STEP2_text_variants_gen.ipynb     # LLM-generated 8-variant target texts
|   |-- __STEP3_eeg_preproc.ipynb           # EEG preprocessing (bandpass, baseline, epoch)
|   |-- __STEP4_eeg_text_match_split.ipynb  # Align EEG epochs to text, train/val/test split
|   `-- tmp/                        # Processed pickle files (not tracked by git)
|
|-- experiments/
|   |-- train_gated.py              # Gated attention + nucleus sampling example
|   |-- train_with_jepa_encoder-small.py  # JEPA encoder + Flan-T5-Small
|   |-- train_with_jepa_encoder_bart.py   # JEPA encoder + BART backbone
|   |-- train_cli.py                # CLI-driven training (argparse)
|   |-- train_cls.py                # Classification-only training variant
|   |-- train_energy.py             # Energy-based contrastive loss variant
|   |-- train_bart.py               # BART stand-alone training
|   |-- sweep_train.py              # WandB sweep for hyperparameter search
|   |-- sweep_eval.py               # WandB sweep evaluation runner
|   `-- run_eval.py                 # Standalone evaluation from a checkpoint
|
|-- train.py                        # Main from-scratch fine-tuning script (multi-GPU)
|-- train_with_jepa_encoder.py      # Fine-tuning with JEPA encoder initialisation
|-- train_with_jepa_encoder2.py     # Alternative JEPA fine-tuning variant
|
|-- demo/
|   |-- app.py                      # Gradio interactive demo
|   |-- inference.py                # Live inference helpers (model loading, generation)
|   |-- visualise.py                # EEG visualisation (butterfly, spectrogram, feature space)
|   |-- create_demo_df.py           # Build demo dataframe with pre-computed results
|   `-- cache/                      # Per-sample JSON inference cache
|
|-- runs/
|   |-- v1/                         # GRAPE-GLIM v1 checkpoint (epoch=199)
|   `-- v2/                         # GRAPE-GLIM v2 checkpoint (epoch=199, noise-augmented)
|
|-- evaluations/                    # Saved per-run evaluation outputs
|-- results/                        # Final aggregated result dataframes
|-- thesis/                         # Thesis figures and export scripts
|-- requirements.txt
`-- environment.yml
```

---

## Environment Setup

**Python version:** 3.9 or later (tested on 3.10 and 3.13).

**Recommended:** Create a dedicated conda environment.

```bash
conda create -n grape-glim python=3.10 -y
conda activate grape-glim
```

**Install PyTorch with CUDA.** Check https://pytorch.org for the correct command for your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

The core dependencies are:

| Package | Minimum version | Purpose |
|---|---|---|
| torch | 2.8.0 | Deep learning framework |
| lightning | 2.4.0 | Training loop, DDP, checkpointing |
| torchmetrics | 1.3.1 | BLEU, ROUGE, WER, retrieval metrics |
| transformers | 4.45.0 | Flan-T5, BART, tokenisers |
| wandb | 0.12.10 | Experiment tracking |
| numpy | 1.26.4 | Array operations |
| pandas | 2.2.x | Dataframe-based dataset |
| scipy | any | Signal processing for EEG |
| scikit-learn | any | Linear probes (LogisticRegression) |
| gradio | any | Interactive demo |
| tensorboard | any | Pretraining loss curves |

**Pretraining-specific extras** (only required for Stage 1):

```bash
pip install -r pretraining/requirements.txt
```

---

## Data Preparation

The processed dataset is stored as pandas pickle files under `data/tmp/`. These files are not tracked by git due to their size. Run the four numbered notebooks in order if starting from raw ZuCo `.mat` files:

```
data/__STEP1_text_extract_revise.ipynb        # Extract and clean raw sentences from ZuCo XML/mat
data/__STEP2_text_variants_gen.ipynb          # Generate 8 paraphrase variants per sentence via LLM
data/__STEP3_eeg_preproc.ipynb                # Bandpass filter, epoch alignment, artefact rejection
data/__STEP4_eeg_text_match_split.ipynb       # Match EEG segments to sentences, train/val/test split
```

The final outputs expected by the training scripts are:

```
data/tmp/zuco_eeg_label_8variants.df          # Training dataset: EEG + text + 8-variant labels
data/tmp/zuco_eeg_128ch_1280len.df            # Pretraining dataset: EEG + subject labels only
data/tmp/zuco_eeg_label_benchmarking.df       # Benchmarking dataset for probe evaluation
data/tmp/zuco_eeg_to_text_demo.df             # 10-sample demo subset
```

Each row in `zuco_eeg_label_8variants.df` contains:

| Column | Type | Shape | Description |
|---|---|---|---|
| eeg | np.ndarray | (1280, 128) | EEG signal, 1280 timesteps x 128 channels |
| mask | np.ndarray | (1280,) | Binary mask: 1 for valid timesteps, 0 for padding |
| input text | str | - | "To English: " + raw sentence |
| target text | str | - | One of 8 paraphrases (rotated per epoch during training) |
| lexical simplification (v0/v1) | str | - | Paraphrase variants generated by LLM |
| semantic clarity (v0/v1) | str | - | Semantic rewrite variants |
| syntax simplification (v0/v1) | str | - | Syntactic rewrite variants |
| naive rewritten | str | - | Generic rewrite |
| naive simplified | str | - | Simplified version |
| subject | str | - | Participant ID (e.g. "ZAB") |
| dataset | str | - | "ZuCo1" or "ZuCo2" |
| task | str | - | "task1", "task2", or "task3" |
| sentiment label | str | - | "positive", "negative", "neutral", or "nan" |
| relation label | str | - | Relation type (e.g. "employment") or "nan" |
| text uid | int | - | Unique sentence identifier used by GLIMSampler |
| phase | str | - | "train", "val", or "test" |

GLIMSampler ensures every batch contains samples from distinct text UIDs, which is required for the symmetric CLIP contrastive loss to produce meaningful negatives.

---

## Stage 1: Signal-JEPA Pretraining

### Pretraining Architecture

The pretrainer (`pretraining/pretrain_glim_encoder.py`) wraps GLIM's `EncoderBlock` modules in a JEPA-style masked prediction framework borrowed from EEG2Rep.

**Core components:**

- **Patch embedding.** The EEG input of shape (B, 1280, 128) is divided into non-overlapping patches of size 8 along the time dimension, giving 160 patches of flattened size 8 x 128 = 1024. These are projected to `emb_size` (default 128) via a linear layer followed by LayerNorm and GELU.

- **Positional encoding.** Fixed 1D sinusoidal positional embeddings are added to the patches. These are not learned and not updated during training.

- **Context encoder.** Six `EncoderBlock` modules from `model/modules.py` operating on the visible (unmasked) patches only. Prompt injection, temporal modulation, and causal masking are all disabled during pretraining because the objective is purely self-supervised.

- **Target encoder.** An exponential moving average (EMA) copy of the context encoder. Its parameters are never directly differentiated; they are updated via `momentum_update()` after each optimiser step with momentum coefficient 0.99 by default. The target encoder sees the full (unmasked) sequence.

- **Semantic Subsequence Preserving (SSP) masking.** At each forward pass, two contiguous chunks of patches are chosen as the visible set (approximately 50% of patches). The complementary patches form the masked set. SSP masking preserves temporal semantics because the visible context is always a contiguous subsequence of the original signal rather than random scattered positions.

- **Cross-attention predictor.** Two cross-attention blocks attend from learnable mask tokens (with positional encoding for the masked positions) to the context encoder output. This bridges the masked positions to the visible context representations without directly exposing the target encoder output to the context encoder.

- **VICReg loss.** The loss function has three components operating on the predicted representations for masked positions:
  - Alignment: MSE between predicted and target (EMA) representations.
  - Variance: Penalises dimensions whose standard deviation falls below 1 to prevent representational collapse.
  - Covariance: Penalises off-diagonal covariance to encourage feature decorrelation.

**Why self-supervised pretraining helps.** Standard EEG-to-text fine-tuning trains on relatively few paired examples (approximately 17,000 sentence-level recordings). The pretrained encoder is exposed to the broader distribution of EEG signals across all subjects and corpora without requiring text labels, learning to produce temporally coherent representations that capture subject-invariant neural patterns.

### Running the Pretrainer

All pretraining commands are run from the project root unless stated otherwise.

**Basic run (recommended starting point):**

```bash
cd pretraining
python run_pretrain.py \
    --data_path ../data/tmp/zuco_eeg_128ch_1280len.df \
    --output_dir Results/GLIM_Pretrain1 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --n_blocks 6 \
    --emb_size 128 \
    --num_heads 8 \
    --patch_size 8 \
    --mask_ratio 0.5 \
    --momentum 0.99 \
    --gpu 0
```

**With gated attention in the pretrainer:**

```bash
cd pretraining
python run_pretrain.py \
    --data_path ../data/tmp/zuco_eeg_128ch_1280len.df \
    --output_dir Results/GLIM_Pretrain_Gated \
    --epochs 100 \
    --use_gated_attention \
    --gpu 0
```

**Using the EEG2Rep-style CLI runner** (operates on separate benchmark datasets such as Crowdsource or DREAMER):

```bash
cd pretraining
python main.py \
    --data_dir Dataset/Crowdsource \
    --Training_mode Rep-Learning \
    --epochs 200 \
    --batch_size 128 \
    --lr 1e-3 \
    --emb_size 16 \
    --layers 4 \
    --num_heads 8 \
    --patch_size 8 \
    --mask_ratio 0.5 \
    --momentum 0.99 \
    --gpu 0
```

This runner uses `running.py` which calls either `Rep_Learning` or `Supervised` training modes. It is suitable for in-domain or cross-domain pretraining on third-party EEG datasets.

### Pretraining Configuration Reference

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `../data/tmp/zuco_eeg_128ch_1280len.df` | Path to the EEG-only pretraining dataset |
| `--output_dir` | `Results/GLIM_Pretrain` | Directory for checkpoints and TensorBoard logs |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 64 | Batch size. Larger batches stabilise VICReg variance/covariance terms |
| `--lr` | 1e-4 | Adam-W learning rate |
| `--weight_decay` | 0.01 | L2 regularisation |
| `--n_blocks` | 6 | Number of EncoderBlocks in context and target encoders. Must match the EEG encoder in the downstream GLIM model |
| `--emb_size` | 128 | Internal embedding dimension. Must match `hidden_dim` in GLIM |
| `--num_heads` | 8 | Number of attention heads per block |
| `--patch_size` | 8 | Time-axis patch size. 1280 / 8 = 160 patches |
| `--mask_ratio` | 0.5 | Fraction of patches to mask (approximately) |
| `--momentum` | 0.99 | EMA coefficient for target encoder update |
| `--use_gated_attention` | False | Enable sigmoid gates in the pretrained encoder blocks |
| `--linear_probe_interval` | 5 | Run subject-ID linear probe every N epochs to monitor representation quality |
| `--gpu` | 0 | GPU index. Use -1 for CPU |
| `--seed` | 42 | Random seed for data split and weight initialisation |

### Monitoring and Checkpoints

TensorBoard logs are saved to `Results/<output_dir>/tensorboard/`. Launch with:

```bash
tensorboard --logdir pretraining/Results/GLIM_Pretrain1/tensorboard
```

The following scalars are tracked:
- `loss/total`, `loss/align`, `loss/std`, `loss/cov` per epoch
- `accuracy/linear_probe` every `--linear_probe_interval` epochs (logistic regression on subject-ID classification using frozen encoder features)

Two checkpoint files are saved:

| File | Description |
|---|---|
| `Results/<dir>/best_model.pth` | Best model by linear probe accuracy. Used for weight transfer |
| `Results/<dir>/final_model.pth` | Model at the end of training regardless of probe accuracy |

Each checkpoint contains:
- `model_state_dict`: Full pretrainer including EMA target encoder and predictor
- `encoder_state_dict`: Extracted context encoder weights pre-formatted with `in_blocks.i.*` key naming for direct loading into `glim.eeg_encoder`
- `optimizer_state_dict`
- `epoch`, `accuracy`, `args`

---

## Stage 2: Linear Probe Evaluation

Linear probing trains a logistic regression classifier on top of frozen encoder features extracted from the pretrained model. It is used to verify that pretraining produced semantically meaningful representations before committing to expensive fine-tuning.

### Single-Task Probe

Evaluates subject-ID classification, t-SNE feature visualisation, and feature statistics (effective rank, variance collapse detection).

```bash
cd pretraining
python evaluate_pretrained.py
```

The checkpoint path and data path are hardcoded at the top of `main()`. Edit them if your paths differ:

```python
ckpt_path = 'Results/GLIM_Pretrain1/best_model.pth'
data_path = '../data/tmp/zuco_eeg_128ch_1280len.df'
```

Outputs:
- Console: train/test accuracy, random baseline, per-class report, feature statistics
- File: `Results/GLIM_Pretrain1/feature_tsne.png` (t-SNE coloured by subject ID)

Interpretation guidelines:

| Test accuracy range | Interpretation |
|---|---|
| > 25% | Excellent. Transfer to GLIM is recommended |
| 15-25% | Good. Small benefit expected from transfer |
| 10-15% | Fair. Consider adjusting mask ratio or EMA momentum |
| < 10% | Poor. Model may not have learned useful representations |

### Multi-Task Probe

Evaluates the pretrained encoder on five classification tasks simultaneously using the richer `zuco_eeg_label_8variants.df` which contains sentiment, relation, and reading paradigm labels alongside subject IDs.

```bash
cd pretraining
python evaluate_multitask_probe.py \
    --ckpt Results/GLIM_Pretrain1/best_model.pth \
    --data ../data/tmp/zuco_eeg_label_8variants.df \
    --batch_size 64 \
    --gpu 0 \
    --save_dir Results/GLIM_Pretrain1
```

Tasks evaluated:

| Task | Classes | Rationale |
|---|---|---|
| Subject ID | ~30 | Tests identity discrimination |
| Sentiment classification | 3 (negative, neutral, positive) | Tests semantic capture |
| Relation classification | 9 types | Tests fine-grained semantic structure |
| Reading paradigm (NR vs TSR, excluding SST) | 2 | Tests task-level generalisation |
| Reading paradigm (NR vs TSR, including SST) | 2 | Full paradigm discrimination |

Outputs:
- Console: per-task accuracy table with improvement-over-random multiplier
- File: `Results/<dir>/multitask_tsne.png` (five t-SNE plots side by side)
- File: `Results/<dir>/multitask_probe_summary.png` (bar chart comparing all tasks)
- File: `Results/<dir>/multitask_probe_results.txt` (plain-text summary)

The verdict logic:
- Average improvement > 2.0x over random across all tasks: good transfer candidate
- Average improvement > 1.5x: encoder captures some discriminative signal
- Semantic tasks (sentiment, relation) averaging > 1.5x improvement: the encoder is capturing meaning from EEG, directly supporting the world model hypothesis

### Inspecting the Pretrained Checkpoint

To inspect any pretrained checkpoint without running the full evaluation:

```bash
cd pretraining
python inspect_checkpoint.py
```

Edit the path inside `inspect_checkpoint.py` to point to your checkpoint.

---

## Stage 3: Fine-Tuning GLIM

Fine-tuning trains the complete GRAPE-GLIM pipeline end-to-end. The language model (Flan-T5-Large by default) is kept frozen throughout. Only the EEG encoder, prompt embedder, and cross-modal aligner are trained.

**All fine-tuning commands are run from the project root:**

```bash
cd e:\MSc Files\MSc Project\gated-glim\GLIM   # or equivalent on your system
```

### Option A: Training from Scratch (Baseline)

No pretraining. EEG encoder weights are randomly initialised. This reproduces the GLIM baseline with gated attention added.

```bash
python train.py
```

Key settings inside `train.py`:
- `devices = [0,1,2,3,4,5,6,7]` — edit to match your available GPU indices
- `bsz_train = 72` — total batch size per step
- `lr = 1e-4`
- `use_gated_attention = True`, `gating_type = 'elementwise'`
- Logs and checkpoints saved under `./runs/dev-dist/`
- Checkpointing every 10 epochs (`full_val_interval = 10`)

To run on a single GPU:

```python
# Inside train.py, change:
devices = [0]
```

### Option B: Fine-Tuning with JEPA Encoder (GRAPE-GLIM)

Loads pretrained weights into `model.eeg_encoder.in_blocks` before fine-tuning. This is the primary GRAPE-GLIM training pathway.

```bash
python train_with_jepa_encoder.py
```

The critical block inside the script:

```python
from pretraining.load_pretrained import load_pretrained_encoder, freeze_pretrained_encoder

PRETRAINED_CKPT_PATH = './pretraining/Results/GLIM_Pretrain1/best_model.pth'
FREEZE_ENCODER = False   # Set True to lock pretrained weights; False to fine-tune them

if os.path.exists(PRETRAINED_CKPT_PATH):
    model = load_pretrained_encoder(model, PRETRAINED_CKPT_PATH)
    if FREEZE_ENCODER:
        model = freeze_pretrained_encoder(model, freeze_in_blocks=True)
```

`load_pretrained_encoder` transfers only the `in_blocks` (context encoder) weights. The `out_blocks` (Q-Merger), prompt embedder, and aligner are always trained from scratch regardless of `FREEZE_ENCODER`.

**Weight key mapping during transfer:**

The pretrained checkpoint's `encoder_state_dict` uses keys of the form `in_blocks.i.layer.*`. These are matched to `glim.eeg_encoder.in_blocks.i.layer.*`. Any key present in the pretrained checkpoint but absent from the GLIM encoder is silently skipped. Any shape mismatch raises a warning and the corresponding weight is not transferred.

**Variant: `train_with_jepa_encoder2.py`** follows the same pattern but with slightly different group names and is used for ablation runs with different random seeds.

### Option C: Gated Attention with Nucleus Sampling

Demonstrates the full set of novel components: gated attention in the encoder plus nucleus (top-p) sampling at generation time.

```bash
python experiments/train_gated.py
```

Key differences from the baseline:

```python
# In experiments/train_gated.py:
use_gated_attention = True
gating_type = 'elementwise'      # elementwise sigmoid gate per attention output dimension
generation_strategy = 'nucleus'  # top-p probabilistic sampling instead of beam search
top_p = 0.95                     # probability mass threshold
temperature = 0.7                # lower = less random; 1.0 = unscaled logits
```

**Gating types:**
- `elementwise`: A sigmoid gate multiplied per output element of each attention head. More expressive, adds parameters proportional to `hidden_dim`.
- `headwise`: A single scalar gate per head. Fewer parameters, simpler ablation.

**Generation strategies:**
- `beam`: Deterministic beam search. `num_beams` controls width. Original GLIM behaviour.
- `nucleus`: Samples from the smallest probability-mass-sufficient vocabulary subset at each step. Produces more diverse outputs.
- `greedy`: Deterministic single-best-token selection at each step. Fastest inference.

### Option D: BART Language Model Backbone

Uses `facebook/bart-large-cnn` as the frozen decoder instead of Flan-T5-Large. Requires `embed_dim = 1024` (BART's d_model).

```bash
python experiments/train_with_jepa_encoder_bart.py
```

Or the standalone BART script without JEPA:

```bash
python experiments/train_bart.py
```

### Option E: Small Flan-T5 Variant

Uses `google/flan-t5-small` (embed_dim = 512). Suitable for development on machines with limited GPU memory.

```bash
python experiments/train_with_jepa_encoder-small.py
```

Note: `embed_dim` must be set to 512 (matching `flan-t5-small`'s d_model). The aligner's output projection changes accordingly.

### Fine-Tuning Configuration Reference

The `GLIM` class constructor accepts the following parameters. All parameters are saved to WandB via `save_hyperparameters`.

**EEG dimensions:**

| Parameter | Default | Description |
|---|---|---|
| `input_eeg_len` | 1280 | Input EEG sequence length in timesteps |
| `hidden_eeg_len` | 96 | Compressed sequence length after the EEG encoder's Q-Merger (out_blocks) |
| `input_text_len` | 96 | Maximum tokenised length of the input text fed to the T5 encoder |
| `tgt_text_len` | 64 | Maximum length of generated output tokens |
| `input_dim` | 128 | Number of EEG channels (ZuCo = 128) |
| `hidden_dim` | 256 | Internal hidden dimension of the EEG encoder transformer |
| `embed_dim` | 1024 | Shared cross-modal embedding dimension. Must equal the LM's d_model |

**Language model:**

| Parameter | Default | Description |
|---|---|---|
| `text_model_id` | `"google/flan-t5-large"` | HuggingFace model identifier. Also accepts `"google/flan-t5-small"`, `"facebook/bart-large-cnn"` |

**EEG encoder architecture:**

| Parameter | Default | Description |
|---|---|---|
| `n_in_blocks` | 6 | Number of temporal processing blocks (in_blocks). These receive the pretrained weights |
| `n_out_blocks` | 6 | Number of Q-Merger blocks (out_blocks). Always trained from scratch |
| `in_temporal_modulate` | True | Whether to apply temporal modulation (gating over time) inside in_blocks |
| `out_is_causal` | True | Whether out_blocks use causal (masked) self-attention |
| `num_heads` | 8 | Number of attention heads per block |
| `mlp_ratio` | 4 | Feed-forward layer expansion ratio |
| `dropout` | 0.0 | Dropout rate applied throughout the encoder |

**Prompt embedder:**

| Parameter | Default | Description |
|---|---|---|
| `prompt_nums` | (3, 3, 31) | Number of prompt tokens for (task, dataset, subject) dimensions |
| `prompt_dropout_probs` | (0.0, 1.0, 1.0) | Dropout probabilities per prompt dimension during training. 1.0 = always drop = dataset and subject prompts act as noise augmentation on training; 0.0 = always use |
| `evaluate_prompt_embed` | `'src'` | How to combine prompt embeddings at inference. `'src'` uses the source prompt token; `'zero'` zeros it; `'mean'` averages |
| `prompt_tuning_len` | 0 | If > 0, prepend learnable soft prompt tokens to the T5 encoder input |

**Gated attention:**

| Parameter | Default | Description |
|---|---|---|
| `use_gated_attention` | False | Enable sigmoid gates in EncoderBlock attention output |
| `gating_type` | `'elementwise'` | `'elementwise'` (per-dimension gate) or `'headwise'` (per-head scalar gate) |

**Generation:**

| Parameter | Default | Description |
|---|---|---|
| `generation_strategy` | `'beam'` | `'beam'`, `'nucleus'`, `'greedy'`, or `'energy'` |
| `num_beams` | 2 | Beam width for beam search |
| `top_p` | 0.95 | Cumulative probability mass for nucleus sampling |
| `top_k` | 0 | Top-k vocabulary filter. 0 = disabled (pure nucleus) |
| `temperature` | 0.7 | Logit scaling temperature for nucleus sampling |

**Loss and training:**

| Parameter | Default | Description |
|---|---|---|
| `clip_loss_weight` | 0.5 | Weight lambda for CLIP loss. Final loss = lambda * L_clip + (1-lambda) * L_lm + epsilon * L_commit |
| `commitment_loss_weight` | 0.0 | Weight epsilon for EEG-text embedding commitment loss |
| `commitment_loss_key` | `'mse'` | Commitment loss type: `'mse'` or `'kl_div'` |
| `use_y_mask` | False | If True, pass the input text attention mask to the LM decoder |
| `bsz_train` | 48 | Training batch size (total across all GPUs) |
| `bsz_val` | 24 | Validation batch size |
| `lr` | 1e-5 | Adam learning rate |
| `weight_decay` | 0 | Adam L2 weight decay |
| `full_val_interval` | 10 | Run full evaluation (generation + retrieval + classification) every N epochs |
| `bs_retrieval` | 24 | Batch size used for retrieval metric computation during validation |

**ETES evaluation:**

| Parameter | Default | Description |
|---|---|---|
| `use_etes_eval` | False | Enable EEG-Text Embedding Similarity (ETES) evaluation during full validation |
| `use_energy_loss` | False | Add energy-based contrastive loss to the training objective |
| `energy_loss_weight` | 0.3 | Weight for the energy loss term |
| `energy_type` | `'cosine'` | Energy distance type: `'cosine'`, `'bilinear'`, or `'mlp'` |

### Multi-GPU Distributed Training

GLIM uses PyTorch Lightning with DDP (Distributed Data Parallel). To train on multiple GPUs:

```python
# In train.py:
devices = [0, 1, 2, 3]   # Four GPUs
```

GLIMSampler handles distributed sampling automatically. It extends `DistributedSampler` and ensures that samples within each batch have distinct text UIDs, which is required for the CLIP contrastive loss to compute correct in-batch negatives.

**Important:** `use_distributed_sampler=False` is set in the Trainer because GLIMSampler manages distribution internally. Do not change this.

Checkpoints are saved under the WandB run directory. The directory is printed to the console at startup. Checkpoints are saved every `full_val_interval` epochs with a monitor key of `epoch` (all checkpoints kept, `save_top_k=-1`).

---

## Model Architecture

### EEG Encoder

The EEG encoder has two functionally distinct stages implemented as two `nn.ModuleList` objects inside `EEGEncoder`.

**in_blocks (temporal processing, 6 blocks):**
Each block is an `EncoderBlock` with optional temporal modulation (`in_temporal_modulate=True`). In temporal modulation mode, the block applies a learned gating of the temporal dimension independently of the channel dimension. These blocks receive JEPA pretrained weights when initialised from `train_with_jepa_encoder.py`.

**out_blocks (Q-Merger, 6 blocks):**
Each block is an `EncoderBlock` operating in causal mode (`out_is_causal=True`). This stage compresses the variable-length EEG sequence (1280 timesteps) down to a fixed-length representation (96 query tokens) compatible with the T5/BART encoder's expected sequence length. Always trained from scratch.

Prompt embeddings are injected into the residual stream before each block in both stages during fine-tuning. During JEPA pretraining, prompt injection is disabled.

### Prompt Embedder

The `PromptEmbedder` maps three discrete prompt dimensions (task, dataset, subject) to a single continuous vector via a learnable embedding table. During training, prompt dropout is applied independently per dimension according to `prompt_dropout_probs`. This forces the model to remain functional when subject or dataset information is unavailable, improving generalisation at evaluation time.

At evaluation, the `evaluate_prompt_embed` strategy controls how the (possibly zeroed) prompt is combined:
- `src`: Use the actual source prompt token embedding.
- `zero`: Replace the prompt with a zero vector.
- `mean`: Average over all prompt tokens in the embedding table.

### Cross-Modal Aligner

The `Aligner` bridges the EEG encoder output space and the language model's embedding space.

**embed_eeg:** Applies a linear projection followed by layer normalisation to produce:
- `eeg_embeds`: Shape (B, 96, 1024). Fed to the T5/BART decoder as encoder_outputs.
- `eeg_emb`: Shape (B, 1024). Global EEG embedding used for contrastive retrieval and classification.

**embed_text:** Encodes tokenised text through the frozen T5/BART encoder, then pools the hidden states to produce a (B, 1024) text embedding.

**align_emb_vector:** Computes symmetric cosine similarity logits between `eeg_emb` and `text_emb` for the CLIP loss and retrieval evaluation.

### Gated Attention

Gated attention extends the standard scaled dot-product attention output with a learned sigmoid gate. In the `elementwise` variant, each output element is multiplied by an independent gate value:

```
output = sigmoid(gate(x)) * SDPA(Q, K, V)
```

where `gate(x)` is a linear projection of the residual input `x`. This allows the model to selectively suppress or amplify attention outputs per dimension, which is particularly useful for EEG signals where specific frequency bands and electrode regions are more informative depending on the linguistic content.

### Language Model Decoder

The T5 or BART decoder takes `eeg_embeds` as `encoder_outputs` via `BaseModelOutput`. The input text (for teacher forcing during training) is tokenised with the prompt template `"To English: <sentence>"`. The decoder is kept completely frozen throughout: `requires_grad_(False)` is applied immediately after loading from HuggingFace and the model is excluded from all optimizer parameter groups. The checkpoint saving code (`on_save_checkpoint`) strips all `text_model.*` keys from the state dict because they would make checkpoints unnecessarily large.

---

## Evaluation and Testing

Full evaluation on the test set is run from a saved checkpoint using `experiments/run_eval.py` or the test mode of the Lightning Trainer.

**Run evaluation from a checkpoint:**

```bash
python experiments/run_eval.py \
    --checkpoint ./runs/v1/epoch=199-step=397600.ckpt \
    --data_path ./data/tmp/zuco_eeg_label_8variants.df \
    --gpu 0
```

**Metrics computed during full validation and test:**

| Metric | Description |
|---|---|
| BLEU-1 to BLEU-4 @MTV | BLEU against all 8 paraphrase targets (multi-target variant) |
| BLEU-1 to BLEU-4 @RAW | BLEU against the original input sentence |
| ROUGE-1 F / P / R @MTV | ROUGE-1 against all 8 targets |
| ROUGE-1 F / P / R @RAW | ROUGE-1 against original input |
| WER | Word Error Rate against original input |
| Retrieval Top-1/5/10 | EEG-to-text retrieval accuracy within batch |
| Sentiment ACC-1 | Zero-shot sentiment classification (3-class) |
| Relation ACC-1 / Top-3 | Zero-shot relation classification (9-class) |
| Corpus ACC | Zero-shot corpus discrimination (movie reviews vs. biographies) |

Classification is zero-shot: label embeddings are computed with prompt templates such as `"Sentiment classification: It is <label>."` and compared against the EEG embedding via cosine similarity. No classifier head is trained.

**Noise robustness evaluation.** The `GLIMDataModule` accepts `eval_noise_input=True`, which replaces the test EEG with standard Gaussian noise at evaluation time. This measures whether the model genuinely decodes from EEG or exploits text-only biases.

---

## Gradio Demo

The interactive demo visualises live inference for any of the 10 selected demo samples using the v1 or v2 checkpoint.

**Requirements:**

```bash
pip install gradio scipy mne
```

**Launch:**

```bash
cd demo
python app.py
```

This opens a browser at `http://localhost:7860`.

**Modes:**

- **Static (pre-computed):** Displays results cached during training evaluation. No GPU required.
- **Live Inference:** Runs the model on the selected sample in real time. Requires a CUDA-capable GPU with sufficient free VRAM (approximately 3GB for v1, 2.5GB for v2 due to model size differences). The first click loads the model into memory; subsequent clicks on any sample use the cached model.

**Note on Windows page file.** Live inference loads Flan-T5-Large (approximately 800MB) plus the GLIM checkpoint (237MB for v1, 160MB for v2) into GPU memory. If the system page file is too small relative to available RAM, the model load may fail with `OSError: The paging file is too small (error 1455)`. To resolve this, go to System Properties > Advanced > Performance Settings > Advanced > Virtual Memory and increase the page file size to at least 8GB, or close memory-intensive applications before running.

**Charts available:**
- Butterfly Plot: Overlay of all 128 EEG channels over time
- EEG Feature Space: UMAP/t-SNE of word-level EEG segments
- Spectrograms: Time-frequency plots with attention profile overlay

**Outputs displayed:**
- Generated text from EEG
- BLEU and ROUGE generation metrics
- ETES (EEG-Text Embedding Similarity) alignment metrics
- Zero-shot classification probabilities (sentiment, relation, corpus, reading paradigm)

**Pre-warming the cache.** To populate the demo cache for all samples before running the app, use:

```bash
cd demo
python create_demo_df.py --checkpoint v1 v2
```

This runs inference for all 10 demo samples and saves results as JSON files under `demo/cache/`.

---

## Experiment Scripts Reference

| Script | Location | Purpose |
|---|---|---|
| `train.py` | root | From-scratch multi-GPU training (8 GPUs default) |
| `train_with_jepa_encoder.py` | root | JEPA encoder transfer + gated attention, single GPU |
| `train_with_jepa_encoder2.py` | root | Variant of the above with different run group name |
| `experiments/train_gated.py` | experiments/ | Gated attention + nucleus sampling demo |
| `experiments/train_with_jepa_encoder-small.py` | experiments/ | JEPA + Flan-T5-Small (lower VRAM) |
| `experiments/train_with_jepa_encoder_bart.py` | experiments/ | JEPA + BART-Large-CNN backbone |
| `experiments/train_bart.py` | experiments/ | BART from scratch |
| `experiments/train_cls.py` | experiments/ | Classification-only training (no generation) |
| `experiments/train_energy.py` | experiments/ | Energy-based auxiliary loss variant |
| `experiments/train_cli.py` | experiments/ | Argparse-driven training for sweeps |
| `experiments/sweep_train.py` | experiments/ | WandB hyperparameter sweep launcher |
| `experiments/sweep_eval.py` | experiments/ | WandB sweep evaluation runner |
| `experiments/run_eval.py` | experiments/ | Evaluate a saved checkpoint |
| `pretraining/run_pretrain.py` | pretraining/ | Signal-JEPA pretraining |
| `pretraining/main.py` | pretraining/ | EEG2Rep-style pretraining (benchmark datasets) |
| `pretraining/evaluate_pretrained.py` | pretraining/ | Single-task probe + t-SNE |
| `pretraining/evaluate_multitask_probe.py` | pretraining/ | Multi-task probe (5 tasks) |
| `pretraining/evaluate_benchmarking_probe.py` | pretraining/ | Benchmarking probe pipeline (LogReg) |
| `pretraining/evaluate_benchmarking_probe_svm.py` | pretraining/ | Benchmarking probe pipeline (SVM) |

---

## Runs and Checkpoint Layout

```
runs/
|-- v1/
|   `-- epoch=199-step=397600.ckpt    # GRAPE-GLIM v1: gated attention, beam search
|-- v2/
|   `-- epoch=199-step=397600.ckpt    # GRAPE-GLIM v2: gated attention, noise-augmented dataset
|-- dev-dist-test/                    # Earlier test run with WandB artifacts
`-- latest-run/                       # Symlink or copy of the most recent run
```

**v1 checkpoint** (237MB): Flan-T5-Large decoder, hidden_dim=256, 6+6 encoder blocks, gated attention (elementwise), beam search decoding (num_beams=2), trained 200 epochs.

**v2 checkpoint** (160MB): Same architecture. The reduced file size is because the checkpoint was saved mid-run and contains fewer logged WandB tables. The difference in size does not reflect model parameter differences.

**Checkpoint format.** GLIM overrides `on_save_checkpoint` to remove all `text_model.*` keys. This means checkpoints do not contain the frozen language model weights. When loading a checkpoint, the text model is re-downloaded from HuggingFace (or loaded from the local HuggingFace cache). Ensure internet access or a pre-populated HuggingFace cache at `~/.cache/huggingface/`.

---

## Troubleshooting

**"The paging file is too small for this operation to complete" (Windows, error 1455)**

This error occurs when loading the T5-Large model from HuggingFace while available RAM is low. Solutions in order of preference:

1. Close Chrome, VS Code, and other memory-intensive processes before starting the demo or any evaluation script.
2. Increase the Windows page file: Control Panel > System > Advanced > Performance Settings > Advanced > Virtual Memory. Set it to a minimum of 8GB on the system drive.
3. Use the Flan-T5-Small variant (`text_model_id="google/flan-t5-small"`, `embed_dim=512`) which requires less memory.

**"encoder.embed_tokens.weight | MISSING"**

This warning appears when loading Flan-T5 with `tie_word_embeddings=False`. The encoder and decoder embedding matrices are not stored in the pretrained HuggingFace checkpoint because they are normally tied to `shared.weight`. The demo inference code (`demo/inference.py`) handles this automatically by copying `shared.weight` into both `encoder.embed_tokens.weight` and `decoder.embed_tokens.weight` after loading. This warning can be safely ignored when using the demo; the copy ensures correct decoding.

**"No pretrained checkpoint found at ./pretraining/Results/GLIM_Pretrain1/best_model.pth"**

Run Stage 1 pretraining first. The JEPA fine-tuning script falls back to random initialisation if the checkpoint is missing, which will still train a functional model but without the pretraining benefit.

**CUDA out of memory during training**

- Reduce `bsz_train` from 72 to 48 or 32 in the training script.
- Reduce `num_beams` from 2 to 1 during validation-time generation.
- Use `precision='bf16-mixed'` (already the default). Do not switch to fp32.
- If using multiple GPUs, try `strategy='ddp'` and ensure `use_distributed_sampler=False` remains set.

**WandB offline runs**

Uncomment `offline=True` in the `WandbLogger` call:

```python
logger = WandbLogger(project='glim', group=group_name, save_dir=log_dir, offline=True)
```

After training, sync offline runs with: `wandb sync ./runs/<group_name>/wandb/offline-run-*/`

**Checkpoint not loading (strict=False warning)**

All checkpoints are loaded with `strict=False` because the frozen text model keys are stripped at save time. The missing keys are expected and do not indicate a problem as long as the EEG encoder, aligner, and prompt embedder keys are all present.

**Port already in use (Gradio demo)**

```bash
python demo/app.py --port 7861
```

**Generating a public share link:**

```bash
python demo/app.py --share
```