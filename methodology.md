# Methodology

---

## 1. Experiments: Architectural Explorations Before the Final Design

Before arriving at the final architecture — a JEPA-pretrained, gated-attention GLIM with CLS loss — several architectural experiments were conducted to explore alternative approaches to improving EEG-to-Text generation. This section documents the experiments that informed the final design decisions.

### 1.1 Baseline GLIM (CLS Loss Only)

The starting point was the original GLIM architecture, which uses a CLIP-style contrastive learning (CLS) loss for training:

- **Training objective:** A weighted combination of contrastive alignment loss (CLIP loss) and language modelling (LM) loss:

$$\mathcal{L} = \lambda \cdot \mathcal{L}_{CLIP} + (1-\lambda) \cdot \mathcal{L}_{LM}$$

where $\lambda = 0.5$.

- **Decoding:** Beam search with `num_beams=2`.
- **What was learned:** This serves as the baseline for all subsequent experiments. The CLS loss aligns EEG and text embeddings in a shared space via contrastive learning, while the LM loss trains the frozen Flan-T5-Large decoder to generate text from the aligned EEG embeddings.

> **Script:** `train_cls.py`

### 1.2 Energy-Based Model (EBM) Extension

The first major experiment explored adding an **energy-based contrastive loss** to the training objective, inspired by energy-based models:

- **Key additions:**
  - `EnergyContrastiveLoss`: An EBM-style contrastive objective that explicitly models an energy function $E(\text{eeg}, \text{text})$ where lower energy indicates better alignment. Unlike standard InfoNCE, this formulation includes learnable temperature, label smoothing, and embedding dropout.
  - `EnergyGuidedGenerator`: Energy-guided decoding that generates multiple candidate texts and selects the one with the lowest energy (highest EEG-text alignment). This reranking approach was tested with 5 candidates.
  - `ETESEvaluator` (EEG-Text Energy Score): A novel evaluation metric that measures how well generated text aligns with the source EEG signal in the model's embedding space, computed as $\text{ETES} = -\cos(\mathbf{e}_{\text{eeg}}, \mathbf{e}_{\text{text}})$. Lower is better.

- **Configuration:** Energy loss weight of 0.3, cosine energy type, combined with CLS and LM losses.

- **Outcome:** The energy loss was ultimately **removed from training** due to instability and marginal benefit. However, **ETES was retained as a validation-only metric** because it provides a reference-free, EEG-aware evaluation signal that complements BLEU/ROUGE.

> **Script:** `train_energy.py`

### 1.3 Gated Attention (NeurIPS 2025 Best Paper)

The next experiment introduced **Gated Attention** into the EEG encoder, based on the NeurIPS 2025 Best Paper:

- **Mechanism:** After Scaled Dot-Product Attention (SDPA), a learnable sigmoid gate is applied element-wise to the attention output:

$$\text{GatedAttention}(Q,K,V) = \sigma(W_g \cdot \text{SDPA}(Q,K,V)) \odot \text{SDPA}(Q,K,V)$$

- **Gating types explored:**  
  - **Elementwise** (selected): Gate per element — more expressive, learns fine-grained suppression.
  - **Headwise:** Gate per attention head — simpler but less discriminative.

- **XAI logging:** Gate activation statistics (mean, std, sparsity, entropy) and histograms are tracked at each validation step via WandB for interpretability analysis.

- **Outcome:** Gated attention was retained in the final architecture as it provides learnable signal filtering that is particularly beneficial for noisy EEG data.

> **Script:** `train_gated.py`

### 1.4 Nucleus Sampling vs Beam Search

An experiment on **decoding strategy** compared:

| Strategy | Configuration | Description |
|----------|--------------|-------------|
| Beam search | num_beams=1,2,4 | Deterministic, tends toward safe/generic outputs |
| Nucleus sampling | top_p=0.9,0.95; T=0.7 | Probabilistic, more diverse but less consistent |

- **Findings (from decoding sweep):**
  - **Beam-1** gave the best BLEU-1 (0.217) and Quick Score (1.376).
  - **Nucleus (top_p=0.95)** was incomplete but showed higher ETES (0.102), hinting at better EEG alignment despite lower text overlap.
  - Retrieval accuracy (Ret@1 = 0.076) was constant across all decoding strategies, confirming it depends on the encoder, not decoding.

- **Outcome:** Beam search (num_beams=2) was selected as the default for consistency and reproducibility. Nucleus sampling was offered as an option for diversity.

### 1.5 BART-base Language Model

An experiment replaced Flan-T5-Large with **BART-large-CNN** as the frozen language model:

- **Motivation:** Test whether a different pretrained LM architecture would improve generation quality for EEG-to-Text.
- **Outcome:** Flan-T5-Large was retained as the default, as it produced superior results in preliminary comparisons.

> **Script:** `train_bart.py`

### 1.6 Summary of Experiments

| Experiment | What Was Tested | Contribution to Final Architecture |
|-----------|----------------|-----------------------------------|
| Baseline CLS | Standard CLIP loss training | ✅ Core loss retained |
| Energy-Based Model | EBM contrastive loss + energy decoding | ❌ Loss removed; ✅ ETES metric retained |
| Gated Attention | Learnable sigmoid gates on attention | ✅ Retained (elementwise) |
| Nucleus Sampling | Probabilistic decoding | ❌ Not default; available as option |
| BART Language Model | Alternative LM backbone | ❌ Flan-T5-Large preferred |
| JEPA Pretraining | Self-supervised encoder pretraining | ✅ Core contribution (see Section 2) |

---

## 2. Pretraining Methodology

### 2.1 Motivation

The EEG encoder in GLIM is initialised randomly and must learn both EEG representation and cross-modal alignment simultaneously during fine-tuning. This section describes the self-supervised pretraining stage designed to give the encoder a head start by learning an internal model of brain activity dynamics before any text supervision.

### 2.2 Architecture: Signal-JEPA for EEG

We implement a **Joint Embedding Predictive Architecture (JEPA)** to pretrain the EEG encoder component of GLIM. JEPA, proposed by LeCun (2022), learns to predict abstract representations of masked input rather than reconstructing raw signals — forcing the encoder to learn high-level semantic features rather than low-level signal patterns.

#### Components

| Component | Configuration | Purpose |
|-----------|--------------|---------|
| **Context Encoder** | 6× `EncoderBlock`, 128 dim, 8 heads | Encodes visible EEG patches into representations |
| **Target Encoder** | EMA copy of context (momentum $\tau$=0.99) | Provides prediction targets without gradients |
| **Predictor** | 2× Cross-Attention blocks | Predicts masked patch representations from context |
| **Patch Embedding** | patch_size=8 → 160 patches from 1280 timesteps | Converts raw EEG (128 channels) into token sequences |
| **Positional Encoding** | 1D sinusoidal | Preserves temporal order information |

#### Design Decisions

1. **JEPA over MAE:** Prediction in representation space, not raw EEG space. This avoids wasting model capacity on reconstructing low-level signal noise and focuses on learning abstract, semantic features of brain activity.

2. **GLIM-compatible architecture:** The pretraining encoder uses the **exact same `EncoderBlock`** class as the full GLIM model, ensuring zero-friction weight transfer with no adapters or dimension mismatches.

3. **SSP Masking (from EEG2Rep):** Rather than random token masking (designed for images), **Semantic Subsequence Preserving** masking selects contiguous temporal chunks as visible context. This preserves the temporal semantics of EEG signals — the encoder must learn brain activity dynamics, not spatial interpolation.

4. **VICReg Loss:** The loss function combines three terms to prevent representation collapse:

$$\mathcal{L}_{\text{VICReg}} = \alpha \cdot \mathcal{L}_{\text{align}} + \beta \cdot \mathcal{L}_{\text{var}} + \gamma \cdot \mathcal{L}_{\text{cov}}$$

   - **Alignment ($\mathcal{L}_{\text{align}}$):** MSE between predicted and target representations.
   - **Variance ($\mathcal{L}_{\text{var}}$):** Ensures each dimension maintains sufficient variance across the batch.
   - **Covariance ($\mathcal{L}_{\text{cov}}$):** Decorrelates dimensions to prevent redundancy.

   Unlike contrastive methods (SimCLR, CLIP), VICReg does not require negative pairs, which are difficult to define meaningfully for EEG data.

### 2.3 Training Configuration

#### Baseline (GLIM_Pretrain1)

| Parameter | Value |
|-----------|-------|
| Encoder blocks | 6× `EncoderBlock` |
| Embedding dim | 128 |
| Attention heads | 8 |
| Gated attention | **Disabled** |
| Dataset | ZuCo EEG (128 channels, 1280 timesteps) |
| Epochs completed | ~52 |
| Steps per epoch | 5,523 |
| Batch size | 72 |
| Learning rate | 1e-4 (cosine decay) |
| Weight decay | 0.04 → 0.048 (progressive) |
| Mask ratio | 50% |
| Precision | bfloat16 mixed |
| Hardware | Single GPU |
| Training time | ~3.5 hours |

#### Latest (GLIM_Pretrain5)

| Parameter | Value |
|-----------|-------|
| Encoder blocks | 6× `EncoderBlock` |
| Embedding dim | 128 |
| Attention heads | 8 |
| Gated attention | **Enabled** (elementwise) |
| Dataset | ZuCo EEG (128 channels, 1280 timesteps) |
| Epochs | Extended training (200 epochs configured) |
| Batch size | 72 |
| Learning rate | 1e-4 (cosine decay) |
| Weight decay | 0.04 → 0.048 (progressive) |
| Mask ratio | 50% |
| Precision | bfloat16 mixed |
| Hardware | Single GPU |
| Predictor layers | 2 cross-attention blocks |

### 2.4 Pretraining Results

#### 2.4.1 Baseline (GLIM_Pretrain1) Results

**Loss Convergence:**
The training loss followed a characteristic three-phase pattern:

| Phase | Epochs | Loss | Description |
|-------|--------|------|-------------|
| Phase 1 | 1–3 | 0.85 → 0.25 | Rapid initial decrease — basic EEG structure learned |
| Phase 2 | 4–12 | ↑ to ~0.44 | VICReg regularisation activated, forcing diversity |
| Phase 3 | 13–52 | 0.44 → 0.356 | Gradual convergence — semantic feature learning |

**Training Health:**

| Metric | Result | Status |
|--------|--------|--------|
| Final loss | 0.356 (stable 16+ epochs) | ✅ Converged |
| Representation collapse | var = 0.990 constant | ✅ None |
| Gradient stability | No vanishing/exploding | ✅ Stable |
| Loss balance (pred vs reg) | Both decreasing | ✅ Balanced |
| Linear probe accuracy | **27.8%** | ✅ Non-trivial features |

#### 2.4.2 Latest (GLIM_Pretrain5) Results

**Multi-Task Linear Probe Evaluation:**

| Task | Classes | Train Acc | Test Acc | Random Baseline | vs Random |
|------|---------|-----------|----------|-----------------|-----------|
| Subject ID classification | 30 | 100.0% | **99.6%** | 3.3% | **29.9×** |
| Sentiment classification | 3 | 46.8% | **33.6%** | 33.3% | 1.0× |
| Relation classification | 9 | 61.8% | **56.5%** | 11.1% | **5.1×** |

**Interpretation:**
- **Subject ID (99.6%):** The encoder has learned strong subject-specific EEG patterns — individual neural signatures are highly discriminable from frozen representations.
- **Relation (56.5%, 5.1× random):** The encoder has captured meaningful semantic content from EEG — relation types are distinguishable above chance.
- **Sentiment (33.6%, ~random):** Sentiment information is not well-captured by the EEG encoder at this stage, consistent with the inherent difficulty of extracting sentiment from raw EEG signals.

### 2.5 Comparison: Baseline vs Latest Pretraining

| Metric | Baseline (Pretrain1) | Latest (Pretrain5) |
|--------|---------------------|-------------------|
| Gated attention | No | Yes (elementwise) |
| Training epochs | ~52 | Extended (200 configured) |
| Linear probe (overall) | 27.8% | — |
| Subject ID (test acc) | — | **99.6%** |
| Relation (test acc) | — | **56.5%** |
| Sentiment (test acc) | — | 33.6% |
| Representation collapse | None (var = 0.990) | None |
| Loss convergence | 0.85 → 0.356 | Converged |

The latest pretraining configuration with gated attention and longer training produces substantially richer representations, particularly for subject identification (29.9× random) and relation classification (5.1× random).

---

## 3. Fine-Tuning Methodology

### 3.1 Architecture Overview

After pretraining, the encoder weights are transferred to the full GLIM architecture for end-to-end EEG-to-Text generation:

```
Raw EEG (B, 1280, 128)
        ↓
[Prompt Embedder] → prompt conditioning (task, dataset, subject)
        ↓
[EEG Encoder]
  ├── in_blocks (6× EncoderBlock)  ← JEPA pretrained weights
  └── out_blocks (6× DecoderBlock) ← Q-Merger, trained from scratch
        ↓
[Aligner]
  ├── EEG → embed vector (B, 1024)
  ├── Text → embed vector (B, 1024)
  └── CLIP contrastive alignment
        ↓
[Flan-T5-Large Decoder] (frozen)
        ↓
Generated Text
```

#### Component Initialisation

| Component | Initialisation | Training Status |
|-----------|---------------|-----------------|
| EEG Encoder (in_blocks) | ✅ JEPA pretrained weights | Fine-tuned (not frozen) |
| Q-Merger (out_blocks) | Random initialisation | Trained from scratch |
| Aligner | Random initialisation | Trained from scratch |
| Prompt Embedder | Random initialisation | Trained from scratch |
| Gated Attention gates | Learnable (sigmoid) | Trained with encoder |
| Flan-T5-Large (LLM) | HuggingFace pretrained | **Frozen** |

### 3.2 Training (`train.py`)

Training uses PyTorch Lightning with the following configuration:

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam (lr=1e-4, weight_decay=0) |
| Max epochs | 200 |
| Precision | bfloat16 mixed |
| Batch size (train) | 72 |
| Batch size (val) | 24 |
| Full validation interval | Every 10 epochs |
| Checkpointing | Save every `full_val_interval` epochs |
| Random seed | 42 |
| Data | `zuco_eeg_label_8variants.df` |
| Hardware | Multi-GPU (up to 8× GPUs) |

**Loss function:**

$$\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{CLIP}} + (1-\lambda) \cdot \mathcal{L}_{\text{LM}} + \varepsilon \cdot \mathcal{L}_{\text{commitment}}$$

where $\lambda = 0.5$, $\varepsilon = 0$ (commitment loss disabled).

**Data sampling:** A custom `GLIMSampler` samples batches by **text identity** rather than sample index, ensuring all samples within a batch have distinct text identifiers. This is essential for the CLIP contrastive loss to function correctly — without it, duplicate texts in a batch create false negatives.

#### Loading Pretrained Weights

```python
from pretraining.load_pretrained import load_pretrained_encoder

PRETRAINED_CKPT_PATH = './pretraining/Results/GLIM_Pretrain5/best_model.pth'

model = GLIM(...)  # Standard GLIM initialisation
model = load_pretrained_encoder(model, PRETRAINED_CKPT_PATH)
```

The `load_pretrained_encoder` function maps the JEPA context encoder weights to GLIM's `eeg_encoder.in_blocks`, matching keys by name and verifying shape compatibility. Only `in_blocks` weights are transferred; `out_blocks` (Q-Merger) and all other components are trained from scratch.

### 3.3 Testing (`test.py`)

Testing loads a trained checkpoint and evaluates on the held-out test split:

```python
model = GLIM.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    strict=False,
    use_etes_eval=True,
    log_xai=True,
)
trainer.test(model, datamodule=dm)
```

**Metrics computed during testing:**

| Category | Metrics |
|----------|---------|
| Generation | BLEU-1,2,3,4; ROUGE-1 (precision, recall, F-measure); WER |
| Retrieval | Top-1, Top-5, Top-10 accuracy |
| Classification | Corpus accuracy, Relation top-1/top-3, Sentiment top-1 |
| EEG alignment | ETES alignment, ETES total, ETES gap |
| XAI | Gate mean, std, sparsity, entropy; attention heatmaps |

### 3.4 Prediction (`predict.py`)

The unified prediction script runs corpus, relation, and sentiment classification on the test set and computes two types of accuracy:

1. **CLIP-like accuracy (EEG-based):** Computes cosine similarity between EEG embeddings and candidate label embeddings to classify samples. This measures how well the encoder's learned representations discriminate between categories.

2. **CLIP-like accuracy (text-based):** Same approach but using text embeddings (both raw input and generated text) instead of EEG embeddings. This reveals how much task-relevant information the generated text retains.

3. **LLM-based prediction:** Uses a separate LLM to classify generated text into categories. This tests whether the generated text is coherent and informative enough for downstream NLP.

**Tasks:**

| Task | Labels | Description |
|------|--------|-------------|
| Corpus | movie review / personal biography | Binary corpus classification |
| Relation | 9 relation types | Multi-class relation extraction |
| Sentiment | negative / neutral / positive | 3-class sentiment classification |

---

## 4. Fine-Tuning Results and Comparison

### 4.1 Nucleus-Trained vs Beam-Trained (with Energy + Gated Attention)

Two training runs were compared, both using gated attention but with different generation strategies during training:

**Generation metrics (beam decoding at evaluation):**

| Metric | With-Energy (Nucleus-trained) | Without-Energy (Beam-trained) |
|--------|-------------------------------|-------------------------------|
| **BLEU-1** | 0.1827 | **0.2440** |
| **BLEU-2** | 0.0645 | **0.0864** |
| **ROUGE-1** | 0.1621 | **0.2194** |
| **ETES Total** | **-0.027** | -0.399 |
| **Corpus cls acc** | 0.553 | **0.692** |
| **Relation top-1** | **0.235** | 0.032 |
| **Sentiment top-1** | **0.357** | 0.295 |

**Prediction results (predict.py):**

| Task | Metric | With-Energy | Without-Energy |
|------|--------|-------------|----------------|
| Corpus | CLIP-like (EEG) | 0.553 | **0.692** |
| | CLIP-like (text gen) | 0.268 | **0.806** |
| | LLM prediction | 0.195 | **0.960** |
| Relation | CLIP-like (EEG) | **0.235** | 0.032 |
| | CLIP-like (text gen) | **0.119** | 0.093 |
| Sentiment | CLIP-like (EEG) | **0.357** | 0.295 |
| | CLIP-like (text gen) | **0.432** | 0.312 |
| | LLM prediction | 0.017 | **0.305** |

**Key findings:**
- **Without-energy (beam-trained)** excels at text generation quality (BLEU, ROUGE) and corpus classification (including near-perfect LLM-based prediction at 96%).
- **With-energy (nucleus-trained)** excels at ETES (EEG-text alignment), relation classification, and sentiment EEG-based classification.
- The trade-off suggests that energy-based training produces better EEG representations but at the cost of surface-level text quality.

### 4.2 Decoding Strategy Sweep (on Beam-Trained Checkpoint)

| Decoding Config | Quick Score | BLEU-1@MTV | Ret@1 | ETES |
|-----------------|-------------|------------|-------|------|
| Beam, num_beams=1 | **1.376** | **0.217** | 0.076 | -0.039 |
| Beam, num_beams=2 | 1.267 | 0.183 | 0.076 | -0.027 |
| Beam, num_beams=4 | 1.221 | 0.169 | 0.076 | -0.017 |
| Nucleus, top_p=0.9 | 1.221 | 0.169 | 0.076 | -0.017 |
| Nucleus, top_p=0.95 | 0.658 | ~0.001 | 0.076 | **0.102** |

- Beam-1 (greedy) gives the best text quality; Nucleus (top_p=0.95) gives the best ETES but nearly zero BLEU, indicating the generated text diverges from references while staying aligned to the EEG.
- Retrieval accuracy is constant across decoding strategies (depends on encoder, not decoder).

### 4.3 Latest Fine-Tuning Results (Epoch 199 — JEPA-Pretrained + Gated Attention)

This is the final model: JEPA-pretrained EEG encoder with gated attention, fine-tuned for 200 epochs using CLS loss and beam-search decoding.

#### 4.3.1 Classification Results (`predict.py` — Epoch 199)

| Task | Metric | Value |
|------|--------|-------|
| **Corpus** | EEG Accuracy | **0.4121** |
| | Text Acc (Raw) | 0.5072 |
| | Text Acc (Gen) | 0.3909 |
| | LLM Prediction | **0.9121** |
| **Relation** | EEG Acc (Top-1) | 0.1367 |
| | EEG Acc (Top-3) | **0.3653** |
| | Text Acc (Raw) | 0.3786 |
| | Text Acc (Gen) | 0.3020 |
| | LLM Pred (Top-1) | 0.0776 |
| | LLM Pred (Top-3) | **0.7286** |
| **Sentiment** | EEG Accuracy | **0.2854** |
| | Text Acc (Raw) | 0.4197 |
| | Text Acc (Gen) | 0.2878 |
| | LLM Prediction | 0.3094 |

**Key observations:**
- **Corpus classification** is the strongest task: the LLM prediction accuracy of 91.2% shows the generated text is coherent and corpus-discriminative.
- **Relation classification** is most informative with top-3 EEG accuracy at 36.5% and LLM top-3 at 72.9%, indicating relations are captured in the EEG embedding space.
- **Sentiment classification** remains the most difficult task, consistent with the pretraining probe results (33.6% ≈ random for sentiment).

#### 4.3.2 Generation & Alignment Metrics (`test.py` — Epoch 199)

| Metric | Value |
|--------|-------|
| **BLEU-1** | 0.2068 |
| **BLEU-2** | 0.0532 |
| **ROUGE-1** (F-measure@MTV) | 0.2114 |
| **ROUGE-1** (F-measure@RAW) | 0.1132 |
| **ETES alignment** | **−0.451** |
| **ETES reference** | −0.149 |
| **ETES gap** | −0.302 |
| **Ret@1** | 0.072 |
| **Ret@5** | 0.311 |
| **Ret@10** | 0.556 |
| Corpus cls acc (test.py) | 0.412 |
| Relation cls acc (Top-1) | 0.137 |
| Relation cls acc (Top-3) | 0.366 |
| Sentiment cls acc | 0.355 |

#### 4.3.3 XAI Gate Statistics (`test.py` — Epoch 199)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Gate entropy | 0.988 | Near-maximal diversity — gates utilise full capacity |
| Gate mean | 0.472 | ~47% pass-through — the encoder actively filters ~53% of information |
| Gate sparsity | 0.136 | 13.6% of gates are strongly suppressed (< 0.1) |
| Gate std | 0.258 | Moderate selectivity — gates are not trivially uniform |

#### 4.3.4 Comparison with Original GLIM

| Enhancement | Original GLIM | Our Architecture |
|-------------|--------------|------------------|
| Encoder pretraining | ❌ Random init | ✅ JEPA pretrained |
| Gated attention | ❌ Standard SDPA | ✅ Elementwise gating |
| ETES evaluation | ❌ Not available | ✅ Reference-free metric |
| XAI logging | ❌ Not available | ✅ Gate statistics + attention heatmaps |
| Prompt conditioning | ✅ (task, dataset, subject) | ✅ Same, with dropout |
| LM backbone | Flan-T5-Large (frozen) | Same |
| Decoding | Beam search | Beam (default) + nucleus option |

**Noise ablation ($\mathcal{N}_{\text{in}}$):** Following the GLIM paper's evaluation protocol, a noise test replaces real EEG input with Gaussian noise to verify that the model relies on actual EEG signals rather than text priors. This ablation was implemented in `predict_noise_test.py`.

### 4.4 Epoch 59 Results (JEPA-Pretrained + Gated Attention — Earlier Checkpoint)

This checkpoint was saved at epoch 59 of the same training run. It serves as an intermediate comparison point.

#### 4.4.1 Generation & Alignment Metrics (`test.py` — Epoch 59)

| Metric | Value |
|--------|-------|
| **BLEU-1** | 0.2037 |
| **BLEU-2** | 0.0516 |
| **ROUGE-1** (F-measure@MTV) | 0.2058 |
| **ROUGE-1** (F-measure@RAW) | 0.1119 |
| **ETES alignment** | **−0.607** |
| **ETES reference** | −0.276 |
| **ETES gap** | −0.331 |
| **Ret@1** | 0.066 |
| **Ret@5** | 0.306 |
| **Ret@10** | 0.537 |
| Corpus cls acc (test.py) | 0.371 |
| Relation cls acc (Top-1) | 0.212 |
| Relation cls acc (Top-3) | 0.440 |
| Sentiment cls acc | 0.367 |

#### 4.4.2 Classification Results (`predict.py` — Epoch 59)

| Task | Metric | Value |
|------|--------|-------|
| **Corpus** | EEG Accuracy | 0.3709 |
| | Text Acc (Raw) | 0.5679 |
| | Text Acc (Gen) | 0.4062 |
| | LLM Prediction | **0.9339** |
| **Relation** | EEG Acc (Top-1) | **0.2112** |
| | EEG Acc (Top-3) | **0.4398** |
| | Text Acc (Raw) | 0.2980 |
| | Text Acc (Gen) | **0.4031** |
| | LLM Pred (Top-1) | 0.0745 |
| | LLM Pred (Top-3) | 0.6449 |
| **Sentiment** | EEG Accuracy | 0.2686 |
| | Text Acc (Raw) | 0.2830 |
| | Text Acc (Gen) | **0.3405** |
| | LLM Prediction | 0.3022 |

### 4.5 Three-Way Comparison: Epoch 59 vs Epoch 199 vs GLIM (Liu et al.) Paper

> **Note:** GLIM (Liu et al.) results are from [the paper](https://arxiv.org/html/2505.17099v1) (Tables 1 & 2). The original GLIM was trained on 8×RTX-4090D GPUs for 200 epochs with MTV augmentation (143K EEG-text triplets), no JEPA pretraining, no gated attention, and no ETES metric. Our model uses a single GPU with JEPA-pretrained encoder + gated attention.

#### 4.5.1 Generation & Retrieval Metrics (`test.py`)

| Metric | Epoch 59 | Epoch 199 | GLIM Paper | Best |
|--------|----------|-----------|------------|------|
| **BLEU-1 @MTV** | 0.2037 | 0.2068 | **0.2604** | GLIM |
| **BLEU-2 @MTV** | 0.0516 | 0.0532 | **0.1056** | GLIM |
| **ROUGE-1 @RAW** | 0.1119 | 0.1132 | **0.1227** | GLIM |
| **ROUGE-1 F@MTV** | 0.2058 | **0.2114** | — | Ep 199 |
| **Ret@1** | 0.066 | 0.072 | **0.082** | GLIM |
| **Ret@5** | 0.306 | 0.311 | **0.351** | GLIM |
| **Ret@10** | 0.537 | **0.556** | — | Ep 199 |

#### 4.5.2 EEG-Text Alignment (ETES)

| Metric | Epoch 59 | Epoch 199 | GLIM Paper | Best |
|--------|----------|-----------|------------|------|
| **ETES alignment** | −0.607 | **−0.451** | N/A | Ep 199 |
| **ETES reference** | −0.276 | **−0.149** | N/A | Ep 199 |
| **ETES gap** | −0.331 | **−0.302** | N/A | Ep 199 |

> ETES is our novel contribution — it was not available in the original GLIM paper.

#### 4.5.3 Zero-Shot Classification (EEG Embeddings — `test.py`)

| Task | Epoch 59 | Epoch 199 | GLIM Paper | Best |
|------|----------|-----------|------------|------|
| **Corpus** | 0.371 | 0.412 | **0.935** | GLIM |
| **Relation (Top-1)** | 0.212 | 0.137 | **0.325** | GLIM |
| **Relation (Top-3)** | 0.440 | 0.366 | **0.571** | GLIM |
| **Sentiment** | 0.367 | 0.355 | **0.427** | GLIM |

#### 4.5.4 Classification on Generated Text (`predict.py` / LLM-Assisted)

| Task | Epoch 59 | Epoch 199 | GLIM Paper | Best |
|------|----------|-----------|------------|------|
| **Corpus (EEG acc)** | 0.371 | **0.412** | — | Ep 199 |
| **Corpus (LLM pred)** | **0.934** | 0.912 | 0.922 | Ep 59 |
| **Relation EEG (Top-1)** | **0.211** | 0.137 | — | Ep 59 |
| **Relation EEG (Top-3)** | 0.440 | 0.365 | — | Ep 59 |
| **Relation LLM (Top-3)** | 0.645 | **0.729** | 0.563 | Ep 199 |
| **Sentiment (EEG acc)** | 0.269 | 0.285 | — | Ep 199 |
| **Sentiment (LLM pred)** | 0.302 | 0.309 | **0.396** | GLIM |

#### 4.5.5 XAI Gate Statistics (Our Model Only)

| Metric | Epoch 199 | GLIM Paper |
|--------|-----------|------------|
| Gate entropy | 0.988 | N/A |
| Gate mean | 0.472 | N/A |
| Gate sparsity | 0.136 | N/A |
| Gate std | 0.258 | N/A |

> Gated attention and XAI logging are our additions — the original GLIM uses standard scaled dot-product attention.

### 4.6 Key Findings

**Our model (Epoch 199) vs Original GLIM Paper:**

- **Generation gap exists but is narrowing:** Our BLEU-1 (0.207) reaches ~79% of GLIM's (0.260). The gap likely stems from MTV augmentation (143K triplets vs our ~16K) and multi-GPU training (8×4090D vs single GPU).
- **Retrieval is competitive:** Ret@1 (0.072 vs 0.082) and Ret@5 (0.311 vs 0.351) are within striking distance.
- **Classification gap is significant for EEG-embedding tasks:** Corpus (0.412 vs 0.935) and relation (0.366 vs 0.571 top-3) show our model's EEG embeddings are less discriminative — likely because the JEPA encoder was pretrained for general representation learning, not contrastive EEG-text alignment.
- **LLM-based relation classification EXCEEDS the original GLIM:** Our Relation LLM Top-3 (0.729) significantly surpasses GLIM's (0.563), suggesting our generated text captures relational semantics better despite lower raw EEG embedding quality.
- **Corpus LLM prediction is comparable:** Our 0.912–0.934 vs GLIM's 0.922 — essentially equivalent.
- **ETES and XAI are novel contributions** not available in the original GLIM, providing new insights into EEG-text alignment quality and encoder behaviour.

**Epoch 59 vs Epoch 199:**

- **Epoch 199 wins on most metrics:** Better ETES (−0.451 vs −0.607), generation (BLEU, ROUGE), retrieval, and corpus classification.
- **Epoch 59 wins on relation/sentiment EEG classification:** Relation EEG Top-1 (0.211 vs 0.137) and Top-3 (0.440 vs 0.365) are substantially better, suggesting early-epoch representations preserve fine-grained semantic distinctions that are lost with further training.
- **Trade-off:** Longer training shifts the model toward text generation quality at the cost of EEG-level semantic discrimination. The encoder gradually specialises for the LM objective rather than maintaining general-purpose representations.

---

## 5. Evaluation Metrics

### 5.1 Generation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **BLEU-1,2,3,4** | N-gram precision against reference text | Higher ↑ |
| **ROUGE-1** (F1, Precision, Recall) | Unigram overlap with reference | Higher ↑ |
| **WER** | Word error rate vs raw input | Lower ↓ |

### 5.2 Retrieval Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **Ret@1 / Ret@5 / Ret@10** | Top-k retrieval accuracy (EEG→text in batch) | Higher ↑ |

### 5.3 Classification Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **Corpus cls acc** | Binary corpus classification (movie/biography) | Higher ↑ |
| **Relation top-1 / top-3** | 9-class relation classification | Higher ↑ |
| **Sentiment top-1** | 3-class sentiment classification | Higher ↑ |

### 5.4 EEG-Text Energy Score (ETES)

| Metric | Description | Direction |
|--------|-------------|-----------|
| **ETES alignment** | $-\cos(\mathbf{e}_{\text{eeg}}, \mathbf{e}_{\text{gen\_text}})$ | Lower ↓ |
| **ETES reference** | Same, using reference text | Lower ↓ |
| **ETES gap** | Alignment − Reference (positive = generated worse) | Lower ↓ |

ETES is **EEG-aware** and **reference-free capable**, complementing BLEU/ROUGE which only compare text-to-text.

| ETES Range | Interpretation |
|-----------|---------------|
| < −0.8 | Excellent alignment |
| < −0.5 | Good alignment |
| < −0.2 | Fair alignment |
| > 0 | Poor alignment |

### 5.5 XAI Metrics

| Metric | Description |
|--------|-------------|
| Gate mean | Average gate activation (0=fully suppressed, 1=fully passed) |
| Gate std | Standard deviation of gate activations (higher=more selective) |
| Gate sparsity | Fraction of gates < 0.1 (higher=more suppression) |
| Gate entropy | Normalised entropy of gate distribution (higher=more diverse) |

---

## 6. Implementation Details

### 6.1 Data

- **Dataset:** ZuCo (Zurich Cognitive Language Processing Corpus) — EEG recordings from subjects reading sentences from two sources:
  - **ZuCo1:** Movie reviews (Stanford Sentiment Treebank) and Wikipedia relation sentences
  - **ZuCo2:** Additional reading tasks
- **EEG format:** 128 channels × 1280 time steps per sentence
- **Text augmentation:** 8 text variants per sentence (`zuco_eeg_label_8variants.df`)
- **Split:** Standard train/val/test split
- **30 subjects** across both datasets

### 6.2 Model Architecture (Key Dimensions)

| Component | Dimension |
|-----------|-----------|
| EEG input | (B, 1280, 128) — time × channels |
| Hidden EEG length | 96 |
| Input text length | 96 tokens |
| Target text length | 64 tokens |
| EEG dim → hidden dim | 128 → 256 |
| Embedding dim | 1024 (matches Flan-T5-Large) |
| Encoder blocks (in) | 6 |
| Decoder blocks (out) | 6 |
| Attention heads | 8 |
| MLP ratio | 4 |
| Prompt configuration | (3 tasks, 3 datasets, 31 subjects) |
| Prompt dropout | (0.0, 1.0, 1.0) — always drop dataset/subject in training |

### 6.3 Software Stack

| Library | Purpose |
|---------|---------|
| PyTorch + Lightning | Training framework |
| Transformers (HuggingFace) | Flan-T5-Large, BART, tokenisers |
| WandB | Experiment tracking and logging |
| torchmetrics | BLEU, ROUGE, WER, classification accuracy |
| timm | MLP blocks for transformer architecture |

---

## 7. Reproducibility

All experiments use:
- **Fixed random seed:** 42 (via `L.seed_everything(42, workers=True)`)
- **Deterministic matmul:** `torch.set_float32_matmul_precision('medium')`
- **Checkpointing:** Every `full_val_interval` epochs (default: 10)
- **Logging:** WandB for all metrics, including per-sample generation tables

The pretrained encoder checkpoints are saved as `.pth` files containing:
- `model_state_dict`: Full pretraining model weights
- `encoder_state_dict`: Extracted encoder weights for transfer
- `epoch`: Number of pretraining epochs completed
- `accuracy`: Linear probe accuracy at save time
