# ETES: EEG–Text Energy Score

This document describes **ETES** (EEG–Text Energy Score): what it is, how it is measured in this codebase, and why it is useful for evaluating EEG-to-text models.

---

## 1. What is ETES?

**ETES** is an evaluation metric that measures how well **generated text** aligns with the **source EEG signal** in the model’s embedding space. It is implemented in `model/energy.py` as `ETESEvaluator`.

- **Name:** EEG–Text Energy Score (energy = “score” in the sense of an energy-based model: lower = better alignment).
- **Role:** It scores **EEG–text alignment** for the model’s outputs, using the same kind of alignment used during training (EEG and text in a shared space).
- **Difference from BLEU/ROUGE:** BLEU and ROUGE compare generated text to **reference text** (surface overlap). ETES compares **generated text** to the **EEG** via embeddings, so it is **EEG-aware** and can be used in a **reference-free** way (or together with references via the “gap” metric).

In short: **ETES = how well the model’s generated text “matches” the brain signal in the learned representation space.**

---

## 2. How is ETES measured?

### 2.1 Inputs

- **EEG embedding vectors**  
  Obtained from the model’s aligner for each test sample (same pathway as during training). Shape: `(B, D)`.

- **Generated texts**  
  The model’s decoded text for each sample (e.g. beam or nucleus decoding). List of strings, length `B`.

- **Reference texts** (optional)  
  Ground-truth transcripts. If provided, the code also computes **reference energy** and **ETES gap** (see below).

### 2.2 Computation pipeline

1. **Encode generated (and optionally reference) text**  
   - Tokenize → T5 encoder → aligner’s `embed_text()` → text embeddings in the **same space** as EEG embeddings.

2. **Alignment energy (main ETES score)**  
   For each sample \(i\):
   \[
   E_{\text{align}}(i) = -\cos\bigl(\text{eeg\_emb}_i,\, \text{text\_emb}_i\bigr).
   \]
   So:
   - **Lower energy ⇒ higher cosine similarity ⇒ better EEG–text alignment.**
   - In the code: `alignment_energy = -F.cosine_similarity(eeg_norm, text_norm, dim=-1)` (see `ETESEvaluator.compute_alignment_energy`).

3. **Aggregate**  
   - **ETES alignment:** mean of \(E_{\text{align}}\) over the batch (primary metric).  
   - Optional: std, min, max, median, percentiles (e.g. p10, p90) for analysis.

4. **Optional fluency term**  
   If enabled, a fluency (e.g. perplexity-based) term can be added to form a **total energy**; by default only alignment is used, so **etes_total = etes_alignment**.

5. **Reference comparison (when references exist)**  
   - **Reference energy:** same formula, but using **reference text** embeddings instead of generated:  
     \(E_{\text{ref}}(i) = -\cos(\text{eeg\_emb}_i,\, \text{ref\_text\_emb}_i)\).  
   - **ETES gap:** mean over samples of \((\text{alignment\_energy} - \text{reference\_energy})\).  
     - Gap &gt; 0: generated text is, on average, worse aligned to EEG than the reference.  
     - Gap &lt; 0: generated text is, on average, better aligned than the reference (rare).

So **measurement** is: embed EEG and text with the **same aligner**, compute **negative cosine similarity** per sample, then report mean (and optionally reference/gap).

### 2.3 Reported metrics (in logs / `run_eval`)

| Metric | Meaning |
|--------|--------|
| **ETES Alignment** | Mean alignment energy (generated text vs EEG). **Lower is better.** |
| **ETES Total** | Same as alignment if no fluency term; otherwise alignment + fluency. **Lower is better.** |
| **ETES Reference** | Mean energy when using reference text instead of generated. Lower = reference is well aligned to EEG. |
| **ETES Gap** | Mean(generated energy − reference energy). Positive = generated worse than reference. |

### 2.4 Interpretation bands (from code)

Qualitative interpretation of the **alignment** score (same scale as “ETES Alignment” / “ETES Total” when fluency is off):

| ETES (alignment) | Interpretation |
|-------------------|----------------|
| &lt; −0.8 | Excellent alignment |
| &lt; −0.5 | Good alignment |
| &lt; −0.2 | Fair alignment |
| &gt; 0   | Poor alignment (embeddings dissimilar) |

These are implemented in `ETESEvaluator._interpret_score`.

---

## 3. Why is ETES useful for evaluation?

### 3.1 Directly measures EEG–text alignment

- BLEU/ROUGE only compare **text to text** (generated vs reference). They do not use the EEG at all.
- ETES explicitly measures whether the **generated text** is aligned to the **brain signal** in the model’s space. So it answers: “Does the output match what the EEG represents?” rather than “Does the output match one reference string?”

### 3.2 EEG-aware and reference-free capable

- **Reference-free use:** You can compute ETES alignment without any reference transcripts. That is useful when references are missing, expensive, or you want a signal that does not depend on a single gold sentence.
- **With references:** Reference energy and ETES gap show how close generated text gets to the “ideal” (reference) in terms of EEG alignment, not just n-gram overlap.

### 3.3 Matches the training objective

- The model is trained (when energy is used) with an **energy-based contrastive loss** that pushes **low energy** for matched EEG–text pairs and high energy for mismatched pairs.
- ETES uses the **same** embedding space and the **same** notion of energy (negative cosine similarity). So ETES directly reflects what the model was optimized for: EEG–text alignment in that space.

### 3.4 Continuous and fine-grained

- ETES is a **continuous** score, so you can rank systems or checkpoints and see small differences.
- You can also look at **per-sample** ETES (e.g. via `get_sample_scores`) for error analysis and stability.

### 3.5 Complements text-only metrics

- **BLEU/ROUGE:** “Does the output look like this reference?”
- **ETES:** “Does the output match the EEG in the model’s semantic space?”
- A system can have high BLEU but low ETES (e.g. generic or copied text that doesn’t match the EEG), or higher ETES but moderate BLEU (e.g. paraphrases that align better to the signal). So ETES adds information that text-only metrics cannot provide.

### 3.6 When to enable ETES

- Enable with **`--use_energy`** in `run_eval.py` (and the code sets `use_etes_eval=True` when energy is requested).
- ETES requires the **aligner** and **text encoder** used at evaluation to be the same as (or compatible with) the trained model; it is only defined when the model has been trained with the shared EEG–text space (e.g. gated + energy setup).

---

## 4. Summary

| Aspect | ETES |
|--------|------|
| **What** | EEG–Text Energy Score: alignment between EEG and generated text in embedding space. |
| **How** | Embed EEG and text with the model’s aligner; score = −cosine_similarity; lower = better. |
| **Why useful** | EEG-aware, reference-free capable, consistent with energy-based training, continuous, complements BLEU/ROUGE. |

For implementation details, see `model/energy.py` (`ETESEvaluator`) and the evaluation call sites in `model/glim.py` (e.g. logging `etes_alignment`, `etes_total`, `etes_reference`, `etes_gap`).
