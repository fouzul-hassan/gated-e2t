# Comparison: Two Evaluation Approaches and Results

This document compares results from **run-eval-result.txt** (evaluations with `run_eval.py`), **tuning-results.text** (strategy sweep with `sweep_eval.py`), and **prediction-results.txt** (classification/prediction with `predict.py`), including training setup, decoding strategies (beam vs nucleus vs energy), and test/prediction metrics.

---

## 1. What Each File Contains

| Source | Script | Checkpoint(s) | What was varied |
|--------|--------|----------------|-----------------|
| **run-eval-result.txt** | `run_eval.py` | (1) glim-nucleus-gated-energy (2) glim-beam-gated-energy | One eval per checkpoint; default decoding = **beam** (num_beams from model, typically 2). Reports BLEU, ROUGE, ETES, corpus/relation/sentiment cls. |
| **tuning-results.text** | `sweep_eval.py` | glim-beam-gated-energy only | **Decoding** sweep: beam (1,2,4), nucleus (top_p 0.9, 0.95) on same checkpoint. |
| **prediction-results.txt** | `predict.py` | (1) glim-nucleus-gated-energy (2) glim-beam-gated-energy | One prediction run per checkpoint (`--task all --save_results --use_llm`). Reports **clip-like** accuracies (EEG, text raw, text generated) and **llm-pred** acc for Corpus, Relation, Sentiment. No BLEU/ROUGE/ETES. |

So you have:
- **Two training approaches**: nucleus-trained vs beam-trained (different checkpoints).
- **One decoding sweep**: multiple generation strategies on the beam-trained model only (tuning-results).
- **Prediction pipeline**: same two checkpoints run through `predict.py` for classification-style tasks and optional LLM-based predictions (prediction-results).

---

## 2. Training Approaches (from checkpoint names)

| Approach | Checkpoint | Training config (inferred) |
|----------|------------|----------------------------|
| **Nucleus-trained** | `glim-nucleus-gated-energy` | Gated + energy; generation_strategy = nucleus during training (top_p ~0.95, temp ~0.7) |
| **Beam-trained** | `glim-beam-gated-energy` | Gated + energy; generation_strategy = beam during training (e.g. num_beams=2) |

Both use the same data (`zuco_eeg_label_8variants.df`), `--use_energy`, and gated attention. Only the **generation strategy used at training time** differs.

---

## 3. Evaluation / Decoding Config

### 3.1 run-eval-result.txt (run_eval.py, default = beam)

- No `--generation_strategy` / `--num_beams` / `--top_p` in your commands → script default **beam** and model defaults (e.g. num_beams=2) are used.
- So both runs in this file are evaluated with **beam decoding** (same strategy, different checkpoints).

| Run | Checkpoint | Decoding (inferred) | BLEU-1 (gen) | BLEU-2 (gen) | ROUGE-1 | Corpus cls acc | Relation top-1 | Sentiment top-1 | ETES Total |
|-----|------------|----------------------|---------------|--------------|---------|----------------|----------------|-----------------|------------|
| 1 | glim-nucleus-gated-energy | beam (default) | **0.1827** | 0.0645 | 0.1621 | 0.5530 | 0.2347 | 0.3573 | **-0.0265** |
| 2 | glim-beam-gated-energy    | beam (default) | **0.2440** | 0.0864 | 0.2194 | 0.6916 | 0.0316 | 0.2950 | -0.3985 |

So with **same decoding (beam)**:
- **Beam-trained** checkpoint gives higher BLEU-1/ROUGE and corpus accuracy.
- **Nucleus-trained** checkpoint gives better (less negative) ETES (-0.0265 vs -0.3985) and better relation top-1 (0.2347 vs 0.0316).

### 3.2 tuning-results.text (sweep_eval.py on beam-trained checkpoint only)

All rows: checkpoint = **glim-beam-gated-energy**. Only **decoding** strategy and hyperparameters change.

| Decoding config | Quick score | BLEU1@MTV | Ret@1 | ETES |
|-----------------|-------------|-----------|-------|------|
| beam, num_beams=1 | **1.3759** | **0.2170** | 0.0763 | -0.0389 |
| beam, num_beams=2 | 1.2669 | 0.1827 | 0.0763 | -0.0265 |
| beam, num_beams=4 | 1.2212 | 0.1691 | 0.0763 | -0.0166 |
| nucleus, top_p=0.9, T=0.7 | 1.2212 | 0.1691 | 0.0763 | -0.0166 |
| nucleus, top_p=0.95, T=0.7 | 0.6577 | ~0.0011 | 0.0763 | 0.1024 |

Notes:
- **tuning-results.text** is incomplete: last run (nucleus top_p=0.95) was cut at 15/92 test batches; BLEU1@MTV and Quick score for that config may not be final.
- Ret@1 is constant (0.0763) across all sweep configs.
- Best **Quick score** and **BLEU1@MTV** on this sweep: **beam with num_beams=1**.
- Beam-2 in the sweep matches BLEU1@MTV 0.1827 and ETES -0.0265, consistent with run_eval’s beam default on the same checkpoint.

### 3.3 prediction-results.txt (predict.py, both checkpoints)

`predict.py` runs **Corpus**, **Relation**, and **Sentiment** tasks with `--task all --save_results --use_llm`. It reports **clip-like** accuracies (EEG-based, text raw, text generated) and **llm-pred** (LLM-based prediction). Same two checkpoints as run-eval, but no generation metrics (no BLEU/ROUGE/ETES).

**Summary table (from prediction-results.txt):**

| Task | Metric | Nucleus-trained | Beam-trained |
|------|--------|-----------------|--------------|
| **Corpus** | clip-like acc (EEG) | 0.5530 | **0.6916** |
| | clip-like (text raw) | **0.7034** | 0.5915 |
| | clip-like (text gen) | 0.2681 | **0.8062** |
| | llm-pred | 0.1952 | **0.9601** |
| **Relation** | clip-like acc (EEG) | **0.2347** | 0.0316 |
| | clip-like (text raw) | **0.2663** | 0.1643 |
| | clip-like (text gen) | **0.1194** | 0.0929 |
| | llm-pred | N/A | 0.10 |
| **Sentiment** | clip-like acc (EEG) | **0.3573** | 0.2950 |
| | clip-like (text raw) | **0.4772** | 0.4029 |
| | clip-like (text gen) | **0.4317** | 0.3118 |
| | llm-pred | 0.017 | **0.3046** |

Takeaways from prediction-results:
- **Beam-trained** wins on Corpus (EEG and text-gen acc, llm-pred) and on Sentiment llm-pred.
- **Nucleus-trained** wins on Relation (all clip-like accs) and on Sentiment clip-like (EEG, text raw, text gen).
- Corpus EEG acc (0.553 vs 0.692) and Relation EEG acc (0.235 vs 0.032) in prediction-results align with run-eval’s corpus/relation metrics for the same checkpoints.

---

## 4. Side-by-side: Training approach vs decoding

- **run-eval-result.txt** compares **training approach** (nucleus vs beam) with **fixed decoding** (beam); reports BLEU, ROUGE, ETES, classification.
- **prediction-results.txt** compares **training approach** (nucleus vs beam) via `predict.py`; reports clip-like (EEG, text raw, text gen) and llm-pred for Corpus, Relation, Sentiment (no BLEU/ROUGE/ETES).
- **tuning-results.text** compares **decoding** (beam 1/2/4, nucleus 0.9/0.95) with **fixed training** (beam-trained).

So:

| Question | Where to look |
|----------|----------------|
| Nucleus-trained vs beam-trained (same beam decoding) | run-eval-result.txt Run 1 vs Run 2 |
| Nucleus vs beam: classification/prediction (Corpus, Relation, Sentiment + LLM) | prediction-results.txt (both checkpoints, predict.py) |
| Best decoding on beam-trained model | tuning-results.text (best so far: beam-1) |
| Beam vs nucleus vs energy decoding | tuning-results.text has beam and nucleus; energy configs were not completed (all energy_rerank_candidates = None) |

---

## 5. Summary

- **Training**
  - Two setups: **nucleus-trained** vs **beam-trained** (gated + energy, same data).
- **Testing / prediction**
  - **run-eval-result.txt**: Two checkpoints, both evaluated with **beam** decoding. Beam-trained wins on BLEU/ROUGE/corpus accuracy; nucleus-trained wins on ETES and relation top-1.
  - **prediction-results.txt**: Same two checkpoints via `predict.py`. Beam-trained wins on Corpus (EEG, text-gen, llm-pred) and Sentiment llm-pred; nucleus-trained wins on Relation (all clip-like) and Sentiment clip-like. Corpus/Relation EEG accs match run-eval trends.
  - **tuning-results.text**: On the **beam-trained** checkpoint, **beam with num_beams=1** gives the best Quick score and BLEU1@MTV in the sweep; beam-2 matches the run_eval beam results. Nucleus top_p=0.95 run is incomplete.
- **Energy decoding**
  - Sweep was intended to include beam, nucleus, and energy, but no rows with `energy_rerank_candidates` set appear; energy strategy was not compared in these logs.

---

## 6. Quick reference: main metrics

| Metric | Nucleus-trained (beam decode) | Beam-trained (beam decode) | Beam-trained best decode (beam-1) |
|--------|--------------------------------|----------------------------|-----------------------------------|
| BLEU-1 (gen) | 0.1827 | 0.2440 | 0.2170 |
| BLEU-2 (gen) | 0.0645 | 0.0864 | — |
| ROUGE-1 | 0.1621 | 0.2194 | — |
| Corpus cls acc | 0.553 | 0.692 | — |
| Relation cls top-1 | 0.235 | 0.032 | 0.076 |
| Sentiment cls top-1 | 0.357 | 0.295 | — |
| ETES Total | **-0.027** | -0.399 | -0.039 |
| Quick score | — | — | **1.376** |

So: **beam-trained + beam decoding** gives best BLEU/ROUGE/corpus accuracy; **nucleus-trained + beam decoding** gives best ETES and relation top-1; **beam-trained + beam-1** gives best Quick score in the sweep.

**Prediction (predict.py) — from prediction-results.txt:**

| Metric | Nucleus-trained | Beam-trained |
|--------|-----------------|--------------|
| Corpus clip-like (EEG) | 0.553 | **0.692** |
| Relation clip-like (EEG) | **0.235** | 0.032 |
| Sentiment clip-like (EEG) | **0.357** | 0.295 |
| Corpus llm-pred | 0.195 | **0.960** |
| Sentiment llm-pred | 0.017 | **0.305** |

So: beam-trained leads on Corpus and LLM-based prediction; nucleus-trained leads on Relation and on clip-like Sentiment/EEG metrics.
