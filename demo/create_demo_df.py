"""
Create a curated demo dataset from the best-performing test samples.

Selects top-10 by BLEU-1 with task diversity, joins EEG arrays,
and pre-computes all text-based metrics (BLEU/ROUGE/WER).
Classification metrics are pre-computed if --checkpoint is provided.

Usage:
    cd demo
    python create_demo_df.py                              # text metrics only
    python create_demo_df.py --checkpoint v1              # + classification (uses auto-discovered ckpt)
    python create_demo_df.py --checkpoint v1 --checkpoint v2
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import numpy as np
import pandas as pd
from torchmetrics.functional.text import bleu_score, rouge_score, word_error_rate

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_PKL  = '../data/tmp/glim_gen_results.pkl'
EEG_LABEL_DF = '../data/tmp/zuco_eeg_label_8variants.df'
OUTPUT_PATH  = '../data/tmp/zuco_eeg_to_text_demo.df'
CACHE_DIR    = os.path.join(os.path.dirname(__file__), 'cache')
N_SAMPLES    = 10

VARIANT_KEYS = [
    'lexical simplification (v0)', 'lexical simplification (v1)',
    'semantic clarity (v0)',        'semantic clarity (v1)',
    'syntax simplification (v0)',   'syntax simplification (v1)',
    'naive rewritten',              'naive simplified',
]


# ── Metric helpers ────────────────────────────────────────────────────────────
def safe_bleu(gen: str, refs: list[str], n: int) -> float:
    try:
        return bleu_score([gen], [refs], n_gram=n).item()
    except Exception:
        return 0.0


def safe_rouge(gen: str, refs: list[str]) -> dict:
    try:
        d = rouge_score([gen], [refs], rouge_keys='rouge1')
        return {k: v.item() for k, v in d.items()}
    except Exception:
        return {'rouge1_fmeasure': 0.0, 'rouge1_precision': 0.0, 'rouge1_recall': 0.0}


def safe_wer(gen: str, ref: str) -> float:
    try:
        return word_error_rate([gen], [ref]).item()
    except Exception:
        return 1.0


def compute_text_metrics(gen: str, raw_input: str, variants: list[str]) -> dict:
    """Compute all generation metrics for one sample."""
    metrics = {}
    # @MTV  (multi-target variants as references)
    valid_variants = [v for v in variants if isinstance(v, str) and v.strip()]
    for n in [1, 2, 3, 4]:
        metrics[f'bleu{n}_mtv'] = safe_bleu(gen, valid_variants, n)
        metrics[f'bleu{n}_raw'] = safe_bleu(gen, [raw_input], n)
    r_mtv = safe_rouge(gen, valid_variants)
    r_raw = safe_rouge(gen, [raw_input])
    for k in ['fmeasure', 'precision', 'recall']:
        metrics[f'rouge1_{k}_mtv'] = r_mtv[f'rouge1_{k}']
        metrics[f'rouge1_{k}_raw'] = r_raw[f'rouge1_{k}']
    metrics['wer'] = safe_wer(gen, raw_input)
    return metrics


# ── Discovery / selection ─────────────────────────────────────────────────────
def discover_checkpoint(version: str) -> str | None:
    pattern = os.path.join('..', 'runs', version, '*.ckpt')
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def select_diverse_samples(results_df: pd.DataFrame, n: int = N_SAMPLES) -> pd.DataFrame:
    """Select top-BLEU samples with task diversity."""
    results_df = results_df.copy()
    results_df['has_sentiment'] = results_df['sentiment label'].apply(
        lambda x: str(x).strip() not in ('nan', 'None', ''))
    results_df['has_relation'] = results_df['relation label'].apply(
        lambda x: str(x).strip() not in ('nan', 'None', ''))

    # Quota per group: NR-sentiment, TSR-relation, NR-plain
    groups = [
        ('NR+sentiment', results_df[results_df['has_sentiment']]),
        ('TSR+relation', results_df[results_df['has_relation']]),
        ('NR-plain',     results_df[~results_df['has_sentiment'] & ~results_df['has_relation']]),
    ]

    selected, seen_texts = [], set()
    quota_each = max(2, n // len(groups))

    for group_name, grp in groups:
        grp_sorted = grp.sort_values('bleu1_raw', ascending=False)
        count = 0
        for _, row in grp_sorted.iterrows():
            txt = str(row['raw input text']).strip()
            if txt not in seen_texts:
                seen_texts.add(txt)
                selected.append(row)
                count += 1
            if count >= quota_each or len(selected) >= n:
                break

    # Fill remaining slots from global top
    remaining = results_df.sort_values('bleu1_raw', ascending=False)
    for _, row in remaining.iterrows():
        if len(selected) >= n:
            break
        txt = str(row['raw input text']).strip()
        if txt not in seen_texts:
            seen_texts.add(txt)
            selected.append(row)

    return pd.DataFrame(selected[:n]).reset_index(drop=True)


# ── Optional: pre-compute classification metrics using GLIM model ─────────────
def precompute_classification(demo_df: pd.DataFrame, ckpt_path: str, version: str):
    """Run GLIM model on demo samples and cache classification results."""
    from model.glim import GLIM
    from data.datamodule import GLIMDataModule

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Loading {version} checkpoint: {ckpt_path} on {device}")

    model = GLIM.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False, weights_only=False)
    model.setup(stage='test')
    model.eval().to(device)
    tokenizer = model.tokenizer

    # Candidate embeddings
    sentiment_labels = ['negative', 'neutral', 'positive']
    relation_labels  = ['awarding', 'education', 'employment', 'foundation',
                        'job title', 'nationality', 'political affiliation', 'visit', 'marriage']
    corpus_labels    = ["The topic is about: movie, good or bad",
                        "The topic is about: life experiences, relationship"]
    paradigm_labels  = ["Normal reading: the participant reads sentences for general comprehension",
                        "Task-specific reading: the participant reads sentences to extract semantic relations"]

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        se_embs = model.cal_label_embs(sentiment_labels, "Sentiment classification: It is <MASK>.")
        re_embs = model.cal_label_embs(relation_labels,  "Relation classification: It is about <MASK>.")
        co_embs = model.cal_label_embs(corpus_labels)

        paradigm_ids, paradigm_mask = model.tokenize(paradigm_labels, 64)
        paradigm_hid, paradigm_hmask = model.encode_text(paradigm_ids, paradigm_mask)
        paradigm_embs = model.aligner.embed_text(paradigm_hid, paradigm_hmask)

    cache = {}
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for idx, row in demo_df.iterrows():
            eeg  = torch.from_numpy(row['eeg']).unsqueeze(0).to(device)   # (1, T, C)
            mask = torch.from_numpy(row['mask']).unsqueeze(0).to(device)  # (1, T)
            task = row.get('task', 'task1')
            subj = row.get('subject', '<UNK>')
            dataset = row.get('dataset', '<UNK>')
            t_token = '<NR>' if task != 'task3' else '<TSR>'
            prompts = ([t_token], [dataset], [subj])

            # Encode EEG
            prompt_ids   = model.p_embedder.encode(prompts, device=device)
            prompt_embed = model.p_embedder(prompt_ids, model.eval_pembed)
            eeg_hiddens, _ = model.eeg_encoder(eeg, mask, prompt_embed)
            eeg_embeds, eeg_emb = model.aligner.embed_eeg(eeg_hiddens)
            if eeg_emb.dim() == 1: eeg_emb = eeg_emb.unsqueeze(0)

            def cls_probs(embs):
                eeg_n = eeg_emb / eeg_emb.norm(dim=1, keepdim=True)
                cand_n = embs / embs.norm(dim=1, keepdim=True)
                return (eeg_n @ cand_n.T).softmax(dim=-1).squeeze(0).cpu().tolist()

            # Reading paradigm: use <UNK> task prompt
            unk_prompts  = (['<UNK>'], [dataset], [subj])
            unk_pids     = model.p_embedder.encode(unk_prompts, device=device)
            unk_pembed   = model.p_embedder(unk_pids, model.eval_pembed)
            unk_hid, _   = model.eeg_encoder(eeg, mask, unk_pembed)
            _, unk_emb   = model.aligner.embed_eeg(unk_hid)
            if unk_emb.dim() == 1: unk_emb = unk_emb.unsqueeze(0)

            def paradigm_probs():
                eeg_n  = unk_emb / unk_emb.norm(dim=1, keepdim=True)
                cand_n = paradigm_embs / paradigm_embs.norm(dim=1, keepdim=True)
                return (eeg_n @ cand_n.T).softmax(dim=-1).squeeze(0).cpu().tolist()

            # Generate text (greedy, fast)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                gen_ids = model.text_model.generate(
                    encoder_outputs=__import__('transformers').modeling_outputs.BaseModelOutput(eeg_embeds),
                    do_sample=False, num_beams=2, min_length=0, max_length=model.tgt_text_len)
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            result = {
                'gen_text':           gen_text,
                'sentiment_probs':    cls_probs(se_embs),
                'sentiment_labels':   sentiment_labels,
                'relation_probs':     cls_probs(re_embs),
                'relation_labels':    relation_labels,
                'corpus_probs':       cls_probs(co_embs),
                'corpus_labels':      ['movie review', 'biography'],
                'paradigm_probs':     paradigm_probs(),
                'paradigm_labels':    ['NR', 'TSR'],
            }
            # Add text metrics with live gen text
            variants = [row.get(k, '') for k in VARIANT_KEYS]
            result['text_metrics'] = compute_text_metrics(gen_text, str(row['input text']), variants)
            cache[str(idx)] = result
            print(f"  [{idx+1}/{len(demo_df)}] gen: {gen_text[:60]}...")

    cache_path = os.path.join(CACHE_DIR, f'precomputed_{version}.json')
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"  Saved cache → {cache_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='*', default=[],
                        help='Checkpoint versions to pre-compute (e.g. v1 v2)')
    parser.add_argument('--n', type=int, default=N_SAMPLES,
                        help='Number of demo samples')
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Step 1: Load generation results
    print("Step 1: Loading generation results...")
    results_df = pd.read_pickle(RESULTS_PKL)
    print(f"  {len(results_df)} samples  |  columns: {list(results_df.columns)}")

    # Step 2: Compute BLEU-1 per sample
    print("\nStep 2: Computing BLEU-1 scores...")
    scores = [safe_bleu(str(r['gen text']), [str(r['raw input text'])], 1)
              for _, r in results_df.iterrows()]
    results_df['bleu1_raw'] = scores
    print(f"  Range: {min(scores):.4f} – {max(scores):.4f}  |  Mean: {sum(scores)/len(scores):.4f}")

    # Step 3: Select diverse samples
    print(f"\nStep 3: Selecting top-{args.n} diverse samples...")
    selected = select_diverse_samples(results_df, args.n)
    for i, row in selected.iterrows():
        tag = f"sent={row['sentiment label']}" if row['has_sentiment'] else f"rel={row['relation label']}"
        print(f"  [{i+1:02d}] BLEU={row['bleu1_raw']:.3f} | {tag}")
        print(f"        {str(row['raw input text'])[:75]}...")

    # Step 4: Load EEG df and join
    print("\nStep 4: Loading EEG label df (may take ~60s)...")
    eeg_df   = pd.read_pickle(EEG_LABEL_DF)
    test_df  = eeg_df[eeg_df['phase'] == 'test']
    print(f"  Test rows: {len(test_df)}")

    demo_rows = []
    for _, sel in selected.iterrows():
        txt     = str(sel['raw input text']).strip()
        matches = test_df[test_df['input text'].str.strip() == txt]
        if len(matches) == 0:
            print(f"  WARNING: No EEG match for: {txt[:60]}...")
            continue
        eeg_row = matches.iloc[0].copy()
        # Attach columns from generation results
        eeg_row['gen text']  = sel['gen text']
        eeg_row['bleu1_raw'] = sel['bleu1_raw']
        demo_rows.append(eeg_row)

    demo_df = pd.DataFrame(demo_rows).reset_index(drop=True)
    print(f"  Matched {len(demo_df)} / {len(selected)} samples with EEG")

    # Step 5: Pre-compute text metrics
    print("\nStep 5: Computing text metrics (offline)...")
    text_metrics_all = []
    for _, row in demo_df.iterrows():
        gen      = str(row.get('gen text', ''))
        raw_inp  = str(row.get('input text', ''))
        variants = [str(row.get(k, '')) for k in VARIANT_KEYS]
        text_metrics_all.append(compute_text_metrics(gen, raw_inp, variants))

    demo_df['text_metrics'] = text_metrics_all

    # Step 6: Save demo df
    demo_df.to_pickle(OUTPUT_PATH)
    print(f"\nSaved demo df → {OUTPUT_PATH}  ({len(demo_df)} samples)")

    # Step 7 (optional): Pre-compute classification metrics per checkpoint
    for version in (args.checkpoint or []):
        ckpt = discover_checkpoint(version)
        if ckpt is None:
            print(f"\nWARNING: No checkpoint found for {version}")
            continue
        print(f"\nStep 7: Pre-computing classification for {version} ...")
        precompute_classification(demo_df, ckpt, version)

    print("\n✅ Done! Run `python app.py` to launch the demo.")


if __name__ == '__main__':
    main()
