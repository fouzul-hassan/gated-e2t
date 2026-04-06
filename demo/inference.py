"""
Live inference helpers for the GLIM demo.

- Auto-discover checkpoints from ./runs/v1/ and ./runs/v2/
- Load model (singleton per version, cached in memory)
- Run inference on one demo sample: generate text + all metrics
- Disk-based JSON cache so repeated clicks are instant
"""
import sys, os, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torchmetrics.functional.text import bleu_score, rouge_score, word_error_rate
from transformers.modeling_outputs import BaseModelOutput

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

VARIANT_KEYS = [
    'lexical simplification (v0)', 'lexical simplification (v1)',
    'semantic clarity (v0)',        'semantic clarity (v1)',
    'syntax simplification (v0)',   'syntax simplification (v1)',
    'naive rewritten',              'naive simplified',
]

SENTIMENT_LABELS  = ['negative', 'neutral', 'positive']
RELATION_LABELS   = ['awarding', 'education', 'employment', 'foundation',
                     'job title', 'nationality', 'political affiliation', 'visit', 'marriage']
CORPUS_LABELS     = ["The topic is about: movie, good or bad",
                     "The topic is about: life experiences, relationship"]
CORPUS_DISPLAY    = ['movie review', 'biography']
PARADIGM_LABELS   = ["Normal reading: the participant reads sentences for general comprehension",
                     "Task-specific reading: the participant reads sentences to extract semantic relations"]
PARADIGM_DISPLAY  = ['NR', 'TSR']


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_all_checkpoints() -> dict[str, str]:
    """
    Return {version_label: ckpt_path} for all auto-discovered checkpoints.
    Scans ./runs/v1/, ./runs/v2/, ./runs/v3/, ... up to v9.
    Picks the most recently modified .ckpt per version directory.
    """
    base = os.path.join(os.path.dirname(__file__), '..', 'runs')
    found = {}
    for version_dir in sorted(glob.glob(os.path.join(base, 'v*'))):
        version = os.path.basename(version_dir)
        ckpts   = sorted(glob.glob(os.path.join(version_dir, '*.ckpt')),
                         key=os.path.getmtime)
        if ckpts:
            found[version] = ckpts[-1]   # most recently modified = best epoch
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Model singleton cache (stays loaded between Gradio calls)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: dict[str, object] = {}

def load_model(ckpt_path: str, device: torch.device):
    """Load GLIM model; cache in memory so it's only loaded once per checkpoint."""
    if ckpt_path in _model_cache:
        return _model_cache[ckpt_path]

    from model.glim import GLIM
    print(f"  Loading model from {ckpt_path} …")
    model = GLIM.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False, weights_only=False)
    model.setup(stage='test')
    model.eval().to(device)
    _model_cache[ckpt_path] = model
    print("  Model loaded ✓")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bleu(gen, refs, n):
    try:    return bleu_score([gen], [refs], n_gram=n).item()
    except: return 0.0

def _rouge(gen, refs):
    try:
        d = rouge_score([gen], [refs], rouge_keys='rouge1')
        return {k: v.item() for k, v in d.items()}
    except:
        return {'rouge1_fmeasure': 0.0, 'rouge1_precision': 0.0, 'rouge1_recall': 0.0}

def _wer(gen, ref):
    try:    return word_error_rate([gen], [ref]).item()
    except: return 1.0


def compute_all_text_metrics(gen: str, raw_input: str, variants: list[str]) -> dict:
    valid = [v for v in variants if isinstance(v, str) and v.strip()]
    m = {}
    for n in [1, 2, 3, 4]:
        m[f'bleu{n}_mtv'] = _bleu(gen, valid or [raw_input], n)
        m[f'bleu{n}_raw'] = _bleu(gen, [raw_input], n)
    for suffix, refs in [('mtv', valid or [raw_input]), ('raw', [raw_input])]:
        r = _rouge(gen, refs)
        for stat in ['fmeasure', 'precision', 'recall']:
            m[f'rouge1_{stat}_{suffix}'] = r[f'rouge1_{stat}']
    m['wer'] = _wer(gen, raw_input)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(sample_idx: int, row, ckpt_path: str,
                  version_label: str, device: torch.device) -> dict:
    """
    Run full inference on one demo sample.
    Results are disk-cached; subsequent calls are instant.

    Returns a rich dict with generated text + all metrics.
    """
    cache_key  = f"{version_label}_sample_{sample_idx}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    # ── Cache hit ──
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        cached['_from_cache'] = True
        return cached

    # ── Cache miss: run model ──
    model = load_model(ckpt_path, device)

    eeg_np  = row['eeg']   if isinstance(row['eeg'],  np.ndarray) else np.array(row['eeg'],  dtype=np.float32)
    mask_np = row['mask']  if isinstance(row['mask'], np.ndarray) else np.array(row['mask'], dtype=np.int8)
    if eeg_np.shape[0] == 128:   # (C,T) → (T,C)
        eeg_np = eeg_np.T

    eeg_t  = torch.from_numpy(eeg_np).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)

    task    = str(row.get('task',    'task1'))
    subj    = str(row.get('subject', '<UNK>'))
    dataset = str(row.get('dataset', '<UNK>'))
    t_token = '<NR>' if task != 'task3' else '<TSR>'

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # ── Encode EEG with task-aware prompt ──
        prompts      = ([t_token], [dataset], [subj])
        p_ids        = model.p_embedder.encode(prompts, device=device)
        p_embed      = model.p_embedder(p_ids, model.eval_pembed)
        eeg_hiddens, _ = model.eeg_encoder(eeg_t, mask_t, p_embed)
        eeg_embeds, eeg_emb = model.aligner.embed_eeg(eeg_hiddens)
        if eeg_emb.dim() == 1: eeg_emb = eeg_emb.unsqueeze(0)

        # ── Encode EEG with <UNK> task prompt (for reading paradigm) ──
        unk_prompts      = (['<UNK>'], [dataset], [subj])
        unk_pids         = model.p_embedder.encode(unk_prompts, device=device)
        unk_pembed       = model.p_embedder(unk_pids, model.eval_pembed)
        unk_hid, _       = model.eeg_encoder(eeg_t, mask_t, unk_pembed)
        _, unk_emb       = model.aligner.embed_eeg(unk_hid)
        if unk_emb.dim() == 1: unk_emb = unk_emb.unsqueeze(0)

        # ── Classification helper ──
        def cls_probs(cand_emb):
            e = eeg_emb / eeg_emb.norm(dim=1, keepdim=True)
            c = cand_emb / cand_emb.norm(dim=1, keepdim=True)
            return (e @ c.T).softmax(dim=-1).squeeze(0).cpu().tolist()

        def paradigm_probs_fn(cand_emb):
            e = unk_emb / unk_emb.norm(dim=1, keepdim=True)
            c = cand_emb / cand_emb.norm(dim=1, keepdim=True)
            return (e @ c.T).softmax(dim=-1).squeeze(0).cpu().tolist()

        # Pre-compute label embeddings
        se_embs = model.cal_label_embs(SENTIMENT_LABELS, "Sentiment classification: It is <MASK>.")
        re_embs = model.cal_label_embs(RELATION_LABELS,  "Relation classification: It is about <MASK>.")
        co_embs = model.cal_label_embs(CORPUS_LABELS)
        pa_embs = model.cal_label_embs(PARADIGM_LABELS)

        # ── Text generation (beam search) ──
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            gen_ids = model.text_model.generate(
                encoder_outputs=BaseModelOutput(eeg_embeds),
                num_beams=2, min_length=0, max_length=model.tgt_text_len)
        gen_text = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # ── Text metrics ──
    raw_input = str(row.get('input text', ''))
    variants  = [str(row.get(k, '')) for k in VARIANT_KEYS]
    text_metrics = compute_all_text_metrics(gen_text, raw_input, variants)

    # ── Build result dict ──
    result = {
        'gen_text':         gen_text,
        'text_metrics':     text_metrics,

        'sentiment_probs':  cls_probs(se_embs),
        'sentiment_labels': SENTIMENT_LABELS,

        'relation_probs':   cls_probs(re_embs),
        'relation_labels':  RELATION_LABELS,

        'corpus_probs':     cls_probs(co_embs),
        'corpus_labels':    CORPUS_DISPLAY,

        'paradigm_probs':   paradigm_probs_fn(pa_embs),
        'paradigm_labels':  PARADIGM_DISPLAY,

        '_from_cache': False,
        '_version':    version_label,
        '_ckpt':       ckpt_path,
    }

    # ── Save to disk cache ──
    with open(cache_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Cached → {cache_path}")

    return result
