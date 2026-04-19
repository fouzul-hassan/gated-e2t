"""
Zero-Shot SVM Probe on Benchmarking Subjects.

This script trains a Support Vector Machine (linear kernel) on the original ZuCo dataset 
representations (frozen pretrained encoder), and evaluates how well it transfers to 
the held-out Benchmarking subjects (X-cohort).

Tasks evaluated:
  1. Reading Paradigm Classification (NR vs TSR)
  2. Relation Classification (e.g., nationality, employer, etc.)

Note: Evaluates exactly like the official zuco-benchmark codebase.

Usage:
    cd pretraining
    python evaluate_benchmarking_probe_svm.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import h5py
from glob import glob
from scipy import signal
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from collections import Counter

sys.path.insert(0, '..')
sys.path.insert(0, '.')
from pretraining.pretrain_glim_encoder import GLIMEncoderPretrainer

SCRIPT_DIR = os.path.dirname(__file__)
RAW_BENCH_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'raw_data', 'ZuCo2_benchmarking')
RAW_TASK_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'raw_data', 'ZuCo2', 'task_materials')
LABEL_TABLE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'tmp', 'zuco_label_8variants.df')
BENCH_CACHE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'tmp', 'zuco_eeg_label_benchmarking_compat.df')

SUBJECT_KEYS = ['XAH', 'XBB', 'XBD', 'XDT', 'XLS', 'XPB', 'XSE', 'XSS', 'XTR', 'XWS']
SRC_SAMPLE_RATE = 500
TGT_SAMPLE_RATE = 128
TGT_MAX_LEN = 1280
TGT_WIDTH = 128
SRC_CHANNELS = 105
MIN_LEN = int(0.5 * SRC_SAMPLE_RATE)
MAX_LEN = int(10 * SRC_SAMPLE_RATE)

TYPOS = {
    "emp11111ty": "empty", "film.1": "film.", "–": "-", "\u2018s": "'s",
    "\ufffd s": "'s", "`s": "'s", "Maria": "Mari\u0107",
    "1Universidad": "Universidad", "1902\u201419": "1902 - 19",
    "Wuerttemberg": "W\u00fcrttemberg", "long -time": "long-time",
    "Jose": "Jos\u00e9", "Bucher": "B\u00f4cher", "1839 ? May": "1839 - May",
    "G\ufffd n\ufffd ration": "Generation", "Bragan\u00e7a": "Bragana",
    "1837?October": "1837 - October", "nVera-Ellen": "Vera-Ellen",
    "write Ethics": "wrote Ethics", "Adams-Onis": "Adams-On\u00eds",
    "(40 km?)": "(40 km\u00b2)", "(40 km\u02dd)": "(40 km\u00b2)",
    " (IPA: /?g?nz?b?/ ) ": " ", '""Canes""': '"Canes"',
    "111Senator": "Senator", "Creteil": "Cr\u00e9teil",
    "Zoonomia": "Zo\u00f6nomia", "nee Darwin": "n\u00e9e Darwin",
    "Ruthy": "R\u00e9thy", "Eidgenoessische": "Eidgen\u00f6ssische",
    "40 km\ufffd": "40 km\u00b2", "King Leopold": "King L\u00e9opold",
}


def revise_typo(text: str):
    if not isinstance(text, str):
        return text
    for src, tgt in TYPOS.items():
        if src in text:
            text = text.replace(src, tgt)
    return text


def load_zuco2_ordered_sentences(task_dir: str) -> list[str]:
    def load_sentences_from_csvs(prefix, n_files=7):
        sentences = []
        for i in range(1, n_files + 1):
            fpath = os.path.join(task_dir, f'{prefix}_{i}.csv')
            df = pd.read_csv(fpath, sep=';', encoding='utf-8', header=None,
                             names=['paragraph_id', 'sentence_id', 'sentence', 'control'],
                             dtype=str)
            sentences.extend(df['sentence'].tolist())
        return sentences

    nr_sentences = load_sentences_from_csvs('nr')
    tsr_sentences = load_sentences_from_csvs('tsr')
    return [None] + nr_sentences + tsr_sentences


def build_benchmark_dataframe() -> pd.DataFrame:
    print("   [fallback] Rebuilding benchmarking dataframe from raw .mat files...")
    ordered_sentences = load_zuco2_ordered_sentences(RAW_TASK_DIR)
    mat_paths = sorted(glob(os.path.join(RAW_BENCH_DIR, 'results*.mat')))

    records = []
    for mat_path in mat_paths:
        subject_key = os.path.basename(mat_path).replace('results', '').replace('.mat', '')
        if subject_key not in SUBJECT_KEYS:
            continue

        with h5py.File(mat_path, 'r') as mat:
            s = mat['sentenceData']
            n = s['rawData'].shape[0]
            for j in range(n):
                raw_ref = s['rawData'][j][0]
                eeg_raw = mat[raw_ref][:].astype(np.float32)
                if not np.all(np.isfinite(eeg_raw)):
                    continue

                T, ch = eeg_raw.shape
                if ch != SRC_CHANNELS or not (MIN_LEN < T <= MAX_LEN):
                    continue

                sid_ref = s['sentID'][j][0]
                sid_data = mat[sid_ref][()]
                sent_id = int(sid_data.flat[0])
                if sent_id < 1 or sent_id >= len(ordered_sentences):
                    continue

                text = revise_typo(ordered_sentences[sent_id])

                T_new = int(T * TGT_SAMPLE_RATE / SRC_SAMPLE_RATE)
                eeg_ds = signal.resample(eeg_raw, T_new, axis=0)
                eeg_padded = np.pad(eeg_ds, ((0, 0), (0, TGT_WIDTH - SRC_CHANNELS)), 'constant', constant_values=0)
                eeg_final = np.pad(eeg_padded, ((0, TGT_MAX_LEN - T_new), (0, 0)), 'constant', constant_values=0)
                mask = np.zeros(TGT_MAX_LEN, dtype=np.int8)
                mask[:T_new] = 1

                records.append({
                    'eeg': eeg_final,
                    'mask': mask,
                    'text': text,
                    'dataset': 'ZuCo2_benchmarking',
                    'task': 'unknown',
                    'subject': subject_key,
                })

    df = pd.DataFrame(records)
    print(f"   [fallback] Rebuilt benchmarking rows: {len(df)}")

    label_table = pd.read_pickle(LABEL_TABLE_PATH)
    label_deduped = (label_table
                     .sort_values('dataset', ascending=False)
                     .drop_duplicates('input text', keep='first')
                     .reset_index(drop=True))
    text_to_label_id = label_deduped.set_index('input text')['text uid'].to_dict()
    df['label id'] = df['text'].map(text_to_label_id)
    df = df.dropna(subset=['label id']).copy()
    df['label id'] = df['label id'].astype(int)

    df_eeg = df.reindex(columns=['eeg', 'mask', 'subject', 'label id'])
    df_final = df_eeg.merge(label_deduped, left_on='label id', right_on='text uid', how='left')
    df_final['phase'] = 'benchmarking'

    try:
        pd.to_pickle(df_final, BENCH_CACHE_PATH)
        print(f"   [fallback] Cached compatible benchmark dataframe -> {BENCH_CACHE_PATH}")
    except Exception as cache_exc:
        print(f"   [fallback] Cache save skipped: {cache_exc}")

    return df_final


# ── Dataset ──────────────────────────────────────────────────────────────────
class ZuCoMultiLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        eeg = row['eeg']             # (128, 1280) or (1280, 128)
        if eeg.shape[0] == 128:
            eeg = eeg.T              # → (1280, 128)
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)

        subject = str(row.get('subject', 'unknown'))
        sentiment = str(row.get('sentiment label', 'nan'))
        relation = str(row.get('relation label', 'nan'))
        task = str(row.get('task', ''))
        paradigm = 'NR' if ('task1' in task or 'task2' in task) else ('TSR' if 'task3' in task else 'nan')

        return eeg_tensor, subject, sentiment, relation, paradigm


def load_dataset(data_path: str, is_main: bool = True):
    """Load dataset. If main, returns train/test splits. If benchmarking, returns all data."""
    print(f"📂 Loading data from {data_path}")
    if is_main:
        df = pd.read_pickle(data_path)
    else:
        if os.path.exists(BENCH_CACHE_PATH):
            try:
                print(f"   Using compatible cache: {BENCH_CACHE_PATH}")
                df = pd.read_pickle(BENCH_CACHE_PATH)
            except Exception:
                df = build_benchmark_dataframe()
        else:
            try:
                df = pd.read_pickle(data_path)
            except Exception as exc:
                print(f"   [WARN] Could not read benchmark pickle ({type(exc).__name__}: {exc})")
                df = build_benchmark_dataframe()
    
    if is_main:
        train_df = df[df['phase'] == 'train']
        test_df = df[df['phase'] == 'test']
        print(f"   Train samples: {len(train_df)} | Test samples: {len(test_df)}")
        return ZuCoMultiLabelDataset(train_df), ZuCoMultiLabelDataset(test_df)
    else:
        # Benchmarking data
        print(f"   Benchmarking samples: {len(df)}")
        return ZuCoMultiLabelDataset(df)


# ── Feature Extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(model, loader, device):
    """Extract features + all label types from the frozen encoder."""
    model.eval()
    all_features = []
    all_relations = []
    all_paradigms = []

    for batch in loader:
        eeg, _, _, relations, paradigms = batch
        eeg = eeg.to(device)
        features = model(eeg)  # (B, emb_size)
        all_features.append(features.cpu())
        all_relations.extend(relations)
        all_paradigms.extend(paradigms)

    features = torch.cat(all_features, dim=0).numpy()
    return features, all_relations, all_paradigms


# ── Linear Probe ─────────────────────────────────────────────────────────────
def train_and_eval_probe(train_feat, train_labels, 
                         test_feat, test_labels, 
                         bench_feat, bench_labels, task_name):
    # Filter out valid labels (no 'nan')
    train_valid = [(f, l) for f, l in zip(train_feat, train_labels) if l not in ('nan', 'None', '')]
    test_valid = [(f, l) for f, l in zip(test_feat, test_labels) if l not in ('nan', 'None', '')]
    bench_valid = [(f, l) for f, l in zip(bench_feat, bench_labels) if l not in ('nan', 'None', '')]

    if not train_valid or not bench_valid:
        print(f"\n⚠️ Missing data for {task_name}, skipping.")
        return

    train_X, train_y_raw = zip(*train_valid)
    test_X, test_y_raw = zip(*test_valid)
    bench_X, bench_y_raw = zip(*bench_valid)

    # Convert to ndarray
    train_X, test_X, bench_X = np.stack(train_X), np.stack(test_X), np.stack(bench_X)

    # Encoding
    le = LabelEncoder()
    le.fit(list(train_y_raw) + list(test_y_raw) + list(bench_y_raw))
    train_y = le.transform(train_y_raw)
    test_y = le.transform(test_y_raw)
    bench_y = le.transform(bench_y_raw)

    n_classes = len(le.classes_)
    random_baseline = 1.0 / n_classes

    print(f"\n{'─'*60}")
    print(f"📊 {task_name}")
    print(f"{'─'*60}")
    print(f"   Classes ({n_classes}): {list(le.classes_)}")
    print(f"   Random baseline: {100*random_baseline:.1f}%\n")

    # Feature Scaling (Crucial for SVM / Linear probes on embeddings)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    bench_X = scaler.transform(bench_X)

    # Train Head with Tuning
    print("   Tuning SVM Hyperparameters (3-Fold CV)...")
    param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}
    
    # class_weight='balanced' perfectly handles the class imbalance!
    base_clf = SVC(kernel='linear', class_weight='balanced', random_state=42)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    clf = GridSearchCV(
        base_clf,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=1,
        pre_dispatch=1,
        error_score='raise',
    )
    clf.fit(train_X, train_y)

    print(f"   ⭐ Best parameters: {clf.best_params_}")
    best_clf = clf.best_estimator_

    # Predict
    train_acc = accuracy_score(train_y, best_clf.predict(train_X))
    test_acc = accuracy_score(test_y, best_clf.predict(test_X))
    bench_acc = accuracy_score(bench_y, best_clf.predict(bench_X))

    print(f"   ✅ [TRAIN] Accuracy: {100*train_acc:.1f}%")
    print(f"   ✅ [TEST]  Accuracy: {100*test_acc:.1f}%  (Baseline transfer)")
    print(f"   🚀 [BENCH] Accuracy: {100*bench_acc:.1f}%  (Zero-shot Subject Transfer)\n")
    
    if n_classes <= 20:
        print(f"   📋 Benchmarking Per-Class Report:")
        report = classification_report(bench_y, best_clf.predict(bench_X), 
                                       labels=range(n_classes),
                                       target_names=le.classes_, digits=3, zero_division=0)
        for line in report.split('\n'):
            print(f"   {line}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='Results/GLIM_Pretrain1/best_model.pth')
    parser.add_argument('--main_data', type=str, default='../data/tmp/zuco_eeg_label_8variants.df')
    parser.add_argument('--bench_data', type=str, default='../data/tmp/zuco_eeg_label_benchmarking.df')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("  ZERO-SHOT SVM PROBE ON BENCHMARKING DATA")
    print("=" * 60)

    # ── Load checkpoint ──
    print(f"\n📦 Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    ckpt_args = ckpt.get('args', {})

    pos_embed_shape = ckpt['model_state_dict']['pos_embed'].shape
    actual_patch_size = 1280 // pos_embed_shape[1]

    model = GLIMEncoderPretrainer(
        in_len=1280,
        in_dim=128,
        emb_size=ckpt_args.get('emb_size', 128),
        n_blocks=ckpt_args.get('n_blocks', 6),
        num_heads=ckpt_args.get('num_heads', 8),
        patch_size=actual_patch_size,
        mask_ratio=ckpt_args.get('mask_ratio', 0.5),
        momentum=ckpt_args.get('momentum', 0.99),
        use_gated_attention=ckpt_args.get('use_gated_attention', False),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("   ✅ Pretrained Model loaded")

    # ── Load data ──
    main_train, main_test = load_dataset(args.main_data, is_main=True)
    bench_dataset = load_dataset(args.bench_data, is_main=False)

    train_loader = DataLoader(main_train, batch_size=64, shuffle=False)
    test_loader = DataLoader(main_test, batch_size=64, shuffle=False)
    bench_loader = DataLoader(bench_dataset, batch_size=64, shuffle=False)

    # ── Extract features ──
    print("\n🔍 Extracting features from frozen encoder (Main Train)...")
    train_feat, train_rel, train_para = extract_features(model, train_loader, device)

    print("🔍 Extracting features from frozen encoder (Main Test)...")
    test_feat, test_rel, test_para = extract_features(model, test_loader, device)

    print("🔍 Extracting features from frozen encoder (Benchmarking)...")
    bench_feat, bench_rel, bench_para = extract_features(model, bench_loader, device)

    # ── Probes ──
    train_and_eval_probe(train_feat, train_para, 
                         test_feat, test_para, 
                         bench_feat, bench_para, 
                         task_name="Reading Paradigm (NR vs TSR)")

    train_and_eval_probe(train_feat, train_rel, 
                         test_feat, test_rel, 
                         bench_feat, bench_rel, 
                         task_name="Relation Classification")

    print("\n" + "=" * 60)
    print("  ✅ SVM Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
