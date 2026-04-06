"""
Zero-Shot Linear Probe on Benchmarking Subjects.

This script trains a linear probe on the original ZuCo dataset representations (frozen pretrained encoder),
and evaluates how well it transfers to the held-out Benchmarking subjects (X-cohort).

Tasks evaluated:
  1. Reading Paradigm Classification (NR vs TSR)
  2. Relation Classification (e.g., nationality, employer, etc.)

Note: We don't evaluate Subject ID here because the Benchmarking subjects 
are completely disjoint from the training subjects.

Usage:
    cd pretraining
    python evaluate_benchmarking_probe.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from collections import Counter

sys.path.insert(0, '..')
sys.path.insert(0, '.')
from pretraining.pretrain_glim_encoder import GLIMEncoderPretrainer


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
    df = pd.read_pickle(data_path)
    
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

    # Feature Scaling (Crucial for Linear probes on embeddings)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    bench_X = scaler.transform(bench_X)

    # Train Head
    print("   Training Logistic Regression Head...")
    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    clf.fit(train_X, train_y)

    # Predict
    train_acc = accuracy_score(train_y, clf.predict(train_X))
    test_acc = accuracy_score(test_y, clf.predict(test_X))
    bench_acc = accuracy_score(bench_y, clf.predict(bench_X))

    print(f"   ✅ [TRAIN] Accuracy: {100*train_acc:.1f}%")
    print(f"   ✅ [TEST]  Accuracy: {100*test_acc:.1f}%  (Baseline transfer)")
    print(f"   🚀 [BENCH] Accuracy: {100*bench_acc:.1f}%  (Zero-shot Subject Transfer)\n")
    
    if n_classes <= 20:
        print(f"   📋 Benchmarking Per-Class Report:")
        report = classification_report(bench_y, clf.predict(bench_X), 
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
    print("  ZERO-SHOT MULTI-PROBE ON BENCHMARKING DATA")
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
    print("  ✅ Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
