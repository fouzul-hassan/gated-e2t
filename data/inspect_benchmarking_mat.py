"""
Second-pass inspection of ZuCo2 benchmarking .mat file.
Focuses on: sentID, rawData orientation, word-level rawEEG, text recovery.
Run from the GLIM/data directory:
    python inspect_benchmarking_mat.py
"""
import h5py
import numpy as np
import pandas as pd

MAT_PATH = r'./raw_data/ZuCo2_benchmarking/resultsXAH.mat'
LABEL_TABLE_PATH = r'./tmp/zuco_label_8variants.df'

f = h5py.File(MAT_PATH, 'r')
s = f['sentenceData']

# ── 1. sentID — decode as scalar ─────────────────────────────
print("=" * 60)
print("1. sentID (reading as nested ndarray [[730.]], first 10):")
sent_ids = []
for j in range(10):
    val = s['sentID'][j, 0]
    try:
        data = f[val][()]
        sid = int(data.flat[0])  # [[730.]] -> 730
        sent_ids.append(sid)
        print(f"  [{j}] {sid}")
    except Exception as e:
        print(f"  [{j}] ERROR: {e}")

# ── 2. rawData shape and orientation ─────────────────────────
print("\n" + "=" * 60)
print("2. rawData orientation check (first 3 sentences):")
for j in range(3):
    raw_ref = s['rawData'][j][0]
    raw = f[raw_ref][:]
    print(f"  rawData[{j}] shape: {raw.shape}  dtype: {raw.dtype}  finite: {np.all(np.isfinite(raw))}")

# Check last channel empty (like ZuCo2 training)
raw_ref0 = s['rawData'][0][0]
raw0 = f[raw_ref0][:]
print(f"\n  Last column (channel) all-zero? {not raw0[:, -1].any()}")
print(f"  Last row (timepoint) all-zero? {not raw0[-1, :].any()}")

# ── 3. word-level rawEEG ─────────────────────────────────────
print("\n" + "=" * 60)
print("3. word[0].rawEEG — check if refs or actual data:")
word_ref = s['word'][0][0]
word_obj = f[word_ref]
n_words = word_obj['rawEEG'].shape[0]
print(f"  n_words in sentence 0: {n_words}")
for wi in range(min(3, n_words)):
    try:
        raw_item = word_obj['rawEEG'][wi][0]
        # Is it a reference?
        if isinstance(raw_item, h5py.h5r.Reference):
            eeg = f[raw_item][:]
            print(f"  word[{wi}] rawEEG shape: {eeg.shape}  (via reference)")
        else:
            print(f"  word[{wi}] rawEEG direct value: {raw_item}")
    except Exception as e:
        print(f"  word[{wi}] ERROR: {e}")

# ── 4. Match sentID against label table ──────────────────────
print("\n" + "=" * 60)
print("4. Matching sentIDs against existing label table:")
try:
    label_table = pd.read_pickle(LABEL_TABLE_PATH)
    print(f"  Label table loaded: {label_table.shape}")
    print(f"  'text uid' range: {label_table['text uid'].min()} - {label_table['text uid'].max()}")

    # Get all sentIDs
    all_sent_ids = []
    for j in range(s['sentID'].shape[0]):
        val = s['sentID'][j, 0]
        try:
            data = f[val][()]
            all_sent_ids.append(int(data.flat[0]))  # [[730.]] -> 730
        except Exception as e:
            print(f"  WARNING sentID[{j}]: {e}")
            all_sent_ids.append(-1)

    unique_ids = sorted(set(all_sent_ids))
    print(f"\n  Benchmarking sentID range: {min(all_sent_ids)} - {max(all_sent_ids)}")
    print(f"  First 10 sentIDs: {all_sent_ids[:10]}")

    # Try matching
    matches = label_table[label_table['text uid'].isin(unique_ids)]
    print(f"\n  Matched {len(unique_ids)} unique sentIDs -> {matches['text uid'].nunique()} found in label table")
    if len(matches) > 0:
        print(f"  Tasks found: {matches['task'].unique().tolist()}")
        print(f"  Datasets found: {matches['dataset'].unique().tolist()}")
        print(f"  Example text: '{matches.iloc[0]['input text'][:80]}'")
except FileNotFoundError:
    print("  Label table not found - skipping match step")

# ── 5. mean_t1 shape (frequency feature, sentence level) ─────
print("\n" + "=" * 60)
print("5. Sentence-level frequency features (mean_t1):")
t1_ref = s['mean_t1'][0][0]
t1 = f[t1_ref][:]
print(f"  mean_t1[0] shape: {t1.shape}  (expected: channels x 1 or 1 x channels)")

f.close()
print("\n" + "=" * 60)
print("Done.")
