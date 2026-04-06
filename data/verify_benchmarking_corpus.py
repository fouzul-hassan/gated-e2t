"""
Verify: which dataset/task do the benchmarking sentIDs map to?
Run from GLIM/data/:
    python verify_benchmarking_corpus.py
"""
import h5py
import numpy as np
import pandas as pd

MAT_PATH       = r'./raw_data/ZuCo2_benchmarking/resultsXAH.mat'
LABEL_TABLE_PATH = r'./tmp/zuco_label_8variants.df'

# ── Load label table ──────────────────────────────────────────
label_table = pd.read_pickle(LABEL_TABLE_PATH)
print(f"Label table: {label_table.shape}")
print(f"text uid range: {label_table['text uid'].min()} - {label_table['text uid'].max()}")
print(f"Label table columns: {label_table.columns.tolist()}\n")

# ── Get all sentIDs from XAH ──────────────────────────────────
f = h5py.File(MAT_PATH, 'r')
s = f['sentenceData']
n = s['rawData'].shape[0]

all_sent_ids = []
for j in range(n):
    val = s['sentID'][j][0]
    data = f[val][()]
    all_sent_ids.append(int(data.flat[0]))
f.close()

print(f"Total sentIDs in XAH: {len(all_sent_ids)}")
print(f"sentID range: {min(all_sent_ids)} - {max(all_sent_ids)}")
print(f"Unique sentIDs: {len(set(all_sent_ids))}\n")

# ── Match to label table ──────────────────────────────────────
# Each sentID should correspond to text uid in the label table
matched = label_table[label_table['text uid'].isin(all_sent_ids)]

print("=" * 60)
print("Matched rows — Dataset/Task breakdown:")
print(matched.groupby(['dataset', 'task']).size().to_string())

print("\n" + "=" * 60)
print("Unique text uids matched per dataset/task:")
unique_per_group = matched.drop_duplicates('text uid').groupby(['dataset', 'task']).size()
print(unique_per_group.to_string())

print("\n" + "=" * 60)
print("sentIDs NOT found in label table:")
matched_uids = set(matched['text uid'].tolist())
missing = [sid for sid in all_sent_ids if sid not in matched_uids]
print(f"  Count: {len(missing)}")
if missing:
    print(f"  Missing IDs: {missing[:20]}")

print("\n" + "=" * 60)
print("Sample matched rows (first 5):")
first5 = matched[matched['text uid'].isin(all_sent_ids[:5])][['text uid','dataset','task','input text']].drop_duplicates('text uid')
for _, row in first5.iterrows():
    print(f"  uid={row['text uid']} | {row['dataset']}/{row['task']} | '{row['input text'][:60]}...'")
