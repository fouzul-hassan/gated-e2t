"""
STEP4b: Match benchmarking EEG data with label table.

Loads zuco_eeg_benchmarking.df (from STEP3b) and matches each sentence's
'text' field against the label table to retrieve all labels, then sets
phase='benchmarking'.

Output: data/tmp/zuco_eeg_label_benchmarking.df
Schema: identical to zuco_eeg_label_8variants.df but with phase='benchmarking'

Run from the GLIM/data directory:
    python __STEP4b_benchmarking_eeg_text_match.py
"""
import numpy as np
import pandas as pd

EEG_PATH    = './tmp/zuco_eeg_benchmarking.df'
LABEL_PATH  = './tmp/zuco_label_8variants.df'
OUTPUT_PATH = './tmp/zuco_eeg_label_benchmarking.df'

# ── Load data ─────────────────────────────────────────────────
print("Loading dataframes...")
df          = pd.read_pickle(EEG_PATH)
label_table = pd.read_pickle(LABEL_PATH)

print(f"  EEG df:      {df.shape}   cols: {df.columns.tolist()}")
print(f"  Label table: {label_table.shape}")

# ── Check text coverage ───────────────────────────────────────
print("\n" + "=" * 60)
input_texts = set(label_table['input text'].tolist())
matched_mask = df['text'].isin(input_texts)
print(f"Text match rate: {matched_mask.sum()} / {len(df)} "
      f"({matched_mask.mean()*100:.1f}%)")
print(f"Unmatched rows: {(~matched_mask).sum()}")

if (~matched_mask).sum() > 0:
    print("Sample unmatched texts:")
    for t in df[~matched_mask]['text'].unique()[:5]:
        print(f"  '{str(t)[:80]}'")

# ── Build deduped label table: 1 row per text_uid (prefer ZuCo2) ─────
# Sort so ZuCo2 comes first (descending alphabetically: ZuCo2 > ZuCo1).
# drop_duplicates then keeps ZuCo2 where both exist.
print("\n" + "=" * 60)
print("Building text → label_id lookup (preferring ZuCo2 rows)...")

label_deduped = (label_table
                 .sort_values('dataset', ascending=False)   # ZuCo2 first
                 .drop_duplicates('input text', keep='first')
                 .reset_index(drop=True))

print(f"  Original label_table rows: {len(label_table)}")
print(f"  Deduped (1 row per unique text): {len(label_deduped)}")
print(f"  Dataset distribution in deduped:")
print(label_deduped.groupby(['dataset', 'task']).size().to_string())

text_to_label_id = label_deduped.set_index('input text')['text uid'].to_dict()

label_ids = df['text'].map(text_to_label_id)
df['label id'] = label_ids

n_missing = df['label id'].isna().sum()
print(f"label_id assigned: {(~df['label id'].isna()).sum()} / {len(df)}")
print(f"label_id missing:  {n_missing}")

# Drop rows with no label match
if n_missing > 0:
    print(f"Dropping {n_missing} rows with no label match...")
    df = df.dropna(subset=['label id']).copy()
    df['label id'] = df['label id'].astype(int)

# ── Merge with deduped label table (1 row per text_uid) ─────────────
print("\n" + "=" * 60)
print("Merging with deduped label table...")

# Keep EEG columns before merge
df_eeg = df.reindex(columns=['eeg', 'mask', 'subject', 'label id'])
df_merged = df_eeg.merge(
    label_deduped, left_on='label id', right_on='text uid', how='left')

print(f"  Merged shape: {df_merged.shape}  (expected: ({len(df_eeg)}, ...)")
assert df_merged.shape[0] == len(df_eeg), \
    f"Row explosion! {df_merged.shape[0]} != {len(df_eeg)}"
print(f"  ✅ No row explosion confirmed")
print(f"  Columns: {df_merged.columns.tolist()}")

# ── Set phase = 'benchmarking' ────────────────────────────────
df_merged['phase'] = 'benchmarking'

# ── Verify final schema ───────────────────────────────────────
print("\n" + "=" * 60)
print("Final schema:")
expected_cols = [
    'eeg', 'mask', 'subject', 'label id', 'raw text', 'dataset', 'task',
    'control', 'raw label', 'input text', 'text uid', 'sentiment label',
    'relation label', 'phase'
]
for col in expected_cols:
    present = col in df_merged.columns
    print(f"  {'✅' if present else '❌'} {col}")

# ── Stats breakdown ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Records by subject:")
print(df_merged.groupby('subject').size().to_string())

print("\nRecords by dataset/task (from label table):")
print(df_merged.groupby(['dataset', 'task']).size().to_string())

print("\nSentiment label distribution:")
print(df_merged['sentiment label'].value_counts().to_string())

print(f"\nphase: {df_merged['phase'].unique()}")

# ── Save ──────────────────────────────────────────────────────
pd.to_pickle(df_merged, OUTPUT_PATH)
print(f"\n✅ Saved {df_merged.shape[0]} records to {OUTPUT_PATH}")
print("\nSample row:")
row = df_merged.iloc[0]
print(f"  subject:         {row['subject']}")
print(f"  dataset/task:    {row['dataset']} / {row['task']}")
print(f"  text_uid:        {row['text uid']}")
print(f"  input text:      '{str(row['input text'])[:70]}'")
print(f"  sentiment label: {row['sentiment label']}")
print(f"  relation label:  {row['relation label']}")
print(f"  phase:           {row['phase']}")
print(f"  eeg shape:       {row['eeg'].shape}")
