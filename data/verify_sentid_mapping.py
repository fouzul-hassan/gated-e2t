"""
Determine the correct sentID → sentence text mapping.

The benchmarking X-subjects read ZuCo2 sentences (349 NR + 390 TSR = 739).
Their sentIDs (1-739) are the original pre-shuffle sentence positions.
This script figures out the correct sentID → text mapping.

Run from GLIM/data/:
    python verify_sentid_mapping.py
"""
import h5py
import numpy as np
import pandas as pd

MAT_PATH         = r'./raw_data/ZuCo2_benchmarking/resultsXAH.mat'
ZUCO2_NR_DIR     = r'./raw_data/ZuCo2/task_materials'
LABEL_TABLE_PATH = r'./tmp/zuco_label_8variants.df'

# ── Step 1: Build ordered ZuCo2 sentence list from CSV task materials ─
print("=" * 60)
print("Step 1: Load ZuCo2 sentences in original order")

def load_zuco2_sentences(file_dir, prefix, n_files=7):
    """Load all sentences from ZuCo2 task materials in order."""
    import os
    sentences = []
    for i in range(1, n_files + 1):
        fpath = os.path.join(file_dir, f'{prefix}_{i}.csv')
        df = pd.read_csv(fpath, sep=';', encoding='utf-8', header=None,
                         names=['paragraph_id', 'sentence_id', 'sentence', 'control'],
                         dtype=str)
        sentences.extend(df['sentence'].tolist())
    return sentences

nr_sentences  = load_zuco2_sentences(ZUCO2_NR_DIR, 'nr')   # 349 sentences
tsr_sentences = load_zuco2_sentences(ZUCO2_NR_DIR, 'tsr')  # 390 sentences
print(f"  NR sentences loaded: {len(nr_sentences)}")
print(f"  TSR sentences loaded: {len(tsr_sentences)}")

# ── Step 2: Try different orderings of sentID → sentence ──────────────
print("\n" + "=" * 60)
print("Step 2: Get sentIDs and raw data from XAH.mat (first 10)")

f = h5py.File(MAT_PATH, 'r')
s = f['sentenceData']
n = s['rawData'].shape[0]

sent_ids = []
for j in range(n):
    val     = s['sentID'][j][0]
    data    = f[val][()]
    sid     = int(data.flat[0])
    sent_ids.append(sid)

print(f"  First 10 sentIDs: {sent_ids[:10]}")
print(f"  sentID range: {min(sent_ids)}-{max(sent_ids)}")
f.close()

# ── Step 3: Try ordering hypothesis A: NR first then TSR ─────────────
print("\n" + "=" * 60)
print("Step 3: Test Hypothesis A — sentID 1-349=NR, 350-739=TSR")
all_sentences_A = [None] + nr_sentences + tsr_sentences  # 1-indexed

# ── Step 4: Match sentIDs to sentence text and look up in label table ─
label_table = pd.read_pickle(LABEL_TABLE_PATH)

# Apply typo corrections (same as STEP4)
typobook = {"emp11111ty":"empty","film.1":"film.","–":"-","'s":"'s",
            "Maria":"Marić","1Universidad":"Universidad","1902—19":"1902 - 19",
            "Wuerttemberg":"Württemberg","long -time":"long-time",
            "Jose":"José","Bucher":"Bôcher","1839 ? May":"1839 - May",
            "Bragança":"Bragana","1837?October":"1837 - October",
            "nVera-Ellen":"Vera-Ellen","write Ethics":"wrote Ethics",
            "Adams-Onis":"Adams-Onís","111Senator":"Senator",
            "Creteil":"Créteil","Zoonomia":"Zoönomia","1902\ufffd19":"1902 - 19",
            "nee Darwin":"née Darwin","Ruthy":"Réthy",
            "Eidgenoessische":"Eidgenössische","40 km\ufffd":"40 km²",
            "King Leopold":"King Léopold"}

def revise_typo(text):
    if not isinstance(text, str):
        return text
    for src, tgt in typobook.items():
        if src in text:
            text = text.replace(src, tgt)
    return text

input_texts = set(label_table['input text'].tolist())

# Test hypothesis A: NR (1-349) then TSR (350-739)
matched_A = 0
unmatched_A = []
for sid in sent_ids[:50]:  # check first 50
    if 1 <= sid <= len(all_sentences_A) - 1:
        text = revise_typo(all_sentences_A[sid])
        if text in input_texts:
            matched_A += 1
        else:
            unmatched_A.append((sid, text[:60] if text else 'NONE'))

print(f"  Matched (first 50): {matched_A}/50")
if unmatched_A:
    print(f"  Unmatched examples:")
    for sid, t in unmatched_A[:5]:
        print(f"    sentID={sid}: '{t}'")

# ── Step 5: Show a few matched examples to verify correctness ─────────
print("\n" + "=" * 60)
print("Step 5: Show sample sentID → text → label table matches")
for sid in sent_ids[:5]:
    text = revise_typo(all_sentences_A[sid]) if 1 <= sid <= len(all_sentences_A)-1 else None
    match = label_table[label_table['input text'] == text]
    if len(match) > 0:
        row = match.iloc[0]
        print(f"  sentID={sid:3d} → '{text[:55]}...'")
        print(f"            dataset={row['dataset']}, task={row['task']}, text_uid={row['text uid']}")
    else:
        print(f"  sentID={sid:3d} → NO MATCH: '{str(text)[:55]}'")
