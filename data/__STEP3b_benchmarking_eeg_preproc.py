"""
STEP3b: Preprocess ZuCo2 Benchmarking EEG data into .df format.

These 10 held-out subjects (X-prefix) read the same 739 ZuCo2 sentences:
  - sentID 1-370   → ZuCo2 NR  (nr_1.csv ... nr_7.csv, 370 total incl. controls)
  - sentID 371-781 → ZuCo2 TSR (tsr_1.csv ... tsr_7.csv, 411 total incl. controls)
  sentID is the 1-indexed original sentence position (pre-shuffle).

Key differences vs. ZuCo2 training subjects:
  - rawData is (T, 105) — no .T needed (already timepoints × channels)
  - All 105 channels are real → pad to 128 (no trimming)
  - No 'content' field → reconstruct text from sentID + ZuCo2 task CSV files

Output: data/tmp/zuco_eeg_benchmarking.df
Schema:  ['eeg', 'mask', 'text', 'dataset', 'task', 'subject']
         (same as zuco_eeg_128ch_1280len.df for compatibility with STEP4)

Run from the GLIM/data directory:
    python __STEP3b_benchmarking_eeg_preproc.py
"""
import os
import h5py
import scipy
import numpy as np
import pandas as pd
from glob import glob
import rich.progress as rp

# ── Config ────────────────────────────────────────────────────
BENCH_DIR       = './raw_data/ZuCo2_benchmarking'
ZUCO2_TASK_DIR  = './raw_data/ZuCo2/task_materials'
OUTPUT_PATH     = './tmp/zuco_eeg_benchmarking.df'

SUBJECT_KEYS    = ['XAH', 'XBB', 'XBD', 'XDT', 'XLS', 'XPB', 'XSE', 'XSS', 'XTR', 'XWS']

SRC_SAMPLE_RATE = 500
TGT_SAMPLE_RATE = 128
TGT_MAX_LEN     = 1280   # timepoints after downsampling + padding
TGT_WIDTH       = 128    # channels after padding
SRC_CHANNELS    = 105    # real channels (no empty last channel unlike ZuCo1/2 training)

MIN_LEN = int(0.5 * SRC_SAMPLE_RATE)   # 250 samples (0.5s at 500Hz)
MAX_LEN = int(10  * SRC_SAMPLE_RATE)   # 5000 samples (10s at 500Hz)


def load_zuco2_ordered_sentences(task_dir: str) -> list[str]:
    """
    Load ZuCo2 NR + TSR sentences in original experiment order (1-indexed).

    Returns a list where index 0 is None (so sentID=1 → index 1),
    sentID 1..n_nr are NR sentences,
    sentID n_nr+1..n_nr+n_tsr are TSR sentences.
    """
    def load_sentences_from_csvs(prefix, n_files=7):
        sentences = []
        for i in range(1, n_files + 1):
            fpath = os.path.join(task_dir, f'{prefix}_{i}.csv')
            df = pd.read_csv(fpath, sep=';', encoding='utf-8', header=None,
                             names=['paragraph_id', 'sentence_id', 'sentence', 'control'],
                             dtype=str)
            sentences.extend(df['sentence'].tolist())
        return sentences

    nr_sentences  = load_sentences_from_csvs('nr')    # 370 sentences
    tsr_sentences = load_sentences_from_csvs('tsr')   # 411 sentences

    # 1-indexed: index 0 is placeholder
    ordered = [None] + nr_sentences + tsr_sentences
    print(f"  ZuCo2 sentences loaded: {len(nr_sentences)} NR + {len(tsr_sentences)} TSR")
    print(f"  Ordered list length: {len(ordered)} (1-indexed, index 0 = None)")
    return ordered


typobook = {
    "emp11111ty": "empty", "film.1": "film.", "–": "-", "\u2018s": "'s",
    "\ufffd s": "'s", "`s": "'s", "Maria": "Mari\u0107",
    "1Universidad": "Universidad", "1902\u201419": "1902 - 19",
    "Wuerttemberg": "W\u00fcrttemberg", "long -time": "long-time",
    "Jose": "Jos\u00e9", "Bucher": "B\u00f4cher", "1839 ? May": "1839 - May",
    "G\ufffd n\ufffd ration": "Generation", "Bragan\u00e7a": "Bragana",
    "1837?October": "1837 - October", "nVera-Ellen": "Vera-Ellen",
    "write Ethics": "wrote Ethics", "Adams-Onis": "Adams-On\u00eds",
    "(40 km?)": "(40 km\u00b2)", "(40 km\u02dd)": "(40 km\u00b2)",
    " (IPA: /?g?nz?b?g/) ": " ", '""Canes""': '"Canes"',
    "111Senator": "Senator", "Creteil": "Cr\u00e9teil",
    "Zoonomia": "Zo\u00f6nomia", "nee Darwin": "n\u00e9e Darwin",
    "Ruthy": "R\u00e9thy", "Eidgenoessische": "Eidgen\u00f6ssische",
    "40 km\ufffd": "40 km\u00b2", "King Leopold": "King L\u00e9opold",
}

def revise_typo(text: str) -> str:
    if not isinstance(text, str):
        return text
    for src, tgt in typobook.items():
        if src in text:
            text = text.replace(src, tgt)
    return text


def process_benchmarking_subject(mat_path: str,
                                  subject_key: str,
                                  ordered_sentences: list[str]) -> list[dict]:
    """
    Load and preprocess one benchmarking subject's .mat file.
    Returns a list of record dicts compatible with STEP4 matching.
    """
    records = []
    dropped = {'inf': 0, 'length': 0, 'bad_sentid': 0}

    mat = h5py.File(mat_path, 'r')
    s   = mat['sentenceData']
    n   = s['rawData'].shape[0]

    for j in range(n):
        # ── Get EEG ──────────────────────────────────────────
        raw_ref  = s['rawData'][j][0]
        eeg_raw  = mat[raw_ref][:].astype(np.float32)  # (T, 105)

        # Exclude non-finite
        if not np.all(np.isfinite(eeg_raw)):
            dropped['inf'] += 1
            continue

        T, ch = eeg_raw.shape
        assert ch == SRC_CHANNELS, f"Expected {SRC_CHANNELS} ch, got {ch}"

        # Length filter
        if not (MIN_LEN < T <= MAX_LEN):
            dropped['length'] += 1
            continue

        # ── Get sentence text via sentID ──────────────────────
        sid_ref  = s['sentID'][j][0]
        sid_data = mat[sid_ref][()]
        sent_id  = int(sid_data.flat[0])   # e.g. [[730.]] → 730

        if sent_id < 1 or sent_id >= len(ordered_sentences):
            dropped['bad_sentid'] += 1
            continue

        text_raw     = ordered_sentences[sent_id]  # original text from CSV
        text_revised = revise_typo(text_raw)        # typo-corrected

        # ── Downsample time axis (axis=0) ────────────────────
        T_new   = int(T * TGT_SAMPLE_RATE / SRC_SAMPLE_RATE)
        eeg_ds  = scipy.signal.resample(eeg_raw, T_new, axis=0)  # (T_new, 105)

        # ── Pad channels: 105 → 128 ──────────────────────────
        pad_ch      = TGT_WIDTH - SRC_CHANNELS  # 23
        eeg_padded  = np.pad(eeg_ds, ((0, 0), (0, pad_ch)),
                             'constant', constant_values=0)  # (T_new, 128)

        # ── Pad time: T_new → 1280 ───────────────────────────
        pad_t      = TGT_MAX_LEN - T_new
        eeg_final  = np.pad(eeg_padded, ((0, pad_t), (0, 0)),
                            'constant', constant_values=0)  # (1280, 128)

        # ── Mask ─────────────────────────────────────────────
        mask        = np.zeros(TGT_MAX_LEN, dtype=np.int8)
        mask[:T_new] = 1  # 1 = valid, 0 = padded

        records.append({
            'eeg':      eeg_final,
            'mask':     mask,
            'text':     text_revised,  # matches 'text' column in zuco_eeg_128ch_1280len.df
            'dataset':  'ZuCo2_benchmarking',
            'task':     'unknown',     # resolved in STEP4 via text match to label table
            'subject':  subject_key,
        })

    mat.close()
    total_seen = len(records) + sum(dropped.values())
    print(f"  {subject_key}: {len(records)}/{total_seen} recorded  "
          f"[dropped: inf={dropped['inf']}, len={dropped['length']}, "
          f"bad_sid={dropped['bad_sentid']}]")
    return records


def main():
    os.makedirs('./tmp', exist_ok=True)

    print("Loading ZuCo2 task material sentences...")
    ordered_sentences = load_zuco2_ordered_sentences(ZUCO2_TASK_DIR)
    print()

    mat_paths = sorted(glob(f'{BENCH_DIR}/results*.mat'))
    print(f"Found {len(mat_paths)} .mat files:")
    for p in mat_paths:
        print(f"  {os.path.basename(p)}")
    print()

    all_records = []
    with rp.Progress(
        rp.SpinnerColumn(),
        rp.TextColumn("[progress.description]{task.description}"),
        rp.BarColumn(), rp.TaskProgressColumn(),
        "•", rp.TimeElapsedColumn()
    ) as progress:
        task_bar = progress.add_task(
            "Processing benchmarking subjects...", total=len(mat_paths))

        for mat_path in mat_paths:
            fname       = os.path.basename(mat_path)
            subject_key = fname.replace('results', '').replace('.mat', '')

            if subject_key not in SUBJECT_KEYS:
                print(f"  Skipping unexpected subject: {subject_key}")
                progress.advance(task_bar)
                continue

            records = process_benchmarking_subject(
                mat_path, subject_key, ordered_sentences)
            all_records.extend(records)
            progress.advance(task_bar)

    df = pd.DataFrame(all_records)
    print(f"\n{'='*60}")
    print(f"Total records: {df.shape[0]}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Subjects: {df['subject'].value_counts().to_dict()}")
    print(f"\nEEG shape check:  {df.iloc[0]['eeg'].shape}  (expected: (1280, 128))")
    print(f"Mask shape check: {df.iloc[0]['mask'].shape}  (expected: (1280,))")
    print(f"\nSample text[0]: '{df.iloc[0]['text'][:80]}'")

    pd.to_pickle(df, OUTPUT_PATH)
    print(f"\n✅ Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
