import pandas as pd, numpy as np

df = pd.read_pickle('data/tmp/zuco_eeg_label_8variants.df')

lines = []
for i in [0, 100, 5000]:
    r = df.iloc[i]
    e = r['eeg']
    m = r['mask']
    al = int(m.sum())
    
    lines.append('=' * 80)
    lines.append(f'SAMPLE {i}')
    lines.append('=' * 80)
    lines.append(f'  Subject:         {r.subject}')
    lines.append(f'  Dataset:         {r.dataset}')
    lines.append(f'  Task:            {r.task}')
    lines.append(f'  Phase:           {r.phase}')
    lines.append(f'  Input Text:      {str(r["input text"])[:100]}')
    lines.append(f'  Sentiment Label: {r["sentiment label"]}')
    lines.append(f'  Relation Label:  {r["relation label"]}')
    lines.append(f'  Text UID:        {r["text uid"]}')
    lines.append(f'  EEG shape:       {e.shape}   dtype: {e.dtype}')
    lines.append(f'  Actual length:   {al} / 1280  = {round(al/128, 2)}s')
    lines.append(f'  EEG stats:       min={e[:al].min():.4f}  max={e[:al].max():.4f}  mean={e[:al].mean():.6f}  std={e[:al].std():.4f}')
    lines.append(f'  Mask first 10:   {m[:10].tolist()}')
    lines.append('')
    lines.append('  EEG first 5 timepoints x first 8 channels:')
    for t in range(5):
        vals = [f'{e[t,c]:8.4f}' for c in range(8)]
        lines.append(f'    t={t:4d}: [{", ".join(vals)}]')
    lines.append('')
    lines.append('  EEG around padding boundary:')
    for t in range(max(0, al-2), min(al+2, 1280)):
        tag = 'VALID' if t < al else 'PAD  '
        vals = [f'{e[t,c]:8.4f}' for c in range(8)]
        lines.append(f'    t={t:4d} [{tag}]: [{", ".join(vals)}]')
    lines.append('')
    lines.append('  Last 4 channels (125-128, zero-padded):')
    for t in range(3):
        vals = [f'{e[t,c]:8.4f}' for c in range(124, 128)]
        lines.append(f'    t={t:4d}: [{", ".join(vals)}]')
    lines.append('')

with open('_sample_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('Written to _sample_output.txt')
