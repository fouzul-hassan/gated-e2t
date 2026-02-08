"""
Detailed generation debug - check eeg_embeds and try different generation params.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
from transformers.modeling_outputs import BaseModelOutput
from inference import load_model, _patched_setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

DEMO_DF_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tmp', 'zuco_eeg_to_text_demo.df')
demo_df = pd.read_pickle(DEMO_DF_PATH)
row = demo_df.iloc[0]
print(f"Input text: {row.get('input text', '')}")
print(f"Target text: {row.get('gen text', row.get('target text', '?'))}")

ckpt_v1 = os.path.join(os.path.dirname(__file__), '..', 'runs', 'v1', 'epoch=199-step=397600.ckpt')
model = load_model(ckpt_v1, device)
model.eval()

eeg_np  = np.array(row['eeg'],  dtype=np.float32)
mask_np = np.array(row['mask'], dtype=np.int8)
if eeg_np.shape[0] == 128:
    eeg_np = eeg_np.T

eeg_t  = torch.from_numpy(eeg_np).unsqueeze(0).to(device)
mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)

task    = str(row.get('task', 'task1'))
subj    = str(row.get('subject', '<UNK>'))
dataset = str(row.get('dataset', '<UNK>'))
t_token = '<NR>' if task != 'task3' else '<TSR>'

print(f"\nEEG shape: {eeg_t.shape}, mask sum: {mask_t.sum()}")
print(f"Task token: {t_token}, subject: {subj}, dataset: {dataset}")

use_amp = device.type == 'cuda'
with torch.no_grad():
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        prompts = ([t_token], [dataset], [subj])
        p_ids   = model.p_embedder.encode(prompts, device=device)
        p_embed = model.p_embedder(p_ids, model.eval_pembed)
        
        eeg_hiddens, attn_weights = model.eeg_encoder(eeg_t, mask_t, p_embed, need_weights=True)
        eeg_embeds, eeg_emb = model.aligner.embed_eeg(eeg_hiddens)
        
        print(f"\neeg_hiddens shape: {eeg_hiddens.shape}")
        print(f"eeg_embeds shape: {eeg_embeds.shape}")
        print(f"eeg_embeds dtype: {eeg_embeds.dtype}")
        print(f"eeg_embeds stats: min={eeg_embeds.float().min():.4f}, max={eeg_embeds.float().max():.4f}, "
              f"mean={eeg_embeds.float().mean():.4f}, std={eeg_embeds.float().std():.4f}")
        print(f"eeg_embeds has_nan: {eeg_embeds.float().isnan().any()}")
        print(f"eeg_embeds has_inf: {eeg_embeds.float().isinf().any()}")

        # Try generating with different strategies
        for strategy_name, gen_kwargs in [
            ('greedy', dict(do_sample=False, num_beams=1, min_new_tokens=5, max_length=64)),
            ('beam4', dict(num_beams=4, min_new_tokens=5, max_length=64, no_repeat_ngram_size=3)),
            ('beam4_forced', dict(num_beams=4, min_new_tokens=10, max_length=64, no_repeat_ngram_size=3,
                                  length_penalty=2.0, repetition_penalty=1.5)),
            ('nucleus', dict(do_sample=True, top_p=0.9, temperature=0.8, min_new_tokens=5, max_length=64)),
        ]:
            gen_ids = model.text_model.generate(
                encoder_outputs=BaseModelOutput(eeg_embeds),
                **gen_kwargs
            )
            text = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            print(f"\n  [{strategy_name}] gen_ids: {gen_ids[0].tolist()}")
            print(f"  [{strategy_name}] text: '{text}'")

print("\nDone.")
