"""Test inference end-to-end without wandb stdout issues."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, pandas as pd
from inference import run_inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
demo_df = pd.read_pickle(os.path.join('..', 'data', 'tmp', 'zuco_eeg_to_text_demo.df'))

row = demo_df.iloc[0]
input_text = row.get('input text', '?')

# Write to file to avoid wandb stdout encoding issues
with open('test_inference_output.txt', 'w', encoding='utf-8') as f:
    f.write(f"Input: {input_text}\n")
    
result = run_inference(0, row, os.path.join('..', 'runs', 'v1', 'epoch=199-step=397600.ckpt'), 
                       'v1', device)

with open('test_inference_output.txt', 'a', encoding='utf-8') as f:
    f.write(f"gen_text: {repr(result['gen_text'])}\n")
    f.write(f"from_cache: {result.get('_from_cache')}\n")
    f.write(f"bleu1_raw: {result['text_metrics'].get('bleu1_raw')}\n")

print("Done - check test_inference_output.txt")
