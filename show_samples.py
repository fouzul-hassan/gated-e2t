"""
Show sample generated texts from a checkpoint.
Usage: python show_samples.py --checkpoint <path> --num_samples 10
"""

import argparse
import torch
import pandas as pd
from model.glim import GLIM
from data.data_module import GLIMDataModule


def main():
    parser = argparse.ArgumentParser(description='Show sample generated texts')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_path', type=str, default='./data/tmp/zuco_eeg_label_8variants.df')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to show')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Load model
    model = GLIM.load_from_checkpoint(args.checkpoint, strict=False)
    model.eval()
    model.cuda()
    
    # Load data
    dm = GLIMDataModule(pd.read_pickle(args.data_path), batch_size=1)
    dm.setup('test')
    
    dataloader = dm.test_dataloader() if args.split == 'test' else dm.val_dataloader()
    
    print(f"\n{'='*80}")
    print(f"SAMPLE GENERATIONS FROM: {args.checkpoint}")
    print(f"{'='*80}\n")
    
    samples_shown = 0
    with torch.no_grad():
        for batch in dataloader:
            if samples_shown >= args.num_samples:
                break
            
            # Move to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Generate
            outputs = model.shared_forward(batch)
            
            # Decode
            gen_ids = outputs.get('gen_ids', outputs.get('sample_ids'))
            if gen_ids is None:
                print("No generation found in outputs")
                continue
                
            gen_text = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            
            # Get reference text
            ref_text = batch.get('tgt_strs', batch.get('raw_tgt_strs', ['N/A']))[0]
            if isinstance(ref_text, tuple):
                ref_text = ref_text[0]
            
            print(f"Sample {samples_shown + 1}:")
            print(f"  Reference: {ref_text}")
            print(f"  Generated: {gen_text}")
            print()
            
            samples_shown += 1
    
    print(f"{'='*80}")
    print(f"Showed {samples_shown} samples")


if __name__ == '__main__':
    main()
