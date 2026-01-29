"""
GLIM Evaluation Script

Run evaluation metrics (including new ETES energy metric) on a trained checkpoint.
No retraining required.

Usage:
    python run_eval.py --checkpoint_path "path/to/checkpoint.ckpt" \
                       --data_path "./data/tmp/zuco_eeg_label_8variants.df" \
                       --use_energy --use_gated

Arguments:
    --checkpoint_path: Path to the .ckpt file
    --data_path: Path to the .df data file
    --use_energy: Enable energy-based evaluation (ETES)
    --use_gated: Enable gated attention config (must match training)
    --gpus: GPU IDs to use (default: 0)
"""
import os
import argparse
import torch
import warnings
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from model.glim import GLIM
from data.datamodule import GLIMDataModule

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='GLIM Evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    
    # Configuration (must match training)
    parser.add_argument('--use_gated', action='store_true', help='Use if model trained with gated attention')
    parser.add_argument('--use_energy', action='store_true', help='Use if model trained with energy components')
    parser.add_argument('--generation_strategy', type=str, default='beam',
                        choices=['beam', 'nucleus', 'greedy', 'energy'],
                        help='Generation strategy for eval')
    # Decoding knobs (optional overrides; useful for sweeps)
    parser.add_argument('--num_beams', type=int, default=None, help='Beam width (only for beam)')
    parser.add_argument('--top_p', type=float, default=None, help='Nucleus sampling top_p (only for nucleus)')
    parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature (only for nucleus)')
    parser.add_argument('--energy_rerank_candidates', type=int, default=None,
                        help='Num candidates for energy reranking (only for energy decoding)')
    
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs')
    parser.add_argument('--bsz_test', type=int, default=24, help='Batch size')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    devices = [int(x) for x in args.gpus.split(',')]
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    
    # Logger
    logger = WandbLogger(
        project='glim',
        name='eval-run',
        save_dir='./runs/eval',
        offline=True  # Offline by default for eval
    )
    
    # Data
    print(f"Loading data from {args.data_path}...")
    dm = GLIMDataModule(
        data_path=args.data_path,
        eval_noise_input=False,
        bsz_test=args.bsz_test,
        num_workers=4
    )
    
    # Model
    print(f"Loading model from {args.checkpoint_path}...")
    
    # We need to manually override strict loading if args are different
    # But load_from_checkpoint usually handles hparams if saved correctly.
    # We pass overrides just in case.
    
    model = GLIM.load_from_checkpoint(
        args.checkpoint_path,
        map_location=f"cuda:{devices[0]}",
        strict=False,
        
        # Override config for evaluation
        generation_strategy=args.generation_strategy,
        use_etes_eval=args.use_energy, # Ensure ETES is on if energy is requested
        use_energy_loss=False,         # No loss calculation needed for eval
        # Optional decoding overrides
        **({} if args.num_beams is None else {"num_beams": args.num_beams}),
        **({} if args.top_p is None else {"top_p": args.top_p}),
        **({} if args.temperature is None else {"temperature": args.temperature}),
        **({} if args.energy_rerank_candidates is None else {"energy_rerank_candidates": args.energy_rerank_candidates}),
    )
    
    # Trainer
    trainer = L.Trainer(
        accelerator='gpu',
        devices=devices,
        logger=logger,
        precision='bf16-mixed',
    )
    
    # Run Test
    print("Starting evaluation...")
    test_results = trainer.test(model, datamodule=dm)
    print("\n" + "="*80)
    print("EVALUATION COMPLETE - SUMMARY OF KEY METRICS")
    print("="*80)
    
    # Extract and display key metrics
    if test_results and len(test_results) > 0:
        metrics = test_results[0]  # Get first (and usually only) test result dict
        
        # Key metrics to display (using actual logged key names)
        key_metrics = {
            'Generation Metrics': [
                ('test/mean_BLEU1@MTV', 'BLEU-1 (Generated)'),
                ('test/mean_BLEU2@MTV', 'BLEU-2 (Generated)'),
                ('test/mean_ROUGE1@MTV', 'ROUGE-1 Recall (Generated)'),
                ('test/mean_ROUGE1@RAW', 'ROUGE-1 (Raw)'),
            ],
            'Classification Metrics': [
                ('test/mean_corpus_cls_acc', 'Corpus Classification Accuracy'),
                ('test/mean_relation_cls_acc_top01', 'Relation Classification (Top-1)'),
                ('test/mean_relation_cls_acc_top03', 'Relation Classification (Top-3)'),
                ('test/mean_sentiment_cls_acc_top01', 'Sentiment Classification (Top-1)'),
            ],
            'ETES Metrics (if available)': [
                ('test/etes_alignment', 'ETES Alignment'),
                ('test/etes_total', 'ETES Total'),
                ('test/etes_reference', 'ETES Reference'),
                ('test/etes_gap', 'ETES Gap'),
            ],
        }
        
        # Display metrics by category
        for category, metric_list in key_metrics.items():
            print(f"\n{category}:")
            print("-" * 80)
            found_any = False
            for metric_key, metric_name in metric_list:
                # Try different key formats (matching actual logged format)
                value = None
                # Try exact key first, then variations
                for key_format in [
                    metric_key,  # Try exact key first (e.g., 'test/mean_BLEU1@MTV')
                    metric_key.replace('test/', ''),  # Without test/ prefix
                    metric_key.replace('test/mean_', 'mean_'),  # Without test/ prefix but keep mean_
                ]:
                    if key_format in metrics:
                        value = metrics[key_format]
                        break
                
                if value is not None:
                    # Handle tensor values
                    if hasattr(value, 'item'):
                        value = value.item()
                    print(f"  {metric_name:.<50} {value:.4f}")
                    found_any = True
            
            if not found_any and category == 'ETES Metrics (if available)':
                print("  (ETES metrics not available - ensure --use_energy flag is set)")
        
        print("\n" + "="*80)
        print("Full metrics are logged to wandb. Check logs for detailed breakdown.")
        print("="*80)
    else:
        print("Evaluation complete! Check logs for metrics.")

if __name__ == '__main__':
    main()
