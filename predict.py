"""
Unified prediction script for corpus, relation, and sentiment classification.
Converts the functionality of predict_corpus.ipynb, predict_relation.ipynb, and predict_sentiment.ipynb.

Usage:
    python predict.py --checkpoint <path> --task corpus
    python predict.py --checkpoint <path> --task relation
    python predict.py --checkpoint <path> --task sentiment
    python predict.py --checkpoint <path> --task all
"""

import argparse
import torch
import pandas as pd
from rich.progress import track
from rich import print as rprint
from torchmetrics.functional.classification import multiclass_accuracy

from model.glim import GLIM
from data.datamodule import GLIMDataModule


def load_model_and_data(checkpoint_path, data_path, batch_size=24, device='cuda:0'):
    """Load model and data module."""
    device = torch.device(device)
    
    print(f"Loading model from {checkpoint_path}...")
    model = GLIM.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,
    )
    model.setup(stage='test')
    model.eval()
    
    print(f"Loading data from {data_path}...")
    dm = GLIMDataModule(
        data_path=data_path,
        eval_noise_input=False,
        bsz_test=batch_size,
    )
    dm.setup(stage='test')
    
    return model, dm, device


def predict_corpus(model, dm, device):
    """Predict corpus classification (movie review vs personal biography)."""
    print("\n" + "="*80)
    print("CORPUS CLASSIFICATION")
    print("="*80 + "\n")
    
    prefix = "The topic is about: "
    candidates = [
        prefix + "movie, good or bad", 
        prefix + "life experiences, relationship"
    ]
    
    results = []
    with torch.no_grad():
        for batch in track(dm.test_dataloader(), description="Corpus prediction"):
            eeg = batch['eeg'].to(device)
            eeg_mask = batch['mask'].to(device)
            prompts = batch['prompt']
            raw_task_key = batch['raw task key']
            
            # Labels: task1 = 0 (movie), task2/task3 = 1 (biography)
            labels = []
            for t_key in raw_task_key:
                labels.append(0 if t_key == 'task1' else 1)
            labels = torch.tensor(labels, device=device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prob, gen_str = model.predict(eeg, eeg_mask, prompts, candidates, generate=True)
            
            for i in range(len(eeg)):
                results.append({
                    'raw_input_text': batch['raw input text'][i],
                    'gen_text': gen_str[i],
                    'label': labels[i].item(),
                    'prob': prob[i],
                    'pred': prob[i].argmax().item(),
                })
    
    # Calculate accuracy
    probs = torch.stack([torch.tensor(r['prob']) for r in results])
    labels = torch.tensor([r['label'] for r in results])
    acc = multiclass_accuracy(probs, labels.to(probs.device), num_classes=2, top_k=1, average='micro')
    
    print(f"\n[bold green]Corpus Classification Accuracy: {acc.item():.4f}[/bold green]")
    
    # Per-class stats
    correct_0 = sum(1 for r in results if r['label'] == 0 and r['pred'] == 0)
    correct_1 = sum(1 for r in results if r['label'] == 1 and r['pred'] == 1)
    total_0 = sum(1 for r in results if r['label'] == 0)
    total_1 = sum(1 for r in results if r['label'] == 1)
    
    print(f"  Movie Review (class 0): {correct_0}/{total_0} = {correct_0/total_0:.4f}")
    print(f"  Biography (class 1): {correct_1}/{total_1} = {correct_1/total_1:.4f}")
    
    return results, acc.item()


def predict_relation(model, dm, device):
    """Predict relation classification (person-place, person-person, etc.)."""
    print("\n" + "="*80)
    print("RELATION CLASSIFICATION")
    print("="*80 + "\n")
    
    prefix = "The relation type is: "
    relation_types = [
        "place of birth",
        "place of death", 
        "country of nationality",
        "country of administrative divisions",
        "place of headquarters",
        "neighborhood of",
        "company",
        "children",
        "employment history",
        "peer"
    ]
    candidates = [prefix + r for r in relation_types]
    
    # Build label mapping
    label_to_idx = {r: i for i, r in enumerate(relation_types)}
    
    results = []
    with torch.no_grad():
        for batch in track(dm.test_dataloader(), description="Relation prediction"):
            eeg = batch['eeg'].to(device)
            eeg_mask = batch['mask'].to(device)
            prompts = batch['prompt']
            relation_labels = batch.get('relation label', [None] * len(eeg))
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prob, gen_str = model.predict(eeg, eeg_mask, prompts, candidates, generate=True)
            
            for i in range(len(eeg)):
                rel_label = relation_labels[i] if i < len(relation_labels) else None
                label_idx = label_to_idx.get(rel_label, -1) if rel_label else -1
                
                results.append({
                    'raw_input_text': batch['raw input text'][i],
                    'gen_text': gen_str[i],
                    'relation_label': rel_label,
                    'label_idx': label_idx,
                    'prob': prob[i],
                    'pred': prob[i].argmax().item(),
                    'pred_label': relation_types[prob[i].argmax().item()],
                })
    
    # Filter valid labels and calculate accuracy
    valid_results = [r for r in results if r['label_idx'] >= 0]
    if valid_results:
        probs = torch.stack([torch.tensor(r['prob']) for r in valid_results])
        labels = torch.tensor([r['label_idx'] for r in valid_results])
        
        acc_top1 = multiclass_accuracy(probs, labels.to(probs.device), 
                                        num_classes=len(relation_types), top_k=1, average='micro')
        acc_top3 = multiclass_accuracy(probs, labels.to(probs.device), 
                                        num_classes=len(relation_types), top_k=3, average='micro')
        
        print(f"\n[bold green]Relation Classification Accuracy (Top-1): {acc_top1.item():.4f}[/bold green]")
        print(f"[bold green]Relation Classification Accuracy (Top-3): {acc_top3.item():.4f}[/bold green]")
        print(f"Valid samples: {len(valid_results)}")
    else:
        acc_top1 = 0.0
        print("No valid relation labels found in test set")
    
    return results, acc_top1 if valid_results else 0.0


def predict_sentiment(model, dm, device):
    """Predict sentiment classification (positive, negative, neutral)."""
    print("\n" + "="*80)
    print("SENTIMENT CLASSIFICATION")
    print("="*80 + "\n")
    
    prefix = "The sentiment is: "
    sentiment_types = ["very negative", "negative", "neutral", "positive", "very positive"]
    candidates = [prefix + s for s in sentiment_types]
    
    # Build label mapping
    label_to_idx = {s: i for i, s in enumerate(sentiment_types)}
    
    results = []
    with torch.no_grad():
        for batch in track(dm.test_dataloader(), description="Sentiment prediction"):
            eeg = batch['eeg'].to(device)
            eeg_mask = batch['mask'].to(device)
            prompts = batch['prompt']
            sentiment_labels = batch.get('sentiment label', [None] * len(eeg))
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prob, gen_str = model.predict(eeg, eeg_mask, prompts, candidates, generate=True)
            
            for i in range(len(eeg)):
                sent_label = sentiment_labels[i] if i < len(sentiment_labels) else None
                label_idx = label_to_idx.get(sent_label, -1) if sent_label else -1
                
                results.append({
                    'raw_input_text': batch['raw input text'][i],
                    'gen_text': gen_str[i],
                    'sentiment_label': sent_label,
                    'label_idx': label_idx,
                    'prob': prob[i],
                    'pred': prob[i].argmax().item(),
                    'pred_label': sentiment_types[prob[i].argmax().item()],
                })
    
    # Filter valid labels and calculate accuracy
    valid_results = [r for r in results if r['label_idx'] >= 0]
    if valid_results:
        probs = torch.stack([torch.tensor(r['prob']) for r in valid_results])
        labels = torch.tensor([r['label_idx'] for r in valid_results])
        
        acc_top1 = multiclass_accuracy(probs, labels.to(probs.device), 
                                        num_classes=len(sentiment_types), top_k=1, average='micro')
        
        print(f"\n[bold green]Sentiment Classification Accuracy: {acc_top1.item():.4f}[/bold green]")
        print(f"Valid samples: {len(valid_results)}")
        
        # Per-class breakdown
        for i, sent in enumerate(sentiment_types):
            class_results = [r for r in valid_results if r['label_idx'] == i]
            if class_results:
                correct = sum(1 for r in class_results if r['pred'] == i)
                print(f"  {sent}: {correct}/{len(class_results)} = {correct/len(class_results):.4f}")
    else:
        acc_top1 = 0.0
        print("No valid sentiment labels found in test set")
    
    return results, acc_top1 if valid_results else 0.0


def main():
    parser = argparse.ArgumentParser(description='Unified prediction script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_path', type=str, default='./data/tmp/zuco_eeg_label_8variants.df')
    parser.add_argument('--task', type=str, default='all', 
                        choices=['corpus', 'relation', 'sentiment', 'all'],
                        help='Which prediction task to run')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_results', action='store_true', help='Save results to pickle files')
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()
    
    # Load model and data
    model, dm, device = load_model_and_data(
        args.checkpoint, args.data_path, args.batch_size, args.device
    )
    
    all_results = {}
    
    # Run predictions
    if args.task in ['corpus', 'all']:
        results, acc = predict_corpus(model, dm, device)
        all_results['corpus'] = {'results': results, 'accuracy': acc}
    
    if args.task in ['relation', 'all']:
        results, acc = predict_relation(model, dm, device)
        all_results['relation'] = {'results': results, 'accuracy': acc}
    
    if args.task in ['sentiment', 'all']:
        results, acc = predict_sentiment(model, dm, device)
        all_results['sentiment'] = {'results': results, 'accuracy': acc}
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for task, data in all_results.items():
        print(f"  {task.capitalize()} Accuracy: {data['accuracy']:.4f}")
    
    # Save results
    if args.save_results:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        for task, data in all_results.items():
            output_path = os.path.join(args.output_dir, f'{task}_predictions.pkl')
            pd.to_pickle(data['results'], output_path)
            print(f"Saved {task} results to {output_path}")
    
    return all_results


if __name__ == '__main__':
    main()
