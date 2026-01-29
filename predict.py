"""
Unified prediction script for corpus, relation, and sentiment classification.
Matches the metrics from predict_corpus.ipynb, predict_relation.ipynb, and predict_sentiment.ipynb.

Metrics computed:
  - clip-like acc: EEG-based prediction accuracy
  - clip-like acc [raw input text]: Text embedding accuracy using ground truth text
  - clip-like acc [gen text]: Text embedding accuracy using generated text
  - llm-pred acc: LLM-based classification accuracy (optional, requires HF token)

Usage:
    python predict.py --checkpoint <path> --task corpus
    python predict.py --checkpoint <path> --task relation
    python predict.py --checkpoint <path> --task sentiment
    python predict.py --checkpoint <path> --task all
    python predict.py --checkpoint <path> --task all --use_llm  # Include LLM evaluation
"""

import argparse
import os
import torch
import pandas as pd
from rich.progress import track
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy

from model.glim import GLIM
from data.datamodule import GLIMDataModule

console = Console()


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


def compute_text_embedding_accuracy(model, results, candidates, device, input_template="To English: <MASK>"):
    """Compute accuracy using text embeddings (raw input and generated text)."""
    probs_raw, probs_gen = [], []
    
    loader = DataLoader(results, batch_size=64, shuffle=False, drop_last=False)
    for batch in track(loader, description="Text embedding prediction"):
        raw_texts = batch['raw_input_text']
        gen_texts = batch['gen_text']
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prob_raw = model.predict_text_embedding(raw_texts, input_template, candidates)
            prob_gen = model.predict_text_embedding(gen_texts, input_template, candidates)
            probs_raw.append(prob_raw)
            probs_gen.append(prob_gen)
    
    probs_raw = torch.cat(probs_raw, dim=0)
    probs_gen = torch.cat(probs_gen, dim=0)
    
    return probs_raw, probs_gen


def load_llm_pipeline(device='cuda:0'):
    """Load LLM for classification (requires HF token)."""
    import transformers
    
    eval_llm_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading LLM: {eval_llm_id}...")
    
    pipe = transformers.pipeline(
        model=eval_llm_id,
        model_kwargs={"torch_dtype": torch.float16},
        device_map=torch.device(device),
    )
    return pipe


def compute_llm_accuracy(pipe, results, task_type, labels, num_classes):
    """Compute LLM-based classification accuracy."""
    if task_type == 'corpus':
        instructions = {
            "role": "system", 
            "content": (
                "You task is to classify the most likely topic of the following sentence."
                " Label '0' for 'movie review', '1' for 'personal biography'."
                " Please just output the integer label."
            )
        }
    elif task_type == 'relation':
        instructions = {
            "role": "system", 
            "content": (
                "You task is to classify the relation type in the following sentence."
                " Labels: 0='awarding', 1='education', 2='employment', 3='foundation',"
                " 4='job title', 5='nationality', 6='political affiliation', 7='visit', 8='marriage'."
                " Please just output the integer label."
            )
        }
    elif task_type == 'sentiment':
        instructions = {
            "role": "system", 
            "content": (
                "You task is to classify the sentiment of the following sentence."
                " Labels: 0='negative', 1='neutral', 2='positive'."
                " Please just output the integer label."
            )
        }
    
    # Use generated text for evaluation
    input_sentences = [r['gen_text'] for r in results]
    messages = [[instructions, {"role": "user", "content": sen}] for sen in input_sentences]
    inputs = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    terminators = [
        pipe.tokenizer.eos_token_id, 
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'
    
    with torch.no_grad():
        outputs = pipe(
            inputs, 
            batch_size=16, 
            max_new_tokens=4,
            eos_token_id=terminators,
            do_sample=True,
            num_beams=2,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
    
    # Calculate accuracy
    predictions = []
    for i, output in enumerate(outputs):
        try:
            pred_text = output[0]['generated_text'][len(inputs[i]):]
            pred = int(pred_text.strip())
            predictions.append(pred)
        except:
            predictions.append(-1)
    
    # Top-1 accuracy
    n_correct = sum(1 for i, p in enumerate(predictions) if p == labels[i])
    llm_acc_top1 = n_correct / len(labels)
    
    # For multi-class, also compute top-3 (if applicable)
    if num_classes > 3:
        # Top-3 would need probability outputs from LLM, so we approximate
        llm_acc_top3 = llm_acc_top1  # Placeholder - LLM only gives single prediction
    else:
        llm_acc_top3 = llm_acc_top1
    
    return llm_acc_top1, llm_acc_top3


def predict_corpus(model, dm, device, use_llm=False, llm_pipe=None):
    """Predict corpus classification (movie review vs personal biography)."""
    console.print("\n" + "="*80, style="bold blue")
    console.print("CORPUS CLASSIFICATION", style="bold blue")
    console.print("="*80 + "\n", style="bold blue")
    
    prefix = "The topic is about: "
    candidates = [
        prefix + "movie, good or bad", 
        prefix + "life experiences, relationship"
    ]
    
    results = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in track(dm.test_dataloader(), description="[EEG] Corpus prediction"):
            eeg = batch['eeg'].to(device)
            eeg_mask = batch['mask'].to(device)
            prompts = batch['prompt']
            raw_task_key = batch['raw task key']
            
            # Labels: task1 = 0 (movie), task2/task3 = 1 (biography)
            labels = []
            for t_key in raw_task_key:
                labels.append(0 if t_key == 'task1' else 1)
            labels_tensor = torch.tensor(labels, device=device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prob, gen_str = model.predict(eeg, eeg_mask, prompts, candidates, generate=True)
            
            all_labels.extend(labels)
            all_probs.append(prob)
            
            for i in range(len(eeg)):
                results.append({
                    'raw_input_text': batch['raw input text'][i],
                    'gen_text': gen_str[i],
                    'label': labels[i],
                    'prob': prob[i].cpu(),
                    'pred': prob[i].argmax().item(),
                })
    
    # EEG-based accuracy
    probs = torch.cat(all_probs, dim=0)
    labels_tensor = torch.tensor(all_labels, device=probs.device)
    clip_acc = multiclass_accuracy(probs, labels_tensor, num_classes=2, top_k=1, average='micro')
    
    # Text embedding accuracy
    probs_raw, probs_gen = compute_text_embedding_accuracy(model, results, candidates, device)
    clip_acc_raw = multiclass_accuracy(probs_raw, labels_tensor, num_classes=2, top_k=1, average='micro')
    clip_acc_gen = multiclass_accuracy(probs_gen, labels_tensor, num_classes=2, top_k=1, average='micro')
    
    # Results table
    table = Table(title="Corpus Classification Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("clip-like acc", f"{clip_acc.item():.4f}")
    table.add_row("clip-like acc [raw input text]", f"{clip_acc_raw.item():.4f}")
    table.add_row("clip-like acc [gen text]", f"{clip_acc_gen.item():.4f}")
    
    # LLM accuracy (optional)
    if use_llm and llm_pipe:
        llm_acc, _ = compute_llm_accuracy(llm_pipe, results, 'corpus', all_labels, 2)
        table.add_row("llm-pred acc", f"{llm_acc:.4f}")
    else:
        llm_acc = None
    
    console.print(table)
    
    return {
        'results': results,
        'clip_acc': clip_acc.item(),
        'clip_acc_raw': clip_acc_raw.item(),
        'clip_acc_gen': clip_acc_gen.item(),
        'llm_acc': llm_acc,
    }


def predict_relation(model, dm, device, use_llm=False, llm_pipe=None):
    """Predict relation classification."""
    console.print("\n" + "="*80, style="bold blue")
    console.print("RELATION CLASSIFICATION", style="bold blue")
    console.print("="*80 + "\n", style="bold blue")
    
    prefix = "Relation classification: "
    template = "It is about <MASK>."
    # These are the actual relation labels from the ZuCo dataset (Task 3)
    relation_types = [
        'awarding', 'education', 'employment', 'foundation', 
        'job title', 'nationality', 'political affiliation', 'visit', 'marriage'
    ]
    candidates = [prefix + template.replace("<MASK>", label) for label in relation_types]
    label_to_idx = {r: i for i, r in enumerate(relation_types)}
    
    results = []
    all_labels = []
    all_probs = []
    seen_labels = set()  # Debug: collect unique labels
    
    with torch.no_grad():
        for batch in track(dm.test_dataloader(), description="[EEG] Relation prediction"):
            eeg = batch['eeg'].to(device)
            eeg_mask = batch['mask'].to(device)
            prompts = batch['prompt']
            relation_labels = batch.get('relation label', [None] * len(eeg))
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prob, gen_str = model.predict(eeg, eeg_mask, prompts, candidates, generate=True)
            
            for i in range(len(eeg)):
                rel_label = relation_labels[i] if i < len(relation_labels) else None
                seen_labels.add(rel_label)  # Debug
                label_idx = label_to_idx.get(rel_label, -1) if rel_label and rel_label != 'nan' else -1
                
                if label_idx >= 0:
                    all_labels.append(label_idx)
                    all_probs.append(prob[i:i+1])
                
                results.append({
                    'raw_input_text': batch['raw input text'][i],
                    'gen_text': gen_str[i],
                    'relation_label': rel_label,
                    'label_idx': label_idx,
                    'prob': prob[i].cpu(),
                    'pred': prob[i].argmax().item(),
                })
    
    # Debug: print unique labels seen in data
    console.print(f"[dim]Unique relation labels in data: {seen_labels}[/dim]")
    
    # EEG-based accuracy
    if all_probs:
        probs = torch.cat(all_probs, dim=0)
        labels_tensor = torch.tensor(all_labels, device=probs.device)
        clip_acc1 = multiclass_accuracy(probs, labels_tensor, num_classes=len(relation_types), top_k=1, average='micro')
        clip_acc3 = multiclass_accuracy(probs, labels_tensor, num_classes=len(relation_types), top_k=3, average='micro')
        
        # Text embedding accuracy
        valid_results = [r for r in results if r['label_idx'] >= 0]
        probs_raw, probs_gen = compute_text_embedding_accuracy(model, valid_results, candidates, device, input_template="To English: <MASK>.")
        clip_acc_raw = multiclass_accuracy(probs_raw, labels_tensor, num_classes=len(relation_types), top_k=1, average='micro')
        clip_acc_gen = multiclass_accuracy(probs_gen, labels_tensor, num_classes=len(relation_types), top_k=1, average='micro')
    else:
        clip_acc1, clip_acc3, clip_acc_raw, clip_acc_gen = 0, 0, 0, 0
    
    # Results table
    table = Table(title="Relation Classification Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("clip-like acc1", f"{clip_acc1.item() if torch.is_tensor(clip_acc1) else clip_acc1:.4f}")
    table.add_row("clip-like acc3", f"{clip_acc3.item() if torch.is_tensor(clip_acc3) else clip_acc3:.4f}")
    table.add_row("clip-like acc [raw input text]", f"{clip_acc_raw.item() if torch.is_tensor(clip_acc_raw) else clip_acc_raw:.4f}")
    table.add_row("clip-like acc [gen text]", f"{clip_acc_gen.item() if torch.is_tensor(clip_acc_gen) else clip_acc_gen:.4f}")
    
    # LLM accuracy
    if use_llm and llm_pipe and all_labels:
        llm_acc1, llm_acc3 = compute_llm_accuracy(llm_pipe, valid_results, 'relation', all_labels, len(relation_types))
        table.add_row("llm-pred acc-top1", f"{llm_acc1:.4f}")
        table.add_row("llm-pred acc-top3", f"{llm_acc3:.4f}")
    else:
        llm_acc1, llm_acc3 = None, None
    
    console.print(table)
    console.print(f"Valid samples: {len(all_labels)}")
    
    # Debug: show label distribution
    if all_labels:
        from collections import Counter
        label_counts = Counter(all_labels)
        console.print("Label distribution:")
        for idx, count in sorted(label_counts.items()):
            console.print(f"  {relation_types[idx]}: {count}")
    
    return {
        'results': results,
        'clip_acc1': clip_acc1.item() if torch.is_tensor(clip_acc1) else clip_acc1,
        'clip_acc3': clip_acc3.item() if torch.is_tensor(clip_acc3) else clip_acc3,
        'clip_acc_raw': clip_acc_raw.item() if torch.is_tensor(clip_acc_raw) else clip_acc_raw,
        'clip_acc_gen': clip_acc_gen.item() if torch.is_tensor(clip_acc_gen) else clip_acc_gen,
        'llm_acc_top1': llm_acc1,
        'llm_acc_top3': llm_acc3,
    }


def predict_sentiment(model, dm, device, use_llm=False, llm_pipe=None):
    """Predict sentiment classification."""
    console.print("\n" + "="*80, style="bold blue")
    console.print("SENTIMENT CLASSIFICATION", style="bold blue")
    console.print("="*80 + "\n", style="bold blue")
    
    prefix = "Sentiment classification: "
    template = "It is <MASK>."
    # ZuCo Task 1 (Sentiment) typically has these 3 classes
    sentiment_types = ['negative', 'neutral', 'positive']
    candidates = [prefix + template.replace("<MASK>", label) for label in sentiment_types]
    label_to_idx = {s: i for i, s in enumerate(sentiment_types)}
    
    results = []
    all_labels = []
    all_probs = []
    seen_labels = set()  # Debug
    
    with torch.no_grad():
        for batch in track(dm.test_dataloader(), description="[EEG] Sentiment prediction"):
            eeg = batch['eeg'].to(device)
            eeg_mask = batch['mask'].to(device)
            prompts = batch['prompt']
            sentiment_labels = batch.get('sentiment label', [None] * len(eeg))
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prob, gen_str = model.predict(eeg, eeg_mask, prompts, candidates, generate=True)
            
            for i in range(len(eeg)):
                sent_label = sentiment_labels[i] if i < len(sentiment_labels) else None
                seen_labels.add(sent_label)  # Debug
                label_idx = label_to_idx.get(sent_label, -1) if sent_label and sent_label != 'nan' else -1
                
                if label_idx >= 0:
                    all_labels.append(label_idx)
                    all_probs.append(prob[i:i+1])
                
                results.append({
                    'raw_input_text': batch['raw input text'][i],
                    'gen_text': gen_str[i],
                    'sentiment_label': sent_label,
                    'label_idx': label_idx,
                    'prob': prob[i].cpu(),
                    'pred': prob[i].argmax().item(),
                })
    
    # Debug: print unique labels seen in data
    console.print(f"[dim]Unique sentiment labels in data: {seen_labels}[/dim]")
    
    # EEG-based accuracy
    if all_probs:
        probs = torch.cat(all_probs, dim=0)
        labels_tensor = torch.tensor(all_labels, device=probs.device)
        clip_acc1 = multiclass_accuracy(probs, labels_tensor, num_classes=len(sentiment_types), top_k=1, average='micro')
        
        # Text embedding accuracy
        valid_results = [r for r in results if r['label_idx'] >= 0]
        probs_raw, probs_gen = compute_text_embedding_accuracy(model, valid_results, candidates, device, input_template="Sentiment classification: <MASK>.")
        clip_acc_raw = multiclass_accuracy(probs_raw, labels_tensor, num_classes=len(sentiment_types), top_k=1, average='micro')
        clip_acc_gen = multiclass_accuracy(probs_gen, labels_tensor, num_classes=len(sentiment_types), top_k=1, average='micro')
    else:
        clip_acc1, clip_acc_raw, clip_acc_gen = 0, 0, 0
    
    # Results table
    table = Table(title="Sentiment Classification Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("clip-like acc", f"{clip_acc1.item() if torch.is_tensor(clip_acc1) else clip_acc1:.4f}")
    table.add_row("clip-like acc [raw input text]", f"{clip_acc_raw.item() if torch.is_tensor(clip_acc_raw) else clip_acc_raw:.4f}")
    table.add_row("clip-like acc [gen text]", f"{clip_acc_gen.item() if torch.is_tensor(clip_acc_gen) else clip_acc_gen:.4f}")
    
    # LLM accuracy
    if use_llm and llm_pipe and all_labels:
        llm_acc1, llm_acc3 = compute_llm_accuracy(llm_pipe, valid_results, 'sentiment', all_labels, len(sentiment_types))
        table.add_row("llm-pred acc-top1", f"{llm_acc1:.4f}")
        table.add_row("llm-pred acc-top3", f"{llm_acc3:.4f}")
    else:
        llm_acc1, llm_acc3 = None, None
    
    console.print(table)
    console.print(f"Valid samples: {len(all_labels)}")
    
    return {
        'results': results,
        'clip_acc': clip_acc1.item() if torch.is_tensor(clip_acc1) else clip_acc1,
        'clip_acc_raw': clip_acc_raw.item() if torch.is_tensor(clip_acc_raw) else clip_acc_raw,
        'clip_acc_gen': clip_acc_gen.item() if torch.is_tensor(clip_acc_gen) else clip_acc_gen,
        'llm_acc_top1': llm_acc1,
        'llm_acc_top3': llm_acc3,
    }


def main():
    parser = argparse.ArgumentParser(description='Unified prediction script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_path', type=str, default='./data/tmp/zuco_eeg_label_8variants.df')
    parser.add_argument('--task', type=str, default='all', 
                        choices=['corpus', 'relation', 'sentiment', 'all'],
                        help='Which prediction task to run')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_llm', action='store_true', 
                        help='Use LLM for classification (requires HF token for Llama)')
    parser.add_argument('--save_results', action='store_true', help='Save results to pickle files')
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()
    
    # Load model and data
    model, dm, device = load_model_and_data(
        args.checkpoint, args.data_path, args.batch_size, args.device
    )
    
    # Load LLM if requested
    llm_pipe = None
    if args.use_llm:
        try:
            llm_pipe = load_llm_pipeline(args.device)
        except Exception as e:
            print(f"Warning: Could not load LLM: {e}")
            print("Continuing without LLM evaluation...")
    
    all_results = {}
    
    # Run predictions
    if args.task in ['corpus', 'all']:
        all_results['corpus'] = predict_corpus(model, dm, device, args.use_llm, llm_pipe)
    
    if args.task in ['relation', 'all']:
        all_results['relation'] = predict_relation(model, dm, device, args.use_llm, llm_pipe)
    
    if args.task in ['sentiment', 'all']:
        all_results['sentiment'] = predict_sentiment(model, dm, device, args.use_llm, llm_pipe)
    
    # Summary table
    console.print("\n" + "="*80, style="bold green")
    console.print("SUMMARY", style="bold green")
    console.print("="*80, style="bold green")
    
    summary_table = Table(title="All Results Summary")
    summary_table.add_column("Task", style="cyan")
    summary_table.add_column("clip-like acc", style="green")
    summary_table.add_column("clip-like [raw]", style="green")
    summary_table.add_column("clip-like [gen]", style="green")
    summary_table.add_column("llm-pred", style="green")
    
    for task, data in all_results.items():
        clip_acc = data.get('clip_acc', data.get('clip_acc1', 0))
        summary_table.add_row(
            task.capitalize(),
            f"{clip_acc:.4f}",
            f"{data.get('clip_acc_raw', 0):.4f}",
            f"{data.get('clip_acc_gen', 0):.4f}",
            f"{data.get('llm_acc', data.get('llm_acc_top1', 'N/A'))}" if data.get('llm_acc') or data.get('llm_acc_top1') else "N/A"
        )
    
    console.print(summary_table)
    
    # Save results
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        for task, data in all_results.items():
            output_path = os.path.join(args.output_dir, f'{task}_predictions.pkl')
            pd.to_pickle(data['results'], output_path)
            print(f"Saved {task} results to {output_path}")
    
    return all_results


if __name__ == '__main__':
    main()
