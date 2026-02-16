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
    """
    Compute LLM-based classification accuracy.
    Matches the notebook approaches exactly:
    - corpus: integer labels, max_new_tokens=4
    - relation: top-3 text labels, max_new_tokens=16
    - sentiment: text labels, max_new_tokens=8
    """
    # Use generated text for evaluation
    input_sentences = [r['gen_text'] for r in results]
    
    if task_type == 'corpus':
        instructions = {
            "role": "system", 
            "content": (
                "You task is to classify the most likely topic of the following sentence."
                " Label '0' for 'movie review', '1' for 'personal biography'."
                " Please just output the integer label."
            )
        }
        max_new_tokens = 4
    elif task_type == 'relation':
        instructions = {
            "role": "system", 
            "content": (
                "You task is relation extraction. Please choose the top-3 from the below 9 possible labels:\n"
                " awarding, education, employment, foundation, job title, nationality,"
                " political affiliation, visit, and marriage.\n"
                " Please just output the three most likely labels in descending order of probability,"
                " and separate them by commas. Do not output any other words!"
            )
        }
        max_new_tokens = 16
    elif task_type == 'sentiment':
        instructions = {
            "role": "system", 
            "content": (
                "You task is sentiment classification. Please pick the most likely label from:\n"
                " negative, neutral and positive.\n"
                " Please just output your predicted label do not output any other words!"
            )
        }
        max_new_tokens = 8
    
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
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            num_beams=2,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
    
    if task_type == 'corpus':
        # Corpus: integer label matching (notebook approach)
        n_correct = 0
        for i in range(len(outputs)):
            try:
                pred = int(outputs[i][0]['generated_text'][len(inputs[i]):])
                if pred == labels[i]:
                    n_correct += 1
            except:
                pass
        llm_acc_top1 = n_correct / len(labels)
        return llm_acc_top1, llm_acc_top1
    
    elif task_type == 'relation':
        # Relation: top-3 text label matching (notebook approach)
        all_relations = ['awarding', 'education', 'employment', 'foundation', 
                         'job title', 'nationality', 'political affiliation', 'visit', 'marriage']
        label_to_str = {i: r for i, r in enumerate(all_relations)}
        
        n_correct_top1 = 0
        n_correct_top3 = 0
        for i in range(len(outputs)):
            pred_text = outputs[i][0]['generated_text'][len(inputs[i]):].strip()
            pred_labels = [p.strip().lower() for p in pred_text.split(',')]
            true_label = label_to_str.get(labels[i], '')
            
            if len(pred_labels) >= 1 and pred_labels[0] == true_label:
                n_correct_top1 += 1
            if true_label in pred_labels[:3]:
                n_correct_top3 += 1
        
        llm_acc_top1 = n_correct_top1 / len(labels)
        llm_acc_top3 = n_correct_top3 / len(labels)
        return llm_acc_top1, llm_acc_top3
    
    elif task_type == 'sentiment':
        # Sentiment: text label matching (notebook approach)
        all_sentiments = ['negative', 'neutral', 'positive']
        label_to_str = {i: s for i, s in enumerate(all_sentiments)}
        
        n_correct = 0
        for i in range(len(outputs)):
            pred_text = outputs[i][0]['generated_text'][len(inputs[i]):].strip().lower()
            true_label = label_to_str.get(labels[i], '')
            if pred_text == true_label:
                n_correct += 1
        
        llm_acc_top1 = n_correct / len(labels)
        return llm_acc_top1, llm_acc_top1


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
    
    table.add_row("EEG Accuracy", f"{clip_acc.item():.4f}")
    table.add_row("Text Acc (Raw)", f"{clip_acc_raw.item():.4f}")
    table.add_row("Text Acc (Gen)", f"{clip_acc_gen.item():.4f}")
    
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
        clip_acc1 = multiclass_accuracy(probs, labels_tensor, num_classes=len(relation_types), top_k=1, ignore_index=-1, average='micro')
        clip_acc3 = multiclass_accuracy(probs, labels_tensor, num_classes=len(relation_types), top_k=3, ignore_index=-1, average='micro')
        
        # Text embedding accuracy
        valid_results = [r for r in results if r['label_idx'] >= 0]
        probs_raw, probs_gen = compute_text_embedding_accuracy(model, valid_results, candidates, device, input_template="To English: <MASK>.")
        clip_acc_raw = multiclass_accuracy(probs_raw, labels_tensor, num_classes=len(relation_types), top_k=3, ignore_index=-1, average='micro')
        clip_acc_gen = multiclass_accuracy(probs_gen, labels_tensor, num_classes=len(relation_types), top_k=3, ignore_index=-1, average='micro')
    else:
        clip_acc1, clip_acc3, clip_acc_raw, clip_acc_gen = 0, 0, 0, 0
    
    # Results table
    table = Table(title="Relation Classification Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("EEG Acc (Top-1)", f"{clip_acc1.item() if torch.is_tensor(clip_acc1) else clip_acc1:.4f}")
    table.add_row("EEG Acc (Top-3)", f"{clip_acc3.item() if torch.is_tensor(clip_acc3) else clip_acc3:.4f}")
    table.add_row("Text Acc (Raw)", f"{clip_acc_raw.item() if torch.is_tensor(clip_acc_raw) else clip_acc_raw:.4f}")
    table.add_row("Text Acc (Gen)", f"{clip_acc_gen.item() if torch.is_tensor(clip_acc_gen) else clip_acc_gen:.4f}")
    
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
    
    prefix = "Sentiment: "
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
        clip_acc1 = multiclass_accuracy(probs, labels_tensor, num_classes=len(sentiment_types), top_k=1, ignore_index=-1, average='micro')
        
        # Text embedding accuracy
        valid_results = [r for r in results if r['label_idx'] >= 0]
        probs_raw, probs_gen = compute_text_embedding_accuracy(model, valid_results, candidates, device, input_template="Sentiment classification: <MASK>.")
        clip_acc_raw = multiclass_accuracy(probs_raw, labels_tensor, num_classes=len(sentiment_types), top_k=1, ignore_index=-1, average='micro')
        clip_acc_gen = multiclass_accuracy(probs_gen, labels_tensor, num_classes=len(sentiment_types), top_k=1, ignore_index=-1, average='micro')
    else:
        clip_acc1, clip_acc_raw, clip_acc_gen = 0, 0, 0
    
    # Results table
    table = Table(title="Sentiment Classification Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("EEG Accuracy", f"{clip_acc1.item() if torch.is_tensor(clip_acc1) else clip_acc1:.4f}")
    table.add_row("Text Acc (Raw)", f"{clip_acc_raw.item() if torch.is_tensor(clip_acc_raw) else clip_acc_raw:.4f}")
    table.add_row("Text Acc (Gen)", f"{clip_acc_gen.item() if torch.is_tensor(clip_acc_gen) else clip_acc_gen:.4f}")
    
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


def save_results_to_txt(all_results, output_dir, checkpoint_path):
    """Save each task's results to a separate text file."""
    from datetime import datetime
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for task, data in all_results.items():
        txt_path = os.path.join(output_dir, f'{task}_results.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"  {task.upper()} CLASSIFICATION RESULTS\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Timestamp:  {timestamp}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n\n")
            
            # --- Summary Metrics ---
            f.write(f"{'-'*40}\n")
            f.write(f"  Summary Metrics\n")
            f.write(f"{'-'*40}\n")
            
            if task == 'corpus':
                f.write(f"  EEG Accuracy:        {data.get('clip_acc', 0):.4f}\n")
                f.write(f"  Text Acc (Raw):      {data.get('clip_acc_raw', 0):.4f}\n")
                f.write(f"  Text Acc (Gen):      {data.get('clip_acc_gen', 0):.4f}\n")
                if data.get('llm_acc') is not None:
                    f.write(f"  LLM-Pred Acc:        {data['llm_acc']:.4f}\n")
            
            elif task == 'relation':
                f.write(f"  EEG Acc (Top-1):     {data.get('clip_acc1', 0):.4f}\n")
                f.write(f"  EEG Acc (Top-3):     {data.get('clip_acc3', 0):.4f}\n")
                f.write(f"  Text Acc (Raw):      {data.get('clip_acc_raw', 0):.4f}\n")
                f.write(f"  Text Acc (Gen):      {data.get('clip_acc_gen', 0):.4f}\n")
                if data.get('llm_acc_top1') is not None:
                    f.write(f"  LLM-Pred Acc (Top1): {data['llm_acc_top1']:.4f}\n")
                    f.write(f"  LLM-Pred Acc (Top3): {data['llm_acc_top3']:.4f}\n")
            
            elif task == 'sentiment':
                f.write(f"  EEG Accuracy:        {data.get('clip_acc', 0):.4f}\n")
                f.write(f"  Text Acc (Raw):      {data.get('clip_acc_raw', 0):.4f}\n")
                f.write(f"  Text Acc (Gen):      {data.get('clip_acc_gen', 0):.4f}\n")
                if data.get('llm_acc_top1') is not None:
                    f.write(f"  LLM-Pred Acc (Top1): {data['llm_acc_top1']:.4f}\n")
                    f.write(f"  LLM-Pred Acc (Top3): {data['llm_acc_top3']:.4f}\n")
            
            f.write(f"\n")
            
            # --- Per-Sample Predictions ---
            results = data.get('results', [])
            if results:
                f.write(f"{'-'*40}\n")
                f.write(f"  Per-Sample Predictions ({len(results)} samples)\n")
                f.write(f"{'-'*40}\n\n")
                
                if task == 'corpus':
                    label_names = ['movie review', 'personal biography']
                    for i, r in enumerate(results):
                        pred_idx = r.get('pred', -1)
                        label_idx = r.get('label', -1)
                        pred_name = label_names[pred_idx] if 0 <= pred_idx < len(label_names) else '?'
                        label_name = label_names[label_idx] if 0 <= label_idx < len(label_names) else '?'
                        correct = '✓' if pred_idx == label_idx else '✗'
                        f.write(f"[{i+1:4d}] {correct}  Label: {label_name:<20s}  Pred: {pred_name:<20s}\n")
                        f.write(f"        Raw:  {r.get('raw_input_text', '')[:80]}\n")
                        f.write(f"        Gen:  {r.get('gen_text', '')[:80]}\n\n")
                
                elif task == 'relation':
                    relation_types = ['awarding', 'education', 'employment', 'foundation', 
                                      'job title', 'nationality', 'political affiliation', 'visit', 'marriage']
                    for i, r in enumerate(results):
                        pred_idx = r.get('pred', -1)
                        label_idx = r.get('label_idx', -1)
                        pred_name = relation_types[pred_idx] if 0 <= pred_idx < len(relation_types) else '?'
                        label_name = r.get('relation_label', '?')
                        correct = '✓' if pred_idx == label_idx else ('✗' if label_idx >= 0 else '-')
                        f.write(f"[{i+1:4d}] {correct}  Label: {str(label_name):<25s}  Pred: {pred_name:<25s}\n")
                        f.write(f"        Raw:  {r.get('raw_input_text', '')[:80]}\n")
                        f.write(f"        Gen:  {r.get('gen_text', '')[:80]}\n\n")
                
                elif task == 'sentiment':
                    sentiment_types = ['negative', 'neutral', 'positive']
                    for i, r in enumerate(results):
                        pred_idx = r.get('pred', -1)
                        label_idx = r.get('label_idx', -1)
                        pred_name = sentiment_types[pred_idx] if 0 <= pred_idx < len(sentiment_types) else '?'
                        label_name = r.get('sentiment_label', '?')
                        correct = '✓' if pred_idx == label_idx else ('✗' if label_idx >= 0 else '-')
                        f.write(f"[{i+1:4d}] {correct}  Label: {str(label_name):<12s}  Pred: {pred_name:<12s}\n")
                        f.write(f"        Raw:  {r.get('raw_input_text', '')[:80]}\n")
                        f.write(f"        Gen:  {r.get('gen_text', '')[:80]}\n\n")
            
            f.write(f"\n{'='*80}\n")
        
        print(f"Saved {task} results to {txt_path}")


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
    summary_table.add_column("EEG Accuracy", style="green")
    summary_table.add_column("Text Acc (Raw)", style="green")
    summary_table.add_column("Text Acc (Gen)", style="green")
    summary_table.add_column("LLM Prediction", style="green")
    
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
    
    # Always save results to text files
    save_results_to_txt(all_results, args.output_dir, args.checkpoint)
    
    # Optionally save to pickle
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        for task, data in all_results.items():
            output_path = os.path.join(args.output_dir, f'{task}_predictions.pkl')
            pd.to_pickle(data['results'], output_path)
            print(f"Saved {task} pickle to {output_path}")
    
    return all_results


if __name__ == '__main__':
    main()
