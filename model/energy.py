"""
Energy-Based Components for GLIM EEG-to-Text

This module implements three energy-based enhancements:

1. EnergyContrastiveLoss - EBM-style contrastive learning objective
2. EnergyGuidedGenerator - Energy-guided decoding for generation
3. ETESEvaluator - Novel EEG-Text Energy Score evaluation metric

Energy functions provide a principled way to measure and optimize
EEG-text alignment, going beyond simple cosine similarity.

References:
- Energy-Based Models: LeCun et al., "A Tutorial on Energy-Based Learning"
- COLD Decoding: Qin et al., "COLD Decoding" (arXiv:2202.11705)
- Contrastive Learning: Oord et al., "Representation Learning with CPC"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict, List, Tuple
import numpy as np


class EnergyContrastiveLoss(nn.Module):
    """
    Energy-Based Contrastive Loss for EEG-Text alignment.
    
    Unlike standard InfoNCE/CLIP loss, this formulation explicitly models
    an energy function E(eeg, text) where lower energy means better alignment.
    
    The loss encourages:
    - Low energy for positive (matched) EEG-text pairs
    - High energy for negative (mismatched) pairs
    
    Energy function: E(x, y) = -cos_sim(x, y) / temperature
    
    This formulation allows for:
    - More flexible energy landscapes than simple cosine similarity
    - Easy extension with additional energy terms (e.g., fluency, coherence)
    - Principled uncertainty estimation via partition function
    
    Regularization options (to prevent overfitting):
    - label_smoothing: Soften the contrastive targets (prevents overconfident predictions)
    - embedding_dropout: Add dropout to embeddings before energy computation
    - gradient_scale: Scale down gradients from energy loss
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 energy_type: Literal['cosine', 'bilinear', 'mlp'] = 'cosine',
                 hidden_dim: int = 1024,
                 learn_temperature: bool = False,
                 # Regularization options
                 label_smoothing: float = 0.1,  # Smooth targets to prevent overconfidence
                 embedding_dropout: float = 0.1,  # Dropout on embeddings
                 gradient_scale: float = 1.0):  # Scale gradients (< 1.0 reduces impact)
        """
        Args:
            temperature: Softmax temperature (lower = sharper distributions)
            energy_type: Type of energy function
                - 'cosine': E = -cos_sim (standard, efficient)
                - 'bilinear': E = -x^T W y (learnable interaction matrix)
                - 'mlp': E = MLP([x; y; x*y]) (most expressive)
            hidden_dim: Dimension of embeddings
            learn_temperature: Whether to learn temperature as a parameter
            label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
            embedding_dropout: Dropout rate for embeddings during training
            gradient_scale: Scale factor for gradients (1.0 = normal, 0.5 = half gradients)
        """
        super().__init__()
        
        self.energy_type = energy_type
        self.label_smoothing = label_smoothing
        self.gradient_scale = gradient_scale
        
        # Embedding dropout for regularization
        self.embed_dropout = nn.Dropout(p=embedding_dropout)
        
        if learn_temperature:
            # Log-scale for numerical stability
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))
        
        # Learnable energy function components
        if energy_type == 'bilinear':
            self.W = nn.Parameter(torch.eye(hidden_dim) * 0.1)
        elif energy_type == 'mlp':
            self.energy_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            # Initialize to approximate cosine similarity
            nn.init.zeros_(self.energy_mlp[-1].weight)
            nn.init.zeros_(self.energy_mlp[-1].bias)
    
    @property
    def temperature(self) -> float:
        return torch.exp(self.log_temperature)
    
    def compute_energy(self, eeg_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise energy between EEG and text embeddings.
        
        Args:
            eeg_emb: (B, D) EEG embeddings
            text_emb: (B, D) or (B', D) Text embeddings
            
        Returns:
            energy: (B, B') pairwise energy matrix (lower = better match)
        """
        # Normalize embeddings
        eeg_norm = F.normalize(eeg_emb, dim=-1)
        text_norm = F.normalize(text_emb, dim=-1)
        
        if self.energy_type == 'cosine':
            # E = -cos_sim: simple and effective
            similarity = torch.mm(eeg_norm, text_norm.T)
            energy = -similarity / self.temperature
            
        elif self.energy_type == 'bilinear':
            # E = -x^T W y: learnable interaction
            # W is initialized close to identity, so starts similar to cosine
            transformed = torch.mm(eeg_norm, self.W)
            similarity = torch.mm(transformed, text_norm.T)
            energy = -similarity / self.temperature
            
        elif self.energy_type == 'mlp':
            # E = MLP([x; y; x*y]): most expressive
            B1, B2 = eeg_emb.size(0), text_emb.size(0)
            # Expand for pairwise computation
            eeg_exp = eeg_norm.unsqueeze(1).expand(-1, B2, -1)  # (B1, B2, D)
            text_exp = text_norm.unsqueeze(0).expand(B1, -1, -1)  # (B1, B2, D)
            
            # Concatenate features: [eeg, text, eeg*text]
            combined = torch.cat([eeg_exp, text_exp, eeg_exp * text_exp], dim=-1)
            energy = self.energy_mlp(combined).squeeze(-1)  # (B1, B2)
            energy = energy / self.temperature
        
        return energy
    
    def forward(self, eeg_emb: torch.Tensor, text_emb: torch.Tensor, 
                return_metrics: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute EBM contrastive loss with regularization.
        
        Args:
            eeg_emb: (B, D) EEG embeddings
            text_emb: (B, D) Text embeddings (same batch, matched pairs)
            return_metrics: Whether to return additional metrics
            
        Returns:
            Dictionary with 'loss' and optional metrics
        """
        B = eeg_emb.size(0)
        device = eeg_emb.device
        
        # Apply embedding dropout for regularization (only during training)
        if self.training:
            eeg_emb = self.embed_dropout(eeg_emb)
            text_emb = self.embed_dropout(text_emb)
        
        # Compute pairwise energies
        energy = self.compute_energy(eeg_emb, text_emb)  # (B, B)
        
        # Convert to logits (negative energy = higher probability)
        logits = -energy  # (B, B)
        
        # Create targets with label smoothing
        if self.label_smoothing > 0:
            # Soft targets: (1 - smoothing) for correct, smoothing/(B-1) for others
            smooth_value = self.label_smoothing / (B - 1)
            targets = torch.full((B, B), smooth_value, device=device)
            targets.fill_diagonal_(1.0 - self.label_smoothing)
            
            # Cross entropy with soft targets
            log_probs_eeg = F.log_softmax(logits, dim=1)
            log_probs_text = F.log_softmax(logits, dim=0)
            
            loss_eeg_to_text = -(targets * log_probs_eeg).sum(dim=1).mean()
            loss_text_to_eeg = -(targets * log_probs_text).sum(dim=0).mean()
        else:
            # Standard hard targets
            labels = torch.arange(B, device=device)
            loss_eeg_to_text = F.cross_entropy(logits, labels)
            loss_text_to_eeg = F.cross_entropy(logits.T, labels)
        
        loss = (loss_eeg_to_text + loss_text_to_eeg) / 2
        
        # Apply gradient scaling (reduces energy loss impact during backprop)
        if self.gradient_scale != 1.0 and self.training:
            # Scale gradients by detaching and re-adding with scale
            loss = loss * self.gradient_scale + loss.detach() * (1 - self.gradient_scale)
        
        result = {'loss': loss}
        
        if return_metrics:
            # Compute accuracy for monitoring
            with torch.no_grad():
                labels = torch.arange(B, device=device)
                # Lower energy = higher probability of being the match
                pred_eeg_to_text = logits.argmax(dim=1)
                pred_text_to_eeg = logits.argmax(dim=0)
                
                acc_eeg_to_text = (pred_eeg_to_text == labels).float().mean()
                acc_text_to_eeg = (pred_text_to_eeg == labels).float().mean()
                
                positive_energy = energy.diag()
                result['acc_eeg_to_text'] = acc_eeg_to_text
                result['acc_text_to_eeg'] = acc_text_to_eeg
                result['mean_positive_energy'] = positive_energy.mean()
                result['temperature'] = self.temperature
        
        return result


class EnergyGuidedGenerator:
    """
    Energy-Guided Text Generation for EEG-to-Text.
    
    Uses gradient-based or sampling-based methods to guide text generation
    toward outputs that have low energy (high alignment) with the source EEG.
    
    Two modes:
    1. Reranking: Generate N candidates, return lowest energy one
    2. Energy sampling: Bias token probabilities toward low-energy continuations
    
    Based on ideas from:
    - COLD Decoding (Qin et al., 2022)
    - Energy-Based Models for text (Deng et al., 2020)
    """
    
    def __init__(self,
                 text_model,
                 tokenizer,
                 aligner,
                 energy_weight: float = 1.0,
                 n_candidates: int = 5,
                 mode: Literal['rerank', 'energy_sample'] = 'rerank'):
        """
        Args:
            text_model: The language model (e.g., Flan-T5)
            tokenizer: Tokenizer for the language model
            aligner: GLIM aligner module for computing EEG/text embeddings
            energy_weight: How much to weight energy vs. LM probability
            n_candidates: Number of candidates to generate for reranking
            mode: 'rerank' or 'energy_sample'
        """
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.aligner = aligner
        self.energy_weight = energy_weight
        self.n_candidates = n_candidates
        self.mode = mode
    
    def compute_alignment_energy(self, 
                                  eeg_emb: torch.Tensor, 
                                  text_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute energy (negative alignment) between EEG and generated text.
        
        Args:
            eeg_emb: (B, D) EEG embedding vector
            text_ids: (B, L) Generated text token IDs
            
        Returns:
            energy: (B,) Energy for each sample (lower = better)
        """
        # Encode text through the LM encoder
        text_encoder = self.text_model.get_encoder()
        with torch.no_grad():
            text_outputs = text_encoder(input_ids=text_ids)
            text_hidden = text_outputs.last_hidden_state
        
        # Get text embedding through aligner
        text_emb = self.aligner.embed_text(text_hidden, mask=None)
        
        # Energy = negative cosine similarity
        eeg_norm = F.normalize(eeg_emb, dim=-1)
        text_norm = F.normalize(text_emb, dim=-1)
        
        energy = -F.cosine_similarity(eeg_norm, text_norm, dim=-1)
        return energy
    
    @torch.no_grad()
    def generate_with_reranking(self,
                                 eeg_embeds: torch.Tensor,
                                 eeg_emb_vector: torch.Tensor,
                                 max_length: int = 64,
                                 **generate_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple candidates and return the one with lowest energy.
        
        Args:
            eeg_embeds: (B, L, D) EEG sequence embeddings for decoder
            eeg_emb_vector: (B, D) EEG global embedding for energy computation
            max_length: Maximum generation length
            **generate_kwargs: Additional args for model.generate()
            
        Returns:
            best_ids: (B, L) Token IDs of best candidates
            energies: (B,) Energy of selected candidates
        """
        from transformers.modeling_outputs import BaseModelOutput
        
        B = eeg_embeds.size(0)
        device = eeg_embeds.device
        
        # Generate N candidates per sample using nucleus sampling
        all_candidates = []
        for _ in range(self.n_candidates):
            gen_ids = self.text_model.generate(
                encoder_outputs=BaseModelOutput(eeg_embeds),
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                max_length=max_length,
                **generate_kwargs
            )
            all_candidates.append(gen_ids)
        
        # Stack candidates: (n_candidates, B, L)
        # Pad to same length
        max_len = max(c.size(1) for c in all_candidates)
        padded_candidates = []
        for c in all_candidates:
            if c.size(1) < max_len:
                pad = torch.full((B, max_len - c.size(1)), 
                                self.tokenizer.pad_token_id,
                                device=device, dtype=c.dtype)
                c = torch.cat([c, pad], dim=1)
            padded_candidates.append(c)
        
        candidates = torch.stack(padded_candidates, dim=1)  # (B, n_candidates, L)
        
        # Compute energy for each candidate
        energies = torch.zeros(B, self.n_candidates, device=device)
        for i in range(self.n_candidates):
            candidate_ids = candidates[:, i, :]  # (B, L)
            energies[:, i] = self.compute_alignment_energy(eeg_emb_vector, candidate_ids)
        
        # Select lowest energy candidate for each sample
        best_idx = energies.argmin(dim=1)  # (B,)
        best_ids = candidates[torch.arange(B, device=device), best_idx]  # (B, L)
        best_energies = energies[torch.arange(B, device=device), best_idx]  # (B,)
        
        return best_ids, best_energies
    
    def generate(self,
                 eeg_embeds: torch.Tensor,
                 eeg_emb_vector: torch.Tensor,
                 max_length: int = 64,
                 **generate_kwargs) -> Dict[str, torch.Tensor]:
        """
        Main generation method.
        
        Returns:
            Dictionary with 'ids', 'energies', and 'texts'
        """
        if self.mode == 'rerank':
            ids, energies = self.generate_with_reranking(
                eeg_embeds, eeg_emb_vector, max_length, **generate_kwargs
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented yet")
        
        # Decode to text
        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        
        return {
            'ids': ids,
            'energies': energies,
            'texts': texts,
        }


class ETESEvaluator:
    """
    EEG-Text Energy Score (ETES) Evaluator.
    
    A novel evaluation metric for EEG-to-Text models that measures
    how well generated text aligns with the source EEG signal.
    
    Unlike BLEU/ROUGE which only measure surface overlap with references,
    ETES directly measures the semantic alignment between EEG and generated text.
    
    Key advantages:
    1. EEG-aware: Considers the source signal, not just reference text
    2. Reference-free: Can evaluate without ground truth transcriptions
    3. Continuous: Provides fine-grained quality scores
    4. Learnable: Uses the learned EEG-text alignment model
    
    The score is computed as the negative cosine similarity between
    EEG and text embeddings (lower = better alignment).
    
    Score interpretation:
    - ETES < -0.8: Excellent alignment
    - ETES < -0.5: Good alignment  
    - ETES < -0.2: Fair alignment
    - ETES > 0: Poor alignment (embeddings are dissimilar)
    """
    
    def __init__(self, 
                 aligner,
                 text_encoder,
                 tokenizer,
                 include_fluency: bool = False,
                 fluency_model_id: Optional[str] = None):
        """
        Args:
            aligner: Trained GLIM aligner module
            text_encoder: Language model encoder (e.g., T5 encoder)
            tokenizer: Tokenizer for the language model
            include_fluency: Whether to include a fluency (perplexity) term
            fluency_model_id: Model ID for fluency scoring (optional)
        """
        self.aligner = aligner
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.include_fluency = include_fluency
        
        if include_fluency and fluency_model_id:
            from transformers import AutoModelForCausalLM
            self.fluency_model = AutoModelForCausalLM.from_pretrained(fluency_model_id)
        else:
            self.fluency_model = None
    
    @torch.no_grad()
    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode texts to embeddings using the aligner."""
        # Tokenize
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                max_length=96, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Encode through LM encoder
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Get embedding through aligner
        text_emb = self.aligner.embed_text(hidden_states, attention_mask)
        return text_emb
    
    @torch.no_grad()
    def compute_alignment_energy(self,
                                   eeg_emb: torch.Tensor,
                                   text_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment energy between EEG and text embeddings.
        
        Returns:
            energy: (B,) Energy scores (lower = better alignment)
        """
        eeg_norm = F.normalize(eeg_emb, dim=-1)
        text_norm = F.normalize(text_emb, dim=-1)
        
        # Energy = negative cosine similarity
        energy = -F.cosine_similarity(eeg_norm, text_norm, dim=-1)
        return energy
    
    @torch.no_grad()
    def compute_fluency_score(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Compute fluency score based on perplexity.
        Lower perplexity = more fluent = lower energy.
        """
        if self.fluency_model is None:
            return torch.zeros(len(texts), device=device)
        
        ppls = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt').to(device)
            outputs = self.fluency_model(**inputs, labels=inputs['input_ids'])
            ppl = torch.exp(outputs.loss)
            ppls.append(ppl)
        
        # Normalize perplexity to energy scale
        ppl_tensor = torch.stack(ppls)
        fluency_energy = torch.log(ppl_tensor) / 10  # Scale factor
        return fluency_energy
    
    @torch.no_grad()
    def evaluate(self,
                 eeg_emb_vectors: torch.Tensor,
                 generated_texts: List[str],
                 reference_texts: Optional[List[str]] = None,
                 fluency_weight: float = 0.1) -> Dict[str, float]:
        """
        Evaluate generated texts using ETES metric.
        
        Args:
            eeg_emb_vectors: (B, D) EEG embedding vectors
            generated_texts: List of generated text strings
            reference_texts: Optional list of reference texts (for comparison)
            fluency_weight: Weight for fluency term in total energy
            
        Returns:
            Dictionary with ETES scores and statistics
        """
        device = eeg_emb_vectors.device
        B = eeg_emb_vectors.size(0)
        
        # Encode generated texts
        gen_text_emb = self.encode_text(generated_texts, device)
        
        # Compute alignment energy
        alignment_energy = self.compute_alignment_energy(eeg_emb_vectors, gen_text_emb)
        
        # Optionally add fluency term
        if self.include_fluency:
            fluency_energy = self.compute_fluency_score(generated_texts, device)
            total_energy = alignment_energy + fluency_weight * fluency_energy
        else:
            total_energy = alignment_energy
            fluency_energy = torch.zeros(B, device=device)
        
        results = {
            # Primary ETES metric (alignment only)
            'etes_alignment': alignment_energy.mean().item(),
            'etes_alignment_std': alignment_energy.std().item(),
            
            # Total energy (with fluency if enabled)
            'etes_total': total_energy.mean().item(),
            'etes_total_std': total_energy.std().item(),
            
            # Distribution statistics
            'etes_min': total_energy.min().item(),
            'etes_max': total_energy.max().item(),
            'etes_median': total_energy.median().item(),
            
            # Percentile-based metrics
            'etes_p10': torch.quantile(total_energy, 0.1).item(),
            'etes_p90': torch.quantile(total_energy, 0.9).item(),
        }
        
        # If reference texts provided, compute reference energy for comparison
        if reference_texts is not None:
            ref_text_emb = self.encode_text(reference_texts, device)
            ref_energy = self.compute_alignment_energy(eeg_emb_vectors, ref_text_emb)
            results['etes_reference'] = ref_energy.mean().item()
            results['etes_gap'] = (alignment_energy - ref_energy).mean().item()
        
        return results
    
    def get_sample_scores(self,
                          eeg_emb_vectors: torch.Tensor,
                          generated_texts: List[str]) -> List[Dict]:
        """
        Get per-sample ETES scores for detailed analysis.
        
        Returns:
            List of dictionaries with individual sample scores
        """
        device = eeg_emb_vectors.device
        
        gen_text_emb = self.encode_text(generated_texts, device)
        energies = self.compute_alignment_energy(eeg_emb_vectors, gen_text_emb)
        
        sample_scores = []
        for i, (text, energy) in enumerate(zip(generated_texts, energies)):
            sample_scores.append({
                'index': i,
                'text': text,
                'etes': energy.item(),
                'quality': self._interpret_score(energy.item()),
            })
        
        return sample_scores
    
    @staticmethod
    def _interpret_score(energy: float) -> str:
        """Interpret ETES score as qualitative label."""
        if energy < -0.8:
            return 'excellent'
        elif energy < -0.5:
            return 'good'
        elif energy < -0.2:
            return 'fair'
        elif energy < 0.2:
            return 'poor'
        else:
            return 'very_poor'
