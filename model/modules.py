import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from typing import Literal, Optional


class PromptEmbedder(nn.Module):
    """
    Embeds prompt ids into vector representations.
    """
    def __init__(self, 
                 dim = 128,
                 prompt_nums: tuple[int] = (3, 3, 31), 
                 drop_probs: tuple[float] = (0.0, 0.0, 0.0),
                 prompt_keys: dict[str, list[str]] = None,
                 ):
        super().__init__()
        assert prompt_nums == tuple(len(keys) for keys in prompt_keys.values())
        assert all([(prob >= 0.0 and prob <= 1.0) for prob in drop_probs])
        self.all_zero_inject = all([prob == 1.0 for prob in drop_probs])
        self.dim = dim
        self.prompt_nums = prompt_nums
        self.drop_probs = drop_probs
        self.prompt_keys = prompt_keys
        self.embedders = nn.ModuleList([nn.Embedding(num, dim) for num in prompt_nums])
        for embedder in self.embedders:
            nn.init.normal_(embedder.weight, std = 0.02)
        self.sigma = nn.Parameter(torch.tensor([num / sum(prompt_nums) for num in prompt_nums])) 
        
    def forward(self, prompt_ids, 
                eval_pembed: Literal['zero', 'sum', 'mean', 'src'] = 'src', 
                ) -> torch.Tensor:
        '''
        prompt_ids:             (n, 3)
        eval_pembed:            the aggregation method of prompt embed.
        all_zero_inject:        see below
        '''

        if self.all_zero_inject:
            return self.zero_inject(prompt_ids)
        
        p_embed = 0
        for k, (embedder, p_num, prob) in enumerate(zip(self.embedders,
                                                                  self.prompt_nums, 
                                                                  self.drop_probs)):
            pids = prompt_ids[:,k]  # a Tensor of shape (n,), the task/dataset/subject prompt ids in batch

            if self.training:
                pids = self.p_drop(pids, prob)
                p = embedder(pids)

            else:  
                if (eval_pembed == 'zero' and prob > 0.0) or prob == 1.0:
                    # NOTE: `prob == 1.0` indicates that only the 0-th embedding is trained
                    p = embedder(torch.zeros_like(pids))

                elif eval_pembed == 'sum' and prob > 0.0:  # sum 0-th embeddings and the src
                    embedder_sum = nn.EmbeddingBag.from_pretrained(embeddings=embedder.weight,
                                                                   mode='sum',)
                    bags = torch.stack([torch.zeros_like(pids), pids], dim=1)
                    weight = torch.tensor([(p_num-1)*prob, (1-prob)], device=pids.device)
                    weights = (weight / weight.sum()).expand_as(bags)
                    p = embedder_sum(input = bags, per_sample_weights = weights) 

                elif eval_pembed == 'mean':
                    embedder_mean = nn.EmbeddingBag.from_pretrained(embeddings=embedder.weight,
                                                                    mode='mean',)
                    p = embedder_mean(input=torch.arange(1, p_num, device=pids.device).expand(pids.shape[0], -1))
                    # NOTE: ignore 0-th  

                else:  # use the `src` prompt
                    p = embedder(pids)

            p_embed += p * self.sigma[k]
        return p_embed
    
    @torch.no_grad()
    def p_drop(self, src_pids, drop_prob):
        """
        src_pids:     (n) ids of t/d/s in batch
        drop_prob:      float
        """
        assert self.training == True  # NOTE: only on training
        if drop_prob > 0.0:
            drop_mask = torch.rand(src_pids.shape, device=src_pids.device) < drop_prob
            tgt_pids = torch.where(drop_mask, 0, src_pids).to(src_pids)
        else:
            tgt_pids = src_pids
        return tgt_pids
    
    def encode(self, prompts: list[tuple[str]], device=None) -> torch.Tensor:
        '''
        prompts:                [n*('task'), n*('datset'), n*('subject'))
        prompt_ids:             (n, 3)
        '''
        bsz = len(prompts[0])
        prompt_ids = []
        for k, prompt_keys in enumerate(self.prompt_keys.values()):
            ids = [prompt_keys.index(prompts[k][i]) for i in range(bsz)]
            ids = torch.tensor(ids, dtype=torch.int, device=device)
            prompt_ids.append(ids)
        prompt_ids = torch.stack(prompt_ids, dim=-1)
        return prompt_ids
    
    def decode(self, prompt_ids) -> list[tuple[str]]:
        '''
        prompt_ids:             (n, 3)
        prompts:                n*('task', 'datset', 'subject')
        '''
        all_prompts = []
        for ids in prompt_ids:
            prompts = (keys[idx] for keys, idx in zip(self.prompt_keys.values(), ids))
            all_prompts.append(prompts)
        return all_prompts
    
    def zero_inject(self, prompt_ids):
        """
        An add-hoc method to control the prompt injection for the ablation study of GLIM
        """
        # p_t = self.embedders[0](prompt_ids[:,0])
        p_t = self.embedders[0](torch.zeros_like(prompt_ids[:,0]))
        p_d = self.embedders[1](torch.zeros_like(prompt_ids[:,1]))
        p_s = self.embedders[2](torch.zeros_like(prompt_ids[:,2]))
        p_embed = p_t*self.sigma[0] + p_d*self.sigma[1] + p_s*self.sigma[2] 
        return p_embed


class EEGEncoder(nn.Module):
    """
    Q-Merger:           Transformer blocks (encoder + decoder) with learnable queries
    Backbone:           Transformer encoder blocks with adaLN
    """
    def __init__(
        self,
        in_len = 1280,
        out_len = 96,
        in_dim = 128,
        out_dim = 256,
        task_embed_len = 4,
        n_in_blocks = 6, 
        n_out_blocks = 6, 
        in_temporal_modulate = True,
        out_is_causal = True, 
        **block_kwargs,
        ):
        super().__init__()

        self.out_dim = out_dim
        self.task_embed_len = task_embed_len
        self.in_blocks = nn.ModuleList([EncoderBlock(in_dim, in_len, inject_prompt=True, 
                                                     temporal_modulate=in_temporal_modulate,
                                                     is_causal=False, **block_kwargs) for _ in range(n_in_blocks)])
        self.x_proj = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim, eps=1e-6, elementwise_affine=True)
        self.shared_queries = nn.Parameter(torch.randn(1, out_len - task_embed_len, out_dim))
        self.out_blocks = nn.ModuleList([DecoderBlock(out_dim, is_causal=out_is_causal, **block_kwargs)
                                            for _ in range(n_out_blocks)])
        self.norm2 = nn.LayerNorm(out_dim, eps=1e-6, elementwise_affine=True)

        pos_embed = get_1d_sincos_pos_embed_from_grid(in_dim, np.arange(in_len))
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False)


    def forward(self, eeg, mask, p, need_weights=False) -> tuple[torch.Tensor, dict]:
        '''
        input:
            eeg:            (n, t, c)     c is the channel num, t is the num of time points
            mask:           (n, t),       torch.int, 1 for `unmasked`, 
            p:              (n, c)        prompt embedding token for ('task', 'dataset', 'subject')
        output:
            out             (n, l, d)     d for hidden dim
            attn_weights    (n, l, c)     a dict of attention weights for each of i-th decoder block 
        '''
        bsz = eeg.shape[0]
        x = eeg + self.pos_embed                                        # (n, t, c) 
        for i, in_block in enumerate(self.in_blocks):
            x = in_block(x, mask, p)                                    # (n, t, c)

        x = self.norm1(self.x_proj(x))                                  # (n, t, d)
        q = self.shared_queries.expand(bsz, -1, -1)                     # (n, l, d)         

        attn_weights = {}
        for j, out_block in enumerate(self.out_blocks):
            q, attn_w = out_block(q, x, mask, need_weights=need_weights)
            attn_weights.update({j: attn_w})
        out = self.norm2(q)                                             # (n, l, d)
        return out, attn_weights
    
    def get_gate_stats(self) -> dict:
        """
        Aggregate gate statistics from all gated attention layers.
        Returns average stats across all encoder and decoder blocks.
        """
        gate_means = []
        gate_stds = []
        gate_sparsities = []
        
        # Collect from encoder blocks
        for block in self.in_blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'get_gate_stats'):
                stats = block.attn.get_gate_stats()
                gate_means.append(stats['gate_mean'])
                gate_stds.append(stats['gate_std'])
                gate_sparsities.append(stats['gate_sparsity'])
        
        # Collect from decoder blocks
        for block in self.out_blocks:
            if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'get_gate_stats'):
                stats = block.self_attn.get_gate_stats()
                gate_means.append(stats['gate_mean'])
                gate_stds.append(stats['gate_std'])
                gate_sparsities.append(stats['gate_sparsity'])
            if hasattr(block, 'cross_attn') and hasattr(block.cross_attn, 'get_gate_stats'):
                stats = block.cross_attn.get_gate_stats()
                gate_means.append(stats['gate_mean'])
                gate_stds.append(stats['gate_std'])
                gate_sparsities.append(stats['gate_sparsity'])
        
        if len(gate_means) == 0:
            return {'gate_mean': 0.0, 'gate_std': 0.0, 'gate_sparsity': 0.0}
        
        return {
            'gate_mean': sum(gate_means) / len(gate_means),
            'gate_std': sum(gate_stds) / len(gate_stds),
            'gate_sparsity': sum(gate_sparsities) / len(gate_sparsities),
        }


class Aligner(nn.Module):
    '''
    share all modules except queries between modalities
    '''
    def __init__(self, hidden_dim, embed_dim, num_heads, dropout,
                 commitment_loss_key: Literal['mse','kl_div']= 'mse', use_y_mask=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(hidden_dim, embed_dim)
        self.q_x = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True) # NOTE: emb_dim
        self.q_y = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.cross_attn_x = CrossAttention(embed_dim, num_heads, dropout)  
        self.cross_attn_y = CrossAttention(embed_dim, num_heads, dropout)  # not shared anymore!
        self.commitment_loss_key = commitment_loss_key
        self.use_y_mask = use_y_mask

    def forward(self, x, y, y_mask):
        '''
        inputs:
        - x:        (n, l, d)  `eeg` hidden states, output of eeg encoder
        - y:        (n, l, e)  `input_text` hidden states, `encoder_output` of text model
        - y_mask:   (n, l)      
        '''
        x = self.in_proj(x)  # (n, l, e)
        loss_commitment = self.align_embeds(x, y, y_mask)
        x_mask = y_mask if (self.training and self.use_y_mask) else None
        x, x_emb = self.embed_eeg(x, x_mask)
        y_emb = self.embed_text(y, y_mask)
        loss_clip, logits_clip = self.align_emb_vector(x_emb, y_emb)
        return loss_clip, logits_clip, loss_commitment, x, x_emb, y_emb
    
    def embed_eeg(self, x, mask=None):
        '''
        [(n,l,d) -> ] (n,l,e) -> (n,e)
        '''
        if x.shape[-1] == self.hidden_dim:
            x = self.in_proj(x)  # (n, l, e)
        x_emb = self.cross_attn_x(self.q_x.expand(x.shape[0], -1, -1), x, mask)[0].squeeze()  # (n, e)
        return x, x_emb
    
    def embed_text(self, y, y_mask=None):
        '''
        y:     (n, l, e)    
        '''
        y_emb = self.cross_attn_y(self.q_y.expand(y.shape[0], -1, -1), y, y_mask)[0].squeeze()  # (n, e)
        return y_emb
    
    def align_emb_vector(self, x_emb, y_emb):
        bsz = x_emb.shape[0]
        x_normed = x_emb / x_emb.norm(dim=1, keepdim=True)        # (n, e)
        y_normed = y_emb / y_emb.norm(dim=1, keepdim=True)        # (n, e)
        x_logits = x_normed @ y_normed.T                  # (n, n)
        y_logits = x_logits.T

        target = torch.arange(bsz, device=x_emb.device)  # (n)
        
        loss = (F.cross_entropy(x_logits, target) + F.cross_entropy(y_logits, target)) / 2
        return loss, x_logits.detach()
    
    def align_embeds(self, x, y, y_mask):
        if self.training and self.use_y_mask:
            mask = ~y_mask.bool().unsqueeze(-1).expand_as(x)
            x = x.masked_fill_(mask, 0)
            y = y.masked_fill_(mask, 0)
        if self.commitment_loss_key == 'mse':
            loss = F.mse_loss(x, y)
        else:
            loss = F.kl_div(x, y, reduction='batchmean') 
        return loss


############################################# sub modules #############################################


class EncoderBlock(nn.Module):
    """
    A Transformer encoder block with adaptive layernorm (adaLN) from DiT.
    Supports optional gated attention for improved feature learning.
    """
    def __init__(self, hidden_dim, hidden_len, 
                 inject_prompt=True, temporal_modulate=True, is_causal=False, 
                 num_heads=8, mlp_ratio=4, dropout=0.0,
                 use_gated_attention=False, gating_type='elementwise'):
        super().__init__()
        self.inject_prompt = inject_prompt
        self.temporal_modulate = temporal_modulate
        self.is_causal = is_causal
        self.use_gated_attention = use_gated_attention
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine = not inject_prompt)
        # Choose between standard attention and gated attention
        if use_gated_attention:
            self.attn = GatedSelfAttention(hidden_dim, num_heads, dropout, is_causal=is_causal,
                                           gating_type=gating_type)
        else:
            self.attn = SelfAttention(hidden_dim, num_heads, dropout, is_causal=is_causal)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine = not inject_prompt)
        self.mlp = Mlp(hidden_dim, hidden_dim * mlp_ratio, drop=dropout,
                       act_layer=lambda: nn.GELU(approximate="tanh"))
        if inject_prompt:
            self.adaLN = nn.Sequential(nn.SiLU(),
                                       nn.Linear(hidden_dim, 6 * hidden_dim, bias=True))
            # zero-out adaLN:
            nn.init.zeros_(self.adaLN[1].weight)
            nn.init.zeros_(self.adaLN[1].bias)
            if temporal_modulate:
                self.t_adaLN = nn.Sequential(nn.SiLU(), 
                                            nn.Linear(hidden_dim, 2 * hidden_len, bias=True))
                nn.init.zeros_(self.t_adaLN[1].weight)
                nn.init.zeros_(self.t_adaLN[1].bias)
            
    def forward(self, x, mask, p=None):
        # x: (n,l,d)  p: (n,d)
        # mask: (n,l) 1--> unmasked
        if not self.inject_prompt:
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.mlp(self.norm2(x))
            return x
        if self.temporal_modulate:
            x = self.t_modulate(x, p)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(p).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(self.modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def t_modulate(self, x, p):
        scale, shift = self.t_adaLN(p).chunk(2, dim=1)  # (n,d) -> 2* (n,l)
        x = x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)
        return x
    

class DecoderBlock(nn.Module):
    """
    A Transformer decoder block with self-attention + cross-attention.
    Supports optional gated attention for improved cross-modal learning.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.0, is_causal=True,
                 use_gated_attention=False, gating_type='elementwise'):
        super().__init__()
        self.use_gated_attention = use_gated_attention
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        # Choose between standard and gated self-attention
        if use_gated_attention:
            self.self_attn = GatedSelfAttention(dim, num_heads, dropout, is_causal=is_causal,
                                                gating_type=gating_type)
        else:
            self.self_attn = SelfAttention(dim, num_heads, dropout, is_causal=is_causal)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        # Choose between standard and gated cross-attention
        if use_gated_attention:
            self.cross_attn = GatedCrossAttention(dim, num_heads, dropout, gating_type=gating_type)
        else:
            self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.mlp = Mlp(dim, dim * mlp_ratio, drop=dropout,
                       act_layer=lambda: nn.GELU(approximate="tanh"))
        
    def forward(self, q, x, x_mask, need_weights=False) -> tuple:
        # x: (n,l,d)  p: (n,d)
        # mask: (n,l) 1--> unmasked
        q = q + self.self_attn(self.norm1(q), need_weights=need_weights)
        attn_out, attn_weights = self.cross_attn(self.norm2(q), x, x_mask, need_weights=need_weights)
        q = q + attn_out
        q = q + self.mlp(self.norm3(q))
        return q, attn_weights


class SelfAttention(nn.MultiheadAttention):
    def __init__(self, 
                 hidden_dim, 
                 num_heads, 
                 dropout=0, 
                 bias=True,  
                 batch_first=True,
                 is_causal=False):
        super().__init__(hidden_dim, num_heads, dropout, bias, batch_first=batch_first)
        self.num_heads = num_heads
        self.is_causal = is_causal
        # self.mask_converter = AttentionMaskConverter(is_causal)
        
    def forward(self, x, mask=None, need_weights=False):
        if self.is_causal and mask is None:
            B,L,D = x.shape
            mask_3d = torch.triu(
                torch.full((L, L), float('-inf'), dtype=x.dtype, device=x.device),
                diagonal=1).unsqueeze(0)
            mask_3d = mask_3d.expand(B*self.num_heads,-1,-1)
            return super().forward(x, x, x, attn_mask=mask_3d, 
                               is_causal=self.is_causal, need_weights=need_weights)[0]
        elif self.is_causal and mask is not None:
            raise NotImplementedError
        else:
            mask = ~mask.bool() if mask is not None else mask  # None
            return super().forward(x, x, x, key_padding_mask=mask, need_weights=need_weights)[0]


class CrossAttention(nn.MultiheadAttention):
    def __init__(self, 
                 hidden_dim, 
                 num_heads, 
                 dropout=0, 
                 bias=True,  
                 batch_first=True):
        super().__init__(hidden_dim, num_heads, dropout, bias, batch_first=batch_first)

    def forward(self, q, x, mask=None, need_weights=False) -> tuple:
        mask = ~mask.bool() if mask is not None else mask
        return super().forward(q, x, x, key_padding_mask=mask, need_weights=need_weights)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


############################################# Gated Attention Modules #############################################
# Based on "Gated Attention for Large Language Models" (NeurIPS 2025 Best Paper)
# Paper: https://arxiv.org/abs/2505.06708
# Key insight: Adding sigmoid gates after SDPA introduces non-linearity and enables query-dependent sparsity


class GatedSelfAttention(nn.Module):
    """
    Self-Attention with Elementwise Gating.
    
    Applies a query-dependent sigmoid gate after the standard SDPA output.
    This introduces non-linearity and enables the model to adaptively suppress
    irrelevant attention outputs, which is particularly useful for noisy EEG signals.
    
    From the Gated Attention paper:
    - Introduces non-linearity upon the low-rank mapping in softmax attention
    - Applies query-dependent sparse gating scores to modulate SDPA output
    - Mitigates 'attention sink' phenomenon and enhances training stability
    """
    def __init__(self, 
                 hidden_dim: int, 
                 num_heads: int, 
                 dropout: float = 0, 
                 bias: bool = True,  
                 batch_first: bool = True,
                 is_causal: bool = False,
                 gating_type: Literal['elementwise', 'headwise'] = 'elementwise'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.is_causal = is_causal
        self.gating_type = gating_type
        
        # Standard multi-head attention
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout, bias, batch_first=batch_first)
        
        # Gating mechanism: query-dependent gates
        if gating_type == 'elementwise':
            # Elementwise: each element of the attention output gets its own gate
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        else:  # headwise
            # Headwise: each attention head gets a single scalar gate
            self.gate_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        
        # Initialize gates to be close to 1 for stable training start
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.ones_(self.gate_proj.bias)  # Sigmoid(1) ≈ 0.73, allows gradual learning
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, need_weights: bool = False):
        """
        Args:
            x: Input tensor of shape (B, L, D)
            mask: Optional attention mask of shape (B, L), 1 for unmasked, 0 for masked
            need_weights: Whether to return attention weights
            
        Returns:
            Gated attention output of shape (B, L, D)
        """
        B, L, D = x.shape
        
        # Handle causal masking
        attn_mask = None
        key_padding_mask = None
        
        if self.is_causal and mask is None:
            attn_mask = torch.triu(
                torch.full((L, L), float('-inf'), dtype=x.dtype, device=x.device),
                diagonal=1).unsqueeze(0)
            attn_mask = attn_mask.expand(B * self.num_heads, -1, -1)
        elif mask is not None:
            key_padding_mask = ~mask.bool()
        
        # Standard attention
        if attn_mask is not None:
            attn_output, attn_weights = self.mha(x, x, x, attn_mask=attn_mask, 
                                                  is_causal=self.is_causal, need_weights=need_weights)
        else:
            attn_output, attn_weights = self.mha(x, x, x, key_padding_mask=key_padding_mask, 
                                                  need_weights=need_weights)
        
        # Compute query-dependent gates from input (not from output)
        gate = torch.sigmoid(self.gate_proj(x))  # (B, L, D) or (B, L, num_heads)
        
        # Store gate statistics for monitoring (detached to avoid affecting gradients)
        with torch.no_grad():
            self._last_gate_mean = gate.mean().item()
            self._last_gate_std = gate.std().item()
            self._last_gate_sparsity = (gate < 0.1).float().mean().item()  # % of gates < 0.1
        
        # Apply gating
        if self.gating_type == 'elementwise':
            # Elementwise gating: gate has shape (B, L, D)
            gated_output = attn_output * gate
        else:
            # Headwise gating: reshape attention output to apply per-head gates
            attn_output = attn_output.view(B, L, self.num_heads, self.head_dim)
            gate = gate.unsqueeze(-1)  # (B, L, num_heads, 1)
            gated_output = (attn_output * gate).view(B, L, D)
        
        return gated_output
    
    def get_gate_stats(self) -> dict:
        """Return the gate statistics from the last forward pass."""
        return {
            'gate_mean': getattr(self, '_last_gate_mean', 0.0),
            'gate_std': getattr(self, '_last_gate_std', 0.0),
            'gate_sparsity': getattr(self, '_last_gate_sparsity', 0.0),
        }


class GatedCrossAttention(nn.Module):
    """
    Cross-Attention with Elementwise Gating.
    
    Similar to GatedSelfAttention but for cross-attention scenarios where
    queries and keys/values come from different sources (e.g., EEG queries
    attending to text representations, or vice versa).
    
    The gate is computed from the query, making it query-dependent and allowing
    the model to dynamically control how much cross-modal information to incorporate.
    """
    def __init__(self, 
                 hidden_dim: int, 
                 num_heads: int, 
                 dropout: float = 0, 
                 bias: bool = True,  
                 batch_first: bool = True,
                 gating_type: Literal['elementwise', 'headwise'] = 'elementwise'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.gating_type = gating_type
        
        # Standard multi-head attention for cross-attention
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout, bias, batch_first=batch_first)
        
        # Gating mechanism: query-dependent gates
        if gating_type == 'elementwise':
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        else:  # headwise
            self.gate_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        
        # Initialize gates to be close to 1 for stable training start
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.ones_(self.gate_proj.bias)
        
    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                need_weights: bool = False) -> tuple:
        """
        Args:
            q: Query tensor of shape (B, Lq, D)
            kv: Key-Value tensor of shape (B, Lkv, D)
            mask: Optional key padding mask of shape (B, Lkv), 1 for unmasked, 0 for masked
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of (gated_output, attention_weights)
            - gated_output: shape (B, Lq, D)
            - attention_weights: shape (B, Lq, Lkv) or None
        """
        B, Lq, D = q.shape
        
        # Handle mask
        key_padding_mask = ~mask.bool() if mask is not None else None
        
        # Standard cross-attention
        attn_output, attn_weights = self.mha(q, kv, kv, key_padding_mask=key_padding_mask, 
                                              need_weights=need_weights)
        
        # Compute query-dependent gates
        gate = torch.sigmoid(self.gate_proj(q))  # (B, Lq, D) or (B, Lq, num_heads)
        
        # Store gate statistics for monitoring (detached to avoid affecting gradients)
        with torch.no_grad():
            self._last_gate_mean = gate.mean().item()
            self._last_gate_std = gate.std().item()
            self._last_gate_sparsity = (gate < 0.1).float().mean().item()
        
        # Apply gating
        if self.gating_type == 'elementwise':
            gated_output = attn_output * gate
        else:
            attn_output = attn_output.view(B, Lq, self.num_heads, self.head_dim)
            gate = gate.unsqueeze(-1)  # (B, Lq, num_heads, 1)
            gated_output = (attn_output * gate).view(B, Lq, D)
        
        return gated_output, attn_weights
    
    def get_gate_stats(self) -> dict:
        """Return the gate statistics from the last forward pass."""
        return {
            'gate_mean': getattr(self, '_last_gate_mean', 0.0),
            'gate_std': getattr(self, '_last_gate_std', 0.0),
            'gate_sparsity': getattr(self, '_last_gate_sparsity', 0.0),
        }