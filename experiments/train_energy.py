"""
Example Training Script: GLIM with Energy-Based Components

This demonstrates the full enhancement suite:
1. Gated Attention (NeurIPS 2025 Best Paper) - Elementwise gating
2. Nucleus Sampling (Holtzman et al. 2019) - Top-p decoding  
3. Energy-Based Loss (Novel) - EBM contrastive objective
4. ETES Evaluation (Novel) - EEG-text alignment metric

Energy-based features:
- use_energy_loss: Add EBM contrastive loss during training
- generation_strategy='energy': Rerank candidates by EEG alignment
- use_etes_eval: Compute ETES metric during validation
"""
import os
import torch
import warnings
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model.glim import GLIM
from data.datamodule import GLIMDataModule

warnings.filterwarnings("ignore", ".*when logging on epoch level in distributed.*")

# Configuration
group_name = 'energy-glim'
log_dir = './runs/' + group_name
os.makedirs(log_dir, exist_ok=True)

devices = [0]
L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision('medium')

# Logger
logger = WandbLogger(project='glim', group=group_name, save_dir=log_dir)

# Callbacks
full_val_interval = 10
callbacks = [
    ModelCheckpoint(monitor='epoch', 
        dirpath=str(logger.experiment.dir) + '-checkpoints',
        save_top_k=-1, every_n_epochs=full_val_interval),
]

# Trainer
trainer = L.Trainer(accelerator='gpu', devices=devices, logger=logger,
                    max_epochs=200, precision='bf16-mixed',
                    enable_checkpointing=True, callbacks=callbacks,
                    use_distributed_sampler=False, num_sanity_val_steps=0)

# Data Module
dm = GLIMDataModule(data_path='./data/tmp/zuco_eeg_label_8variants.df',
                    eval_noise_input=False, bsz_train=72, bsz_val=24, num_workers=4)

# =============================================================================
# GLIM Model with All Enhancements
# =============================================================================
model = GLIM(
    # EEG Encoder
    input_eeg_len=1280, hidden_eeg_len=96,
    input_text_len=96, tgt_text_len=64,
    input_dim=128, hidden_dim=256, embed_dim=1024,
    text_model_id="google/flan-t5-large",
    prompt_nums=(3, 3, 31), prompt_dropout_probs=(0.0, 1.0, 1.0),
    n_in_blocks=6, n_out_blocks=6,
    
    # Phase 1: Gated Attention
    use_gated_attention=True,
    gating_type='elementwise',
    
    # Phase 1: Nucleus Sampling (or 'energy' for energy-guided)
    generation_strategy='nucleus',  # Use 'energy' for energy-guided reranking
    top_p=0.95, temperature=0.7,
    
    # Phase 2: Energy-Based Components
    use_energy_loss=True,           # Enable EBM contrastive loss
    energy_loss_weight=0.3,         # Weight relative to CLIP loss
    energy_type='cosine',           # 'cosine', 'bilinear', or 'mlp'
    use_etes_eval=True,             # Enable ETES evaluation metric
    energy_rerank_candidates=5,     # For energy decoding
    
    # Standard config
    clip_loss_weight=0.5,
    lr=1e-4, bsz_train=dm.bsz_train, bsz_val=dm.bsz_val,
    full_val_interval=full_val_interval,
)

trainer.fit(model, datamodule=dm)
