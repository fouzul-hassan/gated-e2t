"""
Training Script: GLIM with CLS Loss + ETES Validation

This uses the default GLIM training approach:
1. CLS Loss (CLIP-style contrastive alignment) for training
2. ETES (EEG-Text Energy Score) as a validation-only metric

Key features:
    - clip_loss_weight: Weight for contrastive alignment loss (default 0.5)
    - use_etes_eval: Enable ETES metric computation during validation
    - No energy-based loss during training (simpler, more stable)
    
The ETES metric measures EEG-text alignment quality after text generation,
providing a reference-free evaluation of how well generated text matches
the source EEG signal.
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
group_name = 'glim-cls-etes'  # Experiment group name
log_dir = './runs/' + group_name
os.makedirs(log_dir, exist_ok=True)

devices = [0]  # Single GPU; use [0,1,2,...] for multi-GPU
L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision('medium')

# Logger
logger = WandbLogger(project='glim',
                     group=group_name,
                     save_dir=log_dir,
                     )

# Callbacks
full_val_interval = 10
callbacks = [
    ModelCheckpoint(monitor='epoch', 
        dirpath=str(logger.experiment.dir) + '-checkpoints',
        save_top_k=-1,
        every_n_epochs=full_val_interval,
        ),
    ]

# Trainer
trainer = L.Trainer(accelerator='gpu',
                    devices=devices,
                    logger=logger,
                    max_epochs=200,
                    precision='bf16-mixed', 
                    enable_checkpointing=True,
                    callbacks=callbacks,
                    use_distributed_sampler=False,
                    num_sanity_val_steps=0,   
                    )

# Data Module
dm = GLIMDataModule(data_path='./data/tmp/zuco_eeg_label_8variants.df',
                    eval_noise_input=False,
                    bsz_train=72,
                    bsz_val=24,
                    num_workers=4)

# =============================================================================
# GLIM Model with CLS Loss + ETES Validation
# =============================================================================

model = GLIM(
    # EEG Encoder Configuration
    input_eeg_len=1280,
    hidden_eeg_len=96,
    input_text_len=96,
    tgt_text_len=64,
    input_dim=128,
    hidden_dim=256,
    embed_dim=1024,
    
    # Language Model (frozen)
    text_model_id="google/flan-t5-large",
    
    # Prompt Configuration
    prompt_nums=(3, 3, 31),
    prompt_dropout_probs=(0.0, 1.0, 1.0),
    evaluate_prompt_embed='src',
    
    # Transformer Architecture
    n_in_blocks=6,
    n_out_blocks=6,
    in_temporal_modulate=True,
    out_is_causal=True,
    
    # Generation Configuration
    generation_strategy='beam',  # 'beam', 'nucleus', or 'greedy'
    num_beams=2,
    
    # ==========================================================
    # ETES Evaluation (validation-only metric)
    # ==========================================================
    use_etes_eval=True,  # Compute ETES during validation
    
    # ==========================================================
    # Loss Configuration (default CLS loss)
    # No energy loss - using standard CLIP-style contrastive loss
    # ==========================================================
    prompt_tuning_len=0,
    dropout=0,
    clip_loss_weight=0.5,  # Weight for contrastive alignment loss
    commitment_loss_weight=0.0,
    commitment_loss_key='mse',
    use_y_mask=False,
    
    # Training Configuration
    bsz_train=dm.bsz_train,
    bsz_val=dm.bsz_val,
    lr=1e-4,
    weight_decay=0,
    full_val_interval=full_val_interval,
    bs_retrieval=24,
)

# Start training
trainer.fit(model, datamodule=dm)
