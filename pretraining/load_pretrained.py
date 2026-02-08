"""
Utility to load pretrained encoder weights into GLIM model.

After pretraining with run_pretrain.py, use this to transfer the encoder
weights to a full GLIM model for downstream EEG-to-text tasks.
"""
import torch
import logging

logger = logging.getLogger(__name__)


def load_pretrained_encoder(glim_model, pretrain_ckpt_path: str, strict: bool = False):
    """
    Load pretrained EEGEncoder weights into GLIM model.
    
    Only loads the in_blocks (EncoderBlocks) weights. The Q-Merger (out_blocks)
    and other components are trained from scratch.
    
    Args:
        glim_model: GLIM model instance
        pretrain_ckpt_path: Path to pretrained checkpoint (.pth)
        strict: If True, raise error on missing keys
        
    Returns:
        glim_model: Model with pretrained encoder weights
    """
    logger.info(f"Loading pretrained encoder from {pretrain_ckpt_path}")
    
    ckpt = torch.load(pretrain_ckpt_path, map_location='cpu')
    
    if 'encoder_state_dict' in ckpt:
        # Use the extracted encoder state dict
        encoder_state = ckpt['encoder_state_dict']
    else:
        # Extract from full model state dict
        encoder_state = {}
        prefix = 'context_encoder.'
        for key, val in ckpt['model_state_dict'].items():
            if key.startswith(prefix):
                new_key = 'in_blocks.' + key[len(prefix):]
                encoder_state[new_key] = val
    
    # Load into GLIM's eeg_encoder.in_blocks
    current_state = glim_model.eeg_encoder.state_dict()
    
    # Filter keys that exist in GLIM's encoder
    matched_keys = []
    for key, val in encoder_state.items():
        if key in current_state:
            if current_state[key].shape == val.shape:
                current_state[key] = val
                matched_keys.append(key)
            else:
                logger.warning(f"Shape mismatch for {key}: "
                             f"pretrained {val.shape} vs GLIM {current_state[key].shape}")
        else:
            logger.debug(f"Key not found in GLIM: {key}")
    
    # Load the updated state
    glim_model.eeg_encoder.load_state_dict(current_state, strict=strict)
    
    logger.info(f"Loaded {len(matched_keys)} / {len(encoder_state)} pretrained weights")
    
    if 'epoch' in ckpt:
        logger.info(f"Pretrained for {ckpt['epoch']} epochs")
    if 'accuracy' in ckpt:
        logger.info(f"Pretrained linear probe accuracy: {ckpt['accuracy']:.4f}")
    
    return glim_model


def freeze_pretrained_encoder(glim_model, freeze_in_blocks: bool = True):
    """
    Optionally freeze the pretrained encoder blocks.
    
    Args:
        glim_model: GLIM model with loaded pretrained weights
        freeze_in_blocks: If True, freeze in_blocks for fine-tuning
        
    Returns:
        glim_model: Model with frozen encoder (if requested)
    """
    if freeze_in_blocks:
        for param in glim_model.eeg_encoder.in_blocks.parameters():
            param.requires_grad = False
        logger.info("Frozen in_blocks (pretrained encoder)")
    
    return glim_model


if __name__ == "__main__":
    # Test weight loading
    import sys
    sys.path.insert(0, '..')
    
    from model.glim import GLIM
    
    # Create dummy GLIM model
    glim = GLIM(
        input_eeg_len=1280,
        hidden_eeg_len=96,
        input_text_len=96,
        input_dim=128,
        hidden_dim=256,
        embed_dim=1024,
        n_in_blocks=6,
        n_out_blocks=6,
    )
    
    # Check pretrained checkpoint exists
    ckpt_path = 'Results/GLIM_Pretrain/best_model.pth'
    import os
    if os.path.exists(ckpt_path):
        glim = load_pretrained_encoder(glim, ckpt_path)
        print("Successfully loaded pretrained weights")
    else:
        print(f"No checkpoint found at {ckpt_path}")
        print("Run pretraining first with: python run_pretrain.py")
