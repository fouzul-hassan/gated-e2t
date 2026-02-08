"""
ZuCo EEG Dataset Loader for GLIM Pretraining.

Loads ZuCo EEG data from pickle DataFrame and converts to numpy arrays
suitable for EEG2Rep-style self-supervised pretraining.
"""
import os
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_zuco(data_path: str, val_ratio: float = 0.1, test_ratio: float = 0.1,
              seed: int = 42) -> dict:
    """
    Load ZuCo EEG dataset from pickle DataFrame.
    
    Args:
        data_path: Path to .df file (e.g., zuco_eeg_128ch_1280len.df)
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    
    Returns:
        Data dict compatible with EEG2Rep training:
        - 'All_train_data': (N_train, channels, timesteps) for SSL
        - 'train_data', 'train_label': for fine-tuning
        - 'val_data', 'val_label': for validation
        - 'test_data', 'test_label': for testing
        - 'max_len': sequence length
    """
    logger.info(f"Loading ZuCo dataset from {data_path}")
    
    # Load DataFrame
    df = pd.read_pickle(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Extract EEG arrays - each row has 'eeg' with shape (channels, timesteps)
    # ZuCo format: (128 channels, 1280 timesteps)
    eeg_list = df['eeg'].tolist()
    eeg_data = np.stack(eeg_list, axis=0)  # (N, 128, 1280)
    
    # Transpose to (N, timesteps, channels) for GLIM compatibility
    # GLIM expects: (batch, timesteps, channels) = (N, 1280, 128)
    eeg_data = eeg_data.transpose(0, 2, 1)  # (N, 1280, 128)
    
    logger.info(f"EEG data shape: {eeg_data.shape}")
    
    # Create pseudo-labels from subject IDs (for linear probe evaluation)
    if 'subject' in df.columns:
        subjects = df['subject'].values
        unique_subjects = np.unique(subjects)
        subject_to_id = {s: i for i, s in enumerate(unique_subjects)}
        labels = np.array([subject_to_id[s] for s in subjects])
        logger.info(f"Found {len(unique_subjects)} unique subjects")
    else:
        # If no subject column, use zeros
        labels = np.zeros(len(df), dtype=np.int32)
        logger.warning("No 'subject' column found, using zeros as labels")
    
    # Split data: train | val | test
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        eeg_data, labels, test_size=test_ratio, random_state=seed, stratify=labels
    )
    
    val_size = val_ratio / (1 - test_ratio)  # Adjust for remaining data
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=val_size, random_state=seed, 
        stratify=train_val_labels
    )
    
    # All training data for self-supervised learning (includes validation)
    all_train_data = np.concatenate([train_data, val_data], axis=0)
    all_train_labels = np.concatenate([train_labels, val_labels], axis=0)
    
    Data = {
        # For self-supervised pretraining
        'All_train_data': all_train_data,
        'All_train_label': all_train_labels,
        # For fine-tuning / linear probe
        'train_data': train_data,
        'train_label': train_labels,
        'val_data': val_data,
        'val_label': val_labels,
        'test_data': test_data,
        'test_label': test_labels,
        # Metadata
        'max_len': eeg_data.shape[1],  # 1280
        'num_channels': eeg_data.shape[2],  # 128
    }
    
    logger.info(f"All train: {all_train_data.shape}, Test: {test_data.shape}")
    logger.info(f"Train: {train_data.shape}, Val: {val_data.shape}")
    
    return Data


class ZuCoDataset:
    """
    PyTorch-style dataset wrapper for ZuCo EEG data.
    Compatible with EEG2Rep's dataset_class interface.
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray, patch_size: int = 8):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int32)
        self.patch_size = patch_size
    
    def __getitem__(self, idx):
        import torch
        x = torch.tensor(self.data[idx])
        y = torch.tensor(self.labels[idx])
        return x, y, idx
    
    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)
    data_path = "../data/tmp/zuco_eeg_128ch_1280len.df"
    if os.path.exists(data_path):
        data = load_zuco(data_path)
        print(f"Successfully loaded ZuCo dataset")
        print(f"All train shape: {data['All_train_data'].shape}")
        print(f"Number of labels: {data['All_train_label'].max() + 1}")
