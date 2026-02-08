"""
Evaluate pretrained GLIM encoder performance.

This script verifies that the pretrained model:
1. Loads correctly
2. Produces meaningful representations
3. Achieves good linear probe accuracy
4. Shows learned features (t-SNE visualization)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import sys
sys.path.insert(0, '..')

from Dataset.zuco_memmap import load_zuco_memmap
from pretrain_glim_encoder import GLIMEncoderPretrainer
from torch.utils.data import DataLoader


def load_checkpoint(ckpt_path):
    """Load pretrained checkpoint."""
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print(f"\n📊 Checkpoint Info:")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Linear probe accuracy: {ckpt.get('accuracy', 'N/A'):.4f}")
    print(f"  Args: {ckpt.get('args', {})}")
    
    return ckpt


def extract_features(model, loader, device, max_batches=None):
    """Extract features from pretrained encoder."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x = x.to(device)
            features = model(x)  # (B, emb_size)
            all_features.append(features.cpu())
            all_labels.append(y)
    
    return torch.cat(all_features, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def evaluate_linear_probe(train_features, train_labels, test_features, test_labels):
    """Evaluate with linear probe."""
    print("\n🔬 Linear Probe Evaluation:")
    
    # Train logistic regression
    clf = LogisticRegression(max_iter=2000, random_state=42, verbose=1)
    clf.fit(train_features, train_labels)
    
    # Predict
    train_pred = clf.predict(train_features)
    test_pred = clf.predict(test_features)
    
    train_acc = accuracy_score(train_labels, train_pred)
    test_acc = accuracy_score(test_labels, test_pred)
    
    print(f"\n✅ Results:")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Random baseline: {1.0/len(np.unique(test_labels)):.4f}")
    print(f"  Improvement over random: {test_acc / (1.0/len(np.unique(test_labels))):.2f}x")
    
    # Detailed report
    print(f"\n📋 Classification Report:")
    print(classification_report(test_labels, test_pred, digits=3))
    
    return test_acc


def visualize_features(features, labels, save_path='feature_tsne.png'):
    """Visualize features with t-SNE."""
    print("\n🎨 Creating t-SNE visualization...")
    
    # Sample for faster computation
    n_samples = min(2000, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)
    features_sample = features[indices]
    labels_sample = labels[indices]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_sample)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels_sample, cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Subject ID')
    plt.title('t-SNE Visualization of Pretrained EEG Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved to {save_path}")
    plt.close()


def analyze_feature_statistics(features):
    """Analyze feature statistics."""
    print("\n📈 Feature Statistics:")
    
    # Mean and std per dimension
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    
    print(f"  Feature dimension: {features.shape[1]}")
    print(f"  Mean (across dims): {mean.mean():.4f} ± {mean.std():.4f}")
    print(f"  Std (across dims): {std.mean():.4f} ± {std.std():.4f}")
    
    # Check for collapse
    if std.min() < 0.01:
        print(f"  ⚠️ WARNING: Some features have very low variance (min std: {std.min():.6f})")
    else:
        print(f"  ✅ No feature collapse detected (min std: {std.min():.4f})")
    
    # Effective rank
    _, s, _ = np.linalg.svd(features - features.mean(axis=0), full_matrices=False)
    s_normalized = s / s.sum()
    effective_rank = np.exp(-(s_normalized * np.log(s_normalized + 1e-10)).sum())
    
    print(f"  Effective rank: {effective_rank:.1f} / {features.shape[1]}")
    print(f"  Rank utilization: {effective_rank / features.shape[1] * 100:.1f}%")


def main():
    # Configuration
    ckpt_path = 'Results/GLIM_Pretrain1/best_model.pth'
    data_path = '../data/tmp/zuco_eeg_128ch_1280len.df'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("PRETRAINED MODEL EVALUATION")
    print("="*60)
    
    # Load checkpoint
    ckpt = load_checkpoint(ckpt_path)
    
    # Create model
    print("\n🔧 Creating model...")
    args = ckpt.get('args', {})
    
    # Infer actual patch_size from checkpoint (args may be incorrect)
    pos_embed_shape = ckpt['model_state_dict']['pos_embed'].shape
    n_patches = pos_embed_shape[1]
    actual_patch_size = 1280 // n_patches
    
    print(f"  Detected {n_patches} patches in checkpoint")
    print(f"  => Inferred patch_size: {actual_patch_size} (args said: {args.get('patch_size')})")
    
    model = GLIMEncoderPretrainer(
        in_len=1280,
        in_dim=128,
        emb_size=args.get('emb_size', 128),
        n_blocks=args.get('n_blocks', 6),
        num_heads=args.get('num_heads', 8),
        patch_size=actual_patch_size,  # Use inferred value
        mask_ratio=args.get('mask_ratio', 0.5),
        momentum=args.get('momentum', 0.99),
        use_gated_attention=args.get('use_gated_attention', False),
    ).to(device)
    
    print(f"  Model config: emb_size={args.get('emb_size')}, n_blocks={args.get('n_blocks')}, "
          f"patch_size={actual_patch_size}, num_heads={args.get('num_heads')}")
    
    # Load weights
    model.load_state_dict(ckpt['model_state_dict'])
    print("  ✅ Weights loaded successfully")
    
    # Load data
    print("\n📂 Loading data...")
    datasets = load_zuco_memmap(data_path, seed=42)
    
    train_loader = DataLoader(datasets['train'], batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(datasets['test'], batch_size=64, shuffle=False, num_workers=2)
    
    # Extract features
    print("\n🔍 Extracting features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    
    print(f"  Train features: {train_features.shape}")
    print(f"  Test features: {test_features.shape}")
    
    # Analyze statistics
    analyze_feature_statistics(test_features)
    
    # Linear probe evaluation
    test_acc = evaluate_linear_probe(train_features, train_labels, test_features, test_labels)
    
    # Visualize
    visualize_features(test_features, test_labels, 'Results/GLIM_Pretrain1/feature_tsne.png')
    
    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    if test_acc > 0.25:
        print("✅ EXCELLENT: Model learned meaningful representations!")
        print(f"   Test accuracy: {test_acc:.2%} (8x better than random)")
    elif test_acc > 0.15:
        print("✅ GOOD: Model shows decent performance")
        print(f"   Test accuracy: {test_acc:.2%}")
    elif test_acc > 0.10:
        print("⚠️ FAIR: Model learned some patterns but could improve")
        print(f"   Test accuracy: {test_acc:.2%}")
    else:
        print("❌ POOR: Model may not have learned useful features")
        print(f"   Test accuracy: {test_acc:.2%}")
    
    print("\n🎯 Recommendation:")
    if test_acc > 0.20:
        print("   This model is ready to transfer to GLIM!")
    else:
        print("   Consider retraining with adjusted hyperparameters")
    
    print("="*60)


if __name__ == '__main__':
    main()
