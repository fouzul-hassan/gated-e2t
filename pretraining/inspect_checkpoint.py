"""Quick script to inspect checkpoint structure."""
import torch

ckpt = torch.load('Results/GLIM_Pretrain1/best_model.pth', map_location='cpu')

print("Checkpoint keys:", ckpt.keys())
print("\nModel state dict keys:")
for k, v in ckpt['model_state_dict'].items():
    if 'pos_embed' in k or 'patch_embed' in k:
        print(f"  {k}: {v.shape}")

print("\nArgs:", ckpt.get('args', {}))

# Calculate actual patch size from pos_embed shape
pos_embed_shape = ckpt['model_state_dict']['pos_embed'].shape
n_patches = pos_embed_shape[1]
print(f"\nNumber of patches in checkpoint: {n_patches}")
print(f"Expected patches with patch_size=8: {1280 // 8} = 160")
print(f"Expected patches with patch_size=80: {1280 // 80} = 16")
print(f"\n=> Actual patch_size used: 80 (not 8!)")
