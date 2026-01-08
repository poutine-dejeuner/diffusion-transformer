'''
Generate samples from trained diffusion model.
'''
import torch
import numpy as np
import os
from dit import DiT
from diff_utils import Diffusion
from config import config
from utils import Config

def main():
    # Configuration
    model_dir = 'model_1'
    n_samples = 10
    batch_size = 100  # Generate in batches
    steps = 100
    
    # Load configuration
    conf = Config(config, model_dir)
    img_size = conf.img_size
    if isinstance(img_size, (tuple, list)):
        img_h, img_w = img_size
    else:
        img_h, img_w = img_size, img_size
    dim = conf.dim
    patch_size = conf.patch_size
    depth = conf.depth
    heads = conf.heads
    mlp_dim = conf.mlp_dim
    k = conf.k
    channels = conf.channels
    
    # Setup device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Initialize model
    model = DiT(img_size, dim, patch_size, depth, heads, mlp_dim, k, in_channels=channels)
    model.to(device)
    
    # Load checkpoint
    ckpt_path = os.path.join(model_dir, 'best_ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found at {ckpt_path}')
    
    print(f'Loading checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['ema'])  # Use EMA weights
    model.eval()
    print(f"Loaded checkpoint from iter {ckpt['iter']}, FID: {ckpt['fid']:.4f}")
    
    # Initialize diffusion
    diffusion = Diffusion()
    
    # Generate samples
    print(f'Generating {n_samples} samples of size ({img_h}, {img_w})...')
    all_samples = []
    n_batches = max(1, n_samples // batch_size)
    actual_batch_size = min(batch_size, n_samples)
    
    with torch.no_grad():
        for i in range(n_batches):
            print(f'Batch {i+1}/{n_batches}')
            sz = (actual_batch_size, channels, img_h, img_w)
            samples = diffusion.sample(model, sz, steps=steps)
            all_samples.append(samples.cpu().numpy())
    
    # Concatenate all samples
    all_samples = np.concatenate(all_samples, axis=0)
    print(f'Generated {all_samples.shape[0]} samples with shape {all_samples.shape}')
    
    # Save to file
    output_path = os.path.join(model_dir, 'images.npy')
    np.save(output_path, all_samples)
    print(f'Samples saved to {output_path}')

if __name__ == '__main__':
    main()
