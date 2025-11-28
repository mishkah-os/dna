#!/usr/bin/env python3
"""
Generate Weights from Trained DNA

This example shows how to use a trained DNA to generate weight matrices.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from dna import PatternMiner, WeightCoordinateMapper

def main():
    print("🧬 Generating Weights from DNA")
    print("=" * 60)

    # 1. Load trained DNA
    print("\n📥 Loading trained DNA checkpoint...")
    checkpoint_path = Path('./output/checkpoints/best_model.pt')

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please run 01_basic_pattern_mining.py first!")
        return

    miner = PatternMiner(
        dna_type='spectral',
        hidden_dim=128,
        num_layers=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    miner.load_checkpoint(checkpoint_path)
    print("✅ DNA loaded successfully")

    # 2. Generate weights for a new matrix
    print("\n🎨 Generating weight matrix (512x512)...")

    rows, cols = 512, 512
    layer_idx = 5  # Layer 5

    # Create normalized coordinate grid
    x = np.linspace(-1, 1, rows)
    y = np.linspace(-1, 1, cols)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Normalize layer coordinate
    max_layers = 12  # Assume 12 layers total
    z = 2 * (layer_idx / (max_layers - 1)) - 1

    # Create coordinate array
    coords = np.stack([
        xx.flatten(),           # x
        yy.flatten(),           # y
        np.full(rows * cols, z),  # z (layer)
        np.zeros(rows * cols)   # w (type: attention)
    ], axis=-1)

    coords = torch.from_numpy(coords).float()

    # 3. Generate weights using DNA
    print("🧬 Querying DNA network...")
    with torch.no_grad():
        weights = miner.dna(coords.to(miner.device))
        weight_matrix = weights.cpu().numpy().reshape(rows, cols)

    print(f"✅ Generated {rows}x{cols} weight matrix")
    print(f"   Min: {weight_matrix.min():.6f}")
    print(f"   Max: {weight_matrix.max():.6f}")
    print(f"   Mean: {weight_matrix.mean():.6f}")
    print(f"   Std: {weight_matrix.std():.6f}")

    # 4. Visualize the generated matrix
    print("\n📊 Creating visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Full matrix
    im1 = axes[0].imshow(weight_matrix, cmap='RdBu_r', aspect='auto')
    axes[0].set_title(f'Generated Weight Matrix ({rows}x{cols})')
    axes[0].set_xlabel('Column Index')
    axes[0].set_ylabel('Row Index')
    plt.colorbar(im1, ax=axes[0])

    # Row profile
    mid_row = weight_matrix[rows//2, :]
    axes[1].plot(mid_row)
    axes[1].set_title(f'Middle Row Profile (row {rows//2})')
    axes[1].set_xlabel('Column Index')
    axes[1].set_ylabel('Weight Value')
    axes[1].grid(True, alpha=0.3)

    # Distribution
    axes[2].hist(weight_matrix.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[2].set_title('Weight Distribution')
    axes[2].set_xlabel('Weight Value')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path('./output/generated_weights.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")

    # 5. Generate weights at different layers
    print("\n🔬 Generating weights across layers...")

    layer_indices = [0, 3, 6, 9, 11]
    fig, axes = plt.subplots(1, len(layer_indices), figsize=(20, 4))

    for idx, layer_idx in enumerate(layer_indices):
        z = 2 * (layer_idx / (max_layers - 1)) - 1

        coords = np.stack([
            xx.flatten(),
            yy.flatten(),
            np.full(rows * cols, z),
            np.zeros(rows * cols)
        ], axis=-1)

        coords = torch.from_numpy(coords).float()

        with torch.no_grad():
            weights = miner.dna(coords.to(miner.device))
            layer_matrix = weights.cpu().numpy().reshape(rows, cols)

        im = axes[idx].imshow(layer_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
        axes[idx].set_title(f'Layer {layer_idx}')
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = Path('./output/generated_weights_multilayer.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved multi-layer visualization to: {output_path}")

    print("\n✅ Done! Generated weights from continuous DNA function")


if __name__ == "__main__":
    main()
