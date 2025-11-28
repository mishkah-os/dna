#!/usr/bin/env python3
"""
Basic Pattern Mining Example

This example shows the simplest way to extract patterns from a pretrained model.
"""

import torch
from transformers import AutoModel
from pathlib import Path

from dna import (
    WeightExtractorForSIREN,
    create_dataloader,
    PatternMiner,
    PatternVisualizer
)

def main():
    print("🧬 DNA Pattern Mining - Basic Example")
    print("=" * 60)

    # 1. Load a small pretrained model
    print("\n📥 Loading TinyBERT model...")
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    model = AutoModel.from_pretrained(model_name)
    print(f"✅ Loaded {sum(p.numel() for p in model.parameters()):,} parameters")

    # 2. Extract weights to dataset
    print("\n📊 Extracting weights to coordinate dataset...")
    extractor = WeightExtractorForSIREN(model)
    dataset, metadata = extractor.extract_to_dataset(
        include_embeddings=True,
        include_attention=True,
        include_ffn=True
    )
    print(f"✅ Created dataset with {len(dataset):,} weight values")
    print(f"   Extracted {metadata['total_weights']:,} total weights")

    # 3. Create data loader
    print("\n🔄 Creating data loader...")
    loader = create_dataloader(
        dataset,
        batch_size=8192,
        shuffle=True,
        num_workers=2
    )

    # 4. Train SIREN DNA to learn patterns
    print("\n🧠 Training SIREN DNA to discover patterns...")
    miner = PatternMiner(
        dna_type='spectral',  # Basic SIREN
        hidden_dim=128,       # Smaller for quick demo
        num_layers=3,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    history = miner.fit(
        train_loader=loader,
        num_epochs=20,  # Quick demo
        save_dir=Path('./output/checkpoints')
    )

    print(f"\n✅ Training complete!")
    print(f"   Final loss: {history['train_loss'][-1]:.6f}")
    print(f"   Final PSNR: {history['psnr'][-1]:.2f} dB")

    # 5. Analyze compression
    original_params = metadata['total_weights']
    dna_params = miner.dna.get_num_params()
    compression_ratio = original_params / dna_params

    print(f"\n📊 Compression Analysis:")
    print(f"   Original: {original_params:,} parameters")
    print(f"   DNA: {dna_params:,} parameters")
    print(f"   Compression: {compression_ratio:.2f}x")
    print(f"   Size reduction: {(1 - 1/compression_ratio) * 100:.1f}%")

    # 6. Reconstruct and visualize
    print("\n🔄 Reconstructing weights...")
    reconstructed = miner.reconstruct_weights(
        dataset.coords,
        denormalize_fn=dataset.denormalize
    )

    original = dataset.denormalize(dataset.values).numpy().flatten()
    reconstructed = reconstructed.numpy().flatten()

    print("\n📈 Creating visualizations...")
    visualizer = PatternVisualizer(output_dir=Path('./output/visualizations'))

    metrics = visualizer.visualize_reconstruction_quality(
        original=original,
        reconstructed=reconstructed,
        coords=dataset.coords.numpy()
    )

    print(f"\n✅ Reconstruction Quality:")
    print(f"   R² Score: {metrics['r2']:.6f}")
    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   MAE: {metrics['mae']:.6f}")

    print("\n✅ Done! Check ./output/ for results")


if __name__ == "__main__":
    main()
