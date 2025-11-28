#!/usr/bin/env python3
"""
Complete Pattern Mining Pipeline for TinyBERT

This script runs the full SIREN-based pattern extraction system:
1. Load pretrained TinyBERT
2. Extract weights as coordinate dataset
3. Train SIREN DNA to learn the manifold
4. Reconstruct weights and evaluate
5. Generate comprehensive visualizations

Usage:
    python scripts/run_pattern_mining.py --model huawei-noah/TinyBERT_General_4L_312D
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModel
import argparse
import logging
from datetime import datetime

from dna.weight_dataset import (
    WeightExtractorForSIREN,
    create_dataloader,
    visualize_coordinate_distribution
)
from dna.pattern_miner import PatternMiner
from dna.pattern_visualizer import PatternVisualizer
from dna.logging_utils import setup_logger

# Setup logging
logger = setup_logger(
    "pattern_mining",
    level="INFO",
    log_file=f"logs/pattern_mining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


def main():
    parser = argparse.ArgumentParser(description="SIREN Pattern Mining for Neural Networks")

    parser.add_argument(
        "--model",
        type=str,
        default="huawei-noah/TinyBERT_General_4L_312D",
        help="HuggingFace model name or path"
    )

    parser.add_argument(
        "--dna-type",
        type=str,
        choices=['spectral', 'hierarchical', 'adaptive'],
        default='hierarchical',
        help="Type of DNA network"
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden layer dimension"
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=5,
        help="Number of SIREN layers"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./pattern_mining_output"),
        help="Output directory"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only create visualizations from existing data"
    )

    args = parser.parse_args()

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    (args.output_dir / "visualizations").mkdir(exist_ok=True)
    (args.output_dir / "data").mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("🧬 SIREN Pattern Mining System")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"DNA Type: {args.dna_type}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80)

    # ============================================================================
    # Phase 1: Load Model and Extract Weights
    # ============================================================================
    if not args.visualize_only:
        logger.info("\n" + "=" * 80)
        logger.info("📥 Phase 1: Loading Model and Extracting Weights")
        logger.info("=" * 80)

        logger.info(f"Loading model from HuggingFace: {args.model}")
        model = AutoModel.from_pretrained(args.model)
        logger.info(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Extract to dataset
        extractor = WeightExtractorForSIREN(model)
        dataset, metadata = extractor.extract_to_dataset(
            include_embeddings=True,
            include_attention=True,
            include_ffn=True
        )

        logger.info(f"✅ Dataset created: {len(dataset):,} weight values")
        logger.info(f"   Total original weights: {metadata['total_weights']:,}")

        # Save dataset and metadata
        torch.save(dataset, args.output_dir / "data" / "weight_dataset.pt")
        import json
        with open(args.output_dir / "data" / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Dataset saved to {args.output_dir / 'data'}")

        # Visualize coordinate distribution
        logger.info("Creating coordinate distribution visualization...")
        visualize_coordinate_distribution(
            dataset.coords.numpy(),
            save_path=args.output_dir / "visualizations" / "coordinate_distribution.png"
        )

    else:
        logger.info("Loading existing dataset...")
        dataset = torch.load(args.output_dir / "data" / "weight_dataset.pt")
        with open(args.output_dir / "data" / "metadata.json", 'r') as f:
            metadata = json.load(f)

    # ============================================================================
    # Phase 2: Initial Visualization (Raw Weights)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("📊 Phase 2: Visualizing Raw Weight Patterns")
    logger.info("=" * 80)

    visualizer = PatternVisualizer(output_dir=args.output_dir / "visualizations" / "raw_weights")

    coords_np = dataset.coords.numpy()
    values_np = dataset.values.numpy().flatten()

    logger.info("Creating raw weight visualizations...")
    visualizer.visualize_weight_manifold_3d(coords_np, values_np)
    visualizer.visualize_spectral_content(coords_np, values_np)
    visualizer.visualize_pattern_clustering(coords_np, values_np)

    logger.info("✅ Raw weight visualizations complete")

    if args.visualize_only:
        logger.info("Visualization-only mode complete!")
        return

    # ============================================================================
    # Phase 3: Train SIREN DNA
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("🧠 Phase 3: Training SIREN DNA to Learn Weight Manifold")
    logger.info("=" * 80)

    # Create data loaders
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")

    # Create pattern miner
    miner = PatternMiner(
        dna_type=args.dna_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        device=args.device
    )

    # Train
    logger.info("Starting training...")
    history = miner.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.output_dir / "checkpoints",
        save_every=10,
        early_stopping_patience=20
    )

    logger.info("✅ Training complete!")

    # Calculate compression ratio
    original_params = metadata['total_weights']
    dna_params = miner.dna.get_num_params()
    compression_ratio = original_params / dna_params

    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPRESSION ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Original weights: {original_params:,}")
    logger.info(f"DNA parameters: {dna_params:,}")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    logger.info(f"Size reduction: {(1 - 1/compression_ratio) * 100:.2f}%")
    logger.info("=" * 80)

    # ============================================================================
    # Phase 4: Reconstruct and Evaluate
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("🔄 Phase 4: Reconstructing Weights from DNA")
    logger.info("=" * 80)

    logger.info("Generating reconstructed weights...")

    # Use full dataset for reconstruction
    full_loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    all_reconstructed = []

    miner.dna.eval()
    with torch.no_grad():
        for coords, _ in full_loader:
            reconstructed = miner.reconstruct_weights(
                coords,
                denormalize_fn=dataset.denormalize
            )
            all_reconstructed.append(reconstructed)

    reconstructed_values = torch.cat(all_reconstructed, dim=0).numpy().flatten()

    logger.info(f"✅ Reconstructed {len(reconstructed_values):,} weights")

    # Denormalize original values for comparison
    original_values_denorm = dataset.denormalize(dataset.values).numpy().flatten()

    # ============================================================================
    # Phase 5: Visualization of Reconstruction
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("📈 Phase 5: Analyzing Reconstruction Quality")
    logger.info("=" * 80)

    recon_visualizer = PatternVisualizer(
        output_dir=args.output_dir / "visualizations" / "reconstruction"
    )

    metrics = recon_visualizer.visualize_reconstruction_quality(
        original_values_denorm,
        reconstructed_values,
        coords_np
    )

    logger.info("\n" + "=" * 80)
    logger.info("📊 RECONSTRUCTION METRICS")
    logger.info("=" * 80)
    logger.info(f"R² Score: {metrics['r2']:.6f}")
    logger.info(f"MSE: {metrics['mse']:.6f}")
    logger.info(f"MAE: {metrics['mae']:.6f}")
    logger.info(f"PSNR: {metrics['psnr']:.2f} dB")
    logger.info("=" * 80)

    # Additional reconstructed weight visualizations
    logger.info("Creating reconstructed weight visualizations...")
    recon_visualizer.visualize_weight_manifold_3d(
        coords_np,
        reconstructed_values,
        title="Reconstructed Weight Manifold"
    )

    # ============================================================================
    # Final Summary
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("✅ PATTERN MINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Original parameters: {original_params:,}")
    logger.info(f"DNA parameters: {dna_params:,}")
    logger.info(f"Compression: {compression_ratio:.2f}x ({(1 - 1/compression_ratio) * 100:.1f}% reduction)")
    logger.info(f"Reconstruction R²: {metrics['r2']:.6f}")
    logger.info(f"Reconstruction PSNR: {metrics['psnr']:.2f} dB")
    logger.info(f"\nAll outputs saved to: {args.output_dir}")
    logger.info("=" * 80)

    # Save final summary
    summary = {
        'model': args.model,
        'dna_type': args.dna_type,
        'original_parameters': original_params,
        'dna_parameters': dna_params,
        'compression_ratio': compression_ratio,
        'reconstruction_metrics': metrics,
        'training_history': history,
        'config': vars(args)
    }

    import json
    with open(args.output_dir / "summary.json", 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        json.dump(summary, f, indent=2, default=convert)

    logger.info(f"Summary saved to: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
