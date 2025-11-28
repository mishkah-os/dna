#!/usr/bin/env python3
"""
Custom DNA Architecture Example

This example shows how to create your own custom SIREN architecture.
"""

import torch
import torch.nn as nn
from pathlib import Path

from dna import SineLayer, WeightExtractorForSIREN, create_dataloader
from transformers import AutoModel


class SkipConnectionDNA(nn.Module):
    """
    Custom SIREN with skip connections.

    Architecture:
        Input → Encoder (3 layers) → Decoder (2 layers) → Output
                    ↓ (skip)
                Decoder
    """

    def __init__(self, coord_dim=4, hidden_dim=256):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim

        # Encoder with SIREN layers
        self.encoder = nn.Sequential(
            SineLayer(coord_dim, hidden_dim, is_first=True, omega_0=30.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0)
        )

        # Decoder with skip connection from input
        self.decoder = nn.Sequential(
            SineLayer(hidden_dim + coord_dim, hidden_dim, omega_0=30.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0),
            nn.Linear(hidden_dim, 1)  # Output layer
        )

    def forward(self, coords):
        # Encode
        encoded = self.encoder(coords)

        # Concatenate skip connection
        combined = torch.cat([encoded, coords], dim=-1)

        # Decode
        output = self.decoder(combined)

        return output

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class MultiScaleDNA(nn.Module):
    """
    Custom multi-scale SIREN.

    Uses different omega values for different coordinate dimensions.
    """

    def __init__(self, coord_dim=4, hidden_dim=256):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim

        # Process each coordinate dimension with different frequency
        self.coord_processors = nn.ModuleList([
            SineLayer(1, hidden_dim//4, is_first=True, omega_0=10.0),   # x: low freq
            SineLayer(1, hidden_dim//4, is_first=True, omega_0=30.0),   # y: mid freq
            SineLayer(1, hidden_dim//4, is_first=True, omega_0=100.0),  # z: high freq
            SineLayer(1, hidden_dim//4, is_first=True, omega_0=30.0),   # type: mid freq
        ])

        # Combine features
        self.combiner = nn.Sequential(
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, coords):
        # Process each dimension separately
        features = []
        for i, processor in enumerate(self.coord_processors):
            coord_i = coords[:, i:i+1]  # Extract i-th coordinate
            feat_i = processor(coord_i)
            features.append(feat_i)

        # Concatenate all features
        combined = torch.cat(features, dim=-1)

        # Process combined features
        output = self.combiner(combined)

        return output

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def train_custom_dna(dna_class, dna_name):
    """Train a custom DNA architecture."""

    print(f"\n{'='*60}")
    print(f"🧬 Training {dna_name}")
    print(f"{'='*60}")

    # Load model and extract weights
    print("\n📥 Loading model...")
    model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    extractor = WeightExtractorForSIREN(model)
    dataset, metadata = extractor.extract_to_dataset()
    print(f"✅ Extracted {len(dataset):,} weights")

    # Create data loader
    loader = create_dataloader(dataset, batch_size=8192, shuffle=True, num_workers=2)

    # Create custom DNA
    print(f"\n🧠 Creating {dna_name}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dna = dna_class(coord_dim=4, hidden_dim=128).to(device)

    print(f"   Parameters: {dna.get_num_params():,}")

    # Training setup
    optimizer = torch.optim.Adam(dna.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Train for a few epochs (demo)
    num_epochs = 5
    print(f"\n🏋️ Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        dna.train()
        total_loss = 0
        num_batches = 0

        for coords, values in loader:
            coords = coords.to(device)
            values = values.to(device)

            # Forward
            predictions = dna(coords)
            loss = criterion(predictions, values)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dna.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"   Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

    # Save model
    save_dir = Path(f'./output/custom_{dna_name.lower().replace(" ", "_")}')
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': dna.state_dict(),
        'architecture': dna_name,
        'num_params': dna.get_num_params()
    }, save_dir / 'model.pt')

    print(f"✅ Saved to {save_dir}")

    return dna


def main():
    print("🧬 Custom DNA Architecture Examples")
    print("=" * 60)

    # Train SkipConnectionDNA
    skip_dna = train_custom_dna(SkipConnectionDNA, "Skip Connection DNA")

    # Train MultiScaleDNA
    multi_dna = train_custom_dna(MultiScaleDNA, "Multi-Scale DNA")

    print("\n" + "="*60)
    print("✅ Custom architectures trained successfully!")
    print("="*60)
    print("\nKey Insights:")
    print("  - You can create any SIREN architecture using SineLayer")
    print("  - Skip connections can improve gradient flow")
    print("  - Multi-scale processing handles different frequency patterns")
    print("  - Custom architectures are easy to integrate with PatternMiner")


if __name__ == "__main__":
    main()
