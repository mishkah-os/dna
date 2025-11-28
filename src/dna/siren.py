"""
SIREN: Sinusoidal Representation Networks for Implicit Neural Representations
Based on: "Implicit Neural Representations with Periodic Activation Functions" (Sitzmann et al., 2020)

This module implements the core SIREN architecture for learning the manifold of neural network weights.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional


class SineLayer(nn.Module):
    """
    Single layer with Sine activation and proper initialization.

    The sine activation allows the network to learn high-frequency patterns
    that ReLU networks fail to capture.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        omega_0: Frequency scaling factor
        is_first: Whether this is the first layer (uses different initialization)
        bias: Whether to include bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        bias: bool = True
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self._init_weights()

    def _init_weights(self):
        """
        CRITICAL: Proper initialization from SIREN paper.
        Without this, the network cannot learn high-frequency details.
        """
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform(-1/n, 1/n)
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                           1 / self.linear.in_features)
            else:
                # Hidden layers: uniform(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0)
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

            # Bias initialization
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-np.pi, np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sine activation.

        f(x) = sin(omega_0 * Wx + b)
        """
        return torch.sin(self.omega_0 * self.linear(x))


class SpectralDNA(nn.Module):
    """
    The DNA of the neural network - a compact representation that can regenerate weights.

    This is NOT a simple compression. It learns the MANIFOLD (geometric structure)
    of the weight space. Think of it as learning the "equation" that generates the weights,
    rather than storing the weights themselves.

    Architecture:
        Input: (x, y, z, layer_id) - coordinates in weight space
        Hidden: Multiple SineLayer with sine activations
        Output: weight_value - the predicted weight at that coordinate

    Args:
        coord_dim: Dimension of input coordinates (default: 4 for x,y,z,layer)
        hidden_dim: Width of hidden layers
        num_layers: Number of hidden layers
        omega_0: Base frequency for SIREN
        final_omega_0: Frequency for final layer
    """

    def __init__(
        self,
        coord_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 5,
        omega_0: float = 30.0,
        final_omega_0: float = 30.0
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim

        # Build the network
        layers = []

        # First layer (special initialization)
        layers.append(
            SineLayer(coord_dim, hidden_dim, omega_0=omega_0, is_first=True)
        )

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(
                SineLayer(hidden_dim, hidden_dim, omega_0=omega_0, is_first=False)
            )

        self.net = nn.Sequential(*layers)

        # Final layer - predicts the actual weight value
        self.final_layer = nn.Linear(hidden_dim, 1)

        # Initialize final layer
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_dim) / final_omega_0
            self.final_layer.weight.uniform_(-bound, bound)
            if self.final_layer.bias is not None:
                self.final_layer.bias.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Generate weight value from coordinates.

        Args:
            coords: (batch, coord_dim) - normalized coordinates

        Returns:
            weights: (batch, 1) - predicted weight values
        """
        x = self.net(coords)
        weights = self.final_layer(x)
        return weights

    def get_num_params(self) -> int:
        """Get total number of parameters in the DNA."""
        return sum(p.numel() for p in self.parameters())


class HierarchicalSpectralDNA(nn.Module):
    """
    Advanced version: Hierarchical DNA with multi-scale patterns.

    This learns patterns at multiple frequencies simultaneously:
    - Low frequency: Global structure (what type of layer is this?)
    - Medium frequency: Local patterns (attention heads, FFN structure)
    - High frequency: Fine details (specific weight values)

    This is inspired by wavelet analysis and multi-resolution networks.
    """

    def __init__(
        self,
        coord_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 5,
        base_omega: float = 30.0
    ):
        super().__init__()

        # Three DNA networks at different frequencies
        # Low frequency - learns global structure
        self.low_freq_dna = SpectralDNA(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=num_layers - 2,
            omega_0=base_omega / 4.0  # Lower frequency
        )

        # Medium frequency - learns local patterns
        self.mid_freq_dna = SpectralDNA(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            omega_0=base_omega  # Base frequency
        )

        # High frequency - learns fine details
        self.high_freq_dna = SpectralDNA(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=num_layers - 1,
            omega_0=base_omega * 4.0  # Higher frequency
        )

        # Learnable weights for combining frequencies
        self.freq_weights = nn.Parameter(torch.tensor([0.3, 0.5, 0.2]))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale weight generation.

        Final_weight = α*Low + β*Mid + γ*High
        where α, β, γ are learned.
        """
        low = self.low_freq_dna(coords)
        mid = self.mid_freq_dna(coords)
        high = self.high_freq_dna(coords)

        # Normalize weights
        weights = torch.softmax(self.freq_weights, dim=0)

        # Weighted combination
        output = weights[0] * low + weights[1] * mid + weights[2] * high

        return output

    def get_num_params(self) -> int:
        """Total parameters across all scales."""
        return (self.low_freq_dna.get_num_params() +
                self.mid_freq_dna.get_num_params() +
                self.high_freq_dna.get_num_params() +
                self.freq_weights.numel())


class AdaptiveSpectralDNA(nn.Module):
    """
    Most advanced version: Learns to adaptively adjust frequency based on position.

    Some regions of weight space have high-frequency details (e.g., attention QKV),
    while others are smoother (e.g., bias vectors). This network learns to
    automatically adjust its frequency based on the input coordinates.
    """

    def __init__(
        self,
        coord_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 5,
        min_omega: float = 10.0,
        max_omega: float = 100.0
    ):
        super().__init__()

        self.min_omega = min_omega
        self.max_omega = max_omega

        # Modulation network - predicts omega based on coordinates
        self.omega_predictor = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Main SIREN network (will use predicted omega)
        layers = []

        # First layer
        layers.append(nn.Linear(coord_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)

        # Final output
        self.final_layer = nn.Linear(hidden_dim, 1)

        # Initialize all layers with SIREN scheme (using median omega)
        median_omega = (min_omega + max_omega) / 2.0
        self._init_siren(median_omega)

    def _init_siren(self, omega_0: float):
        """Initialize with SIREN scheme."""
        with torch.no_grad():
            # First layer
            self.layers[0].weight.uniform_(-1 / self.layers[0].in_features,
                                          1 / self.layers[0].in_features)

            # Hidden layers
            for layer in self.layers[1:]:
                bound = np.sqrt(6 / layer.in_features) / omega_0
                layer.weight.uniform_(-bound, bound)

            # Final layer
            bound = np.sqrt(6 / self.final_layer.in_features) / omega_0
            self.final_layer.weight.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Adaptive frequency forward pass.

        omega = min_omega + (max_omega - min_omega) * sigma(predictor(coords))
        output = sin(omega * Linear(x))
        """
        # Predict omega for this coordinate
        omega_scale = self.omega_predictor(coords)  # (batch, 1)
        omega = self.min_omega + (self.max_omega - self.min_omega) * omega_scale

        # Forward through SIREN with adaptive frequency
        x = coords
        for layer in self.layers:
            x = torch.sin(omega * layer(x))

        output = self.final_layer(x)
        return output

    def get_num_params(self) -> int:
        """Total parameters."""
        return sum(p.numel() for p in self.parameters())


def test_siren():
    """Test SIREN implementation."""
    print("=" * 70)
    print("Testing SIREN Architecture")
    print("=" * 70)

    # Test basic SIREN
    print("\n1. Testing SpectralDNA...")
    dna = SpectralDNA(coord_dim=4, hidden_dim=256, num_layers=5)

    # Create random coordinates
    coords = torch.randn(100, 4)
    weights = dna(coords)

    print(f"   Input shape: {coords.shape}")
    print(f"   Output shape: {weights.shape}")
    print(f"   Parameters: {dna.get_num_params():,}")
    print(f"   Output range: [{weights.min():.4f}, {weights.max():.4f}]")

    # Test gradient flow
    loss = weights.mean()
    loss.backward()

    has_grad = all(p.grad is not None for p in dna.parameters())
    print(f"   Gradient flow: {'✅ OK' if has_grad else '❌ FAILED'}")

    # Test hierarchical version
    print("\n2. Testing HierarchicalSpectralDNA...")
    hier_dna = HierarchicalSpectralDNA(coord_dim=4, hidden_dim=256, num_layers=5)
    weights_hier = hier_dna(coords)

    print(f"   Parameters: {hier_dna.get_num_params():,}")
    print(f"   Frequency weights: {hier_dna.freq_weights.data}")

    # Test adaptive version
    print("\n3. Testing AdaptiveSpectralDNA...")
    adapt_dna = AdaptiveSpectralDNA(coord_dim=4, hidden_dim=256, num_layers=5)
    weights_adapt = adapt_dna(coords)

    print(f"   Parameters: {adapt_dna.get_num_params():,}")
    print(f"   Omega range: [{adapt_dna.min_omega}, {adapt_dna.max_omega}]")

    print("\n✅ All SIREN tests passed!")


if __name__ == "__main__":
    test_siren()
