"""
Weight Dataset: Converts neural network weights into a learnable coordinate-value dataset.

This is the KEY insight: We treat the pre-trained weights as a SIGNAL to be learned,
not as parameters to be fine-tuned.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedModel
import logging

logger = logging.getLogger(__name__)


class WeightCoordinateMapper:
    """
    Maps weight matrix indices to normalized coordinates.

    For a weight matrix W[i,j] in layer L, we create coordinates:
    - x: normalized row index
    - y: normalized column index
    - z: layer index (normalized)
    - w: weight type encoding (Q/K/V/FFN etc.)

    The normalization to [-1, 1] is CRITICAL for SIREN to work properly.
    """

    def __init__(self, max_layers: int = 12):
        self.max_layers = max_layers
        self.weight_type_encoding = {
            'query': 0.0,
            'key': 0.25,
            'value': 0.5,
            'output': 0.75,
            'intermediate': 0.33,
            'ffn_output': 0.66,
            'embedding': -0.5,
            'other': 1.0
        }

    def matrix_to_coordinates(
        self,
        matrix: np.ndarray,
        layer_idx: int,
        weight_type: str = 'other'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a weight matrix to (coordinates, values) pairs.

        Args:
            matrix: Weight matrix (rows, cols)
            layer_idx: Which layer this belongs to
            weight_type: Type of weight (query, key, value, etc.)

        Returns:
            coords: (N, 4) array of [x, y, z, w] coordinates
            values: (N,) array of weight values
        """
        rows, cols = matrix.shape

        # Create meshgrid for row/col indices
        row_indices, col_indices = np.meshgrid(
            np.arange(rows),
            np.arange(cols),
            indexing='ij'
        )

        # Normalize to [-1, 1]
        x = 2 * (row_indices / max(rows - 1, 1)) - 1  # Row coordinate
        y = 2 * (col_indices / max(cols - 1, 1)) - 1  # Col coordinate
        z = 2 * (layer_idx / max(self.max_layers - 1, 1)) - 1  # Layer coordinate
        w = self.weight_type_encoding.get(weight_type, 0.0)  # Type encoding

        # Stack coordinates
        coords = np.stack([
            x.flatten(),
            y.flatten(),
            np.full(rows * cols, z),
            np.full(rows * cols, w)
        ], axis=1).astype(np.float32)

        # Flatten values
        values = matrix.flatten().astype(np.float32)

        return coords, values


class WeightDataset(Dataset):
    """
    PyTorch Dataset for weight coordinates and values.

    Each sample is:
        Input: (x, y, z, w) - coordinate in weight space
        Target: weight_value - the actual weight at that coordinate
    """

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        normalize_values: bool = True
    ):
        """
        Args:
            coords: (N, 4) coordinate array
            values: (N,) value array
            normalize_values: Whether to normalize values (recommended for SIREN)
        """
        self.coords = torch.from_numpy(coords).float()
        self.values = torch.from_numpy(values).float().unsqueeze(1)

        # Normalize values to have zero mean and unit variance
        if normalize_values:
            self.value_mean = self.values.mean()
            self.value_std = self.values.std() + 1e-6
            self.values = (self.values - self.value_mean) / self.value_std
        else:
            self.value_mean = 0.0
            self.value_std = 1.0

        logger.info(f"WeightDataset created: {len(self)} samples")
        logger.info(f"  Coordinate range: [{self.coords.min():.2f}, {self.coords.max():.2f}]")
        logger.info(f"  Value range: [{self.values.min():.2f}, {self.values.max():.2f}]")
        logger.info(f"  Normalization: mean={self.value_mean:.6f}, std={self.value_std:.6f}")

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.values[idx]

    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original scale."""
        return normalized_values * self.value_std + self.value_mean


class WeightExtractorForSIREN:
    """
    Specialized weight extractor for SIREN-based pattern mining.

    Unlike the standard WeightExtractor, this one:
    1. Converts weights to coordinates
    2. Creates a unified dataset across all layers
    3. Provides metadata for reconstruction
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.model.eval()
        self.mapper = WeightCoordinateMapper()

    def extract_to_dataset(
        self,
        include_embeddings: bool = True,
        include_attention: bool = True,
        include_ffn: bool = True
    ) -> Tuple[WeightDataset, Dict]:
        """
        Extract all weights and convert to a coordinate dataset.

        Returns:
            dataset: WeightDataset ready for training
            metadata: Information needed for reconstruction
        """
        logger.info("Extracting weights to coordinate dataset...")

        all_coords = []
        all_values = []
        metadata = {
            'layer_info': [],
            'total_weights': 0,
            'weight_shapes': {}
        }

        # Extract embeddings
        if include_embeddings:
            logger.info("  Extracting embeddings...")
            emb_matrix = self.model.embeddings.word_embeddings.weight.detach().cpu().numpy()

            coords, values = self.mapper.matrix_to_coordinates(
                emb_matrix,
                layer_idx=-1,  # Special index for embeddings
                weight_type='embedding'
            )

            all_coords.append(coords)
            all_values.append(values)

            metadata['weight_shapes']['word_embeddings'] = emb_matrix.shape
            metadata['total_weights'] += emb_matrix.size

        # Extract transformer layers
        for layer_idx, layer in enumerate(self.model.encoder.layer):

            # Attention weights
            if include_attention:
                attention_weights = {
                    'query': layer.attention.self.query.weight.detach().cpu().numpy(),
                    'key': layer.attention.self.key.weight.detach().cpu().numpy(),
                    'value': layer.attention.self.value.weight.detach().cpu().numpy(),
                    'output': layer.attention.output.dense.weight.detach().cpu().numpy(),
                }

                for weight_type, matrix in attention_weights.items():
                    coords, values = self.mapper.matrix_to_coordinates(
                        matrix,
                        layer_idx=layer_idx,
                        weight_type=weight_type
                    )

                    all_coords.append(coords)
                    all_values.append(values)

                    key = f'layer_{layer_idx}_attn_{weight_type}'
                    metadata['weight_shapes'][key] = matrix.shape
                    metadata['total_weights'] += matrix.size

            # FFN weights
            if include_ffn:
                ffn_weights = {
                    'intermediate': layer.intermediate.dense.weight.detach().cpu().numpy(),
                    'ffn_output': layer.output.dense.weight.detach().cpu().numpy(),
                }

                for weight_type, matrix in ffn_weights.items():
                    coords, values = self.mapper.matrix_to_coordinates(
                        matrix,
                        layer_idx=layer_idx,
                        weight_type=weight_type
                    )

                    all_coords.append(coords)
                    all_values.append(values)

                    key = f'layer_{layer_idx}_ffn_{weight_type}'
                    metadata['weight_shapes'][key] = matrix.shape
                    metadata['total_weights'] += matrix.size

        # Concatenate all
        all_coords = np.vstack(all_coords)
        all_values = np.hstack(all_values)

        logger.info(f"  Total weights extracted: {metadata['total_weights']:,}")
        logger.info(f"  Coordinate array shape: {all_coords.shape}")

        # Create dataset
        dataset = WeightDataset(all_coords, all_values, normalize_values=True)

        # Add normalization info to metadata
        metadata['value_mean'] = float(dataset.value_mean)
        metadata['value_std'] = float(dataset.value_std)

        return dataset, metadata


def create_dataloader(
    dataset: WeightDataset,
    batch_size: int = 8192,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for the weight dataset.

    Note: Large batch sizes are GOOD here because we're fitting a function,
    not learning features from diverse samples.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def visualize_coordinate_distribution(coords: np.ndarray, save_path: Optional[str] = None):
    """
    Visualize the distribution of coordinates in 3D space.

    This helps understand the "shape" of the weight manifold.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 5))

    # Sample for visualization (too many points will be slow)
    sample_size = min(10000, len(coords))
    indices = np.random.choice(len(coords), sample_size, replace=False)
    coords_sample = coords[indices]

    # 3D scatter of x, y, z coordinates
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(
        coords_sample[:, 0],  # x
        coords_sample[:, 1],  # y
        coords_sample[:, 2],  # z
        c=coords_sample[:, 3],  # color by weight type
        cmap='viridis',
        s=1,
        alpha=0.5
    )
    ax1.set_xlabel('X (Row)')
    ax1.set_ylabel('Y (Col)')
    ax1.set_zlabel('Z (Layer)')
    ax1.set_title('Weight Coordinate Distribution')
    plt.colorbar(scatter, ax=ax1, label='Weight Type')

    # Distribution histograms
    ax2 = fig.add_subplot(132)
    ax2.hist(coords_sample[:, 0], bins=50, alpha=0.5, label='X (Row)', color='r')
    ax2.hist(coords_sample[:, 1], bins=50, alpha=0.5, label='Y (Col)', color='g')
    ax2.hist(coords_sample[:, 2], bins=50, alpha=0.5, label='Z (Layer)', color='b')
    ax2.set_xlabel('Coordinate Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Coordinate Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Weight type distribution
    ax3 = fig.add_subplot(133)
    unique, counts = np.unique(coords_sample[:, 3], return_counts=True)
    ax3.bar(unique, counts, color='purple', alpha=0.7)
    ax3.set_xlabel('Weight Type Encoding')
    ax3.set_ylabel('Count')
    ax3.set_title('Weight Type Distribution')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Coordinate distribution saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Test with a dummy model
    print("Testing WeightDataset creation...")

    # Create dummy weight matrices
    mapper = WeightCoordinateMapper()

    # Test single matrix
    test_matrix = np.random.randn(100, 50)
    coords, values = mapper.matrix_to_coordinates(test_matrix, layer_idx=0, weight_type='query')

    print(f"\nTest matrix shape: {test_matrix.shape}")
    print(f"Coordinates shape: {coords.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Coordinate range: [{coords.min():.2f}, {coords.max():.2f}]")

    # Create dataset
    dataset = WeightDataset(coords, values)
    print(f"\nDataset size: {len(dataset)}")

    # Test batch
    loader = create_dataloader(dataset, batch_size=256, num_workers=0)
    batch_coords, batch_values = next(iter(loader))
    print(f"Batch coordinates shape: {batch_coords.shape}")
    print(f"Batch values shape: {batch_values.shape}")

    print("\n✅ WeightDataset tests passed!")
