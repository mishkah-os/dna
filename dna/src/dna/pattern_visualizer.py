"""
Advanced Pattern Visualization System

This is your EYES into the weight manifold. We'll create visualizations that reveal:
1. The geometric structure of weight space
2. How patterns cluster and relate
3. The frequency content (spectral analysis)
4. The manifold curvature and topology
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PatternVisualizer:
    """
    Advanced visualization suite for weight patterns.

    This class creates publication-quality visualizations that help understand
    the hidden structure in neural network weights.
    """

    def __init__(self, output_dir: Path = Path("./visualizations")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PatternVisualizer initialized: {self.output_dir}")

    def visualize_weight_manifold_3d(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        title: str = "Weight Manifold in 3D Space",
        save_name: str = "weight_manifold_3d.png",
        sample_size: int = 10000
    ):
        """
        3D visualization of the weight manifold.

        Shows how weights are distributed in coordinate space,
        colored by their actual values.
        """
        # Sample for performance
        if len(coords) > sample_size:
            indices = np.random.choice(len(coords), sample_size, replace=False)
            coords_sample = coords[indices]
            values_sample = values[indices]
        else:
            coords_sample = coords
            values_sample = values

        fig = plt.figure(figsize=(20, 6))
        gs = GridSpec(1, 3, figure=fig)

        # Main 3D plot
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')

        scatter = ax1.scatter(
            coords_sample[:, 0],  # x
            coords_sample[:, 1],  # y
            coords_sample[:, 2],  # z (layer)
            c=values_sample,
            cmap='RdBu_r',
            s=1,
            alpha=0.6,
            vmin=np.percentile(values_sample, 1),
            vmax=np.percentile(values_sample, 99)
        )

        ax1.set_xlabel('X (Row Index)', fontsize=10)
        ax1.set_ylabel('Y (Col Index)', fontsize=10)
        ax1.set_zlabel('Z (Layer Index)', fontsize=10)
        ax1.set_title('3D Weight Distribution', fontsize=12, fontweight='bold')

        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.5, aspect=5)
        cbar.set_label('Weight Value', fontsize=10)

        # 2D projections with density
        ax2 = fig.add_subplot(gs[0, 1])

        hist2d = ax2.hist2d(
            coords_sample[:, 0],
            coords_sample[:, 1],
            bins=100,
            cmap='hot',
            cmin=1
        )

        ax2.set_xlabel('X (Row)', fontsize=10)
        ax2.set_ylabel('Y (Col)', fontsize=10)
        ax2.set_title('XY Projection (Density)', fontsize=12, fontweight='bold')

        cbar2 = plt.colorbar(hist2d[3], ax=ax2)
        cbar2.set_label('Point Density', fontsize=10)

        # Layer-wise distribution
        ax3 = fig.add_subplot(gs[0, 2])

        # Extract layer indices (z coordinate)
        layer_indices = coords_sample[:, 2]
        unique_layers = np.unique(layer_indices)

        for layer in unique_layers:
            mask = layer_indices == layer
            layer_values = values_sample[mask]

            ax3.hist(
                layer_values,
                bins=50,
                alpha=0.3,
                label=f'Layer {int((layer + 1) * 6):.0f}',
                density=True
            )

        ax3.set_xlabel('Weight Value', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('Layer-wise Value Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8, loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"3D manifold visualization saved: {save_path}")

        plt.close()

    def visualize_spectral_content(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        title: str = "Spectral Analysis of Weights",
        save_name: str = "spectral_analysis.png"
    ):
        """
        Frequency domain analysis of weight patterns.

        Uses FFT to reveal the frequency components in the weights.
        This shows whether weights have high-frequency details or are smooth.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Analyze different slices of the weight space
        layer_indices = np.unique(coords[:, 2])

        for idx, layer_z in enumerate(layer_indices[:6]):  # Max 6 layers
            ax = axes[idx // 3, idx % 3]

            # Get weights for this layer
            mask = coords[:, 2] == layer_z
            layer_coords = coords[mask]
            layer_values = values[mask]

            # Sort by x,y to create a grid-like structure
            sort_idx = np.lexsort((layer_coords[:, 1], layer_coords[:, 0]))
            sorted_values = layer_values[sort_idx]

            # Reshape to approximate 2D (may not be perfect grid)
            n_points = len(sorted_values)
            grid_size = int(np.sqrt(n_points))

            if grid_size > 0:
                reshaped = sorted_values[:grid_size**2].reshape(grid_size, grid_size)

                # 2D FFT
                fft2d = np.fft.fft2(reshaped)
                fft_magnitude = np.abs(np.fft.fftshift(fft2d))

                # Log scale for visualization
                fft_log = np.log1p(fft_magnitude)

                im = ax.imshow(fft_log, cmap='viridis', aspect='auto')
                ax.set_title(f'Layer {int((layer_z + 1) * 6):.0f} FFT', fontsize=10, fontweight='bold')
                ax.set_xlabel('Frequency X', fontsize=8)
                ax.set_ylabel('Frequency Y', fontsize=8)

                plt.colorbar(im, ax=ax, label='Log Magnitude')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Spectral analysis saved: {save_path}")

        plt.close()

    def visualize_pattern_clustering(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        title: str = "Weight Pattern Clustering",
        save_name: str = "pattern_clustering.png",
        sample_size: int = 5000
    ):
        """
        Cluster analysis of weight patterns using t-SNE.

        This reveals whether there are distinct "types" of weights
        (e.g., attention vs FFN, different layers, etc.)
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        # Sample
        if len(coords) > sample_size:
            indices = np.random.choice(len(coords), sample_size, replace=False)
            coords_sample = coords[indices]
            values_sample = values[indices]
        else:
            coords_sample = coords
            values_sample = values

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Combine coords and values for embedding
        features = np.column_stack([coords_sample, values_sample.reshape(-1, 1)])

        # 1. PCA embedding
        pca = PCA(n_components=2)
        pca_embedding = pca.fit_transform(features)

        scatter1 = axes[0].scatter(
            pca_embedding[:, 0],
            pca_embedding[:, 1],
            c=values_sample,
            cmap='RdBu_r',
            s=10,
            alpha=0.6
        )

        axes[0].set_title('PCA Embedding', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        plt.colorbar(scatter1, ax=axes[0], label='Weight Value')

        # 2. t-SNE embedding (colored by value)
        logger.info("Computing t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_embedding = tsne.fit_transform(features)

        scatter2 = axes[1].scatter(
            tsne_embedding[:, 0],
            tsne_embedding[:, 1],
            c=values_sample,
            cmap='RdBu_r',
            s=10,
            alpha=0.6
        )

        axes[1].set_title('t-SNE Embedding (by Value)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('t-SNE 1', fontsize=10)
        axes[1].set_ylabel('t-SNE 2', fontsize=10)
        plt.colorbar(scatter2, ax=axes[1], label='Weight Value')

        # 3. t-SNE colored by layer
        scatter3 = axes[2].scatter(
            tsne_embedding[:, 0],
            tsne_embedding[:, 1],
            c=coords_sample[:, 2],  # Layer index
            cmap='tab10',
            s=10,
            alpha=0.6
        )

        axes[2].set_title('t-SNE Embedding (by Layer)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('t-SNE 1', fontsize=10)
        axes[2].set_ylabel('t-SNE 2', fontsize=10)
        cbar3 = plt.colorbar(scatter3, ax=axes[2])
        cbar3.set_label('Layer Index', fontsize=10)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Clustering visualization saved: {save_path}")

        plt.close()

    def visualize_reconstruction_quality(
        self,
        original_values: np.ndarray,
        reconstructed_values: np.ndarray,
        coords: Optional[np.ndarray] = None,
        title: str = "Reconstruction Quality Analysis",
        save_name: str = "reconstruction_quality.png"
    ):
        """
        Comprehensive analysis of how well the DNA reconstructs the original weights.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig)

        # 1. Scatter plot: Original vs Reconstructed
        ax1 = fig.add_subplot(gs[0, 0])

        # Sample for performance
        sample_size = min(10000, len(original_values))
        indices = np.random.choice(len(original_values), sample_size, replace=False)

        ax1.scatter(
            original_values[indices],
            reconstructed_values[indices],
            alpha=0.1,
            s=1,
            color='blue'
        )

        # Perfect reconstruction line
        min_val = min(original_values.min(), reconstructed_values.min())
        max_val = max(original_values.max(), reconstructed_values.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')

        ax1.set_xlabel('Original Weight', fontsize=10)
        ax1.set_ylabel('Reconstructed Weight', fontsize=10)
        ax1.set_title('Original vs Reconstructed', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Calculate R²
        ss_res = np.sum((original_values - reconstructed_values) ** 2)
        ss_tot = np.sum((original_values - original_values.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        ax1.text(0.05, 0.95, f'R² = {r2:.6f}',
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Error distribution
        ax2 = fig.add_subplot(gs[0, 1])

        errors = original_values - reconstructed_values

        ax2.hist(errors, bins=100, color='purple', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Reconstruction Error', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add statistics
        ax2.text(0.05, 0.95,
                f'Mean: {errors.mean():.6f}\nStd: {errors.std():.6f}\nMAE: {np.abs(errors).mean():.6f}',
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 3. Absolute error histogram
        ax3 = fig.add_subplot(gs[0, 2])

        abs_errors = np.abs(errors)

        ax3.hist(abs_errors, bins=100, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Absolute Error', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # 4. Error vs Original value
        ax4 = fig.add_subplot(gs[1, 0])

        ax4.scatter(
            original_values[indices],
            abs_errors[indices],
            alpha=0.1,
            s=1,
            color='green'
        )

        ax4.set_xlabel('Original Weight Value', fontsize=10)
        ax4.set_ylabel('Absolute Error', fontsize=10)
        ax4.set_title('Error vs Weight Magnitude', fontsize=12, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        # 5. Layer-wise error (if coords available)
        ax5 = fig.add_subplot(gs[1, 1])

        if coords is not None:
            layer_indices = coords[:, 2]
            unique_layers = np.unique(layer_indices)

            layer_errors = []
            for layer in unique_layers:
                mask = layer_indices == layer
                layer_err = np.abs(errors[mask]).mean()
                layer_errors.append(layer_err)

            ax5.bar(range(len(unique_layers)), layer_errors, color='teal', alpha=0.7)
            ax5.set_xlabel('Layer Index', fontsize=10)
            ax5.set_ylabel('Mean Absolute Error', fontsize=10)
            ax5.set_title('Error by Layer', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, 'Coordinates not provided',
                    ha='center', va='center', fontsize=12)
            ax5.axis('off')

        # 6. Cumulative error plot
        ax6 = fig.add_subplot(gs[1, 2])

        sorted_errors = np.sort(abs_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

        ax6.plot(sorted_errors, cumulative, linewidth=2, color='red')
        ax6.set_xlabel('Absolute Error', fontsize=10)
        ax6.set_ylabel('Cumulative Percentage (%)', fontsize=10)
        ax6.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # Mark percentiles
        for percentile in [50, 90, 95, 99]:
            idx = int(len(sorted_errors) * percentile / 100)
            error_val = sorted_errors[idx]
            ax6.axvline(x=error_val, color='gray', linestyle=':', alpha=0.5)
            ax6.text(error_val, percentile, f'{percentile}%',
                    fontsize=8, rotation=90, va='bottom')

        # 7-9. Residual plots
        # Q-Q plot
        ax7 = fig.add_subplot(gs[2, 0])

        from scipy import stats
        stats.probplot(errors, dist="norm", plot=ax7)
        ax7.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # Hexbin plot
        ax8 = fig.add_subplot(gs[2, 1])

        hexbin = ax8.hexbin(
            original_values,
            reconstructed_values,
            gridsize=50,
            cmap='YlOrRd',
            mincnt=1
        )

        ax8.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2)
        ax8.set_xlabel('Original', fontsize=10)
        ax8.set_ylabel('Reconstructed', fontsize=10)
        ax8.set_title('Density Plot', fontsize=12, fontweight='bold')
        plt.colorbar(hexbin, ax=ax8, label='Count')

        # Summary statistics table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        # Calculate metrics
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        max_error = np.max(np.abs(errors))
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else np.inf

        stats_text = f"""
        RECONSTRUCTION METRICS
        ═══════════════════════
        R² Score:        {r2:.6f}
        MSE:             {mse:.6f}
        RMSE:            {rmse:.6f}
        MAE:             {mae:.6f}
        Max Error:       {max_error:.6f}
        PSNR:            {psnr:.2f} dB

        Error Percentiles:
        50% (Median):    {np.percentile(abs_errors, 50):.6f}
        90%:             {np.percentile(abs_errors, 90):.6f}
        95%:             {np.percentile(abs_errors, 95):.6f}
        99%:             {np.percentile(abs_errors, 99):.6f}
        """

        ax9.text(0.1, 0.9, stats_text,
                transform=ax9.transAxes,
                fontsize=10,
                verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reconstruction quality visualization saved: {save_path}")

        plt.close()

        return {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'psnr': psnr
        }

    def create_comprehensive_report(
        self,
        coords: np.ndarray,
        original_values: np.ndarray,
        reconstructed_values: Optional[np.ndarray] = None
    ):
        """
        Create all visualizations in one go.
        """
        logger.info("Creating comprehensive visualization report...")

        # 1. Manifold visualization
        self.visualize_weight_manifold_3d(coords, original_values)

        # 2. Spectral analysis
        self.visualize_spectral_content(coords, original_values)

        # 3. Clustering
        self.visualize_pattern_clustering(coords, original_values)

        # 4. Reconstruction quality (if available)
        if reconstructed_values is not None:
            metrics = self.visualize_reconstruction_quality(
                original_values,
                reconstructed_values,
                coords
            )

            logger.info(f"Reconstruction metrics: R²={metrics['r2']:.6f}, PSNR={metrics['psnr']:.2f}dB")

        logger.info(f"All visualizations saved to: {self.output_dir}")


if __name__ == "__main__":
    # Test visualizations with synthetic data
    print("Testing PatternVisualizer...")

    # Create synthetic weight data
    n_samples = 5000

    coords = np.random.randn(n_samples, 4).astype(np.float32)
    coords[:, :3] = np.clip(coords[:, :3], -1, 1)  # Normalize

    values = np.sin(5 * coords[:, 0]) * np.cos(3 * coords[:, 1]) + 0.1 * np.random.randn(n_samples)

    # Add some noise to create reconstruction scenario
    reconstructed = values + 0.05 * np.random.randn(n_samples)

    # Create visualizer
    viz = PatternVisualizer(output_dir=Path("./test_visualizations"))

    # Test comprehensive report
    viz.create_comprehensive_report(coords, values, reconstructed)

    print("\n✅ All visualizations created successfully!")
    print(f"Check: {viz.output_dir}")
