"""
🎨 3D Weight Visualizer
Convert weight matrices to 3D mesh data for Three.js
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Mesh3D:
    """3D Mesh data"""
    vertices: List[List[float]]  # [[x, y, z], ...]
    colors: List[List[float]]    # [[r, g, b], ...]
    indices: List[List[int]]     # [[i1, i2, i3], ...] triangles
    stats: Dict


class WeightVisualizer3D:
    """
    Convert weight matrices to 3D visualization data.
    
    Creates surface meshes where:
    - X, Z = position in matrix
    - Y = weight value (height)
    - Color = weight value
    """
    
    def __init__(self):
        pass
    
    def prepare_3d_data(
        self,
        weights: np.ndarray,
        subsample: int = 10,
        height_scale: float = 5.0,
        width_scale: float = 10.0
    ) -> Mesh3D:
        """
        Prepare 3D mesh data from weights.
        
        Args:
            weights: Weight matrix (2D preferred)
            subsample: Subsample factor for large matrices
            height_scale: Scale for Y-axis (height)
            width_scale: Scale for X/Z axes
            
        Returns:
            Mesh3D object
        """
        # Reshape to 2D if needed
        original_shape = weights.shape
        weights = self._reshape_to_2d(weights)
        
        h, w = weights.shape
        
        # Subsample if too large
        if h > 200 or w > 200:
            weights = weights[::subsample, ::subsample]
            h, w = weights.shape
        
        # Normalize weights
        w_min, w_max = weights.min(), weights.max()
        w_range = w_max - w_min
        if w_range < 1e-8:
            w_range = 1.0
        
        weights_norm = (weights - w_min) / w_range
        
        # Create vertices and colors
        vertices = []
        colors = []
        
        for i in range(h):
            for j in range(w):
                # Position
                x = (j / w) * width_scale
                y = weights_norm[i, j] * height_scale
                z = (i / h) * width_scale
                
                vertices.append([float(x), float(y), float(z)])
                
                # Color based on value
                color = self._value_to_color(weights_norm[i, j])
                colors.append(color)
        
        # Create triangle indices
        indices = []
        for i in range(h - 1):
            for j in range(w - 1):
                # Vertex index for current position
                idx = i * w + j
                
                # Two triangles per quad
                # Triangle 1: top-left
                indices.append([idx, idx + 1, idx + w])
                # Triangle 2: bottom-right
                indices.append([idx + 1, idx + w + 1, idx + w])
        
        # Stats
        stats = {
            "original_shape": list(original_shape),
            "mesh_shape": [h, w],
            "min": float(w_min),
            "max": float(w_max),
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "vertices_count": len(vertices),
            "triangles_count": len(indices)
        }
        
        return Mesh3D(
            vertices=vertices,
            colors=colors,
            indices=indices,
            stats=stats
        )
    
    def _reshape_to_2d(self, weights: np.ndarray) -> np.ndarray:
        """Reshape weights to 2D matrix"""
        if weights.ndim == 1:
            # 1D -> row vector
            return weights.reshape(1, -1)
        elif weights.ndim == 2:
            return weights
        elif weights.ndim == 3:
            # 3D -> flatten last dimension
            return weights.reshape(weights.shape[0], -1)
        elif weights.ndim == 4:
            # 4D (conv) -> flatten
            return weights.reshape(weights.shape[0], -1)
        else:
            # Fallback: flatten to square-ish
            size = weights.size
            side = int(np.sqrt(size))
            return weights.flatten()[:side*side].reshape(side, side)
    
    def _value_to_color(self, value: float) -> List[float]:
        """
        Map normalized value [0, 1] to RGB color.
        
        Color scheme:
        - 0.0 (min) -> Blue
        - 0.5 (mid) -> White
        - 1.0 (max) -> Red
        """
        if value < 0.5:
            # Blue to White
            t = value * 2
            r = t
            g = t
            b = 1.0
        else:
            # White to Red  
            t = (value - 0.5) * 2
            r = 1.0
            g = 1.0 - t
            b = 1.0 - t
        
        return [float(r), float(g), float(b)]


# Singleton
_visualizer: WeightVisualizer3D = None


def get_3d_visualizer() -> WeightVisualizer3D:
    """Get singleton visualizer"""
    global _visualizer
    if _visualizer is None:
        _visualizer = WeightVisualizer3D()
    return _visualizer


if __name__ == "__main__":
    # Test
    visualizer = WeightVisualizer3D()
    
    # Create test weight matrix
    test_weights = np.random.randn(50, 50)
    
    print("Generating 3D mesh...")
    mesh = visualizer.prepare_3d_data(test_weights, subsample=1)
    
    print(f"\nMesh stats:")
    print(f"  Vertices: {mesh.stats['vertices_count']}")
    print(f"  Triangles: {mesh.stats['triangles_count']}")
    print(f"  Shape: {mesh.stats['mesh_shape']}")
    print(f"  Range: [{mesh.stats['min']:.3f}, {mesh.stats['max']:.3f}]")
    
    print("\n✓ 3D visualization working!")
