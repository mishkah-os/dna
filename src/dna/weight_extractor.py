"""
🔬 Weight Extractor
Extract weights from loaded models for visualization
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import torch
from dataclasses import dataclass


@dataclass
class LayerInfo:
    """Layer information"""
    name: str
    shape: tuple
    size: int
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float


class WeightExtractor:
    """
    Extract weights from loaded models.
    
    Supports HuggingFace Transformers models.
    """
    
    def __init__(self):
        self.cache_dir = Path("data/weights_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_weights(self, model_id: str) -> Dict[str, np.ndarray]:
        """
        Extract all weights from model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary mapping layer name to weight array
        """
        import sys
        sys.path.insert(0, 'src')
        from dna.model_runner import get_model_runner
        
        runner = get_model_runner()
        
        # Load model if not already loaded
        if not runner.is_loaded(model_id):
            runner.load_model(model_id)
        
        # Get model
        model = runner._models.get(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        
        # Extract all parameters
        weights = {}
        for name, param in model.named_parameters():
            # Convert to numpy
            weight_np = param.detach().cpu().numpy()
            weights[name] = weight_np
        
        return weights
    
    def get_layer_info(self, model_id: str) -> List[LayerInfo]:
        """
        Get information about all layers.
        
        Returns:
            List of LayerInfo objects
        """
        weights = self.extract_weights(model_id)
        
        layers = []
        for name, weight in weights.items():
            layers.append(LayerInfo(
                name=name,
                shape=weight.shape,
                size=weight.size,
                dtype=str(weight.dtype),
                min_val=float(weight.min()),
                max_val=float(weight.max()),
                mean_val=float(weight.mean()),
                std_val=float(weight.std())
            ))
        
        return layers
    
    def get_layer_weights(self, model_id: str, layer_name: str) -> Optional[np.ndarray]:
        """
        Get weights for specific layer.
        
        Args:
            model_id: Model identifier
            layer_name: Layer name
            
        Returns:
            Weight array or None if not found
        """
        weights = self.extract_weights(model_id)
        return weights.get(layer_name)
    
    def save_weights(self, model_id: str, output_dir: Optional[Path] = None):
        """
        Save all weights to disk.
        
        Args:
            model_id: Model identifier
            output_dir: Output directory (default: cache_dir)
        """
        if output_dir is None:
            output_dir = self.cache_dir / model_id
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        weights = self.extract_weights(model_id)
        
        for name, weight in weights.items():
            # Safe filename
            safe_name = name.replace('.', '_').replace('/', '_')
            np.save(output_dir / f"{safe_name}.npy", weight)
        
        print(f"Saved {len(weights)} layers to {output_dir}")


# Singleton instance
_extractor: Optional[WeightExtractor] = None


def get_weight_extractor() -> WeightExtractor:
    """Get singleton weight extractor"""
    global _extractor
    if _extractor is None:
        _extractor = WeightExtractor()
    return _extractor


if __name__ == "__main__":
    # Test
    extractor = WeightExtractor()
    
    # Extract from tinybert
    print("Extracting weights from tinybert...")
    layers = extractor.get_layer_info("tinybert")
    
    print(f"\nFound {len(layers)} layers:")
    for layer in layers[:5]:  # First 5
        print(f"  {layer.name}: {layer.shape}")
    
    print("\n✓ Weight extraction working!")
