"""
🎨 Weights API - Simplified
Endpoints for weight extraction and 3D visualization
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from typing import Optional
import json
import sys
import numpy as np
import math

# Ensure src is in path
if 'src' not in sys.path:
    sys.path.insert(0, 'src')

router = APIRouter()


# ============================================================================
# Helper Functions (inline to avoid import issues)
# ============================================================================

def extract_model_weights(model_id: str):
    """Extract weights from loaded model"""
    from dna.model_runner import get_model_runner
    
    runner = get_model_runner()
    
    if not runner.is_loaded(model_id):
        raise ValueError(f"Model {model_id} not loaded")
    
    model = runner._models.get(model_id)
    if model is None:
        raise ValueError(f"Model {model_id} not found")
    
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
    
    return weights


def prepare_3d_mesh(weights_array, subsample=10, height_scale=5.0, width_scale=10.0):
    """Prepare 3D mesh data from weights"""
    
    # Reshape to 2D
    if weights_array.ndim == 1:
        weights_array = weights_array.reshape(1, -1)
    elif weights_array.ndim > 2:
        weights_array = weights_array.reshape(weights_array.shape[0], -1)
    
    h, w = weights_array.shape
    
    # Subsample if large
    if h > 200 or w > 200:
        weights_array = weights_array[::subsample, ::subsample]
        h, w = weights_array.shape
    
    # Normalize
    w_min, w_max = weights_array.min(), weights_array.max()
    w_range = w_max - w_min
    if w_range < 1e-8:
        w_range = 1.0
    weights_norm = (weights_array - w_min) / w_range
    
    # Create vertices and colors
    vertices = []
    colors = []
    
    for i in range(h):
        for j in range(w):
            x = (j / w) * width_scale
            y = weights_norm[i, j] * height_scale
            z = (i / h) * width_scale
            
            vertices.append([float(x), float(y), float(z)])
            
            # Color: blue-white-red gradient
            val = weights_norm[i, j]
            if val < 0.5:
                t = val * 2
                colors.append([float(t), float(t), 1.0])
            else:
                t = (val - 0.5) * 2
                colors.append([1.0, float(1.0 - t), float(1.0 - t)])
    
    # Create indices
    indices = []
    for i in range(h - 1):
        for j in range(w - 1):
            idx = i * w + j
            indices.append([idx, idx + 1, idx + w])
            indices.append([idx + 1, idx + w + 1, idx + w])
    
    return {
        "vertices": vertices,
        "colors": colors,
        "indices": indices,
        "stats": {
            "original_shape": list(weights_array.shape),
            "mesh_shape": [h, w],
            "min": float(w_min),
            "max": float(w_max),
            "mean": float(weights_array.mean()),
            "std": float(weights_array.std()),
            "vertices_count": len(vertices),
            "triangles_count": len(indices)
        }
    }


def prepare_heatmap(weights_array, subsample: int = 1, max_side: int = 256):
    """Prepare downsampled normalized matrix for 2D heatmaps."""

    if weights_array.ndim == 1:
        weights_array = weights_array.reshape(1, -1)
    elif weights_array.ndim > 2:
        weights_array = weights_array.reshape(weights_array.shape[0], -1)

    weights_array = weights_array.astype(np.float32)

    rows, cols = weights_array.shape

    row_stride = max(1, subsample, math.ceil(rows / max_side))
    col_stride = max(1, subsample, math.ceil(cols / max_side))

    sampled = weights_array[::row_stride, ::col_stride]

    w_min, w_max = sampled.min(), sampled.max()
    w_range = w_max - w_min if (w_max - w_min) > 1e-8 else 1.0
    normalized = (sampled - w_min) / w_range

    return {
        "matrix": normalized.tolist(),
        "shape": [int(sampled.shape[0]), int(sampled.shape[1])],
        "stride": [int(row_stride), int(col_stride)],
        "stats": {
            "min": float(sampled.min()),
            "max": float(sampled.max()),
            "mean": float(sampled.mean()),
            "std": float(sampled.std()),
        }
    }


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/weights/{model_id}/layers")
async def get_model_layers(model_id: str):
    """Get all layers in a model"""
    try:
        weights = extract_model_weights(model_id)
        
        layers = []
        for name, weight in weights.items():
            layers.append({
                "name": name,
                "shape": list(weight.shape),
                "size": weight.size,
                "dtype": str(weight.dtype),
                "min": float(weight.min()),
                "max": float(weight.max()),
                "mean": float(weight.mean()),
                "std": float(weight.std())
            })
        
        return {
            "model_id": model_id,
            "layers": layers,
            "total_layers": len(layers)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {e}")


@router.get("/weights/{model_id}/layer/{layer_name}/3d")
async def get_layer_3d_data(
    model_id: str,
    layer_name: str,
    subsample: int = 10,
    height_scale: float = 5.0,
    width_scale: float = 10.0
):
    """Get 3D mesh data for a layer"""
    try:
        weights = extract_model_weights(model_id)
        
        if layer_name not in weights:
            raise HTTPException(404, f"Layer '{layer_name}' not found")
        
        layer_weights = weights[layer_name]
        
        mesh = prepare_3d_mesh(
            layer_weights,
            subsample=subsample,
            height_scale=height_scale,
            width_scale=width_scale
        )
        
        return {
            "model_id": model_id,
            "layer_name": layer_name,
            **mesh
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed: {e}")


@router.get("/weights/{model_id}/layer/{layer_name}/heatmap")
async def get_layer_heatmap(
    model_id: str,
    layer_name: str,
    subsample: int = 1,
    max_side: int = 256
):
    """Get downsampled 2D heatmap data for a layer"""

    try:
        weights = extract_model_weights(model_id)

        if layer_name not in weights:
            raise HTTPException(404, f"Layer '{layer_name}' not found")

        layer_weights = weights[layer_name]

        heatmap = prepare_heatmap(layer_weights, subsample=subsample, max_side=max_side)

        return {
            "model_id": model_id,
            "layer_name": layer_name,
            **heatmap,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed: {e}")
