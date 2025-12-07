"""
DNA: Neural Network Pattern Discovery System

A revolutionary system for discovering patterns in neural network weights
using SIREN (Sinusoidal Representation Networks).

Main Components:
    - SIREN: Spectral DNA networks with periodic activations
    - WeightDataset: Coordinate transformation and dataset creation
    - PatternMiner: Training engine for pattern discovery
    - PatternVisualizer: Comprehensive visualization suite
    - ModelZoo: Curated catalog of tiny AI models
    - ModelRunner: Download and run tiny AI models
"""

__version__ = "2.1.0"
__author__ = "Mohammed Mishkah, Mohammed Malik Hussein"
__license__ = "MIT"

# Core SIREN networks
from .siren import (
    SineLayer,
    SpectralDNA,
    HierarchicalSpectralDNA,
    AdaptiveSpectralDNA
)

# Weight dataset and extraction
from .weight_dataset import (
    WeightCoordinateMapper,
    WeightDataset,
    WeightExtractorForSIREN,
    create_dataloader,
    visualize_coordinate_distribution
)

# Pattern mining
from .pattern_miner import PatternMiner

# Visualization
from .pattern_visualizer import PatternVisualizer

# Utilities
from .logging_utils import setup_logger

# Model Zoo - Tiny AI Play Store
from .model_zoo import (
    MODEL_ZOO,
    TinyModel,
    get_model,
    list_models,
    get_catalog,
    get_stats,
    Modality,
    Architecture,
    TaskType,
)

# Model Runner
from .model_runner import (
    TinyModelRunner,
    get_runner,
    RunResult,
    ModelStatus,
)

__all__ = [
    # SIREN networks
    "SineLayer",
    "SpectralDNA",
    "HierarchicalSpectralDNA",
    "AdaptiveSpectralDNA",

    # Dataset
    "WeightCoordinateMapper",
    "WeightDataset",
    "WeightExtractorForSIREN",
    "create_dataloader",
    "visualize_coordinate_distribution",

    # Training
    "PatternMiner",

    # Visualization
    "PatternVisualizer",

    # Utilities
    "setup_logger",
    
    # Model Zoo
    "MODEL_ZOO",
    "TinyModel",
    "get_model",
    "list_models",
    "get_catalog",
    "get_stats",
    "Modality",
    "Architecture",
    "TaskType",
    
    # Model Runner
    "TinyModelRunner",
    "get_runner",
    "RunResult",
    "ModelStatus",
]

