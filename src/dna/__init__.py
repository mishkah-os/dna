"""
DNA: Neural Network Pattern Discovery System

A revolutionary system for discovering patterns in neural network weights
using SIREN (Sinusoidal Representation Networks).

Main Components:
    - SIREN: Spectral DNA networks with periodic activations
    - WeightDataset: Coordinate transformation and dataset creation
    - PatternMiner: Training engine for pattern discovery
    - PatternVisualizer: Comprehensive visualization suite
"""

__version__ = "2.0.0"
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
]
