# API Documentation

**مرجع API الشامل - Complete API Reference**

This document provides detailed API reference for all public classes and functions in the DNA Pattern Mining System.

---

## Table of Contents

1. [SIREN Module (`dna.siren`)](#1-siren-module)
2. [Weight Dataset Module (`dna.weight_dataset`)](#2-weight-dataset-module)
3. [Pattern Miner Module (`dna.pattern_miner`)](#3-pattern-miner-module)
4. [Pattern Visualizer Module (`dna.pattern_visualizer`)](#4-pattern-visualizer-module)
5. [Complete Usage Examples](#5-complete-usage-examples)

---

## 1. SIREN Module

**File**: `src/dna/siren.py`

### 1.1 `SineLayer`

**الطبقة الأساسية - The fundamental building block**

```python
class SineLayer(nn.Module):
    """Single sine activation layer with proper SIREN initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False
    )
```

**Parameters:**
- `in_features` (int): Input dimension
- `out_features` (int): Output dimension
- `omega_0` (float): Frequency of sine activation. Higher = learns higher frequency patterns
  - Default: 30.0
  - Typical range: 10.0 - 100.0
- `is_first` (bool): Whether this is the first layer (uses different initialization)

**Forward:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (batch_size, in_features) input tensor

    Returns:
        (batch_size, out_features) with sine activation applied
    """
```

**Example:**
```python
layer = SineLayer(in_features=4, out_features=256, omega_0=30.0, is_first=True)
x = torch.randn(100, 4)  # 100 samples, 4D coordinates
y = layer(x)  # (100, 256)
```

---

### 1.2 `SpectralDNA`

**الشبكة الأساسية - Basic SIREN Network**

```python
class SpectralDNA(nn.Module):
    """
    Basic SIREN network for learning weight patterns.
    Uses sinusoidal activations to capture spectral information.
    """

    def __init__(
        self,
        coord_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 5,
        omega_0: float = 30.0,
        final_omega_0: float = 30.0
    )
```

**Parameters:**
- `coord_dim` (int): Coordinate dimension (typically 4: x, y, z, type)
- `hidden_dim` (int): Width of hidden layers
  - Recommended: 128-512
  - Larger = more capacity but slower
- `num_layers` (int): Depth of network
  - Recommended: 3-7
  - Deeper = captures more complex patterns
- `omega_0` (float): Frequency for hidden layers (default: 30.0)
- `final_omega_0` (float): Frequency for output layer (default: 30.0)

**Methods:**

```python
def forward(self, coords: torch.Tensor) -> torch.Tensor:
    """
    Generate weight values from coordinates.

    Args:
        coords: (N, coord_dim) normalized coordinates in [-1, 1]

    Returns:
        (N, 1) predicted weight values
    """

def get_num_params(self) -> int:
    """Returns total number of trainable parameters."""
```

**Example:**
```python
# Create DNA network
dna = SpectralDNA(
    coord_dim=4,
    hidden_dim=256,
    num_layers=5
)

# Generate weights
coords = torch.randn(10000, 4)  # 10k coordinates
weights = dna(coords)  # (10000, 1)

print(f"DNA has {dna.get_num_params():,} parameters")
```

---

### 1.3 `HierarchicalSpectralDNA`

**شبكة متعددة المستويات - Multi-Scale Network**

```python
class HierarchicalSpectralDNA(nn.Module):
    """
    Hierarchical SIREN that learns patterns at multiple frequency scales.

    Three parallel branches:
    - Low frequency (ω=10): Smooth global patterns
    - Mid frequency (ω=30): Medium-scale structure
    - High frequency (ω=100): Fine details

    Final output is weighted combination.
    """

    def __init__(
        self,
        coord_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 5,
        low_omega: float = 10.0,
        mid_omega: float = 30.0,
        high_omega: float = 100.0
    )
```

**Parameters:**
- Same as `SpectralDNA` plus:
- `low_omega` (float): Frequency for low-frequency branch (default: 10.0)
- `mid_omega` (float): Frequency for mid-frequency branch (default: 30.0)
- `high_omega` (float): Frequency for high-frequency branch (default: 100.0)

**Architecture:**
```
Input coords (4D)
    ├─> Low Freq Branch (ω=10)  ──┐
    ├─> Mid Freq Branch (ω=30)  ──┼─> Weighted Combination ─> Output
    └─> High Freq Branch (ω=100) ─┘
```

**When to use:**
- When weight patterns have structure at multiple scales
- When you need better reconstruction quality (typically +5-10 dB PSNR)
- When you can afford 3x more parameters

**Example:**
```python
dna = HierarchicalSpectralDNA(
    coord_dim=4,
    hidden_dim=256,
    num_layers=5
)

# Learns smooth trends + medium structure + fine details
weights = dna(coords)
```

---

### 1.4 `AdaptiveSpectralDNA`

**شبكة تكيفية - Location-Aware Adaptive Network**

```python
class AdaptiveSpectralDNA(nn.Module):
    """
    Adaptive SIREN that adjusts frequency based on spatial location.

    Uses a small MLP to predict optimal omega_0 for each coordinate,
    then applies position-dependent frequency scaling.
    """

    def __init__(
        self,
        coord_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 5,
        base_omega: float = 30.0,
        omega_range: Tuple[float, float] = (10.0, 100.0)
    )
```

**Parameters:**
- Same as `SpectralDNA` plus:
- `base_omega` (float): Base frequency (default: 30.0)
- `omega_range` (Tuple[float, float]): Min and max frequency range
  - Default: (10.0, 100.0)
  - Network learns to pick frequencies in this range per location

**How it works:**
1. Small MLP predicts `omega_scale(coord)` ∈ [omega_min, omega_max]
2. Main network uses adapted frequency: `ω = base_omega × scale(coord)`
3. Different regions use different frequencies automatically

**When to use:**
- When some weight regions are smooth and others are highly structured
- For maximum reconstruction quality (best PSNR)
- When you want automatic frequency tuning

**Example:**
```python
dna = AdaptiveSpectralDNA(
    coord_dim=4,
    hidden_dim=256,
    num_layers=5,
    omega_range=(10.0, 100.0)
)

# Automatically adapts frequency based on location
weights = dna(coords)
```

---

## 2. Weight Dataset Module

**File**: `src/dna/weight_dataset.py`

### 2.1 `WeightCoordinateMapper`

**نظام تحويل الإحداثيات - Coordinate Transformation System**

```python
class WeightCoordinateMapper:
    """
    Maps weight matrix indices (layer, row, col) to normalized coordinates.

    Transforms discrete matrix indices into continuous coordinate space:
    (layer_idx, i, j) → (x, y, z, w) ∈ [-1, 1]^4
    """

    def __init__(
        self,
        max_layers: int,
        weight_type_encoding: Optional[Dict[str, float]] = None
    )
```

**Parameters:**
- `max_layers` (int): Maximum number of layers in the model
- `weight_type_encoding` (Dict[str, float]): Encoding for different weight types
  - Default: `{'embedding': -0.5, 'attention': 0.0, 'ffn': 0.5, 'other': 0.0}`

**Methods:**

```python
def matrix_to_coordinates(
    self,
    matrix: np.ndarray,
    layer_idx: int,
    weight_type: str = 'other'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert weight matrix to coordinate-value pairs.

    Args:
        matrix: (rows, cols) weight matrix
        layer_idx: Layer index in model
        weight_type: 'embedding', 'attention', 'ffn', or 'other'

    Returns:
        coords: (rows*cols, 4) normalized coordinates in [-1, 1]
        values: (rows*cols,) flattened weight values
    """

def coordinates_to_matrix_shape(
    self,
    coords: np.ndarray,
    layer_idx: int
) -> Tuple[int, int]:
    """
    Infer original matrix shape from coordinates.

    Args:
        coords: (N, 4) coordinates
        layer_idx: Layer index

    Returns:
        (rows, cols) original matrix shape
    """
```

**Example:**
```python
# Create mapper
mapper = WeightCoordinateMapper(max_layers=12)

# Convert weight matrix to coordinates
weight_matrix = np.random.randn(768, 768)  # e.g., attention weights
coords, values = mapper.matrix_to_coordinates(
    matrix=weight_matrix,
    layer_idx=5,
    weight_type='attention'
)

print(coords.shape)  # (589824, 4) = 768*768 coordinates
print(values.shape)  # (589824,) weight values
print(coords.min(), coords.max())  # -1.0, 1.0
```

---

### 2.2 `WeightDataset`

**مجموعة البيانات - PyTorch Dataset for Training**

```python
class WeightDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for weight coordinates and values.
    Handles normalization and denormalization.
    """

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        normalize_values: bool = True
    )
```

**Parameters:**
- `coords` (np.ndarray): (N, coord_dim) coordinates
- `values` (np.ndarray): (N,) or (N, 1) weight values
- `normalize_values` (bool): Whether to normalize values to [-1, 1]
  - Recommended: True (improves training stability)

**Attributes:**
- `coords` (torch.Tensor): Normalized coordinates
- `values` (torch.Tensor): (Normalized) values
- `value_min`, `value_max` (float): Original value range (for denormalization)

**Methods:**

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        coords: (coord_dim,) coordinate vector
        value: (1,) weight value
    """

def __len__(self) -> int:
    """Returns number of samples."""

def normalize(self, values: torch.Tensor) -> torch.Tensor:
    """Normalize values to [-1, 1]."""

def denormalize(self, values: torch.Tensor) -> torch.Tensor:
    """Denormalize values to original range."""
```

**Example:**
```python
# Create dataset
coords = np.random.randn(100000, 4)
values = np.random.randn(100000)

dataset = WeightDataset(coords, values, normalize_values=True)

# Access samples
coord, value = dataset[0]
print(coord.shape, value.shape)  # (4,), (1,)

# Create dataloader
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8192,
    shuffle=True,
    num_workers=4
)

for batch_coords, batch_values in loader:
    # Train model
    pass
```

---

### 2.3 `WeightExtractorForSIREN`

**مستخرج الأوزان - Weight Extractor for Pretrained Models**

```python
class WeightExtractorForSIREN:
    """
    Extracts weights from pretrained models and converts to SIREN dataset.

    Supports:
    - HuggingFace Transformers (BERT, GPT, etc.)
    - PyTorch models
    - Selective extraction (embeddings, attention, FFN)
    """

    def __init__(
        self,
        model: nn.Module,
        max_layers: Optional[int] = None
    )
```

**Parameters:**
- `model` (nn.Module): Pretrained PyTorch model
- `max_layers` (int, optional): Override max layers (auto-detected if None)

**Methods:**

```python
def extract_to_dataset(
    self,
    include_embeddings: bool = True,
    include_attention: bool = True,
    include_ffn: bool = True,
    include_others: bool = False
) -> Tuple[WeightDataset, Dict]:
    """
    Extract weights and create SIREN-ready dataset.

    Args:
        include_embeddings: Include embedding weights
        include_attention: Include attention weights (Q, K, V, O)
        include_ffn: Include feed-forward weights
        include_others: Include other weights (LayerNorm, etc.)

    Returns:
        dataset: WeightDataset ready for training
        metadata: Dict with extraction statistics
            - 'total_weights': Total number of weights
            - 'num_layers': Number of layers
            - 'weight_types': Breakdown by type
            - 'model_name': Model class name
    """

def get_model_info(self) -> Dict:
    """Get basic model information."""
```

**Example:**
```python
from transformers import AutoModel

# Load pretrained model
model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

# Extract weights
extractor = WeightExtractorForSIREN(model)
dataset, metadata = extractor.extract_to_dataset(
    include_embeddings=True,
    include_attention=True,
    include_ffn=True
)

print(f"Extracted {len(dataset):,} weights")
print(f"Model has {metadata['total_weights']:,} total parameters")
print(f"Types: {metadata['weight_types']}")
```

---

### 2.4 Utility Functions

```python
def create_dataloader(
    dataset: WeightDataset,
    batch_size: int = 8192,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create optimized DataLoader for training.

    Args:
        dataset: WeightDataset
        batch_size: Batch size (default: 8192)
        shuffle: Shuffle data (default: True)
        num_workers: Number of workers (default: 4)
        pin_memory: Pin memory for GPU (default: True)

    Returns:
        DataLoader ready for training
    """

def visualize_coordinate_distribution(
    coords: np.ndarray,
    save_path: Optional[Path] = None,
    max_points: int = 10000
):
    """
    Visualize coordinate distribution in 3D space.

    Args:
        coords: (N, coord_dim) coordinates
        save_path: Where to save plot
        max_points: Max points to plot (for performance)
    """
```

---

## 3. Pattern Miner Module

**File**: `src/dna/pattern_miner.py`

### 3.1 `PatternMiner`

**محرك التدريب - Training Engine**

```python
class PatternMiner:
    """
    Trains SIREN networks to learn weight patterns.

    Handles:
    - Training loop with progress bars
    - Validation and early stopping
    - Learning rate scheduling
    - Checkpointing
    - PSNR metrics
    """

    def __init__(
        self,
        dna_type: str = 'spectral',
        hidden_dim: int = 256,
        num_layers: int = 5,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    )
```

**Parameters:**
- `dna_type` (str): Type of DNA network
  - `'spectral'`: Basic SpectralDNA
  - `'hierarchical'`: HierarchicalSpectralDNA (best quality)
  - `'adaptive'`: AdaptiveSpectralDNA (adaptive frequency)
- `hidden_dim` (int): Hidden layer dimension (128-512)
- `num_layers` (int): Number of SIREN layers (3-7)
- `learning_rate` (float): Adam learning rate
  - Recommended: 1e-4 to 1e-3
- `device` (str): 'cuda' or 'cpu'

**Methods:**

```python
def fit(
    self,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 100,
    save_dir: Optional[Path] = None,
    save_every: int = 10,
    early_stopping_patience: int = 20
) -> Dict:
    """
    Train the DNA network.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (optional)
        num_epochs: Number of epochs
        save_dir: Directory for checkpoints
        save_every: Save checkpoint every N epochs
        early_stopping_patience: Stop if no improvement for N epochs

    Returns:
        training_history: Dict with:
            - 'train_loss': List of training losses
            - 'val_loss': List of validation losses
            - 'psnr': List of PSNR values
            - 'learning_rate': List of learning rates
    """

def reconstruct_weights(
    self,
    coords: torch.Tensor,
    denormalize_fn: Optional[callable] = None
) -> torch.Tensor:
    """
    Reconstruct weights from coordinates.

    Args:
        coords: (N, coord_dim) coordinates
        denormalize_fn: Function to denormalize (e.g., dataset.denormalize)

    Returns:
        weights: (N, 1) reconstructed weight values
    """

def load_checkpoint(self, checkpoint_path: Path):
    """Load pretrained checkpoint."""

@staticmethod
def _calculate_psnr(mse: float, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(MAX^2 / MSE)

    Interpretation:
    - > 40 dB: Excellent reconstruction
    - 30-40 dB: Good reconstruction
    - 20-30 dB: Fair reconstruction
    - < 20 dB: Poor reconstruction
    """
```

**Example:**
```python
# Create pattern miner
miner = PatternMiner(
    dna_type='hierarchical',
    hidden_dim=256,
    num_layers=5,
    learning_rate=1e-4,
    device='cuda'
)

# Train
history = miner.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    save_dir=Path('./checkpoints'),
    save_every=10,
    early_stopping_patience=20
)

# Reconstruct
reconstructed = miner.reconstruct_weights(
    coords=test_coords,
    denormalize_fn=dataset.denormalize
)

print(f"Final PSNR: {history['psnr'][-1]:.2f} dB")
```

**Attributes:**
- `dna` (nn.Module): The trained DNA network
- `optimizer` (torch.optim.Optimizer): Adam optimizer
- `scheduler` (torch.optim.lr_scheduler): ReduceLROnPlateau scheduler
- `best_loss` (float): Best validation loss achieved
- `training_history` (Dict): Training metrics over time

---

## 4. Pattern Visualizer Module

**File**: `src/dna/pattern_visualizer.py`

### 4.1 `PatternVisualizer`

**محرك التصور - Visualization Engine**

```python
class PatternVisualizer:
    """
    Creates comprehensive visualizations of weight patterns.

    Visualizations:
    1. 3D manifold scatter plots
    2. Spectral analysis (FFT)
    3. Pattern clustering (t-SNE)
    4. Reconstruction quality (9-panel comparison)
    5. Training curves
    6. Layer-wise analysis
    """

    def __init__(
        self,
        output_dir: Path = Path("./visualizations"),
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 8)
    )
```

**Parameters:**
- `output_dir` (Path): Directory to save visualizations
- `dpi` (int): Resolution of saved images (default: 150)
- `figsize` (Tuple[int, int]): Default figure size in inches

---

### 4.2 Visualization Methods

#### 4.2.1 3D Manifold Visualization

```python
def visualize_weight_manifold_3d(
    self,
    coords: np.ndarray,
    values: np.ndarray,
    title: str = "Weight Manifold in 3D Space",
    max_points: int = 10000
) -> Path:
    """
    Create 3D scatter plot of weight manifold.

    Args:
        coords: (N, coord_dim) coordinates
        values: (N,) weight values
        title: Plot title
        max_points: Max points to plot (for performance)

    Returns:
        Path to saved figure

    Creates:
        Interactive 3D scatter plot with:
        - Points colored by weight value (RdBu colormap)
        - X, Y, Z axes from first 3 coordinate dimensions
        - Colorbar showing value range
    """
```

**Example:**
```python
visualizer = PatternVisualizer(output_dir=Path("./viz"))

visualizer.visualize_weight_manifold_3d(
    coords=coords_np,
    values=values_np,
    title="TinyBERT Weight Manifold"
)
# Saves to: ./viz/weight_manifold_3d.png
```

---

#### 4.2.2 Spectral Content Analysis

```python
def visualize_spectral_content(
    self,
    coords: np.ndarray,
    values: np.ndarray,
    title: str = "Spectral Content Analysis"
) -> Path:
    """
    Analyze frequency content using FFT.

    Args:
        coords: (N, coord_dim) coordinates
        values: (N,) weight values
        title: Plot title

    Returns:
        Path to saved figure

    Creates:
        2x2 subplot:
        1. Power spectrum (log scale)
        2. Phase spectrum
        3. Cumulative energy
        4. Dominant frequencies

    Shows what frequencies are present in the weight patterns.
    """
```

**Interpretation:**
- **High power at low frequencies**: Smooth, global patterns
- **High power at high frequencies**: Fine details, noise
- **Cumulative energy**: How many frequencies needed to capture X% of energy

---

#### 4.2.3 Pattern Clustering

```python
def visualize_pattern_clustering(
    self,
    coords: np.ndarray,
    values: np.ndarray,
    n_clusters: int = 5,
    max_points: int = 5000,
    title: str = "Weight Pattern Clustering"
) -> Path:
    """
    Discover natural groupings using t-SNE and clustering.

    Args:
        coords: (N, coord_dim) coordinates
        values: (N,) weight values
        n_clusters: Number of clusters for K-means
        max_points: Max points for t-SNE
        title: Plot title

    Returns:
        Path to saved figure

    Creates:
        2x2 subplot:
        1. t-SNE embedding colored by cluster
        2. t-SNE embedding colored by value
        3. Cluster distributions
        4. Cluster statistics
    """
```

**Use cases:**
- Discover if weights naturally form groups
- Identify different "types" of weights
- Validate that similar weights cluster together

---

#### 4.2.4 Reconstruction Quality (Comprehensive)

```python
def visualize_reconstruction_quality(
    self,
    original: np.ndarray,
    reconstructed: np.ndarray,
    coords: Optional[np.ndarray] = None,
    title: str = "Reconstruction Quality Analysis"
) -> Tuple[Path, Dict]:
    """
    Comprehensive reconstruction quality analysis.

    Args:
        original: (N,) original weight values
        reconstructed: (N,) reconstructed values
        coords: (N, coord_dim) coordinates (optional)
        title: Plot title

    Returns:
        save_path: Path to saved figure
        metrics: Dict with:
            - 'r2': R² score (coefficient of determination)
            - 'mse': Mean squared error
            - 'mae': Mean absolute error
            - 'psnr': Peak signal-to-noise ratio (dB)
            - 'pearson': Pearson correlation
            - 'spearman': Spearman correlation

    Creates:
        3x3 grid with 9 subplots:
        1. Scatter: Original vs Reconstructed
        2. Histogram: Original distribution
        3. Histogram: Reconstructed distribution
        4. Residual plot
        5. Q-Q plot (normality test)
        6. Error distribution
        7. Spatial error map (if coords provided)
        8. Cumulative error
        9. Metrics summary table
    """
```

**Interpretation:**
- **R² near 1.0**: Excellent reconstruction
- **PSNR > 40 dB**: Excellent quality
- **Residuals centered at 0**: Unbiased reconstruction
- **Q-Q plot linear**: Errors are normally distributed (good!)

**Example:**
```python
metrics = visualizer.visualize_reconstruction_quality(
    original=original_weights,
    reconstructed=reconstructed_weights,
    coords=coords
)

print(f"R² = {metrics['r2']:.6f}")
print(f"PSNR = {metrics['psnr']:.2f} dB")
```

---

#### 4.2.5 Training History

```python
def visualize_training_history(
    self,
    history: Dict,
    title: str = "Training History"
) -> Path:
    """
    Plot training curves.

    Args:
        history: Dict from PatternMiner.fit() with:
            - 'train_loss'
            - 'val_loss'
            - 'psnr'
            - 'learning_rate'
        title: Plot title

    Returns:
        Path to saved figure

    Creates:
        2x2 subplot:
        1. Train/Val loss
        2. PSNR over time
        3. Learning rate schedule
        4. Loss difference (overfit indicator)
    """
```

---

#### 4.2.6 Layer-wise Analysis

```python
def visualize_layer_analysis(
    self,
    coords: np.ndarray,
    values: np.ndarray,
    layer_indices: np.ndarray,
    title: str = "Layer-wise Weight Analysis"
) -> Path:
    """
    Analyze weight patterns per layer.

    Args:
        coords: (N, coord_dim) coordinates
        values: (N,) weight values
        layer_indices: (N,) layer index for each weight
        title: Plot title

    Returns:
        Path to saved figure

    Creates:
        Violin plots and statistics for each layer:
        - Value distribution per layer
        - Mean/std per layer
        - Outlier detection
    """
```

---

## 5. Complete Usage Examples

### 5.1 Basic Pattern Mining Pipeline

```python
#!/usr/bin/env python3
"""Basic pattern mining example."""

import torch
from transformers import AutoModel
from pathlib import Path

from dna.weight_dataset import WeightExtractorForSIREN, create_dataloader
from dna.pattern_miner import PatternMiner
from dna.pattern_visualizer import PatternVisualizer

# 1. Load pretrained model
model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
print(f"✅ Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")

# 2. Extract weights to dataset
extractor = WeightExtractorForSIREN(model)
dataset, metadata = extractor.extract_to_dataset(
    include_embeddings=True,
    include_attention=True,
    include_ffn=True
)
print(f"✅ Created dataset with {len(dataset):,} weight values")

# 3. Create train/val split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = create_dataloader(train_dataset, batch_size=8192, shuffle=True)
val_loader = create_dataloader(val_dataset, batch_size=8192, shuffle=False)

# 4. Train SIREN DNA
miner = PatternMiner(
    dna_type='hierarchical',  # Best quality
    hidden_dim=256,
    num_layers=5,
    learning_rate=1e-4,
    device='cuda'
)

history = miner.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    save_dir=Path('./checkpoints'),
    early_stopping_patience=20
)

print(f"✅ Training complete! Final PSNR: {history['psnr'][-1]:.2f} dB")

# 5. Reconstruct weights
coords = dataset.coords
reconstructed = miner.reconstruct_weights(coords, denormalize_fn=dataset.denormalize)
original = dataset.denormalize(dataset.values).numpy().flatten()
reconstructed = reconstructed.numpy().flatten()

# 6. Visualize results
visualizer = PatternVisualizer(output_dir=Path('./visualizations'))

# Reconstruction quality
metrics = visualizer.visualize_reconstruction_quality(original, reconstructed, coords.numpy())
print(f"R² Score: {metrics['r2']:.6f}")
print(f"PSNR: {metrics['psnr']:.2f} dB")

# Weight manifold
visualizer.visualize_weight_manifold_3d(coords.numpy(), reconstructed)

# Spectral analysis
visualizer.visualize_spectral_content(coords.numpy(), reconstructed)

# Training curves
visualizer.visualize_training_history(history)

print("✅ All visualizations saved to ./visualizations/")
```

---

### 5.2 Custom Model Integration

```python
"""Integrate with your own PyTorch model."""

import torch.nn as nn
from dna.weight_dataset import WeightCoordinateMapper, WeightDataset

# Your custom model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Extract weights manually
model = MyModel()
mapper = WeightCoordinateMapper(max_layers=3)

all_coords = []
all_values = []

for layer_idx, (name, param) in enumerate(model.named_parameters()):
    if param.dim() == 2:  # Weight matrix
        weight_matrix = param.detach().cpu().numpy()

        coords, values = mapper.matrix_to_coordinates(
            matrix=weight_matrix,
            layer_idx=layer_idx,
            weight_type='other'
        )

        all_coords.append(coords)
        all_values.append(values)

# Create dataset
import numpy as np
coords_combined = np.vstack(all_coords)
values_combined = np.concatenate(all_values)

dataset = WeightDataset(coords_combined, values_combined)

# Now train as usual
# ...
```

---

### 5.3 Advanced: Custom DNA Architecture

```python
"""Create custom DNA variant."""

import torch.nn as nn
from dna.siren import SineLayer

class MyCustomDNA(nn.Module):
    """Custom SIREN variant with skip connections."""

    def __init__(self, coord_dim=4, hidden_dim=256):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            SineLayer(coord_dim, hidden_dim, is_first=True),
            SineLayer(hidden_dim, hidden_dim),
            SineLayer(hidden_dim, hidden_dim)
        )

        # Decoder with skip connection
        self.decoder = nn.Sequential(
            SineLayer(hidden_dim + coord_dim, hidden_dim),  # Skip from input
            SineLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
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

# Use with PatternMiner
miner = PatternMiner(
    dna_type='spectral',  # Will be overridden
    hidden_dim=256,
    num_layers=5
)

# Override with custom DNA
miner.dna = MyCustomDNA(coord_dim=4, hidden_dim=256).to(miner.device)

# Train as usual
# history = miner.fit(...)
```

---

### 5.4 Inference: Generate Weights from DNA

```python
"""Use trained DNA to generate weights."""

from pathlib import Path
import torch
from dna.pattern_miner import PatternMiner
from dna.weight_dataset import WeightCoordinateMapper

# 1. Load trained DNA
miner = PatternMiner(
    dna_type='hierarchical',
    hidden_dim=256,
    num_layers=5,
    device='cuda'
)

miner.load_checkpoint(Path('./checkpoints/best_model.pt'))
print("✅ Loaded trained DNA")

# 2. Create coordinate grid for desired layer
mapper = WeightCoordinateMapper(max_layers=12)

# Generate coordinates for a 768x768 attention matrix in layer 5
rows, cols = 768, 768
layer_idx = 5

# Create mesh grid
import numpy as np
row_indices, col_indices = np.meshgrid(
    np.arange(rows),
    np.arange(cols),
    indexing='ij'
)

# Normalize coordinates
x = 2 * (row_indices.flatten() / (rows - 1)) - 1
y = 2 * (col_indices.flatten() / (cols - 1)) - 1
z = 2 * (layer_idx / 11) - 1  # Assuming 12 layers total
w = 0.0  # attention type

coords = np.stack([x, y, z, np.full_like(x, w)], axis=-1)
coords = torch.from_numpy(coords).float()

# 3. Generate weights
with torch.no_grad():
    weights = miner.dna(coords.to(miner.device))
    weights = weights.cpu().numpy()

# 4. Reshape to matrix
weight_matrix = weights.reshape(rows, cols)

print(f"✅ Generated {rows}x{cols} weight matrix")
print(f"   Min: {weight_matrix.min():.6f}")
print(f"   Max: {weight_matrix.max():.6f}")
print(f"   Mean: {weight_matrix.mean():.6f}")

# Now you can use this matrix in a model!
```

---

### 5.5 Batch Processing Multiple Models

```python
"""Process multiple models in batch."""

from pathlib import Path
from transformers import AutoModel
from dna.weight_dataset import WeightExtractorForSIREN
from dna.pattern_miner import PatternMiner
import json

models_to_process = [
    "huawei-noah/TinyBERT_General_4L_312D",
    "distilbert-base-uncased",
    "prajjwal1/bert-tiny"
]

results = []

for model_name in models_to_process:
    print(f"\n{'='*80}")
    print(f"Processing: {model_name}")
    print(f"{'='*80}")

    # Load model
    model = AutoModel.from_pretrained(model_name)

    # Extract
    extractor = WeightExtractorForSIREN(model)
    dataset, metadata = extractor.extract_to_dataset()

    # Train
    miner = PatternMiner(dna_type='hierarchical', device='cuda')

    from dna.weight_dataset import create_dataloader
    loader = create_dataloader(dataset, batch_size=8192)

    history = miner.fit(
        train_loader=loader,
        num_epochs=50,
        save_dir=Path(f'./checkpoints/{model_name.replace("/", "_")}')
    )

    # Evaluate
    coords = dataset.coords
    reconstructed = miner.reconstruct_weights(coords, dataset.denormalize)

    from sklearn.metrics import r2_score
    original = dataset.denormalize(dataset.values).numpy().flatten()
    reconstructed = reconstructed.numpy().flatten()
    r2 = r2_score(original, reconstructed)

    # Record results
    original_params = metadata['total_weights']
    dna_params = miner.dna.get_num_params()
    compression_ratio = original_params / dna_params

    results.append({
        'model': model_name,
        'original_params': original_params,
        'dna_params': dna_params,
        'compression_ratio': compression_ratio,
        'final_psnr': history['psnr'][-1],
        'r2_score': r2
    })

    print(f"✅ {model_name}: {compression_ratio:.2f}x compression, PSNR={history['psnr'][-1]:.2f}dB")

# Save results
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Processed {len(models_to_process)} models")
```

---

## 6. Performance Guidelines

### 6.1 Recommended Hyperparameters

**For small models (< 10M parameters):**
```python
miner = PatternMiner(
    dna_type='spectral',
    hidden_dim=128,
    num_layers=3,
    learning_rate=1e-3,
    batch_size=8192
)
epochs = 50
```

**For medium models (10M - 100M parameters):**
```python
miner = PatternMiner(
    dna_type='hierarchical',
    hidden_dim=256,
    num_layers=5,
    learning_rate=1e-4,
    batch_size=8192
)
epochs = 100
```

**For large models (> 100M parameters):**
```python
miner = PatternMiner(
    dna_type='hierarchical',
    hidden_dim=512,
    num_layers=7,
    learning_rate=5e-5,
    batch_size=16384
)
epochs = 200
```

### 6.2 Memory Optimization

```python
# If running out of memory:

# 1. Reduce batch size
batch_size = 4096  # Instead of 8192

# 2. Reduce hidden dim
hidden_dim = 128  # Instead of 256

# 3. Use gradient accumulation
for i, (coords, values) in enumerate(train_loader):
    loss = criterion(model(coords), values)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 4. Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(coords)
    loss = criterion(output, values)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 6.3 Speed Optimization

```python
# 1. Pin memory
loader = create_dataloader(dataset, pin_memory=True)

# 2. Use multiple workers
loader = create_dataloader(dataset, num_workers=8)

# 3. Enable cudnn benchmarking
torch.backends.cudnn.benchmark = True

# 4. Compile model (PyTorch 2.0+)
model = torch.compile(miner.dna)
```

---

## 7. Troubleshooting

### 7.1 Common Issues

**Problem**: PSNR stuck at low values (< 20 dB)
```python
# Solutions:
# 1. Increase model capacity
hidden_dim = 512  # Larger network
num_layers = 7

# 2. Try hierarchical DNA
dna_type = 'hierarchical'

# 3. Reduce learning rate
learning_rate = 5e-5

# 4. Train longer
num_epochs = 200
```

**Problem**: NaN losses during training
```python
# Solutions:
# 1. Check for inf/nan in data
assert not torch.isnan(dataset.values).any()
assert not torch.isinf(dataset.values).any()

# 2. Reduce learning rate
learning_rate = 1e-5

# 3. Enable gradient clipping (already default)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Check SIREN initialization
# Make sure omega_0 is reasonable (10-100)
```

**Problem**: Very slow training
```python
# Solutions:
# 1. Increase batch size
batch_size = 16384

# 2. Use more workers
num_workers = 8

# 3. Use GPU
device = 'cuda'

# 4. Reduce model size
hidden_dim = 128
num_layers = 3
```

---

## 8. API Quick Reference

**Core Classes:**
- `SpectralDNA`: Basic SIREN network
- `HierarchicalSpectralDNA`: Multi-scale SIREN (recommended)
- `AdaptiveSpectralDNA`: Adaptive frequency SIREN
- `WeightDataset`: PyTorch dataset for coordinates
- `WeightExtractorForSIREN`: Extract from pretrained models
- `PatternMiner`: Training engine
- `PatternVisualizer`: Visualization suite

**Key Functions:**
- `create_dataloader()`: Create optimized DataLoader
- `visualize_coordinate_distribution()`: Quick coordinate viz
- All visualizer methods for comprehensive analysis

**Typical Workflow:**
1. Extract weights → `WeightExtractorForSIREN`
2. Create dataset → `WeightDataset`
3. Train DNA → `PatternMiner.fit()`
4. Reconstruct → `PatternMiner.reconstruct_weights()`
5. Visualize → `PatternVisualizer` methods

---

## Conclusion

This API provides a complete toolkit for discovering patterns in neural network weights using SIREN-based implicit neural representations.

**القوة الحقيقية - The Real Power:**
- Not just compression, but **pattern discovery**
- Continuous function representation of discrete weights
- Multi-scale analysis from smooth trends to fine details
- Comprehensive visualizations to understand the patterns

**للاستخدام - For Usage:**
- See `scripts/run_pattern_mining.py` for complete pipeline
- See `QUICKSTART.md` for 5-minute start
- See `SIREN_PATTERN_MINING.md` for theoretical background

**الهدف - The Goal:**
Intelligence is negative entropy. These tools help you discover the mathematical laws hidden in trained weights.

---

**محمد مشكاح - محمد مالك حسين**
**DNA: Neural Network Pattern Discovery System**
