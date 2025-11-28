# Examples

This directory contains practical examples demonstrating the DNA Pattern Mining System.

## Quick Start

All examples can be run directly:

```bash
# Make sure you're in the project root
cd /path/to/dna

# Run any example
python examples/01_basic_pattern_mining.py
```

## Available Examples

### 1. Basic Pattern Mining (`01_basic_pattern_mining.py`)

**What it does:**
- Loads TinyBERT pretrained model
- Extracts weights to coordinate dataset
- Trains basic SIREN DNA
- Analyzes compression ratio
- Creates visualizations

**Duration:** ~5 minutes (CPU) / ~2 minutes (GPU)

**Output:**
- `./output/checkpoints/` - Trained DNA models
- `./output/visualizations/` - Quality plots

**Run:**
```bash
python examples/01_basic_pattern_mining.py
```

**Key Learning:**
- End-to-end pattern mining workflow
- How compression ratio is calculated
- Quality metrics (PSNR, R²)

---

### 2. Generate Weights from DNA (`02_generate_weights_from_dna.py`)

**What it does:**
- Loads trained DNA checkpoint
- Generates new weight matrices from continuous function
- Creates weight matrices for different layers
- Visualizes generated patterns

**Prerequisites:** Run example 01 first to create checkpoint

**Duration:** ~1 minute

**Output:**
- `./output/generated_weights.png` - Single matrix visualization
- `./output/generated_weights_multilayer.png` - Multi-layer comparison

**Run:**
```bash
python examples/02_generate_weights_from_dna.py
```

**Key Learning:**
- DNA as a continuous function
- Generating weights from coordinates
- Pattern variation across layers

---

### 3. Custom DNA Architecture (`03_custom_dna_architecture.py`)

**What it does:**
- Demonstrates two custom SIREN architectures:
  - **SkipConnectionDNA**: Uses skip connections for better gradient flow
  - **MultiScaleDNA**: Processes each coordinate dimension at different frequencies
- Trains both architectures
- Saves custom models

**Duration:** ~10 minutes (CPU) / ~3 minutes (GPU)

**Output:**
- `./output/custom_skip_connection_dna/` - Skip connection model
- `./output/custom_multi_scale_dna/` - Multi-scale model

**Run:**
```bash
python examples/03_custom_dna_architecture.py
```

**Key Learning:**
- How to create custom SIREN architectures
- Using SineLayer as building block
- Advanced architectural patterns

---

## Understanding the Outputs

### Checkpoints
Trained DNA models are saved as `.pt` files containing:
- Model state dict
- Optimizer state
- Training history
- Configuration

### Visualizations

**Reconstruction Quality:**
- Scatter plot: original vs reconstructed
- Histograms: value distributions
- Residual plot: errors
- Q-Q plot: error distribution normality

**Metrics:**
- **R²**: Coefficient of determination (1.0 = perfect)
- **PSNR**: Peak Signal-to-Noise Ratio (>40 dB = excellent)
- **MAE**: Mean Absolute Error (lower = better)

## Common Issues

### Out of Memory

**Problem:** GPU runs out of memory

**Solution:**
```python
# Reduce batch size
loader = create_dataloader(dataset, batch_size=4096)  # Instead of 8192

# Or reduce model size
miner = PatternMiner(hidden_dim=64, num_layers=3)  # Smaller network
```

### Slow Training

**Problem:** Training takes too long

**Solution:**
```python
# Reduce epochs
history = miner.fit(train_loader, num_epochs=10)  # Quick demo

# Or use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Checkpoint Not Found

**Problem:** Example 02 can't find checkpoint

**Solution:**
```bash
# Make sure to run example 01 first
python examples/01_basic_pattern_mining.py

# Check that checkpoint exists
ls ./output/checkpoints/best_model.pt
```

## Next Steps

After running these examples:

1. **Read the docs:** See `docs/API_DOCUMENTATION.md` for detailed API
2. **Experiment:** Modify hyperparameters (hidden_dim, num_layers, omega_0)
3. **Try your model:** Replace TinyBERT with your own model
4. **Custom architectures:** Create your own SIREN variants

## Performance Tips

### For Best Quality:
```python
miner = PatternMiner(
    dna_type='hierarchical',  # Multi-scale
    hidden_dim=512,           # Large capacity
    num_layers=7,             # Deep network
    learning_rate=1e-4
)

history = miner.fit(train_loader, num_epochs=200)
```

### For Speed:
```python
miner = PatternMiner(
    dna_type='spectral',  # Basic SIREN
    hidden_dim=128,       # Smaller
    num_layers=3,         # Shallow
    learning_rate=1e-3    # Faster convergence
)

history = miner.fit(train_loader, num_epochs=20)
```

## Questions?

Check the documentation:
- **Quick Start:** `QUICKSTART.md`
- **Full Guide:** `SIREN_PATTERN_MINING.md`
- **API Reference:** `docs/API_DOCUMENTATION.md`
- **Theory:** `docs/THEORETICAL_FOUNDATION.md`

---

**Happy Pattern Mining! 🧬**
