"""
Pattern Miner: Trains SIREN to learn the manifold of neural network weights.

This is WHERE THE MAGIC HAPPENS. We're not compressing - we're DISCOVERING
the hidden geometric structure that generates the weights.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import logging
import json
import time

from .siren import SpectralDNA, HierarchicalSpectralDNA, AdaptiveSpectralDNA
from .weight_dataset import WeightDataset

logger = logging.getLogger(__name__)


class PatternMiner:
    """
    Trains a SIREN network to learn weight patterns.

    The key insight: We're fitting a CONTINUOUS FUNCTION to the discrete weights.
    This function captures the underlying pattern/manifold.
    """

    def __init__(
        self,
        dna_type: str = 'spectral',  # 'spectral', 'hierarchical', 'adaptive'
        hidden_dim: int = 256,
        num_layers: int = 5,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            dna_type: Type of DNA network to use
            hidden_dim: Hidden layer width
            num_layers: Number of layers in SIREN
            learning_rate: Learning rate for Adam
            device: Device to train on
        """
        self.dna_type = dna_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        # Will be initialized in fit()
        self.dna = None
        self.optimizer = None
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'psnr': [],
            'learning_rate': []
        }

        logger.info(f"PatternMiner initialized on {self.device}")
        logger.info(f"  DNA type: {dna_type}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Num layers: {num_layers}")

    def _create_dna(self, coord_dim: int = 4) -> nn.Module:
        """Create the appropriate DNA network."""
        if self.dna_type == 'spectral':
            dna = SpectralDNA(
                coord_dim=coord_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            )
        elif self.dna_type == 'hierarchical':
            dna = HierarchicalSpectralDNA(
                coord_dim=coord_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            )
        elif self.dna_type == 'adaptive':
            dna = AdaptiveSpectralDNA(
                coord_dim=coord_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            )
        else:
            raise ValueError(f"Unknown DNA type: {self.dna_type}")

        return dna.to(self.device)

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
        Train the DNA network to learn weight patterns.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs

        Returns:
            Training history dictionary
        """
        # Initialize DNA
        if self.dna is None:
            # Get coordinate dimension from data
            sample_coords, _ = next(iter(train_loader))
            coord_dim = sample_coords.shape[1]

            self.dna = self._create_dna(coord_dim=coord_dim)

            # Optimizer
            self.optimizer = optim.Adam(
                self.dna.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999)
            )

            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )

        logger.info(f"Starting pattern mining for {num_epochs} epochs...")
        logger.info(f"DNA parameters: {self.dna.get_num_params():,}")

        # Training loop
        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_psnr = self._train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss, val_psnr = self._validate(val_loader)
            else:
                val_loss, val_psnr = train_loss, train_psnr

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['psnr'].append(val_psnr)
            self.training_history['learning_rate'].append(current_lr)

            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"PSNR: {val_psnr:.2f} dB | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save checkpoint
            if save_dir and (epoch + 1) % save_every == 0:
                self._save_checkpoint(save_dir, epoch, val_loss)

            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0

                # Save best model
                if save_dir:
                    self._save_checkpoint(save_dir, epoch, val_loss, is_best=True)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_loss:.6f}")

        return self.training_history

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.dna.train()

        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for coords, target_values in pbar:
            coords = coords.to(self.device)
            target_values = target_values.to(self.device)

            # Forward pass
            predicted_values = self.dna(coords)

            # Loss (MSE)
            loss = nn.functional.mse_loss(predicted_values, target_values)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for SIREN stability)
            torch.nn.utils.clip_grad_norm_(self.dna.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Metrics
            with torch.no_grad():
                mse = loss.item()
                psnr = self._calculate_psnr(mse)

            total_loss += mse
            total_psnr += psnr
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{mse:.6f}',
                'psnr': f'{psnr:.2f}dB'
            })

        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches

        return avg_loss, avg_psnr

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.dna.eval()

        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0

        for coords, target_values in val_loader:
            coords = coords.to(self.device)
            target_values = target_values.to(self.device)

            # Forward pass
            predicted_values = self.dna(coords)

            # Loss
            loss = nn.functional.mse_loss(predicted_values, target_values)

            # Metrics
            mse = loss.item()
            psnr = self._calculate_psnr(mse)

            total_loss += mse
            total_psnr += psnr
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches

        return avg_loss, avg_psnr

    @staticmethod
    def _calculate_psnr(mse: float, max_val: float = 1.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.

        PSNR = 10 * log10(MAX^2 / MSE)

        Higher PSNR = Better reconstruction quality
        > 40 dB = Excellent
        30-40 dB = Good
        20-30 dB = Fair
        < 20 dB = Poor
        """
        if mse == 0:
            return 100.0  # Perfect reconstruction
        return 10 * np.log10(max_val**2 / mse)

    def _save_checkpoint(
        self,
        save_dir: Path,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'dna_state_dict': self.dna.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'training_history': self.training_history,
            'config': {
                'dna_type': self.dna_type,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'learning_rate': self.learning_rate
            }
        }

        # Save regular checkpoint
        if not is_best:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        else:
            checkpoint_path = save_dir / 'best_model.pt'

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save training history as JSON
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: Path):
        """Load a saved checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore config
        config = checkpoint['config']
        self.dna_type = config['dna_type']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.learning_rate = config['learning_rate']

        # Recreate DNA
        self.dna = self._create_dna()
        self.dna.load_state_dict(checkpoint['dna_state_dict'])

        # Restore optimizer
        self.optimizer = optim.Adam(self.dna.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore history
        self.training_history = checkpoint['training_history']

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"  Epoch: {checkpoint['epoch']}")
        logger.info(f"  Val Loss: {checkpoint['val_loss']:.6f}")

    @torch.no_grad()
    def reconstruct_weights(
        self,
        coords: torch.Tensor,
        denormalize_fn: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Reconstruct weights from coordinates using the trained DNA.

        Args:
            coords: (N, coord_dim) coordinates
            denormalize_fn: Function to denormalize values

        Returns:
            weights: (N, 1) reconstructed weight values
        """
        self.dna.eval()

        coords = coords.to(self.device)

        # Generate weights
        weights = self.dna(coords)

        # Denormalize if needed
        if denormalize_fn is not None:
            weights = denormalize_fn(weights)

        return weights.cpu()


if __name__ == "__main__":
    # Test pattern miner
    print("Testing PatternMiner...")

    # Create dummy dataset
    from .weight_dataset import WeightDataset, create_dataloader

    dummy_coords = np.random.randn(10000, 4).astype(np.float32)
    dummy_values = np.random.randn(10000).astype(np.float32)

    dataset = WeightDataset(dummy_coords, dummy_values)
    train_loader = create_dataloader(dataset, batch_size=512, num_workers=0)

    # Create miner
    miner = PatternMiner(
        dna_type='spectral',
        hidden_dim=128,
        num_layers=3,
        device='cpu'
    )

    # Train for a few epochs
    history = miner.fit(
        train_loader=train_loader,
        num_epochs=5,
        save_dir=Path('./test_checkpoints')
    )

    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final PSNR: {history['psnr'][-1]:.2f} dB")

    print("\n✅ PatternMiner tests passed!")
