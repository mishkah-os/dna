"""
SQLAlchemy database models
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database.db import Base


class Model(Base):
    """ML Model metadata"""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    hf_name = Column(String(255), nullable=True)  # HuggingFace model name
    model_type = Column(String(50), nullable=False)  # BERT, GPT, Vision, etc.
    architecture = Column(String(100), nullable=True)  # encoder, decoder, etc.
    num_parameters = Column(Integer, nullable=True)
    num_layers = Column(Integer, nullable=True)
    hidden_size = Column(Integer, nullable=True)
    modality = Column(String(50), nullable=False)  # text, vision, audio
    specialty = Column(String(100), nullable=True)  # general, sentiment, etc.
    file_path = Column(String(500), nullable=True)  # Local file path
    file_size_mb = Column(Float, nullable=True)
    status = Column(String(50), default="ready")  # ready, loading, error
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="model", cascade="all, delete-orphan")


class Experiment(Base):
    """Pattern mining experiment"""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    name = Column(String(255), nullable=False)
    dna_type = Column(String(50), default="hierarchical")  # siren, hierarchical, etc.
    
    # Hyperparameters
    epochs = Column(Integer, default=100)
    learning_rate = Column(Float, default=1e-4)
    siren_layers = Column(Integer, default=5)
    siren_hidden = Column(Integer, default=256)
    
    # Results
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    final_psnr = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Progress tracking
    current_epoch = Column(Integer, default=0)
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    
    # Logs and errors
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    model = relationship("Model", back_populates="experiments")
    patterns = relationship("Pattern", back_populates="experiment", cascade="all, delete-orphan")


class Pattern(Base):
    """Discovered weight pattern"""
    __tablename__ = "patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    
    # Pattern data (stored as JSON)
    coordinates = Column(JSON, nullable=True)  # 3D coordinates for visualization
    signature = Column(JSON, nullable=True)  # Pattern fingerprint/features
    layer_data = Column(JSON, nullable=True)  # Per-layer statistics
    
    # Metrics
    pattern_type = Column(String(50), nullable=True)  # smooth, structured, noisy, etc.
    dominant_frequencies = Column(JSON, nullable=True)
    entropy_measure = Column(Float, nullable=True)
    manifold_dimensionality = Column(Integer, nullable=True)
    
    # File paths
    viz_data_path = Column(String(500), nullable=True)  # Path to full visualization data
    weights_path = Column(String(500), nullable=True)  # Path to saved DNA weights
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="patterns")
