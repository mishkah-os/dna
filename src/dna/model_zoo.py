"""
🏪 Tiny AI Model Zoo
Curated catalog of small AI models that run on personal devices.

This is the "Play Store" of tiny AI - download and run with one click!
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class Modality(str, Enum):
    """Model modality types"""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class Architecture(str, Enum):
    """Model architecture types"""
    ENCODER = "encoder"  # BERT-like
    DECODER = "decoder"  # GPT-like
    ENCODER_DECODER = "encoder-decoder"  # T5-like
    CNN = "cnn"
    VIT = "vit"
    HYBRID = "hybrid"


class TaskType(str, Enum):
    """Supported task types"""
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_EMBEDDINGS = "text-embeddings"
    QUESTION_ANSWERING = "question-answering"
    FILL_MASK = "fill-mask"
    IMAGE_CLASSIFICATION = "image-classification"
    SPEECH_TO_TEXT = "speech-to-text"


@dataclass
class TinyModel:
    """A tiny AI model in our zoo"""
    # Identity
    id: str  # Short unique ID
    name: str  # Display name
    hf_name: str  # HuggingFace model name
    
    # Technical
    params_millions: float
    architecture: Architecture
    modality: Modality
    tasks: List[TaskType]
    
    # Metadata
    description: str
    num_layers: int = 0
    hidden_size: int = 0
    family: str = ""
    specialty: str = "general"
    
    # Requirements
    requires_gpu: bool = False
    estimated_ram_gb: float = 1.0
    
    # Status
    tier: int = 1  # 1 = must-have, 2 = good, 3 = advanced
    tested: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "hf_name": self.hf_name,
            "params_millions": self.params_millions,
            "architecture": self.architecture.value,
            "modality": self.modality.value,
            "tasks": [t.value for t in self.tasks],
            "description": self.description,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "family": self.family,
            "specialty": self.specialty,
            "requires_gpu": self.requires_gpu,
            "estimated_ram_gb": self.estimated_ram_gb,
            "tier": self.tier,
            "tested": self.tested,
        }


# ============================================================================
# 🏪 THE TINY AI MODEL ZOO
# ============================================================================

MODEL_ZOO: Dict[str, TinyModel] = {
    
    # ========== TIER 1: MUST-HAVE (Start here) ==========
    
    "tinybert": TinyModel(
        id="tinybert",
        name="TinyBERT",
        hf_name="huawei-noah/TinyBERT_General_4L_312D",
        params_millions=14.5,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.FILL_MASK, TaskType.TEXT_EMBEDDINGS, TaskType.TEXT_CLASSIFICATION],
        description="TinyBERT 4-layer - fast general-purpose encoder",
        num_layers=4,
        hidden_size=312,
        family="BERT",
        specialty="general",
        requires_gpu=False,
        estimated_ram_gb=0.5,
        tier=1,
        tested=True,
    ),
    
    "electra-small": TinyModel(
        id="electra-small",
        name="ELECTRA Small",
        hf_name="google/electra-small-discriminator",
        params_millions=14.0,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.FILL_MASK, TaskType.TEXT_EMBEDDINGS, TaskType.TEXT_CLASSIFICATION],
        description="ELECTRA - efficient discriminative training, same size as TinyBERT",
        num_layers=12,
        hidden_size=256,
        family="ELECTRA",
        specialty="general",
        requires_gpu=False,
        estimated_ram_gb=0.5,
        tier=1,
        tested=False,
    ),
    
    "minilm-l6": TinyModel(
        id="minilm-l6",
        name="MiniLM L6",
        hf_name="microsoft/MiniLM-L6-v2",
        params_millions=22.7,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_EMBEDDINGS],
        description="Optimized for sentence embeddings and similarity",
        num_layers=6,
        hidden_size=384,
        family="BERT",
        specialty="sentence-embeddings",
        requires_gpu=False,
        estimated_ram_gb=0.5,
        tier=1,
        tested=False,
    ),
    
    "distilbert": TinyModel(
        id="distilbert",
        name="DistilBERT",
        hf_name="distilbert-base-uncased",
        params_millions=66.0,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.FILL_MASK, TaskType.TEXT_EMBEDDINGS, TaskType.TEXT_CLASSIFICATION],
        description="Distilled BERT - strong baseline, 60% smaller than BERT",
        num_layers=6,
        hidden_size=768,
        family="BERT",
        specialty="general",
        requires_gpu=False,
        estimated_ram_gb=1.0,
        tier=1,
        tested=False,
    ),
    
    "distilgpt2": TinyModel(
        id="distilgpt2",
        name="DistilGPT-2",
        hf_name="distilgpt2",
        params_millions=82.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION],
        description="Distilled GPT-2 - text generation on CPU",
        num_layers=6,
        hidden_size=768,
        family="GPT",
        specialty="text-generation",
        requires_gpu=False,
        estimated_ram_gb=1.5,
        tier=1,
        tested=False,
    ),
    
    # ========== TIER 2: VISION ==========
    
    "mobilenet-v2": TinyModel(
        id="mobilenet-v2",
        name="MobileNet V2",
        hf_name="google/mobilenet_v2_1.0_224",
        params_millions=3.5,
        architecture=Architecture.CNN,
        modality=Modality.VISION,
        tasks=[TaskType.IMAGE_CLASSIFICATION],
        description="Ultra-light CNN for image classification",
        num_layers=53,
        hidden_size=0,
        family="MobileNet",
        specialty="image-classification",
        requires_gpu=False,
        estimated_ram_gb=0.3,
        tier=2,
        tested=False,
    ),
    
    "deit-tiny": TinyModel(
        id="deit-tiny",
        name="DeiT Tiny",
        hf_name="facebook/deit-tiny-patch16-224",
        params_millions=5.7,
        architecture=Architecture.VIT,
        modality=Modality.VISION,
        tasks=[TaskType.IMAGE_CLASSIFICATION],
        description="Vision Transformer - tiny version",
        num_layers=12,
        hidden_size=192,
        family="ViT",
        specialty="image-classification",
        requires_gpu=False,
        estimated_ram_gb=0.5,
        tier=2,
        tested=False,
    ),
    
    "mobilevit-small": TinyModel(
        id="mobilevit-small",
        name="MobileViT Small",
        hf_name="apple/mobilevit-small",
        params_millions=5.6,
        architecture=Architecture.HYBRID,
        modality=Modality.VISION,
        tasks=[TaskType.IMAGE_CLASSIFICATION],
        description="CNN + Transformer hybrid by Apple",
        num_layers=0,
        hidden_size=0,
        family="MobileViT",
        specialty="image-classification",
        requires_gpu=False,
        estimated_ram_gb=0.5,
        tier=2,
        tested=False,
    ),
    
    # ========== TIER 2: AUDIO ==========
    
    "whisper-tiny": TinyModel(
        id="whisper-tiny",
        name="Whisper Tiny",
        hf_name="openai/whisper-tiny",
        params_millions=39.0,
        architecture=Architecture.ENCODER_DECODER,
        modality=Modality.AUDIO,
        tasks=[TaskType.SPEECH_TO_TEXT],
        description="OpenAI Whisper - tiny speech recognition",
        num_layers=4,
        hidden_size=384,
        family="Whisper",
        specialty="speech-to-text",
        requires_gpu=False,
        estimated_ram_gb=1.0,
        tier=2,
        tested=False,
    ),
    
    # ========== TIER 3: SPECIALIZED ==========
    
    "albert-base": TinyModel(
        id="albert-base",
        name="ALBERT Base",
        hf_name="albert-base-v2",
        params_millions=11.8,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.FILL_MASK, TaskType.TEXT_EMBEDDINGS],
        description="ALBERT - parameter sharing for tiny size",
        num_layers=12,
        hidden_size=768,
        family="ALBERT",
        specialty="general",
        requires_gpu=False,
        estimated_ram_gb=0.5,
        tier=3,
        tested=False,
    ),
    
    "mobilebert": TinyModel(
        id="mobilebert",
        name="MobileBERT",
        hf_name="google/mobilebert-uncased",
        params_millions=25.3,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.FILL_MASK, TaskType.TEXT_EMBEDDINGS, TaskType.QUESTION_ANSWERING],
        description="MobileBERT - optimized for mobile devices",
        num_layers=24,
        hidden_size=128,
        family="BERT",
        specialty="mobile",
        requires_gpu=False,
        estimated_ram_gb=0.5,
        tier=3,
        tested=False,
    ),
    
    "deberta-v3-small": TinyModel(
        id="deberta-v3-small",
        name="DeBERTa V3 Small",
        hf_name="microsoft/deberta-v3-small",
        params_millions=44.0,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.FILL_MASK, TaskType.TEXT_CLASSIFICATION, TaskType.QUESTION_ANSWERING],
        description="DeBERTa V3 - state-of-the-art for small models",
        num_layers=6,
        hidden_size=768,
        family="DeBERTa",
        specialty="general",
        requires_gpu=False,
        estimated_ram_gb=1.0,
        tier=3,
        tested=False,
    ),
    
    "distilmbert": TinyModel(
        id="distilmbert",
        name="DistilmBERT",
        hf_name="distilbert-base-multilingual-cased",
        params_millions=66.0,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.FILL_MASK, TaskType.TEXT_EMBEDDINGS],
        description="Multilingual DistilBERT - 104 languages",
        num_layers=6,
        hidden_size=768,
        family="BERT",
        specialty="multilingual",
        requires_gpu=False,
        estimated_ram_gb=1.0,
        tier=3,
        tested=False,
    ),
}


# ============================================================================
# ZOO UTILITIES
# ============================================================================

def get_model(model_id: str) -> Optional[TinyModel]:
    """Get a model by ID"""
    return MODEL_ZOO.get(model_id)


def list_models(
    tier: Optional[int] = None,
    modality: Optional[Modality] = None,
    architecture: Optional[Architecture] = None,
    max_params: Optional[float] = None,
) -> List[TinyModel]:
    """List models with optional filters"""
    models = list(MODEL_ZOO.values())
    
    if tier is not None:
        models = [m for m in models if m.tier <= tier]
    if modality is not None:
        models = [m for m in models if m.modality == modality]
    if architecture is not None:
        models = [m for m in models if m.architecture == architecture]
    if max_params is not None:
        models = [m for m in models if m.params_millions <= max_params]
    
    return sorted(models, key=lambda m: (m.tier, m.params_millions))


def list_by_tier(tier: int) -> List[TinyModel]:
    """Get all models of a specific tier"""
    return [m for m in MODEL_ZOO.values() if m.tier == tier]


def get_catalog() -> Dict[str, List[Dict]]:
    """Get full catalog organized by modality"""
    catalog = {
        "text": [],
        "vision": [],
        "audio": [],
        "multimodal": [],
    }
    
    for model in MODEL_ZOO.values():
        catalog[model.modality.value].append(model.to_dict())
    
    return catalog


def get_stats() -> Dict[str, Any]:
    """Get zoo statistics"""
    models = list(MODEL_ZOO.values())
    return {
        "total_models": len(models),
        "by_tier": {
            1: len([m for m in models if m.tier == 1]),
            2: len([m for m in models if m.tier == 2]),
            3: len([m for m in models if m.tier == 3]),
        },
        "by_modality": {
            "text": len([m for m in models if m.modality == Modality.TEXT]),
            "vision": len([m for m in models if m.modality == Modality.VISION]),
            "audio": len([m for m in models if m.modality == Modality.AUDIO]),
        },
        "smallest_params": min(m.params_millions for m in models),
        "largest_params": max(m.params_millions for m in models),
        "total_params_millions": sum(m.params_millions for m in models),
        "tested_count": len([m for m in models if m.tested]),
    }


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Tiny AI Model Zoo")
    print("=" * 50)
    
    stats = get_stats()
    print(f"\nTotal models: {stats['total_models']}")
    print(f"  Tier 1 (must-have): {stats['by_tier'][1]}")
    print(f"  Tier 2 (vision/audio): {stats['by_tier'][2]}")
    print(f"  Tier 3 (specialized): {stats['by_tier'][3]}")
    print(f"\nText models: {stats['by_modality']['text']}")
    print(f"Vision models: {stats['by_modality']['vision']}")
    print(f"Audio models: {stats['by_modality']['audio']}")
    print(f"\nSize range: {stats['smallest_params']}M - {stats['largest_params']}M params")
    
    print("\n" + "=" * 50)
    print("TIER 1 MODELS (Start here):\n")
    
    for model in list_by_tier(1):
        gpu_str = "[GPU]" if model.requires_gpu else "[CPU]"
        print(f"  {gpu_str} {model.name}")
        print(f"       {model.params_millions}M params | {model.family}")
        print(f"       HF: {model.hf_name}")
        print(f"       Tasks: {', '.join(t.value for t in model.tasks)}")
        print()

