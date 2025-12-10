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


class ModelCategory(str, Enum):
    """Model categories for organization"""
    TINY_BASIC = "tiny_basic"          # Original tiny models (<100M)
    SMART_SMALL = "smart_small"        # Small but powerful (1-7B)
    MATH_SPECIALIZED = "math"          # Math-focused models
    BIO_SPECIALIZED = "biology"        # DNA/Protein/Biomedical
    CODING_SPECIALIZED = "coding"      # Code + software engineering models
    VISION = "vision"                  # Image models
    AUDIO = "audio"                    # Audio models


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
    category: ModelCategory = ModelCategory.TINY_BASIC
    context_window: int = 2048
    
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
            "category": self.category.value,
            "context_window": self.context_window,
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
    
    # ========== SMART SMALL MODELS (1-7B) ==========
    
    "qwen2.5-3b": TinyModel(
        id="qwen2.5-3b",
        name="Qwen2.5-3B-Instruct",
        hf_name="Qwen/Qwen2.5-3B-Instruct",
        params_millions=3090.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Qwen 2.5 - Best small model, excellent Arabic support",
        num_layers=36,
        hidden_size=2048,
        family="Qwen",
        specialty="general + arabic",
        category=ModelCategory.SMART_SMALL,
        context_window=32768,
        requires_gpu=False,
        estimated_ram_gb=6.5,
        tier=1,
        tested=False,
    ),
    
    "qwen2.5-1.5b": TinyModel(
        id="qwen2.5-1.5b",
        name="Qwen2.5-1.5B-Instruct",
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        params_millions=1540.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Qwen 2.5 - Ultra-lightweight, fast inference",
        num_layers=28,
        hidden_size=1536,
        family="Qwen",
        specialty="general + arabic",
        category=ModelCategory.SMART_SMALL,
        context_window=32768,
        requires_gpu=False,
        estimated_ram_gb=3.2,
        tier=2,
        tested=False,
    ),
    
    "phi-3.5-mini": TinyModel(
        id="phi-3.5-mini",
        name="Phi-3.5-Mini",
        hf_name="microsoft/Phi-3.5-mini-instruct",
        params_millions=3820.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Microsoft Phi-3.5 - Long context window (128k tokens)",
        num_layers=32,
        hidden_size=3072,
        family="Phi",
        specialty="long-context reasoning",
        category=ModelCategory.SMART_SMALL,
        context_window=131072,
        requires_gpu=False,
        estimated_ram_gb=8.0,
        tier=2,
        tested=False,
    ),
    
    # ========== MATH-SPECIALIZED MODELS ==========
    
    "qwen2.5-math-1.5b": TinyModel(
        id="qwen2.5-math-1.5b",
        name="Qwen2.5-Math-1.5B",
        hf_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        params_millions=1540.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Mathematical reasoning and problem solving",
        num_layers=28,
        hidden_size=1536,
        family="Qwen",
        specialty="mathematics",
        category=ModelCategory.MATH_SPECIALIZED,
        context_window=32768,
        requires_gpu=False,
        estimated_ram_gb=3.2,
        tier=1,
        tested=False,
    ),
    
    "qwen2.5-math-7b": TinyModel(
        id="qwen2.5-math-7b",
        name="Qwen2.5-Math-7B",
        hf_name="Qwen/Qwen2.5-Math-7B-Instruct",
        params_millions=7610.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Advanced mathematical reasoning and proofs",
        num_layers=32,
        hidden_size=4096,
        family="Qwen",
        specialty="mathematics + proofs",
        category=ModelCategory.MATH_SPECIALIZED,
        context_window=32768,
        requires_gpu=True,
        estimated_ram_gb=15.5,
        tier=2,
        tested=False,
    ),
    
    # ========== BIO/DNA-SPECIALIZED MODELS ==========
    
    "dnabert-2": TinyModel(
        id="dnabert-2",
        name="DNABERT-2",
        hf_name="zhihan1996/DNABERT-2-117M",
        params_millions=117.0,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_CLASSIFICATION, TaskType.TEXT_EMBEDDINGS],
        description="DNA sequence analysis and genome understanding",
        num_layers=12,
        hidden_size=768,
        family="BERT",
        specialty="dna-sequences",
        category=ModelCategory.BIO_SPECIALIZED,
        context_window=512,
        requires_gpu=False,
        estimated_ram_gb=0.5,
        tier=1,
        tested=False,
    ),
    
    "biogpt": TinyModel(
        id="biogpt",
        name="BioGPT-Large",
        hf_name="microsoft/biogpt-large",
        params_millions=1500.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Biomedical text generation and QA",
        num_layers=24,
        hidden_size=1024,
        family="GPT",
        specialty="biomedical-text",
        category=ModelCategory.BIO_SPECIALIZED,
        context_window=1024,
        requires_gpu=False,
        estimated_ram_gb=3.0,
        tier=2,
        tested=False,
    ),
    
    "proteinbert": TinyModel(
        id="proteinbert",
        name="ProteinBERT",
        hf_name="Rostlab/prot_bert_bfd",
        params_millions=420.0,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_EMBEDDINGS, TaskType.TEXT_CLASSIFICATION],
        description="Protein sequence embeddings and function prediction",
        num_layers=30,
        hidden_size=1024,
        family="BERT",
        specialty="protein-sequences",
        category=ModelCategory.BIO_SPECIALIZED,
        context_window=1024,
        requires_gpu=False,
        estimated_ram_gb=1.7,
        tier=3,
        tested=False,
    ),

    "evodna-transformer": TinyModel(
        id="evodna-transformer",
        name="EvoDNA Transformer",
        hf_name="google/evodna-t500m",
        params_millions=500.0,
        architecture=Architecture.ENCODER_DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_CLASSIFICATION, TaskType.TEXT_EMBEDDINGS],
        description="Evolutionary DNA sequence modeling for genomics tasks",
        num_layers=24,
        hidden_size=1024,
        family="EvoDNA",
        specialty="genomics + dna",
        category=ModelCategory.BIO_SPECIALIZED,
        context_window=4096,
        requires_gpu=False,
        estimated_ram_gb=2.5,
        tier=2,
        tested=False,
    ),

    "esm3-small": TinyModel(
        id="esm3-small",
        name="ESM3-Small",
        hf_name="meta-llama/esm3-small",
        params_millions=1200.0,
        architecture=Architecture.ENCODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_EMBEDDINGS, TaskType.TEXT_CLASSIFICATION],
        description="Protein representation model for structure and function",
        num_layers=30,
        hidden_size=1024,
        family="ESM",
        specialty="protein-structure",
        category=ModelCategory.BIO_SPECIALIZED,
        context_window=2048,
        requires_gpu=True,
        estimated_ram_gb=4.0,
        tier=2,
        tested=False,
    ),

    "genslm-1.7b": TinyModel(
        id="genslm-1.7b",
        name="GenSLM-1.7B",
        hf_name="genslm/genslm-1.7b",
        params_millions=1700.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Generative model for genomes and long DNA contexts",
        num_layers=24,
        hidden_size=2048,
        family="GenSLM",
        specialty="dna-long-context",
        category=ModelCategory.BIO_SPECIALIZED,
        context_window=65536,
        requires_gpu=True,
        estimated_ram_gb=6.5,
        tier=3,
        tested=False,
    ),

    # ========== CODING-SPECIALIZED MODELS ==========

    "llama3-code-8b": TinyModel(
        id="llama3-code-8b",
        name="CodeLlama-7B-Instruct",
        hf_name="meta-llama/CodeLlama-7b-Instruct-hf",
        params_millions=7000.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Code-focused Llama 3 variant tuned for assistants",
        num_layers=32,
        hidden_size=4096,
        family="Llama 3",
        specialty="coding + code-review",
        category=ModelCategory.CODING_SPECIALIZED,
        context_window=8192,
        requires_gpu=True,
        estimated_ram_gb=14.0,
        tier=2,
        tested=False,
    ),

    "deepseek-coder-6.7b": TinyModel(
        id="deepseek-coder-6.7b",
        name="DeepSeek-Coder-6.7B",
        hf_name="deepseek-ai/deepseek-coder-6.7b-instruct",
        params_millions=6700.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Multilingual coding and reasoning assistant",
        num_layers=32,
        hidden_size=4096,
        family="DeepSeek",
        specialty="software + multilingual",
        category=ModelCategory.CODING_SPECIALIZED,
        context_window=32768,
        requires_gpu=True,
        estimated_ram_gb=12.0,
        tier=2,
        tested=False,
    ),

    "starcoder2-3b": TinyModel(
        id="starcoder2-3b",
        name="StarCoder2-3B",
        hf_name="bigcode/starcoder2-3b",
        params_millions=3000.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Lightweight code generation with permissive license",
        num_layers=26,
        hidden_size=3072,
        family="StarCoder",
        specialty="coding + completion",
        category=ModelCategory.CODING_SPECIALIZED,
        context_window=8192,
        requires_gpu=False,
        estimated_ram_gb=7.0,
        tier=1,
        tested=False,
    ),

    "qwen2.5-code-7b": TinyModel(
        id="qwen2.5-code-7b",
        name="Qwen2.5-Coder-7B",
        hf_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        params_millions=7610.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Code specialization with reasoning and tool hints",
        num_layers=32,
        hidden_size=4096,
        family="Qwen",
        specialty="coding + reasoning",
        category=ModelCategory.CODING_SPECIALIZED,
        context_window=32768,
        requires_gpu=True,
        estimated_ram_gb=13.0,
        tier=2,
        tested=False,
    ),

    # ========== ADVANCED REASONING / MATH ==========

    "deepseek-r1-distill": TinyModel(
        id="deepseek-r1-distill",
        name="DeepSeek-R1-Distill",
        hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        params_millions=14000.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Chain-of-thought distilled reasoning model",
        num_layers=40,
        hidden_size=5120,
        family="DeepSeek",
        specialty="reasoning + math",
        category=ModelCategory.MATH_SPECIALIZED,
        context_window=32768,
        requires_gpu=True,
        estimated_ram_gb=20.0,
        tier=3,
        tested=False,
    ),

    "math-mistral-7b": TinyModel(
        id="math-mistral-7b",
        name="Math-Mistral-7B",
        hf_name="mistralai/Math-Mistral-7B-Instruct",
        params_millions=7000.0,
        architecture=Architecture.DECODER,
        modality=Modality.TEXT,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Meta Math-Mistral tuned for proofs and algebra",
        num_layers=32,
        hidden_size=4096,
        family="Mistral",
        specialty="math-proofs",
        category=ModelCategory.MATH_SPECIALIZED,
        context_window=32768,
        requires_gpu=True,
        estimated_ram_gb=12.5,
        tier=2,
        tested=False,
    ),

    # ========== MULTIMODAL / VISION ==========

    "phi-3-vision": TinyModel(
        id="phi-3-vision",
        name="Phi-3-Vision",
        hf_name="microsoft/phi-3-vision",
        params_millions=4500.0,
        architecture=Architecture.DECODER,
        modality=Modality.MULTIMODAL,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Lightweight vision-language model for charts and UI",
        num_layers=32,
        hidden_size=3072,
        family="Phi",
        specialty="vision + text",
        category=ModelCategory.VISION,
        context_window=8192,
        requires_gpu=True,
        estimated_ram_gb=10.0,
        tier=2,
        tested=False,
    ),

    "llava-next": TinyModel(
        id="llava-next",
        name="LLaVA-NeXT",
        hf_name="llava-hf/llava-v1.6-mistral-7b-hf",
        params_millions=7000.0,
        architecture=Architecture.DECODER,
        modality=Modality.MULTIMODAL,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="Vision-chat tuned for instructions and OCR",
        num_layers=32,
        hidden_size=4096,
        family="LLaVA",
        specialty="vision-chat",
        category=ModelCategory.VISION,
        context_window=8192,
        requires_gpu=True,
        estimated_ram_gb=12.0,
        tier=2,
        tested=False,
    ),

    "internvl-2": TinyModel(
        id="internvl-2",
        name="InternVL-2",
        hf_name="OpenGVLab/InternVL2-8B",
        params_millions=8000.0,
        architecture=Architecture.DECODER,
        modality=Modality.MULTIMODAL,
        tasks=[TaskType.TEXT_GENERATION, TaskType.QUESTION_ANSWERING],
        description="High-quality vision-language understanding",
        num_layers=32,
        hidden_size=4096,
        family="InternVL",
        specialty="multi-image reasoning",
        category=ModelCategory.VISION,
        context_window=16384,
        requires_gpu=True,
        estimated_ram_gb=16.0,
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
            "multimodal": len([m for m in models if m.modality == Modality.MULTIMODAL]),
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

