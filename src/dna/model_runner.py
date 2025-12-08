"""
🚀 Tiny AI Model Runner
Download and run tiny AI models with one click!

Supports:
- Text Generation (GPT-like models)
- Text Embeddings (BERT-like models)
- Text Classification
- Image Classification
- Speech-to-Text
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Generator
from dataclasses import dataclass
from enum import Enum

# Lazy imports for faster startup
_transformers = None
_AutoModel = None
_AutoTokenizer = None
_pipeline = None

logger = logging.getLogger(__name__)


def _ensure_transformers():
    """Lazy load transformers to speed up import"""
    global _transformers, _AutoModel, _AutoTokenizer, _pipeline
    if _transformers is None:
        import transformers
        from transformers import AutoModel, AutoTokenizer, pipeline
        _transformers = transformers
        _AutoModel = AutoModel
        _AutoTokenizer = AutoTokenizer
        _pipeline = pipeline
    return _transformers, _AutoModel, _AutoTokenizer, _pipeline


class ModelStatus(str, Enum):
    """Model download/load status"""
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass
class RunResult:
    """Result of running a model"""
    success: bool
    output: Any
    model_id: str
    task: str
    duration_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "model_id": self.model_id,
            "task": self.task,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class TinyModelRunner:
    """
    🚀 The Tiny AI Model Runner
    
    Download and run tiny AI models on CPU with minimal memory footprint.
    
    Usage:
        runner = TinyModelRunner()
        
        # Download a model
        runner.download("tinybert")
        
        # Run text embeddings
        result = runner.embed("tinybert", "Hello, world!")
        
        # Run text generation
        result = runner.generate("distilgpt2", "Once upon a time", max_length=50)
        
        # Stream text generation
        for token in runner.generate_stream("distilgpt2", "Once upon a time"):
            print(token, end="", flush=True)
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        device: str = "auto",
        low_memory: bool = True,
    ):
        """
        Initialize the model runner.
        
        Args:
            cache_dir: Where to store downloaded models (default: ./data/models)
            device: "cpu", "cuda", or "auto"
            low_memory: If True, use memory-efficient loading
        """
        self.cache_dir = cache_dir or Path("data/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.low_memory = low_memory
        
        # Cache for loaded models and tokenizers
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._pipelines: Dict[str, Any] = {}
        
        # Track status
        self._status: Dict[str, ModelStatus] = {}
        
        logger.info(f"🚀 TinyModelRunner initialized (device={self.device}, cache={self.cache_dir})")
    
    def get_status(self, model_id: str) -> ModelStatus:
        """Get the current status of a model"""
        return self._status.get(model_id, ModelStatus.NOT_DOWNLOADED)
    
    def download(self, model_id: str, force: bool = False) -> bool:
        """
        Download a model from HuggingFace.
        
        Args:
            model_id: Model ID from the zoo
            force: Re-download even if already exists
            
        Returns:
            True if download successful
        """
        from dna.model_zoo import get_model
        
        model_info = get_model(model_id)
        if model_info is None:
            logger.error(f"❌ Unknown model: {model_id}")
            return False
        
        try:
            self._status[model_id] = ModelStatus.DOWNLOADING
            logger.info(f"📥 Downloading {model_info.name} ({model_info.params_millions}M params)...")
            
            _, AutoModel, AutoTokenizer, _ = _ensure_transformers()
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_info.hf_name,
                cache_dir=str(self.cache_dir),
            )
            
            # Download model
            model = AutoModel.from_pretrained(
                model_info.hf_name,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=self.low_memory,
            )
            
            self._status[model_id] = ModelStatus.DOWNLOADED
            logger.info(f"✅ Downloaded {model_info.name}")
            
            # Don't keep in memory unless requested
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return True
            
        except Exception as e:
            self._status[model_id] = ModelStatus.ERROR
            logger.error(f"❌ Failed to download {model_id}: {e}")
            return False
    
    def load(self, model_id: str) -> bool:
        """
        Load a model into memory for inference.
        
        Args:
            model_id: Model ID from the zoo
            
        Returns:
            True if load successful
        """
        if model_id in self._models:
            return True  # Already loaded
        
        from dna.model_zoo import get_model
        
        model_info = get_model(model_id)
        if model_info is None:
            logger.error(f"❌ Unknown model: {model_id}")
            return False
        
        try:
            self._status[model_id] = ModelStatus.LOADING
            logger.info(f"⏳ Loading {model_info.name}...")
            
            _, AutoModel, AutoTokenizer, _ = _ensure_transformers()
            
            # Load tokenizer
            self._tokenizers[model_id] = AutoTokenizer.from_pretrained(
                model_info.hf_name,
                cache_dir=str(self.cache_dir),
            )
            
            # Load model
            self._models[model_id] = AutoModel.from_pretrained(
                model_info.hf_name,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=self.low_memory,
            ).to(self.device).eval()
            
            self._status[model_id] = ModelStatus.READY
            logger.info(f"✅ Loaded {model_info.name}")
            return True
            
        except Exception as e:
            self._status[model_id] = ModelStatus.ERROR
            logger.error(f"❌ Failed to load {model_id}: {e}")
            return False
    
    def unload(self, model_id: str) -> bool:
        """Unload a model from memory"""
        if model_id in self._models:
            del self._models[model_id]
        if model_id in self._tokenizers:
            del self._tokenizers[model_id]
        if model_id in self._pipelines:
            del self._pipelines[model_id]
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self._status[model_id] = ModelStatus.DOWNLOADED
        logger.info(f"🗑️ Unloaded {model_id}")
        return True
    
    def embed(
        self,
        model_id: str,
        texts: Union[str, List[str]],
        pooling: str = "mean",
    ) -> RunResult:
        """
        Get text embeddings from an encoder model.
        
        Args:
            model_id: Model ID (e.g., "tinybert", "minilm-l6")
            texts: Single text or list of texts
            pooling: "mean", "cls", or "max"
            
        Returns:
            RunResult with embeddings as output
        """
        import time
        start = time.perf_counter()
        
        try:
            # Ensure model is loaded
            if model_id not in self._models:
                if not self.load(model_id):
                    return RunResult(
                        success=False,
                        output=None,
                        model_id=model_id,
                        task="embed",
                        error="Failed to load model",
                    )
            
            model = self._models[model_id]
            tokenizer = self._tokenizers[model_id]
            
            # Handle single text
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # Apply pooling
                if pooling == "cls":
                    embeddings = hidden_states[:, 0, :]
                elif pooling == "max":
                    embeddings = hidden_states.max(dim=1).values
                else:  # mean
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            
            duration = (time.perf_counter() - start) * 1000
            
            return RunResult(
                success=True,
                output=embeddings.cpu().numpy().tolist(),
                model_id=model_id,
                task="embed",
                duration_ms=duration,
            )
            
        except Exception as e:
            return RunResult(
                success=False,
                output=None,
                model_id=model_id,
                task="embed",
                error=str(e),
            )
    
    def generate(
        self,
        model_id: str,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> RunResult:
        """
        Generate text from a decoder model.
        
        Args:
            model_id: Model ID (e.g., "distilgpt2")
            prompt: Text prompt to continue
            max_length: Maximum tokens to generate
            temperature: Randomness (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            RunResult with generated text as output
        """
        import time
        start = time.perf_counter()
        
        try:
            # Use pipeline for generation
            if model_id not in self._pipelines:
                _, _, _, pipeline = _ensure_transformers()
                
                from dna.model_zoo import get_model
                model_info = get_model(model_id)
                
                self._pipelines[model_id] = pipeline(
                    "text-generation",
                    model=model_info.hf_name,
                    device=0 if self.device == "cuda" else -1,
                    model_kwargs={
                        "cache_dir": str(self.cache_dir),
                        "low_cpu_mem_usage": self.low_memory,
                    },
                )
                self._status[model_id] = ModelStatus.READY
            
            pipe = self._pipelines[model_id]
            
            # Generate
            result = pipe(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=pipe.tokenizer.eos_token_id,
            )
            
            generated_text = result[0]["generated_text"]
            duration = (time.perf_counter() - start) * 1000
            
            return RunResult(
                success=True,
                output=generated_text,
                model_id=model_id,
                task="generate",
                duration_ms=duration,
            )
            
        except Exception as e:
            return RunResult(
                success=False,
                output=None,
                model_id=model_id,
                task="generate",
                error=str(e),
            )
    
    def generate_stream(
        self,
        model_id: str,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> Generator[str, None, None]:
        """
        Stream text generation token by token.
        
        Args:
            model_id: Model ID (e.g., "distilgpt2")
            prompt: Text prompt to continue
            max_length: Maximum tokens to generate
            temperature: Randomness
            
        Yields:
            Generated tokens one at a time
        """
        try:
            from dna.model_zoo import get_model
            model_info = get_model(model_id)
            
            _, _, AutoTokenizer, _ = _ensure_transformers()
            from transformers import AutoModelForCausalLM
            
            # Load for generation
            tokenizer = AutoTokenizer.from_pretrained(
                model_info.hf_name,
                cache_dir=str(self.cache_dir),
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_info.hf_name,
                cache_dir=str(self.cache_dir),
                low_cpu_mem_usage=self.low_memory,
            ).to(self.device).eval()
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate token by token
            with torch.no_grad():
                for _ in range(max_length):
                    outputs = model(**inputs)
                    next_token_logits = outputs.logits[:, -1, :] / temperature
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits, dim=-1),
                        num_samples=1,
                    )
                    
                    # Decode and yield
                    token_str = tokenizer.decode(next_token[0])
                    yield token_str
                    
                    # Update inputs
                    inputs = {
                        "input_ids": torch.cat([inputs["input_ids"], next_token], dim=-1),
                        "attention_mask": torch.cat([
                            inputs["attention_mask"],
                            torch.ones((1, 1), device=self.device, dtype=torch.long),
                        ], dim=-1),
                    }
                    
                    # Stop on EOS
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            yield f"[ERROR: {e}]"
    
    def classify(
        self,
        model_id: str,
        text: str,
        labels: Optional[List[str]] = None,
    ) -> RunResult:
        """
        Classify text (zero-shot or fine-tuned).
        
        Args:
            model_id: Model ID
            text: Text to classify
            labels: Optional labels for zero-shot classification
            
        Returns:
            RunResult with classification scores
        """
        import time
        start = time.perf_counter()
        
        try:
            _, _, _, pipeline = _ensure_transformers()
            
            from dna.model_zoo import get_model
            model_info = get_model(model_id)
            
            # Use zero-shot classification if labels provided
            if labels:
                pipe = pipeline(
                    "zero-shot-classification",
                    model=model_info.hf_name,
                    device=0 if self.device == "cuda" else -1,
                )
                result = pipe(text, labels)
            else:
                pipe = pipeline(
                    "text-classification",
                    model=model_info.hf_name,
                    device=0 if self.device == "cuda" else -1,
                )
                result = pipe(text)
            
            duration = (time.perf_counter() - start) * 1000
            
            return RunResult(
                success=True,
                output=result,
                model_id=model_id,
                task="classify",
                duration_ms=duration,
            )
            
        except Exception as e:
            return RunResult(
                success=False,
                output=None,
                model_id=model_id,
                task="classify",
                error=str(e),
            )
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self._models.keys())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage in MB for loaded models"""
        usage = {}
        for model_id, model in self._models.items():
            params = sum(p.numel() * p.element_size() for p in model.parameters())
            usage[model_id] = params / (1024 * 1024)
        return usage


# ============================================================================
# Singleton instance
# ============================================================================

_runner_instance: Optional[TinyModelRunner] = None


def get_runner() -> TinyModelRunner:
    """Get the global model runner instance"""
    global _runner_instance
    if _runner_instance is None:
        _runner_instance = TinyModelRunner()
    return _runner_instance


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="🚀 Tiny AI Model Runner")
    parser.add_argument("--model", "-m", default="tinybert", help="Model ID to use")
    parser.add_argument("--task", "-t", choices=["embed", "generate", "classify"], default="embed")
    parser.add_argument("--text", default="Hello, world! This is a test.", help="Input text")
    parser.add_argument("--download", "-d", action="store_true", help="Download model first")
    
    args = parser.parse_args()
    
    runner = TinyModelRunner()
    
    if args.download:
        print(f"📥 Downloading {args.model}...")
        runner.download(args.model)
    
    print(f"\n🚀 Running {args.task} with {args.model}...")
    
    if args.task == "embed":
        result = runner.embed(args.model, args.text)
        if result.success:
            print(f"✅ Embedding shape: {len(result.output)}x{len(result.output[0])}")
            print(f"   First 5 values: {result.output[0][:5]}")
        else:
            print(f"❌ Error: {result.error}")
            
    elif args.task == "generate":
        result = runner.generate(args.model, args.text, max_length=50)
        if result.success:
            print(f"✅ Generated:\n{result.output}")
        else:
            print(f"❌ Error: {result.error}")
            
    elif args.task == "classify":
        result = runner.classify(args.model, args.text)
        if result.success:
            print(f"✅ Classification: {result.output}")
        else:
            print(f"❌ Error: {result.error}")
    
    print(f"\n⏱️ Duration: {result.duration_ms:.1f}ms")
