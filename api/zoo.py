"""
🏪 Tiny AI Play Store API
REST endpoints for browsing, downloading, and running tiny AI models.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio

router = APIRouter()


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class ModelInfo(BaseModel):
    """Model information for API responses"""
    id: str
    name: str
    hf_name: str
    params_millions: float
    architecture: str
    modality: str
    tasks: List[str]
    description: str
    num_layers: int
    hidden_size: int
    family: str
    specialty: str
    requires_gpu: bool
    estimated_ram_gb: float
    tier: int
    tested: bool
    status: str = "not_downloaded"


class CatalogResponse(BaseModel):
    """Full catalog response"""
    text: List[ModelInfo]
    vision: List[ModelInfo]
    audio: List[ModelInfo]
    multimodal: List[ModelInfo]
    stats: Dict[str, Any]


class DownloadRequest(BaseModel):
    """Request to download a model"""
    model_id: str
    force: bool = False


class EmbedRequest(BaseModel):
    """Request for text embeddings"""
    model_id: str
    texts: List[str]
    pooling: str = "mean"  # mean, cls, max


class GenerateRequest(BaseModel):
    """Request for text generation"""
    model_id: str
    prompt: str
    max_length: int = 50
    temperature: float = 1.0
    top_p: float = 0.9
    stream: bool = False


class ClassifyRequest(BaseModel):
    """Request for text classification"""
    model_id: str
    text: str
    labels: Optional[List[str]] = None


class RunResponse(BaseModel):
    """Response from running a model"""
    success: bool
    output: Any
    model_id: str
    task: str
    duration_ms: float
    error: Optional[str] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_runner():
    """Get the model runner instance"""
    from dna.model_runner import get_runner
    return get_runner()


def _get_model_with_status(model_id: str) -> ModelInfo:
    """Get model info with current status"""
    from dna.model_zoo import get_model
    from dna.model_runner import get_runner
    
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    runner = get_runner()
    status = runner.get_status(model_id).value
    
    info = model.to_dict()
    info["status"] = status
    return ModelInfo(**info)


# ============================================================================
# CATALOG ENDPOINTS
# ============================================================================

@router.get("/zoo/catalog", response_model=CatalogResponse)
async def get_catalog():
    """
    Get the full model catalog.
    
    Returns all models organized by modality with current download status.
    """
    from dna.model_zoo import get_catalog as get_zoo_catalog, get_stats, MODEL_ZOO
    from dna.model_runner import get_runner
    
    catalog = get_zoo_catalog()
    stats = get_stats()
    runner = get_runner()
    
    # Add status to each model
    for modality in catalog:
        for model in catalog[modality]:
            model["status"] = runner.get_status(model["id"]).value
    
    return CatalogResponse(
        text=[ModelInfo(**m) for m in catalog["text"]],
        vision=[ModelInfo(**m) for m in catalog["vision"]],
        audio=[ModelInfo(**m) for m in catalog["audio"]],
        multimodal=[ModelInfo(**m) for m in catalog["multimodal"]],
        stats=stats,
    )


@router.get("/zoo/models", response_model=List[ModelInfo])
async def list_models(
    tier: Optional[int] = None,
    modality: Optional[str] = None,
    max_params: Optional[float] = None,
):
    """
    List models with optional filters.
    
    Args:
        tier: Filter by tier (1, 2, or 3)
        modality: Filter by modality (text, vision, audio)
        max_params: Filter by maximum parameters (in millions)
    """
    from dna.model_zoo import list_models as list_zoo_models, Modality
    from dna.model_runner import get_runner
    
    mod = Modality(modality) if modality else None
    models = list_zoo_models(tier=tier, modality=mod, max_params=max_params)
    
    runner = get_runner()
    result = []
    for m in models:
        info = m.to_dict()
        info["status"] = runner.get_status(m.id).value
        result.append(ModelInfo(**info))
    
    return result


@router.get("/zoo/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get details for a specific model"""
    return _get_model_with_status(model_id)


@router.get("/zoo/stats")
async def get_stats():
    """Get model zoo statistics"""
    from dna.model_zoo import get_stats
    return get_stats()


# ============================================================================
# DOWNLOAD & LOAD ENDPOINTS
# ============================================================================

@router.get("/zoo/download/{model_id}/progress")
async def stream_download_progress(model_id: str):
    """
    Stream download progress via Server-Sent Events (SSE).
    
    Frontend should connect to this before starting download.
    """
    from sse_starlette.sse import EventSourceResponse
    from dna.download_manager import get_download_manager
    import asyncio
    
    async def event_generator():
        download_manager = get_download_manager()
        
        # Send initial event
        yield {
            "event": "start",
            "data": json.dumps({"model_id": model_id, "status": "starting"})
        }
        
        # Monitor progress
        last_progress = 0
        while True:
            # Check if download exists in progress
            if model_id in download_manager._downloads:
                progress = download_manager._downloads[model_id]
                
                # Send progress update if changed
                if progress.percent != last_progress:
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "model_id": model_id,
                            "percent": progress.percent,
                            "downloaded_mb": progress.downloaded_mb,
                            "total_mb": progress.total_mb,
                            "speed_mbps": progress.speed_mbps,
                            "eta_seconds": progress.eta_seconds,
                            "status": progress.status
                        })
                    }
                    last_progress = progress.percent
                
                # Check if completed
                if progress.status in ["completed", "failed"]:
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "model_id": model_id,
                            "status": progress.status
                        })
                    }
                    break
            
            await asyncio.sleep(0.5)  # Poll every 500ms
    
    return EventSourceResponse(event_generator())


@router.post("/zoo/download")
async def download_model(request: DownloadRequest, background_tasks: BackgroundTasks):
    """
    Start model download (async background task).
    
    Use GET /zoo/download/{model_id}/progress to monitor progress via SSE.
    """
    from dna.model_zoo import get_model
    from dna.download_manager import get_download_manager
    import logging
    
    logger = logging.getLogger(__name__)
    
    model = get_model(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
    
    runner = _get_runner()
    
    # Check if already downloaded
    current_status = runner.get_status(request.model_id).value
    if current_status in ["downloaded", "ready"] and not request.force:
        return {
            "message": f"{model.name} already downloaded",
            "model_id": request.model_id,
            "status": current_status,
        }
    
    # Set status to downloading immediately
    from dna.model_runner import ModelStatus
    runner._status[request.model_id] = ModelStatus.DOWNLOADING
    
    # Download in background with progress tracking
    def do_download():
        try:
            logger.info(f"Starting download for {model.name} with progress tracking...")
            download_manager = get_download_manager()
            
            # Download with progress callback
            success = download_manager.download_model(
                model_id=request.model_id,
                hf_name=model.hf_name
            )
            if success:
                logger.info(f"Successfully downloaded {model.name}")
            else:
                logger.error(f"Failed to download {model.name}")
        except Exception as e:
            logger.error(f"Error downloading {model.name}: {e}")
    
    background_tasks.add_task(do_download)
    
    return {
        "message": f"Downloading {model.name}... Poll /api/zoo/models/{request.model_id} for status",
        "model_id": request.model_id,
        "status": "downloading",
    }


@router.post("/zoo/load/{model_id}")
async def load_model(model_id: str):
    """Load a model into memory for fast inference"""
    runner = _get_runner()
    success = runner.load(model_id)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_id}'")
    
    return {
        "message": f"Model loaded",
        "model_id": model_id,
        "status": "ready",
    }


@router.post("/zoo/unload/{model_id}")
async def unload_model(model_id: str):
    """Unload a model from memory"""
    runner = _get_runner()
    runner.unload(model_id)
    
    return {
        "message": f"Model unloaded",
        "model_id": model_id,
    }


@router.get("/zoo/loaded")
async def get_loaded_models():
    """Get list of currently loaded models with memory usage"""
    runner = _get_runner()
    loaded = runner.get_loaded_models()
    memory = runner.get_memory_usage()
    
    return {
        "loaded_models": loaded,
        "memory_usage_mb": memory,
        "total_memory_mb": sum(memory.values()),
    }


# ============================================================================
# INFERENCE ENDPOINTS
# ============================================================================

@router.post("/zoo/embed", response_model=RunResponse)
async def embed_text(request: EmbedRequest):
    """
    Get text embeddings from an encoder model.
    
    Supported models: tinybert, electra-small, minilm-l6, distilbert, etc.
    """
    runner = _get_runner()
    result = runner.embed(
        model_id=request.model_id,
        texts=request.texts,
        pooling=request.pooling,
    )
    return RunResponse(**result.to_dict())


@router.post("/zoo/generate", response_model=RunResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text from a decoder model.
    
    Supported models: distilgpt2
    
    For streaming, set stream=true and use /zoo/generate/stream instead.
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="For streaming, use GET /zoo/generate/stream endpoint"
        )
    
    runner = _get_runner()
    result = runner.generate(
        model_id=request.model_id,
        prompt=request.prompt,
        max_length=request.max_length,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return RunResponse(**result.to_dict())


@router.get("/zoo/generate/stream")
async def generate_stream(
    model_id: str,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
):
    """
    Stream text generation token by token.
    
    Returns a Server-Sent Events (SSE) stream.
    """
    runner = _get_runner()
    
    async def generate():
        for token in runner.generate_stream(
            model_id=model_id,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
        ):
            yield f"data: {token}\n\n"
            await asyncio.sleep(0)  # Allow other tasks to run
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/zoo/classify", response_model=RunResponse)
async def classify_text(request: ClassifyRequest):
    """
    Classify text using an encoder model.
    
    If labels are provided, uses zero-shot classification.
    """
    runner = _get_runner()
    result = runner.classify(
        model_id=request.model_id,
        text=request.text,
        labels=request.labels,
    )
    return RunResponse(**result.to_dict())


# ============================================================================
# QUICK RUN ENDPOINT
# ============================================================================

@router.post("/zoo/run")
async def quick_run(
    model_id: str,
    task: str,
    input_text: str,
    options: Optional[Dict[str, Any]] = None,
):
    """
    Quick run endpoint - auto-detects best task for the model.
    
    Args:
        model_id: Model to use
        task: "embed", "generate", "classify"
        input_text: Text to process
        options: Additional options (max_length, temperature, etc.)
    """
    runner = _get_runner()
    options = options or {}
    
    if task == "embed":
        result = runner.embed(model_id, input_text, **options)
    elif task == "generate":
        result = runner.generate(model_id, input_text, **options)
    elif task == "classify":
        result = runner.classify(model_id, input_text, **options)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")
    
    return RunResponse(**result.to_dict())
