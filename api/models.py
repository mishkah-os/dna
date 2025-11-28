"""
Model Zoo API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from pydantic import BaseModel
from pathlib import Path
import shutil

from database.db import get_db
from database.models import Model as DBModel

router = APIRouter()

# Pydantic schemas
class ModelBase(BaseModel):
    name: str
    hf_name: Optional[str] = None
    model_type: str
    architecture: Optional[str] = None
    num_parameters: Optional[int] = None
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    modality: str = "text"
    specialty: Optional[str] = None

class ModelCreate(ModelBase):
    pass

class ModelResponse(ModelBase):
    id: int
    file_path: Optional[str] = None
    file_size_mb: Optional[float] = None
    status: str
    
    class Config:
        from_attributes = True

class DownloadRequest(BaseModel):
    hf_name: str
    model_type: str
    modality: str = "text"


# Model storage directory
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


@router.get("/models", response_model=List[ModelResponse])
async def list_models(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all models"""
    result = await db.execute(
        select(DBModel).offset(skip).limit(limit)
    )
    models = result.scalars().all()
    return models


@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get model by ID"""
    result = await db.execute(
        select(DBModel).where(DBModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model


@router.post("/models/download", response_model=ModelResponse)
async def download_model(
    request: DownloadRequest,
    db: AsyncSession = Depends(get_db)
):
    """Download model from HuggingFace"""
    # Check if model already exists
    result = await db.execute(
        select(DBModel).where(DBModel.hf_name == request.hf_name)
    )
    existing_model = result.scalar_one_or_none()
    
    if existing_model:
        return existing_model
    
    # Create model entry with "loading" status
    new_model = DBModel(
        name=request.hf_name.split('/')[-1],
        hf_name=request.hf_name,
        model_type=request.model_type,
        modality=request.modality,
        status="loading"
    )
    
    db.add(new_model)
    await db.commit()
    await db.refresh(new_model)
    
    # TODO: Background task to download from HuggingFace
    # For now, just mark as ready
    new_model.status = "ready"
    await db.commit()
    await db.refresh(new_model)
    
    return new_model


@router.post("/models/upload", response_model=ModelResponse)
async def upload_model(
    file: UploadFile = File(...),
    name: str = None,
    model_type: str = "unknown",
    modality: str = "text",
    db: AsyncSession = Depends(get_db)
):
    """Upload local model file"""
    # Generate safe filename
    safe_name = name or file.filename.rsplit('.', 1)[0]
    file_ext = Path(file.filename).suffix
    save_path = MODELS_DIR / f"{safe_name}{file_ext}"
    
    # Save file
    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        
        # Create database entry
        new_model = DBModel(
            name=safe_name,
            model_type=model_type,
            modality=modality,
            file_path=str(save_path),
            file_size_mb=file_size_mb,
            status="ready"
        )
        
        db.add(new_model)
        await db.commit()
        await db.refresh(new_model)
        
        return new_model
        
    except Exception as e:
        # Cleanup on error
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete model"""
    result = await db.execute(
        select(DBModel).where(DBModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Delete file if exists
    if model.file_path:
        file_path = Path(model.file_path)
        if file_path.exists():
            file_path.unlink()
    
    # Delete from database
    await db.delete(model)
    await db.commit()
    
    return {"message": "Model deleted successfully"}
