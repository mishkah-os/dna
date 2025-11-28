"""
Experiment API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from database.db import get_db
from database.models import Experiment as DBExperiment, Model as DBModel

router = APIRouter()

# Pydantic schemas
class ExperimentCreate(BaseModel):
    model_id: int
    name: str
    dna_type: str = "hierarchical"
    epochs: int = 100
    learning_rate: float = 1e-4
    siren_layers: int = 5
    siren_hidden: int = 256

class ExperimentResponse(BaseModel):
    id: int
    model_id: int
    name: str
    dna_type: str
    epochs: int
    learning_rate: float
    siren_layers: int
    siren_hidden: int
    status: str
    final_psnr: Optional[float] = None
    final_loss: Optional[float] = None
    duration_seconds: Optional[float] = None
    current_epoch: int
    progress: float
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all experiments"""
    query = select(DBExperiment)
    
    if status:
        query = query.where(DBExperiment.status == status)
    
    query = query.offset(skip).limit(limit).order_by(DBExperiment.created_at.desc())
    
    result = await db.execute(query)
    experiments = result.scalars().all()
    return experiments


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get experiment by ID"""
    result = await db.execute(
        select(DBExperiment).where(DBExperiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return experiment


@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(
    experiment: ExperimentCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create new experiment"""
    # Verify model exists
    result = await db.execute(
        select(DBModel).where(DBModel.id == experiment.model_id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Create experiment
    new_experiment = DBExperiment(
        **experiment.dict(),
        status="pending"
    )
    
    db.add(new_experiment)
    await db.commit()
    await db.refresh(new_experiment)
    
    # TODO: Start background task for pattern mining
    
    return new_experiment


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Start experiment execution"""
    result = await db.execute(
        select(DBExperiment).where(DBExperiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status not in ["pending", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start experiment with status: {experiment.status}"
        )
    
    experiment.status = "running"
    experiment.current_epoch = 0
    experiment.progress = 0.0
    await db.commit()
    
    # TODO: Background task to run experiment
    
    return {"message": "Experiment started", "id": experiment_id}


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Stop running experiment"""
    result = await db.execute(
        select(DBExperiment).where(DBExperiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot stop experiment with status: {experiment.status}"
        )
    
    experiment.status = "stopped"
    await db.commit()
    
    # TODO: Cancel background task
    
    return {"message": "Experiment stopped", "id": experiment_id}


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete experiment"""
    result = await db.execute(
        select(DBExperiment).where(DBExperiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    await db.delete(experiment)
    await db.commit()
    
    return {"message": "Experiment deleted successfully"}
