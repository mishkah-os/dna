"""
Pattern Analysis API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from database.db import get_db
from database.models import Pattern as DBPattern, Experiment as DBExperiment

router = APIRouter()

# Pydantic schemas
class PatternResponse(BaseModel):
    id: int
    experiment_id: int
    pattern_type: Optional[str] = None
    entropy_measure: Optional[float] = None
    manifold_dimensionality: Optional[int] = None
    coordinates: Optional[Dict[str, Any]] = None
    signature: Optional[Dict[str, Any]] = None
    layer_data: Optional[Dict[str, Any]] = None
    dominant_frequencies: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class VisualizationData(BaseModel):
    """3D visualization data for Plotly"""
    x: List[float]
    y: List[float]
    z: List[float]
    colors: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@router.get("/patterns", response_model=List[PatternResponse])
async def list_patterns(
    skip: int = 0,
    limit: int = 100,
    experiment_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all discovered patterns"""
    query = select(DBPattern)
    
    if experiment_id:
        query = query.where(DBPattern.experiment_id == experiment_id)
    
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    patterns = result.scalars().all()
    return patterns


@router.get("/patterns/{pattern_id}", response_model=PatternResponse)
async def get_pattern(
    pattern_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get pattern by ID"""
    result = await db.execute(
        select(DBPattern).where(DBPattern.id == pattern_id)
    )
    pattern = result.scalar_one_or_none()
    
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    return pattern


@router.get("/patterns/{pattern_id}/viz", response_model=VisualizationData)
async def get_pattern_visualization(
    pattern_id: int,
    max_points: int = 10000,
    db: AsyncSession = Depends(get_db)
):
    """Get 3D visualization data for pattern"""
    result = await db.execute(
        select(DBPattern).where(DBPattern.id == pattern_id)
    )
    pattern = result.scalar_one_or_none()
    
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    if not pattern.coordinates:
        raise HTTPException(
            status_code=404,
            detail="No visualization data available for this pattern"
        )
    
    # Extract coordinates (assuming stored as {x: [...], y: [...], z: [...]})
    coords = pattern.coordinates
    
    # Downsample if too many points
    if len(coords.get('x', [])) > max_points:
        step = len(coords['x']) // max_points
        coords = {
            'x': coords['x'][::step],
            'y': coords['y'][::step],
            'z': coords['z'][::step]
        }
    
    # Generate colors based on layer data if available
    colors = None
    if pattern.layer_data:
        # TODO: Implement color mapping based on layer
        pass
    
    return VisualizationData(
        x=coords.get('x', []),
        y=coords.get('y', []),
        z=coords.get('z', []),
        colors=colors,
        metadata={
            'pattern_type': pattern.pattern_type,
            'entropy': pattern.entropy_measure
        }
    )


@router.get("/patterns/compare")
async def compare_patterns(
    pattern_a: int,
    pattern_b: int,
    db: AsyncSession = Depends(get_db)
):
    """Compare two patterns"""
    # Fetch both patterns
    result_a = await db.execute(
        select(DBPattern).where(DBPattern.id == pattern_a)
    )
    result_b = await db.execute(
        select(DBPattern).where(DBPattern.id == pattern_b)
    )
    
    pat_a = result_a.scalar_one_or_none()
    pat_b = result_b.scalar_one_or_none()
    
    if not pat_a or not pat_b:
        raise HTTPException(status_code=404, detail="One or both patterns not found")
    
    # Calculate similarity metrics
    # TODO: Implement actual similarity calculation
    similarity = {
        "pattern_a_id": pattern_a,
        "pattern_b_id": pattern_b,
        "euclidean_distance": None,  # TODO
        "correlation": None,  # TODO
        "entropy_diff": abs((pat_a.entropy_measure or 0) - (pat_b.entropy_measure or 0)),
        "pattern_type_match": pat_a.pattern_type == pat_b.pattern_type
    }
    
    return similarity
