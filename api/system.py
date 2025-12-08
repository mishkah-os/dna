"""
🖥️ System Monitoring API
Monitor CPU, RAM, GPU, and loaded models
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMAS
# ============================================================================

class GPUInfoSchema(BaseModel):
    """GPU information"""
    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_percent: float
    temperature_celsius: Optional[float] = None


class ModelMemorySchema(BaseModel):
    """Model memory usage"""
    model_id: str
    ram_mb: float
    vram_mb: float
    status: str


class SystemStatsSchema(BaseModel):
    """Complete system stats"""
    cpu_count: int
    cpu_percent: float
    cpu_freq_mhz: float
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float
    gpus: List[GPUInfoSchema]
    loaded_models: List[ModelMemorySchema]
    platform: str
    python_version: str


class ProcessInfo(BaseModel):
    """Process information"""
    pid: int
    name: str
    memory_mb: float


class MemoryEstimate(BaseModel):
    """Memory estimate for a model"""
    model_id: str
    ram_gb: float
    vram_gb: float
    can_load: bool
    available_ram_gb: float


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/system/stats", response_model=SystemStatsSchema)
async def get_system_stats():
    """
    Get complete system statistics.
    
    Returns:
        - CPU usage, cores, frequency
        - RAM total, used, available
        - GPU info (NVIDIA only)
        - Loaded models and their memory usage
    """
    try:
        from dna.system_monitor import get_monitor
        
        monitor = get_monitor()
        stats = monitor.get_stats()
        
        # Convert to schema
        return SystemStatsSchema(
            cpu_count=stats.cpu_count,
            cpu_percent=stats.cpu_percent,
            cpu_freq_mhz=stats.cpu_freq_mhz,
            ram_total_gb=stats.ram_total_gb,
            ram_used_gb=stats.ram_used_gb,
            ram_available_gb=stats.ram_available_gb,
            ram_percent=stats.ram_percent,
            gpus=[GPUInfoSchema(**gpu.__dict__) for gpu in stats.gpus],
            loaded_models=[ModelMemorySchema(**m.__dict__) for m in stats.loaded_models],
            platform=stats.platform,
            python_version=stats.python_version,
        )
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/processes")
async def get_top_processes(limit: int = 10):
    """Get top processes by memory usage"""
    try:
        from dna.system_monitor import get_monitor
        
        monitor = get_monitor()
        processes = monitor.get_top_processes(limit=limit)
        
        return {
            "processes": processes,
            "count": len(processes),
        }
        
    except Exception as e:
        logger.error(f"Error getting processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/estimate", response_model=MemoryEstimate)
async def estimate_model_memory(model_id: str):
    """
    Estimate memory needed for a model before loading.
    
    Args:
        model_id: Model ID from zoo
        
    Returns:
        Estimated RAM/VRAM and whether it can be loaded
    """
    try:
        from dna.system_monitor import get_monitor
        
        monitor = get_monitor()
        estimate = monitor.estimate_model_memory(model_id)
        
        return MemoryEstimate(
            model_id=model_id,
            ram_gb=estimate["ram_gb"],
            vram_gb=estimate["vram_gb"],
            can_load=estimate["can_load"],
            available_ram_gb=estimate["available_ram_gb"],
        )
        
    except Exception as e:
        logger.error(f"Error estimating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/kill/{pid}")
async def kill_process(pid: int):
    """
    Kill a process by PID.
    
    ⚠️ WARNING: This is dangerous! Use with caution.
    """
    import psutil
    
    try:
        process = psutil.Process(pid)
        process_name = process.name()
        
        # Safety check - don't kill critical processes
        critical_processes = ["System", "csrss.exe", "winlogon.exe", "services.exe"]
        if process_name in critical_processes:
            raise HTTPException(
                status_code=403,
                detail=f"Cannot kill critical process: {process_name}"
            )
        
        process.terminate()
        
        return {
            "success": True,
            "message": f"Terminated process {process_name} (PID: {pid})",
            "pid": pid,
            "name": process_name,
        }
        
    except psutil.NoSuchProcess:
        raise HTTPException(status_code=404, detail=f"Process {pid} not found")
    except psutil.AccessDenied:
        raise HTTPException(status_code=403, detail=f"Access denied to kill process {pid}")
    except Exception as e:
        logger.error(f"Error killing process {pid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/health")
async def system_health():
    """
    Quick health check for system resources.
    
    Returns warnings if resources are low.
    """
    try:
        from dna.system_monitor import get_monitor
        
        monitor = get_monitor()
        stats = monitor.get_stats()
        
        warnings = []
        
        # Check RAM
        if stats.ram_percent > 90:
            warnings.append({
                "type": "critical",
                "resource": "RAM",
                "message": f"RAM usage critical: {stats.ram_percent:.1f}%",
            })
        elif stats.ram_percent > 80:
            warnings.append({
                "type": "warning",
                "resource": "RAM",
                "message": f"RAM usage high: {stats.ram_percent:.1f}%",
            })
        
        # Check CPU
        if stats.cpu_percent > 90:
            warnings.append({
                "type": "warning",
                "resource": "CPU",
                "message": f"CPU usage high: {stats.cpu_percent:.1f}%",
            })
        
        # Check GPU
        for gpu in stats.gpus:
            gpu_percent = (gpu.memory_used_mb / gpu.memory_total_mb) * 100
            if gpu_percent > 90:
                warnings.append({
                    "type": "critical",
                    "resource": f"GPU {gpu.index}",
                    "message": f"{gpu.name} VRAM critical: {gpu_percent:.1f}%",
                })
        
        return {
            "healthy": len([w for w in warnings if w["type"] == "critical"]) == 0,
            "warnings": warnings,
            "cpu_percent": stats.cpu_percent,
            "ram_percent": stats.ram_percent,
        }
        
    except Exception as e:
        logger.error(f"Error checking system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
