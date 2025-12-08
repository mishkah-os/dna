"""
🖥️ System Monitor
Real-time monitoring of CPU, RAM, GPU, and loaded models
"""

import psutil
import platform
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import GPU monitoring (optional)
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVIDIA = True
except:
    HAS_NVIDIA = False
    logger.warning("NVIDIA GPU monitoring not available")


@dataclass
class GPUInfo:
    """GPU information"""
    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_percent: float
    temperature_celsius: Optional[float] = None


@dataclass
class ModelMemoryInfo:
    """Memory used by a loaded model"""
    model_id: str
    ram_mb: float
    vram_mb: float = 0.0
    status: str = "ready"


@dataclass
class SystemStats:
    """Complete system statistics"""
    # CPU
    cpu_count: int
    cpu_percent: float
    cpu_freq_mhz: float
    
    # RAM
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float
    
    # GPU (if available)
    gpus: List[GPUInfo]
    
    # Models
    loaded_models: List[ModelMemoryInfo]
    
    # System
    platform: str
    python_version: str


class SystemMonitor:
    """
    Monitor system resources in real-time.
    
    Usage:
        monitor = SystemMonitor()
        stats = monitor.get_stats()
        print(f"CPU: {stats.cpu_percent}%")
        print(f"RAM: {stats.ram_used_gb:.1f}/{stats.ram_total_gb:.1f} GB")
    """
    
    def __init__(self):
        """Initialize system monitor"""
        self.has_nvidia = HAS_NVIDIA
        logger.info(f"SystemMonitor initialized (NVIDIA: {self.has_nvidia})")
    
    def get_cpu_info(self) -> Dict[str, float]:
        """Get CPU usage"""
        return {
            "count": psutil.cpu_count(logical=True),
            "percent": psutil.cpu_percent(interval=0.1),
            "freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0.0,
        }
    
    def get_ram_info(self) -> Dict[str, float]:
        """Get RAM usage"""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "used_gb": mem.used / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent": mem.percent,
        }
    
    def get_gpu_info(self) -> List[GPUInfo]:
        """Get GPU information (NVIDIA only)"""
        if not self.has_nvidia:
            return []
        
        gpus = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic info
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except:
                    utilization = 0.0
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = None
                
                gpus.append(GPUInfo(
                    index=i,
                    name=name,
                    memory_total_mb=mem_info.total / (1024**2),
                    memory_used_mb=mem_info.used / (1024**2),
                    memory_free_mb=mem_info.free / (1024**2),
                    utilization_percent=utilization,
                    temperature_celsius=temp,
                ))
            
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
        
        return gpus
    
    def get_model_memory(self) -> List[ModelMemoryInfo]:
        """
        Get memory used by loaded models.
        
        Requires access to model runner singleton.
        """
        try:
            from dna.model_runner import get_runner
            runner = get_runner()
            
            models = []
            for model_id in runner.get_loaded_models():
                # Calculate RAM usage
                ram_usage = runner.get_memory_usage().get(model_id, 0.0)
                
                # TODO: Calculate VRAM usage (if model on GPU)
                vram_usage = 0.0
                
                models.append(ModelMemoryInfo(
                    model_id=model_id,
                    ram_mb=ram_usage,
                    vram_mb=vram_usage,
                    status=runner.get_status(model_id).value,
                ))
            
            return models
            
        except Exception as e:
            logger.error(f"Error getting model memory: {e}")
            return []
    
    def get_stats(self) -> SystemStats:
        """Get complete system statistics"""
        cpu_info = self.get_cpu_info()
        ram_info = self.get_ram_info()
        gpu_info = self.get_gpu_info()
        model_info = self.get_model_memory()
        
        return SystemStats(
            # CPU
            cpu_count=cpu_info["count"],
            cpu_percent=cpu_info["percent"],
            cpu_freq_mhz=cpu_info["freq_mhz"],
            
            # RAM
            ram_total_gb=ram_info["total_gb"],
            ram_used_gb=ram_info["used_gb"],
            ram_available_gb=ram_info["available_gb"],
            ram_percent=ram_info["percent"],
            
            # GPU
            gpus=gpu_info,
            
            # Models
            loaded_models=model_info,
            
            # System
            platform=platform.system(),
            python_version=platform.python_version(),
        )
    
    def estimate_model_memory(self, model_id: str) -> Dict[str, float]:
        """
        Estimate memory needed for a model before loading.
        
        Args:
            model_id: Model ID from zoo
            
        Returns:
            {
                "ram_gb": estimated RAM needed,
                "vram_gb": estimated VRAM needed (if GPU),
                "can_load": whether system has enough memory
            }
        """
        from dna.model_zoo import get_model
        
        model_info = get_model(model_id)
        if not model_info:
            return {"ram_gb": 0, "vram_gb": 0, "can_load": False}
        
        # Rough estimation
        # FP16: 2 bytes per parameter
        # FP32: 4 bytes per parameter
        # Add 20% overhead for activations
        
        params = model_info.params_millions * 1e6
        bytes_per_param = 2  # FP16
        overhead = 1.2
        
        estimated_bytes = params * bytes_per_param * overhead
        estimated_gb = estimated_bytes / (1024**3)
        
        # Check if we can load
        ram_info = self.get_ram_info()
        can_load = ram_info["available_gb"] >= estimated_gb
        
        return {
            "ram_gb": estimated_gb,
            "vram_gb": 0.0,  # TODO: Implement GPU estimation
            "can_load": can_load,
            "available_ram_gb": ram_info["available_gb"],
        }
    
    def get_top_processes(self, limit: int = 5) -> List[Dict[str, any]]:
        """Get top processes by memory usage"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                info = proc.info
                processes.append({
                    "pid": info['pid'],
                    "name": info['name'],
                    "memory_mb": info['memory_info'].rss / (1024**2),
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by memory
        processes.sort(key=lambda x: x['memory_mb'], reverse=True)
        return processes[:limit]


# ============================================================================
# Singleton instance
# ============================================================================

_monitor_instance: Optional[SystemMonitor] = None


def get_monitor() -> SystemMonitor:
    """Get the global system monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
    return _monitor_instance


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import json
    
    monitor = SystemMonitor()
    stats = monitor.get_stats()
    
    print("=" * 60)
    print("System Monitor")
    print("=" * 60)
    
    print(f"\nCPU: {stats.cpu_percent:.1f}% ({stats.cpu_count} cores @ {stats.cpu_freq_mhz:.0f} MHz)")
    print(f"RAM: {stats.ram_used_gb:.1f}/{stats.ram_total_gb:.1f} GB ({stats.ram_percent:.1f}%)")
    
    if stats.gpus:
        print(f"\nGPUs:")
        for gpu in stats.gpus:
            print(f"  [{gpu.index}] {gpu.name}")
            print(f"      Memory: {gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f} MB")
            print(f"      Utilization: {gpu.utilization_percent:.1f}%")
            if gpu.temperature_celsius:
                print(f"      Temperature: {gpu.temperature_celsius}°C")
    else:
        print("\nGPU: Not available")
    
    if stats.loaded_models:
        print(f"\nLoaded Models:")
        for model in stats.loaded_models:
            print(f"  {model.model_id}: {model.ram_mb:.1f} MB RAM")
    
    print(f"\nTop Processes:")
    for proc in monitor.get_top_processes(5):
        print(f"  {proc['name']}: {proc['memory_mb']:.1f} MB")
    
    print(f"\nPlatform: {stats.platform}")
    print(f"Python: {stats.python_version}")
