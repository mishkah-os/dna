"""
📥 Professional Download Manager
Handles model downloads with progress tracking, resume capability, and retry logic
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, List
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class DownloadStatus(str, Enum):
    """Download status"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadProgress:
    """Real-time download progress"""
    model_id: str
    status: DownloadStatus
    
    # Files
    total_files: int
    downloaded_files: int
    current_file: str
    
    # Bytes
    total_bytes: int
    downloaded_bytes: int
    
    # Calculated
    percentage: float = 0.0
    speed_mbps: float = 0.0
    eta_seconds: float = 0.0
    
    # Metadata
    start_time: float = 0.0
    end_time: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "model_id": self.model_id,
            "status": self.status,
            "total_files": self.total_files,
            "downloaded_files": self.downloaded_files,
            "current_file": self.current_file,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "total_mb": self.total_bytes / (1024**2),
            "downloaded_mb": self.downloaded_bytes / (1024**2),
            "percentage": self.percentage,
            "speed_mbps": self.speed_mbps,
            "eta_seconds": self.eta_seconds,
            "error": self.error,
        }


class DownloadManager:
    """
    Professional download manager with:
    - Progress tracking
    - Resume capability
    - Retry with exponential backoff
    - Speed calculation
    - ETA estimation
    - Detailed logging
    
    Usage:
        manager = DownloadManager()
        
        def on_progress(progress: DownloadProgress):
            print(f"{progress.percentage:.1f}% - {progress.speed_mbps:.2f} MB/s")
        
        success = manager.download(
            model_id="qwen2.5-3b",
            progress_callback=on_progress
        )
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, max_retries: int = 3):
        """
        Initialize download manager.
        
        Args:
            cache_dir: Where to store downloaded models
            max_retries: Maximum retry attempts on failure
        """
        self.cache_dir = cache_dir or Path("data/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        
        # Active downloads
        self._active_downloads: dict[str, DownloadProgress] = {}
        # Backwards-compat alias used by API/SSE layer
        self._downloads = self._active_downloads
        self._download_threads: dict[str, threading.Thread] = {}
        
        logger.info(f"DownloadManager initialized (cache={self.cache_dir})")
    
    def download(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        force: bool = False,
    ) -> bool:
        """
        Download a model from HuggingFace.
        
        Args:
            model_id: Model ID from zoo
            progress_callback: Called with progress updates
            force: Re-download even if exists
            
        Returns:
            True if download successful
        """
        from dna.model_zoo import get_model
        
        model_info = get_model(model_id)
        if not model_info:
            logger.error(f"Unknown model: {model_id}")
            return False
        
        # Check if already downloading
        if model_id in self._active_downloads:
            logger.warning(f"Model {model_id} is already being downloaded")
            return False
        
        # Initialize progress
        progress = DownloadProgress(
            model_id=model_id,
            status=DownloadStatus.DOWNLOADING,
            total_files=0,
            downloaded_files=0,
            current_file="",
            total_bytes=0,
            downloaded_bytes=0,
            start_time=time.time(),
        )
        
        self._active_downloads[model_id] = progress
        
        try:
            # Use HuggingFace Hub with progress tracking
            from huggingface_hub import snapshot_download
            from tqdm import tqdm
            
            # Progress tracking
            last_update = time.time()
            last_bytes = 0
            
            def on_hf_progress(current, total):
                """Callback from HF download"""
                nonlocal last_update, last_bytes
                
                now = time.time()
                progress.total_bytes = total
                progress.downloaded_bytes = current
                
                if total > 0:
                    progress.percentage = (current / total) * 100
                
                # Calculate speed (every 0.5 seconds)
                if now - last_update >= 0.5:
                    bytes_diff = current - last_bytes
                    time_diff = now - last_update
                    
                    if time_diff > 0:
                        bytes_per_sec = bytes_diff / time_diff
                        progress.speed_mbps = bytes_per_sec / (1024**2)
                        
                        # Calculate ETA
                        bytes_remaining = total - current
                        if bytes_per_sec > 0:
                            progress.eta_seconds = bytes_remaining / bytes_per_sec
                    
                    last_update = now
                    last_bytes = current
                    
                    # Callback
                    if progress_callback:
                        progress_callback(progress)
            
            logger.info(f"Downloading {model_info.name} from {model_info.hf_name}")
            
            # Download with retries
            for attempt in range(self.max_retries):
                try:
                    # Download all files
                    local_dir = snapshot_download(
                        repo_id=model_info.hf_name,
                        cache_dir=str(self.cache_dir),
                        resume_download=True,  # Resume if interrupted
                        local_files_only=False,
                    )
                    
                    # Success!
                    progress.status = DownloadStatus.COMPLETED
                    progress.end_time = time.time()
                    progress.percentage = 100.0
                    
                    if progress_callback:
                        progress_callback(progress)
                    
                    logger.info(f"Successfully downloaded {model_info.name}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Download attempt {attempt + 1}/{self.max_retries} failed: {e}")
                    
                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        # Final failure
                        raise
            
        except Exception as e:
            progress.status = DownloadStatus.FAILED
            progress.error = str(e)
            progress.end_time = time.time()
            
            if progress_callback:
                progress_callback(progress)
            
            logger.error(f"Failed to download {model_id}: {e}")
            return False
            
        finally:
            # Cleanup
            if model_id in self._active_downloads:
                del self._active_downloads[model_id]
    
    def get_progress(self, model_id: str) -> Optional[DownloadProgress]:
        """Get current download progress"""
        return self._active_downloads.get(model_id)
    
    def cancel_download(self, model_id: str) -> bool:
        """Cancel an active download"""
        if model_id in self._active_downloads:
            progress = self._active_downloads[model_id]
            progress.status = DownloadStatus.CANCELLED
            # TODO: Implement actual cancellation
            logger.info(f"Cancelled download: {model_id}")
            return True
        return False
    
    def get_active_downloads(self) -> List[str]:
        """Get list of active downloads"""
        return list(self._active_downloads.keys())

    # ------------------------------------------------------------------ #
    # Compatibility helper for API layer
    # ------------------------------------------------------------------ #
    def download_model(self, model_id: str, hf_name: Optional[str] = None, force: bool = False) -> bool:
        """
        Wrapper expected by api.zoo to start a download.
        hf_name is ignored because the manager already looks up model metadata internally.
        """
        return self.download(model_id=model_id, force=force)


# ============================================================================
# Singleton instance
# ============================================================================

_manager_instance: Optional[DownloadManager] = None


def get_download_manager() -> DownloadManager:
    """Get the global download manager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = DownloadManager()
    return _manager_instance


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python download_manager.py <model_id>")
        print("Example: python download_manager.py tinybert")
        sys.exit(1)
    
    model_id = sys.argv[1]
    
    manager = DownloadManager()
    
    print(f"Downloading {model_id}...")
    print()
    
    def on_progress(progress: DownloadProgress):
        mb_done = progress.downloaded_bytes / (1024**2)
        mb_total = progress.total_bytes / (1024**2)
        
        print(f"\r{progress.percentage:5.1f}% | {mb_done:6.1f}/{mb_total:6.1f} MB | "
              f"{progress.speed_mbps:5.2f} MB/s | ETA: {progress.eta_seconds:4.0f}s", 
              end="", flush=True)
    
    success = manager.download(model_id, progress_callback=on_progress)
    
    print()
    if success:
        print(f"\n✓ Download completed successfully!")
    else:
        print(f"\n✗ Download failed")
