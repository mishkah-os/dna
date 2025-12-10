"""
📥 Professional Download Manager
Handles model downloads with progress tracking, resume capability, and retry logic
"""

import time
import logging
import os
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
            # Use HuggingFace Hub and track progress per file (coarse but visible)
            from huggingface_hub import HfApi, hf_hub_download
            
            api = HfApi()
            logger.info(f"Downloading {model_info.name} from {model_info.hf_name}")

            # List files to know what to fetch; fail fast if empty
            try:
                info = api.model_info(model_info.hf_name)
                siblings = info.siblings or []
            except Exception as e:
                logger.error(f"Cannot list files for {model_info.hf_name}: {e}")
                progress.status = DownloadStatus.FAILED
                progress.error = str(e)
                if progress_callback:
                    progress_callback(progress)
                return False

            if not siblings:
                progress.status = DownloadStatus.FAILED
                progress.error = "No files found in repository"
                if progress_callback:
                    progress_callback(progress)
                return False
            
            total_bytes = sum([(s.size or 0) for s in siblings])
            progress.total_bytes = total_bytes
            last_update = time.time()
            last_bytes = 0
            
            for attempt in range(self.max_retries):
                try:
                    downloaded = 0
                    for sib in siblings:
                        target_size = sib.size or 0
                        local_path = hf_hub_download(
                            repo_id=model_info.hf_name,
                            filename=sib.rfilename,
                            cache_dir=str(self.cache_dir),
                            resume_download=True,
                            local_files_only=False,
                        )
                        # Use reported size or actual file size if missing
                        file_size = target_size or os.path.getsize(local_path)
                        downloaded += file_size
                        progress.downloaded_bytes = downloaded
                        # If HF didn't provide total size, derive from downloaded so far
                        progress.total_bytes = total_bytes if total_bytes > 0 else max(progress.total_bytes, downloaded)
                        progress.percentage = (progress.downloaded_bytes / progress.total_bytes) * 100 if progress.total_bytes else 0
                        
                        now = time.time()
                        if now - last_update > 0:
                            bytes_diff = progress.downloaded_bytes - last_bytes
                            time_diff = now - last_update
                            progress.speed_mbps = (bytes_diff / time_diff) / (1024**2)
                            bytes_remaining = progress.total_bytes - progress.downloaded_bytes
                            progress.eta_seconds = bytes_remaining / (bytes_diff / time_diff) if bytes_diff and time_diff else 0
                            last_update = now
                            last_bytes = progress.downloaded_bytes
                        
                        if progress_callback:
                            progress_callback(progress)
                    
                    if downloaded == 0:
                        raise RuntimeError("Downloaded zero bytes; aborting")

                    # Success!
                    progress.status = DownloadStatus.COMPLETED
                    progress.end_time = time.time()
                    progress.percentage = 100.0
                    progress.speed_mbps = 0.0
                    progress.eta_seconds = 0.0
                    
                    if progress_callback:
                        progress_callback(progress)
                    
                    logger.info(f"Successfully downloaded {model_info.name}")
                    return True
                    
                except Exception as e:
                    progress.status = DownloadStatus.FAILED
                    progress.error = str(e)
                    if progress_callback:
                        progress_callback(progress)
                    logger.warning(f"Download attempt {attempt + 1}/{self.max_retries} failed: {e}")
                    
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        progress.status = DownloadStatus.DOWNLOADING
                        continue
                    else:
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
    def download_model(
        self,
        model_id: str,
        hf_name: Optional[str] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> bool:
        """
        Wrapper expected by api.zoo to start a download.
        hf_name is ignored because the manager already looks up model metadata internally.
        """
        return self.download(model_id=model_id, force=force, progress_callback=progress_callback)


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
