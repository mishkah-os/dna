"""
DNA Pattern Explorer - FastAPI Backend
Main application entry point

🏪 Includes Tiny AI Play Store - download and run tiny AI models!
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import os
import uvicorn
import sys

# Add src to python path to allow importing dna package
sys.path.append("src")

from api import models, experiments, patterns, zoo, auth, admin, system, weights
from database.db import init_database

# Create FastAPI app
app = FastAPI(
    title="DNA Pattern Explorer",
    description="DNA Neural Network Pattern Mining & Tiny AI Play Store",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include API routers
# Authentication & Authorization
app.include_router(auth.router, prefix="/api", tags=["🔐 Authentication"])
app.include_router(admin.router, prefix="/api", tags=["👥 Admin"])

# Core Features
app.include_router(zoo.router, prefix="/api", tags=["🏪 Tiny AI Zoo"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(experiments.router, prefix="/api", tags=["experiments"])
app.include_router(patterns.router, prefix="/api", tags=["patterns"])

# System monitoring & Weight Visualization
app.include_router(system.router, prefix="/api", tags=["🖥️ System Monitor"])
app.include_router(weights.router, prefix="/api", tags=["🎨 Weight Visualization"])



@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    await init_database()
    print("[OK] Database initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("[SHUTDOWN] DNA Pattern Explorer")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "DNA Pattern Explorer"}

# Mount static files to root (must be last to avoid shadowing API)
static_path = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")


if __name__ == "__main__":
    host = os.getenv("DNA_HOST", "0.0.0.0")
    port = int(os.getenv("DNA_PORT", "8058"))
    reload_enabled = os.getenv("DNA_RELOAD", "").lower() in {"1", "true", "yes", "on"}
    workers_raw = os.getenv("DNA_WORKERS", "1")
    workers = max(1, int(workers_raw)) if workers_raw.isdigit() else 1

    print("=" * 50)
    print("Starting DNA Pattern Explorer...")
    print("=" * 50)
    print(f"Dashboard: http://{host}:{port}")
    print(f"Play Store: http://{host}:{port}/zoo.html")
    print(f"API Docs: http://{host}:{port}/api/docs")
    print("=" * 50)

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload_enabled,
        workers=1 if reload_enabled else workers,
        log_level="info"
    )
