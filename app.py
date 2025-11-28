"""
DNA Pattern Explorer - FastAPI Backend
Main application entry point
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import uvicorn
import sys

# Add src to python path to allow importing dna package
sys.path.append("src")

from api import models, experiments, patterns
from database.db import init_database

# Create FastAPI app
app = FastAPI(
    title="DNA Pattern Explorer",
    description="ML Model Pattern Mining & Visualization System",
    version="1.0.0",
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

# Include API routers
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(experiments.router, prefix="/api", tags=["experiments"])
app.include_router(patterns.router, prefix="/api", tags=["patterns"])


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    await init_database()
    print("✅ Database initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down DNA Pattern Explorer")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "DNA Pattern Explorer"}

# Mount static files to root (must be last to avoid shadowing API)
static_path = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")


if __name__ == "__main__":
    print("🚀 Starting DNA Pattern Explorer...")
    print("📍 Dashboard: http://localhost:8058")
    print("📚 API Docs: http://localhost:8058/api/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8058,
        reload=True,
        log_level="info"
    )
