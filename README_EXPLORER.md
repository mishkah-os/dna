# 🧬 DNA Pattern Explorer

Web application for exploring geometric patterns in ML model weights using SIREN-based pattern mining.

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Start Server**
```bash
python app.py
```

3. **Open Browser**
- Dashboard: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

## Features

### Backend
- ✅ FastAPI async REST API
- ✅ SQLite database with SQLAlchemy
- ✅ Model upload & HuggingFace download
- ✅ Experiment tracking
- ✅ Pattern visualization data

### Frontend
- ✅ Mishkah DSL components
- ✅ Plotly.js 3D visualization
- ✅ Chart.js 2D charts  
- ✅ Arabic/English support (RTL/LTR)
- ✅ Dark/Light themes

## Pages

- `/` - Main dashboard
- `/static/viz.html` - 3D Pattern Visualization
- `/static/models.html` - Model Zoo (TODO)
- `/static/patterns.html` - Pattern Analysis (TODO)
- `/static/experiment.html` - Experiment Runner (TODO)

## API Endpoints

**Models:**
- `GET /api/models` - List models
- `POST /api/models/upload` - Upload model
- `POST /api/models/download` - Download from HuggingFace

**Experiments:**
- `GET /api/experiments` - List experiments
- `POST /api/experiments` - Create experiment
- `POST /api/experiments/{id}/start` - Start experiment

**Patterns:**
- `GET /api/patterns` - List patterns
- `GET /api/patterns/{id}/viz` - Get 3D visualization data
- `GET /api/patterns/compare` - Compare patterns

## Architecture

```
Frontend (Mishkah + Plotly)
    ↕ REST API
Backend (FastAPI)
    ↕ SQLAlchemy
Database (SQLite)
```

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, SQLAlchemy, asyncio
- **Database**: SQLite (aiosqlite)
- **Frontend**: Mishkah DSL, Plotly.js, Chart.js
- **ML**: PyTorch, Transformers, SafeTensors

## Development

```bash
# Run with auto-reload
python app.py

# API Documentation
Open http://localhost:8000/api/docs
```
