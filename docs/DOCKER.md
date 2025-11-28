# 🐳 Docker Deployment Guide

## Quick Start with Docker

### 1. Build Image
```bash
docker-compose build dna-web
```

### 2. Run Web Server
```bash
# CPU version
docker-compose up dna-web

# GPU version (requires nvidia-docker)
docker-compose --profile gpu up dna-web-gpu

# Development mode
docker-compose --profile dev up dna-dev
```

### 3. Access Dashboard
Visit: **http://localhost:8000**

---

## Docker Commands

### Build
```bash
# Build image
docker build -t dna-explorer .

# Build with docker-compose
docker-compose build
```

### Run
```bash
# Run in background
docker-compose up -d dna-web

# View logs
docker-compose logs -f dna-web

# Stop
docker-compose down
```

### Access Container
```bash
# Execute command
docker-compose exec dna-web bash

# View database
docker-compose exec dna-web ls -la /app/data
```

---

## Volumes

Data persists in these volumes:

```yaml
volumes:
  - ./data:/app/data        # Database + uploaded models
  - ./cache:/app/cache      # HuggingFace cache
  - ./logs:/app/logs        # Application logs
```

---

## Environment Variables

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - TRANSFORMERS_CACHE=/app/cache
  - HF_HOME=/app/cache
  - TORCH_HOME=/app/cache
```

---

## Health Check

Container includes health check:
```bash
curl http://localhost:8000/health
```

Returns: `{"status": "healthy"}`

---

## Production Deployment

```bash
# Build production image
docker build -t dna-explorer:prod .

# Run with resource limits
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --memory=8g \
  --cpus=4 \
  --name dna-explorer \
  dna-explorer:prod
```

---

## GPU Support

Requires **nvidia-docker** installed.

```bash
# Test GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Run with GPU
docker-compose --profile gpu up dna-web-gpu
```

---

## Troubleshooting

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Permission denied
```bash
# Fix ownership
sudo chown -R 1000:1000 data/ cache/ logs/
```

### Out of memory
```bash
# Increase limits in docker-compose.yml
mem_limit: 16g
```
