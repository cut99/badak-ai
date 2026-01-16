# Setup Guide

## Prerequisites

### System Requirements

**Minimum (CPU)**:
- CPU: 8+ cores
- RAM: 16GB
- Storage: 50GB free

**Recommended (GPU)**:
- NVIDIA RTX 3060+ (12GB VRAM) atau Mac M1/M2/M3
- RAM: 16GB
- Storage: 50GB free

### Software

- Python 3.9+
- MinIO (running)

---

## Installation

### 1. Create Virtual Environment

```bash
cd ai-worker-implementation/python-worker

python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**GPU-specific**:

```bash
# NVIDIA CUDA
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu==1.16.0

# Mac Metal - included by default
```

### 3. Download AI Models

```bash
# Auto-download on first run, atau pre-download:
python -c "from models.insightface_model import InsightFaceModel; InsightFaceModel()"
python -c "from models.openclip_model import OpenCLIPModel; OpenCLIPModel()"
python -c "from models.blip_model import BLIPModel; BLIPModel()"
```

---

## Configuration

### Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Device (auto-detect if empty)
# DEVICE=cuda
# DEVICE=mps
# DEVICE=cpu

# API Security
API_KEY=generate-secure-key-here
ALLOWED_IPS=127.0.0.1,192.168.1.0/24

# VectorDB
VECTORDB_PATH=./data/vectordb

# Thumbnails
THUMBNAIL_PATH=./data/thumbnails

# Model settings
FACE_SIMILARITY_THRESHOLD=0.6
TAG_THRESHOLD=0.25

# Logging
LOG_LEVEL=INFO
```

### Generate API Key

```bash
openssl rand -base64 32
```

---

## Running

### Development

```bash
cd python-worker
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Systemd Service (Linux)

```ini
# /etc/systemd/system/ai-worker.service
[Unit]
Description=BADAK AI Worker
After=network.target

[Service]
User=badak
WorkingDirectory=/opt/ai-worker/python-worker
Environment="PATH=/opt/ai-worker/python-worker/venv/bin"
ExecStart=/opt/ai-worker/python-worker/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ai-worker
sudo systemctl start ai-worker
```

---

## Testing

### Health Check

```bash
curl http://localhost:8000/health
```

### Process Image

```bash
curl -X POST http://localhost:8000/api/process \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "test-123",
    "image_url": "http://minio:9000/bucket/image.jpg?presigned..."
  }'
```

### Merge Clusters

```bash
curl -X POST http://localhost:8000/api/merge-clusters \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "source_cluster_ids": ["cluster-1", "cluster-2"],
    "target_cluster_id": "cluster-1"
  }'
```

### Get Thumbnail

```bash
curl http://localhost:8000/api/cluster/cluster-123/thumbnail \
  -H "X-API-Key: your-key" \
  --output face.jpg
```

---

## Troubleshooting

### CUDA Not Found

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit if needed
# https://developer.nvidia.com/cuda-downloads
```

### Model Download Fails

```bash
# Use proxy
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

### VectorDB Issues

```bash
# Reset VectorDB
rm -rf data/vectordb
# Restart worker - will create fresh DB
```
