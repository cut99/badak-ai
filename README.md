# BADAK AI Worker - Python Implementation

Python AI Worker untuk menggantikan Azure AI services. Semua logic AI (face recognition, embedding, clustering, vision tagging, captioning) berjalan di worker ini.

## 🎯 Overview

### Fitur Utama
- **Face Recognition & Clustering** - InsightFace + VectorDB (ChromaDB)
- **Vision Tagging & Object Detection** - Combined OpenCLIP + BLIP
- **Context Captioning** - BLIP → Indonesian context phrases
- **Async Job Queue** - Non-blocking processing for batch operations
- **Synchronous Processing** - Direct immediate processing for interactive use
- **Auto GPU Detection** - CUDA / Metal / CPU fallback

### API Response (Job Result)
```json
{
  "file_id": "uuid",
  "faces": [{
    "face_id": "uuid",
    "cluster_id": "cluster-123",
    "name": "Jokowi",
    "cluster_name": "Jokowi",
    "bounding_box": [x1, y1, x2, y2],
    "confidence": 0.98,
    "is_new_cluster": false
  }],
  "tags": ["outdoor", "formal", "group photo", "3 orang"],
  "objects": ["person", "chair", "table"],
  "context": "sedang bersalaman"
}
```

---

## 🏗️ Architecture

```
C# Backend (Minimal)              Python AI Worker
┌─────────────────┐              ┌─────────────────────────────┐
│ • Trigger AI    │───────────▶  │ POST /api/process           │
│ • Store results │◀─────────────│ ├─ InsightFace (face+emb)   │
│ • Merge request │───────────▶  │ ├─ OpenCLIP (tags)          │
│ • Get thumbnail │◀─────────────│ ├─ BLIP (context)           │
└─────────────────┘              │ └─ ChromaDB (clustering)    │
                                 │                              │
                                 │ POST /api/process-sync       │
                                 │ POST /api/merge-clusters     │
                                 │ GET  /api/cluster/{id}/thumb │
                                 └─────────────────────────────┘
```

---

## 📁 Project Structure

```
python-worker/
├── main.py                    # FastAPI entry point
├── config.py                  # Configuration
├── requirements.txt
├── .env.example
│
├── api/
│   ├── routes.py              # API endpoints
│   └── schemas.py             # Pydantic models
│
├── models/
│   ├── insightface_model.py   # Face detection + embedding
│   ├── openclip_model.py      # Vision tagging
│   └── blip_model.py          # Context captioning
│
├── services/
│   ├── clustering_service.py  # Face clustering logic
│   ├── vectordb.py            # ChromaDB integration
│   ├── thumbnail_service.py   # Face crop storage
│   ├── image_downloader.py    # Download from presigned URL
│   └── job_queue.py           # Async job management
│
├── middleware/
│   └── security.py            # API key + IP whitelist
│
├── utils/
│   ├── device_detector.py     # GPU/CPU detection
│   └── logger.py
│
├── data/
│   ├── vectordb/              # ChromaDB persistence
│   └── thumbnails/            # Face crop images
│
└── tests/
```

---

## 🚀 Quick Start

```bash
cd python-worker

# Setup environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your settings

# Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/process` | Submit image for async processing (returns job_id) |
| POST | `/api/process-sync` | Process image synchronously (returns result immediately) |
| POST | `/api/batch-process` | Submit batch of images (returns job_id) |
| POST | `/api/merge-clusters` | Submit cluster merge job (returns job_id) |
| GET | `/api/jobs/{job_id}` | Check job status and get results |
| GET | `/api/clusters` | Get paginated gallery of face clusters |
| GET | `/api/cluster/{cluster_id}/thumbnail` | Get face thumbnail image |
| GET | `/health` | Health check and system stats |

---

## 🔐 Security

- **API Key** - Header `X-API-Key` required
- **IP Whitelist** - Only allowed IPs can access
- **Internal Deployment** - No public exposure

---

## 📚 Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Detailed architecture
- [SETUP.md](./SETUP.md) - Setup instructions
- [INSTALL_MACOS.md](./INSTALL_MACOS.md) - MacOS specific installation guide
- [ENHANCEMENTS.md](./ENHANCEMENTS.md) - Planned enhancements
- [TODO.md](./TODO.md) - Implementation tasks
