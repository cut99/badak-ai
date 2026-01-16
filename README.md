# BADAK AI Worker - Python Implementation

Python AI Worker untuk menggantikan Azure AI services. Semua logic AI (face recognition, embedding, clustering, vision tagging, captioning) berjalan di worker ini.

## 🎯 Overview

### Fitur Utama
- **Face Recognition & Clustering** - InsightFace + VectorDB (ChromaDB)
- **Vision Tagging** - OpenCLIP zero-shot classification
- **Context Captioning** - BLIP → Indonesian context phrases
- **Auto GPU Detection** - CUDA / Metal / CPU fallback

### API Response
```json
{
  "file_id": "uuid",
  "faces": [{
    "face_id": "uuid",
    "cluster_id": "cluster-123",
    "bounding_box": [x1, y1, x2, y2],
    "confidence": 0.98
  }],
  "tags": ["outdoor", "formal", "group photo"],
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
│   └── image_downloader.py    # Download from presigned URL
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
| POST | `/api/process` | Process image, return faces + tags + context |
| POST | `/api/merge-clusters` | Merge cluster IDs |
| GET | `/api/cluster/{id}/thumbnail` | Get face thumbnail |
| GET | `/health` | Health check |

---

## 🔐 Security

- **API Key** - Header `X-API-Key` required
- **IP Whitelist** - Only allowed IPs can access
- **Internal Deployment** - No public exposure

---

## 📚 Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Detailed architecture
- [SETUP.md](./SETUP.md) - Setup instructions
- [TODO.md](./TODO.md) - Implementation tasks for Claude Code
