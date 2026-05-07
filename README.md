# BADAK AI Worker - Python Implementation

Python AI Worker untuk menggantikan Azure AI services. Semua logic AI (face recognition, embedding, clustering, vision tagging, captioning, OCR) berjalan di worker ini.

## 🎯 Overview

### Fitur Utama
- **Face Recognition & Clustering** - InsightFace + VectorDB (ChromaDB)
- **Vision Tagging & Object Detection** - Florence-2 (open-vocabulary `<OD>` + `<DENSE_REGION_CAPTION>`)
- **Context Captioning** - Florence-2 (`<MORE_DETAILED_CAPTION>`) → opus-mt EN→ID translation
- **OCR Text Extraction** - Florence-2 (`<OCR>` / `<OCR_WITH_REGION>`)
- **Name Injection** - Florence-2 grounding + positional/group fallback
- **Async Job Queue** - Non-blocking processing for batch operations
- **Synchronous Processing** - Direct immediate processing for interactive use
- **Auto GPU Detection** - CUDA / Metal / CPU fallback

### AI Models

| Model | Purpose | Source |
|-------|---------|--------|
| InsightFace `buffalo_l` | Face detection + 512-dim embedding | InsightFace |
| Florence-2 (`microsoft/Florence-2-base`) | Tagging, object detection, captioning, OCR, grounding | Microsoft |
| opus-mt-en-id (`Helsinki-NLP/opus-mt-en-id`) | English → Indonesian translation | Helsinki-NLP |

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
│ • Merge request │───────────▶  │ ├─ Florence-2 (tags+objects)│
│ • Get thumbnail │◀─────────────│ ├─ Florence-2 (caption)     │
└─────────────────┘              │ ├─ opus-mt (EN→ID)          │
                                 │ ├─ Florence-2 (OCR)         │
                                 │ └─ ChromaDB (clustering)    │
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
│   ├── florence_model.py      # Vision tagging, object detection, OCR (Florence-2)
│   ├── caption_model.py       # Captioning via Florence-2 + name injection
│   └── translation_model.py   # EN→ID translation (opus-mt)
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
- [ENHANCEMENTS.md](./ENHANCEMENTS.md) - Enhancement history (V1)
- [ENHANCEMENTS_V2.md](./ENHANCEMENTS_V2.md) - Enhancement history (V2)
- [TODO.md](./TODO.md) - Implementation tasks
