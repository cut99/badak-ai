# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BADAK AI Worker is a Python FastAPI microservice that handles all AI/ML computation for a C# backend. It provides face recognition & clustering, image tagging, and Indonesian-language context captioning — replacing Azure AI with a self-hosted solution.

## Commands

All commands run from `python-worker/` with the virtualenv activated.

```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac
pip install -r requirements.txt
cp .env.example .env

# Run (development)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run (production)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_api.py -v

# Run a single test by name
pytest tests/test_api.py::test_health_check -v

# Run tests by mark
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m "not slow"

# Health check
curl http://localhost:8000/health
```

## Architecture

### System Role
This Python worker sits behind a **C# backend**. The C# side acts as a thin controller (triggers processing, stores results, proxies thumbnails). This worker owns all AI/ML logic.

### Request Flow
```
Image URL → ImageDownloader → InsightFace (detect faces + 512-dim embeddings)
                            → ClusteringService → ChromaDB (find/create cluster)
                            → ThumbnailService (save face crop if new cluster)
                            → FlorenceModel (open-vocabulary tags + English objects)
                            → CaptionModel (English caption → Indonesian description, name injection)
                            → FlorenceModel OCR (optional, when ENABLE_OCR=true)
                            → Merge context elements into tags
```

### Key Architectural Patterns

**Async vs Sync Processing:**
- `POST /api/process`, `/api/batch-process`, `/api/merge-clusters` — enqueue to `JobQueueService`, return `job_id` immediately
- `POST /api/process-sync` — bypasses queue, runs `process_image_handler()` directly, returns full result
- Poll `GET /api/jobs/{job_id}` for async results (statuses: `queued → processing → completed/failed`)

**Job Queue (`services/job_queue.py`):**
- In-memory FIFO queue with a configurable worker pool (default: 3 workers)
- Handlers are registered by job type (`"process"`, `"batch_process"`, `"merge_clusters"`)
- Jobs are cleaned up after `JOB_RETENTION_HOURS` (default: 24h)
- Duration estimation learns from job history

**Face Clustering (`services/clustering_service.py` + `services/vectordb.py`):**
- Each detected face gets a 512-dim embedding from InsightFace
- ChromaDB searches for the nearest existing embedding (cosine distance)
- If distance < `FACE_SIMILARITY_THRESHOLD` (default: 0.6), face joins that cluster; otherwise a new cluster is created
- Clusters can be merged manually via `/api/merge-clusters`
- Only the first face in a cluster gets a thumbnail (representative image)

**Middleware order in `main.py`:**
CORS → RequestLogging → IPWhitelist → APIKey

Paths exempt from auth: `/health`, `/docs`, `/redoc`, `/openapi.json`

### File Map

| Path | Purpose |
|---|---|
| `main.py` | App entry point, lifespan startup (initializes all services/models), middleware registration |
| `config.py` | Pydantic `Settings` loaded from `.env` |
| `api/routes.py` | All endpoints + job handler functions (`process_image_handler`, `batch_process_handler`, `merge_clusters_handler`) |
| `api/schemas.py` | All Pydantic request/response models |
| `models/insightface_model.py` | Face detection → `Face(bounding_box, embedding, confidence, age, gender)` |
| `models/florence_model.py` | Open-vocabulary tagging via Florence-2 `<OD>` + `<DENSE_REGION_CAPTION>`, object detection, OCR |
| `models/caption_model.py` | Image captioning via Florence-2 `<MORE_DETAILED_CAPTION>` + opus-mt translation, name injection, school age detection |
| `models/translation_model.py` | English → Indonesian translation wrapper (Helsinki-NLP/opus-mt-en-id) |
| `services/clustering_service.py` | find-or-create cluster logic, merge clusters, update names |
| `services/vectordb.py` | ChromaDB wrapper — add/search/update/delete face embeddings |
| `services/job_queue.py` | Async job management, worker pool, TTL cleanup |
| `services/thumbnail_service.py` | Save/retrieve JPEG face crops from `./data/thumbnails/` |
| `services/image_downloader.py` | Async `httpx` image download from presigned URLs |
| `middleware/security.py` | `APIKeyMiddleware`, `IPWhitelistMiddleware`, `RequestLoggingMiddleware` |
| `utils/device_detector.py` | Auto-detect CUDA / MPS / CPU, returns ONNX providers for InsightFace |
| `data/vectordb/` | ChromaDB persistent storage |
| `data/thumbnails/` | Face crop JPEGs keyed by cluster_id |

## Configuration (`.env`)

```env
DEVICE=                          # auto if empty; options: cuda, mps, cpu
API_KEY=your-secure-key
ALLOWED_IPS=127.0.0.1,192.168.1.0/24
VECTORDB_PATH=./data/vectordb
THUMBNAIL_PATH=./data/thumbnails
FACE_SIMILARITY_THRESHOLD=0.6   # lower = stricter clustering
TAG_THRESHOLD=0.25
TAG_TOP_K=10
TAG_LANGUAGE=id                  # id=Indonesian, en=English
JOB_QUEUE_MAX_WORKERS=3
JOB_RETENTION_HOURS=24
ENABLE_AGE_DETECTION=true
ENABLE_OCR=false                # Set to true to enable OCR text extraction
LOG_LEVEL=INFO
```

## Testing

Tests live in `python-worker/tests/`. Key files:
- `test_api.py` — endpoint and security tests
- `test_models.py` — InsightFace, OpenCLIP, BLIP model tests
- `test_clustering.py` — ClusteringService + VectorDB tests
- `test_cluster_naming.py` — cluster naming
- `test_sync_route_verification.py` — sync processing

`pytest.ini` sets `asyncio_mode = auto`. Test marks: `unit`, `integration`, `slow`.

For local file testing: `python tests/scripts/test_local_file.py`
