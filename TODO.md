# TODO - Claude Code Implementation Tasks

Task list untuk implementasi AI Worker. Gunakan file ini sebagai panduan.

---

## Phase 1: Project Setup

### 1.1 Create Project Structure
- [x] Create `python-worker/` directory
- [x] Create subdirectories: `api/`, `models/`, `services/`, `middleware/`, `utils/`, `data/`, `tests/`
- [x] Create `__init__.py` in each directory

### 1.2 Create Configuration Files
- [x] Create `requirements.txt` with dependencies:
  ```
  fastapi==0.109.0
  uvicorn==0.27.0
  pydantic==2.5.3
  insightface==0.7.3
  onnxruntime==1.16.0
  open-clip-torch==2.24.0
  transformers==4.36.0
  torch==2.1.0
  chromadb==0.4.22
  pillow==10.2.0
  numpy==1.26.3
  opencv-python==4.9.0.80
  httpx==0.26.0
  python-dotenv==1.0.0
  ```
- [x] Create `.env.example`
- [x] Create `config.py` - load environment variables

### 1.3 Device Detection
- [x] Create `utils/device_detector.py`
  - Detect CUDA availability
  - Detect MPS (Mac Metal) availability
  - Fallback to CPU
  - Return device type and ONNX providers

---

## Phase 2: AI Models

### 2.1 InsightFace Model
- [x] Create `models/insightface_model.py`
  - Load `buffalo_l` model
  - Method: `detect_faces(image) -> List[Face]`
  - Face contains: bounding_box, embedding (512-dim), confidence, age, gender
  - Use device-appropriate ONNX provider

### 2.2 OpenCLIP Model
- [x] Create `models/openclip_model.py`
  - Load `ViT-B-32` model
  - Define tag templates (indoor, outdoor, meeting, ceremony, etc.)
  - Method: `get_tags(image, threshold=0.25) -> List[str]`
  - Use device-appropriate torch device

### 2.3 BLIP Model
- [x] Create `models/blip_model.py`
  - Load `blip-image-captioning-base`
  - Define CONTEXT_MAPPING (English keyword → Indonesian phrase)
  - Method: `get_context(image) -> str`
  - Match generated caption to closest Indonesian context phrase

---

## Phase 3: Services

### 3.1 VectorDB Service
- [x] Create `services/vectordb.py`
  - Initialize ChromaDB with persistent storage
  - Collection: `face_embeddings` with cosine distance
  - Method: `add_face(face_id, embedding, cluster_id, metadata)`
  - Method: `search_similar(embedding, top_k) -> List[Result]`
  - Method: `update_cluster(face_ids, new_cluster_id)`
  - Method: `get_faces_by_cluster(cluster_id) -> List[Face]`
  - Method: `delete_cluster(cluster_id)`

### 3.2 Clustering Service
- [x] Create `services/clustering_service.py`
  - Method: `find_or_create_cluster(embedding) -> cluster_id`
    - Search VectorDB for similar face
    - If similarity >= 0.6, return existing cluster_id
    - Else create new cluster, save to VectorDB, return new cluster_id
  - Method: `merge_clusters(source_ids, target_id) -> merged_count`
    - Update all faces from source clusters to target
    - Delete source thumbnails

### 3.3 Thumbnail Service
- [x] Create `services/thumbnail_service.py`
  - Method: `save_thumbnail(cluster_id, face_crop)`
    - Save cropped face image as cluster thumbnail
    - Only save if not exists (first face = representative)
  - Method: `get_thumbnail(cluster_id) -> bytes`
  - Method: `delete_thumbnail(cluster_id)`

### 3.4 Image Downloader
- [x] Create `services/image_downloader.py`
  - Method: `download_image(url) -> PIL.Image`
  - Use httpx async client
  - Handle errors gracefully

---

## Phase 4: Security Middleware

### 4.1 API Key Middleware
- [x] Create `middleware/security.py`
  - Function: `verify_api_key(request)`
    - Check `X-API-Key` header
    - Raise 401 if invalid

### 4.2 IP Whitelist Middleware
- [x] Add to `middleware/security.py`
  - Function: `verify_ip_whitelist(request)`
    - Check client IP against ALLOWED_IPS
    - Support CIDR notation (e.g., 192.168.1.0/24)
    - Raise 403 if not allowed

### 4.3 Request Logging
- [x] Add to `middleware/security.py`
  - Middleware: `log_requests`
    - Log IP, method, path, status, duration

---

## Phase 5: API Endpoints

### 5.1 API Schemas
- [x] Create `api/schemas.py`
  ```python
  class ProcessRequest(BaseModel):
      file_id: str
      image_url: str

  class FaceResult(BaseModel):
      face_id: str
      cluster_id: str
      bounding_box: List[int]
      confidence: float
      is_new_cluster: bool

  class ProcessResponse(BaseModel):
      file_id: str
      faces: List[FaceResult]
      tags: List[str]
      context: str

  class MergeRequest(BaseModel):
      source_cluster_ids: List[str]
      target_cluster_id: str

  class MergeResponse(BaseModel):
      success: bool
      merged_count: int
      target_cluster_id: str
  ```

### 5.2 API Routes
- [x] Create `api/routes.py`
  - `POST /api/process`
    1. Download image from URL
    2. Run InsightFace → get faces + embeddings
    3. For each face: find_or_create_cluster → get cluster_id
    4. Crop face, save thumbnail if new cluster
    5. Run OpenCLIP → get tags
    6. Run BLIP → get context
    7. Return ProcessResponse

  - `POST /api/merge-clusters`
    1. Call clustering_service.merge_clusters
    2. Return MergeResponse

  - `GET /api/cluster/{cluster_id}/thumbnail`
    1. Get thumbnail from thumbnail_service
    2. Return image/jpeg

  - `GET /health`
    1. Return status, device, vectordb count, models loaded

### 5.3 Main Application
- [x] Create `main.py`
  - Create FastAPI app
  - Add CORS middleware (internal only)
  - Add security middlewares
  - Include routes
  - Startup event: load all models

---

## Phase 6: Utilities

### 6.1 Logger
- [x] Create `utils/logger.py`
  - Configure structured logging
  - Log to file and console

---

## Phase 7: Testing

### 7.1 Unit Tests
- [x] Create `tests/test_models.py`
  - Test InsightFace detection
  - Test OpenCLIP tagging
  - Test BLIP context matching

- [x] Create `tests/test_clustering.py`
  - Test find_or_create_cluster
  - Test merge_clusters

- [x] Create `tests/test_api.py`
  - Test /api/process endpoint
  - Test /api/merge-clusters endpoint
  - Test /api/cluster/{id}/thumbnail endpoint
  - Test security (invalid API key, blocked IP)

---

## Phase 8: Integration

### 8.1 C# Backend Integration
- [ ] Update C# untuk hit Python API
- [ ] Simpan cluster_id dari response ke database
- [ ] Implement merge forward ke Python

---

## Quick Reference

### File Structure
```
python-worker/
├── main.py
├── config.py
├── requirements.txt
├── .env.example
├── api/
│   ├── __init__.py
│   ├── routes.py
│   └── schemas.py
├── models/
│   ├── __init__.py
│   ├── insightface_model.py
│   ├── openclip_model.py
│   └── blip_model.py
├── services/
│   ├── __init__.py
│   ├── vectordb.py
│   ├── clustering_service.py
│   ├── thumbnail_service.py
│   └── image_downloader.py
├── middleware/
│   ├── __init__.py
│   └── security.py
├── utils/
│   ├── __init__.py
│   ├── device_detector.py
│   └── logger.py
├── data/
│   ├── vectordb/
│   └── thumbnails/
└── tests/
    ├── test_models.py
    ├── test_clustering.py
    └── test_api.py
```

### Run Order
1. Phase 1 → Setup project
2. Phase 2 → Implement models (can test individually)
3. Phase 3 → Implement services
4. Phase 4 → Add security
5. Phase 5 → Create API
6. Phase 6 → Add logging
7. Phase 7 → Test everything
8. Phase 8 → Integrate with C#
