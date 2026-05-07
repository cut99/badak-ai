# Architecture Documentation

## System Overview

Semua AI logic berjalan di Python worker. C# backend minimal - hanya trigger, simpan hasil, dan proxy.

```
┌─────────────────── C# Backend (Minimal) ──────────────────┐
│                                                            │
│  • Hit Python API untuk process image                     │
│  • Simpan cluster_id hasil ke database                    │
│  • Forward merge request dari user                        │
│  • Proxy untuk serve thumbnail                            │
│                                                            │
└─────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌─────────────────── Python AI Worker ──────────────────────┐
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                  REST API (FastAPI)                   │ │
│  │                                                        │ │
│  │  POST /api/process                                    │ │
│  │  POST /api/process-sync                               │ │
│  │  POST /api/batch-process                              │ │
│  │  POST /api/merge-clusters                             │ │
│  │  GET  /api/jobs/{job_id}                              │ │
│  │  GET  /api/clusters                                   │ │
│  │  GET  /api/cluster/{id}/thumbnail                     │ │
│  │  GET  /health                                          │ │
│  └──────────────────────────────────────────────────────┘ │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ InsightFace  │  │  Florence-2  │  │   opus-mt    │    │
│  │ Face + Embed │  │ Tags+Caption │  │   EN → ID    │    │
│  │              │  │  +OCR+Ground │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              VectorDB (ChromaDB)                      │ │
│  │  • 512-dim face embeddings                           │ │
│  │  • Cluster assignments                               │ │
│  │  • Similarity search (cosine)                        │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              Thumbnail Storage                        │ │
│  │  • Cropped face per cluster                          │ │
│  │  • Local filesystem                                   │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Process Image

```
C# Backend                          Python Worker
    │                                    │
    │  POST /api/process                 │
    │  {file_id, image_url}              │
    │──────────────────────────────────▶│
    │                                    │
    │                              1. Download image
    │                              2. InsightFace → faces + embeddings
    │                              3. Search VectorDB for similar face
    │                              4. Assign/create cluster_id
    │                              5. Florence-2 <OD> + <DENSE_REGION_CAPTION> → tags
    │                              6. Florence-2 <OD> → objects (English)
    │                              7. Florence-2 <MORE_DETAILED_CAPTION> → caption
    │                              8. opus-mt → translate to Indonesian
    │                              9. Florence-2 <CAPTION_TO_PHRASE_GROUNDING> → name injection
    │                              10. Florence-2 <OCR> → text extraction (optional)
    │                              11. Save thumbnail (if new cluster)
    │                                    │
    │  Response:                         │
    │  {faces: [{cluster_id}],           │
    │   tags: [...],                     │
    │   objects: [...],                  │
    │   context: "sedang bersalaman"}    │
    │◀──────────────────────────────────│
    │                                    │
    │  Save cluster_id to database       │
    │                                    │
```

### 2. Merge Clusters

```
User clicks "Merge Person A & B"
         │
         ▼
C# Backend                          Python Worker
    │                                    │
    │  POST /api/merge-clusters          │
    │  {source: ["A","B"],               │
    │   target: "A"}                     │
    │──────────────────────────────────▶│
    │                                    │
    │                              1. Update all embeddings
    │                                 from B → cluster_id = A
    │                              2. Delete cluster B thumbnail
    │                              3. Return merged count
    │                                    │
    │  {success: true,                   │
    │   merged_count: 45}                │
    │◀──────────────────────────────────│
    │                                    │
    │  Update database references        │
    │                                    │
```

---

## AI Models

### InsightFace (Face Recognition)

**Model**: `buffalo_l`
**Output**:
- Bounding box `[x1, y1, x2, y2]`
- 512-dim embedding vector
- Detection confidence (0-1)
- Age estimation
- Gender (0=female, 1=male)

**Device Support**:
- CUDA → `CUDAExecutionProvider`
- Mac → `CoreMLExecutionProvider`
- CPU → `CPUExecutionProvider`

### Florence-2 (Vision — Tags, Objects, Captions, OCR)

**Model**: `microsoft/Florence-2-base`
**Shared Instance**: Loaded once, used by both `FlorenceModel` and `CaptionModel`

**Tasks Used**:

| Task | Purpose | Output |
|------|---------|--------|
| `<OD>` | Object detection → tag labels + object list | `{labels: [...], bboxes: [...]}` |
| `<DENSE_REGION_CAPTION>` | Dense region descriptions → additional tags | `{labels: [...]}` |
| `<MORE_DETAILED_CAPTION>` | Full English caption | `"Two women shaking hands..."` |
| `<CAPTION_TO_PHRASE_GROUNDING>` | Ground caption phrases to bboxes (for name injection) | `{bboxes: [...], labels: [...]}` |
| `<OCR>` | Plain text extraction | `{text: "..."}` |
| `<OCR_WITH_REGION>` | OCR with bounding boxes | `{quad_boxes: [...], labels: [...]}` |

**Tag Pipeline**:
1. Run `<OD>` → get object labels
2. Run `<DENSE_REGION_CAPTION>` → get descriptive phrases
3. Clean prefixes ("a ", "an ", "the ")
4. Filter valid tags (1-8 words, no special chars)
5. Deduplicate (case-insensitive)
6. Translate to Indonesian via opus-mt (if `TAG_LANGUAGE=id`)

### opus-mt (English → Indonesian Translation)

**Model**: `Helsinki-NLP/opus-mt-en-id` (MarianMT)
**Purpose**: Translate tags, captions, and descriptions to Indonesian

**Features**:
- Batch translation: single forward pass for multiple strings
- Graceful degradation: returns original English on error
- CPU-only: MarianMT is small and fast on CPU

### Caption & Name Injection (CaptionModel)

**Caption Flow**:
1. Florence-2 `<MORE_DETAILED_CAPTION>` → English caption
2. Clean "The image shows..." prefixes for SPOK output
3. Name injection (3 strategies):
   - **Grounding**: `<CAPTION_TO_PHRASE_GROUNDING>` + IoU matching with known faces
   - **Positional**: Replace "a woman", "the man" with known names
   - **Group fallback**: Replace "two women" → "Sri Mulyani I and a woman"
4. Translate to short Indonesian phrase (≤10 words)
5. Translate to full Indonesian description (sentence-by-sentence)

---

## Face Clustering

### Algorithm

```python
def find_or_create_cluster(embedding: np.ndarray) -> str:
    """
    1. Search VectorDB for similar face (top 1)
    2. If similarity >= 0.6, return existing cluster_id
    3. Else create new cluster, return new cluster_id
    """
    results = vectordb.search(embedding, top_k=1)
    
    if results and results[0].distance <= 0.4:  # cosine distance
        return results[0].metadata["cluster_id"]
    else:
        new_cluster_id = str(uuid4())
        vectordb.add(embedding, metadata={"cluster_id": new_cluster_id})
        return new_cluster_id
```

### VectorDB Schema (ChromaDB)

```python
collection = client.create_collection(
    name="face_embeddings",
    metadata={"hnsw:space": "cosine"}
)

# Each document
{
    "id": "face-uuid",
    "embedding": [512 floats],
    "metadata": {
        "cluster_id": "cluster-uuid",
        "file_id": "original-file-uuid",
        "bounding_box": "[x1,y1,x2,y2]",
        "created_at": "2024-01-16T08:00:00Z"
    }
}
```

### Merge Operation

```python
def merge_clusters(source_ids: list, target_id: str) -> int:
    """
    Update all faces from source clusters to target cluster.
    """
    count = 0
    for source_id in source_ids:
        if source_id == target_id:
            continue
        
        # Get all faces with source cluster_id
        faces = collection.get(
            where={"cluster_id": source_id}
        )
        
        # Update to target cluster_id
        for face in faces["ids"]:
            collection.update(
                ids=[face],
                metadatas=[{"cluster_id": target_id}]
            )
            count += 1
        
        # Delete source thumbnail
        thumbnail_service.delete(source_id)
    
    return count
```

---

## Security

### 1. API Key Middleware

```python
async def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-Key")
    if api_key != settings.API_KEY:
        raise HTTPException(401, "Invalid API key")
```

### 2. IP Whitelist Middleware

```python
ALLOWED_IPS = ["127.0.0.1", "192.168.1.0/24", "10.0.0.0/8"]

async def verify_ip(request: Request):
    client_ip = request.client.host
    for allowed in ALLOWED_IPS:
        if "/" in allowed:
            if ip_address(client_ip) in ip_network(allowed):
                return
        elif client_ip == allowed:
            return
    raise HTTPException(403, f"IP {client_ip} not allowed")
```

### 3. Request Logging

```python
async def log_request(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    logger.info(
        f"[{request.client.host}] "
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration:.2f}s)"
    )
    return response
```

---

## Performance

### Benchmarks

| Device | Per Image | 200k Photos |
|--------|-----------|-------------|
| RTX 3060 | ~0.3s | ~17 hours |
| Mac M2 | ~0.5s | ~28 hours |
| CPU (16-core) | ~2.5s | ~139 hours |

### Optimization Tips

1. **Batch processing** - Group images for VectorDB operations
2. **Async download** - Use httpx async for image download
3. **Model caching** - Load models once at startup
4. **Connection pooling** - Reuse ChromaDB connections
5. **Shared Florence-2** - Single model instance for tags + captions

---

## Backup & Recovery

### VectorDB Backup

```bash
# Backup
cp -r data/vectordb/ backup/vectordb_$(date +%Y%m%d)/

# Restore
cp -r backup/vectordb_20240116/ data/vectordb/
```

### Thumbnail Backup

```bash
# Backup
tar -czvf thumbnails_backup.tar.gz data/thumbnails/

# Restore
tar -xzvf thumbnails_backup.tar.gz -C data/
```

### Recovery from Corruption

Jika VectorDB corrupt, perlu re-process semua foto:
1. Delete `data/vectordb/`
2. Re-run batch processing untuk semua file
