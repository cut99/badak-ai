# REQUIREMENTS.md — BADAK AI Worker: Vision Intelligence Upgrade

## Functional Requirements

### FR-01: Open-vocabulary image tagging
Replace the static 117-item OpenCLIP tag dictionary with Florence-2-based open-vocabulary tagging.
- `get_tags(image) → List[str]` must return Indonesian-language tags without a predefined label list
- Tags must be derived from `<OD>` labels and `<DENSE_REGION_CAPTION>` region phrases
- All tags must be translated to Indonesian via opus-mt-en-id
- Must respect `TAG_TOP_K` config to cap result count
- **Acceptance:** Calling `get_tags()` on an arbitrary image returns a non-empty Indonesian tag list with no hardcoded vocabulary

### FR-02: Open-vocabulary object detection
Replace the static ~170-item OpenCLIP object noun list with Florence-2 `<OD>` output.
- `get_objects(image) → List[str]` must return English object labels
- Must deduplicate results
- **Acceptance:** Calling `get_objects()` returns English noun strings without any predefined noun list

### FR-03: Fluent Indonesian caption generation
Replace brittle BLIP keyword-mapping with genuine model-based captioning + translation.
- `get_context_comprehensive(image) → dict` must return the same dict structure:
  `{english_caption, indonesian_phrase, indonesian_description, elements}`
- `english_caption` — Florence-2 `<MORE_DETAILED_CAPTION>` output
- `indonesian_phrase` — short translated phrase (≤ 10 words)
- `indonesian_description` — full fluent Indonesian paragraph (1–3 sentences)
- `elements` dict — decomposed people/activity/setting/objects/mood structure (same shape as before, consumed by routes.py merge logic)
- **Acceptance:** Description for a handshake photo returns a coherent Indonesian sentence, not "foto bersama"

### FR-04: SPOK caption with name injection
When face cluster names are available, inject them into the caption.
- `get_context_comprehensive(image, known_faces=None)` accepts optional `known_faces` list
  `[{"name": str, "bbox": [x1, y1, x2, y2]}, ...]`
- Uses `<CAPTION_TO_PHRASE_GROUNDING>` to match face regions to caption phrases
- Post-processes generic references ("a man", "a woman", "the man") → known names
- Falls back gracefully when names are unknown or grounding confidence is low
- **Acceptance:** Image with a named cluster produces caption containing the person's name; image with unnamed clusters produces a generic but grammatical description

### FR-05: Built-in OCR
Expose Florence-2's native OCR capabilities via a new model method.
- `get_ocr(image, with_regions=False) → dict`
  - `with_regions=False` → `{"text": str}` (flat extraction via `<OCR>`)
  - `with_regions=True` → `{"text": str, "regions": [{"text": str, "bbox": [8 coords]}]}` (via `<OCR_WITH_REGION>`)
- OCR results populated in `ProcessResponse.ocr` (optional field, `null` when disabled)
- Controlled by `ENABLE_OCR` config setting (default: `false`)
- **Acceptance:** Image containing text returns extracted string in `ocr.text`; `ENABLE_OCR=false` returns `null`

### FR-06: Identical API contract
No breaking changes to the existing API.
- All existing `ProcessResponse` fields unchanged: `file_id`, `faces`, `tags`, `objects`, `context`, `context_detail`
- `ContextDetail` fields unchanged: `english_caption`, `indonesian_phrase`, `indonesian_description`
- New `ocr` field added as `Optional[OcrResult]` (absent = `null` — non-breaking)
- `routes.py` logic for merging tags, objects, context from model outputs — unchanged
- **Acceptance:** Existing C# client receives identical JSON shape for all previously existing fields

### FR-07: CPU-only operation
All models must load and run inference on CPU without CUDA or MPS.
- Florence-2 and opus-mt must not require GPU at startup or inference time
- `DEVICE=cpu` must work correctly
- **Acceptance:** Worker starts and processes images successfully with `DEVICE=cpu` and no GPU present

## Non-Functional Requirements

### NFR-01: Model size budget
Total new model download ≤ 4 GB.
- Florence-2-base: ~450 MB
- opus-mt-en-id: ~300 MB
- Combined: ~750 MB (well within budget)

### NFR-02: Preserved test coverage
All existing test files must continue to pass after model replacement.
- `tests/test_api.py` — endpoint and security tests: must pass unchanged
- `tests/test_clustering.py` — face clustering tests: must pass unchanged
- `tests/test_models.py` — model tests: must be updated to test new models, not old ones

### NFR-03: Graceful degradation
Model failures must not crash the worker; existing error handling patterns apply.
- If Florence-2 inference fails, return empty tags/objects and fallback context
- OCR errors must not affect the rest of ProcessResponse

## Decisions (Locked)

| Decision | Choice | Rationale |
|---|---|---|
| Tagging model | Florence-2-base | Open-vocabulary, no predefined labels, strong on object detection and dense region captioning |
| Caption model | Florence-2-base (shared instance) | Reuse weights already loaded for tagging; one model load |
| Translation model | Helsinki-NLP/opus-mt-en-id | ~300 MB, CPU-fast, well-tested for Indonesian, available in `transformers` |
| Model count | One Florence-2 instance shared across tagging + captioning | Avoids loading 2× model weights |
| OCR exposure | Optional `ocr` field on `ProcessResponse` | Zero new endpoints; non-breaking |
| Name injection | Enable when cluster names available | Produces SPOK captions with real subjects |
| GPU requirement | CPU-only (slower is acceptable) | Deployment constraint |

## Deferred / Out of Scope

- Changing InsightFace or face clustering pipeline
- GPU-optimised quantisation or ONNX export of Florence-2
- New API endpoints (OCR as a standalone endpoint)
- Frontend or C# backend changes
- Streaming / incremental inference responses
- Fine-tuning any model on Indonesian government imagery
