# STATE.md — BADAK AI Worker: Vision Intelligence Upgrade

## Current Position

**Active phase:** Phase 3 — OCR + Cleanup
**Status:** Executing Phase 3

**Phase 1 plans:** (completed)
- `01-01-PLAN.md` — TranslationModel wrapper (opus-mt-en-id) — complete
- `01-02-PLAN.md` — FlorenceModel (get_tags + get_objects) + routes wiring — complete

**Phase 2 plans:** (completed)
- `02-01-PLAN.md` — CaptionModel: get_context_comprehensive() with Florence-2 + opus-mt — complete
- `02-02-PLAN.md` — Name injection via CAPTION_TO_PHRASE_GROUNDING + routes.py wiring — complete

**Phase 3 plans:** (in progress)
- `03-01-PLAN.md` — OCR method + schema update (OcrResult, ProcessResponse.ocr field) — complete
- `03-02-PLAN.md` — Remove old models, clean dependencies + dead config — complete
- `03-03-PLAN.md` — Update CLAUDE.md architecture documentation — pending

## Locked Decisions

| Decision | Value | Why locked |
|---|---|---|
| Vision model | `microsoft/Florence-2-base` | Open-vocabulary, handles tags + objects + captions + OCR in one model; ~450 MB fits budget |
| Translation model | `Helsinki-NLP/opus-mt-en-id` | ~300 MB, CPU-fast, in `transformers` (already a dependency), well-tested for Indonesian |
| Model loading | Single shared Florence-2 instance for tagging + captioning | Avoids loading 2× weights; passed to both `FlorenceModel` and `CaptionModel` |
| CPU-only | Required | Deployment constraint — GPU not guaranteed |
| OCR exposure | Optional `ocr` field on `ProcessResponse` | Non-breaking; no new endpoint needed |
| Name injection | Enabled when cluster names available | SPOK captions with real subjects; graceful fallback when unknown |
| API contract | Zero breaking changes | C# backend must not need updates |

## Method Signatures (must be preserved)

These are the exact signatures consumed by `api/routes.py`. New model classes must match them:

```python
# FlorenceModel
get_tags(image: Image.Image, threshold=None, top_k=None, language=None) -> List[str]
get_objects(image: Image.Image, threshold: float = 0.20, top_k: int = 15) -> List[str]
get_ocr(image: Image.Image, with_regions: bool = False) -> dict
  # with_regions=False: {"text": str}
  # with_regions=True:  {"text": str, "regions": [{"text": str, "bbox": [...8 coords]}]}

# CaptionModel
get_context_comprehensive(image: Image.Image, known_faces=None) -> dict
  # returns: {english_caption, indonesian_phrase, indonesian_description, elements}
  # elements: {people, activity, setting, objects, mood}
detect_school_age(caption: str, face_ages: List[int] = None) -> Optional[str]
```

## ProcessResponse Shape (must not change)

```python
ProcessResponse:
  file_id:        str
  faces:          List[FaceResult]
  tags:           List[str]           # Indonesian
  objects:        List[str]           # English
  context:        str                 # Indonesian short phrase
  context_detail: Optional[ContextDetail]
    .english_caption:        str
    .indonesian_phrase:      str
    .indonesian_description: str
  ocr:            Optional[OcrResult]  # NEW — null by default
```

## Config to Add

```python
# config.py additions
ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "false").lower() == "true"
```

## File Map After Migration

```
models/
  florence_model.py      # Florence-2 for tags, objects, OCR
  caption_model.py       # CaptionModel with context + name injection
  translation_model.py   # Shared opus-mt-en-id wrapper
  insightface_model.py   # UNCHANGED
```

## Pending Work

### Phase 1 (completed)
- [x] `models/translation_model.py` — TranslationModel wrapping opus-mt-en-id
- [x] `models/florence_model.py` — FlorenceModel with get_tags() + get_objects()
- [x] Update `main.py` — import FlorenceModel instead of OpenCLIPModel
- [x] Update `requirements.txt` — add sentencepiece, timm, einops
- [x] Update `tests/test_models.py` — test new tag/object methods

### Phase 2 (completed)
- [x] `models/caption_model.py` — CaptionModel with get_context_comprehensive() + detect_school_age()
- [x] Name injection via CAPTION_TO_PHRASE_GROUNDING
- [x] Update `routes.py` — pass known_faces into get_context_comprehensive()
- [x] Update `tests/test_models.py` — caption model tests

### Phase 3 (in progress)
- [x] `get_ocr()` on FlorenceModel
- [x] `OcrResult` schema + `ProcessResponse.ocr` field in schemas.py
- [x] `ENABLE_OCR` config + routes.py OCR call
- [x] Delete `openclip_model.py`, `blip_model.py`
- [x] Remove `open-clip-torch` from requirements.txt
- [ ] Update CLAUDE.md

## Known Issues / Pre-existing

- `CONTEXT_MODE` config setting was removed in 03-02 (dead config)
- `transformers==4.36.0` pinned — verify Florence-2 compatibility before upgrading if needed
