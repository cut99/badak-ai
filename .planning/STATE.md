# STATE.md — BADAK AI Worker: Vision Intelligence Upgrade

## Current Position

**Active phase:** Phase 1 — Translation Model + Florence Tagging
**Status:** Planning complete, ready to execute Phase 1

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
# FlorenceModel (replaces OpenCLIPModel)
get_tags(image: Image.Image, threshold=None, top_k=None, language=None) -> List[str]
get_objects(image: Image.Image, threshold: float = 0.20, top_k: int = 15) -> List[str]

# CaptionModel (replaces BLIPModel)
get_context_comprehensive(image: Image.Image, known_faces=None) -> dict
  # returns: {english_caption, indonesian_phrase, indonesian_description, elements}
  # elements: {people, activity, setting, objects, mood}
detect_school_age(caption: str, face_ages: List[int] = None) -> Optional[str]

# FlorenceModel (new method)
get_ocr(image: Image.Image, with_regions: bool = False) -> dict
  # with_regions=False: {"text": str}
  # with_regions=True:  {"text": str, "regions": [{"text": str, "bbox": [...8 coords]}]}
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

## Dependencies to Change

```
REMOVE: open-clip-torch==2.24.0
KEEP:   transformers==4.36.0  (already installed — covers both Florence-2 and opus-mt)
        torch==2.9.1
ADD:    sentencepiece          (MarianMT tokenizer requirement)
        timm                   (Florence-2 vision backbone requirement)
        einops                 (Florence-2 requirement)
```

## Config to Add

```python
# config.py additions
ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "false").lower() == "true"
```

## File Map After Migration

```
models/
  florence_model.py      # NEW — replaces openclip_model.py
  caption_model.py       # NEW — replaces blip_model.py
  translation_model.py   # NEW — shared opus-mt wrapper
  insightface_model.py   # UNCHANGED
  openclip_model.py      # DELETE in Phase 3
  blip_model.py          # DELETE in Phase 3
```

## Pending Work

### Phase 1 (next)
- [ ] `models/translation_model.py` — TranslationModel wrapping opus-mt-en-id
- [ ] `models/florence_model.py` — FlorenceModel with get_tags() + get_objects()
- [ ] Update `main.py` — import FlorenceModel instead of OpenCLIPModel
- [ ] Update `requirements.txt` — add sentencepiece, timm, einops
- [ ] Update `tests/test_models.py` — test new tag/object methods

### Phase 2
- [ ] `models/caption_model.py` — CaptionModel with get_context_comprehensive() + detect_school_age()
- [ ] Name injection via CAPTION_TO_PHRASE_GROUNDING
- [ ] Update `routes.py` — pass known_faces into get_context_comprehensive()
- [ ] Update `tests/test_models.py` — caption model tests

### Phase 3
- [ ] `get_ocr()` on FlorenceModel
- [ ] `OcrResult` schema + `ProcessResponse.ocr` field in schemas.py
- [ ] `ENABLE_OCR` config + routes.py OCR call
- [ ] Delete `openclip_model.py`, `blip_model.py`
- [ ] Remove `open-clip-torch` from requirements.txt
- [ ] Update CLAUDE.md

## Known Issues / Pre-existing

- `openclip_model.py` and `blip_model.py` have LSP type errors (pre-existing, not introduced by this milestone)
- `CONTEXT_MODE` config setting exists but is unused in routes.py (pre-existing dead config)
- `transformers==4.36.0` pinned — verify Florence-2 compatibility before upgrading if needed
