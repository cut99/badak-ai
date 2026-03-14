# ROADMAP.md — BADAK AI Worker: Vision Intelligence Upgrade

## Milestone: Vision Intelligence Upgrade

Replace static-dictionary OpenCLIP + BLIP with Florence-2 open-vocabulary vision model and fluent Indonesian translation. Add OCR and name-aware SPOK captions. No API breaking changes.

---

### Phase 1: Translation Model + Florence Tagging

**Goal:** Replace `openclip_model.py` with a Florence-2-based tagger that produces open-vocabulary Indonesian tags and English objects — no predefined label lists.

**Requirements:** FR-01, FR-02, FR-07, NFR-01

**Plans:** 2 plans

Plans:
- [ ] 01-01-PLAN.md — Translation model wrapper (opus-mt-en-id)
- [ ] 01-02-PLAN.md — FlorenceModel: get_tags() + get_objects() + main.py wiring

**Success criteria:**
- `get_tags(image)` returns Indonesian tags for arbitrary images without any hardcoded vocabulary
- `get_objects(image)` returns English object labels from Florence-2 `<OD>`
- Worker starts successfully with new model, old OpenCLIP removed from startup
- `pytest tests/test_models.py -m unit` passes

---

### Phase 2: Caption Model + Name Injection

**Goal:** Replace `blip_model.py` with a Florence-2-based caption model that produces fluent Indonesian descriptions and injects known face cluster names into captions (SPOK).

**Requirements:** FR-03, FR-04, NFR-02

**Plans:** 2 plans

Plans:
- [ ] 02-01-PLAN.md — CaptionModel: get_context_comprehensive() with Florence-2 + opus-mt
- [ ] 02-02-PLAN.md — Name injection via CAPTION_TO_PHRASE_GROUNDING + routes.py wiring

**Success criteria:**
- `get_context_comprehensive(image)` returns fluent Indonesian `indonesian_description` (not keyword-mapped)
- `get_context_comprehensive(image, known_faces=[...])` injects names when cluster names available
- `detect_school_age()` continues to work (copied verbatim)
- `pytest tests/test_models.py` passes with updated caption model tests

---

### Phase 3: OCR + Cleanup

**Goal:** Expose Florence-2 OCR as optional `ocr` field in ProcessResponse, remove old model files, update dependencies and docs.

**Requirements:** FR-05, FR-06, NFR-02, NFR-03

**Plans:** 2 plans

Plans:
- [ ] 03-01-PLAN.md — OCR method + schema update (OcrResult, ProcessResponse.ocr field)
- [ ] 03-02-PLAN.md — Remove old models, update requirements.txt, update config.py + CLAUDE.md

**Success criteria:**
- `ENABLE_OCR=true` → `ProcessResponse.ocr.text` populated for images containing text
- `ENABLE_OCR=false` (default) → `ProcessResponse.ocr` is `null`
- `open-clip-torch` removed from requirements.txt
- `openclip_model.py` and `blip_model.py` deleted
- `pytest tests/` passes (all suites)
- `/api/process-sync` end-to-end returns valid ProcessResponse with all original fields intact
