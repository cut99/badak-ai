---
phase: 01-florence-tagging
plan: 02
subsystem: models/florence
tags: [florence-2, vision, cpu, open-vocabulary]

# Dependency graph
requires:
  - phase: 01-01
    provides: TranslationModel wrapper for opus-mt-en-id
provides:
  - FlorenceModel class with get_tags(), get_objects(), run_task()
  - Open-vocabulary tagging via Florence-2 <OD> + <DENSE_REGION_CAPTION>
  - CPU-compatible inference using SDP attention
affects: [02-01, 02-02]

# Tech tracking
tech-stack:
  added: [transformers>=4.40.0, einops]
  patterns: [Florence-2 CPU inference, SDP attention, flash_attn workaround]

key-files:
  created: [python-worker/models/florence_model.py]
  modified: [python-worker/api/routes.py, python-worker/requirements.txt]

key-decisions:
  - "CPU-only: Use SDP attention instead of flash_attn"
  - "Shared model: FlorenceModel instance used for both tagging and captioning"
  - "Public run_task(): CaptionModel (Phase 2) calls it directly"

patterns-established:
  - "FlorenceModel: Open-vocabulary vision model with translation integration"

requirements-completed: [FR-01, FR-02, FR-07, NFR-01, NFR-02]

# Metrics
duration: ~15min (CPU workaround + wiring)
completed: 2026-03-14
---

# Phase 1 Plan 2: Florence Model + Routes Wiring Summary

**FlorenceModel open-vocabulary tagging replacing OpenCLIP with CPU SDP attention**

## What Was Built

- `python-worker/models/florence_model.py` — Florence-2 wrapper:
  - `get_tags(image) → List[str]` — Indonesian tags via Florence-2 + translation
  - `get_objects(image) → List[str]` — English object labels via Florence-2 <OD>
  - `run_task(image, task, text_input) → Dict` — PUBLIC method for Phase 2
  - CPU workaround: patch `get_imports` to remove flash_attn, use `attn_implementation="sdpa"`
- Updated `python-worker/api/routes.py`:
  - Replace OpenCLIPModel → FlorenceModel + TranslationModel
  - Health check shows `florence` and `translation` keys
- Updated `python-worker/requirements.txt`:
  - `transformers>=4.40.0` (was 4.36.0)
  - `einops>=0.6.1` already present

## Verification

```
>>> from models.florence_model import FlorenceModel
>>> fm = FlorenceModel(translation_model=trans, device_type='cpu')
>>> fm.get_model_info()
{'name': 'microsoft/Florence-2-base', 'is_loaded': True, 'device': 'cpu', ...}
>>> fm.get_objects(img)
['flower']
>>> fm.get_tags(img)
['bunga']  # Indonesian translation!
```

## Issues Fixed

**1. [Rule 3 - Blocking] Florence-2 requires flash_attn (CUDA-only)**
- **Found during:** Initial FlorenceModel loading test
- **Issue:** Florence-2 model file requires `flash_attn` which needs CUDA to install
- **Fix:** 
  - Patch `transformers.dynamic_module_utils.get_imports` to remove flash_attn requirement
  - Use `attn_implementation="sdpa"` (scaled dot product attention) instead of flash_attention_2
  - Load with `torch_dtype=torch.float32` for CPU compatibility
- **Files modified:** `python-worker/models/florence_model.py`

## Deviations from Plan

- None — plan executed as written with CPU workaround applied

## Commits

- `cd000a3`: feat(01-florence-tagging): implement FlorenceModel with CPU workaround
