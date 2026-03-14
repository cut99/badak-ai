---
phase: 01-florence-tagging
plan: 01
subsystem: models/translation
tags: [translation, marianmt, indonesian]

# Dependency graph
requires: []
provides:
  - TranslationModel class with translate() and translate_batch() methods
  - English → Indonesian translation via Helsinki-NLP/opus-mt-en-id
affects: [01-02, 02-01, 02-02]

# Tech tracking
tech-stack:
  added: [transformers, torch, sentencepiece]
  patterns: [CPU-only MarianMT inference, batch-first translation]

key-files:
  created: [python-worker/models/translation_model.py]
  modified: [python-worker/requirements.txt]

key-decisions:
  - "CPU-only: MarianMT is small and fast on CPU"
  - "Batch-first design: One forward pass is faster than looping"
  - "Graceful degradation: Return original English on error"

patterns-established:
  - "TranslationModel: Shared wrapper consumed by FlorenceModel and CaptionModel"

requirements-completed: [FR-07, NFR-01]

# Metrics
duration: ~5min (fix + verification)
completed: 2026-03-14
---

# Phase 1 Plan 1: Translation Model Wrapper Summary

**TranslationModel wrapper for opus-mt-en-id providing English→Indonesian translation**

## What Was Built

- `python-worker/models/translation_model.py` — Thin wrapper around Helsinki-NLP/opus-mt-en-id
  - `translate(text: str) → str` — Single string translation
  - `translate_batch(texts: List[str]) → List[str]` — Batch translation
  - `get_model_info() → Dict` — Model metadata
- Added dependencies: `sentencepiece==0.1.99`, `timm>=0.9.0`, `einops>=0.6.1`

## Verification

```
>>> from models.translation_model import TranslationModel
>>> t = TranslationModel()
>>> t.translate('hello world')
'halo dunia'
>>> t.get_model_info()
{'name': 'Helsinki-NLP/opus-mt-en-id', 'is_loaded': True, ...}
```

## Issues Fixed

**1. [Rule 3 - Blocking] Fixed torch import order**
- **Found during:** Initial execution attempt
- **Issue:** `import torch` was at line 131, but `torch.no_grad()` was used at line 102 → NameError
- **Fix:** Moved `import torch` to top of file
- **Files modified:** `python-worker/models/translation_model.py`

## Deviations from Plan

- None — plan executed as written

## Commits

- `b7e8741`: feat(01-florence-tagging): implement TranslationModel wrapper for opus-mt-en-id
