---
phase: 02-caption-name-injection
plan: 01
subsystem: models/caption
tags: [florence-2, caption, translation, indonesian]

# Dependency graph
requires:
  - phase: 01-02
    provides: FlorenceModel with run_task() for captioning + TranslationModel
provides:
  - CaptionModel class with get_context_comprehensive() + detect_school_age()
  - Fluent Indonesian descriptions via Florence-2 + opus-mt translation
affects: [02-02]

# Tech tracking
tech-stack:
  added: [CaptionModel with Florence-2 + opus-mt]
  patterns: [Shared model instances, real translation not keyword-mapping]

key-files:
  created: [python-worker/models/caption_model.py]
  modified: [python-worker/api/routes.py]

key-decisions:
  - "Shared Florence instance: CaptionModel reuses FlorenceModel from Phase 1"
  - "Real translation: opus-mt instead of BLIP keyword-mapping"
  - "detect_school_age: Copied verbatim from BLIPModel for compatibility"

patterns-established:
  - "CaptionModel: Florence-2 for captioning, opus-mt for translation"

requirements-completed: [FR-03, NFR-02]

# Metrics
duration: ~10min
completed: 2026-03-14
---

# Phase 2 Plan 1: Caption Model Summary

**CaptionModel with Florence-2 + opus-mt translation replacing BLIP keyword-mapping**

## What Was Built

- `python-worker/models/caption_model.py`:
  - `get_context_comprehensive(image)` → `{english_caption, indonesian_phrase, indonesian_description, elements}`
  - `detect_school_age(caption, face_ages)` — copied from BLIPModel
  - Uses Florence-2 `<MORE_DETAILED_CAPTION>` for real caption generation
  - Uses opus-mt for genuine EN→ID translation (not keyword templates)
- Updated `python-worker/api/routes.py`:
  - Replace BLIPModel → CaptionModel
  - Health check shows `caption` key

## Verification

```
>>> caption.get_context_comprehensive(img)
{'english_caption': 'The image is a textured background...',
 'indonesian_phrase': 'Gambar ini adalah latar belakang yang bertekstur...',
 'indonesian_description': 'Gambar ini adalah latar belakang yang bertekstur...',
 'elements': {'people': {...}, 'activity': {...}, 'setting': {...}, 'objects': {...}, 'mood': 'neutral'}}
```

## Issues Fixed

- None — plan executed as written

## Commits

- `b9cc08c`: feat(02-caption-name-injection): implement CaptionModel with Florence-2 captioning
