---
phase: 02-caption-name-injection
plan: 02
subsystem: models/caption
tags: [name-injection, spok, face-clustering]

# Dependency graph
requires:
  - phase: 02-01
    provides: CaptionModel with get_context_comprehensive()
provides:
  - Name injection into captions when cluster names available
  - SPOK-style captions with real subject names
affects: []

# Tech tracking
tech-stack:
  added: [Name injection, IoU matching]
  patterns: [CAPTION_TO_PHRASE_GROUNDING, positional fallback]

key-files:
  modified: [python-worker/models/caption_model.py, python-worker/api/routes.py]

key-decisions:
  - "Grounding first: Use Florence-2 CAPTION_TO_PHRASE_GROUNDING for phrase-to-bbox matching"
  - "Positional fallback: Replace 'a man' → name when grounding fails"
  - "IoU threshold: 0.3 for phrase-face matching"

patterns-established:
  - "known_faces collection: Get cluster names from ClusteringService"

requirements-completed: [FR-04]

# Metrics
duration: ~5min
completed: 2026-03-14
---

# Phase 2 Plan 2: Name Injection Summary

**SPOK captions with face cluster name injection**

## What Was Built

- Added name injection methods to CaptionModel:
  - `_inject_names()` — orchestrator for name injection
  - `_ground_caption_to_faces()` — uses Florence-2 `<CAPTION_TO_PHRASE_GROUNDING>`
  - `_apply_name_substitutions()` — replace phrases with names
  - `_apply_positional_substitution()` — fallback: "a man" → "Budi"
  - `_compute_iou()` — bounding box IoU calculation
- Wired `known_faces` collection in routes.py:
  - Get cluster names from `clustering_service.get_cluster_name()`
  - Pass to `caption_model.get_context_comprehensive(known_faces=...)`

## Verification

```
>>> cm._compute_iou([0,0,10,10], [0,0,10,10])
1.0
>>> cm._apply_positional_substitution('A man and a woman shaking hands', [{'name': 'Budi', ...}, {'name': 'Ani', ...}])
'Budi and Ani shaking hands'
```

## Issues Fixed

- None — plan executed as written

## Commits

- `564f829`: feat(02-caption-name-injection): add name injection to CaptionModel
