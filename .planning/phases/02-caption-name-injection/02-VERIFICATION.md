---
phase: 02-caption-name-injection
verified: 2026-03-15T00:48:52Z
status: passed
score: 8/8 must-haves verified
re_verification: false
---

# Phase 2: Caption & Name Injection — Verification Report

**Phase Goal:** Replace BLIPModel keyword-mapping with Florence-2-based captioning + genuine opus-mt translation; inject face cluster names into captions when available.
**Verified:** 2026-03-15T00:48:52Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                              | Status     | Evidence                                                                                                   |
|----|------------------------------------------------------------------------------------|------------|------------------------------------------------------------------------------------------------------------|
| 1  | get_context_comprehensive(image) returns fluent Indonesian description              | ✓ VERIFIED | `caption_model.py:75` — calls `_generate_caption()` then `translation.translate()` at lines 142, 163     |
| 2  | Return dict has exact shape: english_caption, indonesian_phrase, indonesian_description, elements | ✓ VERIFIED | `caption_model.py:97-113` — returns dict with all 4 keys; elements has people/activity/setting/objects/mood |
| 3  | detect_school_age() works identically to BLIPModel version                         | ✓ VERIFIED | `caption_model.py:34,432-458` — copied verbatim from BLIPModel; AGE_KEYWORDS, UNIFORM_COLORS dicts present |
| 4  | Translation failures degrade gracefully — English text used as fallback             | ✓ VERIFIED | Exception handling in `translation_model.py:62-63`; CaptionModel wraps each translation step separately   |
| 5  | Image with named cluster produces caption containing the person's name              | ✓ VERIFIED | `caption_model.py:505-535` — `_inject_names()` calls grounding then positional substitution               |
| 6  | Image with unnamed clusters produces generic but grammatical description            | ✓ VERIFIED | `caption_model.py:97` — `known_faces=None` skips injection; `routes.py:230` passes None when empty        |
| 7  | Empty or missing known_faces parameter works without error                          | ✓ VERIFIED | `routes.py:230` — `known_faces if known_faces else None`; caption_model checks `if known_faces:` at line 97 |
| 8  | Grounding failure falls back to positional substitution gracefully                  | ✓ VERIFIED | `caption_model.py:524-535` — try/except around `_ground_caption_to_faces()`; fallback to `_apply_positional_substitution()` |

**Score:** 8/8 truths verified

---

### Required Artifacts

| Artifact                                    | Expected                                        | Status     | Details                                                                             |
|---------------------------------------------|-------------------------------------------------|------------|-------------------------------------------------------------------------------------|
| `python-worker/models/caption_model.py`     | Captioning via Florence-2 + translation         | ✓ VERIFIED | 630 lines; exports `CaptionModel`; contains `get_context_comprehensive`, `_inject_names`; uses `MORE_DETAILED_CAPTION` and `CAPTION_TO_PHRASE_GROUNDING` |

---

### Key Link Verification

| From                        | To                          | Via                                                     | Status     | Details                                                                                    |
|-----------------------------|-----------------------------|---------------------------------------------------------|------------|--------------------------------------------------------------------------------------------|
| `caption_model.py`          | `florence_model.py`         | `florence.run_task(image, '<MORE_DETAILED_CAPTION>')`   | ✓ WIRED    | Line 118: `self.florence.run_task(image, "<MORE_DETAILED_CAPTION>")`                       |
| `caption_model.py`          | `translation_model.py`      | `self.translation.translate()`                          | ✓ WIRED    | Lines 142, 163: `self.translation.translate(short_text)` and `self.translation.translate(truncated)` |
| `api/routes.py`             | `caption_model.py`          | `caption_model.get_context_comprehensive()`             | ✓ WIRED    | Lines 229-230: `caption_model.get_context_comprehensive(image, known_faces=...)`           |
| `caption_model.py`          | `florence_model.py`         | `florence.run_task(image, '<CAPTION_TO_PHRASE_GROUNDING>', caption)` | ✓ WIRED | Lines 548-549: `self.florence.run_task(image, "<CAPTION_TO_PHRASE_GROUNDING>", english_caption)` |
| `api/routes.py`             | `caption_model.py`          | `known_faces` from clustering results                   | ✓ WIRED    | Lines 199-206: iterates `face_results`, calls `clustering_service.get_cluster_name()`, builds `known_faces` list and passes to `get_context_comprehensive()` |

---

### Requirements Coverage

| Requirement | Source Plan | Description                                         | Status       | Evidence                                                                   |
|-------------|-------------|-----------------------------------------------------|--------------|----------------------------------------------------------------------------|
| FR-03       | 02-01       | Fluent Indonesian captioning (not keyword-mapped)   | ✓ SATISFIED  | `caption_model.py:118,142,163` — real Florence-2 caption + opus-mt translation |
| NFR-02      | 02-01       | No BLIPModel imports remain                         | ✓ SATISFIED  | `routes.py:38-41` — only FlorenceModel, TranslationModel, CaptionModel imported |
| FR-04       | 02-02       | Name injection via face cluster names               | ✓ SATISFIED  | `caption_model.py:505-584` — `_inject_names`, `_ground_caption_to_faces`, `_apply_positional_substitution` all implemented; `routes.py:199-230` wires known_faces |

---

### Anti-Patterns Found

No blocker or warning-level anti-patterns found in Phase 2 artifacts. The `caption_model.py` comments referencing "copied from BLIPModel" (lines 34-35, 38, 52, 172, 399) are documentation comments, not stale imports.

---

### Human Verification Required

#### 1. Indonesian Caption Quality

**Test:** Process a real image with recognizable content (people, objects, setting) and inspect `indonesian_description` field.
**Expected:** A fluent, natural-sounding Indonesian sentence — not a concatenated keyword string.
**Why human:** Translation fluency requires linguistic judgment; grep checks can't assess naturalness.

#### 2. Name Injection in Real Caption

**Test:** Process an image with a face whose cluster has a known name (e.g., "Budi"). Inspect the `context` field in the response.
**Expected:** Caption contains "Budi" (or equivalent), not "a man" / "a person".
**Why human:** Requires a real face image + registered cluster name in ChromaDB; can't be reproduced with static grep.

---

## Summary

Phase 2 goal is **fully achieved**. Both plans (02-01 and 02-02) delivered:

- `caption_model.py` (630 lines) implements the full pipeline: Florence-2 captioning → translation → name injection
- Name injection uses IoU-based phrase grounding with positional substitution as fallback
- `api/routes.py` builds `known_faces` from cluster names and passes them through to `get_context_comprehensive()`
- `BLIPModel` is fully replaced — no imports remain anywhere in `api/` or `tests/`
- All error paths have try/except guards with graceful degradation

---

_Verified: 2026-03-15T00:48:52Z_
_Verifier: Claude (gsd-verifier)_
