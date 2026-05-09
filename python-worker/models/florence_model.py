"""
Florence-2 Vision Model for open-vocabulary tagging, object detection, and OCR.
CPU-compatible version using SDP attention.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from unittest.mock import patch

import torch
from PIL import Image
import numpy as np


logger = logging.getLogger(__name__)


# Store original get_imports to avoid recursion when patched
from transformers.dynamic_module_utils import get_imports as original_get_imports

def _fixed_get_imports(filename: str):
    """Patch to remove flash_attn import requirement."""
    try:
        if not str(filename).endswith("modeling_florence2.py"):
            return original_get_imports(filename)
        imports = original_get_imports(filename)
        if "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports
    except Exception:
        return original_get_imports(filename)


class FlorenceModel:
    """
    Florence-2-base wrapper for open-vocabulary tagging and object detection.

    Key decisions:
    - Shared model instance: Used for both tagging and captioning to avoid loading 2x weights
    - CPU-only: Uses SDP attention instead of flash_attn
    - Public run_task(): Called by CaptionModel in Phase 2 for caption generation
    """

    MODEL_NAME = "microsoft/Florence-2-base"

    def __init__(
        self,
        translation_model,
        device_type: str = "cpu",
        threshold: float = 0.25,
        top_k: int = 10,
        language: str = "id",
    ):
        """
        Initialize Florence-2 model.

        Args:
            translation_model: TranslationModel instance for EN→ID translation
            device_type: "cpu" or "cuda" (cpu-only deployment)
            threshold: Kept for API compatibility (unused by Florence-2)
            top_k: Maximum number of tags/objects to return
            language: "id" for Indonesian, "en" for English
        """
        self.translation_model = translation_model
        requested_device = (device_type or "cpu").lower()
        if requested_device != "cpu":
            logger.warning(
                "FlorenceModel is configured for CPU-only deployment; forcing CPU "
                "instead of requested device '%s'",
                requested_device,
            )
        self.device_type = "cpu"
        self.device = torch.device("cpu")
        self.threshold = threshold
        self.top_k = top_k
        self.language = language

        self._processor = None
        self._model = None
        self._is_loaded = False

        # Apply Florence-2 transformers compatibility monkey patch
        self._apply_florence_patch()
        self._load_model()

    def _apply_florence_patch(self):
        """Monkey patch PretrainedConfig to fix Florence2LanguageConfig AttributeError in newer transformers versions."""
        try:
            from transformers import PretrainedConfig
            if not hasattr(PretrainedConfig, "_original_getattribute"):
                PretrainedConfig._original_getattribute = PretrainedConfig.__getattribute__
                
                def _patched_getattribute(self, key):
                    if key == "forced_bos_token_id":
                        try:
                            return type(self)._original_getattribute(self, key)
                        except AttributeError:
                            return None
                    return type(self)._original_getattribute(self, key)
                    
                PretrainedConfig.__getattribute__ = _patched_getattribute
        except Exception as e:
            logger.warning(f"Could not apply Florence compatibility patch: {e}")

    def _load_model(self):
        """Load Florence-2 processor and model with CPU workaround."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM

            # Patch to remove flash_attn requirement
            with patch(
                "transformers.dynamic_module_utils.get_imports", _fixed_get_imports
            ):
                attn_impl = "sdpa"
                
                try:
                    logger.info(f"Attempting to load {self.MODEL_NAME} from local cache...")
                    self._processor = AutoProcessor.from_pretrained(
                        self.MODEL_NAME, trust_remote_code=True, local_files_only=True
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.MODEL_NAME,
                        trust_remote_code=True,
                        attn_implementation=attn_impl,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        local_files_only=True,
                    )
                except Exception as local_err:
                    logger.info(f"Local cache miss ({local_err}), downloading {self.MODEL_NAME} from Hub. This may take a while...")
                    self._processor = AutoProcessor.from_pretrained(
                        self.MODEL_NAME, trust_remote_code=True
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.MODEL_NAME,
                        trust_remote_code=True,
                        attn_implementation=attn_impl,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                    )
                    
                self._model.to(self.device)
                self._model.eval()

            self._is_loaded = True
            logger.info(f"FlorenceModel loaded: {self.MODEL_NAME} (attn={attn_impl})")
        except Exception as e:
            logger.error(f"Failed to load FlorenceModel: {e}")
            raise

    def run_task(
        self, image: Image.Image, task: str, text_input: str = ""
    ) -> Dict[str, Any]:
        """
        PUBLIC METHOD - Run a Florence-2 task.

        This must be public because CaptionModel (Phase 2) calls it to run
        <MORE_DETAILED_CAPTION> and <CAPTION_TO_PHRASE_GROUNDING> tasks.

        Args:
            image: PIL Image
            task: Florence-2 task string (e.g., "<OD>", "<DENSE_REGION_CAPTION>", "<OCR>")
            text_input: Optional text input for tasks that need it

        Returns:
            Parsed result dict from Florence-2
        """
        try:
            prompt = self._build_prompt(task, text_input)
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self._max_new_tokens_for_task(task),
                    num_beams=1,
                    do_sample=False,
                )

            generated_text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            # Parse the generated text
            parsed = self._processor.post_process_generation(
                generated_text, task=task, image_size=(image.width, image.height)
            )

            return parsed if parsed else {}

        except Exception as e:
            logger.error(f"Florence run_task failed for task {task}: {e}")
            return {}

    @staticmethod
    def _build_prompt(task: str, text_input: str = "") -> str:
        """Build Florence prompt, preserving the task token for text-conditioned tasks."""
        cleaned_text = (text_input or "").strip()
        if cleaned_text:
            return f"{task}{cleaned_text}"
        return task

    @staticmethod
    def _max_new_tokens_for_task(task: str) -> int:
        """Keep CPU inference bounded per task."""
        limits = {
            "<OD>": 512,
            "<DENSE_REGION_CAPTION>": 512,
            "<MORE_DETAILED_CAPTION>": 256,
            "<CAPTION_TO_PHRASE_GROUNDING>": 512,
            "<OCR>": 1024,
            "<OCR_WITH_REGION>": 1024,
        }
        return limits.get(task, 512)

    def get_tags(
        self,
        image: Image.Image,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        language: Optional[str] = None,
        od_result: Optional[Dict[str, Any]] = None,
        dense_result: Optional[Dict[str, Any]] = None,
        context_labels: Optional[List[str]] = None,
        include_dense: bool = False,
    ) -> List[str]:
        """
        Get open-vocabulary tags for an image.

        Uses <OD> (object detection) labels only, filtered for valid tag-like
        words (1-3 words, no special chars, no sentence fragments).

        Args:
            image: PIL Image
            threshold: Override default threshold (unused but kept for API compat)
            top_k: Override default top_k
            language: Override default language ("id" or "en")

        Returns:
            List of Indonesian tags (or English if language="en")
        """
        try:
            top_k = top_k or self.top_k
            language = language or self.language

            all_labels = []

            # Run object detection for base tags
            if od_result is None:
                od_result = self.run_task(image, "<OD>")
            all_labels.extend(self._extract_labels(od_result, "<OD>"))

            # Dense region captions are noisy and expensive on CPU, so only use them
            # when explicitly requested.
            if include_dense:
                if dense_result is None:
                    dense_result = self.run_task(image, "<DENSE_REGION_CAPTION>")
                all_labels.extend(
                    self._extract_labels(dense_result, "<DENSE_REGION_CAPTION>")
                )

            # Clean prefixes like 'a', 'an', 'the' and Florence location tags
            cleaned_labels = []
            for label in all_labels:
                # Remove location tags like loc_858>
                label = re.sub(r'(?:<loc_\d+>|loc_\d+>)', '', label)
                
                cleaned = label.strip()
                lower_cleaned = cleaned.lower()
                for prefix in ["a ", "an ", "the "]:
                    if lower_cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):].strip()
                        lower_cleaned = cleaned.lower()
                        break
                cleaned_labels.append(cleaned)

            # Filter: only keep valid tag-like labels
            valid_labels = [
                label for label in cleaned_labels if self._is_valid_tag(label)
            ]

            # Deduplicate while preserving order (case-insensitive)
            seen = set()
            unique_labels = []
            for label in valid_labels:
                label_lower = label.lower().strip()
                if label_lower and label_lower not in seen:
                    seen.add(label_lower)
                    unique_labels.append(label.strip())

            # Translate to Indonesian if needed
            translated_labels = unique_labels
            if language == "id" and self.translation_model:
                try:
                    translated_labels = (
                        self.translation_model.translate_batch(unique_labels)
                        if unique_labels
                        else []
                    )
                except Exception as e:
                    logger.warning(f"Translation failed, returning English: {e}")

            # Context labels are deterministic Indonesian tags inferred from OCR,
            # caption, objects, and face count. Prioritize them over raw OD labels.
            combined_labels = []
            seen_combined = set()
            for label in (context_labels or []) + translated_labels:
                normalized = self._normalize_tag(label)
                if normalized and normalized not in seen_combined:
                    seen_combined.add(normalized)
                    combined_labels.append(label.strip())

            return combined_labels[:top_k]

        except Exception as e:
            logger.error(f"get_tags failed: {e}")
            return []

    @staticmethod
    def _is_valid_tag(label: str) -> bool:
        """
        Check if a label qualifies as a proper image tag.

        Valid tags are 1-8 words, no special characters, and at least 2 chars.
        Rejects sentence fragments and garbage output from Florence.
        """
        if not label or not label.strip():
            return False

        cleaned = label.strip()

        # Reject sentence fragments and punctuation-heavy captions.
        if re.search(r"[^a-zA-Z0-9\s\-']", cleaned):
            return False

        # Reject if too short
        if len(cleaned) < 2:
            return False

        lower_cleaned = cleaned.lower()
        if lower_cleaned.startswith(("there ", "this ", "that ", "image ", "photo ")):
            return False

        # Keep tags label-like, not dense caption sentences.
        word_count = len(cleaned.split())
        if word_count > 4:
            return False

        return True

    @staticmethod
    def _normalize_tag(label: str) -> str:
        return re.sub(r"\s+", " ", (label or "").strip().lower())

    @staticmethod
    def _extract_labels(result: Optional[Dict[str, Any]], task: str) -> List[str]:
        if not result or task not in result:
            return []
        task_data = result[task]
        if isinstance(task_data, dict):
            labels = task_data.get("labels", [])
            return labels if isinstance(labels, list) else []
        return []

    def get_objects(
        self,
        image: Image.Image,
        threshold: float = 0.20,
        top_k: int = 15,
        od_result: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get English object labels from Florence-2 <OD>.

        Args:
            image: PIL Image
            threshold: Detection threshold (unused but kept for API compat)
            top_k: Maximum number of objects to return

        Returns:
            List of English object labels
        """
        try:
            result = od_result if od_result is not None else self.run_task(image, "<OD>")

            if not result or "<OD>" not in result:
                return []

            labels = self._extract_labels(result, "<OD>")

            # Deduplicate (case-insensitive) while preserving order
            seen = set()
            unique_objects = []
            for label in labels:
                label_lower = label.lower().strip()
                if label_lower and label_lower not in seen:
                    seen.add(label_lower)
                    unique_objects.append(label.strip())

            return unique_objects[:top_k]

        except Exception as e:
            logger.error(f"get_objects failed: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "name": self.MODEL_NAME,
            "is_loaded": self._is_loaded,
            "device": self.device_type,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "language": self.language,
        }

    def get_ocr(self, image: Image.Image, with_regions: bool = False) -> Dict[str, Any]:
        """
        Get OCR text from an image using Florence-2.

        Args:
            image: PIL Image
            with_regions: If True, return detailed region information with bounding boxes

        Returns:
            Dict with "text" key, and optionally "regions" key if with_regions=True
            - with_regions=False: {"text": str}
            - with_regions=True: {"text": str, "regions": [{"text": str, "bbox": [float, ...]}]}
            On error: returns {"text": ""} — never raises, never returns None
        """
        try:
            if with_regions:
                # Use OCR with region detection
                result = self.run_task(image, "<OCR_WITH_REGION>")

                if not result or "<OCR_WITH_REGION>" not in result:
                    logger.warning("OCR_WITH_REGION returned no result")
                    return {"text": "", "regions": []}

                ocr_data = result["<OCR_WITH_REGION>"]

                # Florence-2 returns {"quad_boxes": [...], "labels": [...]}
                quad_boxes = ocr_data.get("quad_boxes", [])
                labels = ocr_data.get("labels", [])

                regions = []
                full_text_parts = []

                # Zip quad_boxes with labels
                for i, (quad, label) in enumerate(zip(quad_boxes, labels)):
                    if label and label.strip():
                        bbox = self._quad_to_bbox(quad)
                        if bbox:
                            regions.append({"text": label.strip(), "bbox": bbox})
                            full_text_parts.append(label.strip())

                return {
                    "text": "\n".join(full_text_parts),
                    "regions": regions,
                }

            else:
                # Simple OCR without regions
                result = self.run_task(image, "<OCR>")

                if not result or "<OCR>" not in result:
                    logger.warning("OCR returned no result")
                    return {"text": ""}

                ocr_data = result["<OCR>"]

                # Florence-2 returns {"<OCR>": "text"}
                if isinstance(ocr_data, str):
                    return {"text": ocr_data}
                elif isinstance(ocr_data, dict):
                    # Sometimes returns nested structure
                    text = ocr_data.get("text", "") or ocr_data.get(
                        "generated_text", ""
                    )
                    return {"text": text}
                else:
                    return {"text": str(ocr_data) if ocr_data else ""}

        except Exception as e:
            logger.error(f"get_ocr failed: {e}")
            # Never raise — return empty result on error
            return {"text": "", "regions": []} if with_regions else {"text": ""}

    @staticmethod
    def _quad_to_bbox(quad: List[float]) -> Optional[List[float]]:
        """Convert Florence OCR quad or bbox coordinates into [x1, y1, x2, y2]."""
        if not quad:
            return None
        if len(quad) >= 8:
            x_coords = [quad[0], quad[2], quad[4], quad[6]]
            y_coords = [quad[1], quad[3], quad[5], quad[7]]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        if len(quad) == 4:
            x1, y1, x2, y2 = quad
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        return None
