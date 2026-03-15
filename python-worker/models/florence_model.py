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
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports


logger = logging.getLogger(__name__)


# Workaround: Remove flash_attn requirement for CPU-only environments
def _fixed_get_imports(filename: str):
    """Patch to remove flash_attn import requirement."""
    try:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        if "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports
    except Exception:
        return get_imports(filename)


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
        self.device_type = device_type
        self.threshold = threshold
        self.top_k = top_k
        self.language = language

        self._processor = None
        self._model = None
        self._is_loaded = False

        self._load_model()

    def _load_model(self):
        """Load Florence-2 processor and model with CPU workaround."""
        try:
            # Patch to remove flash_attn requirement
            with patch(
                "transformers.dynamic_module_utils.get_imports", _fixed_get_imports
            ):
                self._processor = AutoProcessor.from_pretrained(
                    self.MODEL_NAME, trust_remote_code=True
                )

                # Use SDP attention for CPU/MPS (instead of flash_attn)
                attn_impl = "flash_attention_2" if self.device_type == "cuda" else "sdpa"

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_NAME,
                    trust_remote_code=True,
                    attn_implementation=attn_impl,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                )
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
            # Build inputs based on task type
            if text_input:
                inputs = self._processor(
                    text=text_input, images=image, return_tensors="pt"
                )
            else:
                inputs = self._processor(text=task, images=image, return_tensors="pt")

            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
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

    def get_tags(
        self,
        image: Image.Image,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        language: Optional[str] = None,
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
            od_result = self.run_task(image, "<OD>")
            if od_result and "<OD>" in od_result:
                od_data = od_result["<OD>"]
                if "labels" in od_data:
                    all_labels.extend(od_data["labels"])

            # Run dense region caption for detailed descriptors (colors, textures, etc.)
            dense_result = self.run_task(image, "<DENSE_REGION_CAPTION>")
            if dense_result and "<DENSE_REGION_CAPTION>" in dense_result:
                dense_data = dense_result["<DENSE_REGION_CAPTION>"]
                if "labels" in dense_data:
                    all_labels.extend(dense_data["labels"])

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

            # Cap at top_k
            unique_labels = unique_labels[:top_k]

            # Translate to Indonesian if needed
            if language == "id" and self.translation_model:
                try:
                    translated = self.translation_model.translate_batch(unique_labels)
                    return translated
                except Exception as e:
                    logger.warning(f"Translation failed, returning English: {e}")
                    return unique_labels

            return unique_labels

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

        # Reject if contains weird special characters (allow punctuation like , . ')
        if re.search(r'[^a-zA-Z0-9\s\-\,\.\']', cleaned):
            return False

        # Reject if too short
        if len(cleaned) < 2:
            return False

        # Reject if too many words (max 8 words for a detailed tag phrase)
        word_count = len(cleaned.split())
        if word_count > 8:
            return False

        return True

    def get_objects(
        self, image: Image.Image, threshold: float = 0.20, top_k: int = 15
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
            result = self.run_task(image, "<OD>")

            if not result or "<OD>" not in result:
                return []

            od_data = result["<OD>"]
            labels = od_data.get("labels", [])

            # Deduplicate (case-insensitive) while preserving order
            seen = set()
            unique_objects = []
            for label in labels:
                label_lower = label.lower().strip()
                if label_lower and label_lower not in seen:
                    seen.add(label_lower)
                    unique_objects.append(label)

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
                        # Convert quad box (4 points) to bbox (x1, y1, x2, y2)
                        if len(quad) >= 4:
                            x_coords = [quad[0], quad[2], quad[4], quad[6]]
                            y_coords = [quad[1], quad[3], quad[5], quad[7]]
                            bbox = [
                                min(x_coords),
                                min(y_coords),
                                max(x_coords),
                                max(y_coords),
                            ]
                            regions.append({"text": label.strip(), "bbox": bbox})
                            full_text_parts.append(label.strip())

                return {
                    "text": " ".join(full_text_parts),
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
            return {"text": ""}
