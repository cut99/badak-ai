"""
Caption Model using Florence-2 for caption generation and opus-mt for translation.
"""

import logging
from typing import List, Dict, Any, Optional

from PIL import Image
import numpy as np


logger = logging.getLogger(__name__)


# Indonesian count words
INDONESIAN_COUNTS = {
    1: "satu orang",
    2: "dua orang",
    3: "tiga orang",
    5: "beberapa orang",
    10: "sekelompok orang",
    20: "banyak orang",
}


class CaptionModel:
    """
    Caption model using Florence-2 for generation + opus-mt for translation.

    Key decisions:
    - Shared Florence model: Reuses the same FlorenceModel instance
    - Translation: Uses shared TranslationModel for EN→ID
    - Name injection: 3-strategy approach (grounding → positional → group fallback)
    """

    # School age detection keywords
    AGE_KEYWORDS = {
        "elementary": "anak SD",
        "primary school": "anak SD",
        "junior high": "anak SMP",
        "middle school": "anak SMP",
        "senior high": "anak SMA",
        "high school": "anak SMA",
        "child": "anak SD",
        "children": "anak SD",
        "student": None,  # Generic, need more context
        "students": None,
    }

    # Uniform color keywords for Indonesian schools
    UNIFORM_COLORS = {
        "white red": "SD",
        "red white": "SD",
        "white blue": "SMP",
        "blue white": "SMP",
        "white gray": "SMA",
        "gray white": "SMA",
        "white grey": "SMA",
        "grey white": "SMA",
    }

    def __init__(self, florence_model, translation_model):
        """
        Initialize CaptionModel with shared Florence and Translation instances.

        Args:
            florence_model: FlorenceModel instance (shared, not loaded again)
            translation_model: TranslationModel instance (shared)
        """
        self.florence = florence_model
        self.translation = translation_model

    def get_context_comprehensive(
        self, image: Image.Image, known_faces: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive context from image using Florence-2 + translation.

        Args:
            image: PIL Image
            known_faces: Optional list of known faces for name injection (used in 02-02)

        Returns:
            Dict with:
            - english_caption: Original English caption from Florence
            - indonesian_phrase: Short Indonesian phrase (≤10 words)
            - indonesian_description: Full Indonesian description (1-3 sentences)
            - elements: Structured elements dict
        """
        # Generate English caption from Florence-2
        english_caption = self._generate_caption(image)

        # Inject known face names if available
        if known_faces and len(known_faces) > 0:
            english_caption = self._inject_names(image, english_caption, known_faces)

        # Translate to Indonesian phrase (short)
        indonesian_phrase = self._translate_to_short_phrase(english_caption)

        # Translate to Indonesian description (full)
        indonesian_description = self._translate_to_description(english_caption)

        return {
            "english_caption": english_caption,
            "indonesian_phrase": indonesian_phrase,
            "indonesian_description": indonesian_description,
            "elements": {},
        }

    # Common prefixes Florence generates that should be stripped
    _CAPTION_PREFIXES = [
        "The image shows ",
        "The image depicts ",
        "The image features ",
        "The image presents ",
        "The image displays ",
        "This image shows ",
        "This image depicts ",
        "This image features ",
        "In the image, ",
        "In this image, ",
        "The photo shows ",
        "The photograph shows ",
    ]

    def _generate_caption(self, image: Image.Image) -> str:
        """Generate English caption using Florence-2 <MORE_DETAILED_CAPTION>."""
        try:
            result = self.florence.run_task(image, "<MORE_DETAILED_CAPTION>")

            if result and "<MORE_DETAILED_CAPTION>" in result:
                caption = result["<MORE_DETAILED_CAPTION>"]
                if isinstance(caption, list) and len(caption) > 0:
                    raw = caption[0].get("text", "An image")
                elif isinstance(caption, str):
                    raw = caption
                else:
                    return "An image"

                # Strip "The image shows..." prefixes for proper SPOK output
                return self._clean_caption_prefix(raw)

            return "An image"
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "An image"

    def _clean_caption_prefix(self, caption: str) -> str:
        """
        Remove generic image-describing prefixes so the caption reads as a
        proper SPOK sentence (Subject-Predicate-Object-Keterangan).

        Before: "The image shows two women shaking hands in front of flags"
        After:  "Two women shaking hands in front of flags"
        """
        for prefix in self._CAPTION_PREFIXES:
            if caption.lower().startswith(prefix.lower()):
                # Preserve original casing of the remaining text
                cleaned = caption[len(prefix):]
                if cleaned:
                    # Capitalize first letter
                    return cleaned[0].upper() + cleaned[1:]
        return caption

    def _translate_to_short_phrase(self, english_caption: str) -> str:
        """Translate caption to short Indonesian phrase (≤10 words)."""
        try:
            # Take first sentence or first 15 words
            sentences = english_caption.split(".")
            first_part = sentences[0].strip() if sentences else english_caption
            words = first_part.split()[:15]
            short_text = " ".join(words)

            # Translate
            translated = self.translation.translate(short_text)

            # Truncate to 10 words
            translated_words = translated.split()[:10]
            return " ".join(translated_words)

        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return english_caption

    def _translate_to_description(self, english_caption: str) -> str:
        """Translate full caption to Indonesian description."""
        try:
            # Ensure sentences are translated independently
            # MarianMT (opus-mt-en-id) drops sentences when translating long paragraphs
            sentences = [s.strip() for s in english_caption.split(".") if s.strip()]
            to_translate = sentences

            if not to_translate:
                return self.translation.translate(english_caption)

            # Append period to each sentence to help context during translation
            to_translate_with_dots = [s + "." for s in to_translate]

            # Translate batch of sentences
            translated_sentences = self.translation.translate_batch(to_translate_with_dots)

            # Join back into a single paragraph
            return " ".join([s.strip() for s in translated_sentences]).strip()

        except Exception as e:
            logger.warning(f"Description translation failed: {e}")
            return english_caption



    # School age detection methods

    def _detect_school_age_from_caption(self, caption: str) -> Optional[str]:
        """Detect school age from caption keywords."""
        for keyword, age_tag in self.AGE_KEYWORDS.items():
            if keyword in caption and age_tag:
                return age_tag
        return None

    def _classify_age_group(self, age: int) -> Optional[str]:
        """Classify age to school level."""
        if 6 <= age <= 12:
            return "anak SD"
        elif 13 <= age <= 15:
            return "anak SMP"
        elif 16 <= age <= 18:
            return "anak SMA"
        return None

    def _detect_uniform_color(self, caption: str) -> Optional[str]:
        """Detect school uniform color from caption."""
        has_uniform = any(word in caption for word in ["uniform", "wearing", "shirt"])
        if not has_uniform:
            return None

        for color_combo, school_level in self.UNIFORM_COLORS.items():
            if color_combo in caption:
                return {"SD": "anak SD", "SMP": "anak SMP", "SMA": "anak SMA"}.get(
                    school_level
                )

        return None

    def detect_school_age(
        self, caption: str, face_ages: List[int] = None
    ) -> Optional[str]:
        """
        Detect school age using hybrid approach.

        Priority:
        1. Uniform color detection
        2. Age classification from InsightFace
        3. Caption keyword matching
        """
        caption_lower = caption.lower()

        # Priority 1: Uniform color
        uniform_result = self._detect_uniform_color(caption_lower)
        if uniform_result:
            return uniform_result

        # Priority 2: Age from InsightFace
        if face_ages:
            avg_age = sum(face_ages) / len(face_ages)
            age_result = self._classify_age_group(int(avg_age))
            if age_result:
                return age_result

        # Priority 3: Caption keywords
        return self._detect_school_age_from_caption(caption_lower)

    # Name injection methods for SPOK captions

    def _compute_iou(self, bbox_a: List[float], bbox_b: List[float]) -> float:
        """
        Compute Intersection over Union for two bounding boxes.

        Args:
            bbox_a: [x1, y1, x2, y2]
            bbox_b: [x1, y1, x2, y2]

        Returns:
            IoU value between 0 and 1
        """
        # Handle zero-area boxes
        if len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0.0

        x1_a, y1_a, x2_a, y2_a = bbox_a[:4]
        x1_b, y1_b, x2_b, y2_b = bbox_b[:4]

        # Handle zero-area boxes
        if x2_a <= x1_a or y2_a <= y1_a or x2_b <= x1_b or y2_b <= y1_b:
            return 0.0

        # Compute intersection
        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Compute union
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    # Group references that Florence commonly generates
    _GROUP_PATTERNS = [
        # (regex pattern, generic_singular_male, generic_singular_female, generic_neutral)
        # "two women" → names + "a woman" for unknowns
        (r'\btwo women\b', 'a woman', 'a woman', 'a person'),
        (r'\btwo men\b', 'a man', 'a man', 'a person'),
        (r'\btwo people\b', 'a person', 'a person', 'a person'),
        (r'\btwo persons\b', 'a person', 'a person', 'a person'),
        (r'\bthree women\b', 'a woman', 'a woman', 'a person'),
        (r'\bthree men\b', 'a man', 'a man', 'a person'),
        (r'\bthree people\b', 'a person', 'a person', 'a person'),
        (r'\bseveral women\b', 'a woman', 'a woman', 'a person'),
        (r'\bseveral men\b', 'a man', 'a man', 'a person'),
        (r'\bseveral people\b', 'a person', 'a person', 'a person'),
        (r'\ba group of women\b', 'a woman', 'a woman', 'a person'),
        (r'\ba group of men\b', 'a man', 'a man', 'a person'),
        (r'\ba group of people\b', 'a person', 'a person', 'a person'),
    ]

    def _inject_names(
        self, image, english_caption: str, known_faces: Optional[List[Dict]]
    ) -> str:
        """
        Inject known face names into caption.

        Tries three strategies in order:
        1. Grounding-based: Use Florence CAPTION_TO_PHRASE_GROUNDING + IoU
        2. Singular positional: Replace "a woman", "the man" etc.
        3. Group fallback: Replace "two women" → "Sri Mulyani I and a woman"

        Args:
            image: PIL Image
            english_caption: Original English caption
            known_faces: List of {"name": str, "bbox": [x1,y1,x2,y2]}

        Returns:
            Caption with names injected (or original if no names available)
        """
        if not known_faces or len(known_faces) == 0:
            return english_caption

        try:
            # Strategy 1: Grounding-based injection
            substitutions = self._ground_caption_to_faces(
                image, english_caption, known_faces
            )

            if substitutions:
                result = self._apply_name_substitutions(english_caption, substitutions)
                if result != english_caption:
                    return result

            # Strategy 2: Singular positional substitution
            result = self._apply_positional_substitution(english_caption, known_faces)
            if result != english_caption:
                return result

            # Strategy 3: Group reference fallback
            result = self._apply_group_substitution(english_caption, known_faces)
            if result != english_caption:
                return result

            return english_caption

        except Exception as e:
            logger.warning(f"Name injection failed: {e}")
            return english_caption

    def _ground_caption_to_faces(
        self, image, english_caption: str, known_faces: List[Dict]
    ) -> Dict[str, str]:
        """
        Use Florence-2 to ground caption phrases to face bounding boxes.

        Returns:
            Dict mapping phrase → name
        """
        try:
            result = self.florence.run_task(
                image, "<CAPTION_TO_PHRASE_GROUNDING>", english_caption
            )

            if not result or "<CAPTION_TO_PHRASE_GROUNDING>" not in result:
                return {}

            grounding = result["<CAPTION_TO_PHRASE_GROUNDING>"]
            bboxes = grounding.get("bboxes", [])
            labels = grounding.get("labels", [])

            substitutions = {}

            # For each grounded phrase, compute IoU with known faces
            for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                for face in known_faces:
                    iou = self._compute_iou(bbox, face.get("bbox", []))
                    if iou > 0.3:  # Threshold for matching
                        substitutions[label] = face.get("name", "")

            return substitutions

        except Exception as e:
            logger.warning(f"Grounding failed: {e}")
            return {}

    def _apply_name_substitutions(
        self, caption: str, substitutions: Dict[str, str]
    ) -> str:
        """Apply phrase → name substitutions to caption."""
        result = caption
        for phrase, name in substitutions.items():
            if phrase and name:
                result = result.replace(phrase, name)
        return result

    def _apply_positional_substitution(
        self, caption: str, known_faces: List[Dict]
    ) -> str:
        """Replace generic person references with names based on position."""
        import re

        PERSON_PATTERNS = [
            "a man",
            "the man",
            "an man",
            "a woman",
            "the woman",
            "an woman",
            "a person",
            "the person",
            "a boy",
            "the boy",
            "a girl",
            "the girl",
        ]

        result = caption
        name_idx = 0

        for pattern in PERSON_PATTERNS:
            if name_idx >= len(known_faces):
                break

            pattern_re = re.compile(re.escape(pattern), re.IGNORECASE)

            if pattern_re.search(result):
                name = known_faces[name_idx].get("name", "")
                if name:
                    result = pattern_re.sub(name, result, count=1)
                    name_idx += 1

        return result

    def _apply_group_substitution(
        self, caption: str, known_faces: List[Dict]
    ) -> str:
        """
        Replace group references like 'two women' with known names.

        Example:
            known_faces = [{"name": "Sri Mulyani I"}]
            caption = "Two women shaking hands in front of flags"
            result  = "Sri Mulyani I and a woman shaking hands in front of flags"
        """
        import re

        names = [f.get("name", "") for f in known_faces if f.get("name")]
        if not names:
            return caption

        result = caption

        for pattern, generic_m, generic_f, generic_n in self._GROUP_PATTERNS:
            match = re.search(pattern, result, re.IGNORECASE)
            if not match:
                continue

            # Determine which generic term to use based on the matched pattern
            matched_text = match.group(0).lower()
            if "women" in matched_text or "girl" in matched_text:
                generic = generic_f
            elif "men" in matched_text and "women" not in matched_text:
                generic = generic_m
            else:
                generic = generic_n

            # Build replacement: known names + generic for unknowns
            replacement = self._build_name_list(names, generic)

            # Replace the group reference, preserving sentence flow
            result = result[:match.start()] + replacement + result[match.end():]

            return result  # Only replace the first match

        return result

    @staticmethod
    def _build_name_list(names: List[str], generic_term: str) -> str:
        """
        Build a natural-language list of names + generic for unknowns.

        Examples:
            names=["Sri Mulyani I"], generic="a woman"
            → "Sri Mulyani I and a woman"

            names=["Sri Mulyani I", "Janet Yellen"], generic="a woman"
            → "Sri Mulyani I and Janet Yellen"

            names=["Sri Mulyani I"], generic="a person" (3 people detected)
            → "Sri Mulyani I and a person"
        """
        if len(names) == 0:
            return generic_term

        if len(names) == 1:
            return f"{names[0]} and {generic_term}"

        # Multiple known names: join with commas + "and"
        return ", ".join(names[:-1]) + " and " + names[-1]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model info."""
        return {
            "caption_model": "Florence-2 + opus-mt",
            "florence_loaded": self.florence is not None and self.florence._is_loaded,
            "translation_loaded": self.translation is not None
            and self.translation._is_loaded,
        }
