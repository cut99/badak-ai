"""
Caption Model using Florence-2 for caption generation and opus-mt for translation.
Replaces BLIPModel with genuine model-based captioning + translation.
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
    - Shared Florence model: Reuses the same FlorenceModel instance from Phase 1
    - Translation: Uses shared TranslationModel for EN→ID
    - detect_school_age: Copied verbatim from BLIPModel
    - Elements extraction: Keyword-based heuristics from BLIPModel
    """

    # School age detection keywords (copied from BLIPModel)
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

    # Uniform color keywords for Indonesian schools (copied from BLIPModel)
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

        # Extract structured elements (keyword-based)
        elements = self._extract_detailed_elements(english_caption.lower())

        return {
            "english_caption": english_caption,
            "indonesian_phrase": indonesian_phrase,
            "indonesian_description": indonesian_description,
            "elements": elements,
        }

    def _generate_caption(self, image: Image.Image) -> str:
        """Generate English caption using Florence-2 <MORE_DETAILED_CAPTION>."""
        try:
            result = self.florence.run_task(image, "<MORE_DETAILED_CAPTION>")

            if result and "<MORE_DETAILED_CAPTION>" in result:
                caption = result["<MORE_DETAILED_CAPTION>"]
                if isinstance(caption, list) and len(caption) > 0:
                    return caption[0].get("text", "An image")
                elif isinstance(caption, str):
                    return caption

            return "An image"
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "An image"

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
        """Translate full caption to Indonesian description (max 3 sentences)."""
        try:
            # Take max 3 sentences
            sentences = english_caption.split(".")
            truncated = ". ".join(sentences[:3]).strip()

            if not truncated:
                truncated = english_caption[:200]

            # Translate
            translated = self.translation.translate(truncated)
            return translated

        except Exception as e:
            logger.warning(f"Description translation failed: {e}")
            return english_caption

    def _extract_detailed_elements(self, caption: str) -> dict:
        """
        Extract structured elements from caption (keyword-based, copied from BLIPModel).
        """
        return {
            "people": self._extract_people_info(caption),
            "activity": self._extract_activity(caption),
            "setting": self._extract_setting(caption),
            "objects": self._extract_objects(caption),
            "mood": self._extract_mood(caption),
        }

    def _extract_people_info(self, caption: str) -> dict:
        """Extract people count."""
        people_keywords = {
            "one person": {"count": 1, "count_indonesian": "satu orang"},
            "person": {"count": 1, "count_indonesian": "satu orang"},
            "man": {"count": 1, "count_indonesian": "satu orang"},
            "woman": {"count": 1, "count_indonesian": "satu orang"},
            "two people": {"count": 2, "count_indonesian": "dua orang"},
            "two men": {"count": 2, "count_indonesian": "dua orang"},
            "two women": {"count": 2, "count_indonesian": "dua orang"},
            "three people": {"count": 3, "count_indonesian": "tiga orang"},
            "several people": {"count": 5, "count_indonesian": "beberapa orang"},
            "group": {"count": 10, "count_indonesian": "sekelompok orang"},
            "crowd": {"count": 20, "count_indonesian": "banyak orang"},
        }

        for keyword, info in people_keywords.items():
            if keyword in caption:
                return info
        return {"count": 1, "count_indonesian": "seseorang"}

    def _extract_activity(self, caption: str) -> dict:
        """Extract activity."""
        activities = {
            "shaking hands": {"english": "handshake", "indonesian": "bersalaman"},
            "handshake": {"english": "handshake", "indonesian": "bersalaman"},
            "sitting": {"english": "sitting", "indonesian": "duduk"},
            "seated": {"english": "sitting", "indonesian": "duduk"},
            "standing": {"english": "standing", "indonesian": "berdiri"},
            "talking": {"english": "talking", "indonesian": "berbicara"},
            "speaking": {"english": "speaking", "indonesian": "berbicara"},
            "smiling": {"english": "smiling", "indonesian": "tersenyum"},
            "presenting": {"english": "presenting", "indonesian": "presentasi"},
            "meeting": {"english": "meeting", "indonesian": "rapat"},
            "signing": {"english": "signing", "indonesian": "menandatangani"},
            "walking": {"english": "walking", "indonesian": "berjalan"},
            "posing": {"english": "posing", "indonesian": "berpose"},
        }

        for keyword, activity in activities.items():
            if keyword in caption:
                return activity
        return {"english": "gathering", "indonesian": "berkumpul"}

    def _extract_setting(self, caption: str) -> dict:
        """Extract setting/location."""
        settings = {
            "office": {"english": "office", "indonesian": "ruang kantor"},
            "meeting room": {"english": "meeting room", "indonesian": "ruang rapat"},
            "conference": {
                "english": "conference hall",
                "indonesian": "ruang konferensi",
            },
            "auditorium": {"english": "auditorium", "indonesian": "auditorium"},
            "outdoor": {"english": "outdoor", "indonesian": "luar ruangan"},
            "park": {"english": "park", "indonesian": "taman"},
            "garden": {"english": "garden", "indonesian": "taman"},
            "building": {"english": "building", "indonesian": "gedung"},
            "room": {"english": "room", "indonesian": "ruangan"},
            "hall": {"english": "hall", "indonesian": "aula"},
            "stage": {"english": "stage", "indonesian": "panggung"},
        }

        for keyword, setting in settings.items():
            if keyword in caption:
                return setting

        if "outdoor" in caption or "outside" in caption:
            return {"english": "outdoor", "indonesian": "luar ruangan"}
        return {"english": "indoor", "indonesian": "dalam ruangan"}

    def _extract_objects(self, caption: str) -> dict:
        """Extract objects from caption."""
        objects_map = {
            "desk": "meja",
            "table": "meja",
            "chair": "kursi",
            "microphone": "mikrofon",
            "flag": "bendera",
            "document": "dokumen",
            "paper": "kertas",
            "podium": "podium",
            "screen": "layar",
            "banner": "spanduk",
            "laptop": "laptop",
            "computer": "komputer",
            "phone": "telepon",
            "book": "buku",
            "pen": "pena",
            "bag": "tas",
            "camera": "kamera",
            "glasses": "kacamata",
            "door": "pintu",
            "window": "jendela",
            "wall": "dinding",
            "floor": "lantai",
            "light": "lampu",
            "suit": "jas",
            "tie": "dasi",
            "shirt": "kemeja",
            "shoe": "sepatu",
            "hat": "topi",
            "mask": "masker",
            "bottle": "botol",
            "tree": "pohon",
            "flower": "bunga",
            "car": "mobil",
        }

        stopwords = {
            "a",
            "an",
            "the",
            "in",
            "on",
            "at",
            "of",
            "to",
            "with",
            "by",
            "for",
            "from",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "being",
            "been",
            "this",
            "that",
            "these",
            "those",
            "it",
            "he",
            "she",
            "they",
            "sitting",
            "standing",
            "walking",
            "looking",
            "wearing",
            "holding",
            "carrying",
            "talking",
            "smiling",
            "laughing",
            "running",
            "jumping",
            "playing",
            "posing",
            "photo",
            "image",
            "picture",
            "view",
            "scene",
            "background",
            "foreground",
            "left",
            "right",
            "center",
            "top",
            "bottom",
            "side",
            "front",
            "back",
            "man",
            "woman",
            "person",
            "people",
            "boy",
            "girl",
            "men",
            "women",
            "child",
            "children",
            "group",
            "crowd",
            "white",
            "black",
            "red",
            "blue",
            "green",
            "yellow",
            "orange",
            "grey",
            "gray",
        }

        clean_caption = "".join(
            [c if c.isalnum() or c.isspace() else " " for c in caption.lower()]
        )
        words = clean_caption.split()

        found_objects_en = []
        for word in words:
            if len(word) > 2 and word not in stopwords and word not in found_objects_en:
                found_objects_en.append(word)

        found_objects_id = []
        for obj_en, obj_id in objects_map.items():
            if obj_en in caption.lower() and obj_id not in found_objects_id:
                found_objects_id.append(obj_id)

        return {"english": found_objects_en, "indonesian": found_objects_id}

    def _extract_mood(self, caption: str) -> str:
        """Extract mood."""
        if any(word in caption for word in ["formal", "suit", "official", "ceremony"]):
            return "formal"
        elif any(word in caption for word in ["casual", "relaxed", "informal"]):
            return "informal"
        return "neutral"

    # School age detection methods (copied verbatim from BLIPModel)

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

    def _inject_names(
        self, image, english_caption: str, known_faces: Optional[List[Dict]]
    ) -> str:
        """
        Inject known face names into caption.

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
            # Try grounding-based injection first
            substitutions = self._ground_caption_to_faces(
                image, english_caption, known_faces
            )

            if substitutions:
                return self._apply_name_substitutions(english_caption, substitutions)

            # Fallback to positional substitution
            return self._apply_positional_substitution(english_caption, known_faces)

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

            # Case-insensitive replace first occurrence
            import re

            pattern_re = re.compile(re.escape(pattern), re.IGNORECASE)

            if pattern_re.search(result):
                name = known_faces[name_idx].get("name", "")
                if name:
                    result = pattern_re.sub(name, result, count=1)
                    name_idx += 1

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get model info."""
        return {
            "caption_model": "Florence-2 + opus-mt",
            "florence_loaded": self.florence is not None and self.florence._is_loaded,
            "translation_loaded": self.translation is not None
            and self.translation._is_loaded,
        }
