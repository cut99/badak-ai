"""
BLIP model for image captioning with Indonesian context mapping.
Uses blip-image-captioning-base to generate captions and maps them to Indonesian phrases.
"""

import logging
import torch
from typing import Dict, Any, List, Optional
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)


class BLIPModel:
    """Image captioning using BLIP with Indonesian context phrase mapping."""

    # Indonesian context phrases for government/official photos
    CONTEXT_PHRASES = [
        "sedang bersalaman",
        "sedang duduk",
        "sedang berdiri",
        "sedang berbicara",
        "sedang tersenyum",
        "sedang berfoto",
        "sedang rapat",
        "sedang presentasi",
        "sedang makan",
        "sedang berjalan",
        "menerima penghargaan",
        "menandatangani dokumen",
        "upacara bendera",
        "foto bersama",
        "wawancara",
        "konferensi pers"
    ]

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
        "students": None
    }

    # Uniform color keywords for Indonesian schools
    UNIFORM_COLORS = {
        "white red": "SD",      # Putih merah
        "red white": "SD",
        "white blue": "SMP",     # Putih biru
        "blue white": "SMP",
        "white gray": "SMA",     # Putih abu
        "gray white": "SMA",
        "white grey": "SMA",
        "grey white": "SMA"
    }

    # Mapping from English keywords to Indonesian phrases
    CONTEXT_MAPPING = {
        "shaking hands": "sedang bersalaman",
        "handshake": "sedang bersalaman",
        "shake hands": "sedang bersalaman",
        "sitting": "sedang duduk",
        "seated": "sedang duduk",
        "sit": "sedang duduk",
        "standing": "sedang berdiri",
        "stand": "sedang berdiri",
        "talking": "sedang berbicara",
        "speaking": "sedang berbicara",
        "talk": "sedang berbicara",
        "speak": "sedang berbicara",
        "smiling": "sedang tersenyum",
        "smile": "sedang tersenyum",
        "posing": "sedang berfoto",
        "pose": "sedang berfoto",
        "meeting": "sedang rapat",
        "presentation": "sedang presentasi",
        "presenting": "sedang presentasi",
        "eating": "sedang makan",
        "dining": "sedang makan",
        "walking": "sedang berjalan",
        "walk": "sedang berjalan",
        "award": "menerima penghargaan",
        "receiving award": "menerima penghargaan",
        "signing": "menandatangani dokumen",
        "sign": "menandatangani dokumen",
        "flag ceremony": "upacara bendera",
        "ceremony": "upacara bendera",
        "group photo": "foto bersama",
        "group picture": "foto bersama",
        "together": "foto bersama",
        "interview": "wawancara",
        "press": "konferensi pers",
        "press conference": "konferensi pers"
    }

    def __init__(self, device_type: str = "cpu"):
        """
        Initialize BLIP model.

        Args:
            device_type: Device type ("cuda", "mps", or "cpu")
        """
        self.device_type = device_type
        self.device = torch.device(device_type)
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the BLIP image captioning model."""
        try:
            logger.info(f"Loading BLIP image-captioning-base model on device: {self.device_type}")

            # Load processor and model
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)

            # Set model to evaluation mode
            self.model.eval()

            logger.info("BLIP model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise

    def generate_caption(self, image: Image.Image, max_length: int = 50) -> str:
        """
        Generate an English caption for an image.

        Args:
            image: PIL Image object
            max_length: Maximum caption length in tokens

        Returns:
            Generated caption string

        Example:
            >>> model = BLIPModel()
            >>> img = Image.open("photo.jpg")
            >>> caption = model.generate_caption(img)
            >>> print(caption)
            'two men shaking hands in an office'
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")

        try:
            with torch.no_grad():
                # Preprocess image
                inputs = self.processor(image, return_tensors="pt").to(self.device)

                # Generate caption
                output = self.model.generate(**inputs, max_length=max_length)

                # Decode caption
                caption = self.processor.decode(output[0], skip_special_tokens=True)

            logger.debug(f"Generated caption: '{caption}'")
            return caption

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            raise

    def get_context(self, image: Image.Image) -> str:
        """
        Get Indonesian context phrase for an image.

        Generates English caption using BLIP, then maps keywords to
        the closest Indonesian context phrase.

        Args:
            image: PIL Image object

        Returns:
            Indonesian context phrase string

        Example:
            >>> model = BLIPModel()
            >>> img = Image.open("photo.jpg")
            >>> context = model.get_context(img)
            >>> print(context)
            'sedang bersalaman'
        """
        # Generate English caption
        caption = self.generate_caption(image)
        caption_lower = caption.lower()

        # Try to match keywords to Indonesian phrases
        matched_phrase = self._match_context(caption_lower)

        logger.debug(f"Mapped caption '{caption}' to context '{matched_phrase}'")
        return matched_phrase

    def _match_context(self, caption: str) -> str:
        """
        Match English caption to Indonesian context phrase.

        Args:
            caption: English caption (lowercase)

        Returns:
            Indonesian context phrase
        """
        # Check for exact keyword matches
        for keyword, phrase in self.CONTEXT_MAPPING.items():
            if keyword in caption:
                return phrase

        # Fallback: check for partial matches with common words
        if "hand" in caption:
            return "sedang bersalaman"
        elif "sit" in caption or "chair" in caption:
            return "sedang duduk"
        elif "stand" in caption:
            return "sedang berdiri"
        elif "talk" in caption or "speak" in caption or "conversation" in caption:
            return "sedang berbicara"
        elif "smil" in caption:
            return "sedang tersenyum"
        elif "photo" in caption or "picture" in caption or "camera" in caption:
            return "sedang berfoto"
        elif "group" in caption or "people" in caption or "together" in caption:
            return "foto bersama"

        # Default fallback
        return "foto bersama"

    def get_context_with_caption(self, image: Image.Image) -> Dict[str, str]:
        """
        Get both Indonesian context and English caption.

        Args:
            image: PIL Image object

        Returns:
            Dictionary with 'context' (Indonesian) and 'caption' (English) keys

        Example:
            >>> model = BLIPModel()
            >>> img = Image.open("photo.jpg")
            >>> result = model.get_context_with_caption(img)
            >>> print(result)
            {'context': 'sedang bersalaman', 'caption': 'two men shaking hands'}
        """
        caption = self.generate_caption(image)
        context = self._match_context(caption.lower())

        return {
            "caption": caption,
            "context": context
        }

    def get_context_comprehensive(self, image: Image.Image) -> dict:
        """
        Generate comprehensive context information with detailed elements.

        Returns detailed structure with multiple levels of context including
        English caption, Indonesian phrases and description, and structured elements.

        Args:
            image: PIL Image object

        Returns:
            Dictionary with comprehensive context information

        Example:
            >>> model = BLIPModel()
            >>> img = Image.open("photo.jpg")
            >>> result = model.get_context_comprehensive(img)
            >>> print(result)
            {
                'english_caption': 'two men shaking hands in office',
                'indonesian_phrase': 'sedang bersalaman',
                'indonesian_description': 'Dua orang sedang bersalaman di ruang kantor',
                'elements': {...}
            }
        """
        # Generate longer caption
        caption = self.generate_caption(image, max_length=100)
        caption_lower = caption.lower()

        # Map to Indonesian phrase
        indonesian_phrase = self._match_context(caption_lower)

        # Extract detailed elements
        context_elements = self._extract_detailed_elements(caption_lower)

        # Generate Indonesian description
        indonesian_description = self._generate_indonesian_description(
            caption, context_elements, indonesian_phrase
        )

        return {
            "english_caption": caption,
            "indonesian_phrase": indonesian_phrase,
            "indonesian_description": indonesian_description,
            "elements": context_elements
        }

    def _extract_detailed_elements(self, caption: str) -> dict:
        """
        Extract structured information from caption.

        Args:
            caption: English caption (lowercase)

        Returns:
            Dictionary with extracted elements (people, activity, setting, objects, mood)
        """
        elements = {
            "people": self._extract_people_info(caption),
            "activity": self._extract_activity(caption),
            "setting": self._extract_setting(caption),
            "objects": self._extract_objects(caption),
            "mood": self._extract_mood(caption)
        }
        return elements

    def _extract_people_info(self, caption: str) -> dict:
        """Extract people count and description."""
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
            "crowd": {"count": 20, "count_indonesian": "banyak orang"}
        }

        for keyword, info in people_keywords.items():
            if keyword in caption:
                return info

        # Default
        return {"count": 1, "count_indonesian": "seseorang"}

    def _extract_activity(self, caption: str) -> dict:
        """Extract activity information."""
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
            "posing": {"english": "posing", "indonesian": "berpose"}
        }

        for keyword, activity in activities.items():
            if keyword in caption:
                return activity

        return {"english": "gathering", "indonesian": "berkumpul"}

    def _extract_setting(self, caption: str) -> dict:
        """Extract setting/location information."""
        settings = {
            "office": {"english": "office", "indonesian": "ruang kantor"},
            "meeting room": {"english": "meeting room", "indonesian": "ruang rapat"},
            "conference": {"english": "conference hall", "indonesian": "ruang konferensi"},
            "auditorium": {"english": "auditorium", "indonesian": "auditorium"},
            "outdoor": {"english": "outdoor", "indonesian": "luar ruangan"},
            "park": {"english": "park", "indonesian": "taman"},
            "garden": {"english": "garden", "indonesian": "taman"},
            "building": {"english": "building", "indonesian": "gedung"},
            "room": {"english": "room", "indonesian": "ruangan"},
            "hall": {"english": "hall", "indonesian": "aula"},
            "stage": {"english": "stage", "indonesian": "panggung"}
        }

        for keyword, setting in settings.items():
            if keyword in caption:
                return setting

        # Check if indoor/outdoor
        if "outdoor" in caption or "outside" in caption:
            return {"english": "outdoor", "indonesian": "luar ruangan"}

        return {"english": "indoor", "indonesian": "dalam ruangan"}

    def _extract_objects(self, caption: str) -> dict:
        """
        Extract visible objects from caption.
        Returns all significant nouns/words from caption for English,
        ignoring standard stopwords and common verbs.
        For Indonesian, maps known objects from the predefined map.
        """
        # Extended map for translation purposes (used for description generation)
        objects_map = {
            "desk": "meja", "table": "meja", "chair": "kursi", "microphone": "mikrofon",
            "flag": "bendera", "document": "dokumen", "paper": "kertas", "podium": "podium",
            "stage": "panggung", "screen": "layar", "banner": "spanduk", "laptop": "laptop",
            "computer": "komputer", "phone": "telepon", "book": "buku", "pen": "pena",
            "bag": "tas", "camera": "kamera", "glasses": "kacamata", "door": "pintu",
            "window": "jendela", "wall": "dinding", "floor": "lantai", "ceiling": "langit-langit",
            "light": "lampu", "lamp": "lampu", "suit": "jas", "tie": "dasi", "shirt": "kemeja",
            "shoe": "sepatu", "hat": "topi", "mask": "masker", "bottle": "botol", "cup": "cangkir",
            "glass": "gelas", "tree": "pohon", "flower": "bunga", "car": "mobil",
            "motorcycle": "motor", "building": "gedung", "road": "jalan", "shoes": "sepatu",
            "hats": "topi", "books": "buku", "bags": "tas"
        }

        # 1. Extract raw candidates from caption
        # Simple stopword list to filter out non-objects (determiners, pronouns, verbs, colors, etc.)
        stopwords = {
            "a", "an", "the", "in", "on", "at", "of", "to", "with", "by", "for", "from",
            "and", "or", "but", "is", "are", "was", "were", "be", "being", "been",
            "this", "that", "these", "those", "it", "he", "she", "they", "my", "your",
            "sitting", "standing", "walking", "looking", "wearing", "holding", "carrying",
            "talking", "smiling", "laughing", "running", "jumping", "playing", "posing",
            "photo", "image", "picture", "view", "scene", "background", "foreground",
            "left", "right", "center", "top", "bottom", "side", "front", "back",
            "man", "woman", "person", "people", "boy", "girl", "men", "women", "child", "children",
            "group", "crowd", "white", "black", "red", "blue", "green", "yellow", "orange", "grey", "gray"
        }

        # Clean caption: lowercase and remove punctuation
        clean_caption = "".join([c if c.isalnum() or c.isspace() else " " for c in caption.lower()])
        words = clean_caption.split()
        
        found_objects_en = []
        
        # 2. Build English object list from all valid words
        for word in words:
            # Filter out short words and stopwords
            if len(word) > 2 and word not in stopwords:
                # Avoid duplicates
                if word not in found_objects_en:
                    found_objects_en.append(word)

        # 3. Build Indonesian list from map (for description)
        found_objects_id = []
        for obj_en, obj_id in objects_map.items():
            # Check if map key is in the caption
            if obj_en in caption.lower():
                 if obj_id not in found_objects_id:
                    found_objects_id.append(obj_id)

        return {
            "english": found_objects_en,
            "indonesian": found_objects_id
        }

    def _extract_mood(self, caption: str) -> str:
        """Extract mood/formality."""
        if any(word in caption for word in ["formal", "suit", "official", "ceremony"]):
            return "formal"
        elif any(word in caption for word in ["casual", "relaxed", "informal"]):
            return "informal"
        else:
            return "neutral"

    def _generate_indonesian_description(self, caption: str, elements: dict, phrase: str) -> str:
        """
        Generate detailed Indonesian description from elements.

        Uses template-based approach for reliability.

        Args:
            caption: Original English caption
            elements: Extracted structured elements
            phrase: Short Indonesian phrase

        Returns:
            Detailed Indonesian description (1-2 sentences)
        """
        parts = []

        # Add people count
        if elements.get("people"):
            parts.append(elements["people"]["count_indonesian"])

        # Add activity
        if elements.get("activity"):
            parts.append(f"sedang {elements['activity']['indonesian']}")

        # Add setting
        if elements.get("setting"):
            parts.append(f"di {elements['setting']['indonesian']}")

        # Add mood
        if elements.get("mood") == "formal":
            parts.append("formal")

        # Add objects if any
        objects_id = elements.get("objects", {}).get("indonesian", [])
        if objects_id:
            # Only add first 2 objects to keep description concise
            parts.append(f"dengan {', '.join(objects_id[:2])}")

        # Build description
        if parts:
            description = " ".join(parts).capitalize()
        else:
            description = phrase.capitalize()

        return description

    def _detect_school_age_from_caption(self, caption: str) -> Optional[str]:
        """
        Detect school age from BLIP caption keywords.

        Args:
            caption: English caption (lowercase)

        Returns:
            "anak SD", "anak SMP", "anak SMA", or None
        """
        for keyword, age_tag in self.AGE_KEYWORDS.items():
            if keyword in caption and age_tag:
                return age_tag
        return None

    def _classify_age_group(self, age: int) -> Optional[str]:
        """
        Classify age to school level (SD/SMP/SMA).

        Args:
            age: Estimated age from InsightFace

        Returns:
            "anak SD", "anak SMP", "anak SMA", or None
        """
        if 6 <= age <= 12:
            return "anak SD"
        elif 13 <= age <= 15:
            return "anak SMP"
        elif 16 <= age <= 18:
            return "anak SMA"
        return None

    def _detect_uniform_color(self, caption: str) -> Optional[str]:
        """
        Detect Indonesian school uniform color from caption.

        Args:
            caption: English caption (lowercase)

        Returns:
            "anak SD", "anak SMP", "anak SMA", or None
        """
        # Check for uniform keywords first
        has_uniform = any(word in caption for word in ["uniform", "wearing", "shirt"])

        if not has_uniform:
            return None

        # Check for color combinations
        for color_combo, school_level in self.UNIFORM_COLORS.items():
            if color_combo in caption:
                if school_level == "SD":
                    return "anak SD"
                elif school_level == "SMP":
                    return "anak SMP"
                elif school_level == "SMA":
                    return "anak SMA"

        # Check for individual colors with uniform mention
        if ("red" in caption or "merah" in caption) and ("white" in caption or "putih" in caption):
            return "anak SD"
        elif ("blue" in caption or "biru" in caption) and ("white" in caption or "putih" in caption):
            return "anak SMP"
        elif ("gray" in caption or "grey" in caption or "abu" in caption) and ("white" in caption or "putih" in caption):
            return "anak SMA"

        return None

    def detect_school_age(self, caption: str, face_ages: List[int] = None) -> Optional[str]:
        """
        Detect school age using hybrid approach (caption + age + uniform).

        Priority:
        1. Uniform color detection (most reliable for Indonesian schools)
        2. Age classification from InsightFace
        3. Caption keyword matching

        Args:
            caption: English caption from BLIP
            face_ages: List of ages from InsightFace face detection

        Returns:
            "anak SD", "anak SMP", "anak SMA", or None
        """
        caption_lower = caption.lower()

        # Priority 1: Uniform color detection
        uniform_result = self._detect_uniform_color(caption_lower)
        if uniform_result:
            logger.debug(f"School age detected from uniform: {uniform_result}")
            return uniform_result

        # Priority 2: Age classification from InsightFace
        if face_ages:
            # Take average age of detected faces
            avg_age = sum(face_ages) / len(face_ages)
            age_result = self._classify_age_group(int(avg_age))
            if age_result:
                logger.debug(f"School age detected from face age ({avg_age}): {age_result}")
                return age_result

        # Priority 3: Caption keyword matching
        caption_result = self._detect_school_age_from_caption(caption_lower)
        if caption_result:
            logger.debug(f"School age detected from caption: {caption_result}")
            return caption_result

        return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": "blip-image-captioning-base",
            "framework": "Transformers (Salesforce BLIP)",
            "device_type": self.device_type,
            "num_context_phrases": len(self.CONTEXT_PHRASES),
            "context_phrases": self.CONTEXT_PHRASES,
            "num_mappings": len(self.CONTEXT_MAPPING),
            "num_age_keywords": len(self.AGE_KEYWORDS),
            "num_uniform_colors": len(self.UNIFORM_COLORS),
            "is_loaded": self.model is not None
        }
