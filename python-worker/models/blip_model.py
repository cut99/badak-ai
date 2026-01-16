"""
BLIP model for image captioning with Indonesian context mapping.
Uses blip-image-captioning-base to generate captions and maps them to Indonesian phrases.
"""

import logging
import torch
from typing import Dict, Any, List
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

    def get_all_context_phrases(self) -> List[str]:
        """
        Get list of all available Indonesian context phrases.

        Returns:
            List of Indonesian context phrase strings
        """
        return self.CONTEXT_PHRASES.copy()

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
            "is_loaded": self.model is not None
        }
