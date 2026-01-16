"""
OpenCLIP model for vision tagging using zero-shot classification.
Uses ViT-B-32 model for predefined tag classification.
"""

import logging
import torch
from typing import List, Dict, Any
from PIL import Image
import open_clip

logger = logging.getLogger(__name__)


class OpenCLIPModel:
    """Vision tagging using OpenCLIP zero-shot classification."""

    # Predefined tags for government/official context
    TAGS = [
        # Environment
        "indoor",
        "outdoor",
        # Formality
        "formal",
        "informal",
        # Activities
        "meeting",
        "ceremony",
        "presentation",
        "conference",
        # Government context
        "official event",
        "signing ceremony",
        "award ceremony",
        # Group types
        "group photo",
        "portrait",
        "candid"
    ]

    def __init__(self, device_type: str = "cpu", threshold: float = 0.25):
        """
        Initialize OpenCLIP model.

        Args:
            device_type: Device type ("cuda", "mps", or "cpu")
            threshold: Minimum confidence threshold for tag inclusion (default: 0.25)
        """
        self.device_type = device_type
        self.threshold = threshold
        self.device = torch.device(device_type)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the OpenCLIP ViT-B-32 model."""
        try:
            logger.info(f"Loading OpenCLIP ViT-B-32 model on device: {self.device_type}")

            # Load model and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='laion2b_s34b_b79k',
                device=self.device
            )

            # Get tokenizer
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

            # Set model to evaluation mode
            self.model.eval()

            logger.info("OpenCLIP model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            raise

    def get_tags(self, image: Image.Image, threshold: float = None) -> List[str]:
        """
        Get tags for an image using zero-shot classification.

        Args:
            image: PIL Image object
            threshold: Confidence threshold (uses instance threshold if not specified)

        Returns:
            List of tag strings that exceed the confidence threshold

        Example:
            >>> model = OpenCLIPModel()
            >>> img = Image.open("photo.jpg")
            >>> tags = model.get_tags(img)
            >>> print(tags)
            ['outdoor', 'formal', 'group photo']
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        threshold = threshold if threshold is not None else self.threshold

        try:
            with torch.no_grad():
                # Preprocess image
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                # Create text prompts for each tag
                text_prompts = [f"a photo of {tag}" for tag in self.TAGS]
                text_input = self.tokenizer(text_prompts).to(self.device)

                # Get image and text features
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)

                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Calculate similarity (cosine similarity)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len(self.TAGS))

                # Filter tags by threshold
                selected_tags = []
                for value, index in zip(values, indices):
                    confidence = value.item()
                    if confidence >= threshold:
                        tag = self.TAGS[index]
                        selected_tags.append(tag)
                        logger.debug(f"Tag '{tag}' selected with confidence {confidence:.3f}")

            logger.debug(f"Selected {len(selected_tags)} tags from image")
            return selected_tags

        except Exception as e:
            logger.error(f"Error getting tags: {e}")
            raise

    def get_tags_with_scores(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Get tags with their confidence scores.

        Args:
            image: PIL Image object

        Returns:
            List of dictionaries with 'tag' and 'score' keys, sorted by score descending

        Example:
            >>> model = OpenCLIPModel()
            >>> img = Image.open("photo.jpg")
            >>> tags = model.get_tags_with_scores(img)
            >>> for item in tags:
            ...     print(f"{item['tag']}: {item['score']:.3f}")
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            with torch.no_grad():
                # Preprocess image
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                # Create text prompts for each tag
                text_prompts = [f"a photo of {tag}" for tag in self.TAGS]
                text_input = self.tokenizer(text_prompts).to(self.device)

                # Get image and text features
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)

                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len(self.TAGS))

                # Create results
                results = []
                for value, index in zip(values, indices):
                    results.append({
                        "tag": self.TAGS[index],
                        "score": value.item()
                    })

            return results

        except Exception as e:
            logger.error(f"Error getting tags with scores: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "framework": "OpenCLIP",
            "device_type": self.device_type,
            "threshold": self.threshold,
            "num_tags": len(self.TAGS),
            "tags": self.TAGS,
            "is_loaded": self.model is not None
        }
