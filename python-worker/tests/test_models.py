"""
Unit tests for AI models.
Tests InsightFace, OpenCLIP, and BLIP models.
"""

import pytest
import numpy as np
from PIL import Image
import os

from models.insightface_model import InsightFaceModel, Face
from models.openclip_model import OpenCLIPModel
from models.blip_model import BLIPModel


# Test fixtures
@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a 640x480 RGB image with random colors
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')


@pytest.fixture
def face_crop_image():
    """Create a smaller face-like image for testing."""
    # Create a 200x200 RGB image
    img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')


class TestInsightFaceModel:
    """Test cases for InsightFace model."""

    def test_model_initialization(self):
        """Test that InsightFace model initializes successfully."""
        model = InsightFaceModel(device_type="cpu", onnx_providers=["CPUExecutionProvider"])
        assert model is not None
        assert model.model is not None
        assert model.device_type == "cpu"

    def test_model_info(self):
        """Test getting model information."""
        model = InsightFaceModel(device_type="cpu", onnx_providers=["CPUExecutionProvider"])
        info = model.get_model_info()

        assert info["model_name"] == "buffalo_l"
        assert info["framework"] == "InsightFace"
        assert info["device_type"] == "cpu"
        assert info["embedding_dim"] == 512
        assert info["is_loaded"] is True

    def test_detect_faces_returns_list(self, sample_image):
        """Test that detect_faces returns a list."""
        model = InsightFaceModel(device_type="cpu", onnx_providers=["CPUExecutionProvider"])
        faces = model.detect_faces(sample_image)

        assert isinstance(faces, list)
        # Note: May return empty list if no faces detected in random image

    def test_face_object_structure(self):
        """Test Face dataclass structure."""
        embedding = np.random.rand(512).astype(np.float32)
        face = Face(
            bounding_box=[10, 20, 100, 120],
            embedding=embedding,
            confidence=0.95,
            age=30,
            gender=1
        )

        assert face.bounding_box == [10, 20, 100, 120]
        assert len(face.embedding) == 512
        assert 0.0 <= face.confidence <= 1.0
        assert face.age == 30
        assert face.gender == 1

    def test_crop_face(self, sample_image):
        """Test face cropping functionality."""
        model = InsightFaceModel(device_type="cpu", onnx_providers=["CPUExecutionProvider"])
        bounding_box = [50, 50, 200, 200]

        cropped = model.crop_face(sample_image, bounding_box)

        assert isinstance(cropped, Image.Image)
        # Check that cropped image dimensions are reasonable
        assert cropped.width > 0
        assert cropped.height > 0


class TestOpenCLIPModel:
    """Test cases for OpenCLIP model."""

    def test_model_initialization(self):
        """Test that OpenCLIP model initializes successfully."""
        model = OpenCLIPModel(device_type="cpu", threshold=0.25)
        assert model is not None
        assert model.model is not None
        assert model.device_type == "cpu"
        assert model.threshold == 0.25

    def test_model_info(self):
        """Test getting model information."""
        model = OpenCLIPModel(device_type="cpu")
        info = model.get_model_info()

        assert info["model_name"] == "ViT-B-32"
        assert info["pretrained"] == "laion2b_s34b_b79k"
        assert info["framework"] == "OpenCLIP"
        assert info["device_type"] == "cpu"
        assert info["num_tags"] > 0
        assert "tags" in info
        assert info["is_loaded"] is True

    def test_predefined_tags(self):
        """Test that predefined tags are available."""
        model = OpenCLIPModel(device_type="cpu")

        assert len(model.TAGS) > 0
        # Check for some expected tags
        assert "indoor" in model.TAGS
        assert "outdoor" in model.TAGS
        assert "formal" in model.TAGS

    def test_get_tags_returns_list(self, sample_image):
        """Test that get_tags returns a list of strings."""
        model = OpenCLIPModel(device_type="cpu", threshold=0.25)
        tags = model.get_tags(sample_image)

        assert isinstance(tags, list)
        # All items should be strings
        for tag in tags:
            assert isinstance(tag, str)

    def test_get_tags_with_custom_threshold(self, sample_image):
        """Test get_tags with custom threshold."""
        model = OpenCLIPModel(device_type="cpu", threshold=0.25)

        # Lower threshold should potentially return more tags
        tags_low = model.get_tags(sample_image, threshold=0.1)
        tags_high = model.get_tags(sample_image, threshold=0.5)

        assert isinstance(tags_low, list)
        assert isinstance(tags_high, list)
        # Lower threshold should have >= tags than higher threshold
        assert len(tags_low) >= len(tags_high)

    def test_get_tags_with_scores(self, sample_image):
        """Test get_tags_with_scores returns proper structure."""
        model = OpenCLIPModel(device_type="cpu")
        results = model.get_tags_with_scores(sample_image)

        assert isinstance(results, list)
        for item in results:
            assert "tag" in item
            assert "score" in item
            assert isinstance(item["tag"], str)
            assert isinstance(item["score"], float)
            assert 0.0 <= item["score"] <= 1.0


class TestBLIPModel:
    """Test cases for BLIP model."""

    def test_model_initialization(self):
        """Test that BLIP model initializes successfully."""
        model = BLIPModel(device_type="cpu")
        assert model is not None
        assert model.model is not None
        assert model.processor is not None
        assert model.device_type == "cpu"

    def test_model_info(self):
        """Test getting model information."""
        model = BLIPModel(device_type="cpu")
        info = model.get_model_info()

        assert info["model_name"] == "blip-image-captioning-base"
        assert info["framework"] == "Transformers (Salesforce BLIP)"
        assert info["device_type"] == "cpu"
        assert info["num_context_phrases"] > 0
        assert info["num_mappings"] > 0
        assert info["is_loaded"] is True

    def test_context_phrases_available(self):
        """Test that Indonesian context phrases are defined."""
        model = BLIPModel(device_type="cpu")
        phrases = model.get_all_context_phrases()

        assert len(phrases) > 0
        # Check for some expected phrases
        assert "sedang bersalaman" in phrases
        assert "sedang duduk" in phrases
        assert "foto bersama" in phrases

    def test_context_mapping_available(self):
        """Test that context mapping dictionary is defined."""
        model = BLIPModel(device_type="cpu")

        assert len(model.CONTEXT_MAPPING) > 0
        # Check some key mappings
        assert "shaking hands" in model.CONTEXT_MAPPING
        assert model.CONTEXT_MAPPING["shaking hands"] == "sedang bersalaman"

    def test_generate_caption_returns_string(self, sample_image):
        """Test that generate_caption returns a string."""
        model = BLIPModel(device_type="cpu")
        caption = model.generate_caption(sample_image)

        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_get_context_returns_indonesian_phrase(self, sample_image):
        """Test that get_context returns an Indonesian phrase."""
        model = BLIPModel(device_type="cpu")
        context = model.get_context(sample_image)

        assert isinstance(context, str)
        assert len(context) > 0
        # Context should be one of the predefined phrases
        assert context in model.CONTEXT_PHRASES

    def test_get_context_with_caption(self, sample_image):
        """Test get_context_with_caption returns both fields."""
        model = BLIPModel(device_type="cpu")
        result = model.get_context_with_caption(sample_image)

        assert isinstance(result, dict)
        assert "caption" in result
        assert "context" in result
        assert isinstance(result["caption"], str)
        assert isinstance(result["context"], str)
        assert result["context"] in model.CONTEXT_PHRASES

    def test_match_context_with_known_keyword(self):
        """Test _match_context with known keywords."""
        model = BLIPModel(device_type="cpu")

        # Test direct keyword match
        context = model._match_context("two men shaking hands")
        assert context == "sedang bersalaman"

        context = model._match_context("people sitting at table")
        assert context == "sedang duduk"

        context = model._match_context("group photo outdoors")
        assert context == "foto bersama"

    def test_match_context_fallback(self):
        """Test _match_context fallback behavior."""
        model = BLIPModel(device_type="cpu")

        # Test with unrecognized content (should return default fallback)
        context = model._match_context("abstract pattern colors")
        assert context in model.CONTEXT_PHRASES  # Should return a valid phrase


# Integration test
class TestModelIntegration:
    """Integration tests combining multiple models."""

    def test_full_pipeline_with_sample_image(self, sample_image):
        """Test complete processing pipeline with all models."""
        # Initialize all models
        insightface = InsightFaceModel(device_type="cpu", onnx_providers=["CPUExecutionProvider"])
        openclip = OpenCLIPModel(device_type="cpu")
        blip = BLIPModel(device_type="cpu")

        # Process image
        faces = insightface.detect_faces(sample_image)
        tags = openclip.get_tags(sample_image)
        context = blip.get_context(sample_image)

        # Verify outputs
        assert isinstance(faces, list)
        assert isinstance(tags, list)
        assert isinstance(context, str)
        assert context in blip.CONTEXT_PHRASES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
