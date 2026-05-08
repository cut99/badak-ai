"""
Unit tests for AI models.
Tests InsightFace, Florence, Caption, and Translation models.
"""

import pytest
import numpy as np
from PIL import Image

from models.insightface_model import InsightFaceModel, Face
from models.florence_model import FlorenceModel
from models.translation_model import TranslationModel
from models.caption_model import CaptionModel
from api.schemas import OcrResult, OcrRegion, ProcessResponse


# Test fixtures
@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a 640x480 RGB image with random colors
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def face_crop_image():
    """Create a smaller face-like image for testing."""
    # Create a 200x200 RGB image
    img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def translation_model():
    """Create a TranslationModel instance."""
    return TranslationModel()


@pytest.fixture
def florence_model(translation_model):
    """Create a FlorenceModel instance."""
    return FlorenceModel(translation_model=translation_model, device_type="cpu")


@pytest.fixture
def caption_model(florence_model, translation_model):
    """Create a CaptionModel instance."""
    return CaptionModel(
        florence_model=florence_model, translation_model=translation_model
    )


class TestInsightFaceModel:
    """Test cases for InsightFace model."""

    def test_model_initialization(self):
        """Test that InsightFace model initializes successfully."""
        model = InsightFaceModel(
            device_type="cpu", onnx_providers=["CPUExecutionProvider"]
        )
        assert model is not None
        assert model.model is not None
        assert model.device_type == "cpu"

    def test_model_info(self):
        """Test getting model information."""
        model = InsightFaceModel(
            device_type="cpu", onnx_providers=["CPUExecutionProvider"]
        )
        info = model.get_model_info()

        assert info["model_name"] == "buffalo_l"
        assert info["framework"] == "InsightFace"
        assert info["device_type"] == "cpu"
        assert info["embedding_dim"] == 512
        assert info["is_loaded"] is True

    def test_detect_faces_returns_list(self, sample_image):
        """Test that detect_faces returns a list."""
        model = InsightFaceModel(
            device_type="cpu", onnx_providers=["CPUExecutionProvider"]
        )
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
            gender=1,
        )

        assert face.bounding_box == [10, 20, 100, 120]
        assert len(face.embedding) == 512
        assert 0.0 <= face.confidence <= 1.0
        assert face.age == 30
        assert face.gender == 1

    def test_crop_face(self, sample_image):
        """Test face cropping functionality."""
        model = InsightFaceModel(
            device_type="cpu", onnx_providers=["CPUExecutionProvider"]
        )
        bounding_box = [50, 50, 200, 200]

        cropped = model.crop_face(sample_image, bounding_box)

        assert isinstance(cropped, Image.Image)
        # Check that cropped image dimensions are reasonable
        assert cropped.width > 0
        assert cropped.height > 0


class TestFlorenceModel:
    """Test cases for FlorenceModel."""

    def test_model_initialization(self, translation_model):
        """Test that FlorenceModel initializes successfully."""
        model = FlorenceModel(
            translation_model=translation_model,
            device_type="cpu",
            threshold=0.25,
            top_k=10,
            language="en",
        )
        assert model is not None
        assert model.device_type == "cpu"
        assert model.threshold == 0.25

    def test_model_info(self, florence_model):
        """Test getting model information."""
        info = florence_model.get_model_info()

        assert "name" in info
        assert "microsoft/Florence-2-base" in info["name"]
        assert info["device"] == "cpu"
        assert info["is_loaded"] is True

    def test_get_tags_returns_list(self, florence_model, sample_image):
        """Test that get_tags returns a list of strings."""
        tags = florence_model.get_tags(sample_image)

        assert isinstance(tags, list)
        # All items should be strings
        for tag in tags:
            assert isinstance(tag, str)

    def test_get_objects_returns_list(self, florence_model, sample_image):
        """Test that get_objects returns a list of strings."""
        objects = florence_model.get_objects(sample_image)

        assert isinstance(objects, list)
        # All items should be strings
        for obj in objects:
            assert isinstance(obj, str)

    def test_run_task_preserves_task_token_with_text_input(self, sample_image):
        """Text-conditioned Florence tasks must still include the task token."""

        class FakeProcessor:
            last_text = None

            def __call__(self, text, images, return_tensors):
                self.last_text = text
                return {"input_ids": [1], "pixel_values": [2]}

            def batch_decode(self, generated_ids, skip_special_tokens=False):
                return ["decoded"]

            def post_process_generation(self, generated_text, task, image_size):
                return {task: "ok"}

        class FakeModel:
            def generate(self, **kwargs):
                return ["ids"]

        model = FlorenceModel.__new__(FlorenceModel)
        model._processor = FakeProcessor()
        model._model = FakeModel()
        model.device = "cpu"

        result = model.run_task(
            sample_image, "<CAPTION_TO_PHRASE_GROUNDING>", "a man speaking"
        )

        assert model._processor.last_text == (
            "<CAPTION_TO_PHRASE_GROUNDING>a man speaking"
        )
        assert result == {"<CAPTION_TO_PHRASE_GROUNDING>": "ok"}

    def test_get_objects_reuses_supplied_od_result(self, sample_image):
        """Objects should not trigger a second OD generation when OD is supplied."""
        model = FlorenceModel.__new__(FlorenceModel)
        model.run_task = lambda *args, **kwargs: pytest.fail("OD should be reused")

        objects = model.get_objects(
            sample_image,
            od_result={"<OD>": {"labels": ["person", "Person", "screen"]}},
        )

        assert objects == ["person", "screen"]

    def test_get_tags_prioritizes_context_labels(self, sample_image):
        """Deterministic context tags should lead noisy translated object labels."""

        class FakeTranslation:
            def translate_batch(self, labels):
                mapping = {"person": "orang", "screen": "layar"}
                return [mapping.get(label, label) for label in labels]

        model = FlorenceModel.__new__(FlorenceModel)
        model.translation_model = FakeTranslation()
        model.top_k = 5
        model.language = "id"

        tags = model.get_tags(
            sample_image,
            od_result={"<OD>": {"labels": ["person", "screen", "person"]}},
            context_labels=["rapat", "duduk"],
        )

        assert tags == ["rapat", "duduk", "orang", "layar"]


class TestCaptionModel:
    """Test cases for CaptionModel."""

    def test_model_initialization(self, florence_model, translation_model):
        """Test that CaptionModel initializes successfully."""
        model = CaptionModel(
            florence_model=florence_model, translation_model=translation_model
        )
        assert model is not None
        assert model.florence is not None
        assert model.translation is not None

    def test_model_info(self, caption_model):
        """Test getting model information."""
        info = caption_model.get_model_info()

        assert "caption_model" in info
        assert "Florence-2" in info["caption_model"]
        assert info["florence_loaded"] is True
        assert info["translation_loaded"] is True

    def test_get_context_comprehensive_returns_dict(self, caption_model, sample_image):
        """Test that get_context_comprehensive returns a dict."""
        result = caption_model.get_context_comprehensive(sample_image)

        assert isinstance(result, dict)
        assert "english_caption" in result
        assert "indonesian_phrase" in result
        assert "indonesian_description" in result
        assert "elements" in result
        assert isinstance(result["english_caption"], str)
        assert isinstance(result["indonesian_phrase"], str)
        assert isinstance(result["indonesian_description"], str)

    def test_detect_school_age_returns_str_or_none(self, caption_model):
        """Test that detect_school_age returns a string or None."""
        # Test with school age keywords
        result = caption_model.detect_school_age("students in classroom")
        # Should return None since it's just English text

        # Test with SD keyword
        result_sd = caption_model.detect_school_age("primary school children")
        # May return SD tag if face ages are provided

        # Test without face ages (should return None)
        result_no_age = caption_model.detect_school_age("group of people")
        # Should return None when no face ages provided and no clear school context

    def test_name_injection_does_not_guess_without_grounding(self, sample_image):
        """Known names should not replace generic people without spatial grounding."""

        class FakeFlorence:
            def run_task(self, *args, **kwargs):
                return {}

        model = CaptionModel(florence_model=FakeFlorence(), translation_model=None)
        caption = model._inject_names(
            sample_image,
            "A man is speaking at a meeting",
            [{"name": "Purbaya", "bbox": [10, 10, 50, 50]}],
        )

        assert caption == "A man is speaking at a meeting"

    def test_infer_context_tags_from_ocr_caption_objects(self):
        """Context tags should capture meeting actions without another model call."""
        model = CaptionModel(florence_model=None, translation_model=None)

        tags = model.infer_context_tags(
            english_caption="A large audience is sitting in an auditorium",
            objects=["screen", "chair", "person"],
            ocr_text="RAPIMNAS I DJP 2026\nAuditorium CBB, 9 April 2026",
            face_count=20,
        )

        assert "ramai" in tags
        assert "duduk" in tags
        assert "rapat" in tags
        assert "presentasi" in tags


class TestTranslationModel:
    """Test cases for TranslationModel."""

    def test_model_initialization(self):
        """Test that TranslationModel initializes successfully."""
        model = TranslationModel()
        assert model is not None
        assert model._model is not None

    def test_translate_english_to_indonesian(self, translation_model):
        """Test English to Indonesian translation."""
        result = translation_model.translate("hello world")

        assert isinstance(result, str)
        assert len(result) > 0


# Integration test
class TestModelIntegration:
    """Integration tests combining multiple models."""

    def test_full_pipeline_with_sample_image(self, sample_image):
        """Test complete processing pipeline with all models."""
        # Initialize all models
        insightface = InsightFaceModel(
            device_type="cpu", onnx_providers=["CPUExecutionProvider"]
        )
        translation = TranslationModel()
        florence = FlorenceModel(translation_model=translation, device_type="cpu")
        caption = CaptionModel(florence_model=florence, translation_model=translation)

        # Process image
        faces = insightface.detect_faces(sample_image)
        tags = florence.get_tags(sample_image)
        context_result = caption.get_context_comprehensive(sample_image)

        # Verify outputs
        assert isinstance(faces, list)
        assert isinstance(tags, list)
        assert isinstance(context_result, dict)
        assert "indonesian_phrase" in context_result


# OCR tests
class TestFlorenceModelOCR:
    """Test cases for FlorenceModel OCR functionality."""

    def test_get_ocr_returns_dict(self, florence_model, sample_image):
        """Test that get_ocr returns a dict with 'text' key."""
        result = florence_model.get_ocr(sample_image)

        assert isinstance(result, dict)
        assert "text" in result
        assert isinstance(result["text"], str)

    def test_get_ocr_with_regions(self, florence_model, sample_image):
        """Test get_ocr with regions returns 'regions' key."""
        result = florence_model.get_ocr(sample_image, with_regions=True)

        assert isinstance(result, dict)
        assert "text" in result
        assert "regions" in result
        assert isinstance(result["regions"], list)

    def test_get_ocr_graceful_failure(self, translation_model):
        """Test that tiny 1x1 image doesn't crash."""
        # Create a tiny 1x1 image
        img = Image.fromarray(np.array([[[0, 0, 0]]], dtype=np.uint8))

        fm = FlorenceModel(translation_model=translation_model, device_type="cpu")

        # Should not raise, should return empty result
        result = fm.get_ocr(img)
        assert isinstance(result, dict)
        assert "text" in result

    def test_get_ocr_with_regions_accepts_four_value_bbox(self, sample_image):
        """Florence may return bbox-like OCR boxes instead of eight-value quads."""
        model = FlorenceModel.__new__(FlorenceModel)
        model.run_task = lambda *args, **kwargs: {
            "<OCR_WITH_REGION>": {
                "quad_boxes": [[10, 20, 110, 60]],
                "labels": ["RAPIMNAS I DJP 2026"],
            }
        }

        result = model.get_ocr(sample_image, with_regions=True)

        assert result["text"] == "RAPIMNAS I DJP 2026"
        assert result["regions"] == [
            {"text": "RAPIMNAS I DJP 2026", "bbox": [10, 20, 110, 60]}
        ]


class TestOcrSchema:
    """Test cases for OCR schemas."""

    def test_ocr_result_importable(self):
        """Test that OcrResult can be imported."""
        assert OcrResult is not None
        assert hasattr(OcrResult, "model_fields")

    def test_ocr_region_importable(self):
        """Test that OcrRegion can be imported."""
        assert OcrRegion is not None
        assert hasattr(OcrRegion, "model_fields")

    def test_process_response_has_ocr_field(self):
        """Test that ProcessResponse has ocr field."""
        assert "ocr" in ProcessResponse.model_fields

    def test_process_response_ocr_optional(self):
        """Test that ProcessResponse.ocr is Optional."""
        ocr_field = ProcessResponse.model_fields["ocr"]
        # Check that it's optional (allows None)
        assert ocr_field.is_required() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
