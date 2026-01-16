"""
InsightFace model for face detection and embedding extraction.
Uses buffalo_l model with ONNX Runtime.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


@dataclass
class Face:
    """Face detection result with embedding and metadata."""
    bounding_box: List[int]  # [x1, y1, x2, y2]
    embedding: np.ndarray    # 512-dim face embedding
    confidence: float        # Detection confidence (0-1)
    age: Optional[int]       # Estimated age
    gender: Optional[int]    # 0=female, 1=male


class InsightFaceModel:
    """Face detection and recognition using InsightFace buffalo_l model."""

    def __init__(self, device_type: str = "cpu", onnx_providers: List[str] = None):
        """
        Initialize InsightFace model.

        Args:
            device_type: Device type ("cuda", "mps", or "cpu")
            onnx_providers: List of ONNX Runtime providers
        """
        self.device_type = device_type
        self.onnx_providers = onnx_providers or ["CPUExecutionProvider"]
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the InsightFace buffalo_l model."""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            logger.info(f"Loading InsightFace buffalo_l model with providers: {self.onnx_providers}")

            # Initialize FaceAnalysis with buffalo_l model
            self.model = FaceAnalysis(
                name="buffalo_l",
                providers=self.onnx_providers
            )

            # Prepare model with default context (input size 640x640)
            self.model.prepare(ctx_id=0 if self.device_type == "cuda" else -1, det_size=(640, 640))

            logger.info("InsightFace model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise

    def detect_faces(self, image: Image.Image) -> List[Face]:
        """
        Detect faces in an image and extract embeddings.

        Args:
            image: PIL Image object

        Returns:
            List of Face objects with bounding boxes, embeddings, and metadata

        Example:
            >>> model = InsightFaceModel()
            >>> img = Image.open("photo.jpg")
            >>> faces = model.detect_faces(img)
            >>> for face in faces:
            ...     print(f"Face at {face.bounding_box} with confidence {face.confidence:.2f}")
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Convert PIL Image to numpy array (RGB -> BGR for OpenCV)
            img_array = np.array(image)
            if img_array.ndim == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif img_array.shape[2] == 3:  # RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

            # Detect faces
            detected_faces = self.model.get(img_array)

            # Convert to Face objects
            faces = []
            for face_data in detected_faces:
                # Extract bounding box [x1, y1, x2, y2]
                bbox = face_data.bbox.astype(int).tolist()

                # Extract 512-dim embedding
                embedding = face_data.embedding

                # Extract confidence score
                confidence = float(face_data.det_score)

                # Extract age and gender (if available)
                age = int(face_data.age) if hasattr(face_data, 'age') and face_data.age is not None else None
                gender = int(face_data.gender) if hasattr(face_data, 'gender') and face_data.gender is not None else None

                face = Face(
                    bounding_box=bbox,
                    embedding=embedding,
                    confidence=confidence,
                    age=age,
                    gender=gender
                )
                faces.append(face)

            logger.debug(f"Detected {len(faces)} faces in image")
            return faces

        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            raise

    def crop_face(self, image: Image.Image, bounding_box: List[int], padding: int = 20) -> Image.Image:
        """
        Crop a face from an image using bounding box.

        Args:
            image: PIL Image object
            bounding_box: [x1, y1, x2, y2]
            padding: Additional padding around the face (default: 20 pixels)

        Returns:
            Cropped PIL Image of the face
        """
        x1, y1, x2, y2 = bounding_box

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)

        # Crop the face
        face_crop = image.crop((x1, y1, x2, y2))
        return face_crop

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": "buffalo_l",
            "framework": "InsightFace",
            "device_type": self.device_type,
            "onnx_providers": self.onnx_providers,
            "embedding_dim": 512,
            "is_loaded": self.model is not None
        }
