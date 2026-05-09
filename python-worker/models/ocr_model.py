import logging
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class OcrModel:
    """
    Dedicated OCR model using PaddleOCR to replace Florence-2's hallucination-prone OCR.
    PaddleOCR is lightweight, fast on CPU, and supports Latin characters well.
    """

    def __init__(self, gpu: bool = False):
        self.gpu = gpu
        self._ocr = None
        self._is_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            from paddleocr import PaddleOCR
            logger.info(f"Loading PaddleOCR model (gpu={self.gpu})...")
            # 'en' language covers Indonesian (Latin alphabet) well
            self._ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=self.gpu, show_log=False)
            self._is_loaded = True
            logger.info("PaddleOCR loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR: {e}")
            raise

    def get_ocr(self, image: Image.Image, with_regions: bool = False) -> Dict[str, Any]:
        """
        Extract OCR text from image using PaddleOCR.
        
        Returns:
            Dict with "text" key and optionally "regions" key
            Format matches Florence-2's get_ocr output for API compatibility
        """
        try:
            if not self._is_loaded or not self._ocr:
                logger.warning("PaddleOCR not loaded")
                return {"text": "", "regions": []} if with_regions else {"text": ""}

            # Convert PIL image to numpy array (RGB)
            img_np = np.array(image.convert('RGB'))
            # PaddleOCR usually expects BGR format from OpenCV, so convert RGB to BGR
            img_np = img_np[:, :, ::-1]

            # run paddleocr
            # cls=True enables text direction classification
            results = self._ocr.ocr(img_np, cls=True)

            regions = []
            full_text_parts = []

            # handle case where no text is found (results is [None] or [])
            if not results or results[0] is None:
                return {"text": "", "regions": []} if with_regions else {"text": ""}

            for res in results:
                if not res:
                    continue
                for line in res:
                    bbox, (text, prob) = line
                    if text and text.strip():
                        clean_text = text.strip()
                        # bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        # Convert to [x1, y1, x2, y2]
                        try:
                            xs = [pt[0] for pt in bbox]
                            ys = [pt[1] for pt in bbox]
                            flattened_bbox = [min(xs), min(ys), max(xs), max(ys)]
                        except Exception:
                            flattened_bbox = None

                        regions.append({
                            "text": clean_text,
                            "bbox": flattened_bbox,
                            "confidence": float(prob)
                        })
                        full_text_parts.append(clean_text)

            result_text = " \n ".join(full_text_parts)

            if with_regions:
                return {
                    "text": result_text,
                    "regions": regions
                }
            else:
                return {"text": result_text}

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return {"text": "", "regions": []} if with_regions else {"text": ""}
