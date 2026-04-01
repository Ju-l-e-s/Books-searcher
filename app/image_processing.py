import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self):
        self._model = None
        self._reader = None
        self._is_custom_model = False

    @property
    def model(self):
        """Lazy-load YOLO model on first use instead of at import time."""
        if self._model is None:
            custom_model_path = Path(__file__).resolve().parent.parent / "book-spines.pt"
            if custom_model_path.exists():
                self._model = YOLO(str(custom_model_path))
                self._is_custom_model = True
                logger.info("Loaded custom book-spines YOLO model.")
            else:
                logger.warning(
                    "book-spines.pt not found at %s. "
                    "To use a custom YOLO model, place your trained book-spines.pt "
                    "in the project root. Falling back to yolov8n.pt (auto-downloaded). "
                    "See README for training instructions.",
                    custom_model_path,
                )
                self._model = YOLO("yolov8n.pt")
                self._is_custom_model = False
                logger.info("Loaded standard yolov8n YOLO model (fallback).")
        return self._model

    @property
    def reader(self):
        """Lazy-load EasyOCR reader on first use."""
        if self._reader is None:
            self._reader = easyocr.Reader(['fr', 'en'], gpu=False)
            logger.info("EasyOCR reader initialized (fr, en).")
        return self._reader

    def process_image(self, image_bytes: bytes) -> list[str]:
        """Main pipeline: segment -> preprocess -> OCR"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        # 1. Segmentation
        results = self.model(img)
        spines = []

        for result in results:
            if not hasattr(result, 'boxes'):
                continue
            for box in result.boxes:
                # Fix B2: convert Tensor to int before comparison
                cls_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)

                # If using standard YOLO, class 73 is 'book'
                # If using custom model, we take all detections
                if cls_id == 73 or self._is_custom_model:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img[y1:y2, x1:x2]

                    # 2. Preprocessing
                    processed_crop = self._preprocess_crop(crop)

                    # 3. OCR
                    text = self._ocr_crop(processed_crop)
                    if text:
                        spines.append(text)

        # Normalize before deduplication (case-insensitive, stripped)
        seen = set()
        unique_spines = []
        for s in spines:
            normalized = s.lower().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_spines.append(s)

        return unique_spines

    def _preprocess_crop(self, crop):
        """Apply basic deskewing or rotation if the spine is horizontal."""
        h, w = crop.shape[:2]
        if w > h:
            # Likely a horizontal spine, rotate it
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        # Grayscale helps EasyOCR accuracy
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return gray

    def _ocr_crop(self, crop) -> str:
        """Extract text from a single spine crop."""
        results = self.reader.readtext(crop, detail=0)
        return " ".join(results).strip()


# Singleton instance (models are lazy-loaded on first request)
processor = ImageProcessor()
