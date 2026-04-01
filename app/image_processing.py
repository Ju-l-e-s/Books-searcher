import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import io
import logging

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
            try:
                self._model = YOLO("book-spines.pt")
                self._is_custom_model = True
                logger.info("Loaded custom book-spines YOLO model.")
            except Exception:
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
        # Cast to Any to satisfy the linter when external types are not found
        from typing import Any
        results: Any = self.model(img)
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
                    logger.info(f"📦 Détection Livre #{len(spines)+1}: [{x1}, {y1}, {x2}, {y2}] - Taille crop: {crop.shape}")

                    # 2. Preprocessing
                    processed_crop = self._preprocess_crop(crop)

                    # 3. OCR
                    text = self._ocr_crop(processed_crop)
                    if text:
                        logger.info(f"  └─ 📝 OCR Lu: '{text}'")
                        # Filter out standalone small numbers (1-3 digits) as they pollute search
                        if text.isdigit() and len(text) <= 3:
                            logger.info(f"  └─ ⚠️ Nombre court ignoré: {text}")
                            continue
                        spines.append(text)
                    else:
                        logger.warning(f"  └─ ❌ OCR n'a rien pu lire dans ce crop.")
        
        # Normalize and deduplicate (case-insensitive, stripped)
        spines = sorted(list(set(spines)), key=len, reverse=True)
        unique_spines = []
        for s in spines:
            is_subset = False
            for existing in unique_spines:
                # If this text is already contained in a longer string, skip it
                if s.lower() in existing.lower():
                    is_subset = True
                    break
            if not is_subset:
                unique_spines.append(s)

        logger.info(f"🎯 Total textes uniques après filtrage: {len(unique_spines)}")
        return unique_spines

    def _preprocess_crop(self, crop):
        """Prepare crop for OCR: rotate if vertical and enhance contrast."""
        h, w = crop.shape[:2]
        
        # If it's a vertical spine (tall), rotate it to be horizontal (wide)
        # Most French/European books are bottom-to-top, so 90 CW makes them L-to-R.
        # Even if it's top-to-bottom, EasyOCR often handles upside-down better than vertical.
        if h > w:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        # Grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced

    def _ocr_crop(self, crop) -> str:
        """Extract text from a single spine crop."""
        # We use a lower contrast threshold and other params to be more sensitive
        results = self.reader.readtext(crop, detail=0, paragraph=True)
        return " ".join(results).strip()


# Singleton instance (models are lazy-loaded on first request)
processor = ImageProcessor()
