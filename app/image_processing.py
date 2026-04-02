import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import io
import logging
import re

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
        # We lower confidence threshold to 0.15 and use a smaller overlap (iou) 
        # to force YOLO to find individual spines even if they are stacked.
        from typing import Any
        results: Any = self.model(img, conf=0.15, iou=0.3)
        spines = []

        for result in results:
            if not hasattr(result, 'boxes'):
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                if cls_id == 73 or self._is_custom_model:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Ensure within bounds
                    h_img, w_img = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0: continue

                    # 2. Advanced OCR for this specific box
                    texts = self._process_single_spine(crop)
                    spines.extend(texts)
        
        # 3. Final cleaning
        unique_spines = self._filter_unique_texts(spines)
        logger.info(f"🎯 Total textes uniques envoyés au resolver: {len(unique_spines)}")
        return unique_spines

    def _process_single_spine(self, crop) -> list[str]:
        """OCR a single detected spine with orientation checks."""
        h, w = crop.shape[:2]
        
        # Orient crop horizontally for OCR
        if h > w:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h

        # Try OCR in both directions (normal and 180 reversed)
        # because book spines can be oriented either way.
        results_normal = self._ocr_raw(crop)
        
        # Flip and try again
        crop_flipped = cv2.rotate(crop, cv2.ROTATE_180)
        results_flipped = self._ocr_raw(crop_flipped)
        
        # Build text strings
        text_normal = " ".join(results_normal).strip()
        text_flipped = " ".join(results_flipped).strip()
        
        final_texts = []
        if len(text_normal) > 5: final_texts.append(text_normal)
        if len(text_flipped) > 5: final_texts.append(text_flipped)
        
        # If the crop is thick, it might be a merged detection. Try a split.
        if h > (w * 0.15):
            mid = h // 2
            for part in [crop[0:mid, :], crop[mid:h, :]]:
                t = " ".join(self._ocr_raw(part)).strip()
                if len(t) > 5: final_texts.append(t)

        return final_texts

    def _ocr_raw(self, crop) -> list[str]:
        """Perform raw OCR on a crop and return list of text segments."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Normalize to improve readability
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return self.reader.readtext(norm, detail=0, paragraph=True)

    def _filter_unique_texts(self, texts: list[str]) -> list[str]:
        """Rigorous deduplication: remove overlaps and garbage."""
        # Clean segments
        cleaned = []
        for t in texts:
            # Remove non-alphanumeric noise at edges
            t = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', t).strip()
            if len(t) > 4:
                cleaned.append(t)
        
        # Dedup and remove subsets
        cleaned = sorted(list(set(cleaned)), key=len, reverse=True)
        unique = []
        for t in cleaned:
            if not any(t.lower() in other.lower() for other in unique):
                unique.append(t)
        return unique


# Singleton instance
processor = ImageProcessor()
