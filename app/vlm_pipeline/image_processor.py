# app/vlm_pipeline/image_processor.py
import cv2
import numpy as np
import base64
from typing import List
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = YOLO("yolo11n.pt")
        except Exception:
            _model = YOLO("yolov8n.pt")
    return _model

def extract_spines(image_bytes: bytes) -> List[np.ndarray]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return []
    model = get_model()
    results = model(img, conf=0.15, iou=0.3)
    crops = []
    for result in results:
        if not hasattr(result, 'boxes'): continue
        for box in result.boxes:
            cls_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            if cls_id == 73 or True:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h_img, w_img = img.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    h, w = crop.shape[:2]
                    if h < w:
                         crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                    crops.append(crop)
    return crops

def build_mosaic(cropped_images: List[np.ndarray], max_per_mosaic: int = 10) -> List[str]:
    if not cropped_images: return []
    mosaics_b64 = []
    for i in range(0, len(cropped_images), max_per_mosaic):
        batch = cropped_images[i:i+max_per_mosaic]
        max_h = max(img.shape[0] for img in batch) + 40
        total_w = sum(img.shape[1] for img in batch) + (10 * len(batch))
        mosaic = np.full((max_h, total_w, 3), 255, dtype=np.uint8)
        current_x = 0
        for j, img in enumerate(batch):
            h, w = img.shape[:2]
            book_id = i + j + 1
            cv2.putText(mosaic, str(book_id), (current_x + (w//2) - 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            mosaic[40:40+h, current_x:current_x+w] = img
            current_x += w + 10
        _, buffer = cv2.imencode('.jpg', mosaic)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        mosaics_b64.append(b64_str)
    return mosaics_b64
