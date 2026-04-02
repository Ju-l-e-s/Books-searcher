# tests/test_vlm_image_processor.py
import numpy as np
import base64
import cv2
from app.vlm_pipeline.image_processor import build_mosaic

def test_build_mosaic_empty():
    assert build_mosaic([]) == []

def test_build_mosaic_single_image():
    # Create a dummy image (100x50 red)
    img = np.zeros((100, 50, 3), dtype=np.uint8)
    img[:, :] = [0, 0, 255] # BGR
    
    mosaics = build_mosaic([img])
    assert len(mosaics) == 1
    assert isinstance(mosaics[0], str)
    
    # Decode and check dimensions
    encoded = mosaics[0] 
    image_data = base64.b64decode(encoded)
    nparr = np.frombuffer(image_data, np.uint8)
    decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # max_h = 100 + 40 = 140
    # total_w = 50 + 10 = 60
    assert decoded_img.shape[0] == 140
    assert decoded_img.shape[1] == 60

def test_build_mosaic_multiple_batches():
    # Create 15 dummy images to test max_per_mosaic=10
    imgs = [np.zeros((100, 50, 3), dtype=np.uint8) for _ in range(15)]
    
    mosaics = build_mosaic(imgs, max_per_mosaic=10)
    assert len(mosaics) == 2
    
    # First mosaic should have 10 images
    image_data_1 = base64.b64decode(mosaics[0])
    decoded_img_1 = cv2.imdecode(np.frombuffer(image_data_1, np.uint8), cv2.IMREAD_COLOR)
    # 10 images * 50px + 10 * 10px spacing = 500 + 100 = 600
    assert decoded_img_1.shape[1] == 600
    
    # Second mosaic should have 5 images
    image_data_2 = base64.b64decode(mosaics[1])
    decoded_img_2 = cv2.imdecode(np.frombuffer(image_data_2, np.uint8), cv2.IMREAD_COLOR)
    # 5 images * 50px + 5 * 10px spacing = 250 + 50 = 300
    assert decoded_img_2.shape[1] == 300
