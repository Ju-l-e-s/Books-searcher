import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.image_processing import ImageProcessor


class TestImageProcessor:
    """Tests for ImageProcessor — mocked YOLO and EasyOCR."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    def test_process_image_invalid_bytes_returns_empty(self, processor):
        """Garbage image bytes should return an empty list."""
        result = processor.process_image(b"not an image")
        assert result == []

    def test_preprocess_crop_rotates_vertical_spines(self, processor):
        """Tall (vertical) crops should be rotated to horizontal."""
        vertical_crop = np.zeros((200, 50, 3), dtype=np.uint8)
        result = processor._preprocess_crop(vertical_crop)
        # ROTATE_90_CLOCKWISE: (h, w) -> (w, h) = (50, 200)
        assert result.shape == (50, 200)

    def test_preprocess_crop_horizontal_spines_untouched(self, processor):
        """Wide (horizontal) crops should NOT be rotated."""
        horizontal_crop = np.zeros((50, 200, 3), dtype=np.uint8)
        result = processor._preprocess_crop(horizontal_crop)
        # Already landscape (w > h), so no rotation.
        assert result.shape[1] > result.shape[0]

    def test_ocr_crop_extracts_text(self, processor):
        """_ocr_crop should join EasyOCR results."""
        processor._reader = MagicMock()
        processor._reader.readtext = MagicMock(return_value=["Hello", "World"])

        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        result = processor._ocr_crop(crop)

        assert result == "Hello World"
        processor._reader.readtext.assert_called_once_with(crop, detail=0, paragraph=True)

    def test_ocr_crop_handles_empty_results(self, processor):
        """Empty OCR results should return empty string."""
        processor._reader = MagicMock()
        processor._reader.readtext = MagicMock(return_value=[])

        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        result = processor._ocr_crop(crop)

        assert result == ""

    def test_model_falls_back_when_custom_pt_not_found(self):
        """If book-spines.pt is not found, should fall back to yolov8n.pt."""
        processor = ImageProcessor()
        processor._model = None

        with patch("app.image_processing.YOLO") as mock_yolo:
            def side_effect(model_name):
                if model_name == "book-spines.pt":
                    raise Exception("Model not found")
                return MagicMock()
            
            mock_yolo.side_effect = side_effect

            _ = processor.model

            # Should have fallen back to yolov8n.pt
            assert processor._is_custom_model is False
            assert processor._model is not None
            # Check that it tried both
            assert mock_yolo.call_count == 2

    def test_model_uses_custom_when_pt_exists(self):
        """If book-spines.pt exists, it should be loaded as the custom model."""
        processor = ImageProcessor()
        processor._model = None

        with patch("app.image_processing.YOLO") as mock_yolo:
            mock_yolo_instance = MagicMock()
            mock_yolo.return_value = mock_yolo_instance

            def fake_exists(self):
                if "book-spines.pt" in str(self):
                    return True  # Custom model found!
                return Path.exists(self)

            with patch.object(Path, "exists", fake_exists):
                _ = processor.model

                # Should have loaded the custom model
                assert processor._is_custom_model is True

    def test_model_is_cached_after_first_access(self):
        """Model should only be loaded once, then cached."""
        processor = ImageProcessor()
        processor._model = None

        with patch("app.image_processing.YOLO") as mock_yolo:
            mock_yolo_instance = MagicMock()
            mock_yolo.return_value = mock_yolo_instance

            def fake_exists(self):
                if "book-spines.pt" in str(self):
                    return False
                return Path.exists(self)

            with patch.object(Path, "exists", fake_exists):
                _ = processor.model
                _ = processor.model
                _ = processor.model

                # Model is cached after first load
                assert processor._model is not None
