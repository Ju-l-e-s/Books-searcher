# tests/test_vlm_extractor.py
import pytest
from unittest.mock import MagicMock, patch
import os
import json
from app.vlm_pipeline.vlm_extractor import extract_metadata_from_mosaic

@pytest.mark.asyncio
async def test_extract_metadata_from_mosaic_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"books": [{"id": 1, "title": "Test Book", "author": "Test Author", "publisher": "Test Publisher"}]}'
    mock_client.models.generate_content.return_value = mock_response
    
    with patch("app.vlm_pipeline.vlm_extractor.genai.Client", return_value=mock_client), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
        
        result = await extract_metadata_from_mosaic("ZmFrZSBpbWFnZSBkYXRh") # base64 for "fake image data"
        
        assert "books" in result
        assert len(result["books"]) == 1
        assert result["books"][0]["title"] == "Test Book"
        mock_client.models.generate_content.assert_called_once()

@pytest.mark.asyncio
async def test_extract_metadata_from_mosaic_no_api_key():
    with patch.dict(os.environ, {}, clear=True):
        # We need to ensure GEMINI_API_KEY is not in os.environ
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
        result = await extract_metadata_from_mosaic("ZmFrZSBpbWFnZSBkYXRh")
        assert result == {"books": []}

@pytest.mark.asyncio
async def test_extract_metadata_from_mosaic_error():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API Error")
    
    with patch("app.vlm_pipeline.vlm_extractor.genai.Client", return_value=mock_client), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
        
        result = await extract_metadata_from_mosaic("ZmFrZSBpbWFnZSBkYXRh")
        assert result == {"books": []}

@pytest.mark.asyncio
async def test_extract_metadata_from_mosaic_markdown_json():
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Testing the case where it returns markdown JSON blocks
    mock_response.text = '```json\n{"books": [{"id": 1, "title": "Test Book", "author": "Test Author", "publisher": "Test Publisher"}]}\n```'
    mock_client.models.generate_content.return_value = mock_response
    
    with patch("app.vlm_pipeline.vlm_extractor.genai.Client", return_value=mock_client), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
        
        result = await extract_metadata_from_mosaic("ZmFrZSBpbWFnZSBkYXRh")
        
        assert "books" in result
        assert len(result["books"]) == 1
        assert result["books"][0]["title"] == "Test Book"
