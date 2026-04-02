# tests/test_vlm_isbn_resolver.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.vlm_pipeline.isbn_resolver import resolve_isbn_async

@pytest.mark.asyncio
async def test_resolve_isbn_async_perfect_match():
    # Case: Perfect match for title and author, but no publisher provided in book_dict
    book_dict = {"title": "L'Etranger", "author": "Albert Camus", "publisher": None}
    
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "items": [
                {
                    "volumeInfo": {
                        "title": "L'Etranger",
                        "authors": ["Albert Camus"],
                        "publisher": "Gallimard",
                        "industryIdentifiers": [{"type": "ISBN_13", "identifier": "9782070360024"}]
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp
        
        result = await resolve_isbn_async(book_dict)
        assert result == "9782070360024"

@pytest.mark.asyncio
async def test_resolve_isbn_async_handles_none_from_api():
    # Case: Google Books returns None for some fields
    book_dict = {"title": "L'Etranger", "author": "Albert Camus"}
    
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "items": [
                {
                    "volumeInfo": {
                        "title": None,
                        "authors": None,
                        "publisher": None,
                        "industryIdentifiers": [{"type": "ISBN_13", "identifier": "9782070360024"}]
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp
        
        # Should not crash
        result = await resolve_isbn_async(book_dict)
        assert result is None or isinstance(result, str)
