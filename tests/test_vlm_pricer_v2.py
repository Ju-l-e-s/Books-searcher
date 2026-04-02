# tests/test_vlm_pricer.py
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from app.vlm_pipeline.pricer import get_prices_async

@pytest.mark.asyncio
async def test_get_prices_async_success():
    isbn = "9782253006329"
    
    # Mock responses for Momox and RecycLivre
    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "momox.fr" in url:
            resp.json.return_value = {"buybackPrice": 5.50}
        elif "recyclivre.com" in url:
            resp.text = "Nous vous proposons 3,20 € pour ce livre"
        return resp

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        result = await get_prices_async(isbn)
        assert result == {"momox": 5.50, "recyclivre": 3.20}

@pytest.mark.asyncio
async def test_get_prices_async_momox_only():
    isbn = "9782253006329"
    
    async def mock_get(url, **kwargs):
        resp = MagicMock()
        if "momox.fr" in url:
            resp.status_code = 200
            resp.json.return_value = {"price": 10.0}
        else:
            resp.status_code = 404
        return resp

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        result = await get_prices_async(isbn)
        assert result == {"momox": 10.0, "recyclivre": 0.0}

@pytest.mark.asyncio
async def test_get_prices_async_recyclivre_only():
    isbn = "9782253006329"
    
    async def mock_get(url, **kwargs):
        resp = MagicMock()
        if "recyclivre.com" in url:
            resp.status_code = 200
            resp.text = "Prix: 7.50€"
        else:
            resp.status_code = 500
        return resp

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        result = await get_prices_async(isbn)
        assert result == {"momox": 0.0, "recyclivre": 7.50}

@pytest.mark.asyncio
async def test_get_prices_async_timeout():
    isbn = "9782253006329"
    
    async def slow_get(*args, **kwargs):
        await asyncio.sleep(6) # Longer than 5s timeout
        return MagicMock()

    with patch("httpx.AsyncClient.get", side_effect=slow_get):
        result = await get_prices_async(isbn)
        assert result == {"momox": 0.0, "recyclivre": 0.0}

@pytest.mark.asyncio
async def test_get_prices_async_exception():
    isbn = "9782253006329"
    
    async def error_get(*args, **kwargs):
        raise Exception("Network Error")

    with patch("httpx.AsyncClient.get", side_effect=error_get):
        result = await get_prices_async(isbn)
        assert result == {"momox": 0.0, "recyclivre": 0.0}

@pytest.mark.asyncio
async def test_get_prices_async_different_momox_keys():
    isbn = "9782253006329"
    
    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "momox.fr" in url:
            resp.json.return_value = {"totalPrice": 12.34}
        else:
            resp.status_code = 404
        return resp

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        result = await get_prices_async(isbn)
        assert result["momox"] == 12.34
