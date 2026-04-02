import asyncio
import pytest
import httpx
from unittest.mock import MagicMock, patch, AsyncMock
from app.pricer import Pricer
from app.infrastructure.cache import InMemoryCache

@pytest.mark.asyncio
async def test_get_prices_cache_hit():
    cache = InMemoryCache()
    isbn = "9782253006329"
    cached_value = {"momox": 1.23, "recyclivre": 4.56}
    cache.set(isbn, cached_value, 3600)
    
    pricer = Pricer(cache=cache)
    result = await pricer.get_prices(isbn)
    
    assert result == cached_value
    await pricer.close()

@pytest.mark.asyncio
async def test_get_prices_timeout():
    # Mock cache to always miss
    cache = InMemoryCache()
    pricer = Pricer(cache=cache)
    
    isbn = "9782253006329"
    
    # Mock _fetch_momox and _fetch_recyclivre to take long
    async def slow_fetch(*args, **kwargs):
        await asyncio.sleep(10)
        return 5.0

    with patch.object(pricer, '_fetch_momox', side_effect=slow_fetch), \
         patch.object(pricer, '_fetch_recyclivre', side_effect=slow_fetch):
        
        # We expect it to return 0.0 after 4s timeout
        start_time = asyncio.get_event_loop().time()
        result = await pricer.get_prices(isbn)
        end_time = asyncio.get_event_loop().time()
        
        assert result == {"momox": 0.0, "recyclivre": 0.0}
        assert end_time - start_time < 5.0 # Should be around 4.0s
    
    await pricer.close()

@pytest.mark.asyncio
async def test_get_prices_error_handling():
    cache = InMemoryCache()
    pricer = Pricer(cache=cache)
    isbn = "9782253006329"
    
    # Mock one success and one error
    async def error_fetch(*args, **kwargs):
        raise Exception("API Error")
        
    async def success_fetch(*args, **kwargs):
        return 10.0

    with patch.object(pricer, '_fetch_momox', side_effect=error_fetch), \
         patch.object(pricer, '_fetch_recyclivre', side_effect=success_fetch):
        
        result = await pricer.get_prices(isbn)
        assert result == {"momox": 0.0, "recyclivre": 10.0}

    await pricer.close()

@pytest.mark.asyncio
async def test_fetch_momox_success():
    cache = InMemoryCache()
    pricer = Pricer(cache=cache)
    isbn = "9782253006329"
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"buybackPrice": 5.50}
    mock_resp.text = '{"buybackPrice": 5.50}'

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        price = await pricer._fetch_momox(isbn)
        assert price == 5.50

    await pricer.close()
