import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app
from app.models import BookResult

client = TestClient(app)

@pytest.mark.asyncio
async def test_analyze_shelf_texts_success():
    # Mock data for resolution
    mock_resolved = [
        [{"title": "The Great Gatsby", "isbn": "9780743273565", "author": "F. Scott Fitzgerald", "score": 95.0}]
    ]
    
    # Mock data for pricing
    mock_prices = [
        {"momox": 5.5, "recyclivre": 4.2}
    ]

    with patch("app.main.resolver.resolve", new_callable=AsyncMock) as mock_resolve, \
         patch("app.main.pricer.get_prices", new_callable=AsyncMock) as mock_pricer:
        
        mock_resolve.side_effect = mock_resolved
        mock_pricer.side_effect = mock_prices

        request_data = {
            "items": [
                {"text": "Great Gatsby", "bbox_ratio": 0.05}
            ]
        }
        
        response = client.post("/analyze-shelf-texts", json=request_data)
        
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 1
        assert results[0]["title"] == "The Great Gatsby"
        assert results[0]["isbn"] == "9780743273565"
        assert results[0]["price_momox"] == 5.5
        assert results[0]["price_recyclivre"] == 4.2
        assert results[0]["confidence_score"] == 0.95

@pytest.mark.asyncio
async def test_analyze_shelf_texts_empty():
    request_data = {"items": []}
    response = client.post("/analyze-shelf-texts", json=request_data)
    assert response.status_code == 200
    assert response.json() == []

@pytest.mark.asyncio
async def test_analyze_shelf_texts_no_results():
    with patch("app.main.resolver.resolve", new_callable=AsyncMock) as mock_resolve:
        mock_resolve.return_value = []
        
        request_data = {
            "items": [
                {"text": "Unknown Book"}
            ]
        }
        
        response = client.post("/analyze-shelf-texts", json=request_data)
        assert response.status_code == 200
        assert response.json() == []
