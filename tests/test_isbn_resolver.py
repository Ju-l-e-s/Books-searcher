import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.isbn_resolver import ISBNResolver


class TestISBNResolver:
    """Tests for ISBNResolver — mocked HTTP calls."""

    @pytest.fixture
    def resolver(self):
        return ISBNResolver()

    @pytest.fixture
    def mock_httpx_client(self, resolver):
        client = AsyncMock()
        resolver.client = client
        return client

    async def test_resolve_short_query_returns_empty(self, resolver):
        """Very short strings (< 5 chars) should be rejected early."""
        result = await resolver.resolve("abc")
        assert result == []

    async def test_resolve_empty_query_returns_empty(self, resolver):
        result = await resolver.resolve("")
        assert result == []

    async def test_resolve_whitespace_only_returns_empty(self, resolver):
        result = await resolver.resolve("   ")
        assert result == []

    async def test_google_books_returns_candidates(self, resolver, mock_httpx_client):
        """Google Books should return formatted candidates with ISBN, dimensions, and cover."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "items": [
                {
                    "volumeInfo": {
                        "title": "Le Petit Prince",
                        "authors": ["Antoine de Saint-Exupéry"],
                        "publisher": "Gallimard",
                        "publishedDate": "1943-04-06",
                        "industryIdentifiers": [
                            {"type": "ISBN_13", "identifier": "9783125798100"}
                        ],
                        "dimensions": {
                            "height": "18.0 cm",
                            "width": "12.0 cm",
                            "thickness": "1.0 cm"
                        },
                        "imageLinks": {
                            "thumbnail": "http://example.com/cover.jpg"
                        }
                    }
                }
            ]
        })
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        results = await resolver._search_google_books("petit prince")

        assert len(results) == 1
        assert results[0]["title"] == "Le Petit Prince"
        assert results[0]["isbn"] == "9783125798100"
        assert results[0]["source"] == "google_books"
        assert results[0]["year"] == "1943"
        assert results[0]["cover_url"] == "http://example.com/cover.jpg"
        assert results[0]["dimensions"] == {"height": 18.0, "width": 12.0, "thickness": 1.0}

    def test_parse_dim(self, resolver):
        """Verify _parse_dim handles various dimension string formats."""
        assert resolver._parse_dim("18.0 cm") == 18.0
        assert resolver._parse_dim("18,5 cm") == 18.5
        assert resolver._parse_dim("20 cm") == 20.0
        assert resolver._parse_dim(None) is None
        assert resolver._parse_dim("") is None
        assert resolver._parse_dim("N/A") is None

    async def test_open_library_returns_candidates(self, resolver, mock_httpx_client):
        """Open Library should return formatted candidates with cover URL."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "docs": [
                {
                    "title": "Les Misérables",
                    "author_name": ["Victor Hugo"],
                    "publisher": ["Folio"],
                    "first_publish_year": 1862,
                    "isbn": ["9782253098517"]
                }
            ]
        })
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        results = await resolver._search_open_library("misérables")

        assert len(results) == 1
        assert results[0]["title"] == "Les Misérables"
        assert results[0]["isbn"] == "9782253098517"
        assert results[0]["source"] == "open_library"
        assert results[0]["year"] == "1862"
        assert results[0]["cover_url"] == "https://covers.openlibrary.org/b/isbn/9782253098517-M.jpg"
        assert results[0]["dimensions"] is None

    async def test_google_books_no_isbn_skipped(self, resolver, mock_httpx_client):
        """Books without ISBN should be filtered out."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "items": [
                {
                    "volumeInfo": {
                        "title": "Unknown Book",
                        "authors": ["Author"],
                        # no industryIdentifiers
                    }
                }
            ]
        })
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        results = await resolver._search_google_books("unknown")
        assert results == []

    async def test_open_library_returns_candidates(self, resolver, mock_httpx_client):
        """Open Library should return formatted candidates."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "docs": [
                {
                    "title": "Les Misérables",
                    "author_name": ["Victor Hugo"],
                    "isbn": ["9782253098517"]
                }
            ]
        })
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        results = await resolver._search_open_library("misérables")

        assert len(results) == 1
        assert results[0]["title"] == "Les Misérables"
        assert results[0]["isbn"] == "9782253098517"
        assert results[0]["source"] == "open_library"

    async def test_resolve_combines_and_ranks_both_apis(self, resolver, mock_httpx_client):
        """resolve() should run both APIs in parallel and merge results."""
        mock_gb_response = AsyncMock()
        mock_gb_response.raise_for_status = MagicMock()
        mock_gb_response.json = MagicMock(return_value={
            "items": [
                {
                    "volumeInfo": {
                        "title": "Le Petit Prince",
                        "authors": ["Saint-Exupéry"],
                        "industryIdentifiers": [
                            {"type": "ISBN_13", "identifier": "9783125798100"}
                        ]
                    }
                }
            ]
        })

        mock_ol_response = AsyncMock()
        mock_ol_response.raise_for_status = MagicMock()
        mock_ol_response.json = MagicMock(return_value={
            "docs": [
                {
                    "title": "Le Petit Prince",
                    "author_name": ["Saint-Exupéry"],
                    "isbn": ["9783125798100"]
                }
            ]
        })

        async def get_side_effect(url, **kwargs):
            if "googleapis" in url:
                return mock_gb_response
            return mock_ol_response

        mock_httpx_client.get = AsyncMock(side_effect=get_side_effect)

        results = await resolver.resolve("petit prince")

        assert len(results) > 0
        # Should be sorted by score descending
        scores = [r.get("score", 0) for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_resolve_http_errors_are_handled(self, resolver, mock_httpx_client):
        """HTTP errors in one API should not crash resolve()."""
        mock_gb_response = AsyncMock()
        mock_gb_response.raise_for_status = MagicMock(side_effect=Exception("Network error"))
        mock_httpx_client.get = AsyncMock(return_value=mock_gb_response)

        # Should not raise — exceptions are caught
        results = await resolver.resolve("some book")
        assert isinstance(results, list)

    async def test_close_closes_client(self, resolver):
        """close() should cleanly close the httpx client."""
        resolver.client = AsyncMock()
        await resolver.close()
        resolver.client.aclose.assert_called_once()

    async def test_resolve_uses_cache(self, mock_httpx_client):
        """resolve() should return cached results if available."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = [{"title": "Cached Book", "isbn": "123", "score": 90.0}]
        resolver = ISBNResolver(cache=mock_cache)
        resolver.client = mock_httpx_client
        
        results = await resolver.resolve("some query")
        
        assert len(results) == 1
        assert results[0]["title"] == "Cached Book"
        # Verify no API calls were made
        mock_httpx_client.get.assert_not_called()
        mock_cache.get.assert_called_once()

    async def test_scoring_prioritizes_publisher(self, resolver, mock_httpx_client):
        """Verify that candidates with matching publishers are scored higher."""
        from app.isbn_resolver import _compute_score
        
        ocr = "Le Petit Prince Gallimard"
        cand_generic = {"title": "Le Petit Prince", "author": "Saint-Exupéry", "publisher": "Other"}
        cand_publisher = {"title": "Le Petit Prince", "author": "Saint-Exupéry", "publisher": "Gallimard"}
        
        score_generic = _compute_score(ocr, cand_generic, None, None)
        score_publisher = _compute_score(ocr, cand_publisher, None, None)
        
        assert score_publisher > score_generic
        # ps = 100 * 0.40 = 40 pts difference
        assert score_publisher - score_generic == 40.0

    async def test_penalty_anti_noise(self, resolver):
        """Verify that 'Fiche de lecture' titles receive a heavy penalty."""
        from app.isbn_resolver import _compute_score
        
        ocr = "L'Étranger Camus"
        cand_real = {"title": "L'Étranger", "author": "Albert Camus", "publisher": "Gallimard"}
        cand_noise = {"title": "Fiche de lecture: L'Étranger", "author": "Collectif", "publisher": "Gallimard"}
        
        score_real = _compute_score(ocr, cand_real, "Albert Camus", None)
        score_noise = _compute_score(ocr, cand_noise, "Albert Camus", None)
        
        # -80 pts penalty
        assert score_noise < score_real
        assert (score_real - score_noise) >= 80.0

    async def test_scoring_with_bbox_ratio(self, resolver):
        """Verify that bbox_ratio < 0.1 adds a 'Poche' format bonus (fallback)."""
        from app.isbn_resolver import _compute_score
        
        ocr = "Le Petit Prince"
        cand = {"title": "Le Petit Prince", "author": "Saint-Exupéry", "publisher": "Gallimard"}
        
        score_normal = _compute_score(ocr, cand, None, 0.5)
        score_poche = _compute_score(ocr, cand, None, 0.05)
        
        # fs = 80 * 0.10 = 8 pts bonus (fallback)
        assert score_poche > score_normal
        assert score_poche - score_normal == 8.0

    async def test_scoring_with_triangulation(self, resolver):
        """Verify that matching bbox_ratio with candidate dimensions adds a 100% format bonus."""
        from app.isbn_resolver import _compute_score
        
        ocr = "Le Petit Prince"
        # Candidate ratio = 12/18 = 0.66
        cand = {
            "title": "Le Petit Prince", 
            "dimensions": {"height": 18.0, "width": 12.0}
        }
        
        # Matches (bbox_ratio = 0.66)
        score_match = _compute_score(ocr, cand, None, 0.66)
        # No match (bbox_ratio = 0.3)
        score_no_match = _compute_score(ocr, cand, None, 0.3)
        
        # fs = 100 * 0.10 = 10 pts bonus for match
        assert score_match > score_no_match
        assert score_match - score_no_match == 10.0
