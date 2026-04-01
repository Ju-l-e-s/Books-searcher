import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.scraper import ValuationScraper, _parse_price


# ---------------------------------------------------------------------------
# Helper: build a mock Playwright page whose locator returns the given price
# ---------------------------------------------------------------------------

def _make_page_with_price(price_text: str) -> AsyncMock:
    """Return a mock Playwright page whose first matching locator yields *price_text*."""
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
    mock_page.wait_for_timeout = AsyncMock()

    locator = AsyncMock()
    locator.count = AsyncMock(return_value=1)
    locator.inner_text = AsyncMock(return_value=price_text)
    locator.first = locator

    mock_page.locator = MagicMock(return_value=locator)
    return mock_page


def _make_page_body_only(body_text: str) -> AsyncMock:
    """Return a mock page where no selector matches but *body_text* contains a price."""
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.inner_text = AsyncMock(return_value=body_text)

    empty_locator = AsyncMock()
    empty_locator.count = AsyncMock(return_value=0)
    empty_locator.first = empty_locator
    mock_page.locator = MagicMock(return_value=empty_locator)
    return mock_page


def _make_browser_with_page(page: AsyncMock):
    """Return a mock browser that produces the given page from new_context().new_page()."""
    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=page)
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    return mock_browser, mock_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scraper():
    return ValuationScraper()


# ---------------------------------------------------------------------------
# _parse_price unit tests
# ---------------------------------------------------------------------------

class TestParsePrice:
    def test_comma_decimal(self):
        assert _parse_price("5,90 €") == 5.90

    def test_dot_decimal(self):
        assert _parse_price("Prix : 12.50€") == 12.50

    def test_no_price_returns_none(self):
        assert _parse_price("Aucun résultat") is None

    def test_first_price_wins(self):
        assert _parse_price("3,00 € ou 7,50 €") == 3.00


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

class TestCache:
    async def test_cache_hit_skips_network(self, scraper):
        """Cached result is returned without any network call."""
        scraper._price_cache["9781234567890"] = {"momox": 7.5, "recyclivre": 4.0}
        result = await scraper.get_prices("9781234567890")
        assert result == {"momox": 7.5, "recyclivre": 4.0}

    async def test_result_is_cached_after_first_fetch(self, scraper):
        """A fetched result is stored so the second call uses the cache."""
        scraper._client = None
        scraper._browser = None
        isbn = "9780000000001"

        first = await scraper.get_prices(isbn)
        assert isbn in scraper._price_cache
        assert scraper._price_cache[isbn] == first

        # Second call must return identical object (cache hit)
        second = await scraper.get_prices(isbn)
        assert second == first

    async def test_no_client_no_browser_returns_zeroes(self, scraper):
        """Without HTTP client or browser both prices default to 0.0."""
        scraper._client = None
        scraper._browser = None
        result = await scraper.get_prices("9781234567890")
        assert result == {"momox": 0.0, "recyclivre": 0.0}


# ---------------------------------------------------------------------------
# Lifecycle: start / stop
# ---------------------------------------------------------------------------

class TestLifecycle:
    async def test_start_creates_http_client(self, scraper):
        """start() must create an active httpx.AsyncClient."""
        with patch("app.scraper.async_playwright") as mock_apw:
            mock_pw = MagicMock()
            mock_pw.chromium.launch = AsyncMock(return_value=MagicMock())
            mock_apw.return_value.start = AsyncMock(return_value=mock_pw)
            await scraper.start()

        assert scraper._client is not None
        await scraper._client.aclose()

    async def test_start_launches_chromium_headless(self, scraper):
        """start() must launch a headless Chromium browser via Playwright."""
        mock_browser = MagicMock()
        mock_pw = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)

        with patch("app.scraper.async_playwright") as mock_apw:
            mock_apw.return_value.start = AsyncMock(return_value=mock_pw)
            await scraper.start()

        mock_pw.chromium.launch.assert_called_once_with(headless=True)
        assert scraper._browser is mock_browser

    async def test_stop_closes_all_resources(self, scraper):
        """stop() must close the HTTP client, browser, and Playwright instance."""
        scraper._client = AsyncMock()
        scraper._browser = AsyncMock()
        scraper._playwright = AsyncMock()

        await scraper.stop()

        scraper._client.aclose.assert_called_once()
        scraper._browser.close.assert_called_once()
        scraper._playwright.stop.assert_called_once()
        assert scraper._client is None
        assert scraper._browser is None
        assert scraper._playwright is None

    async def test_stop_is_idempotent_when_never_started(self, scraper):
        """stop() on a fresh scraper must not raise."""
        await scraper.stop()  # all attributes are None — should be a no-op

    async def test_ensure_playwright_is_idempotent(self, scraper):
        """_ensure_playwright() must not launch a second browser if one exists."""
        existing_browser = MagicMock()
        scraper._browser = existing_browser

        with patch("app.scraper.async_playwright") as mock_apw:
            await scraper._ensure_playwright()
            mock_apw.assert_not_called()

        assert scraper._browser is existing_browser


# ---------------------------------------------------------------------------
# Momox scraping
# ---------------------------------------------------------------------------

class TestScrapeMomox:
    async def test_playwright_selector_returns_price(self, scraper):
        """Price is extracted when a CSS selector matches."""
        page = _make_page_with_price("5,90 €")
        mock_browser, mock_context = _make_browser_with_page(page)

        scraper._browser = mock_browser
        result = await scraper._scrape_momox("9781234567890")

        assert result == 5.90
        mock_context.close.assert_called_once()

    async def test_playwright_body_regex_fallback(self, scraper):
        """Falls back to body regex when no selector matches."""
        page = _make_page_body_only("Prix de rachat : 12,50 €")
        mock_browser, mock_context = _make_browser_with_page(page)

        scraper._browser = mock_browser
        result = await scraper._scrape_momox("9781234567890")

        assert result == 12.50
        mock_context.close.assert_called_once()

    async def test_no_browser_no_client_returns_none(self, scraper):
        """Returns None gracefully when no browser and no HTTP client are set."""
        scraper._browser = None
        scraper._client = None
        result = await scraper._scrape_momox("9781234567890")
        assert result is None

    async def test_page_goto_exception_returns_none(self, scraper):
        """Returns None when page.goto raises an exception."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("Navigation error"))
        mock_browser, mock_context = _make_browser_with_page(mock_page)

        scraper._browser = mock_browser
        result = await scraper._scrape_momox("9781234567890")

        assert result is None

    async def test_http_json_flat_response(self, scraper):
        """Parses price from a flat JSON API response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"price": 3.50}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        scraper._client = mock_client
        scraper._browser = None
        result = await scraper._scrape_momox("9781234567890")

        assert result == 3.50

    async def test_http_json_nested_items_response(self, scraper):
        """Parses price from a nested items list in the JSON response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": [{"buybackPrice": 4.00}]}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        scraper._client = mock_client
        scraper._browser = None
        result = await scraper._scrape_momox("9781234567890")

        assert result == 4.00

    async def test_http_html_regex_fallback(self, scraper):
        """Falls back to regex when the HTTP response is HTML, not JSON."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("not JSON")
        mock_resp.text = "<html>Rachat : 2,99 €</html>"
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        scraper._client = mock_client
        scraper._browser = None
        result = await scraper._scrape_momox("9781234567890")

        assert result == 2.99

    async def test_http_non_200_tries_next_url(self, scraper):
        """Skips non-200 responses and returns None if all URLs fail."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        scraper._client = mock_client
        scraper._browser = None
        result = await scraper._scrape_momox("9781234567890")

        assert result is None


# ---------------------------------------------------------------------------
# RecycLivre scraping
# ---------------------------------------------------------------------------

class TestScrapeRecyclivre:
    async def test_playwright_selector_returns_price(self, scraper):
        """Price is extracted when a CSS selector matches."""
        page = _make_page_with_price("8,75 €")
        mock_browser, mock_context = _make_browser_with_page(page)

        scraper._browser = mock_browser
        result = await scraper._scrape_recyclivre("9781234567890")

        assert result == 8.75
        mock_context.close.assert_called_once()

    async def test_playwright_body_regex_fallback(self, scraper):
        """Falls back to body regex when no selector matches."""
        page = _make_page_body_only("Prix reprise : 6,30 €")
        mock_browser, mock_context = _make_browser_with_page(page)

        scraper._browser = mock_browser
        result = await scraper._scrape_recyclivre("9781234567890")

        assert result == 6.30

    async def test_no_browser_no_client_returns_none(self, scraper):
        """Returns None gracefully when neither browser nor client is set."""
        scraper._browser = None
        scraper._client = None
        result = await scraper._scrape_recyclivre("9781234567890")
        assert result is None

    async def test_http_json_response(self, scraper):
        """Parses price from a JSON HTTP response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"price": 5.00}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        scraper._client = mock_client
        scraper._browser = None
        result = await scraper._scrape_recyclivre("9781234567890")

        assert result == 5.00

    async def test_http_html_regex_fallback(self, scraper):
        """Falls back to regex when the HTTP response body is HTML."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("not JSON")
        mock_resp.text = "Nous vous proposons 4,20 € pour ce livre."
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        scraper._client = mock_client
        scraper._browser = None
        result = await scraper._scrape_recyclivre("9781234567890")

        assert result == 4.20

    async def test_page_goto_exception_returns_none(self, scraper):
        """Returns None when page navigation raises an exception."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("Timeout"))
        mock_browser, _ = _make_browser_with_page(mock_page)

        scraper._browser = mock_browser
        result = await scraper._scrape_recyclivre("9781234567890")

        assert result is None
