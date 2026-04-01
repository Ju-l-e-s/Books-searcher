import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.scraper import ValuationScraper


class TestValuationScraper:
    """Tests for ValuationScraper — mocked Playwright/browser."""

    @pytest.fixture
    def scraper(self):
        return ValuationScraper()

    async def test_get_prices_uses_cache(self, scraper):
        """If ISBN is cached, no browser call should be made."""
        scraper._browser = None  # would crash if called
        scraper._price_cache["9781234567890"] = {"momox": 7.5, "recyclivre": 4.0}
        result = await scraper.get_prices("9781234567890")
        assert result == {"momox": 7.5, "recyclivre": 4.0}

    async def test_get_prices_no_browser_returns_zeroes(self, scraper):
        """get_prices should return zero prices if browser not started."""
        scraper._browser = None
        scraper._price_cache = {}
        result = await scraper.get_prices("9781234567890")
        assert result == {"momox": 0.0, "recyclivre": 0.0}

    async def test_start_launches_chromium(self, scraper):
        """start() should launch a Chromium browser in headless mode."""
        mock_browser = MagicMock()
        mock_started_pw = MagicMock()
        mock_started_pw.chromium = MagicMock()
        mock_started_pw.chromium.launch = AsyncMock(return_value=mock_browser)

        mock_playwright = MagicMock()
        mock_playwright.start = AsyncMock(return_value=mock_started_pw)

        scraper._browser = None
        scraper._playwright = None

        with patch("app.scraper.async_playwright", return_value=mock_playwright):
            await scraper.start()

        mock_started_pw.chromium.launch.assert_called_once_with(headless=True)
        assert scraper._browser is mock_browser

    async def test_stop_closes_browser_and_playwright(self, scraper):
        """stop() should cleanly close browser and playwright."""
        mock_browser = AsyncMock()
        mock_playwright = AsyncMock()
        scraper._browser = mock_browser
        scraper._playwright = mock_playwright

        await scraper.stop()

        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()
        assert scraper._browser is None
        assert scraper._playwright is None

    async def test_scrape_momox_returns_price(self, scraper):
        """_scrape_momox should return price when selector matches."""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        # Chain: locator(selector).first.count() → 1, inner_text() → "5,90 €"
        price_locator = AsyncMock()
        price_locator.count.return_value = 1
        price_locator.inner_text = AsyncMock(return_value="5,90 €")
        # locator().first returns the same locator
        price_locator.first = price_locator

        mock_page.locator = MagicMock(return_value=price_locator)
        mock_page.inner_text = AsyncMock(return_value="some text")

        scraper._browser = mock_browser
        result = await scraper._scrape_momox("9781234567890")

        assert result == 5.90
        mock_context.close.assert_called_once()

    async def test_scrape_momox_fallback_regex(self, scraper):
        """_scrape_momox should fall back to body regex when no selector matches."""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        empty_locator = AsyncMock()
        empty_locator.count.return_value = 0
        empty_locator.first = empty_locator
        mock_page.locator = MagicMock(return_value=empty_locator)
        mock_page.inner_text = AsyncMock(return_value="Buyback price: 12,50 €")

        scraper._browser = mock_browser
        result = await scraper._scrape_momox("9781234567890")

        assert result == 12.50
        mock_context.close.assert_called_once()

    async def test_scrape_recyclivre_returns_price(self, scraper):
        """_scrape_recyclivre should return price when selector matches."""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        price_locator = AsyncMock()
        price_locator.count.return_value = 1
        price_locator.inner_text = AsyncMock(return_value="8,75 €")
        price_locator.first = price_locator

        mock_page.locator = MagicMock(return_value=price_locator)
        mock_page.inner_text = AsyncMock(return_value="recyclivre")

        scraper._browser = mock_browser
        result = await scraper._scrape_recyclivre("9781234567890")

        assert result == 8.75
        mock_context.close.assert_called_once()

    async def test_scrape_momox_no_browser_returns_none(self, scraper):
        """_scrape_momox should return None gracefully if no browser."""
        scraper._browser = None
        result = await scraper._scrape_momox("9781234567890")
        assert result is None

    async def test_scrape_recyclivre_no_browser_returns_none(self, scraper):
        """_scrape_recyclivre should return None gracefully if no browser."""
        scraper._browser = None
        result = await scraper._scrape_recyclivre("9781234567890")
        assert result is None

    async def test_get_prices_caches_after_scrape(self, scraper):
        """After scraping, result should be cached."""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        price_locator = AsyncMock()
        price_locator.count.return_value = 1
        price_locator.inner_text = AsyncMock(side_effect=["5,00 €", "3,00 €"])
        price_locator.first = price_locator

        mock_page.locator = MagicMock(return_value=price_locator)
        mock_page.inner_text = AsyncMock(return_value="text")

        scraper._browser = mock_browser
        isbn = "978-new-isbn-123"
        result = await scraper.get_prices(isbn)

        assert isbn in scraper._price_cache
        assert scraper._price_cache[isbn] == result

    async def test_scrape_momox_handles_page_exception(self, scraper):
        """_scrape_momox should return None if page.goto raises."""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("Navigation error"))
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        scraper._browser = mock_browser
        result = await scraper._scrape_momox("9781234567890")

        assert result is None
