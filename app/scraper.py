import asyncio
import re
import logging
from typing import Dict, Optional

import httpx
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}

# Momox France — known JSON API endpoints (fastest path)
MOMOX_API_URLS = [
    "https://www.momox.fr/ajax/sell/additem/?ean={isbn}",
    "https://www.momox.fr/ajax/sell/additem/?isbn={isbn}",
    "https://www.momox.fr/fr_FR/volumes/search/?q={isbn}",
]
# Playwright fallback page
MOMOX_PAGE_URL = "https://www.momox.fr/livres/{isbn}/"

# RecycLivre reprise (buyback) pages
RECYCLIVRE_URL = "https://www.recyclivre.com/reprise/?isbn={isbn}"
RECYCLIVRE_SEARCH_URL = "https://www.recyclivre.com/shop/0-0/search/?q={isbn}"

# CSS selectors tried in order for each site
MOMOX_SELECTORS = [
    ".ScreenMediumItemBuyPrice__price",
    "[data-testid='buy-price']",
    "[data-testid='price']",
    ".buyback-price",
    ".price",
]
RECYCLIVRE_SELECTORS = [
    ".buyback-price",
    ".product-buyback-price",
    "[data-buyback-price]",
    ".price-buyback",
    ".reprise-price",
    ".price",
]

_PRICE_RE = re.compile(r"(\d+[.,]\d{2})\s*€")


def _parse_price(text: str) -> Optional[float]:
    """Extract the first euro price from a text string."""
    match = _PRICE_RE.search(text)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


class ValuationScraper:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(3)
        self._client: Optional[httpx.AsyncClient] = None
        self._price_cache: dict[str, Dict[str, float]] = {}
        self._playwright = None
        self._browser = None

    async def start(self):
        """Initialize the HTTP client and Playwright browser."""
        self._client = httpx.AsyncClient(
            headers=HEADERS,
            timeout=httpx.Timeout(15.0),
            follow_redirects=True,
        )
        await self._ensure_playwright()
        logger.info("Scraper started.")

    async def stop(self):
        """Close HTTP client and Playwright browser."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Scraper stopped.")

    async def _ensure_playwright(self):
        """Start a headless Chromium browser if not already running."""
        if self._browser is not None:
            return
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            logger.info("Playwright browser started.")
        except Exception as exc:
            logger.warning("Could not start Playwright: %s", exc)

    async def get_prices(self, isbn: str) -> Dict[str, float]:
        """Return Momox and RecycLivre buyback prices for the given ISBN.

        Results are cached in-memory so each ISBN is only fetched once per
        scraper lifetime.
        """
        if isbn in self._price_cache:
            logger.debug("Cache hit for ISBN %s", isbn)
            return self._price_cache[isbn]

        async with self.semaphore:
            momox_price, recyclivre_price = await asyncio.gather(
                self._scrape_momox(isbn),
                self._scrape_recyclivre(isbn),
                return_exceptions=True,
            )

            result = {
                "momox": float(momox_price) if isinstance(momox_price, (int, float)) else 0.0,
                "recyclivre": float(recyclivre_price) if isinstance(recyclivre_price, (int, float)) else 0.0,
            }
            self._price_cache[isbn] = result
            return result

    async def _scrape_momox(self, isbn: str) -> Optional[float]:
        """Fetch Momox buyback price.

        Strategy:
        1. Try each known JSON API endpoint via httpx (fast, no browser).
        2. If all fail, navigate to the Momox book page with Playwright.
        3. Try a list of CSS selectors; fall back to a body-text regex.
        """
        # --- HTTP API attempt ---
        if self._client:
            for url_tpl in MOMOX_API_URLS:
                try:
                    resp = await self._client.get(url_tpl.format(isbn=isbn))
                    if resp.status_code != 200:
                        continue
                    try:
                        data = resp.json()
                        # Flat response
                        for key in ("buybackPrice", "buyPrice", "price", "buy_price"):
                            if data.get(key) is not None:
                                return float(data[key])
                        # Nested list response
                        for item in data.get("items") or data.get("results") or []:
                            for key in ("buybackPrice", "buyPrice", "price"):
                                if item.get(key) is not None:
                                    return float(item[key])
                    except Exception:
                        price = _parse_price(resp.text)
                        if price is not None:
                            return price
                except Exception as exc:
                    logger.debug("Momox HTTP request failed (%s) for %s: %s", url_tpl, isbn, exc)

        # --- Playwright fallback ---
        if self._browser is None:
            return None

        try:
            context = await self._browser.new_context(user_agent=HEADERS["User-Agent"])
            page = await context.new_page()
            try:
                await page.goto(
                    MOMOX_PAGE_URL.format(isbn=isbn),
                    timeout=15000,
                    wait_until="domcontentloaded",
                )
                await page.wait_for_timeout(1500)

                for selector in MOMOX_SELECTORS:
                    try:
                        el = page.locator(selector).first
                        if await el.count() > 0:
                            text = await el.inner_text(timeout=3000)
                            price = _parse_price(text)
                            if price is not None:
                                return price
                    except Exception:
                        continue

                body = await page.inner_text("body", timeout=5000)
                return _parse_price(body)
            finally:
                await context.close()
        except Exception as exc:
            logger.debug("Momox Playwright failed for %s: %s", isbn, exc)

        return None

    async def _scrape_recyclivre(self, isbn: str) -> Optional[float]:
        """Fetch RecycLivre buyback price.

        Strategy:
        1. Try the reprise page and search page via httpx.
        2. If both fail, navigate with Playwright.
        3. Try CSS selectors; fall back to body-text regex.
        """
        # --- HTTP attempt ---
        if self._client:
            for url_tpl in (RECYCLIVRE_URL, RECYCLIVRE_SEARCH_URL):
                try:
                    resp = await self._client.get(url_tpl.format(isbn=isbn))
                    if resp.status_code != 200:
                        continue
                    try:
                        data = resp.json()
                        for key in ("buybackPrice", "price", "reprise_price"):
                            if data.get(key) is not None:
                                return float(data[key])
                    except Exception:
                        pass
                    price = _parse_price(resp.text)
                    if price is not None:
                        return price
                except Exception as exc:
                    logger.debug("RecycLivre HTTP request failed (%s) for %s: %s", url_tpl, isbn, exc)

        # --- Playwright fallback ---
        if self._browser is None:
            return None

        try:
            context = await self._browser.new_context(user_agent=HEADERS["User-Agent"])
            page = await context.new_page()
            try:
                await page.goto(
                    RECYCLIVRE_URL.format(isbn=isbn),
                    timeout=15000,
                    wait_until="domcontentloaded",
                )
                await page.wait_for_timeout(1500)

                for selector in RECYCLIVRE_SELECTORS:
                    try:
                        el = page.locator(selector).first
                        if await el.count() > 0:
                            text = await el.inner_text(timeout=3000)
                            price = _parse_price(text)
                            if price is not None:
                                return price
                    except Exception:
                        continue

                body = await page.inner_text("body", timeout=5000)
                return _parse_price(body)
            finally:
                await context.close()
        except Exception as exc:
            logger.debug("RecycLivre Playwright failed for %s: %s", isbn, exc)

        return None


# Singleton instance
scraper = ValuationScraper()
