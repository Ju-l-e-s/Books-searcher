import asyncio
import re
import logging
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}

# Momox France — try multiple known API patterns
MOMOX_API_URLS = [
    "https://www.momox.fr/ajax/sell/additem/?ean={isbn}",
    "https://www.momox.fr/ajax/sell/additem/?isbn={isbn}",
    "https://www.momox.fr/fr_FR/volumes/search/?q={isbn}",
    "https://www.momox.fr/fr/ajax/sell/item/?ean={isbn}",
]
# Keep a single string alias for backward compat with Playwright fallback
MOMOX_API_URL = MOMOX_API_URLS[2]

# RecycLivre reprise (buyback) page — ISBN lookup
RECYCLIVRE_URL = "https://www.recyclivre.com/reprise/?isbn={isbn}"
# Fallback search URL
RECYCLIVRE_SEARCH_URL = "https://www.recyclivre.com/shop/0-0/search/?q={isbn}"


class ValuationScraper:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(3)
        self._client: Optional[httpx.AsyncClient] = None
        self._price_cache: dict[str, Dict[str, float]] = {}
        # Playwright is only used as fallback; keep optional
        self._playwright = None
        self._browser = None

    async def start(self):
        """Initialize the HTTP client pool — call at app startup."""
        self._client = httpx.AsyncClient(
            headers=HEADERS,
            timeout=httpx.Timeout(15.0),
            follow_redirects=True,
        )
        logger.info("HTTP client started.")

    async def stop(self):
        """Close HTTP client and optional Playwright browser — call at app shutdown."""
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
        """Lazily start Playwright only if needed."""
        if self._browser is not None:
            return
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            logger.info("Playwright browser started (fallback).")
        except Exception as e:
            logger.warning("Could not start Playwright: %s", e)

    async def get_prices(self, isbn: str) -> Dict[str, float]:
        """Fetch prices from Momox and RecycLivre for a given ISBN."""
        if isbn in self._price_cache:
            logger.debug("Cache hit for ISBN %s", isbn)
            return self._price_cache[isbn]

        async with self.semaphore:
            momox_task = self._scrape_momox(isbn)
            recyclivre_task = self._scrape_recyclivre(isbn)
            prices_results = await asyncio.gather(momox_task, recyclivre_task, return_exceptions=True)

            p0: float = 0.0
            p1: float = 0.0

            if len(prices_results) > 0:
                val0 = prices_results[0]
                if isinstance(val0, (float, int)):
                    p0 = float(val0)

            if len(prices_results) > 1:
                val1 = prices_results[1]
                if isinstance(val1, (float, int)):
                    p1 = float(val1)

            result = {"momox": p0, "recyclivre": p1}
            self._price_cache[isbn] = result
            return result

    async def _scrape_momox(self, isbn: str) -> Optional[float]:
        """
        Fetch Momox buyback price via their JSON API.
        Falls back to Playwright page scrape if the API fails.
        """
        # --- httpx attempt (try all known API endpoints) ---
        if self._client:
            for url_template in MOMOX_API_URLS:
                try:
                    url = url_template.format(isbn=isbn)
                    resp = await self._client.get(url)
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            # Momox API may return: {"items": [{"buybackPrice": 1.50}]}
                            # or {"buyPrice": 1.50} or {"price": 1.50}
                            items = data.get("items") or data.get("results") or []
                            if items:
                                for item in items:
                                    price = (
                                        item.get("buybackPrice")
                                        or item.get("buyPrice")
                                        or item.get("price")
                                        or item.get("buy_price")
                                    )
                                    if price is not None:
                                        return float(price)
                            else:
                                # Flat response
                                price = (
                                    data.get("buybackPrice")
                                    or data.get("buyPrice")
                                    or data.get("price")
                                )
                                if price is not None:
                                    return float(price)
                        except Exception:
                            # Not JSON — try regex on raw text
                            match = re.search(r'(\d+[.,]\d{2})\s*\u20ac', resp.text)
                            if match:
                                return float(match.group(1).replace(",", "."))
                except Exception as e:
                    logger.debug("Momox httpx request failed (%s) for %s: %s", url_template, isbn, e)

        # --- Playwright fallback ---
        await self._ensure_playwright()
        if self._browser is None:
            return None

        try:
            context = await self._browser.new_context(user_agent=HEADERS["User-Agent"])
            page = await context.new_page()
            try:
                # Try a browseable Momox page before falling back to the API URL
                for url in [
                    f"https://www.momox.fr/livres/{isbn}/",
                    MOMOX_API_URL.format(isbn=isbn),
                ]:
                    try:
                        r = await page.goto(url, timeout=12000, wait_until="domcontentloaded")
                        if r and r.status < 400:
                            await page.wait_for_timeout(1500)
                            break
                    except Exception:
                        continue
                selectors = [
                    ".ScreenMediumItemBuyPrice__price",
                    "[data-testid='price']",
                    ".price",
                    ".price-value",
                    ".offer-price",
                ]
                for selector in selectors:
                    try:
                        el = page.locator(selector).first
                        if await el.count() > 0:
                            price_text = await el.inner_text(timeout=3000)
                            return float(price_text.replace("\u20ac", "").replace(",", ".").strip())
                    except Exception:
                        continue
                try:
                    body_text = await page.inner_text("body", timeout=3000)
                    match = re.search(r'(\d+[.,]\d{2})\s*\u20ac', body_text)
                    if match:
                        return float(match.group(1).replace(",", "."))
                except Exception:
                    pass
                return None
            finally:
                await context.close()
        except Exception as e:
            logger.debug("Momox Playwright fallback failed for %s: %s", isbn, e)
            return None

    async def _scrape_recyclivre(self, isbn: str) -> Optional[float]:
        """
        Fetch RecycLivre buyback price via httpx.
        Falls back to Playwright if needed.
        """
        # --- httpx attempt (reprise page first, then search) ---
        if self._client:
            for url_template in (RECYCLIVRE_URL, RECYCLIVRE_SEARCH_URL):
                try:
                    url = url_template.format(isbn=isbn)
                    resp = await self._client.get(url)
                    if resp.status_code == 200:
                        # Try JSON
                        try:
                            data = resp.json()
                            price = (
                                data.get("buybackPrice")
                                or data.get("price")
                                or data.get("reprise_price")
                            )
                            if price is not None:
                                return float(price)
                        except Exception:
                            pass
                        # Try regex on HTML
                        match = re.search(r'(\d+[.,]\d{2})\s*\u20ac', resp.text)
                        if match:
                            return float(match.group(1).replace(",", "."))
                except Exception as e:
                    logger.debug("RecycLivre httpx request failed (%s) for %s: %s", url_template, isbn, e)

        # --- Playwright fallback ---
        await self._ensure_playwright()
        if self._browser is None:
            return None

        try:
            context = await self._browser.new_context(user_agent=HEADERS["User-Agent"])
            page = await context.new_page()
            try:
                url = RECYCLIVRE_URL.format(isbn=isbn)
                await page.goto(url, timeout=15000)
                selectors = [
                    ".buyback-price",
                    ".product-buyback-price",
                    "[data-buyback-price]",
                    ".price-buyback",
                ]
                for selector in selectors:
                    try:
                        el = page.locator(selector).first
                        if await el.count() > 0:
                            price_text = await el.inner_text(timeout=3000)
                            return float(price_text.replace("\u20ac", "").replace(",", ".").strip())
                    except Exception:
                        continue
                try:
                    body_text = await page.inner_text("body", timeout=3000)
                    match = re.search(r'(\d+[.,]\d{2})\s*\u20ac', body_text)
                    if match:
                        return float(match.group(1).replace(",", "."))
                except Exception:
                    pass
                return None
            finally:
                await context.close()
        except Exception as e:
            logger.debug("RecycLivre Playwright fallback failed for %s: %s", isbn, e)
            return None


# Singleton instance
scraper = ValuationScraper()
