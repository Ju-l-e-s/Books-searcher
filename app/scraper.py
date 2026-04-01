import asyncio
import random
import logging
from functools import lru_cache
from playwright.async_api import async_playwright, Playwright, Browser
from typing import Dict, Optional

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
]


class ValuationScraper:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(3)
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._price_cache: dict[str, Dict[str, float]] = {}

    async def start(self):
        """Start the persistent browser pool — call at app startup."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        logger.info("Playwright browser pool started.")

    async def stop(self):
        """Close browser and playwright — call at app shutdown."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Playwright browser pool stopped.")

    async def get_prices(self, isbn: str) -> Dict[str, float]:
        """Fetch prices from Momox and RecycLivre for a given ISBN."""
        # Check cache first
        if isbn in self._price_cache:
            logger.debug("Cache hit for ISBN %s", isbn)
            return self._price_cache[isbn]

        async with self.semaphore:
            if self._browser is None:
                logger.error("Browser not started. Call scraper.start() first.")
                return {"momox": 0.0, "recyclivre": 0.0}

            momox_task = self._scrape_momox(isbn)
            recyclivre_task = self._scrape_recyclivre(isbn)

            # Using return_exceptions=True to handle possible individual scrape failures
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

            result = {
                "momox": p0,
                "recyclivre": p1
            }

            # Cache the result
            self._price_cache[isbn] = result
            return result

    async def _scrape_momox(self, isbn: str) -> Optional[float]:
        """Scrape Momox buyback price using the persistent browser."""
        if self._browser is None:
            return None
        
        # Explicitly narrowed for the linter
        browser = self._browser
        context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
        page = await context.new_page()
        try:
            url = f"https://www.momox.fr/media/vendre/isbn/{isbn}/"
            await page.goto(url, timeout=15000)

            # Try multiple selectors that Momox may use
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
                        price = float(
                            price_text.replace("€", "").replace(",", ".").strip()
                        )
                        return price
                except Exception:
                    continue

            # Fallback: search for any €-containing text on the page
            try:
                body_text = await page.inner_text("body", timeout=3000)
                import re
                match = re.search(r'(\d+[.,]\d{2})\s*€', body_text)
                if match:
                    return float(match.group(1).replace(",", "."))
            except Exception:
                pass

            return None
        except Exception as e:
            logger.debug("Momox scraping failed for %s: %s", isbn, e)
            return None
        finally:
            await context.close()

    async def _scrape_recyclivre(self, isbn: str) -> Optional[float]:
        """Scrape RecycLivre buyback price using the persistent browser."""
        if self._browser is None:
            return None

        # Explicitly narrowed for the linter
        browser = self._browser
        context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
        page = await context.new_page()
        try:
            url = f"https://www.recyclivre.com/recherche?q={isbn}"
            await page.goto(url, timeout=15000)

            # Try multiple selectors
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
                        price = float(
                            price_text.replace("€", "").replace(",", ".").strip()
                        )
                        return price
                except Exception:
                    continue

            # Fallback regex
            try:
                body_text = await page.inner_text("body", timeout=3000)
                import re
                match = re.search(r'(\d+[.,]\d{2})\s*€', body_text)
                if match:
                    return float(match.group(1).replace(",", "."))
            except Exception:
                pass

            return None
        except Exception as e:
            logger.debug("RecycLivre scraping failed for %s: %s", isbn, e)
            return None
        finally:
            await context.close()


# Singleton instance
scraper = ValuationScraper()
