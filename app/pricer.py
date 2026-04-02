"""Pricer — Lambda-ready buyback price fetcher.

Uses httpx only (no Playwright/Chromium).  Both Momox and RecycLivre are
queried in parallel with a hard 4-second timeout to cap Lambda execution cost.
Results are cached in DynamoDB (TTL = 24 h) to avoid redundant HTTP calls.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# ── HTTP headers (mobile UA — lighter bot fingerprint) ────────────────────────
_MOBILE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# ── Momox endpoints (fastest-first) ──────────────────────────────────────────
_MOMOX_APIS = [
    "https://www.momox.fr/ajax/sell/additem/?ean={isbn}",
    "https://www.momox.fr/ajax/sell/additem/?isbn={isbn}",
    "https://www.momox.fr/fr_FR/volumes/search/?q={isbn}",
]

# ── RecycLivre endpoints ──────────────────────────────────────────────────────
_RECYCLIVRE_URLS = [
    "https://www.recyclivre.com/reprise/?isbn={isbn}",
    "https://www.recyclivre.com/shop/0-0/search/?q={isbn}",
]

_PRICE_RE = re.compile(r"(\d+[.,]\d{2})\s*€")

from app.infrastructure.cache import CacheProvider, get_default_cache

# ── cache configuration ───────────────────────────────────────────────────────
_PRICE_TABLE_NAME: Optional[str] = os.environ.get("PRICE_CACHE_TABLE")
_PRICE_TTL = 24 * 3600  # 24 hours


# ── price parsing ─────────────────────────────────────────────────────────────

def _parse_price(text: str) -> Optional[float]:
    match = _PRICE_RE.search(text)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def _extract_from_json(data: dict) -> Optional[float]:
    """Try common buyback-price keys in a JSON response dict."""
    for key in ("buybackPrice", "buyPrice", "price", "buy_price", "reprise_price"):
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    for container_key in ("items", "results"):
        for item in data.get(container_key) or []:
            for key in ("buybackPrice", "buyPrice", "price"):
                val = item.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
    return None


# ── domain: price fetching ────────────────────────────────────────────────────

class Pricer:
    """Fetch Momox + RecycLivre buyback prices via direct httpx requests."""

    def __init__(self, cache: Optional[CacheProvider] = None) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._cache = cache or get_default_cache(_PRICE_TABLE_NAME)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=_MOBILE_HEADERS,
                timeout=httpx.Timeout(4.0),  # strict Lambda timeout
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_prices(self, isbn: str) -> Dict[str, float]:
        """Return {momox, recyclivre} buyback prices. Returns 0.0 when not found."""
        # Cache read
        cached = await asyncio.to_thread(self._cache.get, isbn)
        if cached is not None:
            logger.debug("Price cache hit for ISBN %s", isbn)
            return cached

        # Run fetchers in parallel with a hard timeout
        momox_task = asyncio.create_task(self._fetch_momox(isbn))
        recyc_task = asyncio.create_task(self._fetch_recyclivre(isbn))

        done, pending = await asyncio.wait(
            [momox_task, recyc_task],
            timeout=4.0
        )

        for task in pending:
            task.cancel()
            logger.warning("Price fetch timed out for ISBN %s", isbn)

        momox_price = 0.0
        if momox_task in done:
            try:
                momox_price = momox_task.result() or 0.0
            except Exception as exc:
                logger.error("Momox fetch failed: %s", exc)

        recyclivre_price = 0.0
        if recyc_task in done:
            try:
                recyclivre_price = recyc_task.result() or 0.0
            except Exception as exc:
                logger.error("RecycLivre fetch failed: %s", exc)

        result: Dict[str, float] = {
            "momox": float(momox_price),
            "recyclivre": float(recyclivre_price),
        }

        # Cache only if we got at least one non-zero price (optional choice)
        # or cache always to avoid retrying if both are zero.
        await asyncio.to_thread(self._cache.set, isbn, result, _PRICE_TTL)
        return result

    async def _fetch_momox(self, isbn: str) -> Optional[float]:
        client = self._get_client()
        for url_tpl in _MOMOX_APIS:
            try:
                # Add referer to look more like a real browser navigation
                headers = {"Referer": "https://www.momox.fr/fr_FR/volumes/sell/"}
                resp = await client.get(url_tpl.format(isbn=isbn), headers=headers)
                if resp.status_code != 200:
                    continue
                try:
                    price = _extract_from_json(resp.json())
                    if price is not None:
                        return price
                except Exception:
                    pass
                price = _parse_price(resp.text)
                if price is not None:
                    return price
            except Exception as exc:
                logger.debug("Momox request failed (%s) for %s: %s", url_tpl, isbn, exc)
        return None

    async def _fetch_recyclivre(self, isbn: str) -> Optional[float]:
        client = self._get_client()
        for url_tpl in _RECYCLIVRE_URLS:
            try:
                headers = {"Referer": "https://www.recyclivre.com/reprise/"}
                resp = await client.get(url_tpl.format(isbn=isbn), headers=headers)
                if resp.status_code != 200:
                    continue
                try:
                    price = _extract_from_json(resp.json())
                    if price is not None:
                        return price
                except Exception:
                    pass
                price = _parse_price(resp.text)
                if price is not None:
                    return price
            except Exception as exc:
                logger.debug("RecycLivre request failed (%s) for %s: %s", url_tpl, isbn, exc)
        return None


# Singleton instance
pricer = Pricer()
