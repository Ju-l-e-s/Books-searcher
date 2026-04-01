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
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
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

# ── DynamoDB cache (infrastructure layer) ─────────────────────────────────────
_PRICE_TABLE_NAME: Optional[str] = os.environ.get("PRICE_CACHE_TABLE")
_PRICE_TTL = 24 * 3600  # 24 hours


class _DynamoCache:
    def __init__(self, table_name: Optional[str]) -> None:
        self._table_name = table_name
        self._table: Any = None

    def _get_table(self) -> Any:
        if self._table is None and self._table_name:
            try:
                import boto3
                self._table = boto3.resource("dynamodb").Table(self._table_name)
            except Exception as exc:
                logger.debug("DynamoDB unavailable: %s", exc)
        return self._table

    def get(self, key: str) -> Optional[Any]:
        table = self._get_table()
        if not table:
            return None
        try:
            resp = table.get_item(Key={"pk": key})
            item = resp.get("Item")
            if item and int(item.get("ttl", 0)) > int(time.time()):
                return json.loads(item["value"])
        except Exception as exc:
            logger.debug("DynamoDB get error: %s", exc)
        return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        table = self._get_table()
        if not table:
            return
        try:
            table.put_item(Item={
                "pk": key,
                "value": json.dumps(value),
                "ttl": int(time.time()) + ttl_seconds,
            })
        except Exception as exc:
            logger.debug("DynamoDB put error: %s", exc)


_price_cache = _DynamoCache(_PRICE_TABLE_NAME)


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

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None

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
        # Cache read (sync boto3 in thread)
        cached = await asyncio.to_thread(_price_cache.get, isbn)
        if cached is not None:
            logger.debug("Price cache hit for ISBN %s", isbn)
            return cached

        try:
            momox_price, recyclivre_price = await asyncio.wait_for(
                asyncio.gather(
                    self._fetch_momox(isbn),
                    self._fetch_recyclivre(isbn),
                    return_exceptions=True,
                ),
                timeout=4.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Price fetch timed out for ISBN %s", isbn)
            momox_price = None
            recyclivre_price = None

        result: Dict[str, float] = {
            "momox": float(momox_price) if isinstance(momox_price, (int, float)) else 0.0,
            "recyclivre": float(recyclivre_price) if isinstance(recyclivre_price, (int, float)) else 0.0,
        }

        await asyncio.to_thread(_price_cache.set, isbn, result, _PRICE_TTL)
        return result

    async def _fetch_momox(self, isbn: str) -> Optional[float]:
        client = self._get_client()
        for url_tpl in _MOMOX_APIS:
            try:
                resp = await client.get(url_tpl.format(isbn=isbn))
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
                resp = await client.get(url_tpl.format(isbn=isbn))
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
