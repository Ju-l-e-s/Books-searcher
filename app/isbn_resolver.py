"""ISBN Resolver — weighted triangulation engine (Lambda-ready).

Scoring breakdown (max ~100 pts before penalties):
  - Title  30 %  : fuzz.token_set_ratio vs OCR text
  - Author 20 %  : +60 raw bonus when author token found in OCR
  - Publisher 40%: keyword match (Gallimard, Folio, Pocket …)
  - Format 10 %  : bbox w/h < 0.1 → "Poche" hint

Anti-noise penalties applied after scoring:
  - -80 pts for study/analysis titles ("Fiche de lecture", "Analyse", …)
  - Penalise year editions (ex. "Larousse 2012") when OCR has no year
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Optional

import httpx
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ── publisher keywords → discriminate editions ────────────────────────────────
PUBLISHER_KEYWORDS = {
    "gallimard", "folio", "pocket", "flammarion", "hachette", "grasset",
    "seuil", "actes sud", "albin michel", "calmann", "fayard", "plon",
    "robert laffont", "stock", "minuit", "denoël", "le livre de poche",
    "j'ai lu", "points", "babel", "rivages",
}

# Anti-noise title tokens — penalise these candidates
_NOISE_TOKENS = re.compile(
    r"\b(fiche\s+de\s+lecture|analyse|étude|bd|illustré)\b", re.IGNORECASE
)

# Year pattern for anti-millesime check
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# ── DynamoDB cache (domain-level, optional) ───────────────────────────────────
_ISBN_TABLE_NAME: Optional[str] = os.environ.get("ISBN_CACHE_TABLE")
_ISBN_TTL = 30 * 24 * 3600  # 30 days


class _DynamoCache:
    """Thin, fault-tolerant DynamoDB wrapper (synchronous boto3)."""

    def __init__(self, table_name: Optional[str]) -> None:
        self._table_name = table_name
        self._table: Any = None

    def _get_table(self) -> Any:
        if self._table is None and self._table_name:
            try:
                import boto3  # imported lazily — not available in tests
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
                "value": json.dumps(value, ensure_ascii=False),
                "ttl": int(time.time()) + ttl_seconds,
            })
        except Exception as exc:
            logger.debug("DynamoDB put error: %s", exc)


_isbn_cache = _DynamoCache(_ISBN_TABLE_NAME)


# ── scoring helpers ───────────────────────────────────────────────────────────

def _score_title(ocr: str, candidate_title: str) -> float:
    """30 % weight: fuzz.token_set_ratio, returns 0-100."""
    return fuzz.token_set_ratio(ocr.lower(), candidate_title.lower())


def _score_author(ocr: str, candidate_author: str, primary_author: Optional[str]) -> float:
    """20 % weight: +60 raw pts when any author token appears in OCR.

    If a primary author is known, secondary authors (illustrateurs, préfaciers)
    are ignored — only the primary author is checked.
    """
    author_to_check = primary_author if primary_author else candidate_author
    if not author_to_check:
        return 0.0
    # Split on common separators; check each token
    tokens = re.split(r"[\s,.\-]+", author_to_check.lower())
    tokens = [t for t in tokens if len(t) >= 3]
    ocr_lower = ocr.lower()
    if any(t in ocr_lower for t in tokens):
        return 60.0
    return 0.0


def _score_publisher(ocr: str, candidate_publisher: str) -> float:
    """40 % weight: binary keyword match, returns 0 or 100."""
    combined = (ocr + " " + candidate_publisher).lower()
    if any(kw in combined for kw in PUBLISHER_KEYWORDS):
        return 100.0
    return 0.0


def _score_format(bbox_ratio: Optional[float]) -> float:
    """10 % weight: w/h < 0.1 → 'Poche' format, returns 0 or 100."""
    if bbox_ratio is not None and bbox_ratio < 0.1:
        return 100.0
    return 0.0


def _apply_penalties(score: float, candidate_title: str, ocr: str) -> float:
    """Apply anti-noise and anti-millesime deductions."""
    # Anti-noise: study / analysis editions
    if _NOISE_TOKENS.search(candidate_title):
        score -= 80.0

    # Anti-millesime: penalise year editions when OCR has no year
    if _YEAR_RE.search(candidate_title) and not _YEAR_RE.search(ocr):
        score -= 40.0

    return score


def _compute_score(
    ocr: str,
    candidate: dict,
    primary_author: Optional[str],
    bbox_ratio: Optional[float],
) -> float:
    title = candidate.get("title", "")
    author = candidate.get("author", "")
    publisher = candidate.get("publisher", "")

    ts = _score_title(ocr, title) * 0.30
    as_ = _score_author(ocr, author, primary_author) * 0.20
    ps = _score_publisher(ocr, publisher) * 0.40
    fs = _score_format(bbox_ratio) * 0.10

    raw = ts + as_ + ps + fs
    return _apply_penalties(raw, title, ocr)


def _detect_primary_author(ocr: str, candidates: list[dict]) -> Optional[str]:
    """Return the first author whose tokens appear in the OCR text.

    Used to suppress secondary authors (illustrateurs, préfaciers).
    """
    for cand in candidates:
        author = cand.get("author", "")
        tokens = re.split(r"[\s,.\-]+", author.lower())
        tokens = [t for t in tokens if len(t) >= 3]
        if tokens and any(t in ocr.lower() for t in tokens):
            return author
    return None


# ── external API helpers ──────────────────────────────────────────────────────

class ISBNResolver:
    GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
    OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        await self.client.aclose()
        logger.info("ISBNResolver HTTP client closed.")

    async def resolve(
        self,
        query: str,
        bbox_ratio: Optional[float] = None,
    ) -> list[dict]:
        """Resolve OCR text to the most likely ISBN using weighted scoring."""
        if not query or len(query.strip()) < 5:
            return []

        query = query.strip()
        cache_key = hashlib.sha256(query.encode()).hexdigest()

        # Cache read (run sync boto3 in thread to avoid blocking event loop)
        cached = await asyncio.to_thread(_isbn_cache.get, cache_key)
        if cached is not None:
            logger.debug("ISBN cache hit for query hash %s", cache_key[:8])
            return cached

        # Parallel API queries
        gb_task = self._search_google_books(query)
        ol_task = self._search_open_library(query)
        results = await asyncio.gather(gb_task, ol_task, return_exceptions=True)

        candidates: list[dict] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Search API failed: %s", result)
                continue
            if isinstance(result, list):
                candidates.extend(result)

        if not candidates:
            return []

        # Detect primary author to suppress secondary authors
        primary_author = _detect_primary_author(query, candidates)

        # Weighted scoring + penalties
        for cand in candidates:
            cand["score"] = _compute_score(query, cand, primary_author, bbox_ratio)

        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top = candidates[:3]

        # Cache write
        await asyncio.to_thread(_isbn_cache.set, cache_key, top, _ISBN_TTL)

        return top

    async def _search_google_books(self, query: str) -> list[dict]:
        try:
            resp = await self.client.get(
                self.GOOGLE_BOOKS_URL, params={"q": query, "maxResults": 5}
            )
            resp.raise_for_status()
            data = resp.json()
            candidates = []
            for item in data.get("items", []):
                info = item.get("volumeInfo", {})
                isbns = [
                    ident["identifier"]
                    for ident in info.get("industryIdentifiers", [])
                    if ident.get("type") in ("ISBN_13", "ISBN_10")
                ]
                if not isbns:
                    continue
                candidates.append({
                    "title": info.get("title", ""),
                    "author": ", ".join(info.get("authors", [])),
                    "publisher": info.get("publisher", ""),
                    "isbn": isbns[0],
                    "source": "google_books",
                })
            return candidates
        except Exception as exc:
            logger.debug("Google Books search failed: %s", exc)
            return []

    async def _search_open_library(self, query: str) -> list[dict]:
        try:
            resp = await self.client.get(
                self.OPEN_LIBRARY_URL, params={"q": query, "limit": 5}
            )
            resp.raise_for_status()
            data = resp.json()
            candidates = []
            for doc in data.get("docs", []):
                isbn_list = doc.get("isbn", [])
                if not isbn_list:
                    continue
                candidates.append({
                    "title": doc.get("title", ""),
                    "author": ", ".join(doc.get("author_name", [])),
                    "publisher": ", ".join(doc.get("publisher", [])),
                    "isbn": isbn_list[0],
                    "source": "open_library",
                })
            return candidates
        except Exception as exc:
            logger.debug("Open Library search failed: %s", exc)
            return []


# Singleton instance
resolver = ISBNResolver()
