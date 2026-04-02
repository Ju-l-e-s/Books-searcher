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

# Secondary author tokens to suppress/penalize
_SECONDARY_AUTHOR_RE = re.compile(
    r"\b(illustrateur|illustration|préface|introduction|postface|collectif)\b", re.IGNORECASE
)

# Year pattern for anti-millesime check
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens of length >= 3."""
    tokens = re.split(r"[\s,.\-]+", text.lower())
    return [t for t in tokens if len(t) >= 3]


from app.infrastructure.cache import CacheProvider, get_default_cache

# ── cache configuration ───────────────────────────────────────────────────────
_ISBN_TABLE_NAME: Optional[str] = os.environ.get("ISBN_CACHE_TABLE")
_ISBN_TTL = 30 * 24 * 3600  # 30 days


# ── scoring helpers ───────────────────────────────────────────────────────────

def _score_title(ocr: str, candidate_title: str) -> float:
    """30 % weight: fuzz.token_set_ratio, returns 0-100."""
    return fuzz.token_set_ratio(ocr.lower(), candidate_title.lower())


def _score_author(ocr: str, candidate_author: str, primary_author: Optional[str]) -> float:
    """20 % weight: +60 raw pts when any author token appears in OCR.

    If a primary author is known, secondary authors (illustrateurs, préfaciers)
    are ignored — only the primary author is checked.
    """
    ocr_lower = ocr.lower()
    cand_author_lower = candidate_author.lower()
    
    bonus = 0.0
    if primary_author:
        # Check if this candidate's author matches the detected primary author
        primary_tokens = _tokenize(primary_author)
        if any(t in cand_author_lower for t in primary_tokens):
            bonus = 60.0
    else:
        # Direct check
        tokens = _tokenize(cand_author_lower)
        if any(t in ocr_lower for t in tokens):
            bonus = 60.0

    # Suppress/penalize secondary authors
    if bonus > 0 and _SECONDARY_AUTHOR_RE.search(cand_author_lower):
        bonus -= 20.0  # Apply a penalty to secondary-author heavy candidates

    return bonus


def _score_publisher(ocr: str, candidate_publisher: str) -> float:
    """40 % weight: binary keyword match, returns 0 or 100."""
    ocr_lower = ocr.lower()
    cand_pub_lower = candidate_publisher.lower()
    # Both OCR and candidate must share a publisher keyword
    for kw in PUBLISHER_KEYWORDS:
        if kw in cand_pub_lower and kw in ocr_lower:
            return 100.0
    return 0.0


def _score_format(bbox_ratio: Optional[float], dimensions: Optional[dict] = None) -> float:
    """10 % weight: w/h ratio matching or 'Poche' hint.
    
    If dimensions are available, we triangulate by comparing the ratio.
    Otherwise, we fallback to a generic 'Poche' hint for narrow spines.
    """
    if bbox_ratio is None:
        return 0.0

    # 1. Triangulation if dimensions are available (height & width)
    if dimensions and dimensions.get("height") and dimensions.get("width"):
        # Real physical ratio (width / height)
        # Note: bbox_ratio from YOLO is also width/height (approx)
        cand_ratio = float(dimensions["width"]) / float(dimensions["height"])
        
        # 20% tolerance for perspective distortion
        if 0.8 * cand_ratio <= bbox_ratio <= 1.2 * cand_ratio:
            return 100.0

    # 2. Fallback: narrow spine (< 0.1) likely means "Poche" format
    if bbox_ratio < 0.1:
        return 80.0  # Slightly less than an exact dimension match

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
    dimensions = candidate.get("dimensions")

    ts = _score_title(ocr, title) * 0.30
    as_ = _score_author(ocr, author, primary_author) * 0.20
    ps = _score_publisher(ocr, publisher) * 0.40
    fs = _score_format(bbox_ratio, dimensions) * 0.10

    raw = ts + as_ + ps + fs
    return _apply_penalties(raw, title, ocr)


def _detect_primary_author(ocr: str, candidates: list[dict]) -> Optional[str]:
    """Return the first author whose tokens appear in the OCR text.

    Used to suppress secondary authors (illustrateurs, préfaciers).
    """
    ocr_lower = ocr.lower()
    fallback_author = None
    
    for cand in candidates:
        author = cand.get("author", "")
        if not author:
            continue
            
        tokens = _tokenize(author)
        if tokens and any(t in ocr_lower for t in tokens):
            if not _SECONDARY_AUTHOR_RE.search(author.lower()):
                # Found a "clean" primary author
                return author
            if fallback_author is None:
                # Keep first matching secondary author as fallback
                fallback_author = author
                
    return fallback_author


# ── external API helpers ──────────────────────────────────────────────────────

class ISBNResolver:
    GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
    OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"

    def __init__(self, cache: Optional[CacheProvider] = None) -> None:
        self.client = httpx.AsyncClient(timeout=10.0)
        self._cache = cache or get_default_cache(_ISBN_TABLE_NAME)

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

        # Cache read
        cached = await asyncio.to_thread(self._cache.get, cache_key)
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
        await asyncio.to_thread(self._cache.set, cache_key, top, _ISBN_TTL)

        return top

    async def _search_google_books(self, query: str) -> list[dict]:
        """Search Google Books for the given query."""
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
                
                # Extract dimensions
                dims_info = info.get("dimensions", {})
                dims = {
                    "height": self._parse_dim(dims_info.get("height")),
                    "width": self._parse_dim(dims_info.get("width")),
                    "thickness": self._parse_dim(dims_info.get("thickness")),
                } if dims_info else None

                candidates.append({
                    "title": info.get("title", ""),
                    "author": ", ".join(info.get("authors", [])),
                    "publisher": info.get("publisher", ""),
                    "year": info.get("publishedDate", "")[:4],
                    "isbn": isbns[0],
                    "source": "google_books",
                    "cover_url": info.get("imageLinks", {}).get("thumbnail"),
                    "dimensions": dims,
                })
            return candidates
        except Exception as exc:
            logger.debug("Google Books search failed: %s", exc)
            return []

    @staticmethod
    def _parse_dim(dim_str: Optional[str]) -> Optional[float]:
        """Extract numeric dimension from string (e.g. '18.0 cm')."""
        if not dim_str: return None
        # Extract numbers from strings like "18.0 cm"
        match = re.search(r"(\d+[.,]?\d*)", dim_str)
        if match:
            return float(match.group(1).replace(",", "."))
        return None

    async def _search_open_library(self, query: str) -> list[dict]:
        """Search Open Library for the given query."""
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
                
                isbn = isbn_list[0]
                candidates.append({
                    "title": doc.get("title", ""),
                    "author": ", ".join(doc.get("author_name", [])),
                    "publisher": ", ".join(doc.get("publisher", [])),
                    "year": str(doc.get("first_publish_year", "")),
                    "isbn": isbn,
                    "source": "open_library",
                    "cover_url": f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg",
                    "dimensions": None,
                })
            return candidates
        except Exception as exc:
            logger.debug("Open Library search failed: %s", exc)
            return []


# Singleton instance
resolver = ISBNResolver()
