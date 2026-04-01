import httpx
from rapidfuzz import fuzz
from typing import List
import asyncio
import logging

logger = logging.getLogger(__name__)


class ISBNResolver:
    GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
    OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        """Properly close the HTTP client — call at app shutdown."""
        await self.client.aclose()
        logger.info("ISBNResolver HTTP client closed.")

    async def resolve(self, query: str) -> List[dict]:
        """Resolve OCR text to its most likely ISBN."""
        if not query or len(query.strip()) < 5:
            return []

        query = query.strip()

        # Run both searches in parallel
        gb_task = self._search_google_books(query)
        ol_task = self._search_open_library(query)

        results = await asyncio.gather(gb_task, ol_task, return_exceptions=True)

        candidates: List[dict] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Search API failed: %s", result)
                continue
            if isinstance(result, list):
                candidates.extend(result)

        # Fuzzy matching score
        for cand in candidates:
            score = fuzz.token_sort_ratio(query.lower(), cand['title'].lower())
            cand['score'] = float(score)

        # Sort by best score
        candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        # Return top 3 candidates by selecting them individually to avoid linter indexing issues
        return [candidates[i] for i in range(len(candidates)) if i < 3]

    async def _search_google_books(self, query: str) -> List[dict]:
        try:
            params = {"q": query, "maxResults": 5}
            resp = await self.client.get(self.GOOGLE_BOOKS_URL, params=params)
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
                if isbns:
                    candidates.append({
                        "title": info.get("title", ""),
                        "author": ", ".join(info.get("authors", [])),
                        "isbn": isbns[0],
                        "source": "google_books"
                    })
            return candidates
        except Exception as e:
            logger.debug("Google Books search failed: %s", e)
            return []

    async def _search_open_library(self, query: str) -> List[dict]:
        try:
            params = {"q": query, "limit": 5}
            resp = await self.client.get(self.OPEN_LIBRARY_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            candidates = []
            for doc in data.get("docs", []):
                isbn_list = doc.get("isbn", [])
                if isbn_list:
                    candidates.append({
                        "title": doc.get("title", ""),
                        "author": ", ".join(doc.get("author_name", [])),
                        "isbn": isbn_list[0],
                        "source": "open_library"
                    })
            return candidates
        except Exception as e:
            logger.debug("Open Library search failed: %s", e)
            return []


# Singleton instance
resolver = ISBNResolver()
