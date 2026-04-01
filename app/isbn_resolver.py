import httpx
from rapidfuzz import fuzz
from typing import List
import asyncio
import logging
import re
import os
from dotenv import load_dotenv

# Charge les variables depuis le fichier .env
load_dotenv()

logger = logging.getLogger(__name__)


class ISBNResolver:
    GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
    OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
        self.api_key = os.getenv("GOOGLE_BOOKS_API_KEY")

    async def close(self):
        """Properly close the HTTP client — call at app shutdown."""
        await self.client.aclose()
        logger.info("ISBNResolver HTTP client closed.")

    async def resolve(self, query: str) -> List[dict]:
        """Resolve OCR text to its most likely ISBN."""
        if not query or len(query.strip()) < 4:
            return []

        # 1. Clean query for search
        # Remove common OCR noise, artifacts, and punctuation at start/end
        clean_query = query.upper()
        clean_query = clean_query.replace("LIVRE DE POCHE", "").replace("LIVRE POCHE", "")
        # Remove leading/trailing non-alphanumeric chars (like ~ or 3)
        clean_query = re.sub(r'^[^A-Z0-9]+', '', clean_query)
        clean_query = re.sub(r'[^A-Z0-9]+$', '', clean_query)
        # Remove leading/trailing digits followed/preceded by space (OCR artifacts like '3 ')
        clean_query = re.sub(r'^\s*\d+\s+', '', clean_query)
        clean_query = re.sub(r'\s+\d+\s*$', '', clean_query)
        clean_query = clean_query.strip()
        
        if not clean_query:
            return []
            
        search_query = clean_query
        print(f"DEBUG: Recherche ISBN pour: '{search_query}'")

        # 2. Parallel API calls
        gb_task = self._search_google_books(search_query)
        ol_task = self._search_open_library(search_query)

        results = await asyncio.gather(gb_task, ol_task, return_exceptions=True)

        candidates: List[dict] = []
        for result in results:
            if isinstance(result, Exception):
                print(f"DEBUG: Erreur API (Grave): {result}")
                continue
            if isinstance(result, list):
                candidates.extend(result)

        if not candidates:
            print(f"DEBUG: Aucun candidat trouvé par l'API pour '{search_query}'")
            return []

        for cand in candidates:
            # 1. Clean title for scoring and display
            t = cand.get('title', '')
            # Remove subtitles and study guide notes (anything after -, :, ()
            t = re.sub(r'[\(\-\:].*', '', t)
            # Remove specific noise words
            for word in ["édition", "illustré", "illustre", "richement", "complet", "intégrale"]:
                t = re.compile(re.escape(word), re.IGNORECASE).sub("", t)

            # Smart author stripping from title
            # We KEEP the author if the title is "Journal d'..." or "Mémoires de..."
            # Otherwise, we strip "de [Author]" or "par [Author]" at the end.
            t_lower = t.lower().strip()
            is_protected = t_lower.startswith("journal") or t_lower.startswith("mémoire") or t_lower.startswith("memoire")
            
            if not is_protected:
                author_raw = str(cand.get('author', ''))
                for part in author_raw.split(','):
                    part = part.strip()
                    if len(part) > 3:
                        # Strip " de Author", " par Author", etc. at the end
                        t = re.sub(re.escape(part), '', t, flags=re.IGNORECASE)
                        t = re.sub(r"['\s]+(de|par|d'|d’)\s*$", "", t, flags=re.IGNORECASE)

            t = t.strip().strip("'").strip("’")
            title_clean = t.lower()

            # Clean authors
            authors_str = str(cand.get('author', ''))
            authors_list = [re.sub(r'[,\s]+$', '', a).strip() for a in authors_str.split(',')]
            
            # Blacklist of study-guide/analysis authors and publishers
            analysis_junk = ["lepetitlitteraire", "fichesdelecture", "brumont", "noiret", "viteux", "leloup", "ramain", "millot", "ozanam"]
            authors_list = [a for a in authors_list if a and a.lower().replace(" ", "") not in analysis_junk]
            
            # If the query contains a specific author name, only keep that one in the list
            query_lower = clean_query.lower()
            matching_authors = []
            for a in authors_list:
                a_parts = [p.lower() for p in re.split(r'[,\s]+', a) if len(p) > 2]
                if any(p in query_lower for p in a_parts):
                    matching_authors.append(a)
            
            if matching_authors:
                cand['author'] = ", ".join(matching_authors)
            else:
                # Fallback: take the first one if it's not in the junk list
                cand['author'] = authors_list[0] if authors_list else "Unknown"

            # Scoring
            # Use token_set_ratio which is more robust to word order and extra words
            base_score = fuzz.token_set_ratio(query_lower, title_clean)
                
            # Author Bonus: Extremely important
            author_bonus = 0
            if cand['author'] != "Unknown":
                author_lower = cand['author'].lower()
                author_parts = [p.strip() for p in re.split(r'[,\s]+', author_lower) if len(p.strip()) > 2]
                match_count = sum(1 for part in author_parts if part in query_lower)
                
                if match_count > 0:
                    author_bonus = 60 + (match_count * 20)
            
            # Penalty System
            penalty = 0
            full_title_lower = str(cand.get('title', '')).lower()
            
            # 1. Study guides / Analysis
            junk_keywords = [
                "fiche de lecture", "analyse", "summary", "chapitre", 
                "profil d'une oeuvre", "profil d'une œuvre", "bac", "concours",
                "étalage", "pédagogique", "expliquée", "expliquer", "étudier", "étude"
            ]
            if any(kw in full_title_lower for kw in junk_keywords):
                penalty += 80
            
            # 2. Graphic novels / Illustrated (unless in query)
            if any(kw in full_title_lower for kw in ["illustré", "illustre", "bande dessinée", "bd", "graphic novel"]):
                if "bd" not in query_lower and "dessinée" not in query_lower:
                    penalty += 50
            
            # 3. Year/Edition mismatch (e.g. Larousse 2012 when query is just Larousse)
            # If query doesn't have a year but title has one, it's likely a specific edition we don't want
            year_match = re.search(r'\b(20\d{2}|19\d{2})\b', full_title_lower)
            if year_match and not re.search(r'\b(20\d{2}|19\d{2})\b', query_lower):
                penalty += 40

            # Final Score calculation
            score = base_score + author_bonus - penalty
            cand['score'] = float(max(0.0, min(100.0, score)))
            cand['title'] = t 
            print(f"DEBUG: Candidat trouvé: {cand['title']} par {cand['author']} (Score: {cand['score']} [Base: {base_score}, Bonus: {author_bonus}, Penalty: {penalty}])")

        # Sort by best score
        candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        # Return top 3 candidates
        return candidates[:3]

    async def _search_google_books(self, query: str) -> List[dict]:
        try:
            params = {"q": query, "maxResults": 20}
            if self.api_key:
                params["key"] = self.api_key
            
            resp = await self.client.get(self.GOOGLE_BOOKS_URL, params=params)
            if resp.status_code != 200:
                print(f"DEBUG: Google Books status {resp.status_code}: {resp.text}")
                return []
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
                    authors = [a.strip() for a in info.get("authors", []) if a and a.strip()]
                    candidates.append({
                        "title": info.get("title", ""),
                        "author": ", ".join(authors),
                        "isbn": isbns[0],
                        "source": "google_books"
                    })
            return candidates
        except Exception as e:
            print(f"DEBUG: Google Books error for '{query}': {e}")
            return []

    async def _search_open_library(self, query: str) -> List[dict]:
        try:
            params = {"q": query, "limit": 20}
            resp = await self.client.get(self.OPEN_LIBRARY_URL, params=params)
            if resp.status_code != 200:
                print(f"DEBUG: Open Library status {resp.status_code}: {resp.text}")
                return []
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
            print(f"DEBUG: Open Library error for '{query}': {e}")
            return []


# Singleton instance
resolver = ISBNResolver()
