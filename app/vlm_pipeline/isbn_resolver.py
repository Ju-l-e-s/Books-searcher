# app/vlm_pipeline/isbn_resolver.py
import httpx
from rapidfuzz import fuzz
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def resolve_isbn_async(book_dict: Dict[str, Any]) -> Optional[str]:
    title = book_dict.get("title")
    author = book_dict.get("author")
    publisher_target = book_dict.get("publisher", "")
    
    if not title:
        return None
        
    query = f"intitle:{title}"
    if author:
        query += f"+inauthor:{author}"
        
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://www.googleapis.com/books/v1/volumes",
                params={"q": query, "maxResults": 5}
            )
            resp.raise_for_status()
            data = resp.json()
            
            best_isbn = None
            best_score = 0.0
            
            for item in data.get("items", []):
                info = item.get("volumeInfo", {})
                # Extract ISBNs
                isbns = [i["identifier"] for i in info.get("industryIdentifiers", []) if i.get("type") in ("ISBN_13", "ISBN_10")]
                if not isbns:
                    continue
                
                c_title = info.get("title") or ""
                c_authors = info.get("authors", [])
                c_author = " ".join(c_authors) if isinstance(c_authors, list) else str(c_authors)
                c_publisher = info.get("publisher") or ""
                
                # Weights: Publisher (40%), Title (40%), Author (20%)
                s_pub = fuzz.token_set_ratio(str(publisher_target).lower(), str(c_publisher).lower()) if publisher_target and c_publisher else 0
                s_title = fuzz.token_set_ratio(str(title).lower(), str(c_title).lower()) if title and c_title else 0
                s_author = fuzz.token_set_ratio(str(author).lower(), str(c_author).lower()) if author and c_author else 0
                
                total_score = (s_pub * 0.4) + (s_title * 0.4) + (s_author * 0.2)
                
                if total_score > best_score:
                    best_score = total_score
                    best_isbn = isbns[0]
            
            # Threshold to avoid false positives (>= 60.0 to allow exact Title+Author matches)
            return best_isbn if best_score >= 60.0 else None
            
    except Exception as e:
        logger.warning(f"ISBN resolution failed for {title}: {e}")
        return None
