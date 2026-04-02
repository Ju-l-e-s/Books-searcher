# Bouquiniste Pro Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the "Bouquiniste Pro" feature, enabling multi-candidate selection per book spine with visual confirmation (covers), dimension triangulation, and advanced export options (Excel/CSV).

**Architecture:** Shift from a 1:1 (Spine -> Book) to a 1:N (Spine -> Candidates) model. Update the backend to return the top 3 candidates per detected spine and the frontend to display them in a `<select>` menu with automatic cover updates.

**Tech Stack:** FastAPI, Pydantic v2, Google Books API, SheetJS (for browser-side Excel export).

---

### Task 1: Backend Models Update

**Files:**
- Modify: `app/models.py`

- [ ] **Step 1: Update models for multi-candidate and dimensions**

```python
from pydantic import BaseModel, computed_field
from typing import List, Optional


class BookDimensions(BaseModel):
    height: Optional[float] = None
    width: Optional[float] = None
    thickness: Optional[float] = None

    def __str__(self) -> str:
        if not self.height: return "Unknown"
        return f"{self.height}x{self.width or '?'}x{self.thickness or '?'} cm"


class BookCandidate(BaseModel):
    model_config = {"frozen": False}

    title: str
    isbn: str
    author: Optional[str] = None
    publisher: Optional[str] = None
    year: Optional[str] = None
    price_momox: float = 0.0
    price_recyclivre: float = 0.0
    confidence_score: float = 0.0
    cover_url: Optional[str] = None
    dimensions: Optional[BookDimensions] = None

    @computed_field
    @property
    def max_price(self) -> float:
        return max(self.price_momox, self.price_recyclivre)


class SpineResult(BaseModel):
    original_text: str
    bbox_ratio: Optional[float] = None
    candidates: List[BookCandidate]


class AnalysisResponse(BaseModel):
    results: List[SpineResult]


class OCRItem(BaseModel):
    text: str
    bbox_ratio: Optional[float] = None


class AnalysisRequest(BaseModel):
    items: List[OCRItem]
```

- [ ] **Step 2: Commit**

```bash
git add app/models.py
git commit -m "feat(models): update for multi-candidate SpineResult"
```

---

### Task 2: ISBN Resolver Enhancement

**Files:**
- Modify: `app/isbn_resolver.py`
- Test: `tests/test_isbn_resolver.py`

- [ ] **Step 1: Update `_search_google_books` to extract dimensions and cover URL**

```python
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

    def _parse_dim(self, dim_str: Optional[str]) -> Optional[float]:
        if not dim_str: return None
        # Extract numbers from strings like "18.0 cm"
        match = re.search(r"(\d+[.,]?\d*)", dim_str)
        if match:
            return float(match.group(1).replace(",", "."))
        return None
```

- [ ] **Step 2: Update `_search_open_library` to extract cover URL**

```python
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
                
                isbn = isbn_list[0]
                candidates.append({
                    "title": doc.get("title", ""),
                    "author": ", ".join(doc.get("author_name", [])),
                    "publisher": ", ".join(doc.get("publisher", [])),
                    "year": str(doc.get("first_publish_year", "")),
                    "isbn": isbn,
                    "source": "open_library",
                    "cover_url": f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg",
                    "dimensions": None, # Open Library rarely provides dimensions in search
                })
            return candidates
        except Exception as exc:
            logger.debug("Open Library search failed: %s", exc)
            return []
```

- [ ] **Step 3: Verify the Top 3 are returned with correct fields**

Run: `pytest tests/test_isbn_resolver.py -v`

- [ ] **Step 4: Commit**

```bash
git add app/isbn_resolver.py
git commit -m "feat(isbn): return top 3 with dimensions and cover URLs"
```

---

### Task 3: API & Synthesis Logic Refactoring

**Files:**
- Modify: `app/synthesis.py`
- Modify: `app/main.py`

- [ ] **Step 1: Update `synthesizer.merge_and_sort` to handle `SpineResult`**

```python
    @staticmethod
    def merge_and_sort_spines(spines: List[SpineResult]) -> List[SpineResult]:
        """Deduplicate by top ISBN across spines and sort by best candidate's max price."""
        if not spines:
            return []
            
        # For simplicity in this version, we keep all spines but ensure candidates are sorted
        for spine in spines:
            spine.candidates.sort(key=lambda x: x.max_price, reverse=True)
            
        spines.sort(key=lambda x: x.candidates[0].max_price if x.candidates else 0, reverse=True)
        return spines
```

- [ ] **Step 2: Update `_process_ocr_results` in `app/main.py`**

```python
async def _process_ocr_results(ocr_items: List[OCRItem]) -> List[SpineResult]:
    """Shared logic for resolving ISBNs and fetching prices for multiple candidates."""
    if not ocr_items:
        return []

    resolve_tasks = [resolver.resolve(item.text, item.bbox_ratio) for item in ocr_items]
    resolved_results: List[Any] = await asyncio.gather(*resolve_tasks, return_exceptions=True)

    spine_results: List[SpineResult] = []
    
    # 1. Map to SpineResult and BookCandidate
    for item, candidates in zip(ocr_items, resolved_results):
        if isinstance(candidates, Exception) or not candidates:
            continue
        
        book_candidates = []
        for cand in candidates:
            book_candidates.append(BookCandidate(
                title=str(cand.get("title", "Unknown")),
                isbn=str(cand.get("isbn", "Unknown")),
                author=cand.get("author"),
                publisher=cand.get("publisher"),
                year=cand.get("year"),
                confidence_score=float(cand.get("score", 0.0)) / 100.0,
                cover_url=cand.get("cover_url"),
                dimensions=cand.get("dimensions"),
            ))
            
        spine_results.append(SpineResult(
            original_text=item.text,
            bbox_ratio=item.bbox_ratio,
            candidates=book_candidates
        ))

    if not spine_results:
        return []

    # 2. Fetch prices in parallel for ALL candidates (max 3 per spine)
    all_isbn_to_fetch = set()
    for spine in spine_results:
        for cand in spine.candidates:
            all_isbn_to_fetch.add(cand.isbn)
            
    # Use a dict to store price results to avoid redundant calls
    price_tasks = {isbn: pricer.get_prices(isbn) for isbn in all_isbn_to_fetch}
    isbns = list(price_tasks.keys())
    prices_list: List[Any] = await asyncio.gather(*price_tasks.values(), return_exceptions=True)
    
    price_map = {}
    for isbn, prices in zip(isbns, prices_list):
        if not isinstance(prices, Exception):
            price_map[isbn] = prices

    # 3. Assign prices back to candidates
    for spine in spine_results:
        for cand in spine.candidates:
            prices = price_map.get(cand.isbn, {})
            cand.price_momox = float(prices.get("momox", 0.0))
            cand.price_recyclivre = float(prices.get("recyclivre", 0.0))

    # 4. Sort and return
    return synthesizer.merge_and_sort_spines(spine_results)
```

- [ ] **Step 3: Commit**

```bash
git add app/synthesis.py app/main.py
git commit -m "feat(api): update synthesis to return multi-candidate SpineResult"
```

---

### Task 4: Frontend UI (Multi-Edition & Covers)

**Files:**
- Modify: `app/static/index.html`

- [ ] **Step 1: Update Table Headers and UI Logic**

Update `displayResults` to:
- Create a `select` for editions.
- Display `cover_url` in an `img` tag.
- Display dimensions.
- Update the row data when the `select` changes.

- [ ] **Step 2: Commit**

```bash
git add app/static/index.html
git commit -m "feat(ui): implement multi-edition selection with automatic cover display"
```

---

### Task 5: Frontend Export (CSV/Excel & Selection)

**Files:**
- Modify: `app/static/index.html`

- [ ] **Step 1: Add selection checkboxes and SheetJS for Excel export**

- [ ] **Step 2: Implement "Export All" and "Export Selection" for CSV and Excel**

- [ ] **Step 3: Commit**

```bash
git add app/static/index.html
git commit -m "feat(export): add Excel export and selection-based export"
```
