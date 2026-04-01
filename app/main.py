from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import logging
from pathlib import Path
from typing import List

from app.image_processing import processor
from app.isbn_resolver import resolver
from app.scraper import scraper
from app.synthesis import synthesizer
from app.models import BookResult

logger = logging.getLogger(__name__)

app = FastAPI(title="BookShelf Sniper")

# Fix B9: Use absolute path relative to this file, not CWD
_BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")


@app.on_event("startup")
async def startup():
    """Initialize the browser pool for scraping."""
    await scraper.start()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup browser pool and HTTP clients."""
    await scraper.stop()
    await resolver.close()


@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = _BASE_DIR / "static" / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.post("/analyze-shelf", response_model=List[BookResult])
async def analyze_shelf(image: UploadFile = File(...)):
    # Validate content type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas une image valide.")

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Le fichier image est vide.")

        # 1. Segment & OCR -> List of text strings
        # Run CPU-bound OCR in a thread to avoid blocking the event loop
        ocr_texts = await asyncio.to_thread(processor.process_image, image_bytes)
        if not ocr_texts:
            return []

        # 2. Resolve ISBNs (Parallel)
        resolve_tasks = [resolver.resolve(text) for text in ocr_texts]
        # Cast to Any to satisfy the linter when external types are not found
        from typing import Any
        resolved_results: Any = await asyncio.gather(*resolve_tasks, return_exceptions=True)

        # Flatten and keep only the best candidate per OCR text
        potential_books = []
        for candidates in resolved_results:
            if isinstance(candidates, Exception):
                logger.warning("ISBN resolution failed for a text: %s", candidates)
                continue
            if isinstance(candidates, list) and len(candidates) > 0:
                best = candidates[0]
                if isinstance(best, dict):
                    potential_books.append(BookResult(
                        title=str(best.get('title', 'Unknown')),
                        isbn=str(best.get('isbn', 'Unknown')),
                        author=best.get('author'),
                        confidence_score=float(best.get('score', 0)) / 100.0
                    ))

        if not potential_books:
            return []

        # 3. Scrape Prices (Parallel with semaphore in the scraper)
        scraping_tasks = [scraper.get_prices(book.isbn) for book in potential_books]
        prices_list = await asyncio.gather(*scraping_tasks, return_exceptions=True)

        # Update book results with prices
        for book, prices in zip(potential_books, prices_list):
            if isinstance(prices, Exception):
                logger.warning("Scraping failed for ISBN %s: %s", book.isbn, prices)
                continue
            if isinstance(prices, dict):
                book.price_momox = float(prices.get('momox', 0.0))
                book.price_recyclivre = float(prices.get('recyclivre', 0.0))

        # 4. Merge, deduplicate and sort
        final_results = synthesizer.merge_and_sort(potential_books)
        return final_results

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during shelf analysis")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {type(e).__name__}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
