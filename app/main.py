from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import logging
from pathlib import Path
from typing import List, Any

from app.image_processing import processor
from app.isbn_resolver import resolver
from app.pricer import pricer
from app.synthesis import synthesizer
from app.models import BookResult, AnalysisRequest, OCRItem

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await resolver.close()
    await pricer.close()


app = FastAPI(title="BookShelf Sniper", lifespan=lifespan)

# Mangum handler — wraps the ASGI app for AWS Lambda + API Gateway
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None  # local dev without mangum installed

_BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = _BASE_DIR / "static" / "index.html"
    return index_path.read_text(encoding="utf-8")


async def _process_ocr_results(ocr_items: List[OCRItem]) -> List[BookResult]:
    """Shared logic for resolving ISBNs and fetching prices."""
    if not ocr_items:
        return []

    # 1. Resolve ISBNs in parallel (weighted scoring + cache)
    resolve_tasks = [resolver.resolve(item.text, item.bbox_ratio) for item in ocr_items]
    resolved_results: List[Any] = await asyncio.gather(*resolve_tasks, return_exceptions=True)

    potential_books: List[BookResult] = []
    for candidates in resolved_results:
        if isinstance(candidates, Exception):
            logger.warning("ISBN resolution failed: %s", candidates)
            continue
        if isinstance(candidates, list) and candidates:
            best = candidates[0]
            if isinstance(best, dict):
                potential_books.append(BookResult(
                    title=str(best.get("title", "Unknown")),
                    isbn=str(best.get("isbn", "Unknown")),
                    author=best.get("author"),
                    confidence_score=float(best.get("score", 0.0)) / 100.0,
                ))

    if not potential_books:
        return []

    # 2. Fetch prices in parallel via httpx (cache enabled)
    price_tasks = [pricer.get_prices(book.isbn) for book in potential_books]
    prices_list: List[Any] = await asyncio.gather(*price_tasks, return_exceptions=True)

    for book, prices in zip(potential_books, prices_list):
        if isinstance(prices, Exception):
            logger.warning("Pricing failed for ISBN %s: %s", book.isbn, prices)
            continue
        if isinstance(prices, dict):
            book.price_momox = float(prices.get("momox", 0.0))
            book.price_recyclivre = float(prices.get("recyclivre", 0.0))

    # 3. Deduplicate and sort by best price
    return synthesizer.merge_and_sort(potential_books)


@app.post("/analyze-shelf", response_model=List[BookResult])
async def analyze_shelf(image: UploadFile = File(...)):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas une image valide.")

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Le fichier image est vide.")

        # 1. Segment & OCR → list of text strings (CPU-bound, run in thread)
        ocr_texts: List[str] = await asyncio.to_thread(processor.process_image, image_bytes)
        if not ocr_texts:
            return []

        # Convert simple strings to OCRItem objects
        ocr_items = [OCRItem(text=text) for text in ocr_texts]

        return await _process_ocr_results(ocr_items)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error during shelf analysis")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")


@app.post("/analyze-shelf-texts", response_model=List[BookResult])
async def analyze_shelf_texts(request: AnalysisRequest):
    """Direct ingestion of OCR results (texts + optional bbox ratios)."""
    try:
        return await _process_ocr_results(request.items)
    except Exception:
        logger.exception("Unexpected error during shelf text analysis")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
