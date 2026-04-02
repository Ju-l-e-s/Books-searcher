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
from app.models import (
    SpineResult, 
    BookCandidate, 
    BookDimensions, 
    AnalysisRequest, 
    OCRItem
)

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


async def _process_ocr_results(ocr_items: List[OCRItem]) -> List[SpineResult]:
    """Shared logic for resolving ISBNs and fetching prices for multiple candidates."""
    if not ocr_items:
        logger.info("No OCR items to process.")
        return []

    logger.info(f"Processing {len(ocr_items)} OCR items...")
    resolve_tasks = [resolver.resolve(item.text, item.bbox_ratio) for item in ocr_items]
    resolved_results: List[Any] = await asyncio.gather(*resolve_tasks, return_exceptions=True)

    spine_results: List[SpineResult] = []
    
    # 1. Map to SpineResult and BookCandidate
    for item, candidates in zip(ocr_items, resolved_results):
        if isinstance(candidates, Exception) or not candidates:
            if isinstance(candidates, Exception):
                logger.warning(f"ISBN resolution failed for '{item.text}': {candidates}")
            else:
                logger.warning(f"No ISBN candidates found for '{item.text}'")
            continue
        
        logger.info(f"Found {len(candidates)} candidates for '{item.text}'")
        book_candidates = []
        for cand in candidates:
            logger.info(f"  - Candidate: {cand.get('title')} ({cand.get('isbn')}) - Score: {cand.get('score')}")
            # Note: cand is a dict from resolver.resolve
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
        logger.info("No spine results after ISBN resolution.")
        return []

    # 2. Fetch prices in parallel for ALL candidates (using a set of ISBNs)
    all_isbns = {c.isbn for s in spine_results for c in s.candidates}
    logger.info(f"Fetching prices for {len(all_isbns)} unique ISBNs...")
    price_tasks = {isbn: pricer.get_prices(isbn) for isbn in all_isbns}
    
    isbns_to_fetch = list(price_tasks.keys())
    prices_list = await asyncio.gather(*price_tasks.values(), return_exceptions=True)
    
    price_map = {}
    for isbn, result in zip(isbns_to_fetch, prices_list):
        if not isinstance(result, Exception):
            price_map[isbn] = result
            logger.info(f"  - Prices for {isbn}: Momox={result.get('momox')}, RecycLivre={result.get('recyclivre')}")
        else:
            logger.warning(f"Pricing failed for ISBN {isbn}: {result}")

    # 3. Assign prices back to candidates
    for spine in spine_results:
        for cand in spine.candidates:
            prices = price_map.get(cand.isbn, {})
            cand.price_momox = float(prices.get("momox", 0.0))
            cand.price_recyclivre = float(prices.get("recyclivre", 0.0))

    # 4. Sort and return
    results = synthesizer.merge_and_sort_spines(spine_results)
    logger.info(f"Final results count: {len(results)}")
    return results


@app.post("/analyze-shelf", response_model=List[SpineResult])
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


@app.post("/analyze-shelf-texts", response_model=List[SpineResult])
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
