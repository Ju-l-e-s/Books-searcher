# BookShelf Sniper VLM Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a VLM-based (Gemini 2.0 Flash) architecture for book spine extraction, replacing traditional OCR with an AI-driven mosaic analysis approach.

**Architecture:** The system will segment spines using YOLOv11, build a single mosaic image to save API tokens, pass the mosaic to Gemini 2.0 Flash to extract structured JSON metadata, resolve ISBNs using RapidFuzz, and fetch prices asynchronously with a DynamoDB cache layer.

**Tech Stack:** Python 3.12+, FastAPI, `google-genai` (Gemini 2.0 Flash), YOLOv11, `httpx`, `rapidfuzz`, `boto3`.

---

### Task 1: Module 1 - Segmentation and Mosaic (`app/vlm_pipeline/image_processor.py`)

**Files:**
- Create: `app/vlm_pipeline/image_processor.py`
- Create: `tests/test_vlm_image_processor.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
import numpy as np
import cv2
from app.vlm_pipeline.image_processor import extract_spines, build_mosaic

def test_build_mosaic_creates_image():
    # Create dummy images
    img1 = np.zeros((100, 50, 3), dtype=np.uint8)
    img2 = np.zeros((120, 60, 3), dtype=np.uint8)
    
    mosaics = build_mosaic([img1, img2], max_per_mosaic=2)
    assert len(mosaics) == 1
    assert isinstance(mosaics[0], str) # Base64 string
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vlm_image_processor.py -v`
Expected: FAIL with ModuleNotFoundError or ImportError

- [ ] **Step 3: Write implementation**

```python
import cv2
import numpy as np
import base64
from typing import List
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

# Lazy load model
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = YOLO("yolo11n.pt")
        except Exception:
            _model = YOLO("yolov8n.pt") # Fallback for local dev if 11 not downloaded
    return _model

def extract_spines(image_bytes: bytes) -> List[np.ndarray]:
    """Isolate spines using YOLO."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    model = get_model()
    results = model(img, conf=0.15, iou=0.3)
    crops = []
    
    for result in results:
        if not hasattr(result, 'boxes'): continue
        for box in result.boxes:
            cls_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            if cls_id == 73 or True: # 73 is book in COCO, or accept custom model
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h_img, w_img = img.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    # Basic upright check
                    h, w = crop.shape[:2]
                    if h < w:
                         crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                    crops.append(crop)
    return crops

def build_mosaic(cropped_images: List[np.ndarray], max_per_mosaic: int = 10) -> List[str]:
    """Combine crops into a mosaic with ID labels, return base64 list."""
    if not cropped_images: return []
    
    mosaics_b64 = []
    for i in range(0, len(cropped_images), max_per_mosaic):
        batch = cropped_images[i:i+max_per_mosaic]
        
        # Determine grid dimensions
        max_h = max(img.shape[0] for img in batch) + 40 # 40px for label
        total_w = sum(img.shape[1] for img in batch) + (10 * len(batch))
        
        mosaic = np.full((max_h, total_w, 3), 255, dtype=np.uint8)
        
        current_x = 0
        for j, img in enumerate(batch):
            h, w = img.shape[:2]
            # Draw ID
            book_id = i + j + 1
            cv2.putText(mosaic, str(book_id), (current_x + (w//2) - 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Paste image
            mosaic[40:40+h, current_x:current_x+w] = img
            current_x += w + 10
            
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', mosaic)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        mosaics_b64.append(b64_str)
        
    return mosaics_b64
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vlm_image_processor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/vlm_pipeline/image_processor.py tests/test_vlm_image_processor.py
git commit -m "feat(vlm): add image segmentation and mosaic builder"
```

---

### Task 2: Module 2 - VLM Extraction (`app/vlm_pipeline/vlm_extractor.py`)

**Files:**
- Create: `app/vlm_pipeline/vlm_extractor.py`
- Create: `tests/test_vlm_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from unittest.mock import patch, AsyncMock
from app.vlm_pipeline.vlm_extractor import extract_metadata_from_mosaic

@pytest.mark.asyncio
async def test_extract_metadata_success():
    with patch('app.vlm_pipeline.vlm_extractor.genai') as mock_genai:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.text = '{"books": [{"id": 1, "title": "Test", "author": "Auth", "publisher": "Pub"}]}'
        mock_client.models.generate_content.return_value = mock_response
        
        result = await extract_metadata_from_mosaic("fake_base64")
        assert len(result["books"]) == 1
        assert result["books"][0]["title"] == "Test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vlm_extractor.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
import json
import logging
from typing import Dict, Any, List
import os

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un extracteur de données bibliographiques. Analyse cette mosaïque de tranches de livres. Chaque livre a un numéro au-dessus. Retourne UNIQUEMENT un objet JSON valide avec la structure suivante : {"books": [{"id": 1, "title": "...", "author": "...", "publisher": "..."}]}. Remplace par null si illisible. Ne génère aucun texte hors du JSON."""

async def extract_metadata_from_mosaic(mosaic_base64: str) -> Dict[str, List[Dict[str, Any]]]:
    """Send mosaic to Gemini 2.0 Flash to extract JSON metadata."""
    if not genai:
        logger.error("google-genai SDK not installed")
        return {"books": []}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set")
        return {"books": []}

    try:
        client = genai.Client(api_key=api_key)
        
        # Format the base64 string for Gemini
        image_part = types.Part.from_bytes(
            data=base64.b64decode(mosaic_base64),
            mime_type="image/jpeg"
        )
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[SYSTEM_PROMPT, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        text = response.text
        # Clean potential markdown wrapping
        if text.startswith("```json"):
            text = text[7:-3]
            
        data = json.loads(text)
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse VLM JSON: {e}. Raw text: {response.text}")
        return {"books": []}
    except Exception as e:
        logger.error(f"VLM extraction failed: {e}")
        return {"books": []}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vlm_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/vlm_pipeline/vlm_extractor.py tests/test_vlm_extractor.py
git commit -m "feat(vlm): add Gemini 2.0 Flash extractor"
```

---

### Task 3: Module 3 - ISBN Triangulation Engine (`app/vlm_pipeline/isbn_resolver.py`)

**Files:**
- Create: `app/vlm_pipeline/isbn_resolver.py`
- Create: `tests/test_vlm_isbn_resolver.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from unittest.mock import patch, AsyncMock
from app.vlm_pipeline.isbn_resolver import resolve_isbn_async

@pytest.mark.asyncio
async def test_resolve_isbn_async():
    book_dict = {"title": "L'Etranger", "author": "Albert Camus", "publisher": "Gallimard"}
    
    with patch('httpx.AsyncClient.get') as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "items": [
                {
                    "volumeInfo": {
                        "title": "L'Étranger",
                        "authors": ["Albert Camus"],
                        "publisher": "Gallimard",
                        "industryIdentifiers": [{"type": "ISBN_13", "identifier": "9782070360024"}]
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp
        
        result = await resolve_isbn_async(book_dict)
        assert result == "9782070360024"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vlm_isbn_resolver.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
import httpx
from rapidfuzz import fuzz
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def resolve_isbn_async(book_dict: Dict[str, Any]) -> Optional[str]:
    """Resolve structured VLM metadata to an ISBN via Google Books."""
    title = book_dict.get("title")
    author = book_dict.get("author")
    publisher_target = book_dict.get("publisher", "")
    
    if not title: return None
    
    query = f"intitle:{title}"
    if author: query += f"+inauthor:{author}"
    
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
                isbns = [i["identifier"] for i in info.get("industryIdentifiers", []) if i.get("type") in ("ISBN_13", "ISBN_10")]
                if not isbns: continue
                
                c_title = info.get("title", "")
                c_author = " ".join(info.get("authors", []))
                c_publisher = info.get("publisher", "")
                
                # Weights: Publisher (40%), Title (40%), Author (20%)
                s_pub = fuzz.token_set_ratio(publisher_target.lower(), c_publisher.lower()) if publisher_target else 0
                s_title = fuzz.token_set_ratio(title.lower(), c_title.lower())
                s_author = fuzz.token_set_ratio(author.lower(), c_author.lower()) if author else 0
                
                total_score = (s_pub * 0.4) + (s_title * 0.4) + (s_author * 0.2)
                
                if total_score > best_score:
                    best_score = total_score
                    best_isbn = isbns[0]
                    
            # Threshold to avoid false positives
            return best_isbn if best_score > 60.0 else None
            
    except Exception as e:
        logger.warning(f"ISBN resolution failed for {title}: {e}")
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vlm_isbn_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/vlm_pipeline/isbn_resolver.py tests/test_vlm_isbn_resolver.py
git commit -m "feat(vlm): add ISBN triangulation engine"
```

---

### Task 4: Module 4 - Pricer (`app/vlm_pipeline/pricer.py`)

**Files:**
- Create: `app/vlm_pipeline/pricer.py`
- Create: `tests/test_vlm_pricer.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from unittest.mock import patch, AsyncMock
from app.vlm_pipeline.pricer import get_prices_async

@pytest.mark.asyncio
async def test_get_prices_async():
    with patch('httpx.AsyncClient.get') as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"price": 5.50} # Mock format
        mock_get.return_value = mock_resp
        
        result = await get_prices_async("9781234567890")
        assert "momox" in result
        assert "recyclivre" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vlm_pricer.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
import asyncio
import httpx
import logging
from typing import Dict
import re

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
    "Accept": "application/json, text/html, */*",
}

async def fetch_momox(client: httpx.AsyncClient, isbn: str) -> float:
    try:
        resp = await client.get(f"https://www.momox.fr/ajax/sell/additem/?ean={isbn}")
        if resp.status_code == 200:
            data = resp.json()
            for key in ("buybackPrice", "price", "buyPrice"):
                if data.get(key) is not None:
                    return float(data[key])
    except Exception:
        pass
    return 0.0

async def fetch_recyclivre(client: httpx.AsyncClient, isbn: str) -> float:
    try:
        resp = await client.get(f"https://www.recyclivre.com/reprise/?isbn={isbn}")
        if resp.status_code == 200:
            match = re.search(r"(\d+[.,]\d{2})\s*€", resp.text)
            if match:
                return float(match.group(1).replace(",", "."))
    except Exception:
        pass
    return 0.0

async def get_prices_async(isbn: str) -> Dict[str, float]:
    """Fetch prices from Momox and RecycLivre concurrently with strict timeout."""
    async with httpx.AsyncClient(headers=HEADERS, timeout=5.0) as client:
        try:
            m_price, r_price = await asyncio.wait_for(
                asyncio.gather(
                    fetch_momox(client, isbn),
                    fetch_recyclivre(client, isbn),
                    return_exceptions=True
                ),
                timeout=5.0
            )
            
            return {
                "momox": m_price if isinstance(m_price, float) else 0.0,
                "recyclivre": r_price if isinstance(r_price, float) else 0.0
            }
        except asyncio.TimeoutError:
            logger.warning(f"Price fetch timed out for {isbn}")
            return {"momox": 0.0, "recyclivre": 0.0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vlm_pricer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/vlm_pipeline/pricer.py tests/test_vlm_pricer.py
git commit -m "feat(vlm): add concurrent pricer"
```

---

### Task 5: Module 5 - Lambda Orchestration (`app/vlm_pipeline/handler.py`)

**Files:**
- Create: `app/vlm_pipeline/handler.py`

- [ ] **Step 1: Write implementation**

```python
import json
import asyncio
import logging
import time
import os
from typing import Dict, Any, List

from .image_processor import extract_spines, build_mosaic
from .vlm_extractor import extract_metadata_from_mosaic
from .isbn_resolver import resolve_isbn_async
from .pricer import get_prices_async

logger = logging.getLogger(__name__)

# DynamoDB Cache setup
try:
    import boto3
    dynamodb = boto3.resource('dynamodb')
    PRICE_TABLE = os.environ.get('PRICE_CACHE_TABLE')
    table = dynamodb.Table(PRICE_TABLE) if PRICE_TABLE else None
except ImportError:
    table = None

async def get_cached_price(isbn: str) -> Dict[str, float]:
    if not table: return None
    try:
        resp = table.get_item(Key={'isbn': isbn})
        item = resp.get('Item')
        if item and int(item.get('ttl', 0)) > int(time.time()):
            return {"momox": float(item.get('momox', 0)), "recyclivre": float(item.get('recyclivre', 0))}
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
    return None

async def set_cached_price(isbn: str, prices: Dict[str, float]):
    if not table: return
    try:
        table.put_item(Item={
            'isbn': isbn,
            'momox': str(prices['momox']),
            'recyclivre': str(prices['recyclivre']),
            'ttl': int(time.time()) + 86400 # 24h
        })
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")

async def process_single_book(book: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ISBN and fetch price for a single book."""
    isbn = await resolve_isbn_async(book)
    if not isbn:
        book["isbn"] = None
        book["prices"] = {"momox": 0.0, "recyclivre": 0.0}
        return book
        
    book["isbn"] = isbn
    
    # Check cache
    cached_prices = await asyncio.to_thread(asyncio.run, get_cached_price(isbn)) if table else None
    
    if cached_prices:
        book["prices"] = cached_prices
    else:
        prices = await get_prices_async(isbn)
        book["prices"] = prices
        if table:
            await asyncio.to_thread(asyncio.run, set_cached_price(isbn, prices))
            
    return book

async def process_shelf(image_bytes: bytes) -> List[Dict[str, Any]]:
    """Full pipeline execution."""
    spines = await asyncio.to_thread(extract_spines, image_bytes)
    if not spines: return []
    
    mosaics = await asyncio.to_thread(build_mosaic, spines)
    if not mosaics: return []
    
    # Process mosaics (sequentially to avoid Gemini rate limits, or parallel if tier allows)
    all_books = []
    for mosaic in mosaics:
        metadata = await extract_metadata_from_mosaic(mosaic)
        books = metadata.get("books", [])
        all_books.extend(books)
        
    # Parallel processing of all extracted books
    tasks = [process_single_book(book) for book in all_books]
    completed_books = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter exceptions and sort by best price
    final_list = [b for b in completed_books if not isinstance(b, Exception)]
    final_list.sort(key=lambda x: max(x.get("prices", {}).get("momox", 0), x.get("prices", {}).get("recyclivre", 0)), reverse=True)
    
    return final_list

def lambda_handler(event, context):
    """AWS Lambda entry point."""
    # Assuming event is an SQS message containing base64 image or S3 url
    # For MVP, assuming event contains base64 image string
    import base64
    
    try:
        body = json.loads(event['Records'][0]['body'])
        image_b64 = body.get("image_base64")
        if not image_b64:
             return {"statusCode": 400, "body": "Missing image_base64"}
             
        image_bytes = base64.b64decode(image_b64)
        
        # Run async event loop
        results = asyncio.run(process_shelf(image_bytes))
        
        return {
            "statusCode": 200,
            "body": json.dumps(results)
        }
    except Exception as e:
        logger.exception("Lambda processing failed")
        return {"statusCode": 500, "body": str(e)}
```

- [ ] **Step 2: Commit**

```bash
git add app/vlm_pipeline/handler.py
git commit -m "feat(vlm): add lambda orchestration handler"
```
