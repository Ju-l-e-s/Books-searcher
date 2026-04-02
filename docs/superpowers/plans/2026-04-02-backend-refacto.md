# Backend Refacto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the "BookShelf Sniper" business logic to a clean, modular, and performance-optimized Python backend ready for serverless execution.

**Architecture:** Hexagonal Architecture (Separation of Domain and Infrastructure). The domain logic (ISBN resolution, pricing) will be isolated from infrastructure (HTTP clients, DynamoDB cache).

**Tech Stack:** FastAPI, Pydantic v2, httpx, rapidfuzz, boto3 (optional for local).

---

### Task 1: Infrastructure - Cache Abstraction

**Files:**
- Create: `app/infrastructure/cache.py`
- Modify: `app/isbn_resolver.py`, `app/pricer.py`

- [ ] **Step 1: Create the Cache Interface and DynamoDB implementation**

```python
import json
import logging
import os
import time
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

class CacheProvider(Protocol):
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any, ttl_seconds: int) -> None: ...

class DynamoDBCache:
    def __init__(self, table_name: Optional[str]) -> None:
        self._table_name = table_name
        self._table: Any = None

    def _get_table(self) -> Any:
        if self._table is None and self._table_name:
            try:
                import boto3
                self._table = boto3.resource("dynamodb").Table(self._table_name)
            except Exception as exc:
                logger.debug("DynamoDB unavailable: %s", exc)
        return self._table

    def get(self, key: str) -> Optional[Any]:
        table = self._get_table()
        if not table: return None
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
        if not table: return
        try:
            table.put_item(Item={
                "pk": key,
                "value": json.dumps(value, ensure_ascii=False),
                "ttl": int(time.time()) + ttl_seconds,
            })
        except Exception as exc:
            logger.debug("DynamoDB put error: %s", exc)

class InMemoryCache:
    def __init__(self):
        self._data = {}
    def get(self, key: str) -> Optional[Any]:
        item = self._data.get(key)
        if item and item["ttl"] > time.time():
            return item["value"]
        return None
    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._data[key] = {"value": value, "ttl": time.time() + ttl_seconds}
```

- [ ] **Step 2: Update `app/isbn_resolver.py` to use the new cache abstraction**

- [ ] **Step 3: Update `app/pricer.py` to use the new cache abstraction**

### Task 2: Domain - ISBN Resolver Refinement

**Files:**
- Modify: `app/isbn_resolver.py`
- Test: `tests/test_isbn_resolver.py`

- [ ] **Step 1: Refine the scoring logic to strictly match the 30/20/40/10 split**
- [ ] **Step 2: Ensure "Priority Author" logic is robust (detect primary author from OCR)**
- [ ] **Step 3: Verify Anti-Noise (-80 pts) and Anti-Millesime (-40 pts) penalties**
- [ ] **Step 4: Run tests to verify scoring accuracy**

### Task 3: Domain - Pricer Refinement

**Files:**
- Modify: `app/pricer.py`

- [ ] **Step 1: Ensure robust Mobile Headers are used**
- [ ] **Step 2: Ensure strict 4s timeout with `asyncio.wait_for`**
- [ ] **Step 3: Implement clean error handling (return 0.0 if any API fails)**

### Task 4: API - New OCR Text Endpoint

**Files:**
- Modify: `app/main.py`
- Modify: `app/models.py`

- [ ] **Step 1: Add `AnalysisRequest` model to `app/models.py`**
- [ ] **Step 2: Implement `/analyze-shelf-texts` endpoint in `app/main.py`**

```python
@app.post("/analyze-shelf-texts", response_model=List[BookResult])
async def analyze_shelf_texts(request: AnalysisRequest):
    # Process list of texts directly
    ...
```

### Task 5: Cleanup & Packaging

- [ ] **Step 1: Remove `app/scraper.py`**
- [ ] **Step 2: Update `requirements.txt` with minimal dependencies**
- [ ] **Step 3: Final verification run**
