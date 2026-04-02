# app/vlm_pipeline/handler.py
import json
import asyncio
import logging
import time
import os
import base64
from typing import Dict, Any, List, Optional

from .image_processor import extract_spines, build_mosaic
from .vlm_extractor import extract_metadata_from_mosaic
from .isbn_resolver import resolve_isbn_async
from .pricer import get_prices_async

logger = logging.getLogger(__name__)

# DynamoDB Cache setup
table = None
try:
    import boto3
    PRICE_TABLE = os.environ.get('PRICE_CACHE_TABLE')
    if PRICE_TABLE:
        table = boto3.resource('dynamodb').Table(PRICE_TABLE)
except Exception as e:
    logger.debug(f"DynamoDB initialization skipped: {e}")

async def get_cached_price(isbn: str) -> Optional[Dict[str, float]]:
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
    isbn = await resolve_isbn_async(book)
    if not isbn:
        book.update({"isbn": None, "prices": {"momox": 0.0, "recyclivre": 0.0}})
        return book
    book["isbn"] = isbn
    cached_prices = await get_cached_price(isbn)
    if cached_prices:
        book["prices"] = cached_prices
    else:
        prices = await get_prices_async(isbn)
        book["prices"] = prices
        await set_cached_price(isbn, prices)
    return book

async def process_shelf(image_bytes: bytes) -> List[Dict[str, Any]]:
    spines = extract_spines(image_bytes)
    if not spines: return []
    mosaics = build_mosaic(spines)
    all_books = []
    for mosaic in mosaics:
        metadata = await extract_metadata_from_mosaic(mosaic)
        all_books.extend(metadata.get("books", []))
    tasks = [process_single_book(book) for book in all_books]
    completed = await asyncio.gather(*tasks, return_exceptions=True)
    results = [b for b in completed if not isinstance(b, Exception)]
    results.sort(key=lambda x: max(x.get("prices", {}).values()), reverse=True)
    return results

def lambda_handler(event, context):
    try:
        # Support both direct call and SQS
        if 'Records' in event:
            body = json.loads(event['Records'][0]['body'])
        else:
            body = event
        image_b64 = body.get("image_base64")
        if not image_b64: return {"statusCode": 400, "body": "Missing image_base64"}
        image_bytes = base64.b64decode(image_b64)
        results = asyncio.run(process_shelf(image_bytes))
        return {"statusCode": 200, "body": json.dumps(results)}
    except Exception as e:
        logger.exception("Lambda failed")
        return {"statusCode": 500, "body": str(e)}
