# app/vlm_pipeline/pricer.py
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
            # Try various keys where price could be stored
            for key in ("buybackPrice", "price", "buyPrice", "totalPrice"):
                if key in data and data[key] is not None:
                    return float(data[key])
    except Exception as e:
        logger.debug(f"Momox fetch failed for {isbn}: {e}")
    return 0.0

async def fetch_recyclivre(client: httpx.AsyncClient, isbn: str) -> float:
    try:
        resp = await client.get(f"https://www.recyclivre.com/reprise/?isbn={isbn}")
        if resp.status_code == 200:
            # Match prices like "12,34 €" or "12.34 €"
            match = re.search(r"(\d+[.,]\d{2})\s*€", resp.text)
            if match:
                return float(match.group(1).replace(",", "."))
    except Exception as e:
        logger.debug(f"RecycLivre fetch failed for {isbn}: {e}")
    return 0.0

async def get_prices_async(isbn: str) -> Dict[str, float]:
    async with httpx.AsyncClient(headers=HEADERS, timeout=5.0) as client:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    fetch_momox(client, isbn),
                    fetch_recyclivre(client, isbn),
                    return_exceptions=True
                ),
                timeout=5.0
            )
            
            m_price = results[0] if len(results) > 0 else 0.0
            r_price = results[1] if len(results) > 1 else 0.0
            
            return {
                "momox": float(m_price) if isinstance(m_price, (int, float)) else 0.0,
                "recyclivre": float(r_price) if isinstance(r_price, (int, float)) else 0.0
            }
        except asyncio.TimeoutError:
            logger.warning(f"Price fetch timed out for {isbn}")
            return {"momox": 0.0, "recyclivre": 0.0}
        except Exception as e:
            logger.error(f"Unexpected error in get_prices_async for {isbn}: {e}")
            return {"momox": 0.0, "recyclivre": 0.0}
