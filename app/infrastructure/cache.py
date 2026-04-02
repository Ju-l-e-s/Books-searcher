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

def get_default_cache(table_name: Optional[str] = None) -> CacheProvider:
    """Factory to return DynamoDBCache if table_name is set, else InMemoryCache."""
    if table_name:
        return DynamoDBCache(table_name)
    return InMemoryCache()
