import time
from unittest.mock import MagicMock, patch
import pytest
from app.infrastructure.cache import InMemoryCache, DynamoDBCache, get_default_cache

def test_in_memory_cache_get_set():
    cache = InMemoryCache()
    cache.set("key1", "value1", 10)
    assert cache.get("key1") == "value1"
    assert cache.get("nonexistent") is None

def test_in_memory_cache_expiry():
    cache = InMemoryCache()
    cache.set("key1", "value1", -1)  # Expired immediately
    assert cache.get("key1") is None

def test_get_default_cache():
    # No table name -> InMemoryCache
    cache = get_default_cache(None)
    assert isinstance(cache, InMemoryCache)
    
    # Table name -> DynamoDBCache
    cache = get_default_cache("my-table")
    assert isinstance(cache, DynamoDBCache)
    assert cache._table_name == "my-table"

@patch("boto3.resource")
def test_dynamodb_cache_get_hit(mock_resource):
    mock_table = MagicMock()
    mock_resource.return_value.Table.return_value = mock_table
    mock_table.get_item.return_value = {
        "Item": {
            "pk": "key1",
            "value": '"value1"',
            "ttl": int(time.time()) + 100
        }
    }
    
    cache = DynamoDBCache("test-table")
    assert cache.get("key1") == "value1"
    mock_table.get_item.assert_called_once_with(Key={"pk": "key1"})

@patch("boto3.resource")
def test_dynamodb_cache_get_miss(mock_resource):
    mock_table = MagicMock()
    mock_resource.return_value.Table.return_value = mock_table
    mock_table.get_item.return_value = {}
    
    cache = DynamoDBCache("test-table")
    assert cache.get("key1") is None

@patch("boto3.resource")
def test_dynamodb_cache_get_expired(mock_resource):
    mock_table = MagicMock()
    mock_resource.return_value.Table.return_value = mock_table
    mock_table.get_item.return_value = {
        "Item": {
            "pk": "key1",
            "value": '"value1"',
            "ttl": int(time.time()) - 100
        }
    }
    
    cache = DynamoDBCache("test-table")
    assert cache.get("key1") is None

@patch("boto3.resource")
def test_dynamodb_cache_set(mock_resource):
    mock_table = MagicMock()
    mock_resource.return_value.Table.return_value = mock_table
    
    cache = DynamoDBCache("test-table")
    with patch("time.time", return_value=1000):
        cache.set("key1", {"a": 1}, 60)
        
    mock_table.put_item.assert_called_once_with(Item={
        "pk": "key1",
        "value": '{"a": 1}',
        "ttl": 1060
    })
