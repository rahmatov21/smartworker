"""
Tests for LRUCache and TTL Expiration.
"""

import time
from project.src.cache import LRUCache


def test_lru_cache_basic_ops():
    cache = LRUCache(maxsize=2, ttl_seconds=10.0)
    cache.set("k1", "v1")
    cache.set("k2", "v2")

    assert cache.get("k1") == "v1"
    assert cache.get("k2") == "v2"
    assert cache.get("k3") is None


def test_lru_eviction():
    cache = LRUCache(maxsize=2, ttl_seconds=10.0)
    cache.set("a", 1)
    cache.set("b", 2)
    # Touch 'a' so 'b' becomes least recently used
    _ = cache.get("a")
    cache.set("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == 3


def test_lru_ttl_expiration():
    cache = LRUCache(maxsize=10, ttl_seconds=0.05)
    cache.set("temp", "value")
    assert cache.get("temp") == "value"

    time.sleep(0.06)
    assert cache.get("temp") is None


def test_cache_stats():
    cache = LRUCache(maxsize=5)
    cache.set("x", 100)
    _ = cache.get("x")  # hit
    _ = cache.get("y")  # miss

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["maxsize"] == 5
