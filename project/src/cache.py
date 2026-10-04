"""
High-Performance Thread-Safe LRU Cache with TTL Expiration.
Used for caching query tokenizations, embeddings, and similarity matrices.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class LRUCache:
    """
    Thread-safe Least Recently Used (LRU) cache with optional Time-To-Live (TTL).
    """

    def __init__(self, maxsize: int = 1024, ttl_seconds: Optional[float] = 300.0):
        self.maxsize = max(1, maxsize)
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieves an item from cache, respecting TTL."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            value, timestamp = self._cache[key]
            if self.ttl is not None and (time.time() - timestamp) > self.ttl:
                # Expired
                del self._cache[key]
                self.evictions += 1
                self.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Stores a key-value pair in cache, evicting oldest if full."""
        with self._lock:
            now = time.time()
            if key in self._cache:
                self._cache[key] = (value, now)
                self._cache.move_to_end(key)
                return

            if len(self._cache) >= self.maxsize:
                # Evict oldest
                self._cache.popitem(last=False)
                self.evictions += 1

            self._cache[key] = (value, now)

    def stats(self) -> Dict[str, Any]:
        """Returns cache efficiency metrics."""
        with self._lock:
            total = self.hits + self.misses
            hit_ratio = round((self.hits / total) * 100.0, 2) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio_percent": hit_ratio,
                "evictions": self.evictions,
                "ttl_seconds": self.ttl,
            }

    def clear(self) -> None:
        """Purges all entries."""
        with self._lock:
            self._cache.clear()
