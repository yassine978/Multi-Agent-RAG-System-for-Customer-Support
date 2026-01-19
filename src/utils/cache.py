"""Caching utilities for optimizing performance and reducing costs.

This module provides caching mechanisms for:
- Vector search results
- DataFrame query results
- LLM responses
"""

import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from .config import CACHE_ENABLED, CACHE_TTL
from .logger import setup_logger

logger = setup_logger(__name__)

# Type variable for generic functions
F = TypeVar('F', bound=Callable[..., Any])


class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, ttl: int = CACHE_TTL):
        """Initialize cache.

        Args:
            ttl: Time to live in seconds
        """
        self.ttl = ttl
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.hits = 0
        self.misses = 0

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Create a cache key from function arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Hash string as cache key
        """
        # Create a stable string representation
        key_data = {
            "args": args,
            "kwargs": kwargs,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            self.misses += 1
            return None

        value, timestamp = self.cache[key]
        current_time = time.time()

        # Check if expired
        if current_time - timestamp > self.ttl:
            del self.cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": f"{hit_rate:.2f}%",
            "cached_items": len(self.cache),
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)


# Global cache instances
_vector_search_cache = SimpleCache(ttl=CACHE_TTL)
_dataframe_query_cache = SimpleCache(ttl=CACHE_TTL)
_llm_response_cache = SimpleCache(ttl=CACHE_TTL)


def cached(
    cache_instance: Optional[SimpleCache] = None,
    enabled: bool = CACHE_ENABLED,
) -> Callable[[F], F]:
    """Decorator to cache function results.

    Args:
        cache_instance: Cache instance to use (default: new instance)
        enabled: Whether caching is enabled

    Returns:
        Decorated function
    """
    if cache_instance is None:
        cache_instance = SimpleCache()

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled:
                return func(*args, **kwargs)

            # Generate cache key
            cache_key = cache_instance._make_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache HIT for {func.__name__}")
                return cached_result

            # Cache miss - call function
            logger.debug(f"Cache MISS for {func.__name__}")
            result = func(*args, **kwargs)

            # Store in cache
            cache_instance.set(cache_key, result)

            return result

        # Add cache management methods
        wrapper.cache_clear = cache_instance.clear  # type: ignore
        wrapper.cache_stats = cache_instance.get_stats  # type: ignore

        return cast(F, wrapper)

    return decorator


def cache_vector_search(func: F) -> F:
    """Decorator for caching vector search results.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    return cached(_vector_search_cache)(func)


def cache_dataframe_query(func: F) -> F:
    """Decorator for caching DataFrame query results.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    return cached(_dataframe_query_cache)(func)


def cache_llm_response(func: F) -> F:
    """Decorator for caching LLM responses.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    return cached(_llm_response_cache)(func)


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches.

    Returns:
        Dictionary with stats for each cache
    """
    return {
        "vector_search": _vector_search_cache.get_stats(),
        "dataframe_query": _dataframe_query_cache.get_stats(),
        "llm_response": _llm_response_cache.get_stats(),
    }


def clear_all_caches() -> None:
    """Clear all caches."""
    _vector_search_cache.clear()
    _dataframe_query_cache.clear()
    _llm_response_cache.clear()
    logger.info("All caches cleared")


def cleanup_all_caches() -> Dict[str, int]:
    """Clean up expired entries from all caches.

    Returns:
        Dictionary with number of entries removed from each cache
    """
    return {
        "vector_search": _vector_search_cache.cleanup_expired(),
        "dataframe_query": _dataframe_query_cache.cleanup_expired(),
        "llm_response": _llm_response_cache.cleanup_expired(),
    }


if __name__ == "__main__":
    # Test caching
    print("Testing cache utilities...")

    @cached()
    def slow_function(x: int) -> int:
        """Simulates a slow function."""
        time.sleep(0.1)
        return x * 2

    # First call - cache miss
    start = time.time()
    result1 = slow_function(5)
    time1 = time.time() - start
    print(f"First call: {result1} (took {time1:.3f}s)")

    # Second call - cache hit
    start = time.time()
    result2 = slow_function(5)
    time2 = time.time() - start
    print(f"Second call: {result2} (took {time2:.3f}s)")

    print(f"Speedup: {time1/time2:.1f}x")

    # Show stats
    stats = slow_function.cache_stats()  # type: ignore
    print(f"\nCache stats: {stats}")
