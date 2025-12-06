import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Optional, Any

CACHE_FILE = os.path.join("storage", "cache_store.json")
MAX_CACHE_SIZE = 1000  # Maximum entries in memory cache
PERSIST_INTERVAL = 30  # Persist to disk every 30 seconds

# Ensure storage directory exists
def _ensure_storage_dir():
    """Ensure storage directory exists."""
    storage_dir = os.path.dirname(CACHE_FILE)
    if storage_dir and not os.path.exists(storage_dir):
        os.makedirs(storage_dir, exist_ok=True)

class LRUCache:
    """LRU Cache implementation with size limit."""
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove oldest (first item)
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def delete(self, key: str):
        self.cache.pop(key, None)
    
    def clear(self):
        self.cache.clear()
    
    def to_dict(self):
        return dict(self.cache)

class CacheManager:
    """
    High-performance cache manager with in-memory caching and background persistence.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CacheManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._memory_cache = LRUCache(MAX_CACHE_SIZE)
        self._dirty = False
        self._last_persist = time.time()
        self._persist_lock = threading.Lock()
        self._initialized = True
        
        # Load existing cache from disk
        self._load_from_disk()
        
        # Start background persistence thread
        self._start_persistence_thread()
                
    def _get_key(self, text, prefix="query"):
        hash_digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"{prefix}:{hash_digest}"
    
    def _load_from_disk(self):
        """Load cache from disk into memory."""
        _ensure_storage_dir()  # Ensure storage directory exists
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Load into memory cache
                    for key, value in data.items():
                        self._memory_cache.set(key, value)
            except Exception:
                # If loading fails, start with empty cache
                pass
    
    def _persist_to_disk(self):
        """Persist cache to disk (called in background thread)."""
        if not self._dirty:
            return
        
        with self._persist_lock:
            if not self._dirty:
                return
            
            try:
                _ensure_storage_dir()  # Ensure storage directory exists
                # Get current cache state
                cache_data = self._memory_cache.to_dict()
                
                # Write to disk
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2)
                
                self._dirty = False
                self._last_persist = time.time()
            except Exception:
                # If persistence fails, continue (cache still in memory)
                pass
    
    def _start_persistence_thread(self):
        """Start background thread for periodic persistence."""
        def persist_worker():
            while True:
                time.sleep(PERSIST_INTERVAL)
                if self._dirty:
                    self._persist_to_disk()
        
        thread = threading.Thread(target=persist_worker, daemon=True)
        thread.start()
    
    def _persist_async(self):
        """Mark cache as dirty for background persistence."""
        self._dirty = True
        # Persist immediately if it's been a while
        if time.time() - self._last_persist > PERSIST_INTERVAL:
            self._persist_to_disk()

    def get_cached_response(self, query):
        """Get cached response (fast in-memory lookup)."""
        key = self._get_key(query, "query")
        return self._memory_cache.get(key)
        
    def set_cached_response(self, query, response):
        """Set cached response (fast in-memory write, async persistence)."""
        key = self._get_key(query, "query")
        if response is None:
            self._memory_cache.delete(key)
        else:
            self._memory_cache.set(key, response)
        self._persist_async()
            
    def get_cached_extraction(self, file_path):
        """Get cached file extraction."""
        stat = os.stat(file_path)
        key = f"file:{file_path}:{stat.st_mtime}"
        return self._memory_cache.get(key)
        
    def set_cached_extraction(self, file_path, content):
        """Set cached file extraction."""
        stat = os.stat(file_path)
        key = f"file:{file_path}:{stat.st_mtime}"
        self._memory_cache.set(key, content)
        self._persist_async()
    
    def get_cached_intent(self, query):
        """Get cached intent classification (fast in-memory lookup)."""
        key = self._get_key(query, "intent")
        return self._memory_cache.get(key)
    
    def set_cached_intent(self, query, intent_data):
        """Set cached intent classification (fast in-memory write, async persistence)."""
        key = self._get_key(query, "intent")
        self._memory_cache.set(key, intent_data)
        self._persist_async()
    
    def get_cached_query_result(self, query, query_type):
        """Get cached query result (SQL or RAG)."""
        key = self._get_key(f"{query_type}:{query}", "query_result")
        return self._memory_cache.get(key)
    
    def set_cached_query_result(self, query, query_type, result, ttl=3600):
        """Set cached query result with TTL."""
        key = self._get_key(f"{query_type}:{query}", "query_result")
        cache_entry = {
            "result": result,
            "timestamp": time.time(),
            "ttl": ttl
        }
        self._memory_cache.set(key, cache_entry)
        self._persist_async()
    
    def clear_cache(self):
        """Clear all caches."""
        self._memory_cache.clear()
        self._dirty = True
        self._persist_to_disk()
    
    def get_cache_stats(self):
        """Get cache statistics."""
        return {
            "size": len(self._memory_cache.cache),
            "max_size": self._memory_cache.max_size,
            "dirty": self._dirty,
            "last_persist": self._last_persist
        }
