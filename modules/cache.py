import hashlib
import json
import os

CACHE_FILE = "cache_store.json"

class CacheManager:
    def __init__(self):
        if not os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'w') as f:
                json.dump({}, f)
                
    def _get_key(self, text, prefix="query"):
        hash_digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"{prefix}:{hash_digest}"

    def get_cached_response(self, query):
        key = self._get_key(query, "query")
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        return data.get(key)
        
    def set_cached_response(self, query, response):
        key = self._get_key(query, "query")
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        if response is None:
            # Remove the cached entry if response is None
            data.pop(key, None)
        else:
            data[key] = response
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f)
            
    def get_cached_extraction(self, file_path):
        # Hash file content or use mtime + path
        stat = os.stat(file_path)
        key = f"file:{file_path}:{stat.st_mtime}"
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        return data.get(key)
        
    def set_cached_extraction(self, file_path, content):
        stat = os.stat(file_path)
        key = f"file:{file_path}:{stat.st_mtime}"
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        data[key] = content
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f)
