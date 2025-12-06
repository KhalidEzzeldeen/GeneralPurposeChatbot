# Performance Improvement Guide

## Current Performance Bottlenecks

### 1. **File-Based Caching (Major Bottleneck)**
**Problem:**
- Every cache read/write opens, reads, parses, and writes the entire JSON file
- File I/O operations are blocking and slow
- Cache file grows over time, making reads/writes slower

**Impact:** High - Every query involves 2-4 file operations

**Solution:**
- Use in-memory cache (dict) with periodic disk persistence
- Use Redis for distributed caching (if multiple instances)
- Implement LRU cache with size limits
- Batch cache writes (write every N operations or on timer)
- Use faster serialization (msgpack instead of JSON)

---

### 2. **No Streamlit Caching for Expensive Operations**
**Problem:**
- `initialize_system()` is called on every page load/rerun
- Creates new LLM clients, embedding models, and vector store connections
- No caching of index, LLM, or database connections

**Impact:** Very High - Reinitializes everything on each interaction

**Solution:**
- Use `@st.cache_resource` for:
  - LLM client initialization
  - Embedding model initialization
  - VectorStoreIndex (if possible without async issues)
  - Database connection pool
- Cache schema summary (only refresh on demand)
- Cache intent classifier instance

---

### 3. **Schema Summary Generated on Every App Start**
**Problem:**
- `get_schema_summary()` queries database on every app initialization
- Executes multiple SQL queries (table names, columns, DISTINCT values)
- No caching of schema information

**Impact:** High - Slow app startup, especially with large databases

**Solution:**
- Cache schema summary in memory/file
- Only refresh when user clicks "Scan Schema" button
- Store schema summary in config or separate cache file
- Use `@st.cache_data` with TTL for schema summary
- Lazy load: Only generate when needed for intent classification

---

### 4. **LLM Calls for Intent Classification (Every Query)**
**Problem:**
- Every user query triggers an LLM call for intent classification
- LLM calls are slow (network latency + processing time)
- Even with caching, first-time queries are slow

**Impact:** Very High - Adds 1-3 seconds per query

**Solution:**
- **Aggressive intent caching**: Cache intent classifications more aggressively
- **Batch classification**: Classify multiple queries at once if possible
- **Faster model**: Use smaller/faster model for classification (separate from main LLM)
- **Parallel processing**: Run intent classification in parallel with other operations
- **Confidence-based caching**: Cache low-confidence classifications longer
- **Pre-classification**: Pre-classify common query patterns

---

### 5. **Synchronous Database Operations**
**Problem:**
- All database operations are synchronous and block the UI
- Schema queries, SQL queries, and data fetching all block
- No connection pooling or query optimization

**Impact:** High - Database queries can take 1-5 seconds

**Solution:**
- **Connection pooling**: Reuse database connections
- **Async queries**: Use async database operations (if compatible with Streamlit)
- **Query optimization**: Add indexes, optimize SQL queries
- **Background processing**: Run expensive queries in background threads
- **Result caching**: Cache SQL query results
- **Pagination**: Limit result sets, use pagination

---

### 6. **Vector Store Operations (ChromaDB)**
**Problem:**
- Vector similarity search can be slow with large knowledge bases
- No caching of search results
- Embedding generation for queries happens every time

**Impact:** Medium-High - RAG queries can take 2-5 seconds

**Solution:**
- **Cache embeddings**: Cache query embeddings
- **Index optimization**: Use HNSW index for faster similarity search
- **Result caching**: Cache RAG search results
- **Limit search scope**: Reduce `similarity_top_k` if not needed
- **Pre-compute embeddings**: Pre-embed common queries
- **ChromaDB optimization**: Tune ChromaDB settings for performance

---

### 7. **No Connection Pooling**
**Problem:**
- New database connections created for each operation
- No reuse of connections
- Connection overhead on every query

**Impact:** Medium - Adds 100-500ms per database operation

**Solution:**
- **SQLAlchemy connection pool**: Configure connection pooling
- **Singleton pattern**: Reuse database engine instance
- **Connection limits**: Set appropriate pool size
- **Connection timeout**: Configure connection timeouts

---

### 8. **Inefficient Cache File Operations**
**Problem:**
- Cache file is read/written synchronously
- Entire file is loaded into memory for every operation
- No file locking (potential race conditions)

**Impact:** Medium - Adds 50-200ms per cache operation

**Solution:**
- **In-memory cache**: Keep cache in memory, persist periodically
- **Async file I/O**: Use async file operations
- **File locking**: Implement proper file locking
- **Database-backed cache**: Use SQLite or Redis instead of JSON file

---

### 9. **No Streaming Responses**
**Problem:**
- User waits for complete response before seeing anything
- No progressive rendering
- Poor perceived performance

**Impact:** Medium - Affects user experience, not actual speed

**Solution:**
- **Stream LLM responses**: Show tokens as they're generated
- **Progressive UI updates**: Update UI incrementally
- **Skeleton loading**: Show loading placeholders

---

### 10. **Redundant Operations**
**Problem:**
- Schema summary generated even when not needed
- Multiple database connections for same operation
- Re-initialization of objects that don't change

**Impact:** Medium - Wastes resources

**Solution:**
- **Lazy loading**: Only load what's needed when needed
- **Singleton pattern**: Reuse expensive objects
- **Conditional execution**: Skip operations when not needed

---

## Performance Improvement Strategy (Priority Order)

### Phase 1: Quick Wins (High Impact, Low Effort)
1. **Add Streamlit caching** for `initialize_system()` and schema summary
2. **Implement in-memory cache** with periodic persistence
3. **Cache schema summary** (only refresh on demand)
4. **Add connection pooling** for database

**Expected Improvement:** 50-70% faster response times

---

### Phase 2: Medium Effort (High Impact)
1. **Optimize cache operations** (in-memory + async persistence)
2. **Aggressive intent caching** (cache more, longer)
3. **Query result caching** (cache SQL and RAG results)
4. **Optimize vector search** (tune ChromaDB, reduce top_k)

**Expected Improvement:** Additional 20-30% improvement

---

### Phase 3: Advanced Optimizations (Medium Impact)
1. **Use faster classification model** (smaller model for intent)
2. **Parallel processing** (run operations in parallel)
3. **Streaming responses** (improve perceived performance)
4. **Database query optimization** (indexes, query tuning)

**Expected Improvement:** Additional 10-20% improvement

---

## Specific Implementation Recommendations

### 1. Cache Manager Optimization
**Current:** File-based JSON cache
**Recommended:** 
- In-memory dictionary (fast access)
- Background thread for persistence (non-blocking)
- LRU eviction policy (limit memory usage)
- Optional: Redis for distributed caching

**Code Pattern:**
```python
class CacheManager:
    def __init__(self):
        self._memory_cache = {}  # Fast in-memory cache
        self._dirty = False  # Track if cache needs saving
        self._load_from_disk()  # Load existing cache
        
    def get_cached_response(self, query):
        # Fast in-memory lookup
        return self._memory_cache.get(key)
    
    def set_cached_response(self, query, response):
        # Fast in-memory write
        self._memory_cache[key] = response
        self._dirty = True
        # Persist in background (non-blocking)
        self._persist_async()
```

---

### 2. Streamlit Caching
**Current:** No caching of expensive operations
**Recommended:**
```python
@st.cache_resource
def get_llm_client(model_name, temperature):
    # Cache LLM client
    return setup_llm_engine(...)

@st.cache_resource
def get_vector_index(chroma_path):
    # Cache vector index
    return VectorStoreIndex.from_vector_store(...)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_schema_summary():
    # Cache schema summary
    return db_mgr.get_schema_summary()
```

---

### 3. Schema Summary Caching
**Current:** Generated on every app start
**Recommended:**
- Store in config file or separate cache
- Only regenerate when user clicks "Scan Schema"
- Use `@st.cache_data` with long TTL
- Lazy load: Only generate when needed

**Code Pattern:**
```python
# In Settings page
if st.button("Scan & Ingest Schema"):
    schema = db_mgr.scan_schema()
    config.set("schema_summary", schema)  # Store in config
    st.cache_data.clear()  # Clear cache

# In Home.py
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_cached_schema_summary():
    # Load from config or generate
    return config.get("schema_summary") or db_mgr.get_schema_summary()
```

---

### 4. Intent Classification Optimization
**Current:** LLM call for every query
**Recommended:**
- More aggressive caching (cache similar queries)
- Use smaller/faster model for classification
- Cache based on query similarity (not exact match)
- Pre-classify common patterns

**Code Pattern:**
```python
def classify_with_similarity_cache(self, query):
    # Check cache for similar queries
    similar_query = self._find_similar_cached_query(query)
    if similar_query:
        return self._memory_cache[similar_query]
    
    # Only call LLM if not cached
    result = self.llm.classify(query)
    self._cache_similar_queries(query, result)
    return result
```

---

### 5. Database Connection Pooling
**Current:** New connection for each operation
**Recommended:**
```python
# Singleton database engine with connection pool
_db_engine = None

def get_db_engine():
    global _db_engine
    if _db_engine is None:
        _db_engine = create_engine(
            uri,
            pool_size=5,  # Reuse connections
            max_overflow=10,
            pool_pre_ping=True  # Verify connections
        )
    return _db_engine
```

---

### 6. Query Result Caching
**Current:** No caching of SQL/RAG results
**Recommended:**
- Cache SQL query results (same query = same result)
- Cache RAG search results (same query = same context)
- Invalidate cache when data changes

**Code Pattern:**
```python
def query_with_cache(self, query, query_type):
    cache_key = f"{query_type}:{hash(query)}"
    
    # Check cache
    cached = cache_mgr.get(cache_key)
    if cached:
        return cached
    
    # Execute query
    result = self._execute_query(query)
    
    # Cache result
    cache_mgr.set(cache_key, result, ttl=3600)
    return result
```

---

### 7. Vector Search Optimization
**Current:** Full similarity search every time
**Recommended:**
- Reduce `similarity_top_k` if not needed
- Cache query embeddings
- Use HNSW index in ChromaDB
- Pre-filter by metadata if possible

**Code Pattern:**
```python
# Cache query embeddings
@st.cache_data
def get_cached_embedding(query):
    return embed_model.get_query_embedding(query)

# Use cached embedding for search
embedding = get_cached_embedding(query)
results = vector_store.query(embedding, top_k=3)  # Reduced from 5
```

---

## Performance Metrics to Track

1. **Response Time**: Time from query to response
2. **Cache Hit Rate**: Percentage of queries served from cache
3. **LLM Call Count**: Number of LLM calls per query
4. **Database Query Time**: Time spent on database operations
5. **Vector Search Time**: Time spent on similarity search
6. **Memory Usage**: Cache size and memory consumption

---

## Expected Performance Improvements

### Current Performance (Estimated)
- First query: 5-10 seconds
- Cached query: 2-4 seconds
- Intent classification: 1-3 seconds
- Database query: 1-5 seconds
- RAG query: 2-5 seconds

### After Phase 1 Optimizations
- First query: 2-4 seconds (50-60% faster)
- Cached query: 0.5-1 second (75% faster)
- Intent classification: 0.1-0.5 seconds (cached)
- Database query: 0.5-2 seconds (pooled)
- RAG query: 1-3 seconds (optimized)

### After All Optimizations
- First query: 1-2 seconds (80% faster)
- Cached query: 0.2-0.5 seconds (90% faster)
- Intent classification: 0.05-0.2 seconds
- Database query: 0.3-1 second
- RAG query: 0.5-2 seconds

---

## Implementation Priority

1. **Critical (Do First)**
   - Streamlit caching for expensive operations
   - In-memory cache with persistence
   - Schema summary caching

2. **High Priority**
   - Connection pooling
   - Aggressive intent caching
   - Query result caching

3. **Medium Priority**
   - Vector search optimization
   - Streaming responses
   - Database query optimization

4. **Low Priority (Nice to Have)**
   - Faster classification model
   - Parallel processing
   - Advanced caching strategies

---

## Notes

- **Streamlit Limitations**: Streamlit reruns the entire script on each interaction, so caching is crucial
- **Memory vs Speed Trade-off**: More caching = faster but more memory usage
- **Cache Invalidation**: Need strategy for when to invalidate cache (data changes, config changes)
- **Testing**: Measure performance before/after each optimization
- **Monitoring**: Add performance logging to identify remaining bottlenecks

