# How the System Determines Database vs Knowledge Base Requests

## Overview

The system uses a **two-tier classification approach**:
1. **Primary**: LLM-based intelligent classification (understands context and intent)
2. **Fallback**: Keyword-based classification (fast and reliable backup)

---

## 🔄 Classification Flow

```
User Query
    ↓
┌─────────────────────────────────────┐
│  Step 1: Check Cache                │
│  (Has this query been classified     │
│   before?)                           │
└─────────────────────────────────────┘
    ↓ (Not cached)
┌─────────────────────────────────────┐
│  Step 2: LLM Classification          │
│  (Send query to LLM with prompt)     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Step 3: Parse LLM Response         │
│  (Extract intent from JSON)          │
└─────────────────────────────────────┘
    ↓ (If LLM fails)
┌─────────────────────────────────────┐
│  Step 4: Keyword Fallback           │
│  (Match keywords in query)           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Step 5: Route to Tool               │
│  - database → SQL Engine            │
│  - knowledge_base → RAG Engine       │
│  - both → Both tools                 │
└─────────────────────────────────────┘
```

---

## 🧠 Method 1: LLM-Based Classification (Primary)

### How It Works

The LLM receives a **structured prompt** that explains the two tools and classification rules:

```python
# The prompt sent to LLM (from intent_classifier.py, line 33-52)
"""
You are an intelligent query router. Analyze the user's question and determine 
which tool should be used to answer it.

Available tools:
1. **knowledge_base** (RAG): Use for questions about documents, policies, 
   text content, images, audio/video transcripts, general information from 
   uploaded files.

2. **database** (SQL): Use for quantitative queries, counting, aggregations, 
   listing records, querying structured data from database tables.

Classification rules:
- Questions about "how many", "count", "sum", "total", "list all", 
  "show records" → database
- Questions about policies, documents, "what is", "explain", 
  "tell me about" → knowledge_base
- Questions that need both structured data AND document context → both
- If uncertain, prefer knowledge_base as it's more general

User query: "{query}"

Respond with ONLY a JSON object:
{
    "intent": "knowledge_base" | "database" | "both" | "unknown",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}
"""
```

### Example LLM Classifications

#### Example 1: Database Query
```
User: "How many users are registered in the system?"

LLM Response:
{
    "intent": "database",
    "confidence": 0.95,
    "reasoning": "Query asks for a count of records, which requires database query"
}

→ Routes to SQL Engine
```

#### Example 2: Knowledge Base Query
```
User: "What is the company policy on remote work?"

LLM Response:
{
    "intent": "knowledge_base",
    "confidence": 0.92,
    "reasoning": "Question about policy document content, needs RAG retrieval"
}

→ Routes to RAG Engine
```

#### Example 3: Complex Query (Both)
```
User: "Show me the total revenue and explain the revenue calculation policy"

LLM Response:
{
    "intent": "both",
    "confidence": 0.88,
    "reasoning": "Needs database for revenue total and knowledge base for policy explanation"
}

→ Routes to Both SQL and RAG, combines results
```

#### Example 4: Ambiguous Query
```
User: "Tell me about users"

LLM Response:
{
    "intent": "knowledge_base",
    "confidence": 0.65,
    "reasoning": "Ambiguous query, defaulting to knowledge base for general information"
}

→ Routes to RAG Engine (with low confidence warning)
```

---

## 🔍 Method 2: Keyword-Based Classification (Fallback)

### When It's Used

- LLM classification fails (error, timeout)
- LLM returns invalid JSON
- System is in fallback mode

### How It Works

The system checks for specific keywords in the query:

```python
# Database keywords (from intent_classifier.py, line 174-178)
db_keywords = [
    "how many", "count", "list", "sum", "total", "database", 
    "table", "rows", "records", "users", "from database",
    "show all", "select", "query database", "aggregate"
]

# Knowledge base keywords (line 181-184)
kb_keywords = [
    "what is", "explain", "tell me about", "describe", 
    "policy", "document", "file", "content", "information about"
]
```

### Classification Logic

```python
needs_db = any(keyword in query_lower for keyword in db_keywords)
needs_kb = any(keyword in query_lower for keyword in kb_keywords)

if needs_db and needs_kb:
    intent = "both"
elif needs_db:
    intent = "database"
elif needs_kb:
    intent = "knowledge_base"
else:
    intent = "knowledge_base"  # Default fallback
```

### Example Keyword Classifications

```
Query: "How many users are there?"
→ Matches "how many" → database

Query: "What is the policy on X?"
→ Matches "what is" and "policy" → knowledge_base

Query: "List all users and explain the user management policy"
→ Matches "list" (db) and "explain" + "policy" (kb) → both

Query: "Hello"
→ No keywords matched → knowledge_base (default)
```

---

## 📊 Decision Matrix

| Query Type | Keywords Present | LLM Understanding | Final Intent |
|------------|------------------|-------------------|--------------|
| "How many users?" | "how many" | Count query | **database** |
| "What is the policy?" | "what is", "policy" | Document query | **knowledge_base** |
| "List users and explain policy" | "list" + "explain" | Both needed | **both** |
| "Tell me about X" | "tell me about" | General info | **knowledge_base** |
| "Count records in table" | "count", "table" | Database query | **database** |
| "Explain the procedure" | "explain" | Document query | **knowledge_base** |

---

## 🎯 Key Differences: LLM vs Keywords

### LLM Classification Advantages ✅

1. **Context Understanding**: Understands meaning, not just words
   - "Show me user statistics" → database (understands "statistics" = count/aggregate)
   - "What do users say about X?" → knowledge_base (understands "say" = content)

2. **Handles Ambiguity**: Can reason about unclear queries
   - "Tell me about users" → Can infer if it's about user data (db) or user policies (kb)

3. **Conversation Context**: Uses previous messages for better classification
   - Previous: "How many users?" → database
   - Follow-up: "What are their roles?" → Can infer still database-related

4. **Complex Queries**: Understands when both tools are needed
   - "Show revenue and explain the calculation method" → both

### Keyword Classification Advantages ✅

1. **Fast**: No LLM call needed
2. **Reliable**: Always works, no API dependency
3. **Predictable**: Same keywords always route the same way
4. **Fallback Safety**: Works when LLM is unavailable

---

## 🔧 Implementation Details

### Code Flow in Home.py

```python
# Step 1: Try LLM classification
if chat_engine.get("intent_classifier"):
    intent_classification = chat_engine["intent_classifier"].classify(
        query=prompt,
        conversation_history=conversation_history
    )

# Step 2: Fallback to keywords if LLM fails
if not intent_classification:
    intent_classification = classify_with_keywords(prompt)

# Step 3: Extract intent
intent = intent_classification.get("intent", "knowledge_base")

# Step 4: Route based on intent
if intent == "database":
    # Use SQL engine
elif intent == "both":
    # Use both SQL and RAG
else:
    # Use RAG engine (knowledge_base)
```

### Caching

Intent classifications are **cached** to avoid redundant LLM calls:

```python
# Check cache first
cached_intent = cache_manager.get_cached_intent(query)
if cached_intent:
    return cached_intent  # Return immediately, no LLM call

# After LLM classification
cache_manager.set_cached_intent(query, classification)
```

This means:
- First time: "How many users?" → LLM call → database
- Second time: "How many users?" → Cache hit → database (instant)

---

## 📝 Real-World Examples

### Example 1: Clear Database Query
```
User: "Count all records in the users table"

LLM Analysis:
- "Count" = quantitative operation
- "records in table" = structured data
→ Intent: database (confidence: 0.98)

Action: Routes to SQL Engine
```

### Example 2: Clear Knowledge Base Query
```
User: "Explain the company's remote work policy"

LLM Analysis:
- "Explain" = informational request
- "policy" = document content
→ Intent: knowledge_base (confidence: 0.95)

Action: Routes to RAG Engine
```

### Example 3: Ambiguous Query
```
User: "Tell me about users"

LLM Analysis:
- Could mean: user data (database) or user policies (knowledge base)
- Context: Previous conversation about policies
→ Intent: knowledge_base (confidence: 0.70)

Action: Routes to RAG Engine (with low confidence)
```

### Example 4: Combined Query
```
User: "Show me the total revenue and explain how it's calculated"

LLM Analysis:
- "total revenue" = database query (aggregation)
- "explain how it's calculated" = knowledge base (documentation)
→ Intent: both (confidence: 0.90)

Action: Routes to both SQL and RAG, combines results
```

---

## 🎓 Summary

The system determines database vs knowledge base requests through:

1. **LLM Intelligence**: Understands context, meaning, and intent
2. **Keyword Matching**: Fast fallback for reliability
3. **Caching**: Performance optimization
4. **Conversation Context**: Uses previous messages for better classification
5. **Confidence Scoring**: Indicates how certain the classification is

The combination of LLM + keyword fallback ensures:
- ✅ **Smart routing** for complex queries (LLM)
- ✅ **Reliable routing** when LLM fails (keywords)
- ✅ **Fast routing** through caching
- ✅ **Flexible routing** for combined queries (both)

