# Routing Fix for Database Queries

## Problem
After implementing schema-aware routing, database queries like "from database" and "Drainage Suction" were being incorrectly routed to the knowledge base instead of the database, even though they worked before.

## Root Cause
The LLM-based classification was overriding explicit database keyword requests. When users said "from database" or made follow-up queries after "from database", the system wasn't prioritizing these explicit signals.

## Solution

### 1. Explicit Keyword Priority
Added explicit database keyword checking that happens **BEFORE** LLM classification:

```python
explicit_db_keywords = ["from database", "query database", "database query", "show from database"]
has_explicit_db_request = any(keyword in prompt_lower for keyword in explicit_db_keywords)
```

If explicit keywords are detected, the system routes directly to database with high confidence (0.95), bypassing LLM classification.

### 2. Follow-up Query Detection
Detects if the previous message contained a database request:

```python
if len(st.session_state.messages) >= 2:
    prev_user_msg = st.session_state.messages[-2].get("content", "").lower()
    previous_was_db_request = any(keyword in prev_user_msg for keyword in explicit_db_keywords)
```

This ensures that follow-up queries like "Drainage Suction" after "from database" are correctly routed to the database.

### 3. Better Error Handling
Instead of silently falling back to RAG when SQL fails, the system now:
- Shows the actual SQL error message
- Indicates that routing was correct but SQL execution failed
- Provides helpful error context

### 4. Debug Information
Added routing debug info to help troubleshoot:
```
🔀 Routing: database (confidence: 0.95) - Explicit database request detected
```

## How It Works Now

### Scenario 1: Explicit Database Request
```
User: "from database"
→ Detects "from database" keyword
→ Routes to database (confidence: 0.95)
→ Executes SQL query
```

### Scenario 2: Follow-up Query
```
User: "from database"
→ Routes to database

User: "Drainage Suction"
→ Detects previous message had "from database"
→ Routes to database (follow-up)
→ Executes SQL query for "Drainage Suction"
```

### Scenario 3: LLM Classification (when no explicit keywords)
```
User: "How many records are in the sales table?"
→ No explicit keywords
→ LLM classifies based on query content
→ Routes to database (if LLM determines it's a database query)
```

## Testing

To verify the fix works:

1. **Test explicit request:**
   - Query: "from database"
   - Should route to database immediately

2. **Test follow-up:**
   - Query 1: "from database"
   - Query 2: "Drainage Suction"
   - Query 2 should route to database

3. **Test error handling:**
   - If SQL fails, should show error message
   - Should NOT silently fall back to RAG

## Files Modified

- `Home.py`: Added explicit keyword checking and follow-up detection
- `modules/intent_classifier.py`: Enhanced keyword list with explicit database requests

