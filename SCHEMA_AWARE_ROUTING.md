# Schema-Aware Routing Enhancement

## Overview

The intent classifier now uses database schema information to make more accurate routing decisions. When the database schema is scanned (via the "Scan & Ingest Schema" button in Settings), the system knows what tables and columns exist, allowing it to better identify database-related queries.

## How It Works

### 1. Schema Scanning

When you click **"Scan & Ingest Schema"** in Settings:
- The system connects to your database
- Scans all tables and their columns
- Saves this information to `data/db_schema_scan.txt`
- Ingests it into the knowledge base

### 2. Schema Summary Extraction

The `DatabaseManager.get_schema_summary()` method extracts a concise summary:
```
Database has 2 table(s): sales, publicservices

Table 'sales': sale_id, listing_id, purpose, emirate, community, developer, ...
Table 'publicservices': service_name_arabic, service_name_english, description_arabic, ...
```

### 3. Enhanced Classification Prompt

The LLM now receives schema information in the classification prompt:

```
Database Schema Information:
Database has 2 table(s): sales, publicservices

Table 'sales': sale_id, listing_id, purpose, emirate, community, ...
Table 'publicservices': service_name_arabic, service_name_english, ...

Use this schema to identify if the query is asking about database tables/columns.
If the query mentions table names, column names, or asks about data in these tables, route to database.
```

### 4. Improved Routing Accuracy

With schema information, the LLM can now:

✅ **Recognize table/column names** in queries
- "Show me sales data" → database (knows 'sales' is a table)
- "What's in the emirate column?" → database (knows 'emirate' is a column)

✅ **Better understand database queries**
- "List all records from sales table" → database (explicit table reference)
- "How many services are in publicservices?" → database (knows 'publicservices' is a table)

✅ **Distinguish similar queries**
- "What is a sale?" → knowledge_base (general question)
- "What is in the sales table?" → database (specific table reference)

## Example Classifications

### Example 1: Table Name Recognition
```
User Query: "Show me all records from the sales table"

Schema Context: Table 'sales': sale_id, listing_id, purpose, ...

LLM Analysis:
- Query mentions "sales table" which exists in schema
- Asking for records from a specific table
→ Intent: database (confidence: 0.98)

Action: Routes to SQL Engine
```

### Example 2: Column Name Recognition
```
User Query: "What values are in the emirate column?"

Schema Context: Table 'sales': ..., emirate, ...

LLM Analysis:
- Query mentions "emirate column" which exists in schema
- Asking about column data
→ Intent: database (confidence: 0.95)

Action: Routes to SQL Engine
```

### Example 3: Ambiguous Query (Better Resolution)
```
User Query: "Tell me about sales"

Without Schema:
- Ambiguous: Could be about sales policies (knowledge_base) or sales data (database)
→ Intent: knowledge_base (confidence: 0.60)

With Schema:
- Schema shows 'sales' is a table name
- Query likely refers to sales data
→ Intent: database (confidence: 0.85)

Action: Routes to SQL Engine (more accurate!)
```

### Example 4: General Query (Still Works)
```
User Query: "What is the company's sales policy?"

Schema Context: Table 'sales': ...

LLM Analysis:
- Query asks about "sales policy" (document/policy content)
- Not asking about table data
→ Intent: knowledge_base (confidence: 0.90)

Action: Routes to RAG Engine (correctly distinguishes!)
```

## Benefits

1. **Higher Accuracy**: Schema-aware routing reduces misrouting
2. **Table/Column Recognition**: Can identify database queries by table/column names
3. **Better Context**: LLM understands what data is available in the database
4. **Reduced Ambiguity**: Schema helps resolve ambiguous queries
5. **Automatic**: Works automatically once schema is scanned

## Implementation Details

### Files Modified

1. **`modules/database.py`**
   - Added `get_schema_summary()` method
   - Extracts table names and columns without needing file access

2. **`modules/intent_classifier.py`**
   - Added `schema_summary` parameter to `__init__()`
   - Enhanced classification prompt to include schema context
   - Uses schema to identify table/column references

3. **`Home.py`**
   - Gets schema summary when initializing intent classifier
   - Passes schema information to classifier

### Code Flow

```
1. User clicks "Scan & Ingest Schema" in Settings
   ↓
2. DatabaseManager.scan_schema() scans database
   ↓
3. Schema saved to data/db_schema_scan.txt and ingested
   ↓
4. On app start, DatabaseManager.get_schema_summary() extracts summary
   ↓
5. IntentClassifier receives schema_summary
   ↓
6. Classification prompt includes schema context
   ↓
7. LLM uses schema to make better routing decisions
```

## Usage

### Step 1: Scan Schema
1. Go to Settings → Database Connection tab
2. Configure database connection
3. Click **"Scan & Ingest Schema"**
4. Wait for scan to complete

### Step 2: Use Enhanced Routing
The system automatically uses schema information for routing. No additional configuration needed!

### Step 3: Test
Try queries like:
- "Show me data from the sales table"
- "What columns are in publicservices?"
- "Count records in sales"
- "What is the sales policy?" (should route to knowledge_base)

## Fallback Behavior

If schema information is not available:
- System falls back to keyword-based classification
- Still works, just without schema awareness
- No errors or failures

## Future Enhancements

- [ ] Cache schema summary to avoid repeated database connections
- [ ] Show schema info in UI for user reference
- [ ] Auto-refresh schema when database structure changes
- [ ] Support for schema changes detection

