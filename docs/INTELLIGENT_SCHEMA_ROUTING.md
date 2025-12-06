# Intelligent Schema-Aware Routing

## Overview

The system now uses **intelligent schema understanding** to route queries, rather than relying on keywords. The LLM analyzes the actual database schema and sample data to determine if a query is asking about information that exists in the database.

## How It Works

### 1. Enhanced Schema Summary

The `get_schema_summary()` method now includes:

- **Table names**: All tables in the database
- **Column names**: All columns with their types
- **Sample data**: Actual values from text columns (up to 5 rows)
- **Content examples**: Shows what kind of data is stored (e.g., service names, locations, etc.)

Example schema summary:
```
Database contains 2 table(s): sales, publicservices

**Table: publicservices**
Columns: service_name_arabic, service_name_english, description_english, ...
Sample data:
  Example row 1: service_name_english='Request for Drainage Suction - Residential Houses', description_english='Residents of Sharjah can request...'
  Example row 2: service_name_english='Rental Indicators Inquiry Service', ...
  Sample service_name_english values: Request for Drainage Suction, Rental Indicators Inquiry Service, ...
```

### 2. Intelligent Classification Prompt

The LLM receives detailed instructions to analyze the schema:

```
Analysis process:
1. Look at the sample data in each table - what actual content/values are stored?
2. Check if the user's query mentions or asks about:
   - Any table names (e.g., "sales", "publicservices")
   - Any column names (e.g., "service_name_english", "emirate")
   - Any actual data values that appear in the sample data (e.g., "Drainage Suction", "Dubai")
3. If the query is asking about information that appears to be in the database tables, route to database.
4. If the query is asking about general policies or documents, route to knowledge_base.
```

### 3. Content Matching

The LLM can now match queries to actual database content:

**Example 1: Service Name Match**
```
User Query: "Drainage Suction"

Schema Analysis:
- publicservices table has service_name_english column
- Sample data shows: "Request for Drainage Suction - Residential Houses"
- Query matches actual database content

→ Intent: database (confidence: 0.95)
→ Routes to SQL Engine ✅
```

**Example 2: Location Match**
```
User Query: "Show me sales in Dubai"

Schema Analysis:
- sales table has emirate column
- Sample data shows emirate values include "Dubai"
- Query is asking about data in sales table

→ Intent: database (confidence: 0.92)
→ Routes to SQL Engine ✅
```

**Example 3: General Policy Query**
```
User Query: "What is the company's remote work policy?"

Schema Analysis:
- No table contains "remote work policy"
- No column or sample data matches this query
- This is likely a document/policy question

→ Intent: knowledge_base (confidence: 0.90)
→ Routes to RAG Engine ✅
```

## Benefits

✅ **Intelligent Understanding**: LLM understands what data exists in the database  
✅ **Content Matching**: Matches queries to actual database values, not just keywords  
✅ **No User Knowledge Required**: Users don't need to know if data is in database or knowledge base  
✅ **Accurate Routing**: Reduces misrouting by understanding actual content  
✅ **Natural Queries**: Works with natural language queries like "Drainage Suction"  

## Comparison: Before vs After

### Before (Keyword-Based)
```
User: "Drainage Suction"
→ No keywords matched
→ Routes to knowledge_base ❌
```

### After (Schema-Aware)
```
User: "Drainage Suction"
→ LLM sees "Drainage Suction" in publicservices.service_name_english sample data
→ Understands this is database content
→ Routes to database ✅
```

## Implementation Details

### Schema Summary Enhancement

The `get_schema_summary()` method now:
1. Fetches sample data (5 rows) from each table
2. Extracts text column values (service names, descriptions, etc.)
3. Shows example rows with actual content
4. Lists sample values from key text columns

### Classification Prompt

The intent classifier prompt now:
1. Explains the analysis process step-by-step
2. Instructs LLM to examine sample data
3. Provides examples of content matching
4. Guides LLM to route based on actual data understanding

## Testing

To verify intelligent routing:

1. **Scan Schema**: Go to Settings → Database → "Scan & Ingest Schema"
2. **Test Service Query**: "Drainage Suction" → Should route to database
3. **Test Location Query**: "Show sales in Dubai" → Should route to database
4. **Test Policy Query**: "What is the company policy?" → Should route to knowledge base

## Key Insight

The system no longer relies on users saying "from database" or matching keywords. Instead, it **intelligently understands** what data exists in the database by examining:
- Table structure
- Column names
- **Actual sample data values**

This allows natural queries like "Drainage Suction" to be correctly identified as database queries because the LLM can see that "Drainage Suction" appears in the database schema sample data.

