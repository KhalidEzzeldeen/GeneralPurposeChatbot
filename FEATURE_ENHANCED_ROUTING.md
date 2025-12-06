# Enhanced Intelligent Routing Feature

## Overview

This feature replaces the simple keyword-based routing system with an LLM-based intent classification system that provides more accurate routing between knowledge base (RAG) and database (SQL) tools.

## Implementation Details

### New Module: `modules/intent_classifier.py`

The `IntentClassifier` class uses the LLM to analyze user queries and determine the appropriate tool(s) to use:

- **Intent Types:**
  - `knowledge_base`: For document-related queries
  - `database`: For quantitative/database queries
  - `both`: For queries requiring both sources
  - `unknown`: Fallback when classification is uncertain

- **Features:**
  - LLM-based classification with structured JSON response
  - Conversation history context support
  - Confidence scoring
  - Automatic fallback parsing if JSON parsing fails

### Enhanced Cache Manager

Added intent caching methods to `modules/cache.py`:
- `get_cached_intent(query)`: Retrieve cached intent classification
- `set_cached_intent(query, intent_data)`: Cache intent classification

This improves performance by avoiding redundant LLM calls for similar queries.

### Updated Routing Logic in `Home.py`

The routing system now:
1. **Primary**: Uses LLM-based intent classification
2. **Fallback**: Falls back to keyword-based routing if LLM classification fails
3. **Smart Routing**: 
   - Routes to appropriate tool based on intent
   - Supports "both" intent to combine RAG and SQL results
   - Automatically tries alternative tool if primary tool fails

## Benefits

✅ **Better Intent Understanding**: LLM understands context and nuance better than keywords  
✅ **Handles Complex Queries**: Can classify queries that don't match simple keyword patterns  
✅ **More Accurate Routing**: Reduces misrouting between tools  
✅ **Performance**: Intent caching avoids redundant LLM calls  
✅ **Reliability**: Keyword fallback ensures routing always works  

## Usage Examples

### Knowledge Base Query
```
User: "What is the company policy on remote work?"
Intent: knowledge_base
→ Routes to RAG engine
```

### Database Query
```
User: "How many users are registered in the system?"
Intent: database
→ Routes to SQL engine
```

### Combined Query
```
User: "Show me the total revenue and explain the revenue policy"
Intent: both
→ Routes to both SQL and RAG, combines results
```

## Configuration

The intent classifier uses the same LLM configured in `config.json`:
```json
{
  "llm": {
    "model_name": "qwen2.5:7b-instruct",
    "temperature": 0.0
  }
}
```

## Testing

To test the enhanced routing:

1. **Knowledge Base Queries:**
   - "What is the policy on X?"
   - "Explain the procedure for Y"
   - "Tell me about Z"

2. **Database Queries:**
   - "How many records are in table X?"
   - "List all users"
   - "What is the total count?"

3. **Complex Queries:**
   - "Show me the user count and explain the user management policy"
   - Queries that don't match simple keywords

## Performance Considerations

- **Caching**: Intent classifications are cached to avoid redundant LLM calls
- **Fallback Speed**: Keyword-based fallback is instant if LLM fails
- **Context Window**: Only uses last 6 messages for context to avoid token bloat

## Future Enhancements

- [ ] Fine-tune classification prompt based on usage patterns
- [ ] Add confidence threshold configuration
- [ ] Support for more intent types (e.g., "image_qa", "audio_qa")
- [ ] Analytics on routing accuracy

## Files Changed

- ✅ `modules/intent_classifier.py` (new)
- ✅ `modules/cache.py` (enhanced)
- ✅ `Home.py` (routing logic updated)

## Branch

This feature is implemented in: `feature/enhanced-routing`

