"""
LLM-based Intent Classifier for Enhanced Routing

This module provides intelligent query intent classification to route queries
between knowledge base (RAG) and database (SQL) tools.
"""

from llama_index.core.llms import ChatMessage
from typing import Literal, Optional
import json

IntentType = Literal["knowledge_base", "database", "both", "unknown"]


class IntentClassifier:
    """
    Classifies user queries to determine which tool(s) should be used.
    Uses LLM for intelligent classification with caching support.
    """
    
    def __init__(self, llm, cache_manager=None, schema_summary=None):
        """
        Initialize the intent classifier.
        
        Args:
            llm: The LLM instance to use for classification
            cache_manager: Optional CacheManager instance for caching intent classifications
            schema_summary: Optional database schema summary (tables and columns) to help with routing
        """
        self.llm = llm
        self.cache_manager = cache_manager
        self.schema_summary = schema_summary
        
        # Classification prompt template
        base_prompt = """You are an intelligent query router. Analyze the user's question and determine which tool should be used to answer it.

Available tools:
1. **knowledge_base** (RAG): Use for questions about documents, policies, procedures, text content, images, audio/video transcripts, general information from uploaded files.
2. **database** (SQL): Use for questions about data stored in database tables - queries about records, entities, transactions, services, or any structured data.

{schema_context}

IMPORTANT: The sample data shown above is ONLY for understanding what TYPE of data the table contains, NOT for exact matching. The sample data illustrates the table's purpose and domain, but the table may contain many more records than what's shown in the samples.

Classification logic:
1. **First, determine the table's purpose/domain from the schema:**
   - Look at the table name, column names, and sample data to understand what TYPE of data the table stores
   - Example: If "publicservices" table has columns like "service_name_english", "steps_arabic", and samples show municipal services → this table stores municipal services
   - Example: If "sales" table has columns like "property_id", "price", and samples show property sales → this table stores property sales data
   - **DO NOT** check if the specific entity in the query appears in the sample data - samples are just examples!

2. **Then, check if the user's query matches the table's purpose/domain:**
   - If the user asks about something that matches the table's purpose/domain → route to **database**
   - Example: User asks "Approval Engineering Drawings service" and table stores municipal services → route to **database** (even if "Approval Engineering Drawings" is not in samples)
   - Example: User asks "Parking Subscription steps" and table stores municipal services → route to **database** (even if not in samples)
   - Example: User asks about a property sale and table stores property sales → route to **database**
   - **Key principle**: Match by domain/purpose, NOT by exact presence in sample data

3. **Check if it's about documents/policies:**
   - Questions about "what is", "explain", "tell me about" general policies, procedures, or concepts → **knowledge_base**
   - Questions asking for specific data records, entities, or structured information that matches a table's domain → **database**
   - If the query is about a concept/explanation AND there's no matching table domain → **knowledge_base**

4. **Special cases:**
   - Questions that need both structured data AND document context → **both**
   - If the query could be answered by either, prefer **database** if it matches a table's domain, otherwise **knowledge_base**

Key insight: The user doesn't know where the information is stored. Your job is to understand:
1. What TYPE of data each table stores (from table name, columns, and sample data patterns)
2. Whether the user's query is asking about that TYPE of data
3. Route to database if there's a domain match, regardless of whether the specific entity appears in samples

User query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{
    "intent": "knowledge_base" | "database" | "both" | "unknown",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why this intent was chosen, focusing on table purpose/domain match rather than sample data presence"
}}"""
        
        self.classification_prompt_template = base_prompt

    def classify(self, query: str, conversation_history: Optional[list] = None) -> dict:
        """
        Classify the intent of a user query.
        
        Args:
            query: The user's query string
            conversation_history: Optional list of previous messages for context
            
        Returns:
            dict with keys: intent (IntentType), confidence (float), reasoning (str)
        """
        # Aggressive caching: Check cache first (including similar queries)
        if self.cache_manager:
            # Check exact match
            cached_intent = self.cache_manager.get_cached_intent(query)
            if cached_intent:
                return cached_intent
            
            # Check for similar queries (normalized)
            normalized_query = query.lower().strip()
            cached_intent = self.cache_manager.get_cached_intent(normalized_query)
            if cached_intent:
                # Cache the normalized version for future use
                self.cache_manager.set_cached_intent(query, cached_intent)
                return cached_intent
        
        # Build schema context if available
        schema_context = ""
        if self.schema_summary:
            schema_context = f"""Database Schema Information:
{self.schema_summary}

CRITICAL: The sample data and distinct values shown above are ONLY examples to help you understand what TYPE of data each table stores. They are NOT a complete list of what exists in the database. The table may contain many more records than what's shown.

Analysis process:
1. **Understand each table's purpose/domain** (NOT what specific records exist):
   - Look at the table name to understand its general purpose
   - Examine column names to see what information is tracked
   - Review sample data to understand the TYPE/CATEGORY of data (e.g., "municipal services", "property sales", "user accounts")
   - The DISTINCT values show variety, but again, these are just examples of the data TYPE

2. **Match query to table purpose/domain** (NOT to specific sample records):
   - If the user's query is about something that matches the table's purpose/domain → route to **database**
   - Example: If "publicservices" table stores municipal services, and user asks about ANY municipal service → route to database (regardless of whether that specific service appears in samples)
   - Example: If "sales" table stores property sales, and user asks about ANY property sale → route to database
   - **DO NOT** check if the specific entity in the query appears in the sample data - that's not the purpose of samples!

3. **Key principle**: 
   - Sample data = Examples of data TYPE/purpose
   - Your job = Match query to data TYPE/purpose, not to specific sample records
   - If query matches table's domain → route to database (even if specific entity not in samples)

4. **Only route to knowledge_base if:**
   - Query is about general policies, procedures, or documents (not specific data records)
   - Query is about information clearly NOT matching any table's domain
   - Query is asking "what is" or "explain" about concepts, not data retrieval

Example 1: User asks "Approval Engineering Drawings service steps"
- Schema shows "publicservices" table stores municipal services (from table name, columns like service_name_english, steps_arabic, and sample data showing services)
- "Approval Engineering Drawings" is a municipal service → matches table's domain
- Route to database ✅ (even if "Approval Engineering Drawings" not in sample data)

Example 2: User asks "Parking Subscription steps"
- Schema shows "publicservices" table stores municipal services
- "Parking Subscription" is a municipal service → matches table's domain
- Route to database ✅ (regardless of sample data)

Example 3: User asks "What is the company policy on remote work?"
- No table's domain matches "company policies" → route to knowledge_base ✅"""
        else:
            schema_context = "No database schema information available. Use keyword-based classification."
        
        # Prepare the classification prompt
        prompt = self.classification_prompt_template.format(
            schema_context=schema_context,
            query=query
        )
        
        # Add conversation context if available
        messages = []
        if conversation_history:
            # Include last 3 exchanges for context (to avoid token bloat)
            recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            for msg in recent_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                else:
                    role = msg.role if hasattr(msg, 'role') else "user"
                    content = str(msg.content) if hasattr(msg, 'content') else str(msg)
                
                if role in ["user", "assistant"]:
                    messages.append(ChatMessage(role=role, content=content))
        
        messages.append(ChatMessage(role="user", content=prompt))
        
        try:
            # Get classification from LLM
            response = self.llm.complete(prompt)
            response_text = str(response).strip()
            
            # Try to parse JSON response
            # Sometimes LLM adds markdown code blocks, so we need to extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract intent from text
                result = self._parse_fallback(response_text)
            
            # Validate and normalize result
            intent = result.get("intent", "unknown").lower()
            if intent not in ["knowledge_base", "database", "both", "unknown"]:
                intent = "unknown"
            
            classification = {
                "intent": intent,
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "LLM classification")
            }
            
            # Cache the result
            if self.cache_manager:
                self.cache_manager.set_cached_intent(query, classification)
            
            return classification
            
        except Exception as e:
            # Fallback to unknown on error
            classification = {
                "intent": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }
            return classification
    
    def _parse_fallback(self, text: str) -> dict:
        """
        Fallback parser if JSON parsing fails.
        Tries to extract intent from natural language response.
        """
        text_lower = text.lower()
        
        if "knowledge_base" in text_lower or "knowledge base" in text_lower or "rag" in text_lower:
            intent = "knowledge_base"
        elif "database" in text_lower or "sql" in text_lower:
            intent = "database"
        elif "both" in text_lower:
            intent = "both"
        else:
            intent = "unknown"
        
        return {
            "intent": intent,
            "confidence": 0.6,
            "reasoning": "Parsed from text response"
        }


def classify_with_keywords(query: str) -> dict:
    """
    Fallback keyword-based classification.
    Used when LLM classification fails or as a backup.
    
    Args:
        query: The user's query string
        
    Returns:
        dict with intent classification
    """
    query_lower = query.lower()
    
    # Database keywords (prioritize explicit requests)
    db_keywords = [
        "from database", "query database", "database query", "show from database",  # Explicit requests (highest priority)
        "how many", "count", "list", "sum", "total", "database", 
        "table", "rows", "records", "users",
        "show all", "select", "aggregate", "get from db", "fetch from database"
    ]
    
    # Knowledge base keywords
    kb_keywords = [
        "what is", "explain", "tell me about", "describe", 
        "policy", "document", "file", "content", "information about"
    ]
    
    needs_db = any(keyword in query_lower for keyword in db_keywords)
    needs_kb = any(keyword in query_lower for keyword in kb_keywords)
    
    if needs_db and needs_kb:
        intent = "both"
        confidence = 0.7
    elif needs_db:
        intent = "database"
        confidence = 0.8
    elif needs_kb:
        intent = "knowledge_base"
        confidence = 0.8
    else:
        # Default to knowledge_base for general queries
        intent = "knowledge_base"
        confidence = 0.5
    
    return {
        "intent": intent,
        "confidence": confidence,
        "reasoning": "Keyword-based classification"
    }

