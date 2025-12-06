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

Classification logic:
1. **First, check if the query is about data in the database:**
   - Look at the sample data in the schema above
   - If the user's query mentions entities, services, records, or data that matches what's in the sample data → route to **database**
   - Examples: If schema shows "publicservices" table with service names, and user asks about a service → **database**
   - Examples: If schema shows "sales" table with property data, and user asks about properties → **database**

2. **Then check if it's about documents/policies:**
   - Questions about "what is", "explain", "tell me about" policies, procedures, documents → **knowledge_base**
   - General informational questions not about specific data records → **knowledge_base**

3. **Special cases:**
   - Questions that need both structured data AND document context → **both**
   - If the query could be answered by either, prefer **database** if it matches sample data, otherwise **knowledge_base**

Key insight: The user doesn't know where the information is stored. Your job is to understand what they're asking about and match it to the available data sources based on the schema and sample data shown above.

User query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{
    "intent": "knowledge_base" | "database" | "both" | "unknown",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why this intent was chosen, referencing schema data if relevant"
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
        # Check cache first
        if self.cache_manager:
            cached_intent = self.cache_manager.get_cached_intent(query)
            if cached_intent:
                return cached_intent
        
        # Build schema context if available
        schema_context = ""
        if self.schema_summary:
            schema_context = f"""Database Schema Information:
{self.schema_summary}

Use this schema to identify if the query is asking about database tables/columns. If the query mentions table names, column names, or asks about data in these tables, route to database."""
        else:
            schema_context = "No database schema information available."
        
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

