import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from modules.llm_engine import setup_llm_engine
from modules.session import SessionManager
from modules.cache import CacheManager
from modules.config import ConfigManager
from modules.intent_classifier import IntentClassifier, classify_with_keywords
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
# ReActAgent removed - using enhanced LLM-based router instead to avoid async workflow issues
# from llama_index.core.agent import ReActAgent
import uuid
import os
import concurrent.futures
import time

# Page Config
st.set_page_config(page_title="ProBot - Enterprise Assistant", layout="wide")

# Streamlit caching for expensive operations
# Note: Cache key includes model_name to ensure different models are cached separately
@st.cache_resource
def get_llm_and_embedding(model_name, embed_model_name, temperature):
    """Cache LLM and embedding model initialization. Model-specific caching."""
    return setup_llm_engine(
        model_name=model_name, 
        embed_model_name=embed_model_name,
        temperature=temperature
    )

@st.cache_resource
def get_vector_index(chroma_path, embed_model):
    """Cache vector index initialization."""
    if not os.path.exists(chroma_path):
        return None
        
    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection("chatbot_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )
    return index

def initialize_system(model_name, temperature):
    """Initialize system using cached components."""
    # Get cached LLM and embedding
    llm, embed_model = get_llm_and_embedding(
        model_name=model_name,
        embed_model_name="bge-m3",
        temperature=temperature
    )
    
    config = ConfigManager()
    chroma_path = config.get("chroma_path")
    
    # Get cached vector index
    index = get_vector_index(chroma_path, embed_model)
    
    return index, llm

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_cached_schema_summary():
    """Cache schema summary - only refresh when explicitly requested."""
    from modules.database import DatabaseManager
    db_mgr = DatabaseManager()
    return db_mgr.get_schema_summary()

@st.cache_resource
def get_sql_query_engine(_llm):
    """Cache SQL query engine initialization."""
    from modules.database import DatabaseManager
    db_mgr = DatabaseManager()
    if not db_mgr.get_connection_string():
        return None
    return db_mgr.get_sql_query_engine(_llm)

def is_context_dependent_query(prompt: str) -> bool:
    """
    Check if a query contains references that require conversation context.
    Examples: "this service", "it", "that", "the above", "those", etc.
    """
    prompt_lower = prompt.lower()
    context_indicators = [
        "this ", "that ", "these ", "those ",
        " it ", " its ", " it's ",
        " the above", " the previous", " the mentioned",
        " for this", " for that", " for it",
        " of this", " of that", " of it",
        " what are the steps", " how to", " tell me more",
        " explain it", " describe it", " details about it"
    ]
    return any(indicator in prompt_lower for indicator in context_indicators)

def resolve_context_references(prompt: str, conversation_history: list, llm) -> str:
    """
    Resolve context-dependent references in a query using conversation history.
    For example: "what are the steps for this service" -> "what are the steps for Approval Engineering Drawings service"
    
    Returns the resolved query if context-dependent, otherwise returns the original prompt.
    """
    if not is_context_dependent_query(prompt):
        return prompt
    
    if not conversation_history or len(conversation_history) < 3:
        return prompt
    
    # Get recent conversation context (exclude current message which is the last one)
    # We want the previous messages to understand what "this service" refers to
    recent_history = conversation_history[-7:-1] if len(conversation_history) > 7 else conversation_history[:-1]
    
    # Need at least one previous exchange (user + assistant) to have context
    if len(recent_history) < 2:
        return prompt
    
    # Build context summary
    context_parts = []
    for msg in recent_history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = msg.role if hasattr(msg, 'role') else ""
            content = str(msg.content) if hasattr(msg, 'content') else str(msg)
        
        if role == "user":
            context_parts.append(f"User: {content}")
        elif role == "assistant":
            # Extract key information from assistant responses (service names, entities, etc.)
            context_parts.append(f"Assistant: {content[:200]}")  # First 200 chars to capture service names
    
    if not context_parts:
        return prompt
    
    context = "\n".join(context_parts)
    
    # Use LLM to resolve references
    resolution_prompt = f"""You are a context resolution assistant. Your job is to resolve references like "this service", "it", "that", etc. in the user's query by replacing them with the actual entities mentioned in the conversation context.

Conversation context (previous messages):
{context}

User's current query: "{prompt}"

Task: Replace any references (like "this service", "it", "that", "the above", etc.) in the user's query with the actual entities, services, or items mentioned in the conversation context.

Rules:
1. If the query says "this service" or "for this service", find the service name mentioned in the conversation context (usually in the Assistant's previous response or User's previous question)
2. If the query says "it" or "that", find what it refers to from the context
3. Keep the rest of the query exactly as is - only replace the reference words
4. If you cannot determine what the reference refers to, return the original query unchanged
5. Only replace references, don't change the query structure or meaning
6. Be specific - if the context mentions "Approval Engineering Drawings service", replace "this service" with "Approval Engineering Drawings service"

Examples:
- Query: "what are the steps for this service" + Context mentions "Approval Engineering Drawings service" → "what are the steps for Approval Engineering Drawings service"
- Query: "tell me more about it" + Context mentions "Contaminated soil removal" → "tell me more about Contaminated soil removal"

Return ONLY the resolved query, nothing else. Do not add explanations or additional text.

Resolved query:"""
    
    try:
        response = llm.complete(resolution_prompt)
        resolved = str(response).strip()
        
        # Clean up the response (remove quotes, extra text, etc.)
        # Remove markdown code blocks if present
        if "```" in resolved:
            # Extract text between code blocks
            parts = resolved.split("```")
            if len(parts) > 1:
                resolved = parts[1].strip()
                if resolved.startswith("text") or resolved.startswith("plain"):
                    resolved = resolved.split("\n", 1)[-1].strip()
        
        resolved = resolved.strip('"').strip("'").strip()
        
        # Remove any explanatory text that might come after the query
        # Look for common patterns like "The resolved query is:" etc.
        lines = resolved.split("\n")
        if len(lines) > 1:
            # Take the first line that looks like a query (has question words or is substantial)
            for line in lines:
                line = line.strip()
                if line and (len(line) > 10 or any(word in line.lower() for word in ["what", "how", "tell", "show", "get", "find"])):
                    resolved = line
                    break
        
        # If the resolved query is too different or empty, use original
        if not resolved or len(resolved) < len(prompt) * 0.5:
            return prompt
        
        # If resolved query is the same as original, return it (might be correct if no references)
        if resolved.lower() == prompt.lower():
            return prompt
        
        return resolved
    except Exception as e:
        # If resolution fails, return original prompt
        import logging
        logging.warning(f"Context resolution failed: {str(e)}")
        return prompt

def get_conversation_context_hash(conversation_history: list, max_messages: int = 4) -> str:
    """
    Generate a hash of recent conversation context for cache key generation.
    This ensures context-dependent queries use different cache keys based on conversation history.
    """
    if not conversation_history or len(conversation_history) < 3:
        return ""
    
    # Get last few messages (excluding current) for context
    recent = conversation_history[-max_messages-1:-1] if len(conversation_history) > max_messages+1 else conversation_history[:-1]
    
    # Extract key information from recent messages
    context_texts = []
    for msg in recent:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = str(msg.content) if hasattr(msg, 'content') else str(msg)
        # Extract first 50 chars of each message for context hash
        if content:
            context_texts.append(content[:50])
    
    if context_texts:
        import hashlib
        context_str = "|".join(context_texts)
        return hashlib.md5(context_str.encode()).hexdigest()[:8]
    return ""

def build_context_aware_query(prompt: str, conversation_history: list) -> str:
    """
    Build a context-aware query by including relevant conversation history.
    This helps the LLM understand references like "this service", "it", "that", etc.
    """
    if not conversation_history or len(conversation_history) < 3:
        # No history or only current message, return prompt as-is
        return prompt
    
    # Get the last few exchanges, excluding the current user message (last item)
    # Get last 5 messages before current (2-3 Q&A pairs)
    recent_history = conversation_history[-6:-1] if len(conversation_history) > 6 else conversation_history[:-1]
    
    # Build context from recent history (excluding current message)
    context_parts = []
    for msg in recent_history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = msg.role if hasattr(msg, 'role') else ""
            content = str(msg.content) if hasattr(msg, 'content') else str(msg)
        
        if role == "user":
            context_parts.append(f"User: {content}")
        elif role == "assistant":
            context_parts.append(f"Assistant: {content}")
    
    # If we have context, prepend it to the query
    if context_parts:
        context = "\n".join(context_parts)
        # Build enhanced query with context
        enhanced_query = f"""Previous conversation context:
{context}

Current question: {prompt}

Please answer the current question, using the conversation context to understand any references (like "this service", "it", "that", etc.). If the current question refers to something mentioned in the previous conversation, use that context to provide a relevant answer."""
        return enhanced_query
    
    return prompt

def display_sql_debug_info(sql_response, response_text_container=None):
    """
    Display SQL debug information (query and raw results) in an expandable section.
    
    Args:
        sql_response: The SQL response object from the query engine
        response_text_container: Optional container to display in (for better placement)
    """
    if not sql_response:
        return
    
    config = ConfigManager()
    debug_config = config.get("debug")
    show_sql_debug = debug_config.get("show_sql_debug", False) if debug_config else False
    
    if not show_sql_debug:
        return
    
    # Get metadata from SQL response
    metadata = {}
    if hasattr(sql_response, 'metadata'):
        metadata = sql_response.metadata
    elif isinstance(sql_response, dict):
        metadata = sql_response
    
    sql_query = metadata.get("sql_query")
    
    if sql_query:
        # Get raw results from metadata
        raw_results = metadata.get("result")
        col_keys = metadata.get("col_keys", [])
        
        container = response_text_container if response_text_container else st
        with container.expander("🔍 SQL Debug Information", expanded=False):
            st.subheader("Generated SQL Query")
            st.code(sql_query, language="sql")
            
            if raw_results is not None:
                st.subheader("Raw Database Results")
                if isinstance(raw_results, list) and len(raw_results) > 0:
                    # Format as table if we have column keys
                    if col_keys and len(col_keys) > 0:
                        import pandas as pd
                        try:
                            # Convert to DataFrame for better display
                            df = pd.DataFrame(raw_results, columns=col_keys)
                            # Limit rows shown (first 50 rows)
                            display_df = df.head(50)
                            st.dataframe(display_df, width='stretch')
                            if len(df) > 50:
                                st.caption(f"Showing first 50 of {len(df)} rows")
                        except Exception:
                            # Fallback to text display
                            st.text(str(raw_results[:500]))  # First 500 chars
                            if len(str(raw_results)) > 500:
                                st.caption("(Truncated - showing first 500 characters)")
                    else:
                        # No column keys, show as text
                        results_str = str(raw_results)
                        st.text(results_str[:1000])  # First 1000 chars
                        if len(results_str) > 1000:
                            st.caption("(Truncated - showing first 1000 characters)")
                else:
                    st.info("No results returned from database query.")
            else:
                st.info("Raw results not available in response metadata.")
            
            # Show metadata keys for debugging
            if st.checkbox("Show all metadata keys", key=f"show_metadata_keys_{hash(str(sql_query))}"):
                st.json(metadata)

def main():
    st.title("🤖 ProBot: Enterprise Assistant")
    
    config = ConfigManager()
    llm_conf = config.get("llm")
    
    # Session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    session_mgr = SessionManager(use_redis=False)
    cache_mgr = CacheManager()

    # Sidebar showing current config
    with st.sidebar:
        st.header("Status")
        from modules.model_registry import get_model_info
        model_info = get_model_info(llm_conf['model_name'])
        model_display = model_info.get('name', llm_conf['model_name'])
        speed_info = model_info.get('speed', '')
        if speed_info:
            st.info(f"**Model:** {model_display}\n⚡ **Speed:** {speed_info}")
        else:
            st.info(f"**Model:** {model_display}")
        st.caption("Go to 'Settings' page to change model.")
        if st.button("New Chat"):
            # Create a new session ID for a fresh start
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    # Initialize Chat History
    if "messages" not in st.session_state:
        previous = session_mgr.get_session(st.session_state.session_id)
        if previous:
             st.session_state.messages = previous
        else:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your enterprise data today?"}]

    # Load resources - cached, so first load is slow but subsequent loads are fast
    # Only show spinner if we're actually loading (not from cache)
    if 'system_initialized' not in st.session_state:
        with st.spinner("🔄 Initializing system (first time may take a moment)..."):
            index, llm = initialize_system(
                model_name=llm_conf["model_name"], 
                temperature=llm_conf["temperature"]
            )
            st.session_state.system_initialized = True
    else:
        # Subsequent loads should be fast due to caching
        index, llm = initialize_system(
            model_name=llm_conf["model_name"], 
            temperature=llm_conf["temperature"]
        )
    
    if index is None:
        st.warning("⚠️ Knowledge Base not found! Please upload files in 'Settings'.")
        # Allow chat without index (pure LLM)? Optionally. For now restrict to RAG.
        # But user might want to chat with DB only. 
        # Let's allowing falling back to pure LLM if index missing is often better, 
        # BUT for RAG bot, let's keep it strict or just show warning.
    
    # Init Engine - Lazy loading for database components
    chat_engine = None
    if index:
        # --- Tool 1: Knowledge Base (RAG) ---
        # Create a chat engine that maintains conversation state
        # This is better than a simple query engine for context-aware conversations
        from llama_index.core.chat_engine import CondenseQuestionChatEngine
        from llama_index.core.memory import ChatMemoryBuffer
        
        # Create base query engine for the chat engine
        base_query_engine = index.as_query_engine(similarity_top_k=3)
        
        # Create memory buffer for conversation history (will be updated dynamically)
        history = [
            ChatMessage(role=m["role"], content=m["content"]) 
            for m in st.session_state.messages 
            if m["role"] in ["user", "assistant"]
        ]
        memory = ChatMemoryBuffer.from_defaults(chat_history=history)
        
        # Create chat engine with conversation memory
        # This will be used for context-dependent queries
        rag_chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=base_query_engine,
            memory=memory,
            llm=llm,
            verbose=False
        )
        
        # Also keep a simple query engine for non-conversational queries
        rag_query_engine = base_query_engine
        
        rag_tool = QueryEngineTool(
            query_engine=rag_query_engine,
            metadata=ToolMetadata(
                name="knowledge_base",
                description=(
                    "Use this tool to answer questions about company policies, uploaded documents, "
                    "videos, images, and general text data. Do not use this for counting database rows."
                ),
            ),
        )
        
        # Store chat engine separately for context-aware queries
        rag_chat_engine_for_context = rag_chat_engine
        
        tools = [rag_tool]
        
        # --- Tool 2: Database (SQL) - Cached and lazy loaded ---
        sql_engine = None
        try:
            # Use cached SQL query engine
            sql_engine = get_sql_query_engine(llm)
        except Exception:
            # Silently fail - database might not be configured
            sql_engine = None
        
        if sql_engine:
            sql_tool = QueryEngineTool(
                query_engine=sql_engine,
                metadata=ToolMetadata(
                    name="database_tool",
                    description=(
                        "Use this tool to answer quantitative questions about the database, "
                        "such as counting rows, listing users, summing values, or querying specific tables. "
                        "If the user asks 'how many' or 'list from database', use this."
                    ),
                ),
            )
            tools.append(sql_tool)
        
        # Enhanced intelligent router with LLM-based intent classification
        # Get cached database schema summary for better routing decisions (lazy load, only when SQL engine exists)
        schema_summary = None
        if sql_engine:
            try:
                # Only load schema summary if SQL engine is available
                # This is cached, so subsequent calls are fast
                schema_summary = get_cached_schema_summary()
            except Exception:
                # If schema loading fails, continue without it
                schema_summary = None
        
        # Initialize intent classifier with caching and schema information
        intent_classifier = IntentClassifier(
            llm=llm, 
            cache_manager=cache_mgr,
            schema_summary=schema_summary
        )
        
        chat_engine = {
            "type": "enhanced_router",
            "llm": llm,
            "rag_engine": rag_query_engine,  # Simple query engine for direct queries
            "rag_chat_engine": rag_chat_engine_for_context,  # Chat engine for context-aware queries
            "sql_engine": sql_engine if sql_engine else None,
            "system_prompt": llm_conf["system_prompt"],
            "intent_classifier": intent_classifier
        }

    # Display Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask about your data..."):
        # Get conversation history BEFORE adding current message
        conversation_history = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
        
        # Check if query is context-dependent
        is_context_dependent = is_context_dependent_query(prompt)
        
        # Build context-aware cache key for context-dependent queries
        cache_key = prompt
        if is_context_dependent and conversation_history:
            context_hash = get_conversation_context_hash(conversation_history)
            if context_hash:
                cache_key = f"{prompt}::ctx:{context_hash}"
        
        # Check Cache with context-aware key
        cached_resp = cache_mgr.get_cached_response(cache_key)
        # Don't use cached error responses - clear them
        if cached_resp:
            if "Error" in cached_resp or "error" in cached_resp.lower() or "run_agent" in cached_resp:
                # Clear the bad cached response
                cache_mgr.set_cached_response(cache_key, None)  # This will overwrite with None
                cached_resp_to_use = None
            else:
                cached_resp_to_use = cached_resp
        else:
            cached_resp_to_use = None
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        session_mgr.append_message(st.session_state.session_id, "user", prompt)
        
        # Update conversation history to include current message
        conversation_history = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages

        with st.chat_message("assistant"):
            if cached_resp_to_use:
                response_text = cached_resp_to_use + " *(cached)*"
                st.markdown(response_text)
            else:
                with st.spinner("Thinking..."):
                    if chat_engine:
                        response = None
                        response_text = ""
                        sql_response_for_debug = None  # Initialize for SQL debug display
                        try:
                            # Enhanced intelligent routing with LLM-based intent classification
                            if isinstance(chat_engine, dict) and chat_engine.get("type") in ["enhanced_router", "simple_router"]:
                                # STEP 1: Resolve context-dependent references BEFORE routing
                                resolved_query = prompt
                                if is_context_dependent:
                                    # Debug: Show that we're attempting context resolution
                                    st.info(f"🔍 Detected context-dependent query. Resolving references...")
                                    resolved_query = resolve_context_references(
                                        prompt, 
                                        conversation_history, 
                                        chat_engine["llm"]
                                    )
                                    if resolved_query != prompt:
                                        st.success(f"✅ Resolved: \"{prompt}\" → \"{resolved_query}\"")
                                    else:
                                        st.warning(f"⚠️ Context resolution returned original query. May need more conversation history.")
                                
                                # STEP 2: Use resolved query for intent classification
                                intent_classification = None
                                if chat_engine.get("type") == "enhanced_router" and chat_engine.get("intent_classifier"):
                                    try:
                                        # Use resolved query for classification
                                        intent_classification = chat_engine["intent_classifier"].classify(
                                            query=resolved_query,  # Use resolved query instead of original
                                            conversation_history=conversation_history
                                        )
                                    except Exception as e:
                                        # If LLM classification fails, fall back to keyword-based
                                        st.warning(f"Intent classification failed, using keyword fallback: {str(e)}")
                                        intent_classification = None
                                
                                # Fallback to keyword-based if LLM classification failed or not available
                                if not intent_classification:
                                    intent_classification = classify_with_keywords(resolved_query)  # Use resolved query
                                
                                intent = intent_classification.get("intent", "knowledge_base")
                                confidence = intent_classification.get("confidence", 0.5)
                                reasoning = intent_classification.get("reasoning", "")
                                
                                # Check admin routing preference (override intent if set)
                                routing_config = config.get("routing")
                                routing_mode = routing_config.get("mode", "auto") if routing_config else "auto"
                                
                                if routing_mode != "auto":
                                    # Admin has set a specific routing mode - override intent
                                    original_intent = intent
                                    intent = routing_mode
                                    reasoning = f"Admin routing mode: {routing_mode} (overridden from {original_intent})"
                                    confidence = 1.0  # Full confidence when admin sets routing
                                    st.info(f"⚙️ **Admin Override:** Routing set to '{routing_mode}' mode")
                                
                                # Show routing info for debugging
                                st.info(f"🔀 Routing: {intent} (confidence: {confidence:.2f}) - {reasoning}")
                                
                                # Use resolved query for all subsequent operations
                                query_to_use = resolved_query
                                
                                # Route based on intent
                                if intent == "database":
                                    if not chat_engine["sql_engine"]:
                                        response_text = "⚠️ **Error:** Database connection not available. Please configure database in Settings."
                                        response = None
                                    else:
                                        # Build context-aware cache key for database queries
                                        db_cache_key = query_to_use
                                        if is_context_dependent and conversation_history:
                                            context_hash = get_conversation_context_hash(conversation_history)
                                            if context_hash:
                                                db_cache_key = f"{query_to_use}::ctx:{context_hash}"
                                        
                                        # Check cache first for SQL query results (use resolved query)
                                        cached_result = cache_mgr.get_cached_query_result(db_cache_key, "sql")
                                        if cached_result:
                                            # Check if cache entry is still valid (TTL)
                                            if time.time() - cached_result.get("timestamp", 0) < cached_result.get("ttl", 3600):
                                                response_text = cached_result["result"] + " *(cached)*"
                                                response = None  # Cached results don't have source_nodes
                                            else:
                                                cached_result = None  # Expired
                                        
                                        if not cached_result:
                                            # Route to SQL engine with resolved query
                                            try:
                                                sql_response = chat_engine["sql_engine"].query(query_to_use)  # Use resolved query
                                                response_text = str(sql_response)
                                                response = sql_response
                                                # Store SQL response for debug display
                                                sql_response_for_debug = sql_response
                                                # Cache the result with context-aware key
                                                cache_mgr.set_cached_query_result(db_cache_key, "sql", response_text, ttl=3600)
                                            except ValueError as e:
                                                # SQL validation error (unsafe query detected)
                                                error_msg = str(e)
                                                response_text = f"🚫 **SQL Safety Validation Failed:**\n\n{error_msg}\n\n*The system detected an unsafe SQL query. This system only supports SELECT queries (read-only operations). If you asked about 'removing', 'deleting', or 'updating', please rephrase your question to request information instead.*"
                                                response = None
                                                st.error("Unsafe SQL query detected and blocked")
                                            except Exception as e:
                                                # Show detailed error for database queries
                                                error_msg = str(e)
                                                response_text = f"⚠️ **Database Query Error:** {error_msg}\n\n*The query was correctly routed to the database, but the SQL execution failed. Please check your query or database connection.*"
                                                response = None
                                                st.error(f"SQL Query Failed: {error_msg}")
                                        
                                elif intent == "both" and chat_engine["sql_engine"]:
                                    # Try both tools in parallel for better performance
                                    try:
                                        # Use chat engine for context-dependent RAG queries
                                        if is_context_dependent and chat_engine.get("rag_chat_engine"):
                                            chat_history = [
                                                ChatMessage(role=m["role"], content=m["content"]) 
                                                for m in conversation_history 
                                                if m["role"] in ["user", "assistant"]
                                            ]
                                            chat_engine["rag_chat_engine"]._memory.set(chat_history)
                                            rag_query_func = lambda q: chat_engine["rag_chat_engine"].chat(q)
                                        else:
                                            rag_query_func = lambda q: chat_engine["rag_engine"].query(q)
                                        
                                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                                            # Submit both queries in parallel with resolved query
                                            sql_future = executor.submit(chat_engine["sql_engine"].query, query_to_use)
                                            rag_future = executor.submit(rag_query_func, query_to_use)
                                            
                                            # Wait for both to complete
                                            sql_response = sql_future.result(timeout=30)
                                            rag_result = rag_future.result(timeout=30)
                                            
                                            sql_text = str(sql_response)
                                            # Handle both chat response and query response
                                            if hasattr(rag_result, 'response'):
                                                rag_text = str(rag_result.response)
                                                rag_response = rag_result
                                            else:
                                                rag_text = str(rag_result)
                                                rag_response = rag_result
                                            
                                            # Combine results
                                            response_text = f"**Database Results:**\n{sql_text}\n\n**Knowledge Base Information:**\n{rag_text}"
                                            response = rag_response  # Use RAG response for source_nodes
                                            
                                            # Store SQL response for debug display
                                            sql_response_for_debug = sql_response
                                            
                                            # Cache both results with resolved query
                                            cache_mgr.set_cached_query_result(query_to_use, "sql", sql_text, ttl=3600)
                                            cache_mgr.set_cached_query_result(query_to_use, "rag", rag_text, ttl=3600)
                                    except concurrent.futures.TimeoutError:
                                        st.warning("Query timeout, trying sequentially...")
                                        # Fallback to sequential if parallel fails
                                        try:
                                            sql_response = chat_engine["sql_engine"].query(query_to_use)  # Use resolved query
                                            sql_text = str(sql_response)
                                            # Use appropriate RAG engine based on context
                                            if is_context_dependent and chat_engine.get("rag_chat_engine"):
                                                chat_history = [
                                                    ChatMessage(role=m["role"], content=m["content"]) 
                                                    for m in conversation_history[:-1]  # Exclude current message
                                                    if m["role"] in ["user", "assistant"]
                                                ]
                                                chat_engine["rag_chat_engine"]._memory.set(chat_history)
                                                chat_result = chat_engine["rag_chat_engine"].chat(query_to_use)  # Use resolved query
                                                rag_text = str(chat_result)
                                                rag_response = chat_result if hasattr(chat_result, 'source_nodes') else None
                                            else:
                                                rag_response = chat_engine["rag_engine"].query(query_to_use)  # Use resolved query
                                                rag_text = str(rag_response)
                                            response_text = f"**Database Results:**\n{sql_text}\n\n**Knowledge Base Information:**\n{rag_text}"
                                            response = rag_response
                                            # Store SQL response for debug display
                                            sql_response_for_debug = sql_response
                                        except Exception:
                                            # If both fail, try just RAG
                                            if is_context_dependent and chat_engine.get("rag_chat_engine"):
                                                chat_history = [
                                                    ChatMessage(role=m["role"], content=m["content"]) 
                                                    for m in conversation_history[:-1]  # Exclude current message
                                                    if m["role"] in ["user", "assistant"]
                                                ]
                                                chat_engine["rag_chat_engine"]._memory.set(chat_history)
                                                rag_response = chat_engine["rag_chat_engine"].chat(query_to_use)  # Use resolved query
                                            else:
                                                rag_response = chat_engine["rag_engine"].query(query_to_use)  # Use resolved query
                                            response_text = str(rag_response)
                                            response = rag_response
                                    except Exception as sql_error:
                                        # If SQL fails, just use RAG
                                        if is_context_dependent and chat_engine.get("rag_chat_engine"):
                                            chat_history = [
                                                ChatMessage(role=m["role"], content=m["content"]) 
                                                for m in conversation_history[:-1]  # Exclude current message
                                                if m["role"] in ["user", "assistant"]
                                            ]
                                            chat_engine["rag_chat_engine"]._memory.set(chat_history)
                                            chat_result = chat_engine["rag_chat_engine"].chat(query_to_use)  # Use resolved query
                                            response_text = str(chat_result)
                                            response = chat_result if hasattr(chat_result, 'source_nodes') else None
                                        else:
                                            rag_response = chat_engine["rag_engine"].query(query_to_use)  # Use resolved query
                                            response_text = str(rag_response)
                                            response = rag_response
                                        
                                else:
                                    # Default to RAG engine (knowledge base)
                                    # Build context-aware cache key using resolved query
                                    rag_cache_key = query_to_use
                                    if is_context_dependent and conversation_history:
                                        context_hash = get_conversation_context_hash(conversation_history)
                                        if context_hash:
                                            rag_cache_key = f"{query_to_use}::ctx:{context_hash}"
                                    
                                    # Check cache first for RAG query results
                                    cached_result = cache_mgr.get_cached_query_result(rag_cache_key, "rag")
                                    if cached_result:
                                        if time.time() - cached_result.get("timestamp", 0) < cached_result.get("ttl", 3600):
                                            response_text = cached_result["result"] + " *(cached)*"
                                            response = None
                                        else:
                                            cached_result = None
                                    
                                    if not cached_result:
                                        # Use chat engine for context-dependent queries, simple engine for direct queries
                                        if is_context_dependent and chat_engine.get("rag_chat_engine"):
                                            # Update chat engine memory with latest conversation history (excluding current)
                                            chat_history = [
                                                ChatMessage(role=m["role"], content=m["content"]) 
                                                for m in conversation_history[:-1]  # Exclude current message
                                                if m["role"] in ["user", "assistant"]
                                            ]
                                            chat_engine["rag_chat_engine"]._memory.set(chat_history)
                                            # Use chat engine with resolved query (chat engine will add current message to memory)
                                            chat_response = chat_engine["rag_chat_engine"].chat(query_to_use)  # Use resolved query
                                            response_text = str(chat_response)
                                            # Extract source nodes if available
                                            if hasattr(chat_response, 'source_nodes'):
                                                response = chat_response
                                            else:
                                                # Create a mock response object for compatibility
                                                from llama_index.core.schema import Response
                                                response = Response(response_text)
                                        else:
                                            # Use simple query engine for direct queries with resolved query
                                            rag_response = chat_engine["rag_engine"].query(query_to_use)  # Use resolved query
                                            response_text = str(rag_response)
                                            response = rag_response
                                        # Cache the result with context-aware key
                                        cache_mgr.set_cached_query_result(rag_cache_key, "rag", response_text, ttl=3600)
                                    
                                    # If RAG doesn't give good results and we have SQL, try SQL as fallback
                                    if len(response_text) < 50 and chat_engine["sql_engine"] and intent != "database":
                                        try:
                                            sql_response = chat_engine["sql_engine"].query(query_to_use)  # Use resolved query
                                            if sql_response and len(str(sql_response)) > len(response_text):
                                                response_text = f"**Database Result:**\n{str(sql_response)}"
                                                response = sql_response
                                                # Store SQL response for debug display
                                                sql_response_for_debug = sql_response
                                        except:
                                            pass  # Keep RAG response
                                    
                            else:
                                # Fallback: If chat_engine is not recognized
                                response_text = "⚠️ **Error:** Unknown chat engine type. Please refresh the page."
                                response = None
                        except (RuntimeError, Exception) as e:
                            error_msg = str(e).lower()
                            # Check if it's an event loop related error
                            if any(phrase in error_msg for phrase in ["no running event loop", "event loop", "cannot be called from a running"]):
                                # Fallback: Use RAG query engine directly if agent fails
                                st.warning("Agent unavailable, using direct query engine...")
                                if index:
                                    rag_engine = index.as_query_engine(similarity_top_k=5)
                                    # Resolve context if needed for fallback
                                    fallback_query = prompt
                                    if is_context_dependent:
                                        fallback_query = resolve_context_references(
                                            prompt, 
                                            conversation_history, 
                                            llm
                                        )
                                    # Use chat engine if context-dependent, otherwise simple query
                                    if is_context_dependent:
                                        from llama_index.core.chat_engine import CondenseQuestionChatEngine
                                        from llama_index.core.memory import ChatMemoryBuffer
                                        chat_history = [
                                            ChatMessage(role=m["role"], content=m["content"]) 
                                            for m in conversation_history 
                                            if m["role"] in ["user", "assistant"]
                                        ]
                                        memory = ChatMemoryBuffer.from_defaults(chat_history=chat_history)
                                        fallback_chat_engine = CondenseQuestionChatEngine.from_defaults(
                                            query_engine=rag_engine,
                                            memory=memory,
                                            llm=llm,
                                            verbose=False
                                        )
                                        response = fallback_chat_engine.chat(fallback_query)  # Use resolved query
                                    else:
                                        response = rag_engine.query(fallback_query)  # Use resolved query
                                    response_text = str(response)
                                else:
                                    response_text = f"⚠️ **Error:** {str(e)}"
                                    response = None
                            else:
                                response_text = f"⚠️ **Error talking to AI Service:** {str(e)}\n\n*Tip: Check if Ollama is running and healthy.*"
                                response = None
                        
                        # --- SQL Debug Display (if enabled) ---
                        # Check if we have a stored SQL response for debug (from "both" route or fallback)
                        if 'sql_response_for_debug' in locals() and sql_response_for_debug:
                            display_sql_debug_info(sql_response_for_debug)
                        # Also check if the main response is a SQL response
                        elif response and hasattr(response, 'metadata'):
                            metadata = response.metadata if hasattr(response, 'metadata') else {}
                            if metadata.get("sql_query"):
                                display_sql_debug_info(response)
                        
                        # --- Source Citations & Media Player ---
                        if response and hasattr(response, 'source_nodes') and response.source_nodes:
                                with st.expander("References & Media"):
                                    for node in response.source_nodes:
                                        meta = node.metadata
                                        score = "{:.2f}".format(node.score) if node.score else "N/A"
                                        st.markdown(f"**Source:** {meta.get('source', 'Unknown')} (Score: {score})")
                                        
                                        # Media Deep Linking
                                        if meta.get("type") == "media" and "start_time" in meta:
                                            start_s = int(meta["start_time"])
                                            source_file = meta["source"]
                                            media_path = os.path.join("./data", source_file)
                                            if os.path.exists(media_path):
                                                    st.video(media_path, start_time=start_s)
                                                    st.caption(f"Playing from {start_s}s")
                                        
                                        st.text(node.get_text()[:200] + "...")
                                        st.divider()

                    else:
                        response_text = "Knowledge base is empty. Please go to Settings to upload data."
                        
                    st.markdown(response_text)
                    if chat_engine: # Only cache if real answer
                        # Use context-aware cache key
                        cache_mgr.set_cached_response(cache_key, response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        session_mgr.append_message(st.session_state.session_id, "assistant", response_text)

if __name__ == "__main__":
    main()
