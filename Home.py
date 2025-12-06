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

    # Load resources
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
    
    # Init Engine
    chat_engine = None
    if index:
        # --- Tool 1: Knowledge Base (RAG) ---
        # Create a query engine from the index with optimized settings
        # Reduced similarity_top_k from 5 to 3 for better performance
        rag_query_engine = index.as_query_engine(similarity_top_k=3)
        
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
        
        tools = [rag_tool]
        
        # --- Tool 2: Database (SQL) ---
        from modules.database import DatabaseManager
        db_mgr = DatabaseManager()
        sql_engine = db_mgr.get_sql_query_engine(llm)
        
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
        
        # History Injection
        history = [
            ChatMessage(role=m["role"], content=m["content"]) 
            for m in st.session_state.messages 
            if m["role"] in ["user", "assistant"]
        ]
        memory = ChatMemoryBuffer.from_defaults(chat_history=history)
        
        # Enhanced intelligent router with LLM-based intent classification
        # Get cached database schema summary for better routing decisions
        schema_summary = None
        if sql_engine:
            schema_summary = get_cached_schema_summary()
        
        # Initialize intent classifier with caching and schema information
        intent_classifier = IntentClassifier(
            llm=llm, 
            cache_manager=cache_mgr,
            schema_summary=schema_summary
        )
        
        chat_engine = {
            "type": "enhanced_router",
            "llm": llm,
            "rag_engine": rag_query_engine,
            "sql_engine": sql_engine if sql_engine else None,
            "memory": memory,
            "system_prompt": llm_conf["system_prompt"],
            "intent_classifier": intent_classifier
        }

    # Display Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask about your data..."):
        # Check Cache (skip if it's an error to avoid showing cached errors)
        cached_resp = cache_mgr.get_cached_response(prompt)
        # Don't use cached error responses - clear them
        if cached_resp:
            if "Error" in cached_resp or "error" in cached_resp.lower() or "run_agent" in cached_resp:
                # Clear the bad cached response
                cache_mgr.set_cached_response(prompt, None)  # This will overwrite with None
                cached_resp_to_use = None
            else:
                cached_resp_to_use = cached_resp
        else:
            cached_resp_to_use = None
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        session_mgr.append_message(st.session_state.session_id, "user", prompt)

        with st.chat_message("assistant"):
            if cached_resp_to_use:
                response_text = cached_resp_to_use + " *(cached)*"
                st.markdown(response_text)
            else:
                with st.spinner("Thinking..."):
                    if chat_engine:
                        response = None
                        response_text = ""
                        try:
                            # Enhanced intelligent routing with LLM-based intent classification
                            if isinstance(chat_engine, dict) and chat_engine.get("type") in ["enhanced_router", "simple_router"]:
                                # Get conversation history for context
                                conversation_history = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
                                
                                # Use LLM-based classification with schema understanding (primary method)
                                intent_classification = None
                                if chat_engine.get("type") == "enhanced_router" and chat_engine.get("intent_classifier"):
                                    try:
                                        intent_classification = chat_engine["intent_classifier"].classify(
                                            query=prompt,
                                            conversation_history=conversation_history
                                        )
                                    except Exception as e:
                                        # If LLM classification fails, fall back to keyword-based
                                        st.warning(f"Intent classification failed, using keyword fallback: {str(e)}")
                                        intent_classification = None
                                
                                # Fallback to keyword-based if LLM classification failed or not available
                                if not intent_classification:
                                    intent_classification = classify_with_keywords(prompt)
                                
                                intent = intent_classification.get("intent", "knowledge_base")
                                confidence = intent_classification.get("confidence", 0.5)
                                reasoning = intent_classification.get("reasoning", "")
                                
                                # Show routing info for debugging
                                st.info(f"🔀 Routing: {intent} (confidence: {confidence:.2f}) - {reasoning}")
                                
                                # Route based on intent
                                if intent == "database":
                                    if not chat_engine["sql_engine"]:
                                        response_text = "⚠️ **Error:** Database connection not available. Please configure database in Settings."
                                        response = None
                                    else:
                                        # Check cache first for SQL query results
                                        cached_result = cache_mgr.get_cached_query_result(prompt, "sql")
                                        if cached_result:
                                            # Check if cache entry is still valid (TTL)
                                            if time.time() - cached_result.get("timestamp", 0) < cached_result.get("ttl", 3600):
                                                response_text = cached_result["result"] + " *(cached)*"
                                                response = None  # Cached results don't have source_nodes
                                            else:
                                                cached_result = None  # Expired
                                        
                                        if not cached_result:
                                            # Route to SQL engine
                                            try:
                                                sql_response = chat_engine["sql_engine"].query(prompt)
                                                response_text = str(sql_response)
                                                response = sql_response
                                                # Cache the result
                                                cache_mgr.set_cached_query_result(prompt, "sql", response_text, ttl=3600)
                                            except Exception as e:
                                                # Show detailed error for database queries
                                                error_msg = str(e)
                                                response_text = f"⚠️ **Database Query Error:** {error_msg}\n\n*The query was correctly routed to the database, but the SQL execution failed. Please check your query or database connection.*"
                                                response = None
                                                st.error(f"SQL Query Failed: {error_msg}")
                                        
                                elif intent == "both" and chat_engine["sql_engine"]:
                                    # Try both tools in parallel for better performance
                                    try:
                                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                                            # Submit both queries in parallel
                                            sql_future = executor.submit(chat_engine["sql_engine"].query, prompt)
                                            rag_future = executor.submit(chat_engine["rag_engine"].query, prompt)
                                            
                                            # Wait for both to complete
                                            sql_response = sql_future.result(timeout=30)
                                            rag_response = rag_future.result(timeout=30)
                                            
                                            sql_text = str(sql_response)
                                            rag_text = str(rag_response)
                                            
                                            # Combine results
                                            response_text = f"**Database Results:**\n{sql_text}\n\n**Knowledge Base Information:**\n{rag_text}"
                                            response = rag_response  # Use RAG response for source_nodes
                                            
                                            # Cache both results
                                            cache_mgr.set_cached_query_result(prompt, "sql", sql_text, ttl=3600)
                                            cache_mgr.set_cached_query_result(prompt, "rag", rag_text, ttl=3600)
                                    except concurrent.futures.TimeoutError:
                                        st.warning("Query timeout, trying sequentially...")
                                        # Fallback to sequential if parallel fails
                                        try:
                                            sql_response = chat_engine["sql_engine"].query(prompt)
                                            sql_text = str(sql_response)
                                            rag_response = chat_engine["rag_engine"].query(prompt)
                                            rag_text = str(rag_response)
                                            response_text = f"**Database Results:**\n{sql_text}\n\n**Knowledge Base Information:**\n{rag_text}"
                                            response = rag_response
                                        except Exception:
                                            # If both fail, try just RAG
                                            rag_response = chat_engine["rag_engine"].query(prompt)
                                            response_text = str(rag_response)
                                            response = rag_response
                                    except Exception as sql_error:
                                        # If SQL fails, just use RAG
                                        rag_response = chat_engine["rag_engine"].query(prompt)
                                        response_text = str(rag_response)
                                        response = rag_response
                                        
                                else:
                                    # Default to RAG engine (knowledge base)
                                    # Check cache first for RAG query results
                                    cached_result = cache_mgr.get_cached_query_result(prompt, "rag")
                                    if cached_result:
                                        if time.time() - cached_result.get("timestamp", 0) < cached_result.get("ttl", 3600):
                                            response_text = cached_result["result"] + " *(cached)*"
                                            response = None
                                        else:
                                            cached_result = None
                                    
                                    if not cached_result:
                                        # Use streaming for RAG responses (better UX)
                                        rag_response = chat_engine["rag_engine"].query(prompt)
                                        response_text = str(rag_response)
                                        response = rag_response
                                        # Cache the result
                                        cache_mgr.set_cached_query_result(prompt, "rag", response_text, ttl=3600)
                                    
                                    # If RAG doesn't give good results and we have SQL, try SQL as fallback
                                    if len(response_text) < 50 and chat_engine["sql_engine"] and intent != "database":
                                        try:
                                            sql_response = chat_engine["sql_engine"].query(prompt)
                                            if sql_response and len(str(sql_response)) > len(response_text):
                                                response_text = f"**Database Result:**\n{str(sql_response)}"
                                                response = sql_response
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
                                    response = rag_engine.query(prompt)
                                    response_text = str(response)
                                else:
                                    response_text = f"⚠️ **Error:** {str(e)}"
                                    response = None
                            else:
                                response_text = f"⚠️ **Error talking to AI Service:** {str(e)}\n\n*Tip: Check if Ollama is running and healthy.*"
                                response = None
                        
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
                        cache_mgr.set_cached_response(prompt, response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        session_mgr.append_message(st.session_state.session_id, "assistant", response_text)

if __name__ == "__main__":
    main()
