import streamlit as st
import os
from modules.config import ConfigManager
from modules.knowledge_base import KnowledgeBase

st.set_page_config(page_title="Settings - ProBot", layout="wide")

st.title("⚙️ System Configuration")

config = ConfigManager()
kb = KnowledgeBase()

tab1, tab2, tab3 = st.tabs(["🤖 LLM Settings", "📚 Knowledge Base", "🗄️ Database Connection"])

# --- TAB 1: LLM Settings ---
with tab1:
    st.header("Model Configuration")
    
    from modules.model_registry import MODEL_REGISTRY, get_recommended_models, get_model_info
    
    current_llm = config.get("llm")
    current_model = current_llm.get("model_name", "qwen2.5:7b-instruct")
    
    # Model Selection
    st.subheader("Select LLM Model")
    
    # Get recommended models
    recommended_models = get_recommended_models()
    model_options = list(recommended_models.keys())
    
    # Add current model if not in recommended list
    if current_model not in model_options:
        model_options.insert(0, current_model)
    
    # Model selector with descriptions
    selected_model = st.selectbox(
        "Choose Model",
        options=model_options,
        index=0 if current_model in model_options else 0,
        format_func=lambda x: f"{MODEL_REGISTRY.get(x, {}).get('name', x)} - {MODEL_REGISTRY.get(x, {}).get('speed', 'Unknown')} speed" if x in MODEL_REGISTRY else x,
        help="Select a pre-configured model or use custom model below"
    )
    
    # Show model information
    if selected_model in MODEL_REGISTRY:
        model_info = MODEL_REGISTRY[selected_model]
        with st.expander(f"ℹ️ {model_info['name']} Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Size", model_info['size'])
            with col2:
                st.metric("Speed", model_info['speed'])
            with col3:
                st.metric("Quality", model_info['quality'])
            st.caption(model_info['description'])
    
    # Custom model option
    st.markdown("---")
    st.subheader("Custom Model (Advanced)")
    use_custom = st.checkbox("Use custom model name", value=current_model not in MODEL_REGISTRY)
    
    if use_custom:
        model_name = st.text_input(
            "Custom Ollama Model Name", 
            value=current_model if current_model not in MODEL_REGISTRY else "",
            help="Enter any Ollama model name (e.g., 'custom-model:latest')"
        )
        if not model_name:
            st.warning("Please enter a model name")
            model_name = selected_model
    else:
        model_name = selected_model
    
    # Model Performance Tips
    st.info("💡 **Tip:** For faster responses, try `qwen2.5:3b-instruct` or `llama3.2:3b`. For best quality, use `qwen2.5:7b-instruct`.")
    
    # Temperature and System Prompt
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=current_llm.get("temperature", 0.2))
    with col2:
        st.caption("Lower = more focused, Higher = more creative")
        
    system_prompt = st.text_area("System Prompt", value=current_llm.get("system_prompt", ""), height=150)
    
    # Check model availability
    from modules.llm_engine import check_model_available
    if model_name:
        is_available, error_msg = check_model_available(model_name)
        if is_available:
            st.success(f"✅ Model '{model_name}' is available in Ollama")
        else:
            st.warning(f"⚠️ {error_msg}")
            st.info("💡 Run `ollama pull <model_name>` to install the model")
    
    # Save button
    if st.button("💾 Save LLM Settings", type="primary"):
        if model_name:
            config.set("llm", "model_name", model_name)
            config.set("llm", "temperature", temperature)
            config.set("llm", "system_prompt", system_prompt)
            st.success("✅ Settings saved successfully!")
            st.cache_resource.clear()  # Clear cache to force reload of LLM engine
            st.cache_data.clear()  # Clear data cache too
            st.info("🔄 Please refresh the page or navigate away and back to apply changes.")
        else:
            st.error("Please select or enter a model name")
    
    # Quick Model Switcher
    st.markdown("---")
    st.subheader("Quick Switch")
    st.caption("Quickly switch between recommended models")
    
    fast_models = {
        "🚀 Fastest": "gemma2:2b",
        "⚡ Very Fast": "qwen2.5:3b-instruct",
        "🎯 Balanced": "qwen2.5:7b-instruct",
        "🌟 High Quality": "llama3.1:8b-instruct"
    }
    
    cols = st.columns(len(fast_models))
    for idx, (label, model) in enumerate(fast_models.items()):
        with cols[idx]:
            if st.button(label, key=f"quick_{model}"):
                config.set("llm", "model_name", model)
                config.set("llm", "temperature", temperature)
                config.set("llm", "system_prompt", system_prompt)
                st.success(f"✅ Switched to {MODEL_REGISTRY.get(model, {}).get('name', model)}")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

# --- TAB 2: Knowledge Base ---
with tab2:
    st.header("Manage Knowledge")
    
    st.subheader("Add New Content")
    uploaded_files = st.file_uploader("Upload Documents (PDF, Excel, Images, Audio)", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button(f"Ingest {len(uploaded_files)} Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Save temp and ingest
            data_dir = config.get("data_path")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                file_path = os.path.join(data_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                result = kb.ingest_file(file_path)
                st.toast(f"{uploaded_file.name}: {result}")
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Ingestion Complete!")
            st.cache_resource.clear()

    st.markdown("---")
    st.subheader("Current Index")
    files = kb.get_ingested_files()
    if files:
        st.dataframe(files)
        if st.button("Clear Knowledge Base (⚠️ Irreversible)"):
            kb.clear_index()
            st.warning("Index cleared.")
            st.rerun()
    else:
        st.info("No files in knowledge base.")

# --- TAB 3: Database ---
with tab3:
    st.header("SQL Database Connection")
    current_db = config.get("database")
    
    col1, col2 = st.columns(2)
    with col1:
        host = st.text_input("Host", value=current_db.get("host"))
        port = st.number_input("Port", value=current_db.get("port"), step=1)
        user = st.text_input("Username", value=current_db.get("user"))
    with col2:
        password = st.text_input("Password", value=current_db.get("password"), type="password")
        dbname = st.text_input("Database Name", value=current_db.get("dbname"))
        
    if st.button("Save Database Config"):
        db_conf = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "dbname": dbname
        }
        # In ConfigManager.set, we are setting key in section.
        # But here 'database' is the section.
        # ConfigManager set(section, key, value). 
        # I need to set individual keys or just update the whole dict if I change ConfigManager.
        # Given current ConfigManager implementation:
        for k, v in db_conf.items():
            config.set("database", k, v)
        st.success("Database configuration saved.")

    from modules.database import DatabaseManager
    db_manager = DatabaseManager()

    if st.button("Test Connection"):
        success, msg = db_manager.test_connection()
        if success:
            st.success(f"{msg}")
        else:
            st.error(f"{msg}")

    if st.button("Scan & Ingest Schema"):
        with st.spinner("Scanning database schema..."):
            result = db_manager.scan_schema()
            if "failed" in result.lower():
                st.error(result)
            else:
                st.success(result)
                # Clear caches to force refresh
                st.cache_resource.clear()
                st.cache_data.clear()  # Clear schema summary cache
