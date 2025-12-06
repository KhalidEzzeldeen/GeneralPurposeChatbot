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
    
    current_llm = config.get("llm")
    
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.text_input("Ollama Model Name", value=current_llm.get("model_name", "qwen2.5:7b-instruct"))
        temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=current_llm.get("temperature", 0.2))
        
    system_prompt = st.text_area("System Prompt", value=current_llm.get("system_prompt", ""), height=150)
    
    if st.button("Save LLM Settings"):
        config.set("llm", "model_name", model_name)
        config.set("llm", "temperature", temperature)
        config.set("llm", "system_prompt", system_prompt)
        st.success("Settings saved successfully!")
        st.cache_resource.clear() # Clear cache to force reload of LLM engine

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
                st.cache_resource.clear()
