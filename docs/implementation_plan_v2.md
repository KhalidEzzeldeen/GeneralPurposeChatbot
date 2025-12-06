# Enhanced Implementation Plan: Configurable & Incremental Chatbot

## Goal Description
Transform the current prototype into a fully configurable "Product". 
- **No Code Configuration**: Users can connect Databases and upload files via the UI.
- **Incremental Knowledge**: Users can add files one by one without rebuilding the whole index.
- **Persistence**: All settings (DB credentials, Model choices) are saved securely.

## Proposed Architecture Logic

### 1. The "Settings" Page (`pages/settings.py`)
We will move to a Streamlit **Multi-Page App** structure.
- **Tab 1: LLM Settings**: Select Model (Qwen/Llama), Temperature, System Prompt.
- **Tab 2: Knowledge Base**: 
    - **File Uploader**: Drag & Drop PDF, Excel, Media.
    - **Index Status**: precise count of documents, option to "Clear Index".
- **Tab 3: Database Connections**: 
    - Input fields for `Host`, `Port`, `User`, `Password`, `Database`.
    - "Test Connection" button.
    - "Scan Schema" button (saves schema to context).

### 2. Incremental Ingestion Engine
Refactor `ingest.py` into a `KnowledgeBase` class.
- **Logic**: 
    - When a file is uploaded -> Calculate Hash.
    - Check `ingested_files.json` registry.
    - If new -> Process -> Add to ChromaDB -> Update Registry.
    - This ensures we never re-embed existing files (saving time/compute).

### 3. Unified Configuration Manager
A `ConfigManager` class that reads/writes to `config.json` (or `.env`).
- storing DB credentials securely.
- storing user preferences.

## Suggestions for Improvement (Included in Plan)
1.  **Chat Profiles**: Allow user to save different "Personas" (e.g., "SQL Analyst" vs "HR Helper") with different system prompts.
2.  **Source Visibility**: When the bot answers, show a clickable "Expand Source" button to see exactly which file/row was used.
3.  **Hybrid Search Toggle**: Allow user to toggle "Keyword Search" vs "Semantic Search" in the UI for debugging.

## Detailed Task List

### Phase 1: Structure & Configuration
1.  **Refactor Directory**: Move `app.py` to `main.py` and set up `pages/` folder.
2.  **Config Module**: Create `modules/config.py` to handle saving/loading JSON settings.
3.  **Secrets Management**: Ensure DB passwords are handled safely (not logged).

### Phase 2: UI Implementation
4.  **Create Settings Page**: Implement the Tabs (LLM, Knowledge, Database).
5.  **Database Form**: create input form for SQL credentials.
6.  **Schema Scanner**: Implement function to connect to SQL DB and save text description of tables.

### Phase 3: Interactive Ingestion
7.  **UI Uploader**: Replace local `data/` folder reliance with `st.file_uploader`.
8.  **Incremental Logic**: Update `ingest.py` to accept single file streams and check for duplicates.
9.  **Progress Bars**: Add visual feedback during Video/Audio transcription processing in UI.

### Phase 4: Chat Interface Upgrades
10. **Dynamic System Prompt**: Load system prompt from the config set in Settings page.
11. **Source Citation UI**: Improve the display of retrieved nodes (show filename + page number).

### Phase 5: Verification
12. **End-to-End Test**: 
    - Start fresh.
    - Go to Settings -> Configure DB.
    - Go to Settings -> Upload PDF.
    - Go to Chat -> Ask question combining DB and PDF info.


    1. What is the Database Type?

The application is built using SQLAlchemy, which makes it "Database Agnostic". This means it can theoretically connect to almost any SQL database (PostgreSQL, MySQL, SQLite, Oracle, MS SQL).

Current Default: The code currently defaults to constructing a PostgreSQL connection string (postgresql+psycopg2://...).
How to change it: You can connect to other databases (like MySQL or SQL Server) simply by installing the appropriate Python driver (e.g., mysql-connector-python or pyodbc) and updating the connection string logic.

2. Where are Audio and Video files stored?

Physical Location: When you upload a file via the "Settings" page, the system saves a copy of the raw file into your local c:\Work\Chatbot\data folder.
Knowledge Location: The content (transcribed text and descriptions) is stored in ChromaDB (./chroma_db). The raw audio/video file itself is not stored in the database, only the intelligence extracted from it.

3. What tools are used for Audio and Video?

We use a "Multimodal Pipeline" to turn these files into text the Chatbot can understand:

Audio (Speech-to-Text):
Tool: faster-whisper (an optimized version of OpenAI's Whisper model).
Function: It listens to the audio track and writes down every spoken word into a transcript.
Video (Visual Understanding):
Tool: Qwen2.5-VL (Vision-Language Model).
Function: It looks at images/frames extracted from the video or PDF and writes a detailed textual description of what it sees (charts, people, actions).
Orchestration:
Tool: LlamaIndex.
Function: It manages the process of reading the file, chunking it, sending it to the AI models above, and saving the results to the database.


Phase 5: Polish & Verify

 Verification: Upload new PDF via UI and query it
 Verification: Configure SQL DB and ask analytics question
 Phase 6: Multimedia Deep Linking

 Modify 
multimodal.py
 to return segments {text, timestamp} from Whisper
 Modify 
knowledge_base.py
 to index segments as separate Docs with metadata
 Modify 
Home.py
 UI to check metadata and render st.video(..., start_time)
