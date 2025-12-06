# Offline Multi-Modal RAG Chatbot Implementation Plan

## Goal Description
Build a professional-grade chatbot capable of running entirely offline on Windows. The bot must extract intelligence from diverse data sources:
- **Documents**: PDF, Excel, Text
- **Media**: Images, Audio, Video
- **Data**: SQL Databases

It will use RAG (Retrieval Augmented Generation) to answer customer queries based on this data.

## Technology Selection & Rational

### 1. The LLM (Offline & Open Source)
To ensure privacy, multilingual support (Arabic/English), and strong reasoning capabilities, we will use the **Qwen2.5** family served via **Ollama**.

- **Primary Text/Reasoning Model**: **Qwen2.5-7B-Instruct** (or 14B if hardware permits)
    - *Why?* Superior instruction following, tool usage, and multilingual capabilities compared to Llama 3.1.
- **Vision Model (Images/PDFs)**: **Qwen2.5-VL-7B-Instruct**
    - *Why?* Specialized for document understanding, charts, and complex layouts (OCR + Description).
- **Audio/Video Transcription**: **OpenAI Whisper (Medium)**
    - *Why?* Industry standard for speech-to-text.
- **Embedding Model**: **BAAI/bge-m3**
    - *Why?* Multilingual, long context (8192 tokens), and optimized for RAG.

### 2. The Tech Stack
- **Language**: Python 3.10+
- **Orchestration**: **LlamaIndex**
    - *Why?* Robust implementations for RAG and Agents.
- **Vector Database**: **ChromaDB** (or Qdrant/Milvus as potential upgrades)
- **UI Framework**: **Streamlit** (for Phase 1) -> **FastAPI + React/Streamlit** (for Production Phase)
- **Session Store**: **redis** or **sqlite** for conversation history.
## Proposed Architecture

### 1. Ingestion Layer ([ingest.py](file:///c:/Work/Chatbot/ingest.py))
- **PDF/Docs**: Extract text via `pypdf`. For complex pages (tables/charts), render as image and pass to **Qwen2.5-VL**. Chunk with **bge-m3** embeddings.
- **Excel/CSV**:
  - **Structured**: Load into SQLite/SQLAlchemy for "Text-to-SQL" queries.
  - **Unstructured**: Convert rows to semantic text strings for standard RAG.
- **Images**: Pass through **Qwen2.5-VL**. Prompt: *"Extract all text, fields, and numbers from this image and describe it."*
- **Audio/Video**: Extract audio -> **Whisper** -> Text Chunks.

### 2. Orchestration & Storage Layer
- **Orchestrator**: A "Router" component using Qwen2.5 to decide intent: `["knowledge_rag", "db_analytics", "image_qa"]`.
- **Vector Store**: ChromaDB collecting embeddings from all unstructured data.
- **SQL Store**: SQLite/PostgreSQL for structured Excel/DB data.
- **Session Memory**: Store conversation turns (User/AI) in a persistent store (SQLite/Redis) to handle follow-up questions.
- **Caching**: Cache frequent queries and extraction results to improve performance.

### 3. Retrieval & Generation Layer ([app.py](file:///c:/Work/Chatbot/app.py))
- **Flow**: User Input -> Session Loader -> Orchestrator -> (Tool: RAG | Tool: SQL) -> Context -> Qwen2.5 -> Answer.

## Proposed Changes / File Structure

### Project Root: `c:/Work/Chatbot`

#### [NEW] `requirements.txt`
Dependencies: `llama-index`, `streamlit`, `chromadb`, `ollama`, `faster-whisper`, `pandas`, `openpyxl`, `sqlalchemy`, `pillow`.

#### [NEW] `ingest.py`
Script to scan the `./data` folder and process all files into the vector database.

#### [NEW] `app.py`
The Streamlit application containing the chat interface and session management.

#### [NEW] `modules/`
- `llm_engine.py`: Wrapper for Ollama interactions.
- `multimodal.py`: Functions for Whisper (Audio) and Image description.
- `database.py`: Connectors for SQL databases.

## Verification Plan

### Automated Verification
- Unit tests for each loader (ensure PDF returns text, Audio returns transcript).
- Integration test: Ingest a sample folder -> Query the index -> Check if answer is contained.

### Manual Verification
1. Place a PDF, an Excel sheet, and an Image in `data/`.
2. Run `ingest.py`.
3. Start `app.py`.
4. Ask a question specific to the PDF.
5. Ask a question specific to the Image content.
6. Verify the bot cites the correct source.
