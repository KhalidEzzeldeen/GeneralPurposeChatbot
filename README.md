# ProBot: Enterprise Assistant Chatbot

A professional AI assistant chatbot built with Streamlit and LlamaIndex that can answer questions from both knowledge base (RAG) and database (SQL) sources.

## Features

- **Dual Source Intelligence**: Automatically routes queries between knowledge base (documents) and database (SQL)
- **Multi-Modal Support**: Processes PDFs, Excel, Images, Audio, and Video files
- **Session Management**: Maintains conversation history across page refreshes
- **Caching**: Intelligent caching for faster responses
- **Offline Capable**: Works with local Ollama models (no external API required)
- **Database Integration**: Natural language to SQL queries
- **Knowledge Base**: RAG (Retrieval Augmented Generation) over uploaded documents

## Prerequisites

1. **Python 3.11+**
2. **Ollama**: Install and ensure it's running
   ```bash
   # Pull required models
   ollama pull qwen2.5:7b-instruct
   ollama pull bge-m3
   ollama pull qwen2.5-vl  # Optional: for image understanding
   ```
3. **PostgreSQL** (optional): For database query features

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KhalidEzzeldeen/GeneralPurposeChatbot.git
   cd GeneralPurposeChatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the application**
   - Copy `config.json.example` to `config.json`
   - Update `config.json` with your database credentials (if using database features)
   - Configure your Ollama model settings

5. **Add your data**
   - Place documents in the `data/` folder
   - Supported formats: PDF, TXT, DOCX, MD, XLSX, CSV, Images, Audio, Video

6. **Ingest data** (optional - can also use Settings page)
   ```bash
   python ingest.py
   ```

## Usage

### Start the Application

```bash
streamlit run Home.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application

1. **Settings Page**: Configure LLM models, upload documents, and set up database connection
2. **Main Chat**: Ask questions - the system will automatically route to:
   - **Knowledge Base** for document-related questions
   - **Database** for queries containing keywords like "how many", "count", "list", etc.

### Example Queries

- **Knowledge Base**: "What is the policy for X?", "Tell me about drainage services"
- **Database**: "How many users are in the database?", "List all services from database"

## Architecture

- **Simple Router**: Keyword-based routing between knowledge base and database (no async agent to avoid event loop issues)
- **Session Management**: File-based session storage (can switch to Redis)
- **Caching**: JSON-based response caching
- **Vector Store**: ChromaDB for document embeddings
- **LLM**: Ollama (Qwen2.5) for text generation
- **Embeddings**: BGE-M3 for multilingual embeddings

## Project Structure

```
├── Home.py                 # Main Streamlit application
├── ingest.py              # Data ingestion script
├── modules/
│   ├── llm_engine.py      # LLM configuration
│   ├── knowledge_base.py  # RAG and document processing
│   ├── database.py        # SQL query engine
│   ├── session.py         # Session management
│   ├── cache.py           # Response caching
│   └── config.py          # Configuration management
├── pages/
│   └── 1_Settings.py      # Settings page
├── data/                  # Uploaded documents
└── chroma_db/             # Vector database (auto-generated)
```

## Configuration

Edit `config.json` to customize:
- LLM model and temperature
- Database connection details
- System prompts
- Data and vector store paths

## Notes

- The application uses a simple keyword-based router instead of an async agent for better Streamlit compatibility
- Sessions are stored in `sessions.json` (file-based)
- Cache is stored in `cache_store.json`
- Vector database is stored in `chroma_db/` directory

## License

This project is for enterprise use.
