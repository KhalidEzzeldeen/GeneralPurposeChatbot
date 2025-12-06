# Project Structure

This document describes the organization of the ProBot Enterprise Assistant project.

## Directory Structure

```
Chatbot/
├── Home.py                 # Main Streamlit application entry point
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
│
├── modules/               # Core application modules
│   ├── cache.py          # Caching manager
│   ├── config.py         # Configuration manager
│   ├── database.py       # Database connection and SQL query engine
│   ├── intent_classifier.py  # LLM-based intent classification
│   ├── knowledge_base.py # Vector knowledge base manager
│   ├── llm_engine.py     # LLM and embedding model setup
│   ├── model_registry.py # Pre-configured model registry
│   ├── multimodal.py     # Multimodal processing (images, audio, video)
│   ├── session.py        # Session management
│   └── streaming.py     # Response streaming utilities
│
├── pages/                # Streamlit pages
│   └── 1_Settings.py     # Application settings page
│
├── scripts/              # Utility scripts
│   └── ingest.py         # Data ingestion script
│
├── docs/                 # Documentation files
│   ├── FEATURE_ENHANCED_ROUTING.md
│   ├── HOW_ROUTING_WORKS.md
│   ├── INTELLIGENT_SCHEMA_ROUTING.md
│   ├── MODEL_SELECTION_GUIDE.md
│   ├── PERFORMANCE_IMPROVEMENT_GUIDE.md
│   ├── PERFORMANCE_OPTIMIZATIONS_IMPLEMENTED.md
│   ├── ROADMAP.md
│   ├── ROUTING_FIX.md
│   ├── SCHEMA_AWARE_ROUTING.md
│   ├── implementation_plan.md
│   ├── implementation_plan_v2.md
│   └── task_v2.md
│
├── data/                 # Data files (documents, Excel, etc.)
│   ├── *.xlsx           # Excel files
│   ├── *.docx           # Word documents
│   ├── *.txt            # Text files
│   └── db_schema_scan.txt
│
├── config/               # Configuration templates
│   └── config.json.example
│
├── logs/                 # Application logs (if any)
│
├── chroma_db/           # ChromaDB vector database (auto-generated)
├── cache_store.json     # Response cache (auto-generated)
├── sessions.json        # Session data (auto-generated)
└── config.json          # Runtime configuration (auto-generated, gitignored)
```

## File Descriptions

### Core Application
- **Home.py**: Main Streamlit application with chat interface, routing logic, and query handling
- **requirements.txt**: Python package dependencies

### Modules (`modules/`)
- **cache.py**: In-memory LRU cache with background persistence
- **config.py**: Configuration manager for application settings
- **database.py**: Database connection pooling, schema scanning, SQL query engine
- **intent_classifier.py**: LLM-based intent classification for routing
- **knowledge_base.py**: Vector knowledge base management and file ingestion
- **llm_engine.py**: LLM and embedding model initialization
- **model_registry.py**: Pre-configured model registry with metadata
- **multimodal.py**: Image, audio, and video processing
- **session.py**: User session management (file-based or Redis)
- **streaming.py**: Response streaming utilities

### Pages (`pages/`)
- **1_Settings.py**: Settings page for LLM configuration, knowledge base, and database

### Scripts (`scripts/`)
- **ingest.py**: Standalone script for ingesting files into the knowledge base

### Documentation (`docs/`)
- Feature documentation and implementation guides
- Performance optimization guides
- Routing and schema-aware routing documentation
- Model selection guide

### Data (`data/`)
- User-uploaded documents (Excel, Word, Text)
- Sample data files
- Database schema scans

### Configuration
- **config.json**: Runtime configuration (contains sensitive data, gitignored)
- **config/config.json.example**: Configuration template

### Generated Files (gitignored)
- **chroma_db/**: ChromaDB vector database storage
- **cache_store.json**: Cached query responses
- **sessions.json**: User session data
- **__pycache__/**: Python bytecode cache

## Key Features

1. **Modular Architecture**: Clean separation of concerns with dedicated modules
2. **Documentation**: Comprehensive docs in `docs/` folder
3. **Configuration Management**: Centralized config with example template
4. **Data Organization**: User data in `data/` folder
5. **Scripts**: Utility scripts in `scripts/` folder

## Running the Application

```bash
streamlit run Home.py
```

## Adding New Features

1. Add new modules to `modules/`
2. Add new pages to `pages/`
3. Update documentation in `docs/`
4. Update `requirements.txt` if needed

