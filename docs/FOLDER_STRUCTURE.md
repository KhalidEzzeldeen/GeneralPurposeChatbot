# Project Folder Structure

```
Chatbot - Copy/
│
├── Home.py                 # Main Streamlit application entry point
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── config.json           # Application configuration (gitignored)
│
├── modules/              # Core application modules
│   ├── cache.py         # Caching system (LRU cache with persistence)
│   ├── config.py        # Configuration manager
│   ├── database.py      # Database connection and SQL query engine
│   ├── intent_classifier.py  # LLM-based intent classification
│   ├── knowledge_base.py    # Vector knowledge base management
│   ├── llm_engine.py    # LLM and embedding model setup
│   ├── model_registry.py    # Pre-configured model registry
│   ├── multimodal.py    # Multimodal processing (audio, images)
│   ├── session.py        # Session management
│   └── streaming.py     # Response streaming utilities
│
├── pages/                # Streamlit pages
│   └── 1_Settings.py     # Settings page
│
├── scripts/              # Utility scripts
│   └── ingest.py        # Data ingestion script
│
├── docs/                 # Documentation
│   ├── FEATURE_ENHANCED_ROUTING.md
│   ├── HOW_ROUTING_WORKS.md
│   ├── INTELLIGENT_SCHEMA_ROUTING.md
│   ├── MODEL_SELECTION_GUIDE.md
│   ├── PERFORMANCE_IMPROVEMENT_GUIDE.md
│   ├── PERFORMANCE_OPTIMIZATIONS_IMPLEMENTED.md
│   ├── PROJECT_STRUCTURE.md
│   ├── ROADMAP.md
│   ├── ROUTING_FIX.md
│   ├── SCHEMA_AWARE_ROUTING.md
│   ├── implementation_plan.md
│   ├── implementation_plan_v2.md
│   └── task_v2.md
│
├── data/                 # Data files (documents, Excel, etc.)
│   ├── db_schema_scan.txt
│   ├── Details.docx
│   ├── DetailsGPT.docx
│   ├── installed_models.txt
│   ├── sample_policy.txt
│   ├── Sharjah Applications.xlsx
│   ├── sharjah_municipality_services_sample.xlsx
│   └── Tasks.txt
│
├── storage/              # Runtime storage (gitignored)
│   ├── cache_store.json  # Query cache
│   └── sessions.json     # User sessions
│
├── chroma_db/            # ChromaDB vector database (gitignored)
│
├── logs/                 # Application logs (gitignored)
│
├── config/               # Configuration templates
│   └── config.json.example
│
└── venv/                 # Python virtual environment (gitignored)
```

## Directory Purposes

- **Root**: Main application files and configuration
- **modules/**: Core application logic and utilities
- **pages/**: Streamlit multi-page application pages
- **scripts/**: Utility and helper scripts
- **docs/**: All project documentation
- **data/**: User data files (documents, Excel files, etc.)
- **storage/**: Runtime files (cache, sessions) - auto-created
- **chroma_db/**: Vector database storage
- **logs/**: Application logs
- **config/**: Configuration templates and examples
- **venv/**: Python virtual environment

## File Organization Rules

1. **Keep at root**: Essential files needed to run the app (Home.py, README.md, requirements.txt, config.json)
2. **modules/**: All reusable Python modules
3. **docs/**: All markdown documentation
4. **storage/**: Runtime files that are auto-generated
5. **data/**: User-uploaded or project data files

