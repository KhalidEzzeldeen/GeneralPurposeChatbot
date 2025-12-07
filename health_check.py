#!/usr/bin/env python3
"""
Application Health Check Script
Checks the health of all components in the ProBot Enterprise Assistant application.
"""

import sys
import os
import json
from typing import Dict, List, Tuple
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_status(check_name: str, status: bool, message: str = ""):
    """Print a formatted status line."""
    # Use ASCII-compatible characters for Windows console
    status_icon = f"{Colors.GREEN}[OK]{Colors.RESET}" if status else f"{Colors.RED}[X]{Colors.RESET}"
    status_text = f"{Colors.GREEN}PASS{Colors.RESET}" if status else f"{Colors.RED}FAIL{Colors.RESET}"
    print(f"{status_icon} [{status_text}] {check_name}")
    if message:
        indent = " " * 4
        print(f"{indent}{Colors.YELLOW}-> {message}{Colors.RESET}")

def check_virtual_environment() -> Tuple[bool, str]:
    """Check if running in a virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        venv_path = sys.prefix
        return True, f"Running in virtual environment: {venv_path}"
    return False, "Not running in virtual environment. Activate venv first: .\\venv\\Scripts\\Activate.ps1"

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_config_file() -> Tuple[bool, str]:
    """Check if config.json exists and is valid."""
    if not os.path.exists("config.json"):
        return False, "config.json not found"
    
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = ["llm", "database", "data_path", "chroma_path"]
        missing = [s for s in required_sections if s not in config]
        if missing:
            return False, f"Missing required sections: {', '.join(missing)}"
        
        return True, "Configuration file is valid"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"Error reading config: {str(e)}"

def check_directories() -> Tuple[bool, str]:
    """Check if required directories exist."""
    config = {}
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
    except:
        return False, "Cannot read config.json"
    
    issues = []
    data_path = config.get("data_path", "./data")
    chroma_path = config.get("chroma_path", "./chroma_db")
    
    if not os.path.exists(data_path):
        issues.append(f"Data directory missing: {data_path}")
    if not os.path.exists(chroma_path):
        issues.append(f"ChromaDB directory missing: {chroma_path}")
    
    if issues:
        return False, "; ".join(issues)
    return True, f"Directories exist: {data_path}, {chroma_path}"

def check_python_packages() -> Tuple[bool, str]:
    """Check if required Python packages are installed."""
    required_packages = [
        "streamlit",
        "llama_index",
        "chromadb",
        "sqlalchemy",
        "psycopg2",
        "pandas",
        "ollama",
        "requests"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    return True, f"All {len(required_packages)} required packages installed"

def check_ollama_service() -> Tuple[bool, str]:
    """Check if Ollama service is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_count = len(models)
            return True, f"Ollama is running ({model_count} model(s) available)"
        return False, f"Ollama returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Is it running on localhost:11434?"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"

def check_ollama_models() -> Tuple[bool, str]:
    """Check if required Ollama models are available."""
    try:
        import requests
        with open("config.json", 'r') as f:
            config = json.load(f)
        
        model_name = config.get("llm", {}).get("model_name", "qwen2.5:7b-instruct")
        embed_model = "bge-m3"
        
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Cannot fetch models from Ollama"
        
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        issues = []
        # Check LLM model
        llm_found = any(model_name in name or name in model_name for name in model_names)
        if not llm_found:
            issues.append(f"LLM model '{model_name}' not found")
        
        # Check embedding model
        embed_found = any(embed_model in name or name in embed_model for name in model_names)
        if not embed_found:
            issues.append(f"Embedding model '{embed_model}' not found")
        
        if issues:
            available = ', '.join(model_names[:5]) if model_names else "none"
            return False, f"{'; '.join(issues)}. Available: {available}"
        
        return True, f"Required models available: {model_name}, {embed_model}"
    except Exception as e:
        return False, f"Error checking models: {str(e)}"

def check_database_connection() -> Tuple[bool, str]:
    """Check database connection."""
    try:
        from modules.database import DatabaseManager
        db_mgr = DatabaseManager()
        success, message = db_mgr.test_connection()
        if success:
            return True, "Database connection successful"
        return False, message
    except ImportError:
        return False, "Cannot import DatabaseManager module"
    except Exception as e:
        return False, f"Database check failed: {str(e)}"

def check_chromadb() -> Tuple[bool, str]:
    """Check ChromaDB accessibility."""
    try:
        import chromadb
        with open("config.json", 'r') as f:
            config = json.load(f)
        
        chroma_path = config.get("chroma_path", "./chroma_db")
        
        if not os.path.exists(chroma_path):
            return False, f"ChromaDB path does not exist: {chroma_path}"
        
        # Try to connect to ChromaDB
        db = chromadb.PersistentClient(path=chroma_path)
        collections = db.list_collections()
        
        # Check if chatbot_knowledge collection exists
        collection_names = [c.name for c in collections]
        if "chatbot_knowledge" in collection_names:
            collection = db.get_collection("chatbot_knowledge")
            count = collection.count()
            return True, f"ChromaDB accessible (collection 'chatbot_knowledge' has {count} documents)"
        else:
            return True, "ChromaDB accessible but 'chatbot_knowledge' collection not found (empty knowledge base)"
    except Exception as e:
        return False, f"ChromaDB check failed: {str(e)}"

def check_modules() -> Tuple[bool, str]:
    """Check if all required modules can be imported."""
    required_modules = [
        "modules.config",
        "modules.database",
        "modules.llm_engine",
        "modules.session",
        "modules.cache",
        "modules.intent_classifier",
        "modules.knowledge_base"
    ]
    
    failed = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError as e:
            failed.append(f"{module}: {str(e)}")
    
    if failed:
        return False, f"Failed to import: {'; '.join(failed)}"
    return True, f"All {len(required_modules)} modules importable"

def check_streamlit() -> Tuple[bool, str]:
    """Check if Streamlit can be imported and basic functionality works."""
    try:
        import streamlit as st
        version = st.__version__
        return True, f"Streamlit {version} available"
    except Exception as e:
        return False, f"Cannot import Streamlit: {str(e)}"

def run_health_check():
    """Run all health checks and display results."""
    print_header("ProBot Enterprise Assistant - Health Check")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Configuration File", check_config_file),
        ("Required Directories", check_directories),
        ("Python Packages", check_python_packages),
        ("Streamlit", check_streamlit),
        ("Application Modules", check_modules),
        ("Ollama Service", check_ollama_service),
        ("Ollama Models", check_ollama_models),
        ("Database Connection", check_database_connection),
        ("ChromaDB", check_chromadb),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            status, message = check_func()
            results.append((check_name, status, message))
            print_status(check_name, status, message)
        except Exception as e:
            results.append((check_name, False, f"Exception: {str(e)}"))
            print_status(check_name, False, f"Exception: {str(e)}")
    
    # Summary
    print_header("Summary")
    total = len(results)
    passed = sum(1 for _, status, _ in results if status)
    failed = total - passed
    
    print(f"Total Checks: {total}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
    
    # Overall status
    print()
    if failed == 0:
        print(f"{Colors.BOLD}{Colors.GREEN}[OK] All health checks passed! Application is healthy.{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.BOLD}{Colors.RED}[X] Some health checks failed. Please review the issues above.{Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    exit_code = run_health_check()
    sys.exit(exit_code)

