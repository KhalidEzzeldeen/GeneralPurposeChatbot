from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

def get_model_context_window(model_name):
    """
    Get appropriate context window size based on model.
    Smaller models may have smaller context windows.
    """
    # Model-specific context windows
    small_models = ["gemma2:2b", "phi3:mini", "qwen2.5:3b-instruct", "llama3.2:3b"]
    if any(m in model_name.lower() for m in ["2b", "3b", "mini"]):
        return 2048  # Smaller context for lighter models
    return 4096  # Default for larger models

def setup_llm_engine(model_name="qwen2.5:7b-instruct", embed_model_name="bge-m3", temperature=0.2):
    """
    Configures the LlamaIndex global settings to use Ollama models.
    Supports multiple model types with optimized settings.
    """
    
    # Get appropriate context window for the model
    context_window = get_model_context_window(model_name)
    
    # Text Generation Model
    llm = Ollama(
        model=model_name, 
        request_timeout=360.0,
        temperature=temperature,
        context_window=context_window,
        additional_kwargs={"num_ctx": context_window}
    )
    
    # Embedding Model (BGE-M3) - consistent across all LLM models
    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0}
    )
    
    # Apply to Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model

def check_model_available(model_name):
    """
    Check if a model is available in Ollama.
    Returns (is_available, error_message)
    """
    try:
        try:
            import requests
        except ImportError:
            return None, "requests library not available. Install with: pip install requests"
        
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Check if model exists (exact match or partial)
            for name in model_names:
                if model_name in name or name in model_name:
                    return True, None
            available = ', '.join(model_names[:5]) if model_names else "none"
            return False, f"Model '{model_name}' not found. Available models: {available}"
        return False, "Could not connect to Ollama. Make sure Ollama is running."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
    except Exception as e:
        return False, f"Error checking model: {str(e)}"
