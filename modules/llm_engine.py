from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

def setup_llm_engine(model_name="qwen2.5:7b-instruct", embed_model_name="bge-m3", temperature=0.2):
    """
    Configures the LlamaIndex global settings to use Ollama models (Qwen + BGE-M3).
    """
    
    # Text Generation Model (Qwen2.5)
    llm = Ollama(
        model=model_name, 
        request_timeout=360.0,
        temperature=temperature,
        context_window=4096,
        additional_kwargs={"num_ctx": 4096}
    )
    
    # Embedding Model (BGE-M3)
    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0}
    )
    
    # Apply to Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model
