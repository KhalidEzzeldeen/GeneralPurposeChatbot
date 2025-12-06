import os
import warnings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from modules.llm_engine import setup_llm_engine
from modules.multimodal import transcribe_audio, describe_image
from modules.cache import CacheManager
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

DATA_DIR = "./data"
CHROMADB_DIR = "./chroma_db"
COLLECTION_NAME = "chatbot_knowledge"

def process_file(file_path):
    cache = CacheManager()
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    documents = []
    
    # Check cache first for expensive extractions (Audio/Image)
    cached_content = cache.get_cached_extraction(file_path)
    if cached_content:
        print(f"Loaded {filename} from cache.")
        return [Document(text=cached_content, metadata={"source": filename, "from_cache": True})]

    print(f"Processing {filename}...")
    
    content_to_cache = None
    
    if ext in ['.pdf', '.txt', '.docx', '.md']:
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        # For text docs, we might not cache the whole text in JSON as it's efficient enough, 
        # but for OCR refined pages we would. Leaving standard loader for now.
        
    elif ext in ['.xlsx', '.csv']:
        try:
            if ext == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            content = df.to_string(index=False)
            documents = [Document(text=content, metadata={"source": filename, "type": "structured_data"})]
            content_to_cache = content
        except Exception as e:
            print(f"Error reading Excel/CSV {filename}: {e}")
            
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        description = describe_image(file_path, model="qwen2.5-vl")
        if description:
            documents = [Document(text=description, metadata={"source": filename, "type": "image", "visual_content_desc": description})]
            content_to_cache = description
            
    elif ext in ['.mp3', '.wav', '.mp4', '.m4a']:
        transcript = transcribe_audio(file_path)
        if transcript:
            documents = [Document(text=transcript, metadata={"source": filename, "type": "media", "transcript": True})]
            content_to_cache = transcript
            
    if content_to_cache:
        cache.set_cached_extraction(file_path, content_to_cache)
        
    return documents

def run_ingestion():
    setup_llm_engine(model_name="qwen2.5:7b-instruct", embed_model_name="bge-m3")
    
    db = chromadb.PersistentClient(path=CHROMADB_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    all_documents = []
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR}. Please put your files there.")
        return

    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    
    if not files:
        print("No files found in data directory.")
        return

    for f in files:
        filepath = os.path.join(DATA_DIR, f)
        docs = process_file(filepath)
        all_documents.extend(docs)
        
    if all_documents:
        print(f"Creating index from {len(all_documents)} documents (using BGE-M3)...")
        VectorStoreIndex.from_documents(
            all_documents, storage_context=storage_context
        )
        print("Ingestion complete. Index updated.")
    else:
        print("No content extracted.")

if __name__ == "__main__":
    run_ingestion()
