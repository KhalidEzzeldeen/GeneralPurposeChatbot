import os
import hashlib
import warnings
import pandas as pd
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from modules.llm_engine import setup_llm_engine
from modules.multimodal import transcribe_audio, describe_image
from modules.config import ConfigManager

# Suppress warnings
warnings.filterwarnings("ignore")

class KnowledgeBase:
    def __init__(self):
        self.config = ConfigManager()
        self.chroma_dir = self.config.get("chroma_path")
        self.collection_name = "chatbot_knowledge"
        
        # Ensure Chroma DB Directory exists
        if not os.path.exists(self.chroma_dir):
            os.makedirs(self.chroma_dir)
            
        self.db = chromadb.PersistentClient(path=self.chroma_dir)
        self.chroma_collection = self.db.get_or_create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Setup Models from Config
        llm_conf = self.config.get("llm")
        setup_llm_engine(model_name=llm_conf["model_name"])

    def _calculate_hash(self, file_path):
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_file_ingested(self, file_hash):
        """Check if file hash is in the registry."""
        ingested_files = self.config.get("state", "ingested_files") or []
        for item in ingested_files:
            if item.get("hash") == file_hash:
                return True
        return False

    def register_file(self, filename, file_hash):
        """Register file as ingested in config."""
        ingested_files = self.config.get("state", "ingested_files") or []
        ingested_files.append({"filename": filename, "hash": file_hash})

        # Wait, ConfigManager.set takes (section, key, value). 
        # But "ingested_files" is a top-level key in my DEFAULT_CONFIG.
        # I need to adjust ConfigManager to handle top-level keys or put it in a section.
        # Let's put it in a "knowledge_base" section or handle root keys.
        # Checking config.py again... set(section, key, value) creates section if missing.
        # So I will use section "state" or just reuse "ingested_files" as section? 
        # Ideally, I should fix ConfigManager to support root keys, but for now I'll use a section.
        self.config.set("state", "ingested_files", ingested_files)

    def process_file(self, file_path):
        """Process a single file and return documents."""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        documents = []
        
        print(f"Processing {filename}...")
        
        if ext in ['.pdf', '.txt', '.docx', '.md']:
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            
        elif ext in ['.xlsx', '.csv']:
            try:
                if ext == '.xlsx':
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                content = df.to_string(index=False)
                documents = [Document(text=content, metadata={"source": filename, "type": "structured_data"})]
            except Exception as e:
                print(f"Error reading Excel/CSV {filename}: {e}")
                
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            description = describe_image(file_path, model="qwen2.5-vl")
            if description:
                documents = [Document(text=description, metadata={"source": filename, "type": "image", "visual_content_desc": description})]
                
        elif ext in ['.mp3', '.wav', '.mp4', '.m4a']:
            transcript_data = transcribe_audio(file_path)
            if transcript_data:
                # Create multiple documents, one for each segment
                for segment in transcript_data:
                    doc = Document(
                        text=segment["text"],
                        metadata={
                            "source": filename,
                            "type": "media",
                            "transcript": True,
                            "start_time": segment["start"],
                            "end_time": segment["end"]
                        }
                    )
                    documents.append(doc)
        
        return documents

    def ingest_file(self, file_path):
        """Ingest a single file if new."""
        if not os.path.exists(file_path):
            return "File not found."
            
        file_hash = self._calculate_hash(file_path)
        
        # Check Registry (Incremental Logic)
        ingested_list = self.config.get("state", "ingested_files") or []
        # Support legacy root key execution if 'state' is empty but root has it (migration edge case), strictly use state now.
        
        for item in ingested_list:
            if item.get("hash") == file_hash:
                print(f"Skipping {os.path.basename(file_path)} (Already Ingested)")
                return "Skipped (Duplicate)"

        docs = self.process_file(file_path)
        
        if docs:
            print(f"Embedding {len(docs)} chunks for {os.path.basename(file_path)}...")
            index = VectorStoreIndex.from_documents(
                docs, storage_context=self.storage_context
            )
            self.register_file(os.path.basename(file_path), file_hash)
            return "Ingested"
        
        return "Failed (No Content)"

    def clear_index(self):
        """Clear the vector database and registry."""
        self.db.delete_collection(self.collection_name)
        self.chroma_collection = self.db.get_or_create_collection(self.collection_name)
        # Reset Registry
        self.config.set("state", "ingested_files", [])
        return "Knowledge Base Cleared."

    def get_ingested_files(self):
         return self.config.get("state", "ingested_files") or []
