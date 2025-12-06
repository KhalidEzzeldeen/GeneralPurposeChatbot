# ProBot: Offline Multi-Modal RAG Chatbot

This project is a professional AI assistant capable of digesting PDF, Excel, Images, Audio, and Video files to answer questions offline.

## Prerequisites
1. **Ollama**: Ensure Ollama is installed and running.
   - Run `ollama pull llama3.1`
   - Run `ollama pull llava`
   - Run `ollama pull nomic-embed-text`
2. **Python Environment**: (Already set up in `venv`)

## How to use

### 1. Add your data
Place all your documents in the `data/` folder.
- **PDFs, Text, Markdown**: Will be indexed as text.
- **Excel/CSV**: Will be converted to text for retrieval.
- **Images**: will be analyzed and described by the Vision Model.
- **Audio/Video**: Will be transcribed by Whisper.

### 2. Ingest Data
Run the ingestion script to process files and build the database:
```powershell
.\venv\Scripts\python ingest.py
```
*Note: The first time you run this, it might download the Whisper model which takes a few minutes.*

### 3. Start the Chatbot
Launch the interface:
```powershell
.\venv\Scripts\streamlit run app.py
```

## features
- **Offline**: No data leaves your machine.
- **Multi-modal**: Understands content from inside images and audio.
- **Professional UI**: Clean Streamlit interface.
