# RAG Pipeline for LLM_Markdown

A complete Retrieval-Augmented Generation (RAG) pipeline that processes multiple file formats from your `LLM_Markdown` folder and provides an interactive query interface.

## Features

✅ **Multi-format Support**: 
- Markdown (`.md`), text (`.txt`), and documentation files
- Source code files (`.py`, `.js`, `.java`, `.c`, `.cpp`, `.ts`, etc.)
- CSV files with structured data extraction
- Images (`.png`, `.jpg`, etc.) with OCR text extraction
- README files from Git repositories

✅ **Smart Processing**:
- File change detection to skip unchanged files
- Recursive directory scanning
- Intelligent text chunking for optimal embedding
- Metadata tracking for source attribution

✅ **Advanced Vector Search**:
- FAISS vector store for fast similarity search
- Maximum Marginal Relevance (MMR) search for diverse results
- Configurable search parameters (k, fetch_k, lambda_mult)
- Persistent storage (saves to disk)
- Incremental updates for new files

✅ **Multiple LLM Support**:
- OpenAI GPT models
- Anthropic Claude models  
- Fallback to open-source embeddings if no API keys

✅ **Web Server Hosting**:
- RESTful API endpoints for external applications
- CORS-enabled for cross-origin requests
- Health checks and system status monitoring
- Debug logging for document retrieval tracking
- Programmatic access to RAG functionality

## Installation

### 1. Install System Dependencies

**macOS:**
```bash
# Install Tesseract for OCR
brew install tesseract

# Optional: Install additional language packs
brew install tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-eng  # English language pack
```

**Windows:**
```bash
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
# Add tesseract to your PATH
```

### 2. Clone and Setup

```bash
# Navigate to your project directory
cd /path/to/your/project

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Setup Configuration

#### Environment Variables
```bash
# Copy the example env file
cp .env.example .env

# Edit .env file with your API keys
nano .env
```

Add your API keys to `.env`:
```env
# At least one is required
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

#### Configuration File
```bash
# Copy the example config file
cp config.example.json config.json

# Edit config file with your paths and settings
nano config.json
```

The `config.json` file contains all configurable settings:
```json
{
  "data_folder": "./documents",
  "vector_store_path": "./faiss_store", 
  "cache_path": "./file_cache.json",
  "text_processing": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_file_size_mb": 1
  },
  "api": {
    "host": "localhost",
    "port": 50001,
    "debug": false
  }
}
```

## Usage

### Quick Start

```bash
# Run the interactive RAG pipeline
python rag_pipeline.py

# Or run the web server API
python rag_api.py
```

The script will:
1. Scan your `LL_Markdown` folder recursively
2. Process all supported file types
3. Create embeddings and vector store
4. Start an interactive query interface

### First Run

When you first run the script:
- It will process all files in `/Users/nayan/Documents/Code/LLM_Markdown/`
- Create embeddings (this may take several minutes)
- Save the vector store to disk (`faiss_store/`)
- Start the interactive query mode

### Subsequent Runs

On subsequent runs:
- Only changed files will be reprocessed
- Existing vector store will be loaded quickly
- New files will be added incrementally

### Interactive Querying

Once the system is ready, you can ask questions like:

```
❓ Your question: What are the main features discussed in the documentation?

❓ Your question: Show me Python functions related to authentication

❓ Your question: What data is in the CSV files?

❓ Your question: Summarize the README files from the repositories
```

Type `quit` or `exit` to stop.

### Web Server API

Start the web server for programmatic access:

```bash
python rag_api.py
```

The API server will run on `http://localhost:50001` by default and provides the following endpoints:

#### API Endpoints

**Health Check:**
```bash
curl http://localhost:50001/health
```

**Query RAG System:**
```bash
curl -X POST http://localhost:50001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main features?", "k": 5}'
```

**Similarity Search (no LLM):**
```bash
# Basic similarity search
curl -X POST http://localhost:50001/similarity_search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication", "k": 5}'

# MMR search for diverse results
curl -X POST http://localhost:50001/similarity_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "GraphQL security vulnerabilities",
    "k": 15,
    "search_type": "mmr",
    "fetch_k": 45,
    "lambda_mult": 0.5
  }'
```

**Rebuild Vector Store:**
```bash
curl -X POST http://localhost:50001/rebuild \
  -H "Content-Type: application/json" \
  -d '{"data_folder": "/path/to/documents"}'
```

**System Status:**
```bash
curl http://localhost:50001/status
```

#### Environment Variables for API

```env
# API server configuration
RAG_API_HOST=localhost
RAG_API_PORT=50001
RAG_API_DEBUG=false
```

## Advanced Search Features

### Maximum Marginal Relevance (MMR) Search

The system now supports MMR search for better document diversity, preventing the retrieval of multiple similar chunks from the same document.

#### Search Types

**1. Similarity Search** - Traditional cosine similarity
- Returns most relevant documents based on semantic similarity
- May return multiple chunks from the same document

**2. MMR Search** - Maximum Marginal Relevance
- Balances relevance with diversity 
- Selects diverse documents from different sources
- Better for comprehensive topic coverage

#### API Parameters

All search endpoints (`/query` and `/similarity_search`) support these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 15 | Number of documents to retrieve |
| `search_type` | string | "mmr" | "similarity" or "mmr" |
| `fetch_k` | int | k*3 | Candidates to fetch for MMR (only for MMR) |
| `lambda_mult` | float | 0.5 | Relevance vs diversity balance (0-1) |

#### Lambda Multiplier Guide

- `1.0` - Pure relevance (same as similarity search)
- `0.7` - Mostly relevance, some diversity
- `0.5` - Balanced relevance and diversity *(recommended)*
- `0.3` - Mostly diversity, some relevance
- `0.0` - Pure diversity (maximum different documents)

#### Example Requests

**High Relevance Search:**
```json
{
  "query": "GraphQL security best practices",
  "k": 10,
  "search_type": "mmr",
  "lambda_mult": 0.8
}
```

**Diverse Topic Coverage:**
```json
{
  "query": "authentication authorization security",
  "k": 15, 
  "search_type": "mmr",
  "fetch_k": 50,
  "lambda_mult": 0.3
}
```

### Debug Logging

The system now includes comprehensive debug logging to track document retrieval:

#### Log Format
```
📂 DEBUG - Retrieved X documents for query: 'your question'
📂 DEBUG - Files fetched: ['file1.md', 'file2.py', 'file3.txt']
```

#### Enabling Debug Logs
Set the logging level to see debug information:
```bash
# In your terminal
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.INFO)"
python rag_api.py
```

Or enable debug mode in config:
```json
{
  "api": {
    "debug": true
  }
}
```

## File Structure

```
Security_RAG_Generator/
├── rag_pipeline.py      # Main RAG pipeline script
├── rag_api.py          # Flask web server API
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── .env               # Your API keys (create this)
├── config.example.json # Configuration template
├── config.json        # Your configuration (create this)
├── faiss_store/       # Vector store files (created automatically)
├── file_cache.json    # File processing cache (created automatically)
└── documents/         # Your documents folder (configurable)
    ├── docs/
    ├── repos/
    ├── images/
    └── ...
```

## Supported File Types

| Type | Extensions | Processing |
|------|-----------|------------|
| Text | `.md`, `.txt`, `.rst` | Direct text extraction |
| Code | `.py`, `.js`, `.java`, `.c`, `.cpp`, `.ts`, etc. | Code-aware processing |
| Images | `.png`, `.jpg`, `.jpeg`, `.gif` | OCR text extraction |
| Data | `.csv`, `.tsv` | Structured data extraction |
| Docs | `README`, `README.md` | Documentation processing |

## Configuration

All configuration is managed through the `config.json` file. Here are the key settings:

### File Paths
```json
{
  "data_folder": "./documents",           // Path to your documents
  "vector_store_path": "./faiss_store",   // Where to store vector database
  "cache_path": "./file_cache.json"       // File processing cache
}
```

### Text Processing
```json
{
  "text_processing": {
    "chunk_size": 1000,          // Text chunk size for embeddings
    "chunk_overlap": 200,        // Overlap between chunks
    "max_file_size_mb": 1,       // Max file size to process
    "encodings": ["utf-8", "utf-16", "latin-1", "cp1252"]
  }
}
```

### Search and Retrieval
```json
{
  "retrieval": {
    "search_type": "mmr",           // "similarity" or "mmr" 
    "k": 15,                        // Number of documents to retrieve
    "fetch_k": 45,                  // Candidates for MMR selection
    "lambda_mult": 0.5,             // Balance: 1.0=relevance, 0.0=diversity
    "content_preview_length": 200   // Preview length in API responses
  }
}
```

### LLM Models
```json
{
  "llm": {
    "anthropic_model": "claude-3-sonnet-20240229",
    "openai_model": "gpt-3.5-turbo",
    "temperature": 0
  },
  "embeddings": {
    "openai_model": "text-embedding-ada-002",
    "huggingface_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

### API Server
```json
{
  "api": {
    "host": "localhost",
    "port": 50001,
    "debug": false
  }
}
```

### Model Priority

The system uses available API keys in this order:
1. Anthropic Claude (if `ANTHROPIC_API_KEY` is set)
2. OpenAI GPT (if `OPENAI_API_KEY` is set)  
3. Local HuggingFace embeddings (fallback)

## Troubleshooting

### Common Issues

**1. Tesseract not found:**
```bash
# Add tesseract path to .env
TESSERACT_CMD=/usr/local/bin/tesseract
```

**2. Memory issues with large files:**
- Files larger than 1MB are automatically skipped
- Adjust the limit in `_read_text_file()` if needed

**3. Encoding errors:**
- The script tries multiple encodings automatically
- Check file encoding if specific files fail

**4. API rate limits:**
- Use smaller batch sizes
- Add delays between API calls
- Consider using local models

### Performance Tips

- Use SSD storage for vector store
- Set `FAISS_ENABLE_GPU=1` if you have CUDA
- Use OpenAI embeddings for better quality
- Keep chunk sizes reasonable (1000-2000 tokens)

## Advanced Usage

### Direct Python Usage

```python
from rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline(data_folder="custom_folder")

# Load and process documents
documents = rag.load_documents()
rag.embed_and_store(documents)

# Setup QA
rag.load_vector_store()
rag.setup_qa_chain()

# Query
result = rag.query_rag("Your question here")
print(result["answer"])
```

### API Integration

```python
import requests

# Query via API
response = requests.post('http://localhost:50001/query', json={
    'question': 'What are the main features?',
    'k': 5
})
result = response.json()
print(result['answer'])

# Basic similarity search
response = requests.post('http://localhost:50001/similarity_search', json={
    'query': 'authentication',
    'k': 5
})
documents = response.json()['results']

# MMR search for diverse results
response = requests.post('http://localhost:50001/similarity_search', json={
    'query': 'GraphQL security vulnerabilities',
    'k': 15,
    'search_type': 'mmr',
    'fetch_k': 45,
    'lambda_mult': 0.5
})
diverse_documents = response.json()['results']
```

### Batch Processing

```python
questions = [
    "What are the main topics?",
    "Show me code examples",
    "Summarize the data"
]

for q in questions:
    result = rag.query_rag(q)
    print(f"Q: {q}")
    print(f"A: {result['answer']}\n")
```

## License

This project is provided as-is for educational and research purposes.