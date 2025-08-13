#!/usr/bin/env python3
"""
Complete RAG Pipeline for LL_Markdown folder
Supports multiple file formats: .md, .txt, .csv, .png, and source code files
Uses FAISS for vector storage and supports multiple LLM providers
"""

import os
import sys
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Core libraries
import pandas as pd
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file {config_path}: {e}")
        return {}

class RAGPipeline:
    def __init__(self, 
                 data_folder: Optional[str] = None,
                 vector_store_path: Optional[str] = None,
                 cache_path: Optional[str] = None,
                 config_path: str = "config.json"):
        """
        Initialize RAG Pipeline
        
        Args:
            data_folder: Path to folder containing documents (overrides config)
            vector_store_path: Path to save FAISS vector store (overrides config)
            cache_path: Path to save file processing cache (overrides config)
            config_path: Path to configuration JSON file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Use provided values or fall back to config, then defaults
        self.data_folder = Path(data_folder or self.config.get('data_folder', './documents'))
        self.vector_store_path = vector_store_path or self.config.get('vector_store_path', './faiss_store')
        self.cache_path = cache_path or self.config.get('cache_path', './file_cache.json')
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.qa_chain = None
        
        # File extensions to process (from config or defaults)
        file_exts = self.config.get('file_extensions', {})
        self.text_extensions = set(file_exts.get('text', ['.md', '.txt', '.rst', '.adoc']))
        self.code_extensions = set(file_exts.get('code', ['.py', '.js', '.java', '.c', '.cpp', '.ts', '.tsx', '.jsx', '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.scala', '.r', '.sql']))
        self.image_extensions = set(file_exts.get('image', ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']))
        self.csv_extensions = set(file_exts.get('csv', ['.csv', '.tsv']))
        
        # Initialize text splitter (from config or defaults)
        text_config = self.config.get('text_processing', {})
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_config.get('chunk_size', 1000),
            chunk_overlap=text_config.get('chunk_overlap', 200),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Other configuration
        self.max_file_size_bytes = text_config.get('max_file_size_mb', 1) * 1024 * 1024
        self.encodings = text_config.get('encodings', ['utf-8', 'utf-16', 'latin-1', 'cp1252'])
        self.csv_preview_rows = self.config.get('csv_processing', {}).get('preview_rows', 10)
        
        # Load file cache
        self.file_cache = self._load_cache()
        
    def _load_cache(self) -> Dict[str, Dict]:
        """Load file processing cache"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save file processing cache"""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.file_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return ""
    
    def _has_file_changed(self, file_path: Path) -> bool:
        """Check if file has changed since last processing"""
        file_str = str(file_path)
        current_hash = self._get_file_hash(file_path)
        
        if file_str not in self.file_cache:
            return True
            
        return self.file_cache[file_str].get('hash') != current_hash
    
    def _update_file_cache(self, file_path: Path):
        """Update cache entry for processed file"""
        file_str = str(file_path)
        self.file_cache[file_str] = {
            'hash': self._get_file_hash(file_path),
            'processed_at': datetime.now().isoformat(),
            'size': file_path.stat().st_size
        }
    
    def _read_text_file(self, file_path: Path) -> str:
        """Read text content from file with encoding detection"""
        for encoding in self.encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path} with {encoding}: {e}")
                continue
        
        logger.warning(f"Could not read {file_path} with any encoding")
        return ""
    
    def _process_image_file(self, file_path: Path) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.warning(f"Could not process image {file_path}: {e}")
            return ""
    
    def _process_csv_file(self, file_path: Path) -> str:
        """Process CSV file into structured text"""
        try:
            df = pd.read_csv(file_path)
            
            # Create structured representation
            text_parts = [f"CSV File: {file_path.name}"]
            text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
            text_parts.append(f"Number of rows: {len(df)}")
            text_parts.append("\nData preview:")
            
            # Add first few rows as text
            for i, row in df.head(self.csv_preview_rows).iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                text_parts.append(f"Row {i+1}: {row_text}")
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"Could not process CSV {file_path}: {e}")
            return ""
    
    def load_documents(self) -> List[Document]:
        """
        Load and process all documents from the data folder
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        if not self.data_folder.exists():
            logger.error(f"Data folder {self.data_folder} does not exist")
            return documents
        
        logger.info(f"Loading documents from {self.data_folder}")
        
        # Walk through all files recursively
        for file_path in self.data_folder.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Skip hidden files and common non-text files
            if file_path.name.startswith('.') and not file_path.name.endswith(('.md', '.txt')):
                continue
            
            # Check if file has changed
            if not self._has_file_changed(file_path):
                logger.debug(f"Skipping unchanged file: {file_path}")
                continue
            
            file_extension = file_path.suffix.lower()
            content = ""
            
            try:
                # Process different file types
                if file_extension in self.text_extensions or file_path.name.lower() in ['readme', 'readme.md', 'readme.txt']:
                    content = self._read_text_file(file_path)
                elif file_extension in self.code_extensions:
                    content = self._read_text_file(file_path)
                    content = f"Source code file ({file_extension}):\n{content}"
                elif file_extension in self.csv_extensions:
                    content = self._process_csv_file(file_path)
                elif file_extension in self.image_extensions:
                    content = self._process_image_file(file_path)
                    if content:
                        content = f"OCR extracted text from image:\n{content}"
                else:
                    # Try to read as text for unknown extensions
                    if file_path.stat().st_size < self.max_file_size_bytes:
                        content = self._read_text_file(file_path)
                
                if content and content.strip():
                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "file_type": file_extension,
                            "file_name": file_path.name,
                            "file_size": file_path.stat().st_size,
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
                    self._update_file_cache(file_path)
                    logger.info(f"Processed: {file_path}")
                else:
                    logger.debug(f"No content extracted from: {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Save cache after processing
        self._save_cache()
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _setup_embeddings(self):
        """Setup embedding model (OpenAI or HuggingFace)"""
        openai_api_key = os.getenv('OPENAI_API_KEY')
        embeddings_config = self.config.get('embeddings', {})
        
        if openai_api_key:
            logger.info("Using OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(
                model=embeddings_config.get('openai_model', 'text-embedding-ada-002'),
                openai_api_key=openai_api_key
            )
        else:
            model_name = embeddings_config.get('huggingface_model', 'sentence-transformers/all-MiniLM-L6-v2')
            device = embeddings_config.get('device', 'cpu')
            logger.info(f"Using HuggingFace embeddings ({model_name})")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device}
            )
    
    def embed_and_store(self, documents: List[Document]) -> bool:
        """
        Create embeddings and store in FAISS vector store
        
        Args:
            documents: List of documents to embed
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents to embed")
            return False
        
        try:
            # Setup embeddings
            self._setup_embeddings()
            
            # Split documents into chunks
            logger.info("Splitting documents into chunks...")
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(texts)} text chunks")
            
            # Check if vector store already exists
            if os.path.exists(f"{self.vector_store_path}/index.faiss"):
                logger.info("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Add new documents
                if texts:
                    logger.info("Adding new documents to existing vector store...")
                    self.vector_store.add_documents(texts)
            else:
                # Create new vector store
                logger.info("Creating new vector store...")
                self.vector_store = FAISS.from_documents(texts, self.embeddings)
            
            # Save vector store
            logger.info(f"Saving vector store to {self.vector_store_path}")
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return False
    
    def _setup_llm(self):
        """Setup LLM (OpenAI, Anthropic, or local)"""
        openai_api_key = os.getenv('OPENAI_API_KEY')
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        llm_config = self.config.get('llm', {})
        temperature = llm_config.get('temperature', 0)
        
        if anthropic_api_key:
            model = llm_config.get('anthropic_model', 'claude-3-sonnet-20240229')
            logger.info(f"Using Anthropic Claude ({model})")
            self.llm = ChatAnthropic(
                model=model,
                anthropic_api_key=anthropic_api_key,
                temperature=temperature
            )
        elif openai_api_key:
            model = llm_config.get('openai_model', 'gpt-3.5-turbo')
            logger.info(f"Using OpenAI GPT ({model})")
            self.llm = ChatOpenAI(
                model=model,
                openai_api_key=openai_api_key,
                temperature=temperature
            )
        else:
            logger.error("No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            return False
        
        return True
    
    def load_vector_store(self) -> bool:
        """Load existing vector store"""
        try:
            if not os.path.exists(f"{self.vector_store_path}/index.faiss"):
                logger.warning("No existing vector store found")
                return False
            
            self._setup_embeddings()
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def setup_qa_chain(self):
        """Setup the QA chain for querying"""
        if not self.vector_store:
            logger.error("Vector store not available")
            return False
        
        if not self._setup_llm():
            return False
        
        # Create custom prompt template
        prompt_template = """You are an AI assistant that answers questions based on the provided context from various documents including markdown files, code, images, and CSV data.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so clearly.

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        retrieval_config = self.config.get('retrieval', {})
        search_type = retrieval_config.get('search_type', 'similarity')
        k = retrieval_config.get('k', 15)
        
        # Setup search kwargs based on search type
        search_kwargs = {"k": k}
        if search_type == 'mmr':
            search_kwargs.update({
                "fetch_k": retrieval_config.get('fetch_k', k * 3),
                "lambda_mult": retrieval_config.get('lambda_mult', 0.5)
            })
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        logger.info("QA chain setup complete")
        return True
    
    def query_rag(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        if not self.qa_chain:
            logger.error("QA chain not setup")
            return {"answer": "QA chain not available", "sources": []}
        
        try:
            # Query the chain
            result = self.qa_chain({"query": question})
            
            # Debug logging: Show what files were retrieved
            source_docs = result.get("source_documents", [])
            retrieved_files = [doc.metadata.get("file_name", "Unknown") for doc in source_docs]
            logger.info(f"📂 DEBUG - Retrieved {len(source_docs)} documents for RAG query: '{question}'")
            logger.info(f"📂 DEBUG - Files fetched: {retrieved_files}")
            
            # Extract sources
            sources = []
            preview_length = self.config.get('retrieval', {}).get('content_preview_length', 200)
            for doc in result.get("source_documents", []):
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "file_name": doc.metadata.get("file_name", "Unknown"),
                    "file_type": doc.metadata.get("file_type", "Unknown"),
                    "content_preview": doc.page_content[:preview_length] + "..." if len(doc.page_content) > preview_length else doc.page_content
                })
            
            return {
                "answer": result["result"],
                "sources": sources,
                "query": question
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return {"answer": f"Error: {e}", "sources": []}

def main():
    """Main function to run the RAG pipeline"""
    print("🔍 RAG Pipeline for LLM_Markdown")
    print("=" * 50)
    
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Check if vector store exists
    if os.path.exists(f"{rag.vector_store_path}/index.faiss"):
        print("📁 Found existing vector store")
        choice = input("Do you want to:\n1. Load existing store\n2. Rebuild from scratch\nChoice (1/2): ").strip()
        
        if choice == "1":
            if rag.load_vector_store():
                print("✅ Vector store loaded")
            else:
                print("❌ Failed to load vector store")
                return
        else:
            print("🔄 Rebuilding from scratch...")
            # Load and process documents
            documents = rag.load_documents()
            if not documents:
                print("❌ No documents found")
                return
            
            # Create embeddings and store
            if not rag.embed_and_store(documents):
                print("❌ Failed to create embeddings")
                return
            print("✅ Vector store created")
    else:
        print("📚 Loading documents...")
        # Load and process documents
        documents = rag.load_documents()
        if not documents:
            print("❌ No documents found")
            return
        
        # Create embeddings and store
        print("🔄 Creating embeddings (this may take a while)...")
        if not rag.embed_and_store(documents):
            print("❌ Failed to create embeddings")
            return
        print("✅ Vector store created")
    
    # Setup QA chain
    print("🤖 Setting up QA chain...")
    if not rag.setup_qa_chain():
        print("❌ Failed to setup QA chain")
        return
    print("✅ QA chain ready")
    
    # Interactive query loop
    print("\n💬 Interactive Query Mode")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 30)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🔍 Searching...")
            result = rag.query_rag(question)
            
            print(f"\n💡 Answer:")
            print(result["answer"])
            
            if result["sources"]:
                print(f"\n📚 Sources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"  {i}. {source['file_name']} ({source['file_type']})")
                    print(f"     Path: {source['source']}")
                    print(f"     Preview: {source['content_preview']}")
                    print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()