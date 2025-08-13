#!/usr/bin/env python3
"""
Flask API for RAG Pipeline
Exposes RAG functionality via REST endpoints for external applications
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import RAGPipeline, load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config("config.json")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global RAG instance
rag_instance: Optional[RAGPipeline] = None

def get_rag_instance() -> RAGPipeline:
    """Get or create RAG pipeline instance"""
    global rag_instance
    if rag_instance is None:
        rag_instance = RAGPipeline(config_path="config.json")
        # Try to load existing vector store
        if not rag_instance.load_vector_store():
            logger.warning("No existing vector store found. Use /rebuild endpoint to create one.")
        else:
            # Setup QA chain if vector store loaded successfully
            if not rag_instance.setup_qa_chain():
                logger.error("Failed to setup QA chain")
    return rag_instance

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        rag = get_rag_instance()
        status = {
            "status": "healthy",
            "vector_store_available": rag.vector_store is not None,
            "qa_chain_available": rag.qa_chain is not None,
            "embeddings_available": rag.embeddings is not None
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_rag():
    """Query the RAG system"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        # Optional parameters
        k = data.get('k', config.get('retrieval', {}).get('k', 15))  # Number of relevant chunks to retrieve
        
        # Get RAG instance and query
        rag = get_rag_instance()
        
        if not rag.qa_chain:
            return jsonify({
                "error": "QA chain not available. Vector store may not be loaded."
            }), 503
        
        # Perform query
        result = rag.query_rag(question, k)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/similarity_search', methods=['POST'])
def similarity_search():
    """Perform similarity search without LLM generation"""
    logger.info(f"📥 DEBUG - Raw request: {request.get_data()}")
    try:
        # Get request data
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        k = data.get('k', config.get('retrieval', {}).get('k', 15))
        search_type = data.get('search_type', config.get('retrieval', {}).get('search_type', 'similarity'))
        fetch_k = data.get('fetch_k', config.get('retrieval', {}).get('fetch_k', k * 3))
        lambda_mult = data.get('lambda_mult', config.get('retrieval', {}).get('lambda_mult', 0.5))
        
        # Debug logging: Show incoming request parameters
        logger.info(f"📋 DEBUG - Incoming request data: {data}")
        logger.info(f"📋 DEBUG - Search parameters: k={k}, search_type='{search_type}', fetch_k={fetch_k}, lambda_mult={lambda_mult}")
        
        # Get RAG instance
        rag = get_rag_instance()
        
        if not rag.vector_store:
            return jsonify({
                "error": "Vector store not available. Use /rebuild endpoint to create one."
            }), 503
        
        # Perform similarity search with enhanced diversity options
        logger.info(f"🔍 DEBUG - Executing {search_type} search...")
        if search_type == 'mmr':
            logger.info(f"🔍 DEBUG - MMR parameters: k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
            docs = rag.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
            logger.info(f"🔍 DEBUG - MMR search completed, returned {len(docs)} docs")
        else:
            logger.info(f"🔍 DEBUG - Similarity parameters: k={k}")
            docs = rag.vector_store.similarity_search(query, k=k)
            logger.info(f"🔍 DEBUG - Similarity search completed, returned {len(docs)} docs")
        
        # Debug logging: Show what files were retrieved
        retrieved_files = [doc.metadata.get("file_name", "Unknown") for doc in docs]
        logger.info(f"📂 DEBUG - Retrieved {len(docs)} documents for query: '{query}'")
        logger.info(f"📂 DEBUG - Files fetched: {retrieved_files}")
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": {
                    "source": doc.metadata.get("source", "Unknown"),
                    "file_name": doc.metadata.get("file_name", "Unknown"),
                    "file_type": doc.metadata.get("file_type", "Unknown"),
                    "file_size": doc.metadata.get("file_size", 0)
                }
            })
        
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in similarity_search endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/rebuild', methods=['POST'])
def rebuild_vector_store():
    """Rebuild the vector store from documents"""
    try:
        global rag_instance
        
        # Get optional data folder path
        data = request.get_json() if request.is_json else {}
        data_folder = data.get('data_folder') if data else None
        
        # Create new RAG instance
        if data_folder:
            rag_instance = RAGPipeline(data_folder=data_folder, config_path="config.json")
        else:
            rag_instance = RAGPipeline(config_path="config.json")
        
        # Load documents
        logger.info("Loading documents...")
        documents = rag_instance.load_documents()
        
        if not documents:
            return jsonify({
                "error": "No documents found to process",
                "documents_processed": 0
            }), 400
        
        # Create embeddings and store
        logger.info("Creating embeddings...")
        success = rag_instance.embed_and_store(documents)
        
        if not success:
            return jsonify({
                "error": "Failed to create embeddings",
                "documents_processed": len(documents)
            }), 500
        
        # Setup QA chain
        if not rag_instance.setup_qa_chain():
            return jsonify({
                "error": "Vector store created but failed to setup QA chain",
                "documents_processed": len(documents)
            }), 500
        
        return jsonify({
            "message": "Vector store rebuilt successfully",
            "documents_processed": len(documents),
            "vector_store_available": True,
            "qa_chain_available": True
        }), 200
        
    except Exception as e:
        logger.error(f"Error in rebuild endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reload', methods=['POST'])
def reload_vector_store():
    """Reload existing vector store"""
    try:
        global rag_instance
        
        # Create new RAG instance and try to load existing store
        rag_instance = RAGPipeline(config_path="config.json")
        
        if not rag_instance.load_vector_store():
            return jsonify({
                "error": "No existing vector store found. Use /rebuild endpoint to create one."
            }), 404
        
        # Setup QA chain
        if not rag_instance.setup_qa_chain():
            return jsonify({
                "error": "Vector store loaded but failed to setup QA chain"
            }), 500
        
        return jsonify({
            "message": "Vector store reloaded successfully",
            "vector_store_available": True,
            "qa_chain_available": True
        }), 200
        
    except Exception as e:
        logger.error(f"Error in reload endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get detailed status of the RAG system"""
    try:
        rag = get_rag_instance()
        
        # Check if vector store files exist
        vector_store_exists = os.path.exists(f"{rag.vector_store_path}/index.faiss")
        cache_exists = os.path.exists(rag.cache_path)
        
        status = {
            "vector_store": {
                "loaded": rag.vector_store is not None,
                "files_exist": vector_store_exists,
                "path": rag.vector_store_path
            },
            "embeddings": {
                "loaded": rag.embeddings is not None,
                "type": "OpenAI" if os.getenv('OPENAI_API_KEY') else "HuggingFace"
            },
            "llm": {
                "available": rag.llm is not None,
                "type": "Anthropic" if os.getenv('ANTHROPIC_API_KEY') else "OpenAI" if os.getenv('OPENAI_API_KEY') else "None"
            },
            "qa_chain": {
                "ready": rag.qa_chain is not None
            },
            "cache": {
                "exists": cache_exists,
                "path": rag.cache_path
            },
            "data_folder": {
                "path": str(rag.data_folder),
                "exists": rag.data_folder.exists()
            }
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting RAG API server...")
    
    # Get configuration from environment or config file
    api_config = config.get('api', {})
    host = os.getenv('RAG_API_HOST', api_config.get('host', 'localhost'))
    port = int(os.getenv('RAG_API_PORT', api_config.get('port', 50001)))
    debug = os.getenv('RAG_API_DEBUG', str(api_config.get('debug', False))).lower() == 'true'
    
    logger.info(f"Server will run on http://{host}:{port}")
    logger.info("Available endpoints:")
    logger.info("  GET  /health          - Health check")
    logger.info("  GET  /status          - Detailed system status") 
    logger.info("  POST /query           - Query RAG system")
    logger.info("  POST /similarity_search - Search without LLM")
    logger.info("  POST /rebuild         - Rebuild vector store")
    logger.info("  POST /reload          - Reload existing vector store")
    
    app.run(host=host, port=port, debug=debug)