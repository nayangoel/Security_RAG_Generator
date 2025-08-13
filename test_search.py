#!/usr/bin/env python3
"""
Test script to debug search issues
"""

import json
import requests

def test_similarity_search():
    """Test the similarity search endpoint with different parameters"""
    
    url = "http://localhost:50001/similarity_search"
    
    # Test 1: Basic request (should use config defaults)
    print("=== Test 1: Basic Request ===")
    basic_request = {
        "query": "GraphQL security vulnerabilities SQL Injection Authorization Bypass DoS"
    }
    
    print(f"Request: {json.dumps(basic_request, indent=2)}")
    try:
        response = requests.post(url, json=basic_request)
        result = response.json()
        print(f"Status: {response.status_code}")
        print(f"Results count: {result.get('count', 0)}")
        if 'results' in result:
            files = [r.get('metadata', {}).get('file_name', 'Unknown') for r in result['results']]
            print(f"Files: {files}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Explicit MMR request
    print("=== Test 2: Explicit MMR Request ===")
    mmr_request = {
        "query": "GraphQL security vulnerabilities SQL Injection Authorization Bypass DoS",
        "k": 15,
        "search_type": "mmr",
        "fetch_k": 45,
        "lambda_mult": 0.5
    }
    
    print(f"Request: {json.dumps(mmr_request, indent=2)}")
    try:
        response = requests.post(url, json=mmr_request)
        result = response.json()
        print(f"Status: {response.status_code}")
        print(f"Results count: {result.get('count', 0)}")
        if 'results' in result:
            files = [r.get('metadata', {}).get('file_name', 'Unknown') for r in result['results']]
            print(f"Files: {files}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: High k similarity search
    print("=== Test 3: High K Similarity Search ===")
    high_k_request = {
        "query": "GraphQL security vulnerabilities SQL Injection Authorization Bypass DoS",
        "k": 20,
        "search_type": "similarity"
    }
    
    print(f"Request: {json.dumps(high_k_request, indent=2)}")
    try:
        response = requests.post(url, json=high_k_request)
        result = response.json()
        print(f"Status: {response.status_code}")
        print(f"Results count: {result.get('count', 0)}")
        if 'results' in result:
            files = [r.get('metadata', {}).get('file_name', 'Unknown') for r in result['results']]
            print(f"Files: {files}")
    except Exception as e:
        print(f"Error: {e}")

def test_direct_faiss():
    """Test FAISS directly to see if the issue is with the vector store"""
    from rag_pipeline import RAGPipeline
    
    print("=== Testing FAISS Vector Store Directly ===")
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline(config_path="config.json")
        
        # Load vector store
        if not rag.load_vector_store():
            print("Failed to load vector store")
            return
        
        query = "GraphQL security vulnerabilities SQL Injection Authorization Bypass DoS"
        
        # Test basic similarity search
        print(f"\n--- Direct Similarity Search (k=15) ---")
        docs = rag.vector_store.similarity_search(query, k=15)
        print(f"Retrieved {len(docs)} documents")
        files = [doc.metadata.get("file_name", "Unknown") for doc in docs]
        print(f"Files: {files[:10]}...")  # Show first 10
        
        # Test MMR search
        print(f"\n--- Direct MMR Search (k=15, fetch_k=45) ---")
        mmr_docs = rag.vector_store.max_marginal_relevance_search(
            query, k=15, fetch_k=45, lambda_mult=0.5
        )
        print(f"Retrieved {len(mmr_docs)} documents")
        mmr_files = [doc.metadata.get("file_name", "Unknown") for doc in mmr_docs]
        print(f"Files: {mmr_files}")
        
        # Show total documents in vector store
        print(f"\n--- Vector Store Info ---")
        print(f"Vector store type: {type(rag.vector_store)}")
        if hasattr(rag.vector_store, 'index') and hasattr(rag.vector_store.index, 'ntotal'):
            print(f"Total vectors in index: {rag.vector_store.index.ntotal}")
        
    except Exception as e:
        print(f"Error in direct test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing RAG Search Functionality")
    print("=" * 50)
    
    # Test API endpoint
    test_similarity_search()
    
    # Test direct FAISS access
    test_direct_faiss()