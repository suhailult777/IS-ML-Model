#!/usr/bin/env python3
"""
Debug RAG system
"""

import sys
import os
sys.path.append('src')

try:
    from src.rag.simple_rag import SimpleRAG
    
    # Create RAG instance
    rag = SimpleRAG()
    
    # Load existing index
    rag.load_index('data/processed/rag_index')
    print("Loaded existing RAG index")
    
    # Test retrieval for treatment question
    question = "list some treatments of aki"
    context_docs = rag.retrieve(question, top_k=3)
    
    print(f"\nQuestion: {question}")
    print(f"Found {len(context_docs)} documents")
    
    for i, (doc, score, metadata) in enumerate(context_docs):
        print(f"\nDocument {i+1} (score: {score:.3f}):")
        print(f"Content (first 300 chars): {doc[:300]}...")
        
        # Test content extraction
        extracted = rag._extract_relevant_content(doc, question)
        print(f"Extracted: {extracted[:200]}...")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
