#!/usr/bin/env python3
"""
Simple test for the cleaned up RAG system
"""

import sys
import os
sys.path.append('src')

from src.rag.simple_rag import SimpleRAG

def test_simple_rag():
    """Test the simplified RAG system"""
    
    # Create RAG instance
    rag = SimpleRAG()
    
    # Create sample documents
    docs = [
        {
            'content': 'The tests most commonly used to diagnose acute kidney injury (AKI) include blood tests to check creatinine and urea nitrogen levels, urinalysis to find signs of kidney disease and kidney failure, and measuring urine output which tracks how much urine is passed each day. These diagnostic tests help doctors confirm or rule out an AKI diagnosis.',
            'metadata': {'source': 'aki_diagnostics.pdf'}
        },
        {
            'content': 'After recovering from an AKI, patients will be at higher risk for developing other health problems such as kidney disease, stroke, heart disease, and other cardiovascular complications. Most people with AKI are already in the hospital.',
            'metadata': {'source': 'aki_recovery.pdf'}
        }
    ]
    
    # Build index
    rag.build_index(docs)
    
    # Test question
    question = "what is the test most commonly used to diagnose AKI"
    
    # Get answer
    result = rag.ask(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print()

if __name__ == "__main__":
    test_simple_rag()
