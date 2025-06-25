#!/usr/bin/env python3
"""
Test treatment question
"""

import sys
import os
sys.path.append('src')

from src.rag.simple_rag import SimpleRAG

def test_treatment_question():
    """Test with treatment question"""
    
    # Create RAG instance
    rag = SimpleRAG()
    
    # Load existing index
    try:
        rag.load_index('data/processed/rag_index')
        print("Loaded existing RAG index")
    except:
        print("Could not load existing index")
        return
    
    # Test with treatment question
    question = "list some treatments of aki"
    
    # Get answer
    result = rag.ask(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print()

if __name__ == "__main__":
    test_treatment_question()
