#!/usr/bin/env python3
"""
Test with the actual user question
"""

import sys
import os
sys.path.append('src')

from src.rag.simple_rag import SimpleRAG

def test_user_question():
    """Test with the user's specific question"""
    
    # Create RAG instance
    rag = SimpleRAG()
    
    # Load existing index if available
    try:
        rag.load_index('data/processed/rag_index')
        print("Loaded existing RAG index")
    except:
        print("Could not load existing index, creating simple test data...")
        docs = [
            {
                'content': 'take over your kidney function until your kidneys The tests most commonly used to diagnosis recover. have an AKI, there are several tests that can be Most people with AKI are already in the hospital ordered to confirm or rule out a diagnosis. Measuring urine output, which tracks how much urine is passed each day Recovery Urinalysis, which is a urine test used to find After recovering from an AKI, you will be at signs of kidney disease and kidney failure higher risk for developing other health problems Blood tests to check creatinine, urea nitrogen such as kidney disease, stroke, heart disease, and other cardiovascular complications.',
                'metadata': {'source': 'aki_medical.pdf'}
            }
        ]
        rag.build_index(docs)
    
    # Test with user's question
    question = "what is the test most commonly used to diagnose AKI"
    
    # Get answer
    result = rag.ask(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print()

if __name__ == "__main__":
    test_user_question()
