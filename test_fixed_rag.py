#!/usr/bin/env python3
"""
Quick test to verify the RAG fixes
"""

import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_text_chunking():
    """Test the improved text chunking"""
    # Read the actual PDF content
    json_content_path = "data/processed/pdf_content.json"
    if os.path.exists(json_content_path):
        with open(json_content_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        full_text = json_data.get('text', '')
        print(f"Original text length: {len(full_text)} characters")
        
        # Import the enhanced system to test chunking
        from enhanced_conversational_ai import EnhancedConversationalAI
        ai = EnhancedConversationalAI()
        
        chunks = ai._split_text_into_chunks(full_text, chunk_size=500)
        
        print(f"\nCreated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({len(chunk)} chars):")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            print("-" * 50)
        
        return chunks
    else:
        print(f"PDF content file not found: {json_content_path}")
        return []

def test_rag_responses():
    """Test the improved RAG responses"""
    from rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers
    
    # Create some test documents with AKI content
    test_docs = [
        {
            'content': 'Acute kidney injury (AKI) is a sudden episode of kidney failure or kidney damage that happens within a few hours or a few days. AKI causes a build-up of waste products in your blood and makes it hard for your kidneys to keep the right balance of fluid in your body.',
            'metadata': {'source': 'aki_definition', 'title': 'AKI Definition'}
        },
        {
            'content': 'Treatment for AKI usually means a hospital stay. In more serious cases, dialysis may be needed to take over kidney function until your kidneys recover. The main goal is to treat the cause of your AKI.',
            'metadata': {'source': 'aki_treatment', 'title': 'AKI Treatment'}
        },
        {
            'content': 'Signs and symptoms of AKI include decreased urine output, swelling in legs and ankles, fatigue, confusion, nausea, and shortness of breath. In some cases, AKI causes no symptoms.',
            'metadata': {'source': 'aki_symptoms', 'title': 'AKI Symptoms'}
        }
    ]
    
    rag = EnhancedRAGWithParsers()
    rag.build_index(test_docs, train_reranker=False)  # Skip training for quick test
    
    test_questions = [
        "what is aki",
        "what is the treatment of aki", 
        "what are the symptoms of aki"
    ]
    
    print("\n" + "="*60)
    print("TESTING IMPROVED RAG RESPONSES")
    print("="*60)
    
    for question in test_questions:
        print(f"\n🙋 Question: {question}")
        response = rag.ask(question)
        print(f"🤖 Response:\n{response['answer']}")
        print("-" * 40)

if __name__ == "__main__":
    print("Testing RAG system fixes...")
    
    print("\n1. Testing text chunking...")
    chunks = test_text_chunking()
    
    print("\n2. Testing RAG responses...")
    test_rag_responses()
    
    print("\n✅ Test completed!")
