#!/usr/bin/env python3
"""
Test script to verify the enhanced RAG system with medical model training
"""

import os
import sys
import json
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_with_training():
    """Test the enhanced RAG system with medical model training"""
    print("Testing Enhanced RAG system with medical model training...")
    
    # Load the JSON data with AKI information
    json_content_path = "data/processed/pdf_content.json"
    if not os.path.exists(json_content_path):
        print(f"❌ Content file not found: {json_content_path}")
        return
    
    print(f"✅ Loading content from {json_content_path}")
    
    with open(json_content_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Create documents from the sections
    documents = []
    
    # Add the sections
    sections = json_data.get('sections', [])
    for section in sections:
        documents.append({
            'content': section.get('content', ''),
            'metadata': {'source': 'pdf_section', 'title': section.get('title', '')}
        })
    
    # Add the full text in chunks
    full_text = json_data.get('text', '')
    if full_text:
        # Split into meaningful chunks
        text_parts = full_text.split('\n\n')
        chunk_size = 500
        current_chunk = ""
        chunk_id = 0
        
        for part in text_parts:
            part = part.strip()
            if not part:
                continue
                
            if len(current_chunk + part) < chunk_size * 2:
                current_chunk += part + " "
            else:
                if current_chunk:
                    documents.append({
                        'content': current_chunk.strip(),
                        'metadata': {'source': 'pdf_chunk', 'chunk_id': chunk_id, 'title': f'AKI Information Part {chunk_id + 1}'}
                    })
                    chunk_id += 1
                current_chunk = part + " "
        
        if current_chunk:
            documents.append({
                'content': current_chunk.strip(),
                'metadata': {'source': 'pdf_chunk', 'chunk_id': chunk_id, 'title': f'AKI Information Part {chunk_id + 1}'}
            })
    
    print(f"✅ Prepared {len(documents)} documents for training")
    
    # Initialize the RAG system
    print("🚀 Initializing Enhanced RAG system...")
    rag = EnhancedRAGWithParsers()
    
    # Build the index with training enabled
    print("🔧 Building index and training models...")
    rag.build_index(documents, train_reranker=True, train_medical_model=True)
    
    # Test questions
    test_questions = [
        "what is aki",
        "what are the symptoms of acute kidney injury",
        "what causes aki",
        "how is aki treated",
        "how is aki diagnosed"
    ]
    
    print("\n" + "="*60)
    print("TESTING ENHANCED RAG WITH TRAINED MEDICAL MODEL")
    print("="*60)
    
    for question in test_questions:
        print(f"\n🙋 Question: {question}")
        print("🔍 Searching...")
        
        try:
            response = rag.ask(question)
            print(f"🤖 AI Response:\n{response['answer']}")
            
            if 'context' in response and response['context']:
                avg_score = sum(score for _, score, _ in response['context']) / len(response['context'])
                print(f"📊 Average retrieval score: {avg_score:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
    
    print("\n✅ Enhanced RAG with medical model training test completed!")

if __name__ == "__main__":
    test_rag_with_training()
