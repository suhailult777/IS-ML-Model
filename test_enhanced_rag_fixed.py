#!/usr/bin/env python3
"""
Test script for the enhanced RAG system after fixes
"""

import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers

def test_enhanced_rag():
    """Test the enhanced RAG system"""
    print("Testing Enhanced RAG system after fixes...")
    
    # Initialize the system
    rag = EnhancedRAGWithParsers()
    
    # Load the JSON data with AKI information
    json_content_path = "data/processed/pdf_content.json"
    if os.path.exists(json_content_path):
        print(f"Loading content from {json_content_path}")
        
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
            # Split into chunks for better retrieval
            words = full_text.split()
            chunk_size = 500
            for i in range(0, len(words), chunk_size // 2):  # 50% overlap
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) > 50:
                    chunk_text = ' '.join(chunk_words)
                    documents.append({
                        'content': chunk_text,
                        'metadata': {'source': 'pdf_chunk', 'chunk_id': i // (chunk_size // 2), 'title': f'AKI Information Part {i // (chunk_size // 2) + 1}'}
                    })
        
        print(f"Built index with {len(documents)} documents")
        
        # Build the index
        rag.build_index(documents, train_reranker=False)  # Skip reranker training for quick test
        
        # Test questions
        test_questions = [
            "what is aki",
            "what are the symptoms of acute kidney injury",
            "what causes aki",
            "how is aki treated"
        ]
        
        print("\n" + "="*60)
        print("TESTING ENHANCED RAG RESPONSES")
        print("="*60)
        
        for question in test_questions:
            print(f"\n🙋 Question: {question}")
            print("🔍 Searching...")
            
            try:
                response = rag.ask(question)
                print(f"🤖 AI Response:\n{response['answer']}")
                print(f"📊 Confidence: {response.get('confidence', 'Unknown')}")
                print(f"📚 Sources used: {response.get('sources', 0)}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 40)
        
        print("\n✅ Enhanced RAG test completed successfully!")
        
    else:
        print(f"❌ Content file not found: {json_content_path}")

if __name__ == "__main__":
    test_enhanced_rag()
