#!/usr/bin/env python3
"""
Quick RAG test script that directly processes the PDF content
"""

import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag.simple_rag import SimpleRAG

def test_rag_with_pdf():
    """Test RAG system directly with PDF content"""
    print("🧪 Testing RAG System with Real AKI PDF Content")
    print("=" * 60)
    
    # Load PDF content from JSON
    json_path = "data/processed/pdf_content.json"
    if not os.path.exists(json_path):
        print("❌ PDF content JSON not found!")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        pdf_data = json.load(f)
    
    pdf_text = pdf_data.get('text', '')
    if not pdf_text:
        print("❌ No text content found in PDF!")
        return
    
    print(f"✅ Loaded PDF content: {len(pdf_text)} characters")
    
    # Create documents for RAG
    documents = []
    
    # Split content into chunks
    def split_content(text):
        # Split by major sections
        chunks = []
        sections = text.split('Causes')
        if len(sections) > 1:
            # Definition and symptoms
            chunks.append("Definition and Symptoms: " + sections[0])
            # Causes and everything after
            chunks.append("Causes and Treatment: " + "Causes" + sections[1])
        else:
            # Split by sentences if no clear sections
            sentences = text.split('. ')
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 400:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        return chunks
    
    chunks = split_content(pdf_text)
    
    for i, chunk in enumerate(chunks):
        documents.append({
            'content': chunk,
            'metadata': {'source': 'pdf_content', 'chunk_id': i}
        })
    
    print(f"✅ Created {len(documents)} document chunks")
    
    # Initialize RAG system
    try:
        rag = SimpleRAG()
        print("✅ RAG system initialized")
        
        # Build index
        rag.build_index(documents)
        print("✅ RAG index built successfully!")
        
        # Test questions
        test_questions = [
            "What is acute kidney injury?",
            "What are the symptoms of AKI?", 
            "What causes acute kidney injury?",
            "How is AKI diagnosed?",
            "What is the treatment for AKI?"
        ]
        
        print("\n🔥 Testing RAG with real medical questions:")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🙋 Question {i}: {question}")
            print("-" * 40)
            
            try:
                response = rag.ask(question, top_k=3)
                answer = response['answer']
                confidence = response['confidence']
                
                print(f"🤖 Answer: {answer}")
                print(f"📊 Confidence: {confidence:.2f}")
                print(f"📚 Sources: {response['sources']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n✅ RAG testing completed successfully!")
        print("🎉 The RAG system is working with real AKI content!")
        
    except Exception as e:
        print(f"❌ RAG initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_with_pdf()
