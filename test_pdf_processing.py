#!/usr/bin/env python3
"""
Test script to let the AI system automatically process the PDF and answer questions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from conversational_ai import ConversationalAI

def test_pdf_processing():
    """Test the AI system with the acute kidney injury PDF"""
    print("🏥 Testing AI System with Acute Kidney Injury PDF")
    print("=" * 60)
    print("🤖 Let the AI system automatically process the PDF...")
    print()
    
    # Initialize the system - it will automatically process the PDF
    ai = ConversationalAI()
    
    print(f"✅ System initialized!")
    print(f"📊 RAG System: {'ACTIVE' if ai.use_rag else 'INACTIVE'}")
    print(f"📑 Document sections processed: {len(getattr(ai, 'document_sections', []))}")
    print(f"💾 Q&A pairs available: {len(getattr(ai, 'qa_pairs', []))}")
    print()
    
    # Test questions about AKI from the PDF
    test_questions = [
        "What is acute kidney injury?",
        "What are the main causes of AKI?", 
        "What are the symptoms of acute kidney injury?",
        "How is AKI diagnosed?",
        "What are the treatment options for acute kidney injury?",
        "What are the stages of AKI?",
        "How can AKI be prevented?"
    ]
    
    print("🧪 Testing questions based on the PDF content:")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🙋 Question {i}: {question}")
        print("-" * 40)
        
        try:
            answer = ai.ask_question(question)
            print(f"🤖 AI: {answer}")
            
            # Show conversation metadata if available
            if ai.conversation_history and len(ai.conversation_history) >= i:
                last_entry = ai.conversation_history[-1]
                if isinstance(last_entry, dict):
                    confidence = last_entry.get('confidence', 'N/A')
                    method = last_entry.get('method', 'N/A')
                    print(f"📊 Confidence: {confidence} | Method: {method}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
    
    print("\n🎉 Testing completed!")
    print(f"📈 Total conversation exchanges: {len(ai.conversation_history)}")

if __name__ == "__main__":
    test_pdf_processing()
