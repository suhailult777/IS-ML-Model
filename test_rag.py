#!/usr/bin/env python3
"""
Enhanced test script for the RAG conversational AI with proper medical knowledge
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from conversational_ai import ConversationalAI

def test_rag_system():
    """Test the RAG system with realistic medical questions"""
    print("🧪 Testing Enhanced RAG Conversational AI System")
    print("=" * 60)
    
    # Initialize the system
    print("🚀 Initializing AI system...")
    ai = ConversationalAI()
    
    # Test questions that should work well with our medical knowledge base
    test_questions = [
        "What is acute kidney injury?",
        "What are the symptoms of chronic kidney disease?", 
        "What causes diabetes?",
        "How is hypertension treated?",
        "What are the stages of chronic kidney disease?",
        "What are the complications of diabetes?",
        "How is acute kidney injury diagnosed?",
        "What medications are used for high blood pressure?"
    ]
    
    print(f"\n🔥 RAG System Status: {'ACTIVE' if ai.use_rag else 'INACTIVE'}")
    print("📚 Testing with comprehensive medical knowledge base")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🙋 Test Question {i}: {question}")
        print("-" * 40)
        
        try:
            answer = ai.ask_question(question)
            print(f"🤖 AI Response:\n{answer}")
            
            # Show conversation history entry
            if ai.conversation_history:
                last_entry = ai.conversation_history[-1]
                if 'confidence' in last_entry:
                    print(f"\n📊 Confidence: {last_entry['confidence']:.2f}")
                if 'method' in last_entry:
                    print(f"🔧 Method: {last_entry['method']}")
                if 'sources' in last_entry:
                    print(f"📚 Sources used: {last_entry['sources']}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
        
        # Add a small pause between questions for readability
        if i < len(test_questions):
            input("Press Enter to continue to next question...")
    
    print("\n✅ RAG Testing completed!")
    print(f"📈 Total conversation entries: {len(ai.conversation_history)}")
    
    # Test conversation summary
    print("\n📋 Conversation Summary:")
    print(ai.get_conversation_summary())

if __name__ == "__main__":
    test_rag_system()
