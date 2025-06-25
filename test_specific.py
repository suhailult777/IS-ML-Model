#!/usr/bin/env python3
"""
Test the specific question mentioned by user
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from conversational_ai import ConversationalAI

def test_specific_question():
    """Test the specific question that was showing issues"""
    print("🧪 Testing Specific Question")
    print("=" * 60)
    
    # Initialize the system
    print("🚀 Initializing AI system...")
    ai = ConversationalAI()
    
    # Test the exact question from user
    question = "what is the test most commonly used to diagnose AKI"
    
    print(f"\n🙋 Question: {question}")
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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_specific_question()
