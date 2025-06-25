#!/usr/bin/env python3
"""
Integration Example: Upgrading Simple Chat to Enhanced RAG
Shows how to modify your existing chat interface to use output parsers
"""

import os
import sys
import json
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpgradedChatInterface:
    """
    Upgraded version of your existing chat interface
    Shows before/after comparison
    """
    
    def __init__(self, use_enhanced_rag=True):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
        self.use_enhanced_rag = use_enhanced_rag
        
        if use_enhanced_rag:
            from src.rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers
            self.rag_system = EnhancedRAGWithParsers()
            print("✅ Using Enhanced RAG with Output Parsers")
        else:
            from src.rag.simple_rag import SimpleRAG
            self.rag_system = SimpleRAG()
            print("📝 Using Simple RAG (original)")
        
        self.is_initialized = False
        self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize the RAG system with sample data"""
        try:
            # Sample medical documents
            documents = [
                {
                    'content': '''Treatment for AKI usually means a hospital stay. During this time, doctors will work to treat the cause of your AKI. In more serious cases, dialysis may be needed to take over kidney function until your kidneys recover. The main goal is to treat the underlying cause and support kidney function during recovery.''',
                    'metadata': {'source': 'medical_guide.pdf', 'section': 'treatment'}
                },
                {
                    'content': '''Signs and symptoms of AKI include decreased urine output, swelling in your legs and ankles, fatigue, confusion, nausea, vomiting, and shortness of breath. Some people may not have any symptoms, especially in the early stages. The symptoms can develop rapidly over hours to days.''',
                    'metadata': {'source': 'medical_guide.pdf', 'section': 'symptoms'}
                },
                {
                    'content': '''Tests most commonly used to diagnose AKI include blood tests to measure creatinine and urea levels, urine tests to check for protein and blood, measuring urine output over 24 hours, and imaging studies like ultrasound to examine kidney structure.''',
                    'metadata': {'source': 'medical_guide.pdf', 'section': 'diagnosis'}
                }
            ]
            
            self.rag_system.build_index(documents)
            self.is_initialized = True
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
    
    def ask_question(self, question: str) -> dict:
        """Ask a question and get response"""
        if not self.is_initialized:
            return {
                'answer': "System not initialized",
                'confidence': 0.0,
                'sources': 0
            }
        
        try:
            response = self.rag_system.ask(question)
            return response
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'confidence': 0.0,
                'sources': 0
            }

def compare_responses():
    """Compare responses between simple and enhanced RAG"""
    
    print("\n🔍 Comparison: Simple RAG vs Enhanced RAG with Output Parsers")
    print("="*80)
    
    # Test questions
    questions = [
        "What are the treatment options for AKI?",
        "What are the symptoms of kidney problems?",
        "How is AKI diagnosed?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📋 Test {i}: {question}")
        print("="*80)
        
        # Simple RAG response
        print("\n📝 SIMPLE RAG (Original):")
        print("-" * 40)
        simple_chat = UpgradedChatInterface(use_enhanced_rag=False)
        simple_response = simple_chat.ask_question(question)
        print(f"Answer: {simple_response['answer']}")
        print(f"Confidence: {simple_response.get('confidence', 0):.3f}")
        
        # Enhanced RAG response
        print("\n✨ ENHANCED RAG (With Output Parsers):")
        print("-" * 40)
        enhanced_chat = UpgradedChatInterface(use_enhanced_rag=True)
        enhanced_response = enhanced_chat.ask_question(question)
        print(enhanced_response['answer'])
        print(f"Confidence: {enhanced_response.get('confidence', 0):.3f}")
        
        print("\n" + "="*80)

def show_integration_steps():
    """Show step-by-step integration guide"""
    
    print("\n🛠️  Integration Guide: Upgrading Your RAG System")
    print("="*80)
    
    print("\n📋 Step 1: Update Dependencies")
    print("-" * 40)
    print("""
Add to requirements.txt:
    langchain>=0.1.0
    langchain-core>=0.1.0
    pydantic>=2.0.0
    """)
    
    print("\n📋 Step 2: Update Imports")
    print("-" * 40)
    print("""
# OLD CODE:
from src.rag.simple_rag import SimpleRAG
rag = SimpleRAG()

# NEW CODE:
from src.rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers
rag = EnhancedRAGWithParsers()
    """)
    
    print("\n📋 Step 3: Same Interface, Better Output")
    print("-" * 40)
    print("""
# The interface remains the same:
response = rag.ask("What are the symptoms?")

# But now you get structured output:
print(response['answer'])  # Formatted markdown with H3 + bullets
print(response['confidence'])  # Confidence score
print(response['sources'])  # Number of sources
print(response['formatted'])  # True for enhanced responses
    """)
    
    print("\n📋 Step 4: Update Your Chat Interface")
    print("-" * 40)
    print("""
# In your existing chat.py:
def process_question(self, question):
    response = self.rag_system.ask(question)
    
    # Display the beautifully formatted response
    print(response['answer'])  # Already formatted!
    
    # Show metadata
    print(f"Confidence: {response['confidence']:.2f}")
    print(f"Sources: {response['sources']}")
    """)
    
    print("\n📋 Step 5: Optional Advanced Features")
    print("-" * 40)
    print("""
# Use system prompt for debugging:
response = rag.ask_with_system_prompt(question)
print(response['full_prompt'])  # See the complete prompt

# Custom confidence handling:
if response['confidence'] > 0.6:
    print("High confidence response")
elif response['confidence'] > 0.3:
    print("Moderate confidence response")
else:
    print("Low confidence - may need clarification")
    """)

def show_system_prompt_usage():
    """Show how to use the system prompt for actual LLM integration"""
    
    print("\n🎯 System Prompt Integration (For Real LLMs)")
    print("="*80)
    
    print("""
The enhanced RAG includes a complete system prompt that follows your specification:

[SYSTEM]
You are an AI assistant that ALWAYS returns answers in clean, markdown-formatted sections.
Strict rules:
1. Use exactly one H3 heading (###) for the title.
2. Use bullet points (`- `) for each item.
3. Bold the key term in each bullet with `**`.
4. No intros, no conclusions, no numbering—just the heading and bullets.

[FEW-SHOT EXAMPLES]
... (includes your examples)

[USER INPUT]
Context: {retrieved_docs}
Question: {user_query}
Answer:

To use with OpenAI/Claude/etc:
    """)
    
    print("""
from src.rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers
import openai  # or anthropic, etc.

rag = EnhancedRAGWithParsers()
response = rag.ask_with_system_prompt("What are the symptoms?")

# Send the complete prompt to your LLM
openai_response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": response['full_prompt']}]
)

# You'll get properly formatted responses!
    """)

if __name__ == "__main__":
    try:
        print("🚀 Enhanced RAG Integration Example")
        print("="*80)
        
        # Show the comparison
        compare_responses()
        
        # Show integration steps
        show_integration_steps()
        
        # Show system prompt usage
        show_system_prompt_usage()
        
        print("\n🎉 Integration guide completed!")
        print("\n📚 Summary:")
        print("   ✅ Same interface as your existing RAG")
        print("   ✨ Professional markdown formatting")
        print("   📋 Structured bullet points with bold terms")
        print("   🎯 Ready for OpenAI/Claude integration")
        print("   🛡️  Type safety with Pydantic models")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
