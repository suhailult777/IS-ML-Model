#!/usr/bin/env python3
"""
Quick Demo: Enhanced RAG with Output Parsers
Shows how to integrate with your existing RAG pipeline
"""

import os
import sys
import json

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

def demo_enhanced_rag():
    """Quick demonstration of the enhanced RAG system"""
    
    print("🚀 Enhanced RAG with Output Parsers - Quick Demo")
    print("="*60)
    
    from src.rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers
    
    # Initialize
    rag = EnhancedRAGWithParsers()
    print("✅ Enhanced RAG system initialized")
    
    # Your existing PDF data (simulated)
    medical_docs = [
        {
            'content': '''Treatment for AKI usually means a hospital stay. During this time, doctors will work to treat the cause of your AKI. In more serious cases, dialysis may be needed to take over kidney function until your kidneys recover. The main goal is to treat the underlying cause and support kidney function during recovery.''',
            'metadata': {'source': 'medical_guide.pdf', 'section': 'treatment'}
        },
        {
            'content': '''Signs and symptoms of AKI include decreased urine output, swelling in your legs and ankles, fatigue, confusion, nausea, vomiting, and shortness of breath. Some people may not have any symptoms, especially in the early stages.''',
            'metadata': {'source': 'medical_guide.pdf', 'section': 'symptoms'}
        }
    ]
    
    # Build index (this replaces your current indexing)
    rag.build_index(medical_docs)
    print("✅ Index built from medical documents")
    
    # Test questions
    questions = [
        "What are the treatment options for AKI?",
        "What symptoms should I look for?",
        "Tell me about kidney problems"
    ]
    
    print(f"\n🔬 Testing {len(questions)} questions with structured output:")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📋 Question {i}: {question}")
        print("-" * 40)
        
        # This is your new RAG call - same interface, better output!
        response = rag.ask(question)
        
        # Display the beautifully formatted response
        print("🤖 AI Response:")
        print(response['answer'])
        print(f"\n📊 Confidence: {response['confidence']:.3f} | Sources: {response['sources']}")
    
    print("\n" + "="*60)
    print("🎯 Key Benefits of Enhanced RAG:")
    print("   ✨ Professional markdown formatting")
    print("   📋 Structured bullet points with bold terms")
    print("   🎯 Consistent H3 headings")
    print("   📊 Confidence scoring")
    print("   🔍 Source tracking")
    print("   🛡️  Type safety with Pydantic")
    
    print("\n💡 Integration Steps:")
    print("   1. Replace: from src.rag.simple_rag import SimpleRAG")
    print("   2. With: from src.rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers")
    print("   3. Same .ask() interface, better formatted output!")

def show_output_parser_structure():
    """Show the structure of the output parser"""
    
    print("\n🏗️  Output Parser Structure")
    print("="*60)
    
    from src.rag.enhanced_rag_with_parsers import MedicalResponse
    
    # Show the Pydantic model structure
    print("📋 Pydantic Model Schema:")
    print("""
class MedicalResponse(BaseModel):
    title: str = Field(description="H3 heading for the topic")
    bullet_points: List[str] = Field(description="Bullet points with bold terms")
    confidence_level: Optional[str] = Field(description="High/Moderate/Low")
    
    def to_formatted_string(self) -> str:
        # Converts to markdown format
    """)
    
    # Example structured data
    example = MedicalResponse(
        title="Treatment Options",
        bullet_points=[
            "**Hospital Care**: Requires hospitalization for monitoring.",
            "**Dialysis**: May be needed in severe cases.",
            "**Address Cause**: Focus on underlying condition."
        ],
        confidence_level="High"
    )
    
    print("\n📄 Example Output:")
    print(example.to_formatted_string())
    
    print("\n🔧 How It Works:")
    print("   1. Document retrieval (same as before)")
    print("   2. Question categorization (treatment/symptoms/diagnosis)")
    print("   3. Structured information extraction")
    print("   4. Pydantic model validation")
    print("   5. Markdown formatting")

if __name__ == "__main__":
    try:
        demo_enhanced_rag()
        show_output_parser_structure()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📚 Next Steps:")
        print("   • Check ENHANCED_RAG_README.md for full documentation")
        print("   • Run test_enhanced_rag.py for comprehensive testing")
        print("   • Try enhanced_chat_with_parsers.py for interactive chat")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
