#!/usr/bin/env python3
"""
Test Enhanced RAG with Output Parsers
Demonstrates the structured output formatting capabilities
"""

import os
import sys
import json
import logging

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from src.rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_rag():
    """Test the enhanced RAG system with sample medical data"""
    
    print("🚀 Testing Enhanced RAG with Output Parsers")
    print("="*60)
    
    # Initialize the RAG system
    rag = EnhancedRAGWithParsers()
    
    # Sample medical documents (based on your AKI data)
    sample_docs = [
        {
            'content': '''Treatment for AKI usually means a hospital stay. During this time, doctors will work to treat the cause of your AKI. In more serious cases, dialysis may be needed to take over kidney function until your kidneys recover. The main goal is to treat the underlying cause and support kidney function during recovery.''',
            'metadata': {'source': 'medical_guide.pdf', 'page': 1, 'section': 'treatment'}
        },
        {
            'content': '''Signs and symptoms of AKI include decreased urine output, swelling in your legs and ankles, fatigue, confusion, nausea, vomiting, and shortness of breath. Some people may not have any symptoms, especially in the early stages. The symptoms can develop rapidly over hours to days.''',
            'metadata': {'source': 'medical_guide.pdf', 'page': 2, 'section': 'symptoms'}
        },
        {
            'content': '''Tests most commonly used to diagnose AKI include blood tests to measure creatinine and urea levels, urine tests to check for protein and blood, measuring urine output over 24 hours, and imaging studies like ultrasound to examine kidney structure. Additional tests may be needed based on the suspected cause.''',
            'metadata': {'source': 'medical_guide.pdf', 'page': 3, 'section': 'diagnosis'}
        },
        {
            'content': '''AKI can be caused by decreased blood flow to the kidneys, direct damage to kidney tissue, or blockage of the urinary tract. Common causes include dehydration, blood loss, certain medications, infections, and kidney stones. Risk factors include advanced age, diabetes, and heart disease.''',
            'metadata': {'source': 'medical_guide.pdf', 'page': 4, 'section': 'causes'}
        },
        {
            'content': '''BLOCKAGE OF THE URINARY TRACT can cause AKI when urine cannot flow properly. This can be due to kidney stones, enlarged prostate in men, bladder cancer, prostate cancer, cervical cancer, or blood clots in the urinary tract. Surgery may be needed to remove the blockage.''',
            'metadata': {'source': 'medical_guide.pdf', 'page': 5, 'section': 'blockage'}
        }
    ]
    
    # Build the index
    print("\n📚 Building RAG index...")
    rag.build_index(sample_docs)
    print("✅ Index built successfully!")
    
    # Test questions with different types
    test_questions = [
        {
            'question': 'What are the treatment options for AKI?',
            'category': 'Treatment'
        },
        {
            'question': 'What are the symptoms of acute kidney injury?',
            'category': 'Symptoms'
        },
        {
            'question': 'How is AKI diagnosed?',
            'category': 'Diagnosis'
        },
        {
            'question': 'What causes acute kidney injury?',
            'category': 'Causes'
        },
        {
            'question': 'What can block the urinary tract?',
            'category': 'Specific Condition'
        },
        {
            'question': 'Tell me about kidney problems',
            'category': 'General'
        }
    ]
    
    print("\n🔬 Testing Structured Output Generation")
    print("="*60)
    
    for i, test_case in enumerate(test_questions, 1):
        question = test_case['question']
        category = test_case['category']
        
        print(f"\n📋 Test {i}: {category}")
        print(f"❓ Question: {question}")
        print("-" * 40)
        
        # Get structured response
        response = rag.ask(question)
        
        # Display the formatted answer
        print(f"🤖 Structured Response:\n{response['answer']}")
        
        # Show metadata
        confidence = response.get('confidence', 0.0)
        sources = response.get('sources', 0)
        print(f"\n📊 Confidence: {confidence:.3f} | Sources: {sources}")
        
        # Show source documents used
        if response.get('retrieved_docs'):
            print(f"📚 Sources used:")
            for j, (content, score, metadata) in enumerate(response['retrieved_docs'], 1):
                section = metadata.get('section', 'unknown')
                print(f"   {j}. Section: {section} (relevance: {score:.3f})")
        
        print("\n" + "="*60)
    
    print("\n🧪 Testing System Prompt Approach")
    print("="*60)
    
    # Test the system prompt approach
    test_question = "What are the benefits of early treatment for AKI?"
    print(f"❓ Question: {test_question}")
    print("-" * 40)
    
    response = rag.ask_with_system_prompt(test_question)
    print(f"🤖 Response:\n{response['answer']}")
    
    if 'full_prompt' in response:
        print(f"\n🔍 System Prompt Preview:")
        prompt_preview = response['full_prompt'][:500] + "..." if len(response['full_prompt']) > 500 else response['full_prompt']
        print(prompt_preview)
    
    print("\n✅ Testing completed successfully!")
    
    # Save the index for future use
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    index_path = os.path.join(data_dir, 'test_rag_index')
    rag.save_index(index_path)
    print(f"💾 Index saved to: {index_path}")
    
    return rag

def demonstrate_output_parser_features():
    """Demonstrate the key features of the output parser"""
    
    print("\n🎯 Output Parser Features Demonstration")
    print("="*60)
    
    from src.rag.enhanced_rag_with_parsers import MedicalResponse
    
    # Example 1: Treatment response
    treatment_response = MedicalResponse(
        title="Treatment Options for AKI",
        bullet_points=[
            "**Hospital Care**: Treatment typically requires hospitalization for monitoring and management.",
            "**Dialysis**: May be necessary in severe cases to support kidney function.",
            "**Address Root Cause**: Primary focus on treating the underlying condition causing the problem."
        ],
        confidence_level="High"
    )
    
    print("📋 Example 1 - Treatment Response:")
    print(treatment_response.to_formatted_string())
    
    # Example 2: Symptoms response
    symptoms_response = MedicalResponse(
        title="Signs and Symptoms",
        bullet_points=[
            "**Decreased Urination**: Reduced urine output or frequency.",
            "**Swelling**: Fluid retention causing swelling in legs, ankles, or face.",
            "**Fatigue**: Persistent tiredness and lack of energy.",
            "**Nausea**: Feeling sick to the stomach, possibly with vomiting."
        ],
        confidence_level="High"
    )
    
    print("\n📋 Example 2 - Symptoms Response:")
    print(symptoms_response.to_formatted_string())
    
    # Example 3: JSON representation
    print("\n📋 Example 3 - JSON Structure:")
    print(json.dumps(treatment_response.dict(), indent=2))

if __name__ == "__main__":
    try:
        # Test the enhanced RAG system
        rag_system = test_enhanced_rag()
        
        # Demonstrate output parser features
        demonstrate_output_parser_features()
        
        print("\n🎉 All tests completed successfully!")
        print("💡 You can now use the enhanced_chat_with_parsers.py for interactive testing")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()
