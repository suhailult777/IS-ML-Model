#!/usr/bin/env python3
"""
Enhanced Chat Interface with Output Parsers
Uses structured output formatting for better GPT-like responses
"""

import os
import json
import logging
from typing import Dict, List, Any
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPDFChatWithParsers:
    """Enhanced chat interface with structured output parsers"""
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
        self.rag_system = EnhancedRAGWithParsers()
        self.is_initialized = False
        
        self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize the RAG system with available data"""
        try:
            # Try to load existing index
            index_path = os.path.join(self.data_dir, 'rag_index')
            if os.path.exists(f"{index_path}_enhanced_rag.faiss"):
                self.rag_system.load_index(index_path)
                self.is_initialized = True
                logger.info("Loaded existing enhanced RAG index")
                return
            
            # If no index exists, build from PDF content
            documents = self._load_documents()
            if documents:
                self.rag_system.build_index(documents)
                self.rag_system.save_index(index_path)
                self.is_initialized = True
                logger.info("Built and saved new enhanced RAG index")
            else:
                logger.warning("No documents found to build RAG index")
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
    
    def _load_documents(self) -> List[Dict]:
        """Load documents from various sources"""
        documents = []
        
        # Load PDF content
        pdf_content_path = os.path.join(self.data_dir, 'pdf_content.txt')
        if os.path.exists(pdf_content_path):
            with open(pdf_content_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks for better retrieval
            chunks = self._split_into_chunks(content, chunk_size=1000)
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'metadata': {
                        'source': 'pdf_content.txt',
                        'chunk_id': i,
                        'type': 'pdf'
                    }
                })
        
        # Load any JSON documents
        json_path = os.path.join(self.data_dir, 'pdf_content.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                if isinstance(json_data, dict) and 'content' in json_data:
                    chunks = self._split_into_chunks(json_data['content'], chunk_size=1000)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            'content': chunk,
                            'metadata': {
                                'source': 'pdf_content.json',
                                'chunk_id': i,
                                'type': 'json'
                            }
                        })
            except Exception as e:
                logger.warning(f"Could not load JSON data: {e}")
        
        logger.info(f"Loaded {len(documents)} document chunks")
        return documents
    
    def _split_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                last_period = text.rfind('.', start + chunk_size - 100, end)
                if last_period > start:
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def ask_question(self, question: str, use_system_prompt: bool = False) -> Dict:
        """Ask a question and get a structured response"""
        if not self.is_initialized:
            return {
                'answer': "### System Not Ready\n\n- **Error**: RAG system is not properly initialized.",
                'confidence': 0.0,
                'sources': 0,
                'formatted': True
            }
        
        try:
            if use_system_prompt:
                response = self.rag_system.ask_with_system_prompt(question)
            else:
                response = self.rag_system.ask(question)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'answer': f"### Error Processing Question\n\n- **Error**: {str(e)}",
                'confidence': 0.0,
                'sources': 0,
                'formatted': True
            }
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n" + "="*60)
        print("🏥 Enhanced Medical AI Assistant with Output Parsers")
        print("="*60)
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'help' for available commands")
        print("-"*60)
        
        if not self.is_initialized:
            print("⚠️  Warning: System not properly initialized. Responses may be limited.")
        
        while True:
            try:
                question = input("\n🤔 Your Question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Take care of your health!")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if question.lower().startswith('debug'):
                    # Debug mode - show system prompt
                    actual_question = question[5:].strip()
                    if actual_question:
                        response = self.ask_question(actual_question, use_system_prompt=True)
                        print(f"\n🤖 AI Response:\n{response['answer']}")
                        print(f"\n📊 Confidence: {response['confidence']:.2f} | Sources: {response['sources']}")
                        if 'full_prompt' in response:
                            print(f"\n🔍 Debug - Full Prompt:\n{response['full_prompt'][:500]}...")
                    continue
                
                # Process the question
                print("\n🔄 Processing your question...")
                response = self.ask_question(question)
                
                # Display the formatted answer
                print(f"\n🤖 AI Response:\n{response['answer']}")
                
                # Show metadata
                confidence = response.get('confidence', 0.0)
                sources = response.get('sources', 0)
                print(f"\n📊 Confidence: {confidence:.2f} | Sources: {sources}")
                
                # Show source information if available
                if response.get('retrieved_docs'):
                    print(f"\n📚 Source Information:")
                    for i, (content, score, metadata) in enumerate(response['retrieved_docs'][:2], 1):
                        source = metadata.get('source', 'Unknown')
                        print(f"   {i}. {source} (relevance: {score:.2f})")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
📋 Available Commands:
   
🔍 Query Types:
   • Treatment questions: "What are the treatment options for AKI?"
   • Symptoms: "What are the symptoms of kidney problems?"
   • Diagnosis: "How is AKI diagnosed?"
   • Causes: "What causes acute kidney injury?"
   
💡 Special Commands:
   • help - Show this help message
   • debug [question] - Show detailed processing information
   • quit/exit/bye - End the conversation

✨ Features:
   • Structured markdown responses
   • Confidence scoring
   • Source citations
   • Medical knowledge from PDF documents
        """
        print(help_text)

def main():
    """Main function to run the enhanced chat interface"""
    try:
        chat = EnhancedPDFChatWithParsers()
        chat.interactive_chat()
    except Exception as e:
        print(f"Failed to start chat interface: {e}")
        logger.error(f"Startup error: {e}")

if __name__ == "__main__":
    main()
