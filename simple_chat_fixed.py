#!/usr/bin/env python3
"""
Simple PDF-based Conversational AI
A lightweight chat interface for interacting with PDF content
"""

import os
import json
import logging
from typing import Dict, List, Any
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePDFChat:
    """Simple chat interface for PDF content"""
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
        self.pdf_content = ""
        self.sections = {}
        self.knowledge_base = {}
        self.qa_pairs = []
        
        self._load_pdf_data()
    
    def _load_pdf_data(self):
        """Load PDF content and processed data"""
        try:
            # Load PDF content
            pdf_content_path = os.path.join(self.data_dir, 'pdf_content.txt')
            if os.path.exists(pdf_content_path):
                with open(pdf_content_path, 'r', encoding='utf-8') as f:
                    self.pdf_content = f.read()
                logger.info(f"Loaded PDF content: {len(self.pdf_content)} characters")
            
            # Load sections
            sections_path = os.path.join(self.data_dir, 'sections.json')
            if os.path.exists(sections_path):
                with open(sections_path, 'r', encoding='utf-8') as f:
                    self.sections = json.load(f)
                logger.info(f"Loaded {len(self.sections)} sections")
            
            # Load knowledge base
            kb_path = os.path.join(self.data_dir, 'knowledge_base.json')
            if os.path.exists(kb_path):
                with open(kb_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} entries")
            
            # Load QA pairs
            qa_path = os.path.join(self.data_dir, 'qa_pairs.json')
            if os.path.exists(qa_path):
                with open(qa_path, 'r', encoding='utf-8') as f:
                    self.qa_pairs = json.load(f)
                logger.info(f"Loaded {len(self.qa_pairs)} QA pairs")
            
        except Exception as e:
            logger.error(f"Error loading PDF data: {e}")
    
    def search_content(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Simple text-based search in PDF content"""
        query_lower = query.lower()
        results = []
        
        # Search in sections (handle both list and dict formats)
        if isinstance(self.sections, list):
            # If sections is a list of objects
            for section_data in self.sections:
                if isinstance(section_data, dict):
                    content = section_data.get('content', '').lower()
                    title = section_data.get('title', '').lower()
                    
                    if query_lower in content or query_lower in title:
                        results.append({
                            'type': 'section',
                            'title': section_data.get('title', ''),
                            'content': section_data.get('content', '')[:500] + "..." if len(section_data.get('content', '')) > 500 else section_data.get('content', ''),
                            'relevance': content.count(query_lower) + title.count(query_lower) * 2
                        })
        elif isinstance(self.sections, dict):
            # If sections is a dictionary
            for section_id, section_data in self.sections.items():
                if isinstance(section_data, dict):
                    content = section_data.get('content', '').lower()
                    title = section_data.get('title', '').lower()
                    
                    if query_lower in content or query_lower in title:
                        results.append({
                            'type': 'section',
                            'title': section_data.get('title', ''),
                            'content': section_data.get('content', '')[:500] + "..." if len(section_data.get('content', '')) > 500 else section_data.get('content', ''),
                            'relevance': content.count(query_lower) + title.count(query_lower) * 2
                        })
        
        # Search in knowledge base
        if isinstance(self.knowledge_base, dict):
            for key, value in self.knowledge_base.items():
                if query_lower in key.lower() or query_lower in str(value).lower():
                    results.append({
                        'type': 'knowledge',
                        'title': key,
                        'content': str(value)[:300] + "..." if len(str(value)) > 300 else str(value),
                        'relevance': key.lower().count(query_lower) + str(value).lower().count(query_lower)
                    })
        
        # Search in QA pairs
        if isinstance(self.qa_pairs, list):
            for qa in self.qa_pairs:
                if isinstance(qa, dict):
                    question = qa.get('question', '').lower()
                    answer = qa.get('answer', '').lower()
                    
                    if query_lower in question or query_lower in answer:
                        results.append({
                            'type': 'qa',
                            'title': qa.get('question', ''),
                            'content': qa.get('answer', ''),
                            'relevance': question.count(query_lower) * 3 + answer.count(query_lower)
                        })
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:max_results]
    
    def generate_response(self, user_input: str) -> str:
        """Generate response based on user input"""
        user_input = user_input.strip()
        
        if not user_input:
            return "Please ask me something about the PDF content."
        
        # Handle greeting
        if any(greeting in user_input.lower() for greeting in ['hello', 'hi', 'hey']):
            return ("Hello! I'm here to help you with questions about the PDF content. "
                   "You can ask me about medical topics, treatments, or any specific information "
                   "from the document. What would you like to know?")
        
        # Handle help requests
        if any(help_word in user_input.lower() for help_word in ['help', 'what can you do']):
            return self._get_help_message()
        
        # Search for relevant content
        search_results = self.search_content(user_input)
        
        if not search_results:
            return ("I couldn't find specific information about that in the PDF. "
                   "Could you try rephrasing your question or asking about a different topic? "
                   "You can ask about medical conditions, treatments, or procedures mentioned in the document.")
        
        # Generate response from search results
        response_parts = []
        response_parts.append("Based on the PDF content, here's what I found:\n")
        
        for i, result in enumerate(search_results, 1):
            if result['type'] == 'qa':
                response_parts.append(f"{i}. Q: {result['title']}")
                response_parts.append(f"   A: {result['content']}\n")
            elif result['type'] == 'section':
                response_parts.append(f"{i}. From section '{result['title']}':")
                response_parts.append(f"   {result['content']}\n")
            elif result['type'] == 'knowledge':
                response_parts.append(f"{i}. {result['title']}: {result['content']}\n")
        
        response_parts.append("\nWould you like me to elaborate on any of these points?")
        
        return "\n".join(response_parts)
    
    def _get_help_message(self) -> str:
        """Return help message"""
        topics = []
        
        # Get sample topics from sections
        if isinstance(self.sections, list):
            section_titles = [data.get('title', '') for data in self.sections[:5] if isinstance(data, dict)]
            topics.extend(section_titles)
        elif isinstance(self.sections, dict):
            section_titles = [data.get('title', '') for data in list(self.sections.values())[:5] if isinstance(data, dict)]
            topics.extend(section_titles)
        
        # Get sample topics from knowledge base
        if isinstance(self.knowledge_base, dict):
            kb_keys = list(self.knowledge_base.keys())[:3]
            topics.extend(kb_keys)
        
        help_msg = [
            "I can help you with questions about the PDF content. Here are some things you can ask:",
            "",
            "• Ask about specific medical topics or conditions",
            "• Request information about treatments or procedures", 
            "• Search for specific terms or concepts",
            "• Ask for explanations of medical terminology",
            ""
        ]
        
        if topics:
            help_msg.append("Some topics available in the document:")
            for topic in topics[:5]:
                if topic and topic.strip():
                    help_msg.append(f"• {topic}")
            help_msg.append("")
        
        help_msg.append("Just type your question and I'll search the PDF for relevant information!")
        
        return "\n".join(help_msg)
    
    def chat_loop(self):
        """Main chat loop"""
        print("=" * 60)
        print("PDF CONVERSATIONAL AI")
        print("=" * 60)
        print("Chat with your PDF content! Type 'quit' or 'exit' to stop.")
        print("Type 'help' to see what you can ask about.")
        print("-" * 60)
        
        if self.pdf_content:
            print(f"📄 Loaded PDF content ({len(self.pdf_content)} characters)")
        if self.sections:
            print(f"📑 Found {len(self.sections)} sections")
        if self.qa_pairs:
            print(f"❓ Loaded {len(self.qa_pairs)} Q&A pairs")
        
        print("\nReady to chat! Ask me anything about the PDF content.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nAI: Goodbye! Thanks for chatting about the PDF content.")
                    break
                
                if not user_input:
                    continue
                
                response = self.generate_response(user_input)
                print(f"\nAI: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nAI: Goodbye! Thanks for chatting.")
                break
            except Exception as e:
                print(f"\nAI: Sorry, I encountered an error: {e}")
                print("Please try again.\n")

def main():
    """Main function"""
    try:
        chat = SimplePDFChat()
        chat.chat_loop()
    except Exception as e:
        logger.error(f"Error starting chat: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
