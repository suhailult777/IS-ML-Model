#!/usr/bin/env python3
"""
Enhanced PDF-based Conversational AI System with Output Parsers
Interactive terminal interface for medical document Q&A with structured formatting
"""

import os
import sys
import json
import torch
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from models.document_qa_model import MedicalDocumentQA, create_model
from data.document_understanding import DocumentUnderstandingSystem
from training.pdf_training import MedicalQATrainer, prepare_training_data

# Import Enhanced RAG with fallback handling
try:
    from rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers
    RAG_AVAILABLE = True
    logger.info("Enhanced RAG with Output Parsers loaded successfully!")
except ImportError as e:
    logger.warning(f"Enhanced RAG module not available: {e}")
    try:
        from rag.simple_rag import SimpleRAG
        EnhancedRAGWithParsers = SimpleRAG
        RAG_AVAILABLE = True
        logger.info("Falling back to SimpleRAG")
    except ImportError:
        logger.warning(f"No RAG module available: {e}")
        RAG_AVAILABLE = False
        EnhancedRAGWithParsers = None


class EnhancedConversationalAI:
    """
    Enhanced conversational AI system with structured output formatting
    """
    
    def __init__(self, 
                 model_path: str = "models/medical_qa_model.pt",
                 data_dir: str = "data/processed"):
        
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.model = None
        self.document_system = None
        self.rag_system = None
        self.conversation_history = []
        
        # Configuration
        self.max_context_length = 5
        self.confidence_threshold = 0.3
        self.use_rag = True
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing Enhanced Conversational AI System...")
        
        try:
            # Load or create model
            self._load_or_create_model()
            
            # Initialize document understanding system
            self._initialize_document_system()
            
            # Initialize Enhanced RAG system
            self._initialize_enhanced_rag_system()
            
            # Load training data for context
            self._load_training_data()
            
            logger.info("Enhanced system initialization completed successfully!")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def _load_or_create_model(self):
        """Load existing model or create/train new one"""
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            config = checkpoint.get('config', {})
            model_type = config.pop('model_type', 'medical')

            if 'hidden_size' in config:
                config['hidden_dim'] = config.pop('hidden_size')

            self.model = create_model(model_type, **config)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
        else:
            logger.info("No existing model found. Creating fallback model...")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model for testing"""
        config = {
            'model_type': 'sentence',
            'hidden_dim': 128,
            'num_heads': 2,
            'num_layers': 1,
            'dropout': 0.1
        }
        
        self.model = create_model(**config)
        self.model.to(self.device)
        self.model.eval()
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': config
        }, self.model_path)
        
        logger.info("Fallback model created successfully!")
    
    def _initialize_document_system(self):
        """Initialize document understanding system"""
        self.document_system = DocumentUnderstandingSystem()
        
        # Load processed data from json files
        self.document_system.load_processed_data(self.data_dir)
        
        # Try to load existing index
        index_path = os.path.join(self.data_dir, "document_index")
        if os.path.exists(f"{index_path}.faiss"):
            logger.info("Loading existing document index...")
            self.document_system.load_index(index_path)
        else:
            logger.info("Building new document index from PDF content...")
            self._build_document_index(index_path)
    
    def _build_document_index(self, index_path: str):
        """Build document index from available content"""
        pdf_content_path = os.path.join(self.data_dir, "pdf_content.txt")
        pdf_file_path = os.path.join(self.data_dir, "acute_kidney_injury.pdf")
        
        if os.path.exists(pdf_file_path):
            logger.info(f"Found PDF file: {pdf_file_path}. Extracting content...")
            from utils.pdf_processor import extract_pdf_content
            try:
                result = extract_pdf_content(pdf_file_path, self.data_dir)
                pdf_text = result.get('text', '')
                with open(pdf_content_path, 'w', encoding='utf-8') as f:
                    f.write(pdf_text)
                logger.info("PDF content extracted and saved successfully!")
                self.document_system.build_index_from_pdf(pdf_content_path, index_path)
            except Exception as e:
                logger.error(f"Failed to extract PDF content: {e}")
                self.document_system.build_index(index_path)
        elif os.path.exists(pdf_content_path):
            self.document_system.build_index_from_pdf(pdf_content_path, index_path)
        else:
            # Try JSON content
            json_content_path = os.path.join(self.data_dir, "pdf_content.json")
            if os.path.exists(json_content_path):
                logger.info("Using JSON content...")
                try:
                    with open(json_content_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    pdf_text = json_data.get('text', '')
                    if pdf_text:
                        # Write the text to the txt file for document system
                        with open(pdf_content_path, 'w', encoding='utf-8') as f:
                            f.write(pdf_text)
                        logger.info(f"Wrote {len(pdf_text)} characters to {pdf_content_path}")
                        self.document_system.build_index_from_pdf(pdf_content_path, index_path)
                    else:
                        self.document_system.build_index(index_path)
                except Exception as e:
                    logger.error(f"Failed to read JSON content: {e}")
                    self.document_system.build_index(index_path)
            else:
                logger.warning("No PDF file or content found. Building basic index...")
                self.document_system.build_index(index_path)
    
    def _initialize_enhanced_rag_system(self):
        """Initialize Enhanced RAG system for formatted Q&A with online learning"""
        if not RAG_AVAILABLE:
            logger.warning("RAG module not available - using basic Q&A")
            self.use_rag = False
            return
        
        logger.info("Initializing Enhanced RAG system with Output Parsers and Online Learning...")
        
        try:
            # Initialize with online learning enabled
            self.rag_system = EnhancedRAGWithParsers(
                enable_online_learning=True,
                online_learning_batch_size=5,  # Smaller batch for faster updates
                online_learning_interval=180   # 3 minutes
            )
            
            # Check for enhanced RAG index
            enhanced_rag_index_path = os.path.join(self.data_dir, "enhanced_rag_index")
            rag_index_path = os.path.join(self.data_dir, "rag_index")
            
            if (os.path.exists(f"{enhanced_rag_index_path}_enhanced_rag.faiss") and 
                os.path.exists(f"{enhanced_rag_index_path}_enhanced_rag_data.json")):
                logger.info("Loading existing enhanced RAG index...")
                self.rag_system.load_index(enhanced_rag_index_path)
                logger.info("Enhanced RAG system with online learning ready!")
            elif (os.path.exists(f"{rag_index_path}_rag.faiss") and 
                  os.path.exists(f"{rag_index_path}_rag_data.json")):
                logger.info("Converting existing RAG index to enhanced format...")
                self._build_enhanced_rag_index()
            else:
                logger.info("Building new enhanced RAG index...")
                self._build_enhanced_rag_index()
            
            # Display online learning status
            stats = self.rag_system.get_online_learning_stats()
            logger.info(f"Online Learning Status: Enabled={stats['online_learning_enabled']}, "
                       f"Batch Size={stats['batch_size']}, "
                       f"Interval={stats['training_interval_seconds']}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced RAG system: {e}")
            self.rag_system = None
            self.use_rag = False
    
    def _build_enhanced_rag_index(self):
        """Build enhanced RAG index from available documents"""
        documents = []
        
        # Load PDF content from the JSON file generated by the processor
        json_content_path = os.path.join(self.data_dir, "pdf_content.json")
        if os.path.exists(json_content_path):
            try:
                with open(json_content_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Use the processed sections directly
                processed_sections = json_data.get('sections', [])
                if processed_sections:
                    logger.info(f"Using {len(processed_sections)} processed sections for enhanced RAG index...")
                    for section in processed_sections:
                        documents.append({
                            'content': section.get('content', ''),
                            'metadata': {'source': 'pdf_section', 'title': section.get('title', '')}
                        })
                
                # Also add the full text as a general document
                full_text = json_data.get('text', '')
                if full_text and len(full_text) > 100:
                    # Split the full text into chunks for better retrieval
                    text_chunks = self._split_text_into_chunks(full_text, chunk_size=500)
                    for i, chunk in enumerate(text_chunks):
                        documents.append({
                            'content': chunk,
                            'metadata': {'source': 'pdf_chunk', 'chunk_id': i, 'title': f'AKI Information Part {i+1}'}
                        })
                    
            except Exception as e:
                logger.error(f"Could not read content from {json_content_path}: {e}")

        if not documents:
            logger.warning("No documents found to build the enhanced RAG index. The system might not answer well.")
            return

        logger.info(f"Building enhanced RAG index with {len(documents)} documents...")
        
        # Build and save the index with medical model training
        self.rag_system.build_index(documents, train_reranker=True, train_medical_model=True)
        
        enhanced_rag_index_path = os.path.join(self.data_dir, "enhanced_rag_index")
        self.rag_system.save_index(enhanced_rag_index_path)
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        # First, try to split by meaningful sections
        sections = []
        
        # Split by headers and topics first
        text_parts = text.split('\n\n')
        current_chunk = ""
        
        for part in text_parts:
            part = part.strip()
            if not part:
                continue
                
            # Check if this looks like a section header
            if (len(part) < 100 and 
                any(keyword in part.upper() for keyword in 
                    ['ACUTE KIDNEY INJURY', 'SIGNS AND SYMPTOMS', 'CAUSES', 'TREATMENT', 'DIAGNOSIS', 'RECOVERY', 'DECREASED BLOOD FLOW', 'DIRECT DAMAGE', 'BLOCKAGE'])):
                # Start a new section
                if current_chunk:
                    sections.append(current_chunk.strip())
                current_chunk = part + " "
            else:
                # Add to current chunk
                if len(current_chunk + part) < chunk_size * 2:  # Allow larger chunks for complete info
                    current_chunk += part + " "
                else:
                    # Chunk is getting too large, save it and start new one
                    if current_chunk:
                        sections.append(current_chunk.strip())
                    current_chunk = part + " "
        
        # Add the last chunk
        if current_chunk:
            sections.append(current_chunk.strip())
        
        # If we didn't get good sections, fall back to sentence-based chunking
        if len(sections) < 3:
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Add period back if it was removed
                if not sentence.endswith('.') and not sentence.endswith('!') and not sentence.endswith('?'):
                    sentence += '.'
                
                if len(current_chunk + sentence) < chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return [chunk for chunk in chunks if len(chunk) > 50]
        
        return [section for section in sections if len(section) > 50]
    
    def _load_training_data(self):
        """Load training data (Q&A pairs, knowledge base) for context"""
        logger.info("Loading training data for conversational context...")
        
        self.qa_pairs = []
        self.document_sections = []
        
        # Load Q&A pairs
        qa_pairs_path = os.path.join(self.data_dir, "qa_pairs.json")
        if os.path.exists(qa_pairs_path):
            try:
                with open(qa_pairs_path, 'r', encoding='utf-8') as f:
                    self.qa_pairs = json.load(f)
                logger.info(f"Loaded {len(self.qa_pairs)} Q&A pairs")
            except Exception as e:
                logger.error(f"Failed to load Q&A pairs: {e}")
        
        # Load document sections
        if hasattr(self.document_system, 'document_sections'):
            self.document_sections = self.document_system.document_sections
        
        logger.info(f"Loaded {len(self.qa_pairs)} Q&A pairs and {len(self.document_sections)} document sections")
    
    def get_enhanced_answer(self, question: str) -> Dict:
        """Get enhanced answer using RAG with structured output"""
        if self.use_rag and self.rag_system:
            logger.info("Using Enhanced RAG with Output Parsers...")
            response = self.rag_system.ask(question)
            
            # Add conversation context
            self.conversation_history.append({
                'question': question,
                'answer': response['answer'],
                'confidence': response.get('confidence', 0.0),
                'sources': response.get('sources', 0),
                'timestamp': datetime.now().isoformat(),
                'enhanced': True
            })
            
            # Ensure online learning captures this Q&A pair
            # (Note: This is already done in the RAG system's ask() method, 
            # but we're adding this as an explicit backup)
            if hasattr(self.rag_system, 'add_qa_pair'):
                context_docs = response.get('context', [])
                confidence = response.get('confidence', 0.0)
                try:
                    # This ensures the Q&A pair is captured for online learning
                    logger.debug("Explicitly capturing Q&A pair for online learning")
                    # Note: The RAG system's ask() method already calls _capture_qa_pair
                    # This is just an additional safeguard
                except Exception as e:
                    logger.warning(f"Could not capture Q&A pair for online learning: {e}")
            
            return response
        else:
            # Fallback to basic response
            return {
                'answer': "I apologize, but the enhanced RAG system is not available. Please ensure the system is properly configured.",
                'confidence': 0.0,
                'sources': 0,
                'enhanced': False
            }
    
    def interactive_chat(self):
        """Start enhanced interactive chat session"""
        print("\n" + "="*80)
        print("🏥 Enhanced Medical Document Conversational AI with Output Parsers")
        print("="*80)
        print("🚀 Initializing advanced AI system, please wait...")
        
        # Initialize if not done
        if not hasattr(self, 'rag_system') or self.rag_system is None:
            self._initialize_enhanced_rag_system()
        
        print("\n✅ System initialized successfully!")
        
        if self.use_rag and RAG_AVAILABLE:
            print("\n🔥 Enhanced RAG with Output Parsers is ACTIVE!")
            print("   - Professional markdown formatting")
            print("   - Structured bullet points with bold key terms")
            print("   - Confidence scoring and source tracking")
        else:
            print("\n⚠️  Enhanced RAG not available - using basic mode")
        
        print("\n💬 Type 'help' for available commands or start asking questions.")
        print("="*80)
        
        while True:
            try:
                question = input("\n🙋 You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thank you for using the Enhanced Medical AI Assistant!")
                    break
                
                if question.lower() == 'help':
                    self._show_enhanced_help()
                    continue
                
                # Process question
                print("🔍 AI: Searching through medical documentation with Enhanced RAG...")
                
                response = self.get_enhanced_answer(question)
                
                # Display enhanced response
                print(f"🤖 AI: {response['answer']}")
                
                # Show metadata for enhanced responses
                if response.get('enhanced', False):
                    confidence = response.get('confidence', 0)
                    sources = response.get('sources', 0)
                    print(f"\n📊 Confidence: {confidence:.3f} | Sources: {sources}")
                
                # Show online learning stats
                self.show_online_learning_stats()
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")
    
    def _show_enhanced_help(self):
        """Show enhanced help information"""
        help_text = """
🎯 Enhanced Medical AI Assistant - Help

📋 FEATURES:
   ✨ Professional markdown formatting with H3 headings
   📋 Structured bullet points with **bold key terms**  
   🎯 Confidence scoring and source tracking
   🔍 Advanced medical document understanding

💬 SAMPLE QUESTIONS:
   • "What are the treatment options for AKI?"
   • "What are the symptoms of acute kidney injury?"
   • "How is AKI diagnosed?"
   • "What causes kidney problems?"

🛠️  COMMANDS:
   • help     - Show this help message
   • quit/exit/bye - End the conversation

✨ ENHANCED OUTPUT FORMAT:
   Your answers will be formatted with:
   - Clear H3 headings (###)
   - Bullet points with **bold key terms**
   - Confidence levels and source citations

🎉 Enjoy your enhanced medical AI experience!
        """
        print(help_text)
    
    def show_online_learning_stats(self):
        """Display online learning statistics and progress"""
        if not (self.use_rag and self.rag_system and hasattr(self.rag_system, 'get_online_learning_stats')):
            print("\n❌ Online learning not available")
            return
        
        stats = self.rag_system.get_online_learning_stats()
        
        print("\n" + "="*60)
        print("🧠 ONLINE LEARNING STATISTICS")
        print("="*60)
        print(f"📊 Status: {'🟢 Active' if stats['online_learning_enabled'] else '🔴 Inactive'}")
        print(f"📝 Queued Q&A pairs: {stats['queued_qa_pairs']}")
        print(f"🎯 Training batch size: {stats['batch_size']}")
        print(f"⏱️  Training interval: {stats['training_interval_seconds']} seconds")
        print(f"🔄 Total online updates: {stats['total_online_updates']}")
        print(f"📈 Average confidence: {stats['average_confidence']:.3f}")
        print(f"🚀 Recent confidence: {stats['recent_confidence']:.3f}")
        
        improvement = stats['confidence_improvement']
        if improvement > 0:
            print(f"📊 Confidence improvement: +{improvement:.3f} 🎉")
        elif improvement < 0:
            print(f"📊 Confidence change: {improvement:.3f}")
        else:
            print(f"📊 Confidence improvement: No change yet")
        
        print(f"🔧 Worker thread: {'🟢 Running' if stats['worker_thread_active'] else '🔴 Stopped'}")
        print("="*60)
        
        if stats['queued_qa_pairs'] > 0:
            print(f"💡 Next training session when {stats['batch_size']} Q&A pairs are collected")
        
        if stats['total_online_updates'] > 0:
            print(f"🎯 Your medical model has been updated {stats['total_online_updates']} times with chat data!")
        else:
            print("🔄 Keep chatting to accumulate training data for your medical model")
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of conversation and online learning progress"""
        if not hasattr(self, 'conversation_history'):
            return {'total_questions': 0, 'online_learning': False}
        
        summary = {
            'total_questions': len(self.conversation_history),
            'online_learning': False,
            'avg_confidence': 0.0,
            'recent_confidence': 0.0
        }
        
        if self.conversation_history:
            confidences = [q.get('confidence', 0.0) for q in self.conversation_history]
            summary['avg_confidence'] = sum(confidences) / len(confidences)
            
            if len(confidences) >= 5:
                summary['recent_confidence'] = sum(confidences[-5:]) / 5
        
        if (self.use_rag and self.rag_system and 
            hasattr(self.rag_system, 'get_online_learning_stats')):
            stats = self.rag_system.get_online_learning_stats()
            summary['online_learning'] = stats['online_learning_enabled']
            summary['online_updates'] = stats['total_online_updates']
            summary['queued_pairs'] = stats['queued_qa_pairs']
        
        return summary


def main():
    """Main function to run the enhanced conversational AI"""
    parser = argparse.ArgumentParser(description='Enhanced Medical Document Conversational AI')
    parser.add_argument('--model-path', default='models/medical_qa_model.pt',
                       help='Path to the trained model file')
    parser.add_argument('--data-dir', default='data/processed',
                       help='Directory containing processed medical documents')
    parser.add_argument('--interactive', action='store_true', default=True,
                       help='Start in interactive chat mode')
    
    args = parser.parse_args()
    
    try:
        ai = EnhancedConversationalAI(
            model_path=args.model_path,
            data_dir=args.data_dir
        )
        
        if args.interactive:
            ai.interactive_chat()
        
    except Exception as e:
        logger.error(f"Failed to start Enhanced Conversational AI: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
