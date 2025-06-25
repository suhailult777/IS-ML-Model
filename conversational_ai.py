#!/usr/bin/env python3
"""
PDF-based Conversational AI System
Interactive terminal interface for medical document Q&A
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

# Remove duplicate logging setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


class ConversationalAI:
    """
    Main conversational AI system for PDF-based medical document Q&A
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
        self.rag_system = None  # Add RAG system
        self.conversation_history = []
        
        # Configuration
        self.max_context_length = 5  # Increased for better conversation flow
        self.confidence_threshold = 0.3  # Minimum confidence for answers
        self.use_rag = True  # Enable RAG by default
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing Conversational AI System...")
        
        try:
            # Load or create model
            self._load_or_create_model()
            
            # Initialize document understanding system
            self._initialize_document_system()
            
            # Initialize RAG system
            self._initialize_rag_system()
            
            # Load training data for context
            self._load_training_data()
            
            logger.info("System initialization completed successfully!")
            
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
            logger.info("No existing model found. Creating and training new model...")
            self._train_new_model()
    
    def _train_new_model(self):
        """Train a new model if none exists"""
        try:
            from training.pdf_training import train_medical_qa_model
            
            logger.info("Training new medical QA model with minimal settings...")
            self.model = train_medical_qa_model(
                data_dir=self.data_dir,
                model_save_path=self.model_path,
                num_epochs=1,  # Reduced for faster startup
                batch_size=2,  # Smaller batch size
                learning_rate=1e-4
            )
            
            logger.info("Model training completed!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.info("Creating simple fallback model...")
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
        self.document_system.load_processed_data(self.data_dir)        # Try to load existing index
        index_path = os.path.join(self.data_dir, "document_index")
        if os.path.exists(f"{index_path}.faiss"):
            logger.info("Loading existing document index...")
            self.document_system.load_index(index_path)
        else:
            logger.info("Building new document index from PDF content...")
            # First try to find and extract from actual PDF file
            pdf_content_path = os.path.join(self.data_dir, "pdf_content.txt")
            pdf_file_path = os.path.join(self.data_dir, "acute_kidney_injury.pdf")
            
            if os.path.exists(pdf_file_path):
                logger.info(f"Found PDF file: {pdf_file_path}. Extracting content...")
                # Extract PDF content automatically
                from utils.pdf_processor import extract_pdf_content
                try:
                    result = extract_pdf_content(pdf_file_path, self.data_dir)
                    pdf_text = result.get('text', '')
                    # Save extracted content to text file
                    with open(pdf_content_path, 'w', encoding='utf-8') as f:
                        f.write(pdf_text)
                    logger.info("PDF content extracted and saved successfully!")
                    self.document_system.build_index_from_pdf(pdf_content_path, index_path)
                except Exception as e:
                    logger.error(f"Failed to extract PDF content: {e}")
                    logger.warning("Building index from processed data...")
                    self.document_system.build_index(index_path)
            elif os.path.exists(pdf_content_path):
                self.document_system.build_index_from_pdf(pdf_content_path, index_path)
            else:
                # Try to use JSON content if available
                json_content_path = os.path.join(self.data_dir, "pdf_content.json")
                if os.path.exists(json_content_path):
                    logger.info("Using JSON content instead...")
                    try:
                        with open(json_content_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        pdf_text = json_data.get('text', '')
                        if pdf_text:
                            # Save to text file for processing
                            with open(pdf_content_path, 'w', encoding='utf-8') as f:
                                f.write(pdf_text)
                            self.document_system.build_index_from_pdf(pdf_content_path, index_path)
                        else:
                            logger.warning("No text content found in JSON. Building index from processed data...")
                            self.document_system.build_index(index_path)                    except Exception as e:
                        logger.error(f"Failed to read JSON content: {e}")
                        self.document_system.build_index(index_path)
                else:
                    logger.warning("No PDF file or PDF content file found. Building index from processed data...")
                    self.document_system.build_index(index_path)
    
    def _initialize_rag_system(self):
        """Initialize Enhanced RAG system for formatted Q&A"""
        if not RAG_AVAILABLE:
            logger.warning("RAG module not available - using basic Q&A")
            self.use_rag = False
            return
        
        logger.info("Initializing Enhanced RAG system...")
        
        try:
            self.rag_system = EnhancedRAGWithParsers()
            # Check if RAG index exists
            rag_index_path = os.path.join(self.data_dir, "rag_index")
            enhanced_rag_index_path = os.path.join(self.data_dir, "enhanced_rag_index")
            
            # Try enhanced RAG index first
            if (os.path.exists(f"{enhanced_rag_index_path}_enhanced_rag.faiss") and 
                os.path.exists(f"{enhanced_rag_index_path}_enhanced_rag_data.json")):
                logger.info("Loading existing enhanced RAG index...")
                self.rag_system.load_index(enhanced_rag_index_path)
                logger.info("Enhanced RAG system ready!")
            # Fallback to old RAG index format
            elif (os.path.exists(f"{rag_index_path}_rag.faiss") and 
                  os.path.exists(f"{rag_index_path}_rag_data.json")):
                logger.info("Loading existing RAG index...")
                # If using enhanced RAG, we need to rebuild the index
                if hasattr(self.rag_system, 'build_index'):
                    logger.info("Converting to enhanced RAG format...")
                    self._build_rag_index(enhanced_rag_index_path)
                else:
                    self.rag_system.load_index(rag_index_path)
                logger.info("RAG system ready!")
            else:
                logger.info("Building new enhanced RAG index...")
                self._build_rag_index(enhanced_rag_index_path)
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced RAG system: {e}")
            self.rag_system = None
            self.use_rag = False
    
    def _build_rag_index(self, rag_index_path: str):
        """Build RAG index from available documents"""
        documents = []
        
        # First, try to use the PDF content directly from JSON
        json_content_path = os.path.join(self.data_dir, "pdf_content.json")
        if os.path.exists(json_content_path):
            try:
                with open(json_content_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                pdf_text = json_data.get('text', '')
                if pdf_text:
                    logger.info("Using PDF content from JSON for RAG index...")
                    # Split the content into meaningful chunks
                    chunks = self._split_medical_content(pdf_text)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            'content': chunk,
                            'metadata': {'source': 'pdf_content', 'chunk_id': i, 'type': 'medical_text'}
                        })
                    logger.info(f"Added {len(chunks)} chunks from PDF content")
            except Exception as e:
                logger.error(f"Failed to load PDF JSON content: {e}")
        
        # Also check for any processed sections from the JSON
        if json_content_path and os.path.exists(json_content_path):
            try:
                with open(json_content_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                sections = json_data.get('sections', [])
                for section in sections:
                    if isinstance(section, dict) and 'content' in section:
                        documents.append({
                            'content': section['content'],
                            'metadata': {'source': 'pdf_sections', 'title': section.get('title', 'Unknown')}
                        })
                logger.info(f"Added {len(sections)} sections from PDF JSON")
            except Exception as e:
                logger.error(f"Failed to load PDF sections: {e}")
        
        # Use document sections if available
        if hasattr(self, 'document_system') and self.document_system and hasattr(self.document_system, 'document_sections'):
            for section in self.document_system.document_sections:
                if isinstance(section, dict) and 'content' in section:
                    documents.append(section)
            logger.info(f"Added {len(self.document_system.document_sections)} document sections")
        
        # Also try to load from text file if available
        pdf_content_path = os.path.join(self.data_dir, "pdf_content.txt")
        if os.path.exists(pdf_content_path):
            try:
                with open(pdf_content_path, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                if text_content:
                    chunks = self._split_medical_content(text_content)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            'content': chunk,
                            'metadata': {'source': 'pdf_text_file', 'chunk_id': i}
                        })
                    logger.info(f"Added {len(chunks)} chunks from PDF text file")
            except Exception as e:
                logger.error(f"Failed to load PDF text file: {e}")
        
        # Load QA pairs as documents
        qa_pairs_path = os.path.join(self.data_dir, "qa_pairs.json")
        if os.path.exists(qa_pairs_path):
            try:
                with open(qa_pairs_path, 'r', encoding='utf-8') as f:
                    qa_pairs = json.load(f)
                
                for qa in qa_pairs:
                    documents.append({
                        'content': f"Q: {qa['question']} A: {qa['answer']}",
                        'metadata': {'source': 'qa_pairs', 'type': 'qa'}
                    })
                logger.info(f"Added {len(qa_pairs)} Q&A pairs")
            except Exception as e:
                logger.error(f"Failed to load QA pairs: {e}")
        
        if documents:
            self.rag_system.build_index(documents)
            self.rag_system.save_index(rag_index_path)
            logger.info(f"RAG index built and saved with {len(documents)} documents!")
        else:
            logger.warning("No documents available for RAG indexing")
            self.use_rag = False
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        import re
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _load_training_data(self):
        """Load training data for context from the document system"""
        if self.document_system:
            self.qa_pairs = self.document_system.qa_pairs
            self.document_sections = self.document_system.document_sections
            logger.info(f"Loaded {len(self.qa_pairs)} Q&A pairs and {len(self.document_sections)} document sections from document system.")
        else:
            logger.warning("Document system not initialized. Cannot load training data.")
            self.qa_pairs = []
            self.document_sections = []
    
    def _get_relevant_context(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get relevant context for a question"""
        if self.document_system:
            try:
                return self.document_system.search(question, top_k=top_k)
            except Exception as e:
                logger.warning(f"Document system search failed: {e}")
        
        logger.warning("Document system not available for search. Returning empty context.")
        return []
    
    def _enhance_answer_with_context(self, question: str, relevant_docs: List[Tuple[str, float]]) -> str:
        """Generate enhanced answer using relevant context"""
        if not relevant_docs:
            return "I couldn't find specific information to answer your question. Could you please rephrase or ask about a different topic?"
        
        # Get the most relevant document
        best_doc, confidence = relevant_docs[0]
        
        if confidence < self.confidence_threshold:
            return f"Based on the available information (confidence: {confidence:.2f}), I found some potentially relevant content, but I'm not very confident. Here's what I found:\n\n{best_doc[:500]}..."
        
        # Create comprehensive answer
        answer_parts = []
        answer_parts.append("Based on the medical documentation guidelines, here's what I found:")
        answer_parts.append("")
        
        # Add primary answer
        answer_parts.append(f"**Primary Information (Confidence: {confidence:.2f}):**")
        answer_parts.append(best_doc[:800] + ("..." if len(best_doc) > 800 else ""))
          # Add additional context if available
        if len(relevant_docs) > 1:
            answer_parts.append("")
            answer_parts.append("**Additional Related Information:**")
            for i, (doc, score) in enumerate(relevant_docs[1:3], 1):  # Show up to 2 more
                answer_parts.append(f"{i}. (Confidence: {score:.2f}) {doc[:300]}...")
        
        return "\n".join(answer_parts)
    
    def _add_to_conversation_history(self, question: str, answer: str, metadata: Dict = None):
        """Enhanced conversation history tracking"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer
        }
        
        # Add metadata if provided
        if metadata:
            entry.update(metadata)
        
        self.conversation_history.append(entry)
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_context_length:
            self.conversation_history = self.conversation_history[-self.max_context_length:]
    
    def ask_question(self, question: str) -> str:
        """Enhanced question processing with RAG"""
        logger.info(f"Processing question: {question[:100]}...")
        
        try:
            # Use RAG if available for better answers
            if self.use_rag and self.rag_system:
                return self._ask_with_rag(question)
            else:
                # Fallback to original method
                relevant_docs = self._get_relevant_context(question)
                answer = self._enhance_answer_with_context(question, relevant_docs)
                self._add_to_conversation_history(question, answer)
                return answer
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"I encountered an error while processing your question. Please try rephrasing or ask a different question. Error: {str(e)}"
    
    def _ask_with_rag(self, question: str) -> str:
        """Process question using RAG system"""
        try:
            # Add conversation context to the question for better continuity
            enhanced_question = self._enhance_question_with_context(question)
            
            # Use RAG to get answer
            rag_response = self.rag_system.ask(enhanced_question, top_k=5)
            
            # Format the response
            answer = rag_response['answer']
            confidence = rag_response['confidence']
            
            # Add conversation flow improvements
            if confidence > 0.6:
                # High confidence - add follow-up suggestions
                follow_ups = self._suggest_follow_up_questions(question, rag_response)
                if follow_ups:
                    answer += "\n\n**You might also want to ask:**"
                    for i, follow_up in enumerate(follow_ups[:3], 1):
                        answer += f"\n{i}. {follow_up}"
            
            elif confidence < 0.4:
                # Low confidence - suggest rephrasing
                answer += "\n\n*If this doesn't fully answer your question, try asking more specifically about symptoms, causes, diagnosis, or treatment.*"
            
            # Add to conversation history with metadata
            self._add_to_conversation_history(question, answer, {
                'confidence': confidence,
                'sources': rag_response['sources'],
                'method': 'rag'
            })
            
            return answer
            
        except Exception as e:
            logger.error(f"RAG processing failed: {e}")
            # Fallback to original method
            relevant_docs = self._get_relevant_context(question)
            answer = self._enhance_answer_with_context(question, relevant_docs)
            self._add_to_conversation_history(question, answer)
            return answer
    
    def _enhance_question_with_context(self, question: str) -> str:
        """Add conversation context to improve question understanding"""
        if not self.conversation_history:
            return question
        
        # Get recent conversation topics
        recent_topics = []
        for exchange in self.conversation_history[-2:]:  # Last 2 exchanges
            prev_question = exchange['question'].lower()
            # Extract key medical terms
            medical_terms = []
            for word in prev_question.split():
                if len(word) > 4 and any(word.endswith(suffix) for suffix in ['itis', 'osis', 'emia', 'uria']):
                    medical_terms.append(word)
            recent_topics.extend(medical_terms)
        
        # If current question is short and we have context, enhance it
        if len(question.split()) < 5 and recent_topics:
            # Simple context enhancement
            if any(word in question.lower() for word in ['what', 'how', 'why', 'when']):
                enhanced = f"{question} (in context of {', '.join(recent_topics[:2])})"
                return enhanced
        
        return question
    
    def _suggest_follow_up_questions(self, question: str, rag_response: Dict) -> List[str]:
        """Generate relevant follow-up questions based on the current question and answer"""
        question_lower = question.lower()
        follow_ups = []
        
        # Analyze the question type and suggest related topics
        if any(word in question_lower for word in ['what is', 'define', 'definition']):
            # Definition questions - suggest symptoms, causes, treatment
            follow_ups.extend([
                "What are the symptoms?",
                "What causes this condition?",
                "How is it treated?"
            ])
        
        elif any(word in question_lower for word in ['symptom', 'sign', 'present']):
            # Symptom questions - suggest causes and treatment
            follow_ups.extend([
                "What are the common causes?",
                "How is this diagnosed?",
                "What are the treatment options?"
            ])
        
        elif any(word in question_lower for word in ['cause', 'why', 'reason']):
            # Cause questions - suggest prevention and treatment
            follow_ups.extend([
                "How can this be prevented?",
                "What are the treatment options?",
                "What are the risk factors?"
            ])
        
        elif any(word in question_lower for word in ['treat', 'therapy', 'cure']):
            # Treatment questions - suggest prognosis and prevention
            follow_ups.extend([
                "What is the prognosis?",
                "Are there any side effects?",
                "How long does treatment take?"
            ])
        
        elif any(word in question_lower for word in ['diagnos', 'test', 'detect']):
            # Diagnosis questions - suggest treatment and prognosis
            follow_ups.extend([
                "What are the treatment options?",
                "What happens if left untreated?",
                "How accurate are these tests?"
            ])
        
        # Add specific medical follow-ups based on detected conditions
        if 'kidney' in question_lower or 'renal' in question_lower:
            follow_ups.extend([
                "What are the stages of kidney disease?",
                "How does diet affect kidney function?",
                "When is dialysis needed?"
            ])
        
        # Filter out questions that are too similar to the current one
        filtered_follow_ups = []
        for follow_up in follow_ups:
            if not self._questions_too_similar(question, follow_up):
                filtered_follow_ups.append(follow_up)
        
        return filtered_follow_ups[:3]  # Return top 3
    
    def _questions_too_similar(self, q1: str, q2: str) -> bool:
        """Check if two questions are too similar"""
        q1_words = set(q1.lower().split())
        q2_words = set(q2.lower().split())
        
        # Remove common words
        common_words = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an'}
        q1_words -= common_words
        q2_words -= common_words
        
        if not q1_words or not q2_words:
            return False
        
        # Check overlap
        overlap = len(q1_words & q2_words) / len(q1_words | q2_words)
        return overlap > 0.6  # More than 60% overlap
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary = ["**Conversation Summary:**", ""]
        for i, exchange in enumerate(self.conversation_history, 1):
            summary.append(f"**Q{i}:** {exchange['question']}")
            summary.append(f"**A{i}:** {exchange['answer'][:200]}...")
            summary.append("")
        
        return "\n".join(summary)
    
    def get_system_info(self) -> str:
        """Enhanced system information including RAG status"""
        info = [
            "**Enhanced Medical Document AI System Information:**",
            "",
            f"- Model: {'Loaded' if self.model else 'Not loaded'}",
            f"- RAG System: {'Active' if self.use_rag and self.rag_system else 'Inactive'}",
            f"- Device: {self.device}",
            f"- Document sections: {len(self.document_sections) if hasattr(self, 'document_sections') else 0}",
            f"- Q&A pairs: {len(self.qa_pairs) if hasattr(self, 'qa_pairs') else 0}",
            f"- Conversation history: {len(self.conversation_history)} exchanges",
            f"- Confidence threshold: {self.confidence_threshold}",
            "",
            "**AI Features:**",
            "- ✓ Advanced retrieval-augmented generation (RAG)",
            "- ✓ Conversational context awareness", 
            "- ✓ Medical document understanding",
            "- ✓ Follow-up question suggestions",
            "- ✓ Confidence scoring",
            "",
            "**Available Commands:**",
            "- Ask any question about medical documentation",
            "- Type 'help' for more commands",
            "- Type 'summary' to see conversation history",
            "- Type 'info' to see this information",
            "- Type 'clear' to clear conversation history",
            "- Type 'quit' or 'exit' to end session"
        ]
        
        return "\n".join(info)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared."
    
    def save_conversation(self, filepath: str = None):
        """Save conversation to file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"conversation_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        
        return f"Conversation saved to {filepath}"
    
    def _split_medical_content(self, text: str) -> List[str]:
        """Split medical content into meaningful chunks for RAG"""
        import re
        
        # Split by medical sections and paragraphs
        chunks = []
        
        # Split by common medical headers
        section_patterns = [
            r'Signs and symptoms?',
            r'Causes?',
            r'Diagnosis',
            r'Treatment',
            r'Recovery',
            r'ACUTE KIDNEY INJURY',
            r'DECREASED BLOOD FLOW',
            r'DIRECT DAMAGE',
            r'BLOCKAGE OF'
        ]
        
        # Try to split by these patterns
        current_chunk = ""
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a header
            is_header = False
            for pattern in section_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_header = True
                    break
            
            if is_header and current_chunk:
                # Save current chunk and start new one
                if len(current_chunk.strip()) > 50:  # Only save meaningful chunks
                    chunks.append(current_chunk.strip())
                current_chunk = line + " "
            else:
                current_chunk += line + " "
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 50:
            chunks.append(current_chunk.strip())
        
        # If no good splits found, split by sentences
        if not chunks:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 30]
            
            # Group sentences into chunks
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 500:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]  # Fallback to original text if no splits
    
    # ...existing code...
def run_interactive_terminal():
    """Run interactive terminal interface"""
    print("=" * 80)
    print("🏥 Enhanced Medical Document Conversational AI with RAG")
    print("=" * 80)
    print("🚀 Initializing advanced AI system, please wait...")
    print()
    
    try:
        # Initialize AI system
        ai = ConversationalAI()
        
        print("✅ System initialized successfully!")
        print()
        if ai.use_rag and ai.rag_system:
            print("🔥 RAG (Retrieval-Augmented Generation) is ACTIVE!")
            print("   - Enhanced answer quality and relevance")
            print("   - Intelligent follow-up suggestions")
            print("   - Advanced conversation context")
        else:
            print("⚠️  RAG system unavailable - using basic Q&A")
        
        print()
        print("💬 Type 'help' for available commands or start asking questions.")
        print("=" * 80)
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("🙋 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🤖 AI: Goodbye! Thank you for using the Enhanced Medical Document AI system.")
                    
                    # Ask if user wants to save conversation
                    if ai.conversation_history:
                        save_choice = input("💾 Would you like to save this conversation? (y/n): ").strip().lower()
                        if save_choice in ['y', 'yes']:
                            filename = ai.save_conversation()
                            print(f"🤖 AI: {filename}")
                    
                    break
                
                elif user_input.lower() == 'help':
                    response = ai.get_system_info()
                
                elif user_input.lower() == 'summary':
                    response = ai.get_conversation_summary()
                
                elif user_input.lower() == 'info':
                    response = ai.get_system_info()
                
                elif user_input.lower() == 'clear':
                    response = ai.clear_history()
                
                elif user_input.lower().startswith('save'):
                    # Extract filename if provided
                    parts = user_input.split()
                    filename = parts[1] if len(parts) > 1 else None
                    response = ai.save_conversation(filename)
                
                else:
                    # Process as question
                    if ai.use_rag:
                        print("🔍 AI: Searching through medical documentation with RAG...")
                    else:
                        print("🔍 AI: Searching through medical documentation...")
                    response = ai.ask_question(user_input)
                
                # Display response
                print(f"🤖 AI: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n\n🤖 AI: Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n🤖 AI: An error occurred: {e}")
                print("Please try again or type 'quit' to exit.")
                print()
    
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Please check your setup and try again.")
        return 1
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Medical Document Conversational AI")
    parser.add_argument('--model-path', default='models/medical_qa_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data-dir', default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--train', action='store_true',
                       help='Force retrain model even if one exists')
    
    args = parser.parse_args()
    
    # Force retrain if requested
    if args.train and os.path.exists(args.model_path):
        os.remove(args.model_path)
        print(f"Removed existing model {args.model_path} for retraining")
    
    # Run interactive terminal
    return run_interactive_terminal()


if __name__ == "__main__":
    sys.exit(main())
