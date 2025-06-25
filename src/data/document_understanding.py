#!/usr/bin/env python3
"""
Document Understanding System for PDF-based Conversational AI
Handles document indexing, retrieval, and comprehension
"""

import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIndex:
    """
    FAISS-based document index for fast similarity search
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self.documents = []
        self.metadata = []
        
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict]):
        """Add documents to the index"""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(documents)} documents to index. Total: {len(self.documents)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents"""
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(score),
                    self.metadata[idx]
                ))
        
        return results
    
    def save(self, filepath: str):
        """Save index to disk"""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save documents and metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        
        logger.info(f"Saved index to {filepath}")
    
    def load(self, filepath: str):
        """Load index from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load documents and metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
        
        logger.info(f"Loaded index from {filepath}")


class DocumentUnderstandingSystem:
    """
    Complete document understanding system
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', embedding_dim: int = 384):
        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = embedding_dim
        self.index = DocumentIndex(embedding_dim)
        self.document_sections: List[Dict] = []
        self.qa_pairs: List[Dict] = []
        self.knowledge_base: Dict = {}

    def load_processed_data(self, processed_data_path: str):
        """Load all processed data files"""
        data_path = Path(processed_data_path)
        
        # Load document sections
        sections_file = data_path / 'sections.json'
        if sections_file.exists():
            with open(sections_file, 'r') as f:
                self.document_sections = json.load(f)
            logger.info(f"Loaded {len(self.document_sections)} document sections.")

        # Load Q&A pairs
        qa_file = data_path / 'qa_pairs.json'
        if qa_file.exists():
            with open(qa_file, 'r') as f:
                self.qa_pairs = json.load(f)
            logger.info(f"Loaded {len(self.qa_pairs)} Q&A pairs.")

        # Load knowledge base
        kb_file = data_path / 'knowledge_base.json'
        if kb_file.exists():
            with open(kb_file, 'r') as f:
                self.knowledge_base = json.load(f)
            logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} entries.")

    def build_index(self, index_path: str):
        """Build the document index from sections and Q&A pairs"""
        logger.info("Building new document index...")
        
        # Combine documents from sections and Q&A
        all_docs = []
        all_metadata = []
        
        # Add document sections
        for i, section in enumerate(self.document_sections):
            all_docs.append(section.get('content', ''))
            all_metadata.append({'source': 'section', 'id': i})
            
        # Add questions from Q&A pairs
        for i, qa in enumerate(self.qa_pairs):
            all_docs.append(qa.get('question', ''))
            all_metadata.append({'source': 'qa', 'id': i, 'answer_ref': qa.get('answer')})

        if not all_docs:
            logger.warning("No documents found to build index. Skipping.")
            return

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_docs)} documents...")
        embeddings = self.model.encode(all_docs, convert_to_tensor=False, show_progress_bar=True)
        
        # Add to index
        self.index.add_documents(all_docs, embeddings, all_metadata)
        
        # Save index
        self.index.save(index_path)

    def load_index(self, index_path: str):
        """Load an existing document index"""
        self.index.load(index_path)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for relevant documents"""
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        results = self.index.search(query_embedding, top_k)
        
        # Return only document and score
        return [(doc, score) for doc, score, meta in results]

    def get_answer_from_qa(self, question: str) -> Optional[str]:
        """Try to find a direct answer from the Q&A pairs"""
        # This is a simple implementation. For better performance, this could also use similarity search.
        for qa in self.qa_pairs:
            if question.lower() in qa.get('question', '').lower():
                return qa.get('answer')
        return None
    
    def process_pdf_content(self, pdf_content: str) -> List[Dict]:
        """Process raw PDF content into meaningful sections"""
        logger.info("Processing PDF content into sections...")
        
        sections = []
        
        # Split content by medical conditions/topics
        # The PDF contains sections like "Acute Kidney Injury", "Heart Failure", etc.
        
        # Split by common medical section headers
        medical_topics = [
            "Acute Kidney Injury", "Acute Tubular Necrosis", "Acute Blood Loss Anemia",
            "BMI", "Chest Pain", "CKD Stage", "Debridement", "Demand Ischemia",
            "Depression", "Encephalopathy", "Functional Quadriplegia", "GI Bleeding",
            "Heart Failure", "HIV/AIDS", "Hypertensive", "Malnutrition", "Pancytopenia",
            "Pneumonia", "Pressure Ulcer", "Respiratory Failure", "Schizophrenia",
            "Sepsis", "Shock", "Syncope", "TIA", "Urosepsis"
        ]
        
        # Find sections by looking for topic headers followed by content
        lines = pdf_content.split('\n')
        current_section = ""
        current_topic = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line contains a medical topic
            found_topic = None
            for topic in medical_topics:
                if topic.lower() in line.lower() and len(line) < 100:  # Header-like line
                    found_topic = topic
                    break
            
            if found_topic:
                # Save previous section if it exists
                if current_section and current_topic:
                    sections.append({
                        'title': current_topic,
                        'content': current_section.strip(),
                        'keywords': self._extract_keywords(current_section),
                        'type': 'medical_condition'
                    })
                
                # Start new section
                current_topic = found_topic
                current_section = line + " "
            else:
                # Add to current section
                if current_topic:
                    current_section += line + " "
        
        # Add final section
        if current_section and current_topic:
            sections.append({
                'title': current_topic,
                'content': current_section.strip(),
                'keywords': self._extract_keywords(current_section),
                'type': 'medical_condition'
            })
        
        # Also create general sections by splitting long content
        if len(sections) < 5:  # If we didn't find many topic sections, split differently
            chunks = self._split_into_chunks(pdf_content, chunk_size=500)
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 100:
                    sections.append({
                        'title': f'Medical Documentation Section {i+1}',
                        'content': chunk.strip(),
                        'keywords': self._extract_keywords(chunk),
                        'type': 'general'
                    })
        
        logger.info(f"Processed PDF into {len(sections)} sections")
        return sections
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        medical_keywords = [
            'acute', 'chronic', 'kidney', 'injury', 'failure', 'respiratory', 'heart',
            'blood', 'pressure', 'diagnosis', 'treatment', 'patient', 'medical',
            'condition', 'symptoms', 'pneumonia', 'sepsis', 'anemia', 'depression',
            'malnutrition', 'shock', 'bleeding', 'infection', 'creatinine', 'urine',
            'output', 'laboratory', 'results'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in medical_keywords if kw in text_lower]
        return found_keywords
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def build_index_from_pdf(self, pdf_content_path: str, index_path: str):
        """Build document index directly from PDF content"""
        logger.info("Building index from PDF content...")
        
        # Load PDF content
        with open(pdf_content_path, 'r', encoding='utf-8') as f:
            pdf_content = f.read()
        
        # Process into sections
        sections = self.process_pdf_content(pdf_content)
        self.document_sections = sections
        
        # Build index
        all_docs = []
        all_metadata = []
        
        for i, section in enumerate(sections):
            all_docs.append(section['content'])
            all_metadata.append({
                'source': 'pdf_section',
                'id': i,
                'title': section['title'],
                'type': section['type'],
                'keywords': section['keywords']
            })
        
        # Add Q&A pairs if available
        for i, qa in enumerate(self.qa_pairs):
            all_docs.append(qa.get('question', ''))
            all_metadata.append({
                'source': 'qa',
                'id': i,
                'answer_ref': qa.get('answer', '')
            })
        
        if not all_docs:
            logger.warning("No documents found to build index.")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_docs)} documents...")
        embeddings = self.model.encode(all_docs, convert_to_tensor=False, show_progress_bar=True)
        
        # Add to index
        self.index.add_documents(all_docs, embeddings, all_metadata)
        
        # Save index
        self.index.save(index_path)
        logger.info("Index built and saved successfully!")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Enhanced search with better result formatting"""
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        results = self.index.search(query_embedding, top_k)
        
        formatted_results = []
        for doc, score, metadata in results:
            # If this is a Q&A result, return the answer
            if metadata.get('source') == 'qa' and metadata.get('answer_ref'):
                formatted_results.append((metadata['answer_ref'], score))
            else:
                # For PDF sections, return the content
                formatted_results.append((doc, score))
        
        return formatted_results


def initialize_document_system(data_dir: str = "data/processed", 
                             rebuild_index: bool = False) -> DocumentUnderstandingSystem:
    """
    Initialize the document understanding system
    
    Args:
        data_dir: Directory containing processed data
        rebuild_index: Whether to rebuild the index from scratch
        
    Returns:
        Initialized DocumentUnderstandingSystem
    """
    logger.info("Initializing Document Understanding System...")
    
    # Create system
    system = DocumentUnderstandingSystem()
    
    # Load processed data
    system.load_processed_data(data_dir)
    
    # Load or build index
    if rebuild_index or not system.load_document_index():
        system.build_document_index()
    
    # Print statistics
    stats = system.get_document_statistics()
    logger.info(f"System initialized with {stats['total_documents']} documents")
    logger.info(f"Categories: {list(stats['category_breakdown'].keys())}")
    
    return system


if __name__ == "__main__":
    # Test the document understanding system
    print("Testing Document Understanding System...")
    
    try:
        # Initialize system
        system = initialize_document_system(rebuild_index=True)
        
        # Test questions
        test_questions = [
            "What is acute kidney injury?",
            "How do you document heart failure?",
            "What are the requirements for depression assessment?",
            "Tell me about chest pain documentation"
        ]
        
        for question in test_questions:
            print(f"\nQ: {question}")
            result = system.answer_question(question)
            print(f"A: {result['answer'][:200]}...")
            print(f"Confidence: {result['confidence']:.3f}")
        
        # Print statistics
        stats = system.get_document_statistics()
        print(f"\nSystem Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error testing system: {e}")
        import traceback
        traceback.print_exc()
