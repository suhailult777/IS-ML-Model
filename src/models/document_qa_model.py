#!/usr/bin/env python3
"""
Document Understanding and Question Answering Model
Transformer-based architecture for PDF content comprehension and conversational AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentEncoder(nn.Module):
    """
    Encodes document sections into dense vector representations
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 hidden_dim: int = 384, dropout: float = 0.1):
        super(DocumentEncoder, self).__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        
        # Load pre-trained sentence transformer
        self.sentence_model = SentenceTransformer(model_name)
        
        # Additional layers for fine-tuning
        self.projection = nn.Sequential(
            nn.Linear(self.sentence_model.get_sentence_embedding_dimension(), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of text segments
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        # Get embeddings from sentence transformer
        with torch.no_grad():
            embeddings = self.sentence_model.encode(texts, convert_to_tensor=True)
        
        # Apply projection layers
        projected = self.projection(embeddings)
        
        return projected


class QuestionEncoder(nn.Module):
    """
    Encodes questions into vector representations
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 hidden_dim: int = 384, dropout: float = 0.1):
        super(QuestionEncoder, self).__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        
        # Load pre-trained sentence transformer
        self.sentence_model = SentenceTransformer(model_name)
        
        # Question-specific projection
        self.projection = nn.Sequential(
            nn.Linear(self.sentence_model.get_sentence_embedding_dimension(), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, questions: List[str]) -> torch.Tensor:
        """
        Encode a list of questions
        
        Args:
            questions: List of question strings
            
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        # Get embeddings from sentence transformer
        with torch.no_grad():
            embeddings = self.sentence_model.encode(questions, convert_to_tensor=True)
        
        # Apply projection layers
        projected = self.projection(embeddings)
        
        return projected


class AttentionMechanism(nn.Module):
    """
    Attention mechanism for document-question interaction
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(AttentionMechanism, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism
        
        Args:
            query: Query tensor (questions)
            key: Key tensor (document sections)
            value: Value tensor (document sections)
            
        Returns:
            Attended features
        """
        attended, attention_weights = self.attention(query, key, value)
        output = self.norm(attended + query)
        
        return output, attention_weights


class DocumentQAModel(nn.Module):
    """
    Complete Document Question Answering Model
    
    Combines document encoding, question encoding, and a classifier
    to score the relevance of a document to a question.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 hidden_dim: int = 384,
                 dropout: float = 0.1,
                 **kwargs): # Accept other args
        super(DocumentQAModel, self).__init__()
        
        # Encoders
        self.document_encoder = DocumentEncoder(model_name, hidden_dim, dropout)
        self.question_encoder = QuestionEncoder(model_name, hidden_dim, dropout)
        
        # Relevance classifier
        self.relevance_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Takes concatenated doc and question embeddings
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) # Outputs a single relevance score
        )
        
    def forward(self, 
                document_sections: List[str], 
                questions: List[str]) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for document QA
        
        Args:
            document_sections: List of document text sections
            questions: List of questions
            
        Returns:
            Tuple of (relevance_scores, None)
        """
        # Encode documents and questions
        doc_embeddings = self.document_encoder(document_sections)
        question_embeddings = self.question_encoder(questions)
        
        # Concatenate features
        combined_features = torch.cat((doc_embeddings, question_embeddings), dim=1)
        
        # Get relevance score
        relevance_scores = self.relevance_classifier(combined_features)
        
        return relevance_scores.view(-1), None # Return scores and None for attention weights
    
    def predict_best_answer(self, 
                           document_sections: List[str], 
                           question: str,
                           top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict the best answer sections for a given question
        
        Args:
            document_sections: List of document sections
            question: Single question string
            top_k: Number of top sections to return
            
        Returns:
            List of (section_text, relevance_score) tuples
        """
        self.eval()
        with torch.no_grad():
            questions = [question] * len(document_sections)
            
            # Get relevance scores
            relevance_scores, _ = self(document_sections, questions)
            
            # Get top k results
            top_k_indices = torch.topk(relevance_scores, k=min(top_k, len(relevance_scores))).indices
            
            results = []
            for idx in top_k_indices:
                section = document_sections[idx.item()]
                score = relevance_scores[idx.item()].item()
                results.append((section, score))
                
        return results


class MedicalDocumentQA(DocumentQAModel):
    """
    Specialized version for medical document QA
    """
    
    def __init__(self, **kwargs):
        # Use a medical-domain model if available
        model_name = kwargs.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        super(MedicalDocumentQA, self).__init__(model_name=model_name, **kwargs)
        
        # Medical-specific keywords and patterns
        self.medical_keywords = [
            'diagnosis', 'treatment', 'symptoms', 'condition', 'patient',
            'medical', 'clinical', 'documentation', 'query', 'template'
        ]
    
    def enhance_medical_context(self, text: str) -> str:
        """
        Enhance text with medical context markers
        """
        enhanced_text = text
        for keyword in self.medical_keywords:
            if keyword.lower() in text.lower():
                enhanced_text = enhanced_text.replace(
                    keyword, f"[MEDICAL:{keyword.upper()}]"
                )
        return enhanced_text


def create_model(model_type: str = "medical", **kwargs) -> DocumentQAModel:
    """
    Factory function to create appropriate model
    
    Args:
        model_type: Type of model ('medical' or 'general')
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    if model_type == "medical":
        return MedicalDocumentQA(**kwargs)
    else:
        return DocumentQAModel(**kwargs)


if __name__ == "__main__":
    # Test the model
    print("Testing Document QA Model...")
    
    # Create model
    model = create_model("medical", hidden_dim=384)
    
    # Test data
    documents = [
        "Acute kidney injury documentation requires creatinine levels and urine output.",
        "Heart failure diagnosis involves echocardiogram and BNP levels.",
        "Depression assessment includes psychiatric consultation and medication review."
    ]
    
    questions = [
        "What is needed for kidney injury documentation?",
        "How do you diagnose heart failure?",
        "What is required for depression assessment?"
    ]
    
    # Test forward pass
    try:
        relevance_scores, attention_weights = model(documents, questions)
        print(f"Model output shape: {relevance_scores.shape}")
        print("Model test successful!")
        
        # Test prediction
        best_answers = model.predict_best_answer(documents, "What is needed for kidney injury?")
        print(f"Best answer: {best_answers[0][0][:100]}...")
        
    except Exception as e:
        print(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()
