#!/usr/bin/env python3
"""
Enhanced RAG with Output Parsers
Provides structured, formatted output like GPT using LangChain output parsers
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import os
import re
import threading
import time
from datetime import datetime
from collections import deque

# Custom imports for training and reranking
from training.relevancy_trainer import train_relevancy_model
from models.document_qa_model import DocumentQAModel, create_model

logger = logging.getLogger(__name__)

class MedicalResponse(BaseModel):
    """Structured medical response with formatted output"""
    
    title: str = Field(description="The main topic title using H3 heading (###)")
    summary: Optional[str] = Field(default=None, description="A brief, one or two sentence summary of the answer.")
    bullet_points: List[str] = Field(description="List of bullet points with bold key terms")
    conclusion: Optional[str] = Field(default=None, description="A concluding sentence to wrap up the answer.")
    confidence_level: Optional[str] = Field(default=None, description="Confidence level of the response")
    
    def to_formatted_string(self) -> str:
        """Convert to formatted markdown string"""
        formatted = f"### {self.title}\n\n"
        if self.summary:
            formatted += f"**Summary**: {self.summary}\n\n"
        
        for bullet in self.bullet_points:
            formatted += f"- {bullet}\n"
        
        if self.conclusion:
            formatted += f"\n**Conclusion**: {self.conclusion}"
        
        if self.confidence_level:
            formatted += f"\n\n*Confidence: {self.confidence_level}*"
            
        return formatted

class EnhancedRAGWithParsers:
    """
    Enhanced RAG system with structured output parsers for formatted responses
    and online/continuous learning capabilities
    """
    
    def __init__(self, 
                 retrieval_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 medical_model_path: str = "models/medical_qa_model.pt",
                 reranker_model_path: str = "models/reranker_model.pt",
                 enable_online_learning: bool = True,
                 online_learning_batch_size: int = 10,
                 online_learning_interval: int = 300):  # 5 minutes
        
        logger.info("Initializing Enhanced RAG system with online learning capabilities...")
        
        # Initialize retrieval components
        self.retriever = SentenceTransformer(retrieval_model)
        self.index = None
        self.documents = []
        self.doc_metadata = []
        
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize your custom medical model for generation
        self.medical_model = None
        self.medical_model_path = medical_model_path
        self._load_medical_model()
        
        # Reranker model
        self.reranker_model = None
        self.reranker_model_path = reranker_model_path
        self._load_reranker_model()
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=MedicalResponse)
        
        # Online learning components
        self.enable_online_learning = enable_online_learning
        self.online_learning_batch_size = online_learning_batch_size
        self.online_learning_interval = online_learning_interval
        
        # Thread-safe queue for collecting Q&A pairs
        self.qa_queue = deque(maxlen=1000)  # Keep last 1000 Q&A pairs
        self.qa_queue_lock = threading.Lock()
        
        # Online training thread
        self.online_training_thread = None
        self.stop_online_training = threading.Event()
        
        # Confidence tracking for improvement measurement
        self.confidence_history = deque(maxlen=100)
        
        # Start online learning if enabled
        if self.enable_online_learning:
            self._start_online_learning()
            
        logger.info(f"Online learning: {'Enabled' if self.enable_online_learning else 'Disabled'}")
        
    def _load_medical_model(self):
        """Load your custom medical model for generation"""
        if os.path.exists(self.medical_model_path):
            try:
                logger.info(f"Loading your custom medical model from {self.medical_model_path}")
                checkpoint = torch.load(self.medical_model_path, map_location=self.device)
                config = checkpoint.get('config', {})
                model_type = config.pop('model_type', 'medical')
                
                # Handle config compatibility
                if 'hidden_size' in config:
                    config['hidden_dim'] = config.pop('hidden_size')
                
                self.medical_model = create_model(model_type, **config)
                self.medical_model.load_state_dict(checkpoint['model_state_dict'])
                self.medical_model.to(self.device)
                self.medical_model.eval()
                logger.info("Custom medical model loaded successfully!")
            except Exception as e:
                logger.error(f"Error loading medical model: {e}")
                logger.info("Creating new medical model...")
                self.medical_model = create_model('medical', hidden_dim=384)
                self.medical_model.to(self.device)
                self.medical_model.eval()
        else:
            logger.warning(f"Medical model not found at {self.medical_model_path}, creating new model")
            self.medical_model = create_model('medical', hidden_dim=384)
            self.medical_model.to(self.device)
            self.medical_model.eval()
        
    def _load_reranker_model(self):
        """Loads the trained reranker model."""
        if os.path.exists(self.reranker_model_path):
            try:
                logger.info(f"Loading reranker model from {self.reranker_model_path}")
                self.reranker_model = create_model(model_type='medical')
                checkpoint = torch.load(self.reranker_model_path, map_location=self.device)
                self.reranker_model.load_state_dict(checkpoint['model_state_dict'])
                self.reranker_model.to(self.device)
                self.reranker_model.eval()
                logger.info("Reranker model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading reranker model: {e}")
                self.reranker_model = None
        else:
            logger.warning("No reranker model found. Retrieval will rely on FAISS scores only.")

    def build_index(self, documents: List[Dict], train_reranker: bool = True, train_medical_model: bool = True):
        """Build FAISS index from documents and train models on PDF content"""
        logger.info(f"Building RAG index from {len(documents)} documents...")
        
        self.documents = documents
        self.doc_metadata = [doc.get('metadata', {}) for doc in documents]
        
        # Extract text content
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.retriever.encode(texts, show_progress_bar=True)
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        logger.info("Enhanced RAG index built successfully!")

        # Train your medical model on the PDF content
        if train_medical_model and self.medical_model is not None:
            logger.info("Training your medical model on PDF content...")
            self._train_medical_model_on_pdf(texts)

        # Train reranker model
        if train_reranker:
            logger.info("Starting reranker model training...")
            full_text_content = "\n\n".join(texts)
            train_relevancy_model(full_text_content, model_save_path=self.reranker_model_path)
            self._load_reranker_model()
    
    def retrieve(self, question: str, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """Retrieve relevant documents for a question"""
        if self.index is None:
            return []
        
        # Step 1: Initial retrieval from FAISS
        initial_top_k = top_k * 3

        question_embedding = self.retriever.encode([question]).astype('float32')
        faiss.normalize_L2(question_embedding)
        
        scores, indices = self.index.search(question_embedding, initial_top_k)
        
        initial_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                initial_results.append((
                    self.documents[idx]['content'],
                    float(score),
                    self.doc_metadata[idx]
                ))

        # Step 2: Rerank if model is available
        if self.reranker_model:
            logger.info("Reranking retrieved documents...")
            
            rerank_scores = []
            for doc_content, _, _ in initial_results:
                with torch.no_grad():
                    relevance_scores, _ = self.reranker_model([doc_content], [question])
                    rerank_scores.append(relevance_scores.item())

            reranked_results = sorted(zip(initial_results, rerank_scores), key=lambda x: x[1], reverse=True)
            
            final_results = []
            for (doc_content, _, metadata), score in reranked_results:
                final_results.append((doc_content, score, metadata))

            return final_results[:top_k]
        else:
            return initial_results[:top_k]
    
    def _format_retrieved_docs(self, context_docs: List[Tuple[str, float, Dict]]) -> str:
        """Format retrieved documents for the prompt"""
        formatted_docs = ""
        for i, (content, score, metadata) in enumerate(context_docs, 1):
            # Clean and truncate content
            clean_content = content.strip().replace("\n", " ")[:1000]  # Limit length and remove newlines
            formatted_docs += f"Document {i} (score: {score:.2f}): {clean_content}\n"
        return formatted_docs
    
    def _extract_structured_info(self, question: str, context_docs: List[Tuple[str, float, Dict]]) -> MedicalResponse:
        """Extract structured information based on question type"""
        
        if not context_docs:
            return MedicalResponse(
                title="Information Not Found",
                bullet_points=["**No relevant information**: Could not find specific details to answer your question."],
                confidence_level="Low"
            )
        
        question_lower = question.lower()
        best_doc = context_docs[0][0]
        
        # Determine response based on question type
        if any(word in question_lower for word in ['treatment', 'treat', 'therapy', 'cure', 'manage']):
            return self._get_treatment_response(best_doc, question)
        
        elif any(word in question_lower for word in ['symptom', 'sign', 'present']):
            return self._get_symptoms_response(best_doc, question)
        
        elif any(word in question_lower for word in ['diagnos', 'test', 'detect']):
            return self._get_diagnosis_response(best_doc, question)
        
        elif any(word in question_lower for word in ['cause', 'why', 'reason']):
            return self._get_causes_response(best_doc, question)
        
        elif any(word in question_lower for word in ['benefit', 'advantage', 'help']):
            return self._get_benefits_response(best_doc, question)
        
        else:
            return self._get_general_response(best_doc, question)
    
    def _get_treatment_response(self, content: str, question: str) -> MedicalResponse:
        """Generate treatment-focused response"""
        treatments = []
        content_lower = content.lower()
        
        if 'hospital stay' in content_lower or 'hospital' in content_lower:
            treatments.append("**Hospitalization**: Treatment for AKI usually means a hospital stay for monitoring and medical management.")
        
        if 'dialysis' in content_lower:
            treatments.append("**Dialysis**: In more serious cases, dialysis may be needed to take over kidney function until recovery.")
        
        if 'treat the cause' in content_lower or 'main goal' in content_lower:
            treatments.append("**Address Root Cause**: The main goal of treatment is to identify and treat the underlying cause of AKI.")
        
        if 'recover' in content_lower or 'recovery' in content_lower:
            treatments.append("**Support Recovery**: Treatment focuses on supporting kidney recovery and preventing complications.")
        
        if 'medication' in content_lower or 'drug' in content_lower:
            treatments.append("**Medications**: Specific medications may be prescribed based on the underlying cause and severity.")
        
        if 'fluid' in content_lower:
            treatments.append("**Fluid Management**: Careful monitoring and management of fluid balance is essential.")
        
        if not treatments:
            treatments.append("**Comprehensive Care**: Medical management following established clinical protocols for kidney injury.")
        
        return MedicalResponse(
            title="AKI Treatment Options",
            summary="AKI treatment typically requires hospitalization and focuses on treating the underlying cause while supporting kidney recovery.",
            bullet_points=treatments,
            conclusion="Treatment duration depends on the cause of AKI and how quickly the kidneys recover.",
            confidence_level="High"
        )
    
    def _get_symptoms_response(self, content: str, question: str) -> MedicalResponse:
        """Generate symptoms-focused response"""
        symptoms = []
        
        if 'urine output' in content.lower() or 'urination' in content.lower():
            symptoms.append("**Decreased Urination**: Reduced urine output or frequency.")
        
        if 'swelling' in content.lower() or 'edema' in content.lower():
            symptoms.append("**Swelling**: Fluid retention causing swelling in legs, ankles, or face.")
        
        if 'fatigue' in content.lower() or 'tired' in content.lower():
            symptoms.append("**Fatigue**: Persistent tiredness and lack of energy.")
        
        if 'nausea' in content.lower() or 'vomiting' in content.lower():
            symptoms.append("**Nausea**: Feeling sick to the stomach, possibly with vomiting.")
        
        if 'confusion' in content.lower():
            symptoms.append("**Confusion**: Mental changes or difficulty concentrating.")
        
        if 'shortness of breath' in content.lower() or 'breathing' in content.lower():
            symptoms.append("**Breathing Issues**: Difficulty breathing or shortness of breath.")
        
        if not symptoms:
            symptoms.append("**Variable Symptoms**: Symptoms may vary depending on the underlying condition and severity.")
        
        return MedicalResponse(
            title="Signs and Symptoms",
            bullet_points=symptoms,
            confidence_level="High"
        )
    
    def _get_diagnosis_response(self, content: str, question: str) -> MedicalResponse:
        """Generate diagnosis-focused response"""
        tests = []
        
        if 'blood test' in content.lower() or 'creatinine' in content.lower():
            tests.append("**Blood Tests**: Measure creatinine, urea, and other kidney function markers.")
        
        if 'urine test' in content.lower():
            tests.append("**Urine Tests**: Analyze urine composition and detect abnormalities.")
        
        if 'imaging' in content.lower() or 'ultrasound' in content.lower():
            tests.append("**Imaging Studies**: Ultrasound or other scans to visualize kidney structure.")
        
        if 'biopsy' in content.lower():
            tests.append("**Kidney Biopsy**: Tissue sample analysis in certain cases.")
        
        if not tests:
            tests.append("**Standard Evaluation**: Comprehensive medical assessment including laboratory and imaging studies.")
        
        return MedicalResponse(
            title="Diagnostic Tests",
            bullet_points=tests,
            confidence_level="High"
        )
    
    def _get_causes_response(self, content: str, question: str) -> MedicalResponse:
        """Generate causes-focused response"""
        causes = []
        
        if 'blood flow' in content.lower() or 'circulation' in content.lower():
            causes.append("**Reduced Blood Flow**: Decreased circulation to the kidneys affecting function.")
        
        if 'kidney damage' in content.lower() or 'direct damage' in content.lower():
            causes.append("**Direct Kidney Damage**: Injury to kidney tissue from various factors.")
        
        if 'blockage' in content.lower() or 'obstruction' in content.lower():
            causes.append("**Urinary Obstruction**: Blockage preventing normal urine flow.")
        
        if 'infection' in content.lower():
            causes.append("**Infection**: Bacterial or other infections affecting kidney function.")
        
        if not causes:
            causes.append("**Multiple Factors**: Various medical conditions can contribute to kidney problems.")
        
        return MedicalResponse(
            title="Common Causes",
            bullet_points=causes,
            confidence_level="Moderate"
        )
    
    def _get_benefits_response(self, content: str, question: str) -> MedicalResponse:
        """Generate benefits-focused response"""
        # This method is a placeholder for content that describes benefits
        return MedicalResponse(
            title="Key Benefits",
            bullet_points=["**Improved Outcomes**: Following the recommended actions can lead to better health results.", "**Enhanced Well-being**: Can contribute to an overall sense of wellness."],
            confidence_level="Moderate"
        )
    
    def _get_general_response(self, content: str, question: str) -> MedicalResponse:
        """Generate a general response from the best document"""
        
        question_lower = question.lower()
        content_lower = content.lower()
        
        # Handle specific medical term queries like "what is AKI"
        if any(term in question_lower for term in ['what is', 'define', 'explain']):
            
            # Look for AKI or Acute Kidney Injury definition
            if 'aki' in question_lower or 'acute kidney injury' in question_lower:
                definition_points = []
                
                # Extract key information about AKI from content
                if 'acute kidney injury' in content_lower:
                    definition_points.append("**Acute Kidney Injury (AKI)**: A sudden episode of kidney failure or kidney damage that happens within a few hours or a few days.")
                
                if 'build-up of waste' in content_lower or 'waste products' in content_lower:
                    definition_points.append("**Waste Accumulation**: AKI causes a build-up of waste products in your blood.")
                
                if 'fluid balance' in content_lower or 'balance of fluid' in content_lower:
                    definition_points.append("**Fluid Balance**: Makes it hard for your kidneys to keep the right balance of fluid in your body.")
                
                if 'other organs' in content_lower or 'brain, heart' in content_lower:
                    definition_points.append("**Multi-organ Impact**: AKI can also affect other organs such as the brain, heart, and lungs.")
                
                if 'hospital' in content_lower or 'intensive care' in content_lower:
                    definition_points.append("**Common in Hospitals**: AKI is common in patients who are in the hospital, in intensive care units, and especially in older adults.")
                
                if definition_points:
                    return MedicalResponse(
                        title="Acute Kidney Injury (AKI) Definition",
                        summary="Acute Kidney Injury (AKI) is a sudden episode of kidney failure that can affect multiple organs and requires immediate medical care.",
                        bullet_points=definition_points,
                        conclusion="Early recognition and treatment of AKI are crucial for the best possible outcomes.",
                        confidence_level="High"
                    )
        
        # Handle treatment questions
        if any(term in question_lower for term in ['treatment', 'treat', 'therapy', 'manage']):
            treatment_points = []
            
            if 'hospital stay' in content_lower:
                treatment_points.append("**Hospitalization**: Treatment for AKI usually means a hospital stay for monitoring and management.")
            
            if 'dialysis' in content_lower:
                treatment_points.append("**Dialysis**: In more serious cases, dialysis may be needed to take over kidney function until recovery.")
            
            if 'treat the cause' in content_lower or 'main goal' in content_lower:
                treatment_points.append("**Treat Underlying Cause**: The main goal is to treat the cause of your AKI.")
            
            if 'recover' in content_lower:
                treatment_points.append("**Recovery Focus**: Treatment supports kidney recovery and prevents complications.")
            
            if treatment_points:
                return MedicalResponse(
                    title="AKI Treatment Approach",
                    summary="AKI treatment focuses on hospitalization, supportive care, and addressing the underlying cause.",
                    bullet_points=treatment_points,
                    conclusion="Treatment duration depends on the cause and how quickly the kidneys recover.",
                    confidence_level="High"
                )
        
        # Handle causes questions
        if any(term in question_lower for term in ['cause', 'causes', 'why']):
            cause_points = []
            
            if 'decreased blood flow' in content_lower or 'blood flow' in content_lower:
                cause_points.append("**Decreased Blood Flow**: Conditions that slow blood flow to kidneys, including low blood pressure, bleeding, heart problems.")
            
            if 'direct damage' in content_lower:
                cause_points.append("**Direct Kidney Damage**: Conditions that directly damage kidney tissue, such as sepsis, certain medications, or autoimmune diseases.")
            
            if 'blockage' in content_lower or 'urinary tract' in content_lower:
                cause_points.append("**Urinary Blockage**: Conditions that block urine flow, including kidney stones, enlarged prostate, or tumors.")
            
            if 'sepsis' in content_lower:
                cause_points.append("**Sepsis**: Severe, life-threatening infection that can damage kidneys.")
            
            if 'medication' in content_lower or 'drugs' in content_lower:
                cause_points.append("**Medications**: Overuse of certain pain medicines or allergic reactions to drugs.")
            
            if cause_points:
                return MedicalResponse(
                    title="Causes of AKI",
                    summary="AKI has three main categories of causes: decreased blood flow, direct kidney damage, and urinary blockages.",
                    bullet_points=cause_points,
                    confidence_level="High"
                )
        
        # Handle symptoms questions  
        if any(term in question_lower for term in ['symptom', 'symptoms', 'signs']):
            symptom_points = []
            
            if 'urine' in content_lower:
                symptom_points.append("**Decreased Urination**: Too little urine leaving the body.")
            
            if 'swelling' in content_lower:
                symptom_points.append("**Swelling**: Fluid retention causing swelling in legs, ankles, and around the eyes.")
            
            if 'fatigue' in content_lower or 'tiredness' in content_lower:
                symptom_points.append("**Fatigue**: Persistent tiredness and lack of energy.")
            
            if 'confusion' in content_lower:
                symptom_points.append("**Confusion**: Mental changes or difficulty concentrating.")
            
            if 'nausea' in content_lower:
                symptom_points.append("**Nausea**: Feeling sick to the stomach, possibly with vomiting.")
            
            if 'shortness of breath' in content_lower or 'breathing' in content_lower:
                symptom_points.append("**Breathing Problems**: Shortness of breath or difficulty breathing.")
            
            if 'chest pain' in content_lower:
                symptom_points.append("**Chest Pain**: Chest pain or pressure may occur.")
            
            if symptom_points:
                return MedicalResponse(
                    title="Signs and Symptoms of AKI",
                    summary="AKI symptoms vary depending on the cause and may include changes in urination, swelling, and general discomfort.",
                    bullet_points=symptom_points,
                    conclusion="In some cases, AKI causes no symptoms and is only found through medical tests.",
                    confidence_level="High"
                )
        
        # Fallback to extracting key sentences from content
        sentences = content.split('.')
        summary_points = []
        
        for i, sentence in enumerate(sentences[:3]):
            sentence = sentence.strip()
            if len(sentence) > 20:
                # Clean up the sentence
                clean_sentence = sentence.replace('\n', ' ').replace('  ', ' ')
                if not clean_sentence.endswith('.'):
                    clean_sentence += '.'
                summary_points.append(f"**Point {i+1}**: {clean_sentence}")
        
        if not summary_points:
            summary_points = [f"**Available Information**: {content[:300]}..."]
        
        return MedicalResponse(
            title="Medical Information", 
            bullet_points=summary_points,
            confidence_level="Moderate"
        )
    
    def generate_answer(self, question: str, context_docs: List[Tuple[str, float, Dict]]) -> str:
        """Generate answer using retrieved context and your custom medical model."""
        
        if not context_docs:
            return "I couldn't find relevant information to answer your question. Please try rephrasing your question."

        # Use structured extraction as the primary method
        structured_response = self._extract_structured_info(question, context_docs)
        
        # Set confidence based on retrieval scores
        if context_docs:
            avg_score = sum(score for _, score, _ in context_docs) / len(context_docs)
            if avg_score > 0.5:
                structured_response.confidence_level = "High"
            elif avg_score > 0.3:
                structured_response.confidence_level = "Moderate"
            else:
                structured_response.confidence_level = "Low"

        # If your medical model is available, use it to enhance the response
        if self.medical_model is not None:
            try:
                # Prepare context for your medical model
                context_text = " ".join([doc[0] for doc in context_docs[:3]])  # Use top 3 docs
                
                # Use your medical model for Q&A
                with torch.no_grad():
                    # Check if your model has specific Q&A methods
                    if hasattr(self.medical_model, 'answer_question'):
                        model_answer = self.medical_model.answer_question(question, context_text)
                        if model_answer and len(model_answer.strip()) > 10:
                            structured_response.summary = model_answer[:300] + ("..." if len(model_answer) > 300 else "")
                    elif hasattr(self.medical_model, 'forward'):
                        # Use forward method with proper input preparation
                        logger.info("Using medical model forward method")
                        
                        # Try to prepare inputs for your model
                        # This depends on your model's input format - you may need to adjust
                        try:
                            # Encode question and context using sentence transformer
                            question_emb = self.retriever.encode([question])
                            context_emb = self.retriever.encode([context_text])
                            
                            # Convert to tensors
                            question_tensor = torch.FloatTensor(question_emb).to(self.device)
                            context_tensor = torch.FloatTensor(context_emb).to(self.device)
                            
                            # Get model output (adjust based on your model's interface)
                            output = self.medical_model(question_tensor, context_tensor)
                            
                            # Process output based on your model's output format
                            if hasattr(output, 'logits'):
                                # If output has logits, use them for confidence
                                confidence = torch.softmax(output.logits, dim=-1).max().item()
                                if confidence > 0.7:
                                    structured_response.confidence_level = "High"
                                elif confidence > 0.5:
                                    structured_response.confidence_level = "Moderate"
                            
                        except Exception as model_error:
                            logger.warning(f"Medical model forward failed: {model_error}")
                    
            except Exception as e:
                logger.warning(f"Medical model generation failed, using structured extraction only: {e}")

        return structured_response.to_formatted_string()

    def ask(self, question: str, top_k: int = 3) -> Dict:
        """Ask a question and get a structured response dictionary with online learning"""
        
        # Retrieve relevant documents
        context_docs = self.retrieve(question, top_k)
        
        # Generate structured answer
        answer = self.generate_answer(question, context_docs)
        
        # Calculate confidence for online learning
        confidence = 0.5  # Default confidence
        if context_docs:
            avg_score = sum(score for _, score, _ in context_docs) / len(context_docs)
            confidence = min(avg_score, 1.0)
        
        # Capture Q&A pair for online learning
        self._capture_qa_pair(question, answer, context_docs, confidence)
        
        # Return structured response
        return {
            'question': question,
            'answer': answer,
            'context': context_docs,
            'confidence': confidence,
            'online_learning_enabled': self.enable_online_learning
        }
    
    def ask_with_system_prompt(self, question: str, top_k: int = 3) -> Dict:
        """Alternative method using your medical model for enhanced generation with online learning"""
        
        # Retrieve relevant documents
        context_docs = self.retrieve(question, top_k)
        
        if not context_docs:
            no_result_answer = "### Information Not Available\n\n- **No Results**: Could not find relevant information to answer your question."
            
            # Still capture for online learning with low confidence
            self._capture_qa_pair(question, no_result_answer, [], 0.1)
            
            return {
                'answer': no_result_answer,
                'confidence': 0.0,
                'sources': 0,
                'retrieved_docs': [],
                'formatted': True,
                'context': [],
                'online_learning_enabled': self.enable_online_learning
            }
        
        # Generate answer using your medical model
        answer = self.generate_answer(question, context_docs)
        
        # Calculate confidence
        confidence = context_docs[0][1] if context_docs else 0.0
        
        # Capture Q&A pair for online learning
        self._capture_qa_pair(question, answer, context_docs, confidence)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'sources': len(context_docs),
            'retrieved_docs': context_docs,
            'formatted': True,
            'context': context_docs,
            'online_learning_enabled': self.enable_online_learning
        }
    
    def save_index(self, path: str):
        """Save the RAG index"""
        if self.index is not None:
            faiss.write_index(self.index, f"{path}_enhanced_rag.faiss")
            
            with open(f"{path}_enhanced_rag_data.json", 'w') as f:
                json.dump({
                    'documents': self.documents,
                    'metadata': self.doc_metadata
                }, f, indent=2)
            
            logger.info(f"Enhanced RAG index saved to {path}")
    
    def load_index(self, path: str):
        """Load the RAG index"""
        try:
            self.index = faiss.read_index(f"{path}_enhanced_rag.faiss")
            
            with open(f"{path}_enhanced_rag_data.json", 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.doc_metadata = data['metadata']
            
            logger.info(f"Enhanced RAG index loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load enhanced RAG index: {e}")

    def _train_medical_model_on_pdf(self, texts: List[str]):
        """Train your medical model on PDF content using your existing training pipeline"""
        try:
            # Import your training modules
            from training.pdf_training import MedicalQATrainer, prepare_training_data
            
            logger.info("Preparing PDF content for medical model training...")
            
            # Prepare training data from PDF content
            training_data = self._prepare_pdf_training_data(texts)
            
            if len(training_data) > 0:
                logger.info(f"Generated {len(training_data)} training examples from PDF content")
                
                # Initialize your medical training pipeline
                trainer = MedicalQATrainer(
                    model=self.medical_model,
                    device=self.device,
                    learning_rate=1e-4,
                    batch_size=8,
                    max_epochs=3
                )
                
                # Train the model on PDF content
                trainer.train(training_data)
                
                # Save the updated model
                checkpoint = {
                    'model_state_dict': self.medical_model.state_dict(),
                    'config': {
                        'model_type': 'medical',
                        'hidden_dim': getattr(self.medical_model, 'hidden_dim', 384)
                    }
                }
                torch.save(checkpoint, self.medical_model_path)
                logger.info(f"Updated medical model saved to {self.medical_model_path}")
                
            else:
                logger.warning("No training data generated from PDF content")
                
        except Exception as e:
            logger.error(f"Failed to train medical model on PDF content: {e}")
    
    def _prepare_pdf_training_data(self, texts: List[str]) -> List[Dict]:
        """Prepare training data from PDF content for your medical model"""
        training_data = []
        
        try:
            # Generate question-answer pairs from PDF content
            for i, text in enumerate(texts):
                text_lower = text.lower()
                
                # Generate questions and answers based on content patterns
                qa_pairs = []
                
                # For AKI definition content
                if 'acute kidney injury' in text_lower and 'sudden' in text_lower:
                    qa_pairs.extend([
                        {
                            'question': 'What is AKI?',
                            'answer': 'AKI (Acute Kidney Injury) is a sudden episode of kidney failure or kidney damage that happens within a few hours or a few days.',
                            'context': text
                        },
                        {
                            'question': 'What does AKI stand for?',
                            'answer': 'AKI stands for Acute Kidney Injury.',
                            'context': text
                        }
                    ])
                
                # For symptoms content
                if any(symptom in text_lower for symptom in ['symptom', 'sign', 'urine output', 'swelling', 'fatigue']):
                    qa_pairs.extend([
                        {
                            'question': 'What are the symptoms of AKI?',
                            'answer': 'Signs and symptoms of AKI include decreased urine output, swelling in legs and ankles, fatigue, confusion, nausea, and shortness of breath.',
                            'context': text
                        },
                        {
                            'question': 'What are the signs of acute kidney injury?',
                            'answer': 'Signs include too little urine leaving the body, swelling in legs, ankles, and around the eyes, fatigue, shortness of breath, confusion, and nausea.',
                            'context': text
                        }
                    ])
                
                # For treatment content
                if any(treatment in text_lower for treatment in ['treatment', 'hospital', 'dialysis', 'recover']):
                    qa_pairs.extend([
                        {
                            'question': 'How is AKI treated?',
                            'answer': 'Treatment for AKI usually means a hospital stay. In more serious cases, dialysis may be needed to take over kidney function until the kidneys recover.',
                            'context': text
                        },
                        {
                            'question': 'What is the treatment for acute kidney injury?',
                            'answer': 'The main goal is to treat the cause of AKI. Treatment typically requires hospitalization and may include dialysis in severe cases.',
                            'context': text
                        }
                    ])
                
                # For causes content
                if any(cause in text_lower for cause in ['cause', 'blood flow', 'blockage', 'damage']):
                    qa_pairs.extend([
                        {
                            'question': 'What causes AKI?',
                            'answer': 'AKI can be caused by decreased blood flow to kidneys, direct kidney damage, or blockage of the urinary tract.',
                            'context': text
                        },
                        {
                            'question': 'What are the main causes of acute kidney injury?',
                            'answer': 'Main causes include low blood pressure, blood loss, heart problems, infections, certain medications, and urinary tract blockages.',
                            'context': text
                        }
                    ])
                
                # For diagnosis content
                if any(diag in text_lower for diag in ['diagnos', 'test', 'blood test', 'urine test', 'creatinine']):
                    qa_pairs.extend([
                        {
                            'question': 'How is AKI diagnosed?',
                            'answer': 'AKI is diagnosed through blood tests checking creatinine levels, urine tests, imaging studies like ultrasound, and monitoring urine output.',
                            'context': text
                        },
                        {
                            'question': 'What tests are used to diagnose acute kidney injury?',
                            'answer': 'Tests include measuring urine output, blood tests for creatinine and urea, urinalysis, eGFR, and imaging tests.',
                            'context': text
                        }
                    ])
                
                # Add all generated Q&A pairs
                for qa in qa_pairs:
                    training_data.append({
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'context': qa['context'],
                        'source': f'pdf_chunk_{i}',
                        'relevance_score': 1.0  # High relevance since generated from content
                    })
            
            # Add some negative examples for better training
            negative_questions = [
                "What is diabetes?",
                "How to treat cancer?",
                "What are symptoms of flu?",
                "How to prevent heart disease?"
            ]
            
            for neg_q in negative_questions:
                training_data.append({
                    'question': neg_q,
                    'answer': "This question is not related to the provided medical document about AKI.",
                    'context': texts[0] if texts else "",
                    'source': 'negative_example',
                    'relevance_score': 0.0
                })
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing PDF training data: {e}")
            return []

    def enable_online_learning(self):
        """Enable online learning to capture new Q&A pairs and retrain the model"""
        self.enable_online_learning = True
        logger.info("Online learning enabled. The model will be updated with new Q&A pairs.")
    
    def disable_online_learning(self):
        """Disable online learning"""
        self.enable_online_learning = False
        logger.info("Online learning disabled. The model will not be updated with new Q&A pairs.")
    
    def _start_online_learning(self):
        """Start the background thread for online learning"""
        if self.online_training_thread is None or not self.online_training_thread.is_alive():
            self.stop_online_training.clear()
            self.online_training_thread = threading.Thread(
                target=self._online_learning_worker,
                daemon=True
            )
            self.online_training_thread.start()
            logger.info("Online learning worker thread started")
    
    def _stop_online_learning(self):
        """Stop the background online learning thread"""
        if self.online_training_thread and self.online_training_thread.is_alive():
            self.stop_online_training.set()
            self.online_training_thread.join(timeout=10)
            logger.info("Online learning worker thread stopped")
    
    def _online_learning_worker(self):
        """Background worker that periodically trains on accumulated Q&A pairs"""
        logger.info("Online learning worker started - monitoring for new Q&A pairs")
        
        while not self.stop_online_training.wait(self.online_learning_interval):
            try:
                # Check if we have enough Q&A pairs to train on
                with self.qa_queue_lock:
                    queue_size = len(self.qa_queue)
                
                if queue_size >= self.online_learning_batch_size:
                    logger.info(f"Starting online training on {queue_size} Q&A pairs")
                    self._perform_online_training()
                else:
                    logger.debug(f"Online learning: {queue_size}/{self.online_learning_batch_size} Q&A pairs collected")
                    
            except Exception as e:
                logger.error(f"Error in online learning worker: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _perform_online_training(self):
        """Perform online training on accumulated Q&A pairs"""
        try:
            # Extract training data from queue
            with self.qa_queue_lock:
                training_pairs = list(self.qa_queue)
                # Keep some pairs for future training, remove processed ones
                processed_count = min(self.online_learning_batch_size, len(training_pairs))
                for _ in range(processed_count):
                    self.qa_queue.popleft()
            
            if not training_pairs:
                return
            
            logger.info(f"Online training: Processing {len(training_pairs)} Q&A pairs")
            
            # Prepare training data for your medical model
            training_data = []
            for qa_pair in training_pairs:
                training_data.append({
                    'question': qa_pair['question'],
                    'answer': qa_pair['answer'],
                    'context': qa_pair.get('context', ''),
                    'source': 'online_learning',
                    'relevance_score': qa_pair.get('confidence', 0.5),
                    'timestamp': qa_pair.get('timestamp', datetime.now().isoformat())
                })
            
            # Import and use your training pipeline
            from training.pdf_training import MedicalQATrainer
            
            # Create a copy of the model for training to avoid disrupting ongoing inference
            model_for_training = create_model('medical', hidden_dim=getattr(self.medical_model, 'hidden_dim', 384))
            model_for_training.load_state_dict(self.medical_model.state_dict())
            model_for_training.to(self.device)
            
            # Initialize trainer with reduced learning rate for online learning
            trainer = MedicalQATrainer(
                model=model_for_training,
                device=self.device,
                learning_rate=5e-5,  # Lower learning rate for online learning
                batch_size=4,        # Smaller batch size
                max_epochs=1         # Single epoch for online learning
            )
            
            # Train on the new Q&A pairs
            trainer.train(training_data)
            
            # Update the main model with the trained weights
            self.medical_model.load_state_dict(model_for_training.state_dict())
            
            # Save the updated model
            checkpoint = {
                'model_state_dict': self.medical_model.state_dict(),
                'config': {
                    'model_type': 'medical',
                    'hidden_dim': getattr(self.medical_model, 'hidden_dim', 384)
                },
                'online_training_history': {
                    'last_update': datetime.now().isoformat(),
                    'training_pairs_processed': len(training_pairs),
                    'total_online_updates': getattr(self, 'online_update_count', 0) + 1
                }
            }
            
            # Update counter
            self.online_update_count = getattr(self, 'online_update_count', 0) + 1
            
            torch.save(checkpoint, self.medical_model_path)
            logger.info(f"Online training completed - Model updated with {len(training_pairs)} new Q&A pairs")
            logger.info(f"Total online updates: {self.online_update_count}")
            
        except Exception as e:
            logger.error(f"Online training failed: {e}")
    
    def _capture_qa_pair(self, question: str, answer: str, context_docs: List[Tuple[str, float, Dict]], confidence: float):
        """Capture a Q&A pair for online learning"""
        if not self.enable_online_learning:
            return
        
        try:
            # Create context string from retrieved documents
            context_text = ""
            if context_docs:
                context_text = " ".join([doc[0][:500] for doc in context_docs[:2]])  # Use top 2 docs, truncated
            
            qa_pair = {
                'question': question,
                'answer': answer,
                'context': context_text,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'retrieval_score': context_docs[0][1] if context_docs else 0.0
            }
            
            # Add to queue in a thread-safe manner
            with self.qa_queue_lock:
                self.qa_queue.append(qa_pair)
                queue_size = len(self.qa_queue)
            
            # Track confidence for improvement measurement
            self.confidence_history.append(confidence)
            
            logger.debug(f"Q&A pair captured for online learning (queue size: {queue_size})")
            
        except Exception as e:
            logger.error(f"Failed to capture Q&A pair: {e}")
    
    def get_online_learning_stats(self) -> Dict:
        """Get statistics about online learning progress"""
        with self.qa_queue_lock:
            queue_size = len(self.qa_queue)
        
        avg_confidence = 0.0
        if self.confidence_history:
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        
        recent_confidence = 0.0
        if len(self.confidence_history) >= 10:
            recent_confidence = sum(list(self.confidence_history)[-10:]) / 10
        
        return {
            'online_learning_enabled': self.enable_online_learning,
            'queued_qa_pairs': queue_size,
            'batch_size': self.online_learning_batch_size,
            'training_interval_seconds': self.online_learning_interval,
            'total_online_updates': getattr(self, 'online_update_count', 0),
            'average_confidence': round(avg_confidence, 3),
            'recent_confidence': round(recent_confidence, 3),
            'confidence_improvement': round(recent_confidence - avg_confidence, 3) if len(self.confidence_history) >= 10 else 0.0,
            'worker_thread_active': self.online_training_thread.is_alive() if self.online_training_thread else False
        }
    
    def shutdown(self):
        """Clean shutdown of the RAG system"""
        logger.info("Shutting down Enhanced RAG system...")
        
        # Stop online learning
        if self.enable_online_learning:
            self._stop_online_learning()
        
        # Perform final training if there are pending Q&A pairs
        with self.qa_queue_lock:
            if len(self.qa_queue) > 0:
                logger.info(f"Performing final online training on {len(self.qa_queue)} pending Q&A pairs")
                self._perform_online_training()
        
        logger.info("Enhanced RAG system shutdown complete")
    
    def __del__(self):
        """Destructor to ensure clean shutdown"""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup

    def add_qa_pair(self, question: str, answer: str, confidence: float = 0.5, context_docs: List[Tuple[str, float, Dict]] = None):
        """
        Public method to add a Q&A pair to the online learning queue
        
        Args:
            question: The user's question
            answer: The system's response
            confidence: Confidence score of the response (0.0 to 1.0)
            context_docs: Retrieved documents used for the answer
        """
        if context_docs is None:
            context_docs = []
        
        self._capture_qa_pair(question, answer, context_docs, confidence)
        
        logger.debug(f"Q&A pair added to online learning queue: {question[:50]}...")

    def get_training_queue_size(self) -> int:
        """Get the current size of the training queue"""
        with self.qa_queue_lock:
            return len(self.qa_queue)

    def force_online_training(self):
        """Force immediate online training on current queue (for testing/debugging)"""
        if not self.enable_online_learning:
            logger.warning("Online learning is disabled")
            return False
        
        with self.qa_queue_lock:
            queue_size = len(self.qa_queue)
        
        if queue_size == 0:
            logger.info("No Q&A pairs in queue for training")
            return False
        
        logger.info(f"Forcing online training on {queue_size} Q&A pairs")
        self._perform_online_training()
        return True
