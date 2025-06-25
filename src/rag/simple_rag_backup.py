#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) System
Clean implementation for medical document Q&A
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class SimpleRAG:
    """
    Simple but effective RAG system for medical Q&A
    """
    
    def __init__(self, 
                 retrieval_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 generation_model: str = "microsoft/DialoGPT-medium"):
        
        logger.info("Initializing Simple RAG system...")
        
        # Initialize retrieval components
        self.retriever = SentenceTransformer(retrieval_model)
        self.index = None
        self.documents = []
        self.doc_metadata = []
        
        # Initialize generation components
        try:
            # Try to load a simple conversation model
            self.generator = pipeline(
                "text-generation",
                model=generation_model,
                tokenizer=generation_model,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256  # GPT-2 pad token
            )
            self.generation_available = True
            logger.info(f"Loaded generation model: {generation_model}")
        except Exception as e:
            logger.warning(f"Could not load generation model: {e}")
            logger.info("Falling back to template-based responses")
            self.generator = None
            self.generation_available = False
    
    def build_index(self, documents: List[Dict]):
        """Build FAISS index from documents"""
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
        
        logger.info("RAG index built successfully!")
    
    def retrieve(self, question: str, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """Retrieve relevant documents for a question"""
        if self.index is None:
            return []
        
        # Encode question
        question_embedding = self.retriever.encode([question]).astype('float32')
        faiss.normalize_L2(question_embedding)
        
        # Search
        scores, indices = self.index.search(question_embedding, top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx]['content'],
                    float(score),
                    self.doc_metadata[idx]                ))
        
        return results
    
    def generate_answer(self, question: str, context_docs: List[Tuple[str, float, Dict]]) -> str:
        """Generate answer using retrieved context"""
        
        if not context_docs:
            return "I couldn't find relevant information to answer your question. Please try rephrasing your question."
        
        # Always use template-based response for better reliability
        return self._template_based_answer(question, context_docs)
    
    def _template_based_answer(self, question: str, context_docs: List[Tuple[str, float, Dict]]) -> str:
        """Generate dynamic template-based answer for any medical content with structured format"""
        
        best_doc, best_score, best_metadata = context_docs[0]
        
        # Analyze question type and extract key medical topics from content
        question_lower = question.lower()
        content_analysis = self._analyze_content_and_question(best_doc, question)
        
        # Determine question type dynamically
        answer_type = self._detect_question_type(question_lower)
        
        # Generate dynamic introduction based on content analysis
        intro = self._generate_dynamic_intro(question, answer_type, content_analysis)
        
        # Generate key points based on actual content
        key_points = self._generate_dynamic_key_points(content_analysis, answer_type, context_docs, question)
        
        # Combine into structured format
        answer_parts = [intro, "", "**Key Points:**"]
        answer_parts.extend(key_points)
        
        # Add confidence indicator
        if best_score > 0.6:
            confidence_text = "High confidence"
        elif best_score > 0.4:
            confidence_text = "Moderate confidence"
        else:
            confidence_text = "Low confidence"
        
        answer_parts.append("")
        answer_parts.append(f"*Confidence: {confidence_text} ({best_score:.2f})*")
        
        return "\n".join(answer_parts)

    def _analyze_content_and_question(self, content: str, question: str) -> Dict:
        """Analyze content to identify medical topics, conditions, and relevant information"""
        clean_content = self._clean_medical_text(content)
        question_lower = question.lower()
        
        analysis = {
            'medical_condition': None,
            'main_topics': [],
            'definitions': [],
            'symptoms': [],
            'causes': [],
            'treatments': [],
            'diagnostic_info': [],
            'risk_factors': [],
            'prevention': [],
            'key_terms': []
        }
        
        # Extract medical condition from question or content
        medical_conditions = [
            'acute kidney injury', 'aki', 'kidney disease', 'renal disease', 'chronic kidney disease',
            'diabetes', 'hypertension', 'heart failure', 'stroke', 'cancer', 'infection',
            'pneumonia', 'asthma', 'copd', 'arthritis', 'depression', 'anxiety'
        ]
        
        for condition in medical_conditions:
            if condition in question_lower or condition in clean_content.lower():
                analysis['medical_condition'] = condition.upper() if condition == 'aki' else condition.title()
                break
        
        # If content is too garbled, use fallback knowledge
        if self._is_content_too_garbled(clean_content):
            return self._get_fallback_analysis(question_lower, analysis['medical_condition'])
          # Extract sentences and categorize them
        sentences = [s.strip() for s in clean_content.split('.') if s.strip() and len(s.strip()) > 10]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Skip irrelevant content
            if any(skip in sentence_lower for skip in ['page', 'updated', 'copyright', 'please consult', 'toll-free', 'kidney.org']):
                continue
            
            # Clean the sentence before categorizing
            clean_sentence = self._clean_and_format_sentence(sentence)
            if not clean_sentence:
                continue
            
            # More flexible categorization to catch actual content
            if any(term in sentence_lower for term in ['definition', 'refers to', 'is defined as', 'means', 'is a condition', 'is a sudden episode', 'aki is']):
                analysis['definitions'].append(clean_sentence)
            elif any(term in sentence_lower for term in ['signs and symptoms', 'symptom', 'sign', 'may include', 'experience', 'feel']):
                # Extract specific symptoms from lists
                if 'may include' in sentence_lower:
                    # Extract the symptoms list after "may include"
                    symptoms_part = sentence[sentence_lower.find('may include') + len('may include'):].strip()
                    # Split on common delimiters and clean
                    symptom_items = [item.strip() for item in symptoms_part.replace(' and ', ',').split(',') if item.strip()]
                    for item in symptom_items[:8]:  # Limit to avoid too many
                        if len(item) > 5 and len(item) < 100:
                            analysis['symptoms'].append(item.strip(' :.'))
                else:
                    analysis['symptoms'].append(clean_sentence)
            elif any(term in sentence_lower for term in ['cause', 'caused by', 'due to', 'result from', 'lead to', 'including:']):
                # Handle cause lists better
                if 'including:' in sentence_lower or 'such as:' in sentence_lower:
                    causes_part = sentence[max(sentence_lower.find('including:'), sentence_lower.find('such as:')) + 10:].strip()
                    cause_items = [item.strip() for item in causes_part.split(',') if item.strip()]
                    for item in cause_items[:6]:
                        if len(item) > 5 and len(item) < 150:
                            analysis['causes'].append(item.strip(' :.'))
                else:
                    analysis['causes'].append(clean_sentence)
            elif any(term in sentence_lower for term in ['treatment', 'therapy', 'manage', 'medication', 'surgery', 'dialysis', 'hospital stay']):
                analysis['treatments'].append(clean_sentence)
            elif any(term in sentence_lower for term in ['diagnos', 'test', 'blood test', 'urine test', 'imaging', 'measuring']):
                analysis['diagnostic_info'].append(clean_sentence)
            elif any(term in sentence_lower for term in ['risk factor', 'increase risk', 'more likely', 'higher risk']):
                analysis['risk_factors'].append(clean_sentence)
            elif any(term in sentence_lower for term in ['prevent', 'avoid', 'reduce risk', 'protection', 'lower your chances']):
                analysis['prevention'].append(clean_sentence)
        
        # Extract key medical terms
        medical_terms = []
        import re
        # Look for medical terminology patterns
        medical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms like AKI, BUN, GFR
            r'\b\w+itis\b',    # Conditions ending in -itis
            r'\b\w+osis\b',    # Conditions ending in -osis
            r'\b\w+emia\b',    # Blood conditions ending in -emia
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, clean_content)
            medical_terms.extend([term for term in matches if len(term) > 2])
        
        analysis['key_terms'] = list(set(medical_terms[:10]))  # Limit to top 10 unique terms
        
        return analysis
    
    def _detect_question_type(self, question_lower: str) -> str:
        """Dynamically detect question type from the question text"""
        # Check for causes first (more specific)
        if any(word in question_lower for word in ['cause', 'causes', 'why', 'reason', 'what causes', 'due to', 'how does', 'why does']):
            return "causes"
        # Check for symptoms
        elif any(word in question_lower for word in ['symptom', 'symptoms', 'sign', 'signs', 'present', 'feel', 'experience', 'how does it feel']):
            return "symptoms"
        # Check for treatment
        elif any(word in question_lower for word in ['treat', 'treatment', 'therapy', 'cure', 'management', 'medicine', 'how to treat']):
            return "treatment"
        # Check for diagnosis
        elif any(word in question_lower for word in ['diagnos', 'diagnosis', 'test', 'testing', 'detect', 'how to', 'check', 'identify']):
            return "diagnosis"
        # Check for prevention
        elif any(word in question_lower for word in ['prevent', 'prevention', 'avoid', 'stop', 'reduce risk', 'protection']):
            return "prevention"
        # Check for definition (most general)
        elif any(word in question_lower for word in ['what is', 'define', 'definition', 'what are', 'what does', 'explain', 'tell me about']):
            return "definition"
        else:
            return "general"

    def _generate_dynamic_intro(self, question: str, answer_type: str, content_analysis: Dict) -> str:
        """Generate dynamic introduction based on content analysis"""
        condition = content_analysis.get('medical_condition', 'this medical condition')
        
        if answer_type == "definition":
            return f"{condition} is a medical condition that requires understanding. Based on the available information, here's what you need to know."
        elif answer_type == "symptoms":
            return f"The symptoms of {condition} can vary between individuals and may develop at different rates. Here are the key signs to watch for."
        elif answer_type == "causes":
            return f"{condition} can result from various factors. Understanding these causes is important for prevention and treatment."
        elif answer_type == "diagnosis":
            return f"Diagnosing {condition} involves several medical tests and evaluations. Early and accurate diagnosis is crucial for effective treatment."
        elif answer_type == "treatment":
            return f"Treatment for {condition} depends on various factors including severity, underlying causes, and individual patient characteristics."
        elif answer_type == "prevention":
            return f"Preventing {condition} involves understanding risk factors and taking appropriate preventive measures."
        else:
            return f"Based on the available medical information about {condition}, here's what you need to know."

    def _generate_dynamic_key_points(self, content_analysis: Dict, answer_type: str, context_docs: List, question: str) -> List[str]:
        """Generate key points dynamically based on content analysis and question type"""
        points = []
        condition = content_analysis.get('medical_condition', 'Medical Condition')
        
        # Always include a definition if available
        if content_analysis['definitions']:
            definition = self._clean_and_format_sentence(content_analysis['definitions'][0])
            points.append(f"**Definition:** {definition}")
        elif condition != 'Medical Condition':
            points.append(f"**About {condition}:** A medical condition requiring attention and proper management.")
        
        # Add content based on question type and available information
        if answer_type == "symptoms" and content_analysis['symptoms']:
            points.append("**Common Symptoms:**")
            for symptom in content_analysis['symptoms'][:5]:
                clean_symptom = self._clean_and_format_sentence(symptom)
                if clean_symptom:
                    points.append(f"  • {clean_symptom}")
        
        elif answer_type == "causes" and content_analysis['causes']:
            points.append("**Main Causes:**")
            for cause in content_analysis['causes'][:5]:
                clean_cause = self._clean_and_format_sentence(cause)
                if clean_cause:
                    points.append(f"  • {clean_cause}")
        
        elif answer_type == "treatment" and content_analysis['treatments']:
            points.append("**Treatment Options:**")
            for treatment in content_analysis['treatments'][:5]:
                clean_treatment = self._clean_and_format_sentence(treatment)
                if clean_treatment:
                    points.append(f"  • {clean_treatment}")
        
        elif answer_type == "diagnosis" and content_analysis['diagnostic_info']:
            points.append("**Diagnostic Methods:**")
            for diagnostic in content_analysis['diagnostic_info'][:5]:
                clean_diagnostic = self._clean_and_format_sentence(diagnostic)
                if clean_diagnostic:
                    points.append(f"  • {clean_diagnostic}")
        
        elif answer_type == "prevention" and content_analysis['prevention']:
            points.append("**Prevention Strategies:**")
            for prevention in content_analysis['prevention'][:5]:
                clean_prevention = self._clean_and_format_sentence(prevention)
                if clean_prevention:
                    points.append(f"  • {clean_prevention}")
        
        # Add risk factors if available
        if content_analysis['risk_factors'] and answer_type in ['causes', 'general', 'definition']:
            points.append("**Risk Factors:**")
            for risk in content_analysis['risk_factors'][:3]:
                clean_risk = self._clean_and_format_sentence(risk)
                if clean_risk:
                    points.append(f"  • {clean_risk}")
        
        # Add key medical terms if relevant
        if content_analysis['key_terms'] and len(content_analysis['key_terms']) > 0:
            points.append("**Key Medical Terms:**")
            for term in content_analysis['key_terms'][:4]:
                points.append(f"  • {term}")
        
        # Add general information if we don't have specific content
        if len(points) <= 1:  # Only definition
            points.append("**Important Information:**")
            # Extract any relevant sentences from the content
            clean_content = self._clean_medical_text(context_docs[0][0])
            sentences = [s.strip() for s in clean_content.split('.') if s.strip() and len(s.strip()) > 20]
            
            relevant_sentences = []
            question_keywords = [word for word in question.lower().split() 
                               if len(word) > 3 and word not in ['what', 'how', 'why', 'when', 'where']]
            
            for sentence in sentences[:10]:  # Check first 10 sentences
                if any(keyword in sentence.lower() for keyword in question_keywords):
                    clean_sentence = self._clean_and_format_sentence(sentence)
                    if clean_sentence and len(clean_sentence) > 20:
                        relevant_sentences.append(clean_sentence)
                        if len(relevant_sentences) >= 3:
                            break            
            if relevant_sentences:
                for sentence in relevant_sentences:
                    points.append(f"  • {sentence}")
            else:
                points.append("  • Please refer to the medical documentation for specific details")
        
        return points
    
    def _clean_and_format_sentence(self, sentence: str) -> str:
        """Clean and format a sentence for display"""
        if not sentence:
            return ""
        
        # Clean the sentence
        clean = sentence.strip()
        
        # Remove common artifacts
        artifacts_to_remove = [
            'DISEASES AND CONDITIONS',
            'ACUTE KIDNEY INJURY (AKI)',
            'Updated May',
            'Page'
        ]
        
        for artifact in artifacts_to_remove:
            clean = clean.replace(artifact, '')
        
        # Fix spacing and formatting
        import re
        clean = re.sub(r'\s+', ' ', clean)
        clean = clean.strip()
        
        # More aggressive garbled text detection
        if self._is_sentence_garbled(clean):
            return ""
        
        # Fix sentence boundaries if they got merged
        clean = self._fix_merged_sentences(clean)
        
        # Take only the first coherent sentence if multiple were merged
        sentences = [s.strip() for s in clean.split('.') if s.strip()]
        if sentences:
            clean = sentences[0]
        
        # Ensure proper sentence ending
        if clean and not clean.endswith(('.', '!', '?', ':')):
            clean += '.'
        
        # Return empty if too short or nonsensical
        if len(clean) < 10 or len(clean) > 300:
            return ""
        
        # Final validation - must contain at least one medical term
        medical_indicators = ['kidney', 'renal', 'acute', 'injury', 'disease', 'condition', 
                            'symptoms', 'treatment', 'diagnosis', 'blood', 'urine', 
                            'creatinine', 'failure', 'function', 'medicine', 'medical',
                            'patient', 'clinical', 'therapy', 'chronic', 'severe']
        
        if not any(indicator in clean.lower() for indicator in medical_indicators):
            return ""
        
        return clean
    
    def _is_sentence_garbled(self, text: str) -> bool:
        """Enhanced check for garbled sentences"""
        if not text or len(text) < 10:
            return True
        
        # Check for excessive capitalization (PDF artifact)
        words = text.split()
        if len(words) == 0:
            return True
        
        uppercase_ratio = sum(1 for word in words if word.isupper()) / len(words)
        if uppercase_ratio > 0.6:
            return True
        
        # Check for disconnected fragments
        if any(pattern in text.upper() for pattern in [
            'FLOW DIRECT DAMAGE', 'BLOCKAGE OF THE', 'DECREASED BLOOD FLOW',
            'THE THE', 'TO TO', 'AND AND', 'OR OR'
        ]):
            return True
        
        # Check for excessive short words (fragments)
        short_words = sum(1 for word in words if len(word) <= 2 and word.isalpha())
        if len(words) > 3 and short_words > len(words) * 0.4:
            return True
        
        # Check for incomplete medical phrases
        incomplete_patterns = [
            r'\bto the\b$', r'\bof the\b$', r'\bin the\b$', r'\bfor the\b$',
            r'^\w{1,3}\s', r'\s\w{1,3}$'
        ]
        
        import re
        for pattern in incomplete_patterns:
            if re.search(pattern, text.lower()):
                return True
        
        return False
    
    def _is_garbled_text(self, text: str) -> bool:
        """Check if text appears to be garbled from PDF extraction"""
        if not text:
            return True
        
        # Count medical terms vs random words
        medical_terms = ['kidney', 'renal', 'acute', 'injury', 'disease', 'condition', 
                        'symptoms', 'treatment', 'diagnosis', 'blood', 'urine', 
                        'creatinine', 'failure', 'function', 'medicine', 'medical']
        
        words = text.lower().split()
        if len(words) < 3:
            return True
        
        # Check for disconnected medical terms (sign of garbled text)
        medical_word_count = sum(1 for word in words if any(term in word for term in medical_terms))
        
        # If too many medical terms are crammed together, it's likely garbled
        if medical_word_count > len(words) * 0.7:
            return True
        
        # Check for common garbled patterns
        garbled_patterns = [
            r'\b\w{1,2}\s+\w{1,2}\s+\w{1,2}',  # Too many short words
            r'[A-Z][a-z]+[A-Z][a-z]+[A-Z][a-z]+',  # CamelCase pattern (merged words)
            r'\b(which|that|are|is|an|the)\s+(which|that|are|is|an|the)',  # Repeated articles
        ]
        
        import re
        for pattern in garbled_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _fix_merged_sentences(self, text: str) -> str:
        """Attempt to fix sentences that were merged during PDF extraction"""
        import re
        
        # Split on common merge patterns
        # Pattern: word + medical term starting with capital (likely new sentence)
        text = re.sub(r'(\w[a-z])\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', r'\1. \2', text)
        
        # Fix specific medical term boundaries
        medical_boundaries = [
            (r'(\w)\s+(Acute kidney)', r'\1. \2'),
            (r'(\w)\s+(Kidney disease)', r'\1. \2'),
            (r'(\w)\s+(Symptoms)', r'\1. \2'),
            (r'(\w)\s+(Treatment)', r'\1. \2'),
            (r'(\w)\s+(Diagnosis)', r'\1. \2'),
        ]
        
        for pattern, replacement in medical_boundaries:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _extract_relevant_content(self, text: str, question: str) -> str:
        """Extract the most relevant part of the text for the question"""
        # Clean the text first
        clean_text = self._clean_medical_text(text)
        
        # Look for specific medical conditions mentioned in the question
        question_lower = question.lower()
        
        # Split text into sentences
        sentences = [s.strip() for s in clean_text.split('.') if s.strip() and len(s.strip()) > 20]
        
        # Find sentences that contain keywords from the question
        question_keywords = [word for word in question_lower.split() 
                           if len(word) > 3 and word not in ['what', 'how', 'why', 'when', 'where', 'tell', 'about']]
        
        relevant_sentences = []
        
        # Look for medical content that's not template text
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Skip template/query text
            if any(skip_phrase in sentence_lower for skip_phrase in [
                'based on your medical judgment', 'please clarify', 'progress notes',
                'none of the above', 'not applicable', 'please specify',
                'in responding to this request', 'thank you', 'updated may'
            ]):
                continue
            
            # Look for actual medical information
            if any(keyword in sentence_lower for keyword in question_keywords):
                relevant_sentences.append(sentence)
            elif any(term in sentence_lower for term in [
                'acute kidney', 'renal', 'creatinine', 'urine', 'kidney disease',
                'symptoms', 'causes', 'treatment', 'diagnosis', 'failure'
            ]):
                # Only include if it contains actual medical content
                if any(medical_term in sentence_lower for medical_term in [
                    'injury', 'disease', 'condition', 'disorder', 'syndrome',
                    'failure', 'dysfunction', 'abnormal', 'elevated', 'decreased'
                ]):
                    relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Prioritize sentences with multiple keywords
            scored_sentences = []
            for sentence in relevant_sentences:
                score = sum(1 for keyword in question_keywords if keyword in sentence.lower())
                scored_sentences.append((sentence, score))            # Sort by score and take best ones - increased to get more complete information
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentences = [s[0] for s in scored_sentences[:6]]  # Increased from 3 to 6 for better coverage
            
            result = '. '.join(best_sentences)
            # Ensure proper sentence ending
            if not result.endswith('.'):
                result += '.'
            # Return the full result without truncation to preserve complete information
            return result
        else:
            # Fallback: provide general medical information based on question type
            return self._generate_fallback_answer(question_lower)
    
    def _generate_fallback_answer(self, question_lower: str) -> str:
        """Generate a helpful fallback answer when specific content isn't found"""
        if 'acute kidney injury' in question_lower or 'aki' in question_lower:
            return ("Acute Kidney Injury (AKI) is a sudden decrease in kidney function that occurs over hours to days. "
                   "It's characterized by a rapid rise in creatinine levels and decreased urine output. "
                   "Common causes include dehydration, medications, infections, and underlying kidney disease.")
        
        elif 'kidney disease' in question_lower or 'renal disease' in question_lower:
            return ("Kidney disease refers to conditions that damage the kidneys and affect their ability to filter waste from blood. "
                   "Symptoms may include swelling, fatigue, changes in urination, and elevated creatinine levels. "
                   "Early detection and treatment are important to prevent progression.")
        
        elif 'symptoms' in question_lower and ('kidney' in question_lower or 'renal' in question_lower):
            return ("Common symptoms of kidney problems include: decreased urine output, swelling in legs/ankles/feet, "
                   "fatigue, shortness of breath, confusion, nausea, weakness, irregular heartbeat, "
                   "chest pain, and high blood pressure. Early kidney disease may have no symptoms.")
        
        elif 'causes' in question_lower and ('kidney' in question_lower or 'renal' in question_lower):
            return ("Common causes of kidney problems include: diabetes, high blood pressure, autoimmune diseases, "
                   "genetic disorders, infections, certain medications, dehydration, obstruction of urine flow, "
                   "and exposure to toxins. Age and family history are also risk factors.")
        
        elif 'diagnosis' in question_lower and ('kidney' in question_lower or 'renal' in question_lower):
            return ("Kidney disease is diagnosed through: blood tests (creatinine, BUN, GFR), urine tests (protein, blood, cells), "
                   "imaging studies (ultrasound, CT scan), and sometimes kidney biopsy. "
                   "GFR (glomerular filtration rate) is used to stage chronic kidney disease.")
        
        elif 'treatment' in question_lower and ('kidney' in question_lower or 'renal' in question_lower):
            return ("Treatment depends on the cause and stage but may include: managing underlying conditions (diabetes, hypertension), "
                   "medications to protect kidney function, dietary changes (low sodium, controlled protein), "
                   "fluid management, and in advanced cases, dialysis or kidney transplant.")
        else:
            return ("I found some relevant medical documentation, but the specific information you're looking for isn't clearly available. "
                   "Please try asking more specific questions about symptoms, causes, diagnosis, or treatment.")
    
    def _clean_medical_text(self, text: str) -> str:
        """Clean and format medical text for better readability and sentence flow."""
        import re
        
        # Remove extra whitespace and formatting artifacts
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove template placeholders and artifacts
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'_+', '', text)
        text = re.sub(r'DISEASES AND CONDITIONS.*?INJURY \(AKI\)', '', text)
        
        # Remove common PDF extraction artifacts
        artifacts = [
            'Updated May', 'Page \d+', 'ACUTE KIDNEY INJURY', 'DISEASES AND CONDITIONS',
            'Figure \d+', 'Table \d+', 'References', 'Bibliography'
        ]
        for artifact in artifacts:
            text = re.sub(artifact, '', text, flags=re.IGNORECASE)
        
        # Fix broken words split across lines (common in PDFs)
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        
        # Fix spacing issues around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])(\w)', r'\1 \2', text)
        
        # More aggressive approach for garbled content
        # Remove obvious fragments and malformed patterns
        text = re.sub(r'\b[A-Z]{3,}\s+[A-Z]{3,}\s+[A-Z]{3,}', '', text)  # Remove chains of caps
        text = re.sub(r'\bTO THE\b|\bOF THE\b|\bIN THE\b|\bFOR THE\b', '', text)  # Remove fragment words
        text = re.sub(r'\bTHE THE\b|\bAND AND\b|\bOR OR\b', '', text)  # Remove duplicated words
        
        # Fix sentence boundaries that got merged incorrectly
        text = re.sub(r'(\w[a-z])([A-Z][a-z])', r'\1. \2', text)
        
        # Fix cases where periods are missing between sentences
        text = re.sub(r'(\w)\s+([A-Z][a-z].*?[a-z])\s+([A-Z][a-z])', r'\1. \2. \3', text)
        
        # Clean up multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Remove page references and dates
        text = re.sub(r'Updated.*?\d{4}.*?page.*?\d+', '', text)
        text = re.sub(r'\d{1,2}\s*\d{1,2}\s*\d{1,2}', '', text)
        
        # Remove repeated words (artifact of bad PDF extraction)
        text = re.sub(r'\b(\w+) \1\b', r'\1', text)
        
        # Remove excessive single letters and fix common artifacts
        text = re.sub(r'\b[a-zA-Z]\b(?!\s[a-zA-Z]\b)', '', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common medical text artifacts
        text = text.replace('AKI AKI', 'AKI')
        text = text.replace('kidney kidney', 'kidney')
        text = text.replace('disease disease', 'disease')
        
        return text.strip()
    
    def _format_answer(self, answer: str, context_docs: List[Tuple[str, float, Dict]]) -> str:
        """Format the generated answer with confidence and sources"""
        
        # Clean up the answer
        answer = answer.strip()
        if not answer:
            return self._template_based_answer("", context_docs)
        
        # Add source information
        best_score = context_docs[0][1] if context_docs else 0.0
        
        formatted_answer = answer
        if not answer.endswith('.'):
            formatted_answer += "."
        
        formatted_answer += f"\n\n*Based on medical documentation (confidence: {best_score:.2f})*"
        
        return formatted_answer
    
    def ask(self, question: str, top_k: int = 3) -> Dict:
        """Main RAG pipeline: retrieve + generate"""
        
        # Retrieve relevant documents
        context_docs = self.retrieve(question, top_k)
        
        # Generate answer
        answer = self.generate_answer(question, context_docs)
        
        # Return structured response
        return {
            'answer': answer,
            'confidence': context_docs[0][1] if context_docs else 0.0,
            'sources': len(context_docs),
            'retrieved_docs': context_docs
        }
    
    def save_index(self, path: str):
        """Save the RAG index"""
        if self.index is not None:
            faiss.write_index(self.index, f"{path}_rag.faiss")
            
            with open(f"{path}_rag_data.json", 'w') as f:
                json.dump({
                    'documents': self.documents,
                    'metadata': self.doc_metadata
                }, f, indent=2)
            
            logger.info(f"RAG index saved to {path}")
    
    def load_index(self, path: str):
        """Load the RAG index"""
        try:
            self.index = faiss.read_index(f"{path}_rag.faiss")
            
            with open(f"{path}_rag_data.json", 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.doc_metadata = data['metadata']
            
            logger.info(f"RAG index loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load RAG index: {e}")
    
    def _extract_complementary_content(self, text: str, question: str, main_content: str) -> str:
        """Extract content that complements the main content without duplication"""
        # Clean the text first
        clean_text = self._clean_medical_text(text)
        
        # Split text into sentences
        sentences = [s.strip() for s in clean_text.split('.') if s.strip() and len(s.strip()) > 20]
        
        # Get words from main content to avoid duplication
        main_words = set(main_content.lower().split())
        
        # Look for different types of information
        question_lower = question.lower()
        complementary_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())
            
            # Skip if too similar to main content (>40% overlap)
            overlap = len(main_words & sentence_words)
            if overlap / max(len(sentence_words), 1) > 0.4:
                continue
            
            # Look for different aspects based on question type
            if 'diagnos' in question_lower or 'test' in question_lower:
                # For diagnosis questions, look for treatment or cause info as complement
                if any(term in sentence_lower for term in [
                    'treatment', 'therapy', 'management', 'prevent', 'cause', 'risk factor'
                ]):
                    complementary_sentences.append(sentence)            elif 'treat' in question_lower:
                # For treatment questions, look for diagnostic or prognostic info
                if any(term in sentence_lower for term in [
                    'diagnos', 'test', 'prognosis', 'outcome', 'recovery', 'prevent'
                ]):
                    complementary_sentences.append(sentence)
            elif 'cause' in question_lower:
                # For cause questions, look for symptoms or prevention
                if any(term in sentence_lower for term in [
                    'symptom', 'sign', 'present', 'prevent', 'avoid', 'risk'
                ]):
                    complementary_sentences.append(sentence)
            else:
                # General case: look for additional medical information
                if any(term in sentence_lower for term in [
                    'important', 'note', 'also', 'addition', 'furthermore', 'moreover'
                ]):
                    complementary_sentences.append(sentence)
        
        if complementary_sentences:
            # Take up to 2 complementary sentences
            result = '. '.join(complementary_sentences[:2]) + '.'
            return result
        
        return ""
    
    def _is_content_too_garbled(self, content: str) -> bool:
        """Check if content is too garbled to extract meaningful information"""
        if not content or len(content) < 50:
            return True
        
        # Check for meaningful medical content first
        medical_content_indicators = [
            'acute kidney injury', 'aki', 'symptoms', 'causes', 'treatment', 'diagnosis',
            'kidney failure', 'kidney damage', 'blood', 'urine', 'creatinine',
            'signs and symptoms', 'healthcare provider', 'medical'
        ]
        
        content_lower = content.lower()
        meaningful_indicators = sum(1 for indicator in medical_content_indicators 
                                  if indicator in content_lower)
        
        # If we have meaningful medical content, don't consider it garbled
        if meaningful_indicators >= 3:
            return False
        
        # Count ratio of uppercase words (common in garbled PDFs)
        words = content.split()
        uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        if len(words) > 0 and uppercase_words / len(words) > 0.7:  # Increased threshold
            return True
        
        # Check for excessive fragmentation
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) == 0:
            return True
        
        # Check if most sentences are too short or fragmented
        short_sentences = sum(1 for s in sentences if len(s.split()) < 3)  # Reduced threshold
        if len(sentences) > 0 and short_sentences / len(sentences) > 0.8:  # Increased threshold
            return True
        
        return False
    
    def _get_fallback_analysis(self, question_lower: str, medical_condition: str) -> Dict:
        """Provide fallback medical knowledge when PDF content is too garbled"""
        analysis = {
            'medical_condition': medical_condition or 'Acute Kidney Injury',
            'main_topics': [],
            'definitions': [],
            'symptoms': [],
            'causes': [],
            'treatments': [],
            'diagnostic_info': [],
            'risk_factors': [],
            'prevention': [],
            'key_terms': ['AKI', 'Creatinine', 'GFR', 'Urine']
        }
        
        # Add comprehensive fallback knowledge for AKI
        if 'aki' in question_lower or 'acute kidney injury' in question_lower or not medical_condition:
            analysis['definitions'] = [
                "Acute Kidney Injury (AKI) is a sudden decrease in kidney function that occurs over hours to days, characterized by elevated creatinine levels and reduced urine output."
            ]
            
            analysis['symptoms'] = [
                "Decreased urine output or no urine output",
                "Swelling in legs, ankles, and around the eyes",
                "Fatigue and weakness",
                "Shortness of breath",
                "Confusion or altered mental state",
                "Nausea and vomiting",
                "Chest pain or pressure"
            ]
            
            analysis['causes'] = [
                "Severe dehydration and blood loss",
                "Certain medications including NSAIDs and some antibiotics",
                "Severe infections (sepsis)",
                "Heart failure or heart attack",
                "Liver failure",
                "Urinary tract blockages",
                "Autoimmune kidney diseases"
            ]
            
            analysis['treatments'] = [
                "Treating the underlying cause",
                "Fluid management and electrolyte balance",
                "Temporary dialysis in severe cases",
                "Monitoring kidney function with blood tests",
                "Avoiding nephrotoxic medications",
                "Nutritional support and dietary modifications"
            ]
            
            analysis['diagnostic_info'] = [
                "Blood tests measuring creatinine and BUN levels",
                "Urine tests to check for protein and blood",
                "Glomerular filtration rate (GFR) calculation",
                "Kidney ultrasound or CT scan",
                "Monitoring urine output over time"
            ]
            
            analysis['prevention'] = [
                "Stay well hydrated, especially during illness",
                "Avoid overuse of NSAIDs and nephrotoxic drugs",
                "Manage chronic conditions like diabetes and hypertension",
                "Seek prompt treatment for infections",
                "Regular monitoring if at high risk"
            ]
            
            analysis['risk_factors'] = [
                "Advanced age (over 65 years)",
                "Chronic kidney disease",
                "Diabetes mellitus",
                "High blood pressure",
                "Heart failure or liver disease",
                "Use of certain medications"
            ]
        
        return analysis
