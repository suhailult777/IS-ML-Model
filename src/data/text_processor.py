#!/usr/bin/env python3
"""
Text Processing and Structuring for PDF-based Conversational AI
Handles text cleaning, section extraction, and knowledge base creation
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalTextProcessor:
    """
    Specialized text processor for medical documentation templates
    """
    
    def __init__(self):
        self.sections = []
        self.knowledge_base = {}
        self.qa_pairs = []
        
    def extract_medical_sections(self, text: str) -> List[Dict]:
        """Extract medical condition sections from the text"""
        sections = []

        # First, split by page markers to get individual templates
        page_pattern = r'page \d+'
        page_splits = re.split(page_pattern, text)

        # Define medical condition patterns that indicate new sections
        condition_patterns = [
            r'Acute Kidney Injury',
            r'Acute Tubular Necrosis',
            r'Acute Blood Loss Anemia',
            r'BMI \d+',
            r'Chest Pain',
            r'CKD Stage',
            r'Debridement',
            r'Demand Ischemia',
            r'Depression',
            r'Encephalopathy',
            r'Functional Quadriplegia',
            r'GI Bleeding',
            r'Heart Failure',
            r'HIV.*AIDS',
            r'Hypertensive',
            r'Malnutrition',
            r'Pancytopenia',
            r'Pneumonia',
            r'Pressure Ulcer',
            r'Respiratory Failure',
            r'Schizophrenia',
            r'Sepsis',
            r'Shock',
            r'Syncope',
            r'TIA',
            r'Urosepsis'
        ]

        # Combine all patterns
        combined_pattern = '|'.join(f'({pattern})' for pattern in condition_patterns)

        # Split text by medical conditions
        condition_splits = re.split(combined_pattern, text, flags=re.IGNORECASE)

        current_title = None
        for i, segment in enumerate(condition_splits):
            if not segment or segment.strip() == '':
                continue

            # Check if this segment is a condition title
            if any(re.search(pattern, segment, re.IGNORECASE) for pattern in condition_patterns):
                current_title = segment.strip()
            elif current_title and len(segment.strip()) > 50:  # Content segment
                # Clean and extract the content
                content = self._clean_section_content(segment)
                if content:
                    sections.append({
                        'title': current_title,
                        'content': content,
                        'type': 'medical_template',
                        'keywords': self._extract_keywords(current_title + ' ' + content)
                    })
                current_title = None

        return sections

    def _clean_section_content(self, content: str) -> str:
        """Clean section content"""
        # Remove extra whitespace and normalize
        content = re.sub(r'\s+', ' ', content).strip()

        # Remove page numbers and dates
        content = re.sub(r'Updated May \d+,\s*\d+\s*\d+\s*\d+', '', content)
        content = re.sub(r'page \d+', '', content)

        # Remove copyright notices
        content = re.sub(r'Pinson Tang.*?LLC.*?Reserved\.?', '', content, flags=re.IGNORECASE)

        return content.strip()

    def _is_section_header(self, line: str) -> bool:
        """Determine if a line is a section header"""
        # Medical condition headers are typically short and contain medical terms
        medical_keywords = [
            'kidney', 'anemia', 'bmi', 'chest pain', 'ckd', 'depression',
            'encephalopathy', 'heart failure', 'hiv', 'hypertensive',
            'malnutrition', 'pneumonia', 'sepsis', 'shock', 'syncope',
            'respiratory failure', 'bleeding', 'debridement', 'ulcer'
        ]
        
        line_lower = line.lower()
        
        # Check if it's a short line with medical keywords
        if len(line.split()) <= 5 and any(keyword in line_lower for keyword in medical_keywords):
            return True
            
        # Check for specific patterns
        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', line) and len(line) < 50:
            return True
            
        return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract medical keywords from text"""
        medical_terms = [
            'acute', 'chronic', 'kidney', 'injury', 'anemia', 'blood', 'loss',
            'chest', 'pain', 'heart', 'failure', 'pneumonia', 'sepsis',
            'shock', 'respiratory', 'failure', 'malnutrition', 'depression',
            'encephalopathy', 'bleeding', 'hypertensive', 'diabetes',
            'infection', 'diagnosis', 'treatment', 'symptoms', 'condition'
        ]
        
        keywords = []
        text_lower = text.lower()
        
        for term in medical_terms:
            if term in text_lower:
                keywords.append(term)
                
        return keywords
    
    def create_qa_pairs(self, sections: List[Dict]) -> List[Dict]:
        """Generate question-answer pairs from medical sections"""
        qa_pairs = []
        
        for section in sections:
            title = section['title']
            content = section['content']
            
            # Generate different types of questions
            questions = [
                f"What is {title}?",
                f"How do you document {title}?",
                f"What are the criteria for {title}?",
                f"What information is needed for {title} documentation?",
                f"Tell me about {title}",
                f"Explain {title} documentation requirements",
                f"What should be included in {title} queries?"
            ]
            
            for question in questions:
                qa_pairs.append({
                    'question': question,
                    'answer': content,
                    'topic': title,
                    'keywords': section['keywords']
                })
        
        return qa_pairs
    
    def create_knowledge_base(self, sections: List[Dict]) -> Dict:
        """Create a structured knowledge base from sections"""
        knowledge_base = {
            'medical_conditions': {},
            'documentation_guidelines': {},
            'query_templates': {},
            'general_info': {}
        }
        
        for section in sections:
            title = section['title']
            content = section['content']
            keywords = section['keywords']
            
            # Categorize based on content
            if 'documentation' in content.lower() or 'medical record' in content.lower():
                knowledge_base['documentation_guidelines'][title] = {
                    'content': content,
                    'keywords': keywords
                }
            elif 'query' in content.lower() or 'clarify' in content.lower():
                knowledge_base['query_templates'][title] = {
                    'content': content,
                    'keywords': keywords
                }
            elif any(keyword in keywords for keyword in ['acute', 'chronic', 'kidney', 'heart', 'pneumonia']):
                knowledge_base['medical_conditions'][title] = {
                    'content': content,
                    'keywords': keywords
                }
            else:
                knowledge_base['general_info'][title] = {
                    'content': content,
                    'keywords': keywords
                }
        
        return knowledge_base
    
    def process_medical_text(self, text_file: str) -> Dict:
        """Main processing function for medical text"""
        logger.info(f"Processing medical text from: {text_file}")
        
        # Read the text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Extract sections
        sections = self.extract_medical_sections(text)
        logger.info(f"Extracted {len(sections)} medical sections")
        
        # Create Q&A pairs
        qa_pairs = self.create_qa_pairs(sections)
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
        
        # Create knowledge base
        knowledge_base = self.create_knowledge_base(sections)
        
        # Store results
        self.sections = sections
        self.qa_pairs = qa_pairs
        self.knowledge_base = knowledge_base
        
        result = {
            'sections': sections,
            'qa_pairs': qa_pairs,
            'knowledge_base': knowledge_base,
            'stats': {
                'total_sections': len(sections),
                'total_qa_pairs': len(qa_pairs),
                'medical_conditions': len(knowledge_base['medical_conditions']),
                'documentation_guidelines': len(knowledge_base['documentation_guidelines']),
                'query_templates': len(knowledge_base['query_templates'])
            }
        }
        
        return result
    
    def save_processed_data(self, output_dir: str):
        """Save processed data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save Q&A pairs
        qa_file = output_path / "qa_pairs.json"
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
        
        # Save knowledge base
        kb_file = output_path / "knowledge_base.json"
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        
        # Save sections
        sections_file = output_path / "sections.json"
        with open(sections_file, 'w', encoding='utf-8') as f:
            json.dump(self.sections, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed data to: {output_dir}")
    
    def search_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search the knowledge base for relevant information"""
        query_lower = query.lower()
        results = []
        
        # Search through all categories
        for category, items in self.knowledge_base.items():
            for title, data in items.items():
                content = data['content']
                keywords = data['keywords']
                
                # Calculate relevance score
                score = 0
                
                # Check for exact matches in title
                if query_lower in title.lower():
                    score += 10
                
                # Check for keyword matches
                for keyword in keywords:
                    if keyword in query_lower:
                        score += 2
                
                # Check for content matches
                if query_lower in content.lower():
                    score += 1
                
                if score > 0:
                    results.append({
                        'title': title,
                        'content': content,
                        'category': category,
                        'score': score,
                        'keywords': keywords
                    })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


def process_pdf_content(pdf_text_file: str, output_dir: str = "data/processed") -> Dict:
    """
    Main function to process PDF content for conversational AI
    
    Args:
        pdf_text_file: Path to extracted PDF text file
        output_dir: Directory to save processed data
        
    Returns:
        Dictionary with processing results
    """
    processor = MedicalTextProcessor()
    result = processor.process_medical_text(pdf_text_file)
    
    # Save processed data
    processor.save_processed_data(output_dir)
    
    return result


if __name__ == "__main__":
    # Test the text processor
    pdf_text_file = "data/processed/pdf_content.txt"
    if os.path.exists(pdf_text_file):
        result = process_pdf_content(pdf_text_file)
        print(f"Processing complete:")
        print(f"- {result['stats']['total_sections']} sections extracted")
        print(f"- {result['stats']['total_qa_pairs']} Q&A pairs generated")
        print(f"- {result['stats']['medical_conditions']} medical conditions")
        print(f"- {result['stats']['documentation_guidelines']} documentation guidelines")
    else:
        print(f"PDF text file not found: {pdf_text_file}")
