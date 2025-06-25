#!/usr/bin/env python3
"""
PDF Processing Utilities for Document Understanding System
Handles PDF text extraction, cleaning, and preprocessing
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print("PDF processing libraries not installed. Please run: pip install PyPDF2 pdfplumber")
    PyPDF2 = None
    pdfplumber = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Comprehensive PDF processing class for extracting and cleaning text content
    """
    
    def __init__(self):
        self.text_content = ""
        self.metadata = {}
        self.sections = []
        
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 library"""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 not installed")
            
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                self.metadata = {
                    'num_pages': len(pdf_reader.pages),
                    'title': pdf_reader.metadata.get('/Title', 'Unknown') if pdf_reader.metadata else 'Unknown'
                }
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                        
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            
        return text
    
    def extract_text_pdfplumber(self, pdf_path: str, keep_layout: bool = True) -> str:
        """Extract text using pdfplumber library (more accurate)"""
        if pdfplumber is None:
            raise ImportError("pdfplumber not installed")
            
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                self.metadata = {
                    'num_pages': len(pdf.pages),
                    'title': pdf.metadata.get('Title', 'Unknown') if pdf.metadata else 'Unknown'
                }
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=2, keep_blank_chars=keep_layout, use_text_flow=keep_layout)
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                        
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Keep paragraph breaks
        text = re.sub(r'(\n\s*){2,}', '\n\n', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\w)(\d)', r'\1 \2', text)  # Add space between word and number
        text = re.sub(r'(\d)(\w)', r'\1 \2', text)  # Add space between number and word
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', ' ', text)
        
        # Normalize whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_tables_as_text(self, pdf_path: str) -> str:
        """Extract tables and convert them to a clean text format"""
        if pdfplumber is None:
            raise ImportError("pdfplumber not installed")
            
        all_tables_text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    if tables:
                        all_tables_text += f"\n--- Tables on Page {page_num + 1} ---\n"
                        for table in tables:
                            # Convert table to a string format
                            table_text = "\n".join(["\t".join(map(str, row)) for row in table])
                            all_tables_text += table_text + "\n\n"
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
        return all_tables_text

    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Extract logical sections from the text using improved rules."""
        sections = []
        
        # Split by newline and filter empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return []

        current_title = "Introduction"
        current_content = ""

        # Regex for potential headers (e.g., ALL CAPS, Title Case, numbered lists)
        header_pattern = re.compile(r'^(?:(\d+\.)|[A-Z][A-Z\s]+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$')

        for i, line in enumerate(lines):
            # A line is a header if it's short and matches a header pattern
            # or if it's followed by an empty line (heuristic)
            is_header = (len(line) < 100 and header_pattern.match(line)) or \
                        (i + 1 < len(lines) and not lines[i+1])

            if is_header and current_content.strip():
                # Save the previous section
                sections.append({'title': current_title, 'content': current_content.strip()})
                current_title = line
                current_content = ""
            else:
                current_content += " " + line
        
        # Add the last section
        if current_content.strip():
            sections.append({'title': current_title, 'content': current_content.strip()})
        
        # If no sections were found, treat the whole content as one
        if not sections:
            sections.append({'title': 'Full Document', 'content': text})
            
        return sections
    
    def process_pdf(self, pdf_path: str, method: str = "pdfplumber") -> Dict:
        """
        Main method to process a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            method: Extraction method ('pdfplumber' or 'pypdf2')
            
        Returns:
            Dictionary containing processed text and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text with layout preservation
        if method == "pdfplumber":
            raw_text = self.extract_text_pdfplumber(pdf_path, keep_layout=True)
        else:
            raw_text = self.extract_text_pypdf2(pdf_path)
            
        # Extract tables separately
        tables_text = self.extract_tables_as_text(pdf_path)
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)

        # Combine main text with table text
        full_content = cleaned_text + "\n\n" + tables_text
        
        # Extract sections from the combined content
        sections = self.extract_sections(full_content)
        
        # Store results
        self.text_content = full_content
        self.sections = sections
        
        result = {
            'raw_text': raw_text,
            'cleaned_text': full_content,
            'sections': sections,
            'metadata': self.metadata,
            'word_count': len(cleaned_text.split()),
            'char_count': len(cleaned_text)
        }
        
        logger.info(f"Extracted {result['word_count']} words from {self.metadata.get('num_pages', 0)} pages")
        
        return result
    
    def save_processed_text(self, output_path: str, format: str = "txt"):
        """Save processed text to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.text_content)
        elif format == "json":
            import json
            data = {
                'text': self.text_content,
                'sections': self.sections,
                'metadata': self.metadata
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        logger.info(f"Saved processed text to: {output_path}")


def extract_pdf_content(pdf_path: str, output_dir: str = "data/processed") -> Dict:
    """
    Convenience function to extract PDF content
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save processed content
        
    Returns:
        Dictionary with processed content
    """
    processor = PDFProcessor()
    result = processor.process_pdf(pdf_path)
    
    # Save processed content
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as text file
    text_path = os.path.join(output_dir, "pdf_content.txt")
    processor.save_processed_text(text_path, format="txt")
    
    # Save as JSON with metadata
    json_path = os.path.join(output_dir, "pdf_content.json")
    processor.save_processed_text(json_path, format="json")
    
    return result


if __name__ == "__main__":
    # Test the PDF processor
    pdf_path = "data/processed/Sample-Physician-Query-Templates-May-2018.pdf"
    if os.path.exists(pdf_path):
        result = extract_pdf_content(pdf_path)
        print(f"Successfully processed PDF: {result['word_count']} words extracted")
    else:
        print(f"PDF file not found: {pdf_path}")
