#!/usr/bin/env python3
"""
Extract content from acute kidney injury PDF and prepare it for the AI system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.pdf_processor import PDFProcessor

def extract_pdf_content():
    """Extract content from the AKI PDF"""
    print("📄 Extracting content from acute_kidney_injury.pdf...")
    
    processor = PDFProcessor()
    pdf_path = "data/processed/acute_kidney_injury.pdf"
    
    try:
        # Try pdfplumber first (more accurate)
        content = processor.extract_text_pdfplumber(pdf_path)
        print(f"✅ Successfully extracted {len(content)} characters using pdfplumber")
    except Exception as e:
        print(f"⚠️ pdfplumber failed: {e}")
        try:
            # Fallback to PyPDF2
            content = processor.extract_text_pypdf2(pdf_path)
            print(f"✅ Successfully extracted {len(content)} characters using PyPDF2")
        except Exception as e2:
            print(f"❌ Both extraction methods failed: {e2}")
            return False
    
    # Clean the content
    clean_content = processor.clean_text(content)
    
    # Save to pdf_content.txt for the AI system to use
    output_path = "data/processed/pdf_content.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(clean_content)
    
    print(f"💾 Saved cleaned content to {output_path}")
    print(f"📊 Content preview (first 300 chars):")
    print("-" * 50)
    print(clean_content[:300] + "...")
    print("-" * 50)
    
    return True

if __name__ == "__main__":
    success = extract_pdf_content()
    if success:
        print("\n🎉 PDF content extraction completed!")
        print("🤖 The AI system can now process this content automatically.")
    else:
        print("\n❌ PDF content extraction failed!")
