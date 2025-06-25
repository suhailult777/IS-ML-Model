#!/usr/bin/env python3
"""
Builds the RAG index from the processed PDF content.
This script initializes the SimpleRAG system, builds the FAISS index,
and triggers the training of the reranker model.
"""

import sys
import os
import logging

# Add src to path
sys.path.append('src')

from rag.simple_rag import SimpleRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to build the RAG index.
    """
    logging.info("Starting the index building process...")

    # Path to the processed PDF content
    pdf_content_path = "data/processed/pdf_content.txt"

    if not os.path.exists(pdf_content_path):
        logging.error(f"Content file not found at {pdf_content_path}.")
        logging.error("Please run `extract_pdf.py` first to generate the content file.")
        return

    logging.info(f"Loading content from {pdf_content_path}")
    with open(pdf_content_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        logging.error("Content file is empty. Cannot build index.")
        return

    # The RAG system expects a list of documents, where each doc is a dictionary.
    # For a single PDF, we can treat the whole content as one document,
    # or split it into smaller documents (e.g., by paragraph).
    # Splitting can sometimes yield better retrieval results. Let's split by paragraphs.
    paragraphs = content.split('\n\n')
    documents = [{'content': p} for p in paragraphs if p.strip()]
    
    logging.info(f"Created {len(documents)} documents from the content file.")

    # Initialize the RAG system
    rag = SimpleRAG()

    # Build the index. This will also trigger the reranker training.
    rag.build_index(documents)

    # Save the index for later use
    index_path = 'data/processed/rag_index'
    rag.save_index(index_path)
    logging.info(f"RAG index and associated files saved to {index_path}")

    logging.info("Index building process completed successfully!")

if __name__ == "__main__":
    main() 