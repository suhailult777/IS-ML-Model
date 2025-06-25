#!/usr/bin/env python3
"""
Relevancy Trainer for RAG
Fine-tunes a model to re-rank retrieved documents for relevancy.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import random
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.document_qa_model import DocumentQAModel, create_model
from training.pdf_training import MedicalQATrainer

logger = logging.getLogger(__name__)

class RelevancyDataset(Dataset):
    """Custom dataset for relevancy training"""
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def _parse_content_into_sections(text_content: str) -> list[dict]:
    """Parses raw text into sections with titles and content."""
    sections = []
    # This is a simplified parser. A more robust one might be needed.
    # It looks for lines that look like headers (e.g., all caps, or short lines).
    lines = text_content.split('\n')
    current_title = "Introduction"
    current_content = ""

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        
        # Simple heuristic for a title: short, and maybe all caps or title case
        if len(stripped_line) < 80 and stripped_line.isupper():
            if current_content.strip():
                sections.append({'title': current_title, 'content': current_content.strip()})
            current_title = stripped_line
            current_content = ""
        else:
            current_content += " " + stripped_line
    
    if current_content.strip():
        sections.append({'title': current_title, 'content': current_content.strip()})
        
    # If no sections were found, treat the whole content as one section
    if not sections:
        sections.append({'title': 'Document', 'content': text_content})

    return sections

def create_training_examples_from_sections(sections: list[dict]) -> list[dict]:
    """Create training examples from sections."""
    training_examples = []
    
    if len(sections) < 2:
        logger.warning("Not enough sections to create negative examples. Creating examples from sentences.")
        if len(sections) == 1:
            sentences = re.split(r'(?<=[.!?])\s+', sections[0]['content'])
            if len(sentences) > 2:
                for i in range(len(sentences)):
                    # Positive example: consecutive sentences
                    if i + 1 < len(sentences):
                        training_examples.append({
                            'question': sentences[i],
                            'context': sentences[i+1],
                            'label': 1
                        })
                    
                    # Negative example: non-consecutive sentences
                    if len(sentences) > 2:
                        random_index = i
                        # Ensure it's not consecutive or the same
                        while abs(random_index - i) <= 1:
                            random_index = random.randint(0, len(sentences) - 1)
                        
                        training_examples.append({
                            'question': sentences[i],
                            'context': sentences[random_index],
                            'label': 0
                        })
        return training_examples

    for i, section in enumerate(sections):
        # Positive example
        training_examples.append({
            'question': f"What is about {section['title']}?", # Synthetic question
            'context': section['content'],
            'label': 1
        })
        
        # Negative example
        random_index = i
        while random_index == i:
            random_index = random.randint(0, len(sections) - 1)
        
        training_examples.append({
            'question': f"What is about {section['title']}?", # Synthetic question
            'context': sections[random_index]['content'],
            'label': 0
        })
        
    logger.info(f"Created {len(training_examples)} training examples from {len(sections)} sections.")
    return training_examples

def collate_fn(batch):
    return batch

def train_relevancy_model(text_content: str, 
                          model_save_path: str = "models/reranker_model.pt",
                          num_epochs: int = 3,
                          batch_size: int = 4):
    """
    Trains a relevancy model from raw text content.
    """
    logger.info("Starting relevancy model training...")
    
    # 1. Parse content into sections
    sections = _parse_content_into_sections(text_content)
    
    if not sections:
        logger.error("Could not parse any sections from the content. Aborting training.")
        return

    # 2. Create training examples
    examples = create_training_examples_from_sections(sections)
    
    if not examples:
        logger.error("Could not create any training examples. Aborting training.")
        return

    # 3. Create Dataset and DataLoader
    dataset = RelevancyDataset(examples)
    # No validation set for simplicity in this integration
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 4. Initialize model and trainer
    # Using 'medical' type as it might have better default settings
    reranker_model = create_model(model_type='medical') 
    trainer = MedicalQATrainer(model=reranker_model, learning_rate=1e-5)

    # 5. Train
    logger.info(f"Training relevancy model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = trainer.train_epoch(dataloader)
        logger.info(f"Train Loss: {train_loss:.4f}")
        trainer.save_model(model_save_path)
        logger.info(f"Saved model after epoch {epoch+1} to {model_save_path}")

    logger.info("Relevancy model training completed!")

if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    
    # Check if pdf_content.txt exists
    if not os.path.exists("data/processed/pdf_content.txt"):
        print("Please run `extract_pdf.py` first to generate `data/processed/pdf_content.txt`")
    else:
        with open("data/processed/pdf_content.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        if content:
            train_relevancy_model(content)
        else:
            print("pdf_content.txt is empty.") 