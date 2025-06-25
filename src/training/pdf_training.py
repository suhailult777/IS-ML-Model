#!/usr/bin/env python3
"""
Training Pipeline for PDF-based Conversational AI
Fine-tunes the document QA model on medical documentation content
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
from pathlib import Path
import random

# Import our models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.document_qa_model import DocumentQAModel, MedicalDocumentQA, create_model
from data.document_understanding import DocumentUnderstandingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalQADataset(Dataset):
    """Custom dataset for Medical Q&A"""
    
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def prepare_training_data(data_dir: str = "data/processed") -> Tuple[List[Dict], List[str]]:
    """
    Prepare training data from processed PDF content
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Tuple of (qa_pairs, document_sections)
    """
    logger.info(f"Preparing training data from {data_dir}")
    
    # Load Q&A pairs
    qa_file = os.path.join(data_dir, "qa_pairs.json")
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    # Load knowledge base for additional documents
    kb_file = os.path.join(data_dir, "knowledge_base.json")
    with open(kb_file, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    
    # Extract document sections
    document_sections = []
    for category, items in knowledge_base.items():
        for title, data in items.items():
            content = data['content']
            if len(content) > 100:  # Only include substantial content
                document_sections.append(content)
    
    # Add answers from Q&A pairs as additional documents
    for qa in qa_pairs:
        if len(qa['answer']) > 100:
            document_sections.append(qa['answer'])
    
    # Remove duplicates
    document_sections = list(set(document_sections))
    
    logger.info(f"Prepared {len(qa_pairs)} Q&A pairs and {len(document_sections)} document sections")
    return qa_pairs, document_sections

def create_training_examples(qa_pairs: List[Dict], document_sections: List) -> List[Dict]:
    """Create training examples from Q&A pairs and document sections."""
    training_examples = []
    
    # Create a dictionary for quick lookup of document sections
    section_dict = {
        i: (s['content'] if isinstance(s, dict) and 'content' in s else s)
        for i, s in enumerate(document_sections)
    }
    
    for pair in qa_pairs:
        question = pair.get('question')
        answer = pair.get('answer')
        
        if question and answer:
            # Simple positive example
            training_examples.append({
                'question': question,
                'context': answer,
                'label': 1
            })
            
            # Simple negative example (random section)
            if section_dict:
                random_section_index = random.choice(list(section_dict.keys()))
                random_section = section_dict[random_section_index]
                if random_section != answer:
                    training_examples.append({
                        'question': question,
                        'context': random_section,
                        'label': 0
                    })

    logger.info(f"Created {len(training_examples)} training examples")
    return training_examples

class MedicalQATrainer:
    """
    Trainer for the medical QA model
    """
    
    def __init__(self, 
                 model: DocumentQAModel,
                 learning_rate: float = 1e-4,
                 device: str = None):
        
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, data_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(data_loader, desc="Training", leave=False):
            questions = [item['question'] for item in batch]
            documents = [item['context'] for item in batch]
            labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            relevance_scores, _ = self.model(documents, questions)
            
            # Ensure predictions and labels have the same shape
            if relevance_scores.shape != labels.shape:
                logger.warning(f"Shape mismatch: predictions {relevance_scores.shape}, labels {labels.shape}")
                # Handle mismatch, e.g., by skipping batch or resizing
                continue

            loss = self.criterion(relevance_scores, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(data_loader)

    def validate_epoch(self, data_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validation", leave=False):
                questions = [item['question'] for item in batch]
                documents = [item['context'] for item in batch]
                labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32).to(self.device)

                relevance_scores, _ = self.model(documents, questions)

                if relevance_scores.shape != labels.shape:
                    logger.warning(f"Shape mismatch during validation: predictions {relevance_scores.shape}, labels {labels.shape}")
                    continue

                loss = self.criterion(relevance_scores, labels)
                total_loss += loss.item()
                
        return total_loss / len(data_loader)
    
    def train(self, 
              train_dataloader: DataLoader, 
              val_dataloader: DataLoader,
              num_epochs: int = 5,
              save_path: str = "models/medical_qa_model.pt"):
        """
        Train the model
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            save_path: Path to save the best model
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch(val_dataloader)
            self.val_losses.append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path)
                logger.info(f"Saved best model with val loss: {val_loss:.4f}")
        
        logger.info("Training completed!")
        return self.train_losses, self.val_losses
    
    def save_model(self, path: str):
        """Save model state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


def train_medical_qa_model(data_dir: str = "data/processed",
                          model_save_path: str = "models/medical_qa_model.pt",
                          num_epochs: int = 5,
                          batch_size: int = 8,
                          learning_rate: float = 1e-4) -> MedicalDocumentQA:
    """
    Main training function
    
    Args:
        data_dir: Directory containing processed data
        model_save_path: Path to save trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model
    """
    logger.info("Starting Medical QA Model Training Pipeline")
    
    # Prepare data
    qa_pairs, document_sections = prepare_training_data(data_dir)
    training_examples = create_training_examples(qa_pairs, document_sections)

    # Split data
    train_size = int(0.8 * len(training_examples))
    val_size = len(training_examples) - train_size
    train_examples, val_examples = random_split(training_examples, [train_size, val_size])

    train_dataset = MedicalQADataset(train_examples)
    val_dataset = MedicalQADataset(val_examples)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create model
    config = {
        'model_type': 'medical',
        'hidden_dim': 384,
        'num_heads': 8,
        'num_layers': 2,
        'dropout': 0.2
    }
    model = create_model(**config)
    
    # Trainer
    trainer = MedicalQATrainer(model, learning_rate=learning_rate)

    # Train the model
    logger.info(f"Starting training for {num_epochs} epochs")
    train_losses, val_losses = trainer.train(
        train_dataloader, val_dataloader, num_epochs=num_epochs
    )

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_save_path)
    
    logger.info(f"Model saved to {model_save_path}")
    return model


if __name__ == "__main__":
    # Run training
    try:
        model = train_medical_qa_model(
            data_dir="data/processed",
            model_save_path="models/medical_qa_model.pt",
            num_epochs=3,  # Reduced for testing
            batch_size=4,  # Smaller batch size for testing
            learning_rate=1e-4
        )
        
        print("Training completed successfully!")
        
        # Test the trained model
        print("\nTesting trained model...")
        test_docs = [
            "Acute kidney injury documentation requires creatinine levels and urine output monitoring.",
            "Heart failure diagnosis involves echocardiogram results and BNP level assessment.",
            "Depression evaluation includes psychiatric consultation and medication review."
        ]
        
        test_question = "What is needed for kidney injury documentation?"
        
        best_answers = model.predict_best_answer(test_docs, test_question, top_k=2)
        print(f"Question: {test_question}")
        print(f"Best answer: {best_answers[0][0][:100]}...")
        print(f"Confidence: {best_answers[0][1]:.3f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
