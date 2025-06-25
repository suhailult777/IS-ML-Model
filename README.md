# 🏥 ML Model with Enhanced RAG and contineous

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> An advanced AI-powered medical document analysis system with real-time learning capabilities, structured output formatting, and professional GPT-like responses.

## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Training & Learning](#training--learning)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## 🔬 Overview

This project is an advanced AI assistant built on a **custom Retrieval-Augmented Generation (RAG) system** with **online learning capabilities**. The system processes all documents (PDFs) and provides intelligent, structured responses to medical queries with real-time model improvement through user interactions.

### 🎯 Purpose

- **all Document Analysis**: Process and understand complex medical literature
- **Intelligent Q&A**: Provide accurate, structured answers to medical questions
- **Continuous Learning**: Improve responses through online learning from user interactions
- **Professional Formatting**: Generate GPT-like structured outputs with confidence scoring

### 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Content   │───▶│   RAG System    │───▶│  AI Assistant   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Document Parser │    │ Vector Database │    │ Online Learning │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ✨ Key Features

### 🎭 Structured Output Formatting

- **Professional markdown responses** with H3 headings and bullet points
- **Bold key terms** for easy scanning and comprehension
- **Confidence scoring** for transparency and reliability
- **Source citations** with document references

### 🧠 Advanced RAG System

- **Semantic search** using sentence transformers
- **Document chunking** with overlap for context preservation
- **FAISS vector indexing** for fast similarity search
- **Custom medical model** integration for domain-specific understanding

### 📚 Learning Capabilities

- **Real-time Q&A capture** from user interactions
- **Background model training** with accumulated conversation data
- **Confidence tracking** and improvement measurement
- **Thread-safe queue management** for concurrent learning

### 🏥 Medical Domain Expertise

- **Acute Kidney Injury (AKI)** specialized knowledge
- **Symptoms, causes, treatment** categorization
- **Medical terminology** understanding
- **Clinical guidelines** integration

## 🏛️ Architecture

### Core Components

```python
EnhancedRAGWithParsers
├── Document Processing
│   ├── PDF Content Extraction
│   ├── Text Chunking & Overlap
│   └── Metadata Preservation
├── Vector Storage & Retrieval
│   ├── FAISS Indexing
│   ├── Sentence Transformers
│   └── Semantic Search
├── Medical Model Integration
│   ├── Custom Medical QA Model
│   ├── Reranker Model
│   └── Domain-Specific Training
├── Output Parsing & Formatting
│   ├── Pydantic Models
│   ├── LangChain Parsers
│   └── Structured Responses
└── Online Learning System
    ├── Q&A Pair Collection
    ├── Background Training
    └── Performance Tracking
```

### Model Architecture

1. **Document Understanding System**

   - PDF text extraction and preprocessing
   - Section-based document parsing
   - Metadata enrichment

2. **Retrieval System**

   - Sentence transformer embeddings (`all-MiniLM-L6-v2`)
   - FAISS vector database for similarity search
   - Context-aware document chunking

3. **Medical QA Model**

   - Custom PyTorch model for medical domain
   - Question-answer pair training
   - Relevancy scoring and reranking

4. **Learning Pipeline**
   - Real-time Q&A pair capture
   - Batch training with user interactions
   - Model performance improvement tracking

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/medical-ai-assistant.git
cd medical-ai-assistant

# Install required packages
pip install -r requirements.txt
```

### Required Packages

```python
# Core ML/AI packages
torch>=2.0.0
transformers>=4.21.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4  # or faiss-gpu for GPU support

# RAG and NLP
langchain>=0.1.0
langchain-core>=0.1.0
pydantic>=2.0.0

# Document processing
pypdf2>=3.0.0
python-docx>=0.8.11

# Utilities
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.64.0
```

## ⚡ Quick Start

### 1. Basic Setup

```python
from src.rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers

# Initialize the RAG system with online learning
rag = EnhancedRAGWithParsers(
    enable_online_learning=True,
    online_learning_batch_size=5,
    online_learning_interval=180  # 3 minutes
)

# Load existing index or build new one
documents = [
    {
        'content': 'Acute kidney injury (AKI) is a sudden episode...',
        'metadata': {'source': 'medical_guide.pdf', 'page': 1}
    }
]
rag.build_index(documents, train_medical_model=True)
```

### 2. Interactive Chat Interface

```bash
# Start the enhanced chat interface
python enhanced_conversational_ai.py

# Or use the simpler chat interface
python enhanced_chat_with_parsers.py
```

### 3. Direct API Usage

```python
# Ask questions and get structured responses
response = rag.ask(\"What are the treatment options for AKI?\")

print(response['answer'])
# Output:
# ### AKI Treatment Options
# - **Hospitalization**: Treatment typically requires a hospital stay for monitoring
# - **Dialysis**: May be needed in severe cases to support kidney function
# - **Address Root Cause**: Primary focus on treating the underlying condition

print(f\"Confidence: {response['confidence']:.3f}\")
print(f\"Sources: {response['sources']}\")
```

## 📊 Usage Examples

### Medical Question Types

#### Treatment Questions

```python
question = \"How is AKI treated?\"
response = rag.ask(question)
```

**Output:**

```markdown
### AKI Treatment Approach

**Summary**: AKI treatment focuses on hospitalization, supportive care, and addressing the underlying cause.

- **Hospitalization**: Treatment for AKI usually means a hospital stay for monitoring and management
- **Dialysis**: In more serious cases, dialysis may be needed to take over kidney function until recovery
- **Treat Underlying Cause**: The main goal is to treat the cause of your AKI
- **Recovery Focus**: Treatment supports kidney recovery and prevents complications

**Conclusion**: Treatment duration depends on the cause and how quickly the kidneys recover.

_Confidence: High_
```

#### Symptom Questions

```python
question = \"What are the symptoms of acute kidney injury?\"
response = rag.ask(question)
```

#### Diagnostic Questions

```python
question = \"How is AKI diagnosed?\"
response = rag.ask(question)
```

### Online Learning Monitoring

```python
# Check online learning statistics
stats = rag.get_online_learning_stats()
print(f\"Queued Q&A pairs: {stats['queued_qa_pairs']}\")
print(f\"Total updates: {stats['total_online_updates']}\")
print(f\"Confidence improvement: {stats['confidence_improvement']:.3f}\")

# Force training for testing
rag.force_online_training()
```

## 📁 Project Structure

```
medical-ai-assistant/
├── 📊 data/
│   ├── processed/
│   │   ├── pdf_content.json          # Processed medical content
│   │   ├── enhanced_rag_index.*      # RAG vector indices
│   │   └── acute_kidney_injury.pdf   # Source medical document
│   ├── integrated/
│   └── raw/
├── 🧠 models/
│   ├── medical_qa_model.pt          # Trained medical QA model
│   ├── reranker_model.pt           # Document reranking model
│   └── best_tagt_model.pt          # Best performing model
├── 🔬 src/
│   ├── rag/
│   │   ├── enhanced_rag_with_parsers.py  # Main RAG system
│   │   ├── simple_rag.py                 # Basic RAG implementation
│   │   └── __init__.py
│   ├── models/
│   │   ├── document_qa_model.py          # Medical QA model architecture
│   │   └── tagt_model.py                 # Custom model implementations
│   ├── training/
│   │   ├── pdf_training.py               # Medical model training
│   │   ├── relevancy_trainer.py          # Relevancy model training
│   │   └── losses.py                     # Custom loss functions
│   ├── data/
│   │   ├── document_understanding.py     # Document processing
│   │   └── text_processor.py             # Text preprocessing
│   └── utils/
│       └── pdf_processor.py              # PDF extraction utilities
├── 💬 Chat Interfaces
│   ├── enhanced_conversational_ai.py     # Full-featured chat interface
│   ├── enhanced_chat_with_parsers.py     # Simple chat interface
│   └── simple_chat_fixed.py              # Basic chat implementation
├── 🧪 tests/
│   ├── test_enhanced_rag.py              # RAG system tests
│   ├── test_pdf_processing.py            # Document processing tests
│   └── test_*.py                         # Various component tests
├── 📋 Configuration
│   ├── requirements.txt                  # Python dependencies
│   ├── start.bat                         # Windows startup script
│   └── start.sh                          # Unix startup script
└── 📚 Documentation
    ├── README.md                         # This file
    ├── ENHANCED_RAG_README.md           # Detailed RAG documentation
    └── LICENSE                           # MIT License
```

## 🎯 Model Details

### Medical QA Model

**Architecture**: Custom PyTorch model with transformer components

```python
class DocumentQAModel(nn.Module):
    def __init__(self, hidden_dim=384, num_heads=8, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = TransformerEncoder(...)
        self.qa_head = QuestionAnsweringHead(...)
        self.relevance_scorer = RelevanceScorer(...)
```

**Training Data**:

- Medical document Q&A pairs
- Acute Kidney Injury (AKI) specialized content
- Synthetic question-answer generation from medical texts
- User interaction data (online learning)

**Performance Metrics**:

- Relevance scoring accuracy
- Answer quality assessment
- Confidence calibration
- Response time optimization

### Retrieval System

**Vector Model**: `sentence-transformers/all-MiniLM-L6-v2`

- 384-dimensional embeddings
- Optimized for semantic similarity
- Fast inference for real-time retrieval

**Index Structure**: FAISS IndexFlatIP

- Cosine similarity search
- Normalized embeddings
- Persistent storage for quick loading

## 🔄 Training & Learning

### Initial Training

```python
# Train on PDF content
from training.pdf_training import MedicalQATrainer

trainer = MedicalQATrainer(
    model=medical_model,
    device='cuda',
    learning_rate=1e-4,
    batch_size=8,
    max_epochs=10
)

# Prepare training data from medical documents
training_data = prepare_training_data(pdf_content)
trainer.train(training_data)
```

### Online Learning Process

1. **Q&A Capture**: Every user interaction is captured

```python
def _capture_qa_pair(self, question, answer, context_docs, confidence):
    qa_pair = {
        'question': question,
        'answer': answer,
        'context': context_text,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }
    self.qa_queue.append(qa_pair)
```

2. **Background Training**: Automatic model updates

```python
def _online_learning_worker(self):
    while not self.stop_online_training.wait(self.online_learning_interval):
        if len(self.qa_queue) >= self.online_learning_batch_size:
            self._perform_online_training()
```

3. **Performance Tracking**: Confidence improvement monitoring

```python
stats = {
    'average_confidence': avg_confidence,
    'recent_confidence': recent_confidence,
    'confidence_improvement': improvement,
    'total_online_updates': update_count
}
```

## 🔧 API Reference

### EnhancedRAGWithParsers

#### Initialization

```python
rag = EnhancedRAGWithParsers(
    retrieval_model: str = \"sentence-transformers/all-MiniLM-L6-v2\",
    medical_model_path: str = \"models/medical_qa_model.pt\",
    enable_online_learning: bool = True,
    online_learning_batch_size: int = 10,
    online_learning_interval: int = 300
)
```

#### Core Methods

**`build_index(documents, train_reranker=True, train_medical_model=True)`**

- Build FAISS index from document list
- Optionally train reranker and medical models
- Save index for persistence

**`ask(question, top_k=3)`**

- Get structured answer with online learning
- Returns dict with answer, confidence, sources
- Automatically captures Q&A for training

**`retrieve(question, top_k=3)`**

- Retrieve relevant documents
- Returns list of (content, score, metadata) tuples
- Uses reranker if available

**`get_online_learning_stats()`**

- Get current learning statistics
- Returns training progress and performance metrics

### MedicalResponse Model

```python
class MedicalResponse(BaseModel):
    title: str                    # H3 heading for response
    summary: Optional[str]        # Brief summary
    bullet_points: List[str]      # Structured bullet points
    conclusion: Optional[str]     # Concluding sentence
    confidence_level: Optional[str]  # High/Moderate/Low
```

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: GPU configuration
CUDA_VISIBLE_DEVICES=0

# Model paths
MEDICAL_MODEL_PATH=models/medical_qa_model.pt
RERANKER_MODEL_PATH=models/reranker_model.pt

# Data directories
DATA_DIR=data/processed
RAG_INDEX_PATH=data/processed/enhanced_rag_index
```

### Configuration Files

**`configs/rag_config.json`**:

```json
{
    \"retrieval_model\": \"sentence-transformers/all-MiniLM-L6-v2\",
    \"chunk_size\": 1000,
    \"chunk_overlap\": 200,
    \"top_k_retrieval\": 5,
    \"confidence_threshold\": 0.3,
    \"online_learning\": {
        \"enabled\": true,
        \"batch_size\": 5,
        \"interval_seconds\": 180,
        \"learning_rate\": 5e-5
    }
}
```

## 🎪 Interactive Features

### Chat Interface Commands

```bash
# In the chat interface
help                    # Show available commands
debug [question]        # Show detailed processing information
quit/exit/bye          # End conversation

# Sample questions to try:
\"What is AKI?\"
\"What are the treatment options for acute kidney injury?\"
\"How is AKI diagnosed?\"
\"What causes kidney problems?\"
```

### Online Learning Dashboard

The chat interface displays real-time learning statistics:

```
🧠 ONLINE LEARNING STATISTICS
============================================================
📊 Status: 🟢 Active
📝 Queued Q&A pairs: 3
🎯 Training batch size: 5
⏱️  Training interval: 180 seconds
🔄 Total online updates: 2
📈 Average confidence: 0.721
🚀 Recent confidence: 0.756
📊 Confidence improvement: +0.035 🎉
🔧 Worker thread: 🟢 Running
============================================================
💡 Next training session when 5 Q&A pairs are collected
🎯 Your medical model has been updated 2 times with chat data!
```

## 🔬 Research & Development

### Current Research Focus

1. **Multi-modal Medical Understanding**

   - Image and chart processing from medical PDFs
   - Integration of visual information with text

2. **Advanced Retrieval Techniques**

   - Hybrid search combining semantic and keyword matching
   - Graph-based knowledge representation

3. **Domain Adaptation**
   - Specialized models for different medical domains
   - Transfer learning from general medical knowledge

### Experimental Features

- **Uncertainty Quantification**: Better confidence estimation
- **Explanation Generation**: Providing reasoning for answers
- **Multi-language Support**: Extending to non-English medical documents

## 🤝 Contributing

We welcome contributions to improve the medical AI assistant!

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/medical-ai-assistant.git
cd medical-ai-assistant

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/
```

### Areas for Contribution

- **Model Improvements**: Better medical domain models
- **New Features**: Additional medical specialties
- **Performance**: Optimization and scaling
- **Documentation**: Examples and tutorials
- **Testing**: More comprehensive test coverage

## 🙏 Acknowledgments

### Inspiration and Collaboration

This project builds upon and integrates work from various open-source medical AI initiatives and academic research:

- **LangChain Community**: For the structured output parsing framework
- **Sentence Transformers**: For semantic embedding models
- **FAISS**: For efficient similarity search
- **PyTorch Community**: For the deep learning framework
- **Medical AI Research**: Various papers on medical NLP and RAG systems

### Data Sources

- **National Kidney Foundation**: AKI educational materials
- **Medical Literature**: Peer-reviewed articles and clinical guidelines
- **Open Medical Datasets**: For model training and validation

### Key Technologies

- **Retrieval-Augmented Generation (RAG)**: Core architecture pattern
- **Transformer Models**: For language understanding and generation
- **Vector Databases**: For efficient document retrieval
- **Online Learning**: For continuous model improvement

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Medical AI Assistant Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/medical-ai-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/medical-ai-assistant/discussions)
- **Email**: medical-ai-support@example.com

## 🗺️ Roadmap

### Version 2.0 (Planned)

- [ ] Multi-modal document processing (images, tables)
- [ ] Advanced medical specialty modules
- [ ] Real-time collaborative learning
- [ ] Web-based interface
- [ ] API service deployment

### Version 1.5 (In Development)

- [x] Enhanced online learning system
- [x] Structured output parsing
- [x] Confidence scoring and tracking
- [ ] Performance optimization
- [ ] Extended medical domain coverage

### Version 1.0 (Current)

- [x] Basic RAG implementation
- [x] PDF document processing
- [x] Medical QA model training
- [x] Interactive chat interface
- [x] Online learning capabilities

---

**🏥 Empowering medical professionals and patients with AI-driven insights and continuous learning.**

_Last updated: June 25, 2025_
