# Enhanced RAG with Output Parsers

This implementation provides a structured output parsing system for your RAG pipeline that delivers GPT-like formatted responses using LangChain's output parsers and Pydantic models.

## 🎯 Key Features

### 1. Structured Output Formatting

- **H3 headings (###)** for main topics
- **Bullet points with bold key terms** for easy scanning
- **Consistent markdown formatting** like professional AI assistants
- **Confidence levels** for transparency

### 2. Output Parser Integration

- Uses **LangChain PydanticOutputParser** for structured data
- **Pydantic models** for data validation and formatting
- **Type safety** with proper schema definitions

### 3. System Prompt Engineering

Based on your specification, the system includes:

- System instructions for consistent formatting
- Few-shot examples for better output quality
- Context and question integration
- Structured response guidelines

## 📁 File Structure

```
src/rag/
├── enhanced_rag_with_parsers.py    # Main enhanced RAG system
├── simple_rag.py                   # Original RAG system
└── __init__.py

enhanced_chat_with_parsers.py       # Interactive chat interface
test_enhanced_rag.py               # Testing and demonstration
requirements.txt                   # Updated dependencies
```

## 🔧 Installation

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **New dependencies added**:

- `langchain>=0.1.0` - Core LangChain functionality
- `langchain-core>=0.1.0` - Output parsers and prompts
- `pydantic>=2.0.0` - Data validation and models

## 🚀 Usage

### Quick Start

```python
from src.rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers

# Initialize the system
rag = EnhancedRAGWithParsers()

# Build index from your documents
documents = [
    {
        'content': 'Your medical document content...',
        'metadata': {'source': 'medical_guide.pdf', 'page': 1}
    }
]
rag.build_index(documents)

# Ask questions and get structured responses
response = rag.ask("What are the treatment options for AKI?")
print(response['answer'])
```

### Interactive Chat

```bash
python enhanced_chat_with_parsers.py
```

### Testing

```bash
python test_enhanced_rag.py
```

## 📋 Output Format Examples

### Treatment Response

```markdown
### Treatment Options

- **Hospital Care**: Treatment typically requires hospitalization for monitoring and management.
- **Dialysis**: May be necessary in severe cases to support kidney function.
- **Address Root Cause**: Primary focus on treating the underlying condition causing the problem.

_Confidence: High_
```

### Symptoms Response

```markdown
### Signs and Symptoms

- **Decreased Urination**: Reduced urine output or frequency.
- **Swelling**: Fluid retention causing swelling in legs, ankles, or face.
- **Fatigue**: Persistent tiredness and lack of energy.
- **Nausea**: Feeling sick to the stomach, possibly with vomiting.

_Confidence: High_
```

## 🏗️ Architecture

### Core Components

1. **MedicalResponse Model** (Pydantic)

   - Structured data model for responses
   - Validation and formatting methods
   - Confidence level tracking

2. **EnhancedRAGWithParsers Class**

   - Document indexing with FAISS
   - Semantic search with sentence transformers
   - Question categorization and routing
   - Structured response generation

3. **Output Parsers**
   - PydanticOutputParser for structured data
   - Automatic formatting to markdown
   - Type validation and error handling

### Question Categories

The system automatically categorizes questions and provides specialized responses:

- **Treatment**: "What are the treatment options...?"
- **Symptoms**: "What are the symptoms...?"
- **Diagnosis**: "How is ... diagnosed?"
- **Causes**: "What causes...?"
- **Benefits**: "What are the benefits...?"
- **General**: Fallback for other questions

## 🔬 System Prompt Template

Your specification is implemented as follows:

````python
system_prompt = '''
[SYSTEM]
You are an AI assistant that ALWAYS returns answers in clean, markdown-formatted sections.
Strict rules:
1. Use exactly one H3 heading (###) for the title.
2. Use bullet points (`- `) for each item.
3. Bold the key term in each bullet with `**`.
4. No intros, no conclusions, no numbering—just the heading and bullets.
5. If the question asks for code, wrap code in triple backticks (```).

[FEW-SHOT EXAMPLES]
### Example 1
Context:
Document 1: "Turmeric has anti-inflammatory properties."
Document 2: "Turmeric is a powerful antioxidant."

Question:
What are the benefits of turmeric?

Answer:
### Benefits of Turmeric
- **Anti-inflammatory**: Reduces swelling and pain.
- **Antioxidant**: Fights free radicals and prevents cell damage.

[USER INPUT]
Context:
{retrieved_docs}

Question:
{user_query}

Answer:
'''
````

## 📊 Response Structure

Each response includes:

```python
{
    'answer': 'Formatted markdown response',
    'confidence': 0.85,  # Similarity score
    'sources': 3,        # Number of documents used
    'retrieved_docs': [...],  # Source documents
    'formatted': True    # Indicates structured formatting
}
```

## 💡 Advanced Features

### 1. Debug Mode

```python
response = rag.ask_with_system_prompt(question)
print(response['full_prompt'])  # See the complete prompt
```

### 2. Custom Confidence Levels

- **High**: >0.6 similarity score
- **Moderate**: 0.3-0.6 similarity score
- **Low**: <0.3 similarity score

### 3. Document Chunking

- Automatic text splitting for better retrieval
- Overlapping chunks for context preservation
- Metadata preservation for source tracking

## 🎛️ Configuration Options

### Model Settings

```python
rag = EnhancedRAGWithParsers(
    retrieval_model="sentence-transformers/all-MiniLM-L6-v2",
    generation_model="microsoft/DialoGPT-medium"
)
```

### Retrieval Parameters

```python
response = rag.ask(question, top_k=5)  # Retrieve top 5 documents
```

### Chunk Settings

```python
chunks = self._split_into_chunks(text, chunk_size=1000, overlap=200)
```

## 🔄 Integration with Existing Code

### Replace Simple RAG

```python
# Old way
from src.rag.simple_rag import SimpleRAG
rag = SimpleRAG()

# New way
from src.rag.enhanced_rag_with_parsers import EnhancedRAGWithParsers
rag = EnhancedRAGWithParsers()
```

### Migration Steps

1. Update imports
2. Replace initialization
3. Use new response format
4. Update UI to handle structured responses

## 🚨 Error Handling

The system includes robust error handling:

```python
try:
    response = rag.ask(question)
except Exception as e:
    response = {
        'answer': f"### Error Processing Question\n\n- **Error**: {str(e)}",
        'confidence': 0.0,
        'sources': 0,
        'formatted': True
    }
```

## 📈 Performance Considerations

- **Chunking**: Smaller chunks (1000 chars) for better precision
- **Overlap**: 200 character overlap for context continuity
- **Caching**: FAISS index saved/loaded for persistence
- **Batch Processing**: Efficient embedding generation

## 🎯 Future Enhancements

1. **Integration with OpenAI/Claude**

   - Replace template-based responses with LLM generation
   - Use the system prompt with actual language models

2. **Custom Parsers**

   - Add parsers for specific medical domains
   - Support for tables, lists, and code blocks

3. **Multi-modal Support**

   - Handle images and charts from PDFs
   - Visual formatting preservation

4. **Advanced Retrieval**
   - Hybrid search (semantic + keyword)
   - Re-ranking for better relevance

## 🎉 Example Output

Running the test shows structured outputs like:

```
📋 Test 1: Treatment
❓ Question: What are the treatment options for AKI?
🤖 Structured Response:
### Treatment Options
- **Hospital Care**: Treatment typically requires hospitalization for monitoring and management.
- **Dialysis**: May be necessary in severe cases to support kidney function.
- **Address Root Cause**: Primary focus on treating the underlying condition causing the problem.
*Confidence: High*
📊 Confidence: 0.622 | Sources: 3
```

This implementation gives you professional, GPT-like formatting while maintaining full control over the content and sources!
