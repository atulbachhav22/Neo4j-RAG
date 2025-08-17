# PDF RAG Knowledge Graph System

A comprehensive system for processing PDF documents, creating knowledge graphs, and performing semantic search with LLM-powered responses.

## Features

- **PDF Processing**: Extract text from PDF files in a specified folder
- **Text Chunking**: Split documents into manageable chunks with overlap
- **Embeddings**: Generate semantic embeddings for text chunks
- **Knowledge Graph**: Store documents and relationships in Neo4j
- **Semantic Search**: Find relevant content using vector similarity
- **LLM Integration**: Generate responses using retrieved context

## Architecture

```
PDF Files → Text Extraction → Chunking → Embeddings → Neo4j Knowledge Graph
                                                              ↓
User Query → Semantic Search → Context Retrieval → LLM → Final Response
```

## System Requirements

This system requires a full Python environment with package management capabilities. The current WebContainer environment only supports Python standard library.

### Required Software

1. **Python 3.9+** with pip
2. **Neo4j Database** (version 5.11+ for vector indexes)
3. **OpenAI API Key** (or local LLM setup)

### Installation in Full Python Environment

```bash
# Clone or download the code
# Install required packages
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm

# Set up Neo4j database
# Run the queries in neo4j_setup.cypher
```

### Environment Variables

Create a `.env` file:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# Document Processing
PDF_FOLDER=./documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.8

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Search Configuration
DEFAULT_TOP_K=5
MAX_CONTEXT_LENGTH=4000
```

## Usage

### 1. Prepare Documents
Place your PDF files in the `documents` folder (or path specified in `PDF_FOLDER`).

### 2. Start Neo4j Database
Ensure Neo4j is running and accessible at the configured URI.

### 3. Run Document Ingestion
```python
from pdf_processor import RAGSystem

# Initialize system
rag_system = RAGSystem("./documents", {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "password"
})

# Ingest documents
rag_system.ingest_documents()
```

### 4. Query the System
```python
# Ask questions
answer = rag_system.query("What are the main topics in the documents?")
print(answer)
```

## Knowledge Graph Structure

### Nodes
- **Document**: Represents PDF files
  - `filename`: Original PDF filename
  - `path`: File path
  - `created_at`: Ingestion timestamp

- **Chunk**: Text segments from documents
  - `id`: Unique chunk identifier
  - `content`: Text content
  - `chunk_index`: Position in document
  - `embedding`: Vector representation

### Relationships
- **CONTAINS**: Document contains chunks
- **SIMILAR_TO**: Semantic similarity between chunks
- **Entity relationships** (Person, Organization, Concept nodes)

## Advanced Features

### Custom Chunking Strategies
Modify the `TextChunker` class to implement different chunking approaches:
- Sentence-based chunking
- Paragraph-based chunking
- Semantic chunking using sentence transformers

### Entity Extraction
Integrate named entity recognition to extract:
- People, organizations, locations
- Key concepts and topics
- Relationships between entities

### Advanced Search
Implement hybrid search combining:
- Vector similarity search
- Graph traversal queries
- Full-text search capabilities

## Performance Optimization

1. **Batch Processing**: Process documents in batches
2. **Parallel Processing**: Use multiprocessing for embedding generation
3. **Neo4j Indexing**: Create appropriate indexes for query performance
4. **Caching**: Cache embeddings and frequent queries

## Troubleshooting

### Common Issues

1. **Memory Usage**: Large PDFs may require chunking adjustments
2. **Neo4j Connection**: Verify database credentials and connectivity
3. **Embedding Generation**: Ensure sufficient GPU memory for large models
4. **PDF Extraction**: Some PDFs may require OCR for scanned documents

## Limitations in WebContainer

This implementation is conceptual and demonstrates the architecture. To run in production:

1. Use a full Python environment with pip
2. Install required packages from `requirements.txt`
3. Set up Neo4j database with vector support
4. Configure OpenAI or local LLM access
5. Handle error cases and edge conditions

## Future Enhancements

- Support for additional document formats (Word, HTML, etc.)
- Real-time document updates and incremental indexing
- Multi-modal support (images, tables in PDFs)
- Advanced reasoning capabilities with graph neural networks
- Explainable AI features showing retrieval paths