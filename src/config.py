"""
Configuration settings for the RAG system
"""

import os
from typing import Dict, Any

# Neo4j Configuration
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "password"),
    "database": os.getenv("NEO4J_DATABASE", "neo4j")
}

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Document Processing Configuration
DOCUMENT_CONFIG = {
    "pdf_folder": os.getenv("PDF_FOLDER", "./documents"),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
    "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "384"))
}

# Search Configuration
SEARCH_CONFIG = {
    "default_top_k": int(os.getenv("DEFAULT_TOP_K", "5")),
    "max_context_length": int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
}