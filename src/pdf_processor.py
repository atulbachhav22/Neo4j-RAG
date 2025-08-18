"""
PDF Processing and Knowledge Graph RAG System
Note: This requires external libraries not available in WebContainer
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

# These imports would be needed in a full Python environment:
# import PyPDF2 or pypdf
# from sentence_transformers import SentenceTransformer
# from neo4j import GraphDatabase
# import openai or transformers
# import numpy as np

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class PDFProcessor:
    """Handles PDF reading and text extraction"""
    
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        In a real implementation, you'd use PyPDF2 or pypdf:
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        """
        # Placeholder for demonstration
        return f"Sample text content from {pdf_path}"
    
    def process_folder(self) -> List[str]:
        """Process all PDF files in the folder"""
        texts = []
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_folder, filename)
                text = self.extract_text_from_pdf(pdf_path)
                texts.append({
                    'filename': filename,
                    'content': text,
                    'path': pdf_path
                })
        return texts

class TextChunker:
    """Handles text chunking strategies"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        text_length = len(text)
        
        for i in range(0, text_length, self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            # Create unique ID for chunk
            chunk_id = hashlib.md5(
                f"{metadata.get('filename', '')}{i}{chunk_text[:50]}".encode()
            ).hexdigest()
            
            chunk_metadata = {
                **metadata,
                'chunk_index': len(chunks),
                'start_char': i,
                'end_char': min(i + self.chunk_size, text_length)
            }
            
            chunks.append(DocumentChunk(
                id=chunk_id,
                content=chunk_text.strip(),
                metadata=chunk_metadata
            ))
            
            if i + self.chunk_size >= text_length:
                break
                
        return chunks

class EmbeddingGenerator:
    """Handles text embedding generation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        # In real implementation:
        # self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for text chunks"""
        for chunk in chunks:
            # In real implementation:
            # chunk.embedding = self.model.encode(chunk.content).tolist()
            
            # Placeholder - random embedding for demo
            import random
            chunk.embedding = [random.random() for _ in range(384)]
            
        return chunks

class Neo4jKnowledgeGraph:
    """Handles Neo4j database operations and knowledge graph creation"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username  
        self.password = password
        self.database = database
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test connection
            with self.driver.session(database=database) as session:
                session.run("RETURN 1")
            print(f"✅ Connected to Neo4j database: {database}")
        except ImportError:
            print("❌ Neo4j driver not installed. Run: pip install neo4j")
            self.driver = None
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None
        
    def get_session(self):
        """Get database session with specified database name"""
        if not self.driver:
            raise Exception("Neo4j driver not initialized")
        return self.driver.session(database=self.database)
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
        
    def create_document_nodes(self, chunks: List[DocumentChunk]):
        """Create document and chunk nodes in Neo4j"""
        if not self.driver:
            print("❌ Neo4j driver not available")
            return
        
        try:
            with self.get_session() as session:
                # Group chunks by document
                documents = {}
                for chunk in chunks:
                    filename = chunk.metadata.get('filename')
                    if filename not in documents:
                        documents[filename] = {
                            'path': chunk.metadata.get('path'),
                            'chunks': []
                        }
                    documents[filename]['chunks'].append(chunk)
                
                # Create document nodes
                for filename, doc_data in documents.items():
                    # Create document node
                    create_doc_query = """
                    MERGE (doc:Document {filename: $filename})
                    SET doc.path = $path,
                        doc.created_at = datetime(),
                        doc.chunk_count = $chunk_count
                    RETURN doc.filename as filename
                    """
                    
                    result = session.run(create_doc_query, {
                        'filename': filename,
                        'path': doc_data['path'],
                        'chunk_count': len(doc_data['chunks'])
                    })
                    
                    doc_result = result.single()
                    if doc_result:
                        print(f"📄 Created document node: {doc_result['filename']}")
                
                # Create chunk nodes and relationships
                chunk_count = 0
                for chunk in chunks:
                    create_chunk_query = """
                    MATCH (doc:Document {filename: $filename})
                    CREATE (chunk:Chunk {
                        id: $chunk_id,
                        content: $content,
                        chunk_index: $chunk_index,
                        start_char: $start_char,
                        end_char: $end_char,
                        content_length: $content_length
                    })
                    SET chunk.embedding = $embedding,
                        chunk.created_at = datetime()
                    CREATE (doc)-[:CONTAINS]->(chunk)
                    RETURN chunk.id as chunk_id
                    """
                    
                    result = session.run(create_chunk_query, {
                        'filename': chunk.metadata.get('filename'),
                        'chunk_id': chunk.id,
                        'content': chunk.content,
                        'chunk_index': chunk.metadata.get('chunk_index', 0),
                        'start_char': chunk.metadata.get('start_char', 0),
                        'end_char': chunk.metadata.get('end_char', 0),
                        'content_length': len(chunk.content),
                        'embedding': chunk.embedding or []
                    })
                    
                    chunk_result = result.single()
                    if chunk_result:
                        chunk_count += 1
                
                print(f"📝 Created {chunk_count} chunk nodes with relationships")
                
        except Exception as e:
            print(f"❌ Error creating document nodes: {e}")
            raise
    
    def create_semantic_relationships(self, chunks: List[DocumentChunk], similarity_threshold: float = 0.8):
        """Create relationships between semantically similar chunks"""
        if not self.driver:
            print("❌ Neo4j driver not available")
            return
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            with self.get_session() as session:
                # Get all chunks with embeddings
                get_chunks_query = """
                MATCH (c:Chunk)
                WHERE c.embedding IS NOT NULL AND size(c.embedding) > 0
                RETURN c.id as id, c.embedding as embedding
                """
                
                result = session.run(get_chunks_query)
                chunk_data = [(record['id'], record['embedding']) for record in result]
                
                if len(chunk_data) < 2:
                    print("⚠️ Not enough chunks with embeddings for similarity calculation")
                    return
                
                print(f"🔍 Calculating similarities for {len(chunk_data)} chunks...")
                
                # Calculate similarities and create relationships
                relationships_created = 0
                batch_size = 100
                
                for i in range(0, len(chunk_data), batch_size):
                    batch = chunk_data[i:i + batch_size]
                    
                    for j, (chunk1_id, embedding1) in enumerate(batch):
                        for k, (chunk2_id, embedding2) in enumerate(chunk_data[i+j+1:], i+j+1):
                            if chunk1_id == chunk2_id:
                                continue
                            
                            # Calculate cosine similarity
                            similarity = cosine_similarity(
                                [embedding1], [embedding2]
                            )[0][0]
                            
                            if similarity > similarity_threshold:
                                create_similarity_query = """
                                MATCH (c1:Chunk {id: $chunk1_id})
                                MATCH (c2:Chunk {id: $chunk2_id})
                                CREATE (c1)-[:SIMILAR_TO {
                                    similarity: $similarity,
                                    created_at: datetime()
                                }]->(c2)
                                """
                                
                                session.run(create_similarity_query, {
                                    'chunk1_id': chunk1_id,
                                    'chunk2_id': chunk2_id,
                                    'similarity': float(similarity)
                                })
                                
                                relationships_created += 1
                
                print(f"🔗 Created {relationships_created} similarity relationships")
                
        except ImportError as e:
            print(f"❌ Missing required packages for similarity calculation: {e}")
            print("Install with: pip install numpy scikit-learn")
        except Exception as e:
            print(f"❌ Error creating semantic relationships: {e}")
            raise
    
    def extract_entities_and_create_graph(self, chunks: List[DocumentChunk]):
        """Extract named entities and create knowledge graph"""
        if not self.driver:
            print("❌ Neo4j driver not available")
            return
        
        try:
            import spacy
            
            # Load spaCy model
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("❌ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                return
            
            with self.get_session() as session:
                entities_created = 0
                relationships_created = 0
                
                for chunk in chunks:
                    # Process text with spaCy
                    doc = nlp(chunk.content)
                    
                    # Extract entities
                    for ent in doc.ents:
                        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT']:
                            # Create entity node
                            create_entity_query = """
                            MERGE (e:Entity {
                                name: $name,
                                type: $entity_type
                            })
                            SET e.created_at = coalesce(e.created_at, datetime())
                            RETURN e.name as name
                            """
                            
                            result = session.run(create_entity_query, {
                                'name': ent.text.strip(),
                                'entity_type': ent.label_
                            })
                            
                            if result.single():
                                entities_created += 1
                            
                            # Create relationship between chunk and entity
                            create_mention_query = """
                            MATCH (c:Chunk {id: $chunk_id})
                            MATCH (e:Entity {name: $entity_name, type: $entity_type})
                            MERGE (c)-[:MENTIONS {
                                start_char: $start_char,
                                end_char: $end_char,
                                confidence: $confidence
                            }]->(e)
                            """
                            
                            session.run(create_mention_query, {
                                'chunk_id': chunk.id,
                                'entity_name': ent.text.strip(),
                                'entity_type': ent.label_,
                                'start_char': ent.start_char,
                                'end_char': ent.end_char,
                                'confidence': 1.0  # spaCy doesn't provide confidence scores
                            })
                            
                            relationships_created += 1
                
                # Create co-occurrence relationships between entities
                cooccurrence_query = """
                MATCH (c:Chunk)-[:MENTIONS]->(e1:Entity)
                MATCH (c)-[:MENTIONS]->(e2:Entity)
                WHERE e1 <> e2
                WITH e1, e2, count(c) as cooccurrence_count
                WHERE cooccurrence_count > 1
                MERGE (e1)-[:CO_OCCURS_WITH {
                    frequency: cooccurrence_count,
                    created_at: datetime()
                }]->(e2)
                """
                
                result = session.run(cooccurrence_query)
                summary = result.consume()
                
                print(f"👥 Created {entities_created} entity nodes")
                print(f"🔗 Created {relationships_created} mention relationships")
                print(f"🤝 Created {summary.counters.relationships_created} co-occurrence relationships")
                
        except ImportError as e:
            print(f"❌ Missing required packages for entity extraction: {e}")
            print("Install with: pip install spacy")
        except Exception as e:
            print(f"❌ Error extracting entities: {e}")
            raise
    
    def get_document_stats(self):
        """Get statistics about the knowledge graph"""
        if not self.driver:
            return {}
        
        try:
            with self.get_session() as session:
                stats_query = """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH ()-[s:SIMILAR_TO]->()
                RETURN 
                    count(DISTINCT d) as document_count,
                    count(DISTINCT c) as chunk_count,
                    count(DISTINCT e) as entity_count,
                    count(DISTINCT s) as similarity_relationships
                """
                
                result = session.run(stats_query)
                stats = result.single()
                
                return {
                    'documents': stats['document_count'],
                    'chunks': stats['chunk_count'],
                    'entities': stats['entity_count'],
                    'similarity_relationships': stats['similarity_relationships']
                }
                
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}
    
    def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5):
        """Search for similar chunks using vector similarity"""
        if not self.driver:
            return []
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            with self.get_session() as session:
                # Get all chunks with embeddings
                get_chunks_query = """
                MATCH (c:Chunk)
                WHERE c.embedding IS NOT NULL AND size(c.embedding) > 0
                RETURN c.id as id, c.content as content, c.embedding as embedding
                """
                
                result = session.run(get_chunks_query)
                chunks = []
                
                for record in result:
                    similarity = cosine_similarity(
                        [query_embedding], [record['embedding']]
                    )[0][0]
                    
                    chunks.append({
                        'id': record['id'],
                        'content': record['content'],
                        'similarity': float(similarity)
                    })
                
                # Sort by similarity and return top_k
                chunks.sort(key=lambda x: x['similarity'], reverse=True)
                return chunks[:top_k]
                
        except Exception as e:
            print(f"❌ Error searching similar chunks: {e}")
            return []
    

class SemanticSearch:
    """Handles semantic search functionality"""
    
    def __init__(self, neo4j_kg: Neo4jKnowledgeGraph, embedding_generator: EmbeddingGenerator):
        self.neo4j_kg = neo4j_kg
        self.embedding_generator = embedding_generator
    
    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Perform semantic search using embeddings"""
        # Generate embedding for query
        # query_embedding = self.embedding_generator.model.encode(query)
        
        # Search in Neo4j using vector similarity
        search_query = """
        MATCH (chunk:Chunk)
        RETURN chunk, gds.similarity.cosine($query_embedding, chunk.embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        # Placeholder results
        print(f"Searching for: '{query}' with top_k={top_k}")
        return []  # Would return actual results from Neo4j

class RAGSystem:
    """Main RAG system orchestrating all components"""
    
    def __init__(self, pdf_folder: str, neo4j_config: Dict[str, str]):
        self.pdf_processor = PDFProcessor(pdf_folder)
        self.chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.knowledge_graph = Neo4jKnowledgeGraph(**neo4j_config)
        self.semantic_search = SemanticSearch(self.knowledge_graph, self.embedding_generator)
        
        # In real implementation, initialize LLM:
        # self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def ingest_documents(self):
        """Complete document ingestion pipeline"""
        print("Starting document ingestion...")
        
        # 1. Extract text from PDFs
        documents = self.pdf_processor.process_folder()
        print(f"Extracted text from {len(documents)} documents")
        
        # 2. Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_text(doc['content'], {
                'filename': doc['filename'],
                'path': doc['path']
            })
            all_chunks.extend(chunks)
        print(f"Created {len(all_chunks)} chunks")
        
        # 3. Generate embeddings
        all_chunks = self.embedding_generator.generate_embeddings(all_chunks)
        print("Generated embeddings for all chunks")
        
        # 4. Store in Neo4j and create knowledge graph
        self.knowledge_graph.create_document_nodes(all_chunks)
        self.knowledge_graph.create_semantic_relationships(all_chunks)
        self.knowledge_graph.extract_entities_and_create_graph(all_chunks)
        print("Stored documents and created knowledge graph")
    
    def query(self, question: str) -> str:
        """Answer question using RAG approach"""
        print(f"Processing query: '{question}'")
        
        # 1. Semantic search
        relevant_chunks = self.semantic_search.search(question, top_k=5)
        
        # 2. Prepare context for LLM
        context = "\n\n".join([chunk.content for chunk in relevant_chunks])
        
        # 3. Generate response with LLM
        prompt = f"""
        Context from documents:
        {context}
        
        Question: {question}
        
        Please provide a comprehensive answer based on the context above.
        """
        
        # In real implementation:
        # response = self.llm.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.choices[0].message.content
        
        return f"Generated response for: '{question}' using {len(relevant_chunks)} relevant chunks"

# Example usage
def main():
    """Main function demonstrating the RAG system"""
    
    # Configuration
    pdf_folder = "./documents"
    neo4j_config = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j", 
        "password": "password"
    }
    
    # Initialize RAG system
    rag_system = RAGSystem(pdf_folder, neo4j_config)
    
    # Ingest documents
    rag_system.ingest_documents()
    
    # Example queries
    questions = [
        "What are the main topics discussed in the documents?",
        "Can you summarize the key findings?",
        "What relationships exist between different concepts?"
    ]
    
    for question in questions:
        answer = rag_system.query(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")

if __name__ == "__main__":
    main()