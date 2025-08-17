#!/usr/bin/env python3
"""
Main script to run the PDF RAG Knowledge Graph System
This demonstrates how the system would be executed in a full Python environment
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'PyPDF2',
        'sentence_transformers', 
        'neo4j',
        'openai',
        'numpy',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_environment():
    """Check environment variables and configuration"""
    required_env_vars = [
        'NEO4J_URI',
        'NEO4J_USERNAME', 
        'NEO4J_PASSWORD',
        'OPENAI_API_KEY'
    ]
    
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n🔧 Create a .env file with these variables")
        return False
    
    print("✅ Environment variables configured")
    return True

def check_neo4j_connection():
    """Test Neo4j database connection"""
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('NEO4J_URI')
        username = os.getenv('NEO4J_USERNAME')
        password = os.getenv('NEO4J_PASSWORD')
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        
        driver.close()
        print("✅ Neo4j connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("🔧 Ensure Neo4j is running and credentials are correct")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = ['documents', 'logs', 'cache']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created/verified directory: {directory}")

def run_ingestion_pipeline():
    """Run the document ingestion pipeline"""
    try:
        # This would import and run the actual system
        # from src.pdf_processor import RAGSystem
        
        print("🚀 Starting document ingestion pipeline...")
        
        # Initialize system
        print("   📋 Initializing RAG system...")
        
        # Process documents
        print("   📄 Processing PDF documents...")
        
        # Generate embeddings
        print("   🧠 Generating embeddings...")
        
        # Store in Neo4j
        print("   💾 Storing in Neo4j knowledge graph...")
        
        # Create relationships
        print("   🔗 Creating semantic relationships...")
        
        print("✅ Document ingestion completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False

def run_query_interface():
    """Start the interactive query interface"""
    print("\n🤖 Starting interactive query interface...")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            query = input("❓ Enter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("🔍 Searching knowledge graph...")
            print("🧠 Generating response with LLM...")
            
            # This would call the actual RAG system
            # response = rag_system.query(query)
            
            # Simulated response
            response = f"Based on the documents in the knowledge graph, here's what I found regarding '{query}': [This would be the actual LLM-generated response based on retrieved context]"
            
            print(f"\n💬 Response: {response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Query failed: {e}")

def main():
    """Main execution function"""
    print("🚀 PDF RAG Knowledge Graph System")
    print("=" * 40)
    
    # Check system requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment configuration
    if not check_environment():
        sys.exit(1)
    
    # Test database connection
    if not check_neo4j_connection():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Check if documents exist
    documents_dir = Path('documents')
    pdf_files = list(documents_dir.glob('*.pdf'))
    
    if not pdf_files:
        print("⚠️  No PDF files found in documents/ directory")
        print("📁 Please add PDF files to process")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files to process")
    
    # Run ingestion pipeline
    if not run_ingestion_pipeline():
        sys.exit(1)
    
    # Start query interface
    run_query_interface()
    
    print("\n👋 Thank you for using the PDF RAG Knowledge Graph System!")

if __name__ == "__main__":
    main()