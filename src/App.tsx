import React, { useState } from 'react';
import { 
  FileText, 
  Database, 
  Brain, 
  Search, 
  MessageSquare, 
  Upload,
  Settings,
  Play,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react';

interface ProcessingStep {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  icon: React.ReactNode;
}

function App() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');

  const processingSteps: ProcessingStep[] = [
    {
      id: 'upload',
      title: 'PDF Upload & Text Extraction',
      description: 'Upload PDF files and extract text content using PyPDF2',
      status: 'completed',
      icon: <FileText className="w-5 h-5" />
    },
    {
      id: 'chunk',
      title: 'Text Chunking',
      description: 'Split documents into overlapping chunks for better context',
      status: 'completed',
      icon: <Settings className="w-5 h-5" />
    },
    {
      id: 'embed',
      title: 'Generate Embeddings',
      description: 'Create semantic embeddings using sentence-transformers',
      status: 'processing',
      icon: <Brain className="w-5 h-5" />
    },
    {
      id: 'store',
      title: 'Store in Neo4j',
      description: 'Save chunks and create knowledge graph relationships',
      status: 'pending',
      icon: <Database className="w-5 h-5" />
    },
    {
      id: 'search',
      title: 'Semantic Search',
      description: 'Find relevant chunks using vector similarity',
      status: 'pending',
      icon: <Search className="w-5 h-5" />
    },
    {
      id: 'llm',
      title: 'LLM Response',
      description: 'Generate final answer using retrieved context',
      status: 'pending',
      icon: <MessageSquare className="w-5 h-5" />
    }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'processing':
        return <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <div className="w-4 h-4 border-2 border-gray-300 rounded-full" />;
    }
  };

  const simulateProcessing = () => {
    setIsProcessing(true);
    // Simulate processing steps
    setTimeout(() => {
      setResponse("Based on the processed documents, I found relevant information about your query. The system retrieved 3 relevant chunks from the knowledge graph and generated this response using the LLM integration.");
      setIsProcessing(false);
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Brain className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">PDF RAG Knowledge Graph System</h1>
              <p className="text-sm text-gray-600">Intelligent document processing with semantic search</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Processing Pipeline */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Processing Pipeline</h2>
              
              <div className="space-y-4">
                {processingSteps.map((step, index) => (
                  <div key={step.id} className="flex items-start space-x-4 p-4 rounded-lg border border-gray-100 hover:border-gray-200 transition-colors">
                    <div className="flex-shrink-0 p-2 bg-gray-50 rounded-lg">
                      {step.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-medium text-gray-900">{step.title}</h3>
                        {getStatusIcon(step.status)}
                      </div>
                      <p className="text-sm text-gray-600 mt-1">{step.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* System Requirements */}
            <div className="bg-white rounded-xl shadow-sm border p-6 mt-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">System Requirements</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h3 className="font-medium text-blue-900 mb-2">Python Environment</h3>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>• Python 3.9+</li>
                    <li>• pip package manager</li>
                    <li>• Virtual environment</li>
                  </ul>
                </div>
                <div className="p-4 bg-green-50 rounded-lg">
                  <h3 className="font-medium text-green-900 mb-2">Required Packages</h3>
                  <ul className="text-sm text-green-800 space-y-1">
                    <li>• PyPDF2 / pypdf</li>
                    <li>• sentence-transformers</li>
                    <li>• neo4j driver</li>
                    <li>• openai / transformers</li>
                  </ul>
                </div>
                <div className="p-4 bg-purple-50 rounded-lg">
                  <h3 className="font-medium text-purple-900 mb-2">Database</h3>
                  <ul className="text-sm text-purple-800 space-y-1">
                    <li>• Neo4j 5.11+</li>
                    <li>• Vector index support</li>
                    <li>• Graph algorithms</li>
                  </ul>
                </div>
                <div className="p-4 bg-orange-50 rounded-lg">
                  <h3 className="font-medium text-orange-900 mb-2">API Keys</h3>
                  <ul className="text-sm text-orange-800 space-y-1">
                    <li>• OpenAI API key</li>
                    <li>• Or local LLM setup</li>
                    <li>• Environment variables</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Query Interface */}
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Query Interface</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Ask a question about your documents
                  </label>
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="What are the main topics discussed in the documents?"
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                    rows={3}
                  />
                </div>
                
                <button
                  onClick={simulateProcessing}
                  disabled={isProcessing || !query.trim()}
                  className="w-full flex items-center justify-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isProcessing ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>Search & Generate</span>
                    </>
                  )}
                </button>
              </div>

              {response && (
                <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-medium text-gray-900 mb-2">Response</h3>
                  <p className="text-sm text-gray-700">{response}</p>
                </div>
              )}
            </div>

            {/* Installation Guide */}
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Start</h2>
              
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">1</div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">Install Dependencies</p>
                    <code className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded mt-1 block">pip install -r requirements.txt</code>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">2</div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">Setup Neo4j</p>
                    <p className="text-xs text-gray-600 mt-1">Run neo4j_setup.cypher queries</p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">3</div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">Configure Environment</p>
                    <p className="text-xs text-gray-600 mt-1">Set API keys and database credentials</p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">4</div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">Run System</p>
                    <code className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded mt-1 block">python pdf_processor.py</code>
                  </div>
                </div>
              </div>
            </div>

            {/* WebContainer Limitation Notice */}
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
              <div className="flex items-start space-x-3">
                <Info className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-medium text-amber-800">WebContainer Limitation</h3>
                  <p className="text-sm text-amber-700 mt-1">
                    This demo shows the system architecture. Full implementation requires a Python environment with external packages.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;