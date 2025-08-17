import React from 'react';
import { ArrowRight, Database, FileText, Brain, Search, MessageSquare } from 'lucide-react';

export const SystemArchitecture: React.FC = () => {
  const architectureSteps = [
    {
      icon: <FileText className="w-8 h-8 text-blue-600" />,
      title: "PDF Processing",
      description: "Extract text from PDF files using PyPDF2",
      details: ["Read PDF files from folder", "Extract text content", "Handle multiple documents"]
    },
    {
      icon: <Settings className="w-8 h-8 text-green-600" />,
      title: "Text Chunking", 
      description: "Split text into manageable chunks",
      details: ["Configurable chunk size", "Overlap for context", "Preserve semantic meaning"]
    },
    {
      icon: <Brain className="w-8 h-8 text-purple-600" />,
      title: "Embeddings",
      description: "Generate semantic embeddings",
      details: ["sentence-transformers", "Vector representations", "Semantic similarity"]
    },
    {
      icon: <Database className="w-8 h-8 text-red-600" />,
      title: "Neo4j Storage",
      description: "Store in knowledge graph",
      details: ["Document nodes", "Chunk relationships", "Vector indexing"]
    },
    {
      icon: <Search className="w-8 h-8 text-orange-600" />,
      title: "Semantic Search",
      description: "Find relevant content",
      details: ["Vector similarity", "Graph traversal", "Context retrieval"]
    },
    {
      icon: <MessageSquare className="w-8 h-8 text-indigo-600" />,
      title: "LLM Response",
      description: "Generate final answer",
      details: ["OpenAI integration", "Context-aware", "Natural language"]
    }
  ];

  return (
    <div className="bg-white rounded-xl shadow-sm border p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">System Architecture</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {architectureSteps.map((step, index) => (
          <div key={index} className="relative">
            <div className="bg-gray-50 rounded-lg p-6 h-full">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-white rounded-lg shadow-sm">
                  {step.icon}
                </div>
                <h3 className="font-semibold text-gray-900">{step.title}</h3>
              </div>
              
              <p className="text-sm text-gray-600 mb-4">{step.description}</p>
              
              <ul className="space-y-1">
                {step.details.map((detail, idx) => (
                  <li key={idx} className="text-xs text-gray-500 flex items-center">
                    <div className="w-1 h-1 bg-gray-400 rounded-full mr-2" />
                    {detail}
                  </li>
                ))}
              </ul>
            </div>
            
            {index < architectureSteps.length - 1 && (
              <div className="hidden lg:block absolute top-1/2 -right-3 transform -translate-y-1/2">
                <ArrowRight className="w-6 h-6 text-gray-400" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};