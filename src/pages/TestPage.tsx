import React, { useState } from 'react';
import { ReactFlow, Node } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { RefreshCw, Plus, Trash2 } from 'lucide-react';
import { nodeTypes, defaultTextWithKeywordsNodeData } from '../nodes';
import type { Word, Keyword, TextWithKeywordsNodeData } from '../nodes/types';

// Test data generator
const generateTestData = (): TextWithKeywordsNodeData => {
  const testPhrases = [
    "the dog is eating a sandwich by paul cezanne",
    "the cat is playing with a ball by vincent van gogh",
    "light bright shining, 2d, futurism, claude monet",
    "the moon is rising, abstract expressionism, rembrandt",
    "a beautiful sunset cubism, picasso, impressionism, edgar degas"
  ];
  
  const randomPhrase = testPhrases[Math.floor(Math.random() * testPhrases.length)];
  const words: Word[] = randomPhrase.split(' ').map(word => ({ value: word }));
  
// Simulate some keywords for testing
const mockKeywords: Keyword[] = [
    {
        entryId: '1',
        value: 'quantum',
        databaseValue: 'Quantum Computing',
        images: ['image1.jpg', 'image2.jpg'],
        isArtist: false,
        type: 'technology',
        aliases: [],
        descriptions: [
            {
                source: 'wikipedia',
                description: 'A type of computing that uses quantum-mechanical phenomena',
                date: '2023-10-01',
            },
            {
                source: 'synth',
                description: 'Recent advancements in quantum computing algorithms',
                date: '2023-09-15',
            }
        ],
        relatedKeywordIds: ['2', '3', '4'],
        relatedKeywordStrings: ['computing', 'physics', 'technology'],
    },
    {
        entryId: '2',
        value: 'machine learning',
        databaseValue: 'Machine Learning',
        images: ['image3.jpg', 'image4.jpg'],
        isArtist: false,
        type: 'technology',
        aliases: [],
        descriptions: [
            {
                source: 'wikipedia',
                description: 'A subset of artificial intelligence that enables systems to learn from data',
                date: '2023-10-01',
            },
            {
                source: 'arxiv',
                description: 'Recent advancements in machine learning algorithms and applications',
                date: '2023-09-15',
            }
        ],
        relatedKeywordIds: ['5', '6', '7'],
        relatedKeywordStrings: ['AI', 'algorithms', 'data science'],
    },
];
  
  // Start with default data and override with test data
  return {
    ...defaultTextWithKeywordsNodeData,
    content: randomPhrase,
    words: words,
    similarTexts: mockKeywords,
    hasNoKeywords: false,
    hasNoSimilarTexts: false,
    provenance: Math.random() > 0.5 ? 'ai' : 'user',
  };
};

const TestPage: React.FC = () => {
  const [nodes, setNodes] = useState<Node<TextWithKeywordsNodeData>[]>([
    {
      id: '1',
      type: 'textwithkeywords', // Use the correct type name from nodeTypes
      position: { x: 250, y: 100 },
      data: generateTestData(),
    },
  ]);
  
  const [refreshCount, setRefreshCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  // Refresh node data
  const handleRefresh = async () => {
    setIsLoading(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    setNodes(currentNodes => 
      currentNodes.map(node => ({
        ...node,
        data: generateTestData(),
      }))
    );
    
    setRefreshCount(prev => prev + 1);
    setIsLoading(false);
  };

  // Add new node
  const handleAddNode = () => {
    const newNode: Node<TextWithKeywordsNodeData> = {
      id: `${nodes.length + 1}`,
      type: 'textwithkeywords', // Use the correct type name
      position: { 
        x: 250 + (nodes.length % 3) * 300, 
        y: 100 + Math.floor(nodes.length / 3) * 250 
      },
      data: generateTestData(),
    };
    
    setNodes(currentNodes => [...currentNodes, newNode]);
  };

  // Clear all nodes
  const handleClear = () => {
    setNodes([]);
    setRefreshCount(0);
  };

  // Toggle AI/User provenance for all nodes
  const toggleProvenance = () => {
    setNodes(currentNodes =>
      currentNodes.map(node => ({
        ...node,
        data: {
          ...node.data,
          provenance: node.data.provenance === 'ai' ? 'user' : 'ai',
        },
      }))
    );
  };

  // Use the default data as a starting point
  const handleUseDefaultData = () => {
    setNodes([{
      id: '1',
      type: 'textwithkeywords',
      position: { x: 250, y: 100 },
      data: { ...defaultTextWithKeywordsNodeData },
    }]);
  };
  // Add mock intersections to test merge functionality
  const addIntersections = () => {
    setNodes(currentNodes => {
      if (currentNodes.length < 2) return currentNodes;
      
      // Make first two nodes intersect with each other
      return currentNodes.map((node, index) => {
        if (index === 0) {
          return {
            ...node,
            data: {
              ...node.data,
              intersections: [{
                id: currentNodes[1].id,
                position: currentNodes[1].position,
                content: currentNodes[1].data.content || ''
              }]
            }
          };
        } else if (index === 1) {
          return {
            ...node,
            data: {
              ...node.data,
              intersections: [{
                id: currentNodes[0].id,
                position: currentNodes[0].position,
                content: currentNodes[0].data.content || ''
              }]
            }
          };
        }
        return node;
      });
    });
  };

  return (
    <div style={{ width: '100vw', height: 'calc(100vh - 48px)', marginTop: '48px', display: 'flex', flexDirection: 'column' }}>
      {/* Control Panel */}
      <div style={{
        padding: '20px',
        backgroundColor: '#f5f5f5',
        borderBottom: '1px solid #ddd',
        display: 'flex',
        alignItems: 'center',
        gap: '15px',
        flexWrap: 'wrap',
      }}>
        <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 'bold' }}>
          TextWithKeywordsNode Test Page
        </h1>
        
        <div style={{ display: 'flex', gap: '10px', marginLeft: 'auto' }}>
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: '1px solid #3b82f6',
              backgroundColor: '#3b82f6',
              color: 'white',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              opacity: isLoading ? 0.6 : 1,
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => !isLoading && (e.currentTarget.style.backgroundColor = '#2563eb')}
            onMouseLeave={(e) => !isLoading && (e.currentTarget.style.backgroundColor = '#3b82f6')}
          >
            <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
            {isLoading ? 'Refreshing...' : 'Refresh Data'}
          </button>
          
          <button
            onClick={handleAddNode}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: '1px solid #10b981',
              backgroundColor: '#10b981',
              color: 'white',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#059669'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#10b981'}
          >
            <Plus size={16} />
            Add Node
          </button>
          
          <button
            onClick={toggleProvenance}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: '1px solid #8b5cf6',
              backgroundColor: '#8b5cf6',
              color: 'white',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#7c3aed'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#8b5cf6'}
          >
            Toggle AI/User
          </button>
          
          <button
            onClick={handleUseDefaultData}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: '1px solid #6b7280',
              backgroundColor: '#6b7280',
              color: 'white',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#4b5563'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#6b7280'}
          >
            Use Default Data
          </button>
          
          <button
            onClick={addIntersections}
            disabled={nodes.length < 2}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: '1px solid #f59e0b',
              backgroundColor: '#f59e0b',
              color: 'white',
              cursor: nodes.length < 2 ? 'not-allowed' : 'pointer',
              opacity: nodes.length < 2 ? 0.6 : 1,
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => nodes.length >= 2 && (e.currentTarget.style.backgroundColor = '#d97706')}
            onMouseLeave={(e) => nodes.length >= 2 && (e.currentTarget.style.backgroundColor = '#f59e0b')}
          >
            Test Intersections
          </button>
          
          <button
            onClick={handleClear}
            disabled={nodes.length === 0}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: '1px solid #ef4444',
              backgroundColor: '#ef4444',
              color: 'white',
              cursor: nodes.length === 0 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              opacity: nodes.length === 0 ? 0.6 : 1,
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => nodes.length > 0 && (e.currentTarget.style.backgroundColor = '#dc2626')}
            onMouseLeave={(e) => nodes.length > 0 && (e.currentTarget.style.backgroundColor = '#ef4444')}
          >
            <Trash2 size={16} />
            Clear All
          </button>
        </div>
      </div>
      
      {/* Info Panel */}
      <div style={{
        padding: '15px 20px',
        backgroundColor: '#fef3c7',
        borderBottom: '1px solid #fbbf24',
        fontSize: '14px',
      }}>
        <strong>Info:</strong> Nodes: {nodes.length} | Refreshes: {refreshCount} | 
        <span style={{ marginLeft: '10px' }}>
          Click on nodes to interact with them. Double-click to edit text. 
          Press Enter to confirm edits and trigger keyword/similar text fetching.
        </span>
      </div>
      
      {/* ReactFlow Canvas */}
      <div style={{ flex: 1 }}>
        <ReactFlow
          nodes={nodes}
          onNodesChange={(changes) => {
            // Handle node changes (position, selection, etc.)
            setNodes((nds) => {
              const updatedNodes = [...nds];
              changes.forEach((change) => {
                if (change.type === 'position' && change.position) {
                  const node = updatedNodes.find((n) => n.id === change.id);
                  if (node) {
                    node.position = change.position;
                  }
                }
                if (change.type === 'select') {
                  const node = updatedNodes.find((n) => n.id === change.id);
                  if (node) {
                    node.selected = change.selected;
                  }
                }
              });
              return updatedNodes;
            });
          }}
          nodeTypes={nodeTypes}
          fitView
          style={{ backgroundColor: '#f9fafb' }}
        >
          {/* You can add Controls, MiniMap, Background here if needed */}
        </ReactFlow>
      </div>
      
      {/* Style for loading animation */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        .loader {
          border: 2px solid #f3f3f3;
          border-top: 2px solid #3498db;
          border-radius: 50%;
          width: 20px;
          height: 20px;
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
};

export default TestPage;