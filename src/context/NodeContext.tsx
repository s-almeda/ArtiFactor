import React, { createContext, useContext, useState, useCallback } from 'react';
import { Node } from '@xyflow/react';

interface NodeContextProps {
  nodes: Node[];
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
  mergeNodes: (node1: Node, node2: Node) => void;
}
``
const NodeContext = createContext<NodeContextProps | undefined>(undefined);

export const NodeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [nodes, setNodes] = useState<Node[]>([]);

  const mergeNodes = useCallback((node1: Node, node2: Node) => {
    // Combine the content of the nodes
    const combinedContent = `${node1.data.content} ${node2.data.content}`;
    const newNode: Node = {
      ...node1,
      data: { ...node1.data, content: combinedContent },
    };

    // Update the nodes state
    setNodes((prevNodes) =>
      prevNodes.filter((node) => node.id !== node1.id && node.id !== node2.id).concat(newNode)
    );
  }, []);

  return (
    <NodeContext.Provider value={{ nodes, setNodes, mergeNodes }}>
      {children}
    </NodeContext.Provider>
  );
};

export const useNodeContext = () => {
  const context = useContext(NodeContext);
  if (!context) {
    throw new Error('useNodeContext must be used within a NodeProvider');
  }
  return context;
};