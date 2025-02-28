import React, { createContext, useContext, useState, useCallback } from 'react';
import { Node,  ReactFlowJsonObject, Viewport} from '@xyflow/react';
import { TextWithKeywordsNodeData } from '../nodes/types';
import { stringToWords } from '../utils/utilityFunctions';



interface NodeContextProps {
  nodes: Node<any>[]; // Keep nodes generic
  currentViewport: Viewport;
  setNodes: React.Dispatch<React.SetStateAction<Node<any>[]>>; // Keep setNodes generic
  nodesToObject: () => ReactFlowJsonObject;
  mergeNodes: (nodesToMerge: { id: string; content: string; position: { x: number; y: number } }[]) => void;
  saveCurrentViewport: (viewport: Viewport) => void;
}

const NodeContext = createContext<NodeContextProps | undefined>(undefined);

export const NodeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [nodes, setNodes] = useState<Node<any>[]>([]); // Keep nodes state generic
  const [currentViewport, setCurrentViewport] = useState<Viewport>({ x: 0, y: 0, zoom: 1 });

  const saveCurrentViewport = useCallback((viewport: Viewport) => {
    setCurrentViewport(viewport);
  }, []);

  const nodesToObject = useCallback((): ReactFlowJsonObject => {
    return {
      nodes: nodes,
      edges: [],
      viewport: currentViewport,
    };
  }, [nodes, currentViewport]);

  const mergeNodes = useCallback((nodesToMerge: { id: string; content: string; position: { x: number; y: number } }[]) => {
    if (nodesToMerge.length === 0) return;

    // Combine the content of the nodes
    const combinedContent = nodesToMerge.map(node => node.content).join(' ');

    // Calculate the average position
    const averagePosition = nodesToMerge.reduce(
      (acc, node) => {
        acc.x += node.position.x;
        acc.y += node.position.y;
        return acc;
      },
      { x: 0, y: 0 }
    );
    averagePosition.x /= nodesToMerge.length;
    averagePosition.y /= nodesToMerge.length;

    const newNode: Node<TextWithKeywordsNodeData> = {
      id: `text-${Date.now()}`,
      type: 'text',
      position: averagePosition,
      data: { content: combinedContent, words: stringToWords(combinedContent), provenance: "user", intersections: [] },
    };

    // Update the nodes state
    setNodes((prevNodes) =>
      prevNodes.filter((node) => !nodesToMerge.some(mergeNode => mergeNode.id === node.id)).concat(newNode)
    );
  }, []);

  return (
    <NodeContext.Provider value={{ nodes, currentViewport, setNodes, mergeNodes, nodesToObject, saveCurrentViewport }}>
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
