import React, { createContext, useContext, useState, useCallback } from 'react';
import { 
  Node, 
  Edge, 
  ReactFlowJsonObject, 
  Viewport,   
  applyNodeChanges,
  applyEdgeChanges} from '@xyflow/react';
import { TextWithKeywordsNodeData } from '../nodes/types';
import { stringToWords } from '../utils/utilityFunctions';
import {useSearchParams} from "react-router-dom";
import { useAppContext } from './AppContext';
import { useCanvasContext } from './CanvasContext';



interface NodeContextProps {
  nodes: Node<any>[]; // Keep nodes generic
  edges: Edge<any>[]; // Keep edges generic 
  currentViewport: Viewport;
  handleOnNodesChange: (changes: any) => void;
  handleOnEdgesChange: (changes: any) => void;
  setNodes: React.Dispatch<React.SetStateAction<Node<any>[]>>; // Keep setNodes generic
  setEdges: React.Dispatch<React.SetStateAction<Edge<any>[]>>; // Keep setEdges generic
  canvasToObject: () => ReactFlowJsonObject;
  mergeNodes: (nodesToMerge: { id: string; content: string; position: { x: number; y: number } }[]) => void;
  saveCurrentViewport: (viewport: Viewport) => void;
}

const NodeContext = createContext<NodeContextProps | undefined>(undefined);

export const NodeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { loginStatus } = useAppContext(); //userID, backend
  const { canvasName, canvasID, quickSaveToBrowser } = useCanvasContext();  //setCanvasName//the nodes as saved to the context and database
  const [searchParams] = useSearchParams();
  const userParam = searchParams.get('user');
  const canvasParam = searchParams.get('canvas');
  
  const [nodes, setNodes] = useState<Node<any>[]>([]); // Keep nodes state generic
  const [edges, setEdges] = useState<Edge<any>[]>([]); // Keep edges state generic
  const [currentViewport, setCurrentViewport] = useState<Viewport>({ x: 0, y: 0, zoom: 1 });

  const saveCurrentViewport = useCallback((viewport: Viewport) => {
    setCurrentViewport(viewport);
  }, []);

  const canvasToObject = useCallback((): ReactFlowJsonObject => {
    //console.log("current nodes:" , nodes)
    return {
      nodes: nodes,
      edges: edges,
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

  const handleOnNodesChange = useCallback(
    (changes: any) => {
      setNodes((nds) => applyNodeChanges(changes, nds));
      quickSaveToBrowser(canvasToObject()); // Save to browser storage
    },
    [setNodes, quickSaveToBrowser, canvasToObject, userParam, canvasParam, loginStatus, canvasID, canvasName]
  );
  
  const handleOnEdgesChange = useCallback(
    (changes: any) => {
      setEdges((eds) => applyEdgeChanges(changes, eds));
      quickSaveToBrowser(canvasToObject()); // Save to browser storage
    },
    [setEdges, quickSaveToBrowser, canvasToObject, userParam, canvasParam, loginStatus, canvasID, canvasName]
  );



  return (
    <NodeContext.Provider value={{ nodes, edges, currentViewport, setNodes, setEdges, mergeNodes, handleOnEdgesChange, handleOnNodesChange, canvasToObject, saveCurrentViewport }}>
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
