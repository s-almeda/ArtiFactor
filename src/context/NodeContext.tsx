import React, { useEffect, createContext, useContext, useState, useCallback } from 'react';
import { 
  Node, 
  Edge, 
  ReactFlowJsonObject, 
  Viewport,   
  applyNodeChanges,
  getIncomers,
  getOutgoers,
  getConnectedEdges,
  applyEdgeChanges,
  addEdge
  } from '@xyflow/react';
import { TextWithKeywordsNodeData, AppNode } from '../nodes/types';
import { stringToWords } from '../utils/utilityFunctions';
import {useSearchParams} from "react-router-dom";
import { useAppContext } from './AppContext';
import { useCanvasContext } from './CanvasContext';

import { debounce } from 'lodash';
import {v4 as uuidv4} from 'uuid';


interface NodeContextProps {
  nodes: Node<any>[]; // Keep nodes generic
  edges: Edge<any>[]; // Keep edges generic 
  currentViewport: Viewport;
  handleOnNodesChange: (changes: any) => void;
  handleOnEdgesChange: (changes: any) => void;
  onNodesDelete: (deleted: any[]) => void;
  deleteNodeById: (nodeId: string) => void;
  drawEdge: (parentNodeId: string, newNodeId: string, updatedNodes: AppNode[]) => void;
  setNodes: React.Dispatch<React.SetStateAction<Node<any>[]>>; // Keep setNodes generic
  setEdges: React.Dispatch<React.SetStateAction<Edge<any>[]>>; // Keep setEdges generic
  canvasToObject: () => ReactFlowJsonObject;
  mergeNodes: (nodesToMerge: { id: string; content: string; position: { x: number; y: number } }[]) => void;
  saveCurrentViewport: (viewport: Viewport) => void;
}

const NodeContext = createContext<NodeContextProps | undefined>(undefined);

export const NodeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
 
  const { loginStatus } = useAppContext(); //userID, backend
  const { canvasName, canvasID, quickSaveToBrowser, saveCanvas } = useCanvasContext();  //setCanvasName//the nodes as saved to the context and database
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
  }, [nodes, edges, currentViewport]);



    
  const autoSaveCanvas = useCallback(
    debounce((canvasObject: { nodes: Node[]; edges: Edge[]; viewport: Viewport; }) => {
      if (loginStatus === "logged in") {
        saveCanvas(canvasObject);
      } else {
        quickSaveToBrowser(canvasObject);
      }
    }, 1000), // 1-second debounce
    [saveCanvas, quickSaveToBrowser, loginStatus]
  );


  const deleteNodeById = useCallback(
  (nodeId: string) => {
    setEdges((currentEdges) => currentEdges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
    setNodes((currentNodes) => currentNodes.filter((node) => node.id !== nodeId));
    autoSaveCanvas(canvasToObject());
  },[]);


  
  

  const mergeNodes = useCallback((nodesToMerge: { id: string; content: string; position: { x: number; y: number } }[]) => {
    if (nodesToMerge.length === 0) return;

    // Combine the content of the nodes
    const combinedContent = nodesToMerge.map(node => node.content).join(' ');

    // Calculate the position below the two parent nodes
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
    const newPosition = { x: averagePosition.x, y: averagePosition.y +210 }; // Position below the parents

    const newNode: Node<TextWithKeywordsNodeData> = {
      id: `text-${uuidv4()}`,
      type: 'text',
      position: newPosition,
      data: { content: combinedContent, words: stringToWords(combinedContent), provenance: "user", intersections: [] },
    };
    // Create new edges for each node in nodesToMerge with the new node as the target
    nodesToMerge.forEach((node) => {
      const newEdge: Edge = {
        id: `edge-${node.id}-${newNode.id}`,
        source: node.id,
        target: newNode.id,
        type: 'default',
      };
      setEdges((prevEdges) => addEdge(newEdge, prevEdges));
    });
    
    // Add the new node to the nodes state
    setNodes((prevNodes) => prevNodes.concat(newNode));

    autoSaveCanvas(canvasToObject());
  }, []);

  const drawEdge = (parentNodeId: string, newNodeId: string, updatedNodes: AppNode[]) => {
    //console.log("Parent Node ID:", parentNodeId, "new node id: ", newNodeId);

    const parentNode = updatedNodes.find((node) => node.id === parentNodeId);
    const newNode = updatedNodes.find((node) => node.id === newNodeId);

    if (parentNode && newNode) {
      //console.log("Parent Node Content:", parentNode.data.content);
      //console.log("New Node Content:", newNode.data.content);

      const newEdge: Edge = {
        id: `edge-${parentNodeId}-${newNodeId}`,
        source: parentNodeId,
        target: newNodeId,
        type: 'default',
      };

      setEdges((eds) => addEdge(newEdge, eds));
    }
  };




  const handleOnNodesChange = useCallback(
    (changes: any) => {
      setNodes((nds) => applyNodeChanges(changes, nds));
    },
    [setNodes, quickSaveToBrowser, canvasToObject, userParam, canvasParam, loginStatus, canvasID, canvasName]
  );
  
  const handleOnEdgesChange = useCallback(
    (changes: any) => { //do not allow edge delete
      const filteredChanges = changes.filter((change: any) => change.type !== 'remove'); // Filter out removals
      setEdges((eds) => applyEdgeChanges(filteredChanges, eds));
    },
    [setEdges, quickSaveToBrowser, canvasToObject, userParam, canvasParam, loginStatus, canvasID, canvasName]
  );

  const onNodesDelete = useCallback(
    (deleted: any[]) => {
      setEdges(
        deleted.reduce((acc, node) => {
          const incomers = getIncomers(node, nodes, edges);
          const outgoers = getOutgoers(node, nodes, edges);
          const connectedEdges = getConnectedEdges([node], edges);

          const remainingEdges = acc.filter(
            (edge: Edge<any>) => !connectedEdges.includes(edge),
          );

          const createdEdges = incomers.flatMap(({ id: source }) =>
            outgoers.map(({ id: target }) => ({
              id: `${source}->${target}`,
              source,
              target,
            })),
          );

          return [...remainingEdges, ...createdEdges];
        }, edges),
      );

      setNodes((prevNodes) =>
        prevNodes.filter((node) => !deleted.some((delNode) => delNode.id === node.id))
      );

    },
    [nodes, edges, canvasID, canvasName, saveCanvas, canvasToObject],
  );



  useEffect(() => {
    autoSaveCanvas(canvasToObject());
  }, [nodes, edges, autoSaveCanvas, canvasToObject]);


  return (
    <NodeContext.Provider value={{ nodes, edges, currentViewport, setNodes, setEdges, drawEdge, mergeNodes, onNodesDelete, handleOnEdgesChange, handleOnNodesChange, canvasToObject, saveCurrentViewport, deleteNodeById }}>
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
