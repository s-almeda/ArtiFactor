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
  useReactFlow} from '@xyflow/react';
import { TextWithKeywordsNodeData, AppNode } from '../nodes/types';
import { stringToWords } from '../utils/utilityFunctions';
import {useSearchParams} from "react-router-dom";
import { useAppContext } from './AppContext';
import { useCanvasContext } from './CanvasContext';
import { useDnD } from './DnDContext';
import { debounce } from 'lodash';


interface NodeContextProps {
  nodes: Node<any>[]; // Keep nodes generic
  edges: Edge<any>[]; // Keep edges generic 
  currentViewport: Viewport;
  handleOnNodesChange: (changes: any) => void;
  handleOnEdgesChange: (changes: any) => void;
  onNodesDelete: (deleted: any[]) => void;
  deleteNodeById: (nodeId: string) => void;
  onNodeDrag: (event: React.MouseEvent, node: Node) => void;
  onNodeDragStop: (event: React.MouseEvent, node: Node) => void;
  setNodes: React.Dispatch<React.SetStateAction<Node<any>[]>>; // Keep setNodes generic
  setEdges: React.Dispatch<React.SetStateAction<Edge<any>[]>>; // Keep setEdges generic
  canvasToObject: () => ReactFlowJsonObject;
  mergeNodes: (nodesToMerge: { id: string; content: string; position: { x: number; y: number } }[]) => void;
  saveCurrentViewport: (viewport: Viewport) => void;
}

const NodeContext = createContext<NodeContextProps | undefined>(undefined);

export const NodeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
 
  const { loginStatus } = useAppContext(); //userID, backend
  const { setDraggableType, setDraggableData } = useDnD();
  const { canvasName, canvasID, quickSaveToBrowser, saveCanvas } = useCanvasContext();  //setCanvasName//the nodes as saved to the context and database
  const [searchParams] = useSearchParams();

  const userParam = searchParams.get('user');
  const canvasParam = searchParams.get('canvas');
  const {getIntersectingNodes} = useReactFlow();
  
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


  const onNodeDrag = useCallback(
    (_: React.MouseEvent, draggedNode: Node) => {
      setDraggableType(draggedNode.type as string);
      setDraggableData(draggedNode.data);
      if (draggedNode.type === "text") {
        updateIntersections(draggedNode, nodes);
      }
    },
      [setNodes, getIntersectingNodes]
    );
    
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



    const updateIntersections = (draggedNode: Node, currentNodes: AppNode[]) => {
      if (!draggedNode.width){
        return currentNodes;
      }
      const intersections = getIntersectingNodes(draggedNode).map((n) => n.id);
      return currentNodes.map((node: AppNode) => {
        if (node.id === draggedNode.id) {
          const updatedIntersections = [
            {
              id: node.id,
              position: node.position,
              content: node.data.content,
            },
            ...intersections.map((id) => {
              const intersectingNode = currentNodes.find((n) => n.id === id);
              if (intersectingNode && intersectingNode.type === "text") {
                //console.log(`${node.data.content} is overlapping with: ${intersectingNode.data.content}`);
                return {
                  id: intersectingNode.id,
                  position: intersectingNode.position,
                  content: intersectingNode.data.content,
                };
              }
              return null;
            }).filter(Boolean),
          ];
          //console.log("updated intersections for node", node.data.content, ": ", updatedIntersections);
          return {
            ...node,
            data: {
              ...node.data,
              intersections: updatedIntersections,
            },
          };
        }
        return node;
      });
    };

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

    autoSaveCanvas(canvasToObject());
  }, []);

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

  // Use autoSaveCanvas instead of saveCanvas
  const onNodeDragStop = useCallback(
    (_: React.MouseEvent, draggedNode: Node) => {
      setNodes((currentNodes: AppNode[]) => updateIntersections(draggedNode, currentNodes));
      //autoSaveCanvas(canvasToObject());
    },
    [setNodes, updateIntersections, autoSaveCanvas, canvasToObject]
  );

  useEffect(() => {
    autoSaveCanvas(canvasToObject());
  }, [nodes, edges, autoSaveCanvas, canvasToObject]);


  return (
    <NodeContext.Provider value={{ nodes, edges, currentViewport, setNodes, setEdges, mergeNodes, onNodesDelete, handleOnEdgesChange, handleOnNodesChange, canvasToObject, saveCurrentViewport, onNodeDragStop, onNodeDrag, deleteNodeById }}>
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
