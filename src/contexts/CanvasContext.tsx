import { createContext, useContext, useState, useCallback } from "react";
import axios from "axios";
import type { ReactNode } from "react";
import type { AppNode } from "../nodes/types";

interface CanvasContextType {
    savedNodes: AppNode[]; //nodes that have been saved to the context, that this context will then also upload to the database
    saveNodes: (nodes: AppNode[]) => void;

    canvasId: string; // unique id for this canvas. will take the form <userID>-<canvas number>, eg shm-1
    setCanvasId: (canvasId: string) => void;
    canvasName: string; // cute user name for the canvas. eg "My Cool Canvas"
    setCanvasName: (canvasId: string) => void; 

    saveCanvas: (newCanvasName?: string) => void;
    loadCanvas: (canvasID: string) => void; //takes a set of nodes from the database and loads them into the React Flow canvas
}

// This context is used to save and load lists of nodes! so users can save and reload canvases! 
// Create Context
const CanvasContext = createContext<CanvasContextType | undefined>(undefined);

// Provider Component
// the backend and the userID are set at the App level (in the App.tsx file!) and passed here!
export const CanvasProvider: React.FC<{ children: ReactNode; userID: string; backend: string }> = ({ children, userID, backend }) => {
    const [savedNodes, setNodes] = useState<AppNode[]>([]);
    const [canvasId, setCanvasId] = useState<string>("none"); //none by default - will get assigned one when a canvas is loaded in or saved for the first time.
    const [canvasName, setCanvasName] = useState<string>("Untitled Canvas");

    const saveNodes = useCallback((nodes: AppNode[]) => { // this function is called every frame by Flow to update the context, so it knows the current state of nodes in the Flow canvas
        const nodesAreDifferent = JSON.stringify(nodes) !== JSON.stringify(savedNodes);
        if (nodesAreDifferent) {
            const diffNodes = nodes.filter((node, index) => JSON.stringify(node) !== JSON.stringify(savedNodes[index]));
            console.log(`Saving ${nodes.length} nodes to context. Different nodes:`, diffNodes);
        }
        setNodes(nodes); //set the nodes in the context to match the nodes just passed to the function
    }, [savedNodes]);

  // Load a canvas by ID
  const loadCanvas = useCallback(async (canvasId: string) => {
    try {
      const response = await axios.get(`${backend}/api/load-canvas/${canvasId}`);
      if (response.data.success) {
        setNodes(response.data.canvas.nodes);
        setCanvasId(canvasId); //set the canvas ID to the one just loaded
        console.log(`Loaded ${savedNodes.length} nodes from canvas: ${canvasId}`);
        setCanvasName(response.data.canvas.name || "Untitled Canvas"); //  Set name if exists, otherwise use default name
      } else {
        console.error("Failed to load canvas:", response.data.error);
      }
    } catch (error) {
      console.error("Error loading canvas:", error);
    }
  }, [backend]);

// Save the current canvas state
const saveCanvas = useCallback(async (newCanvasName?: string) => {
    try {
        if (userID === "") {
            userID = "default";
        }
        const finalCanvasName = newCanvasName?.trim() === "" ? "Untitled Canvas" : (newCanvasName || canvasName).replace(/[^a-zA-Z0-9 _-]/g, ""); // Ensure the name is SQL safe
        const canvasData = {
            userID,
            canvasId,
            canvasName: finalCanvasName,
            nodes: [...savedNodes], // Ensure it's explicitly an array
        };

        console.log("Attempting to save canvas:", canvasData);

        const response = await axios.post(`${backend}/api/save-canvas`, canvasData, {
            headers: { "Content-Type": "application/json" },
        });

        console.log("Canvas saved successfully!", response.data);
    } catch (error) {
        console.error("Error saving canvas:", error);
    }
}, [userID, canvasId, canvasName, savedNodes, backend]);

  return (
    <CanvasContext.Provider value={{ savedNodes, saveNodes, saveCanvas, loadCanvas, canvasId, setCanvasId, canvasName, setCanvasName }}>
      {children}
    </CanvasContext.Provider>
  );
};

// Custom Hook to use the context
export const useCanvasContext = () => {
  const context = useContext(CanvasContext);
  if (!context) {
    throw new Error("useCanvasContext must be used within a CanvasProvider");
  }
  return context;
};
