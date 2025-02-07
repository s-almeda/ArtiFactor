import { createContext, useContext, useState, useCallback } from "react";
import axios from "axios";
import type { ReactNode } from "react";
import type { AppNode } from "../nodes/types";

interface CanvasContextType {
  savedNodes: AppNode[]; //nodes that have been saved to the context, and thus, to the database.
  setNodes: (nodes: AppNode[]) => void;
  saveCanvas: () => void;
  loadCanvas: (canvasID: string) => void;
}

// Create Context
const CanvasContext = createContext<CanvasContextType | undefined>(undefined);

// Provider Component
// the backend is set in the App.tsx file! 
export const CanvasProvider: React.FC<{ children: ReactNode; userID: string; backend: string }> = ({ children, userID, backend }) => {
    const [savedNodes, setNodes] = useState<AppNode[]>([]);

  // Load a canvas by ID
  const loadCanvas = useCallback(async (canvasID: string) => {
    try {
      const response = await axios.get(`${backend}/api/load-canvas/${canvasID}`);
      if (response.data.success) {
        setNodes(response.data.canvas.nodes);
        console.log(`Loaded canvas: ${canvasID}`);
      } else {
        console.error("Failed to load canvas:", response.data.error);
      }
    } catch (error) {
      console.error("Error loading canvas:", error);
    }
  }, [backend]);

  // Save the current canvas state
  const saveCanvas = useCallback(async () => {
    try {
      const canvasData = { userID, savedNodes };
      await axios.post(`${backend}/api/save-canvas`, canvasData, {
        headers: { "Content-Type": "application/json" },
      });
      console.log("Canvas saved successfully!");
    } catch (error) {
      console.error("Error saving canvas:", error);
    }
  }, [savedNodes, backend, userID]);

  return (
    <CanvasContext.Provider value={{ savedNodes, setNodes, saveCanvas, loadCanvas }}>
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
