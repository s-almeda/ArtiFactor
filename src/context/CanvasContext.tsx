import { createContext, useContext, useState, useCallback } from "react";
import axios from "axios";
import { useAppContext } from "./AppContext";
import { useReactFlow, type ReactFlowJsonObject } from "@xyflow/react";

interface CanvasContextType {
  canvasId: string;
  setCanvasId: (canvasId: string) => void;
  canvasName: string;
  setCanvasName: (canvasName: string) => void;
  saveCanvas: (canvasData: ReactFlowJsonObject) => void;
  loadCanvas: (canvasID: string) => void;
}

const CanvasContext = createContext<CanvasContextType | undefined>(undefined);

export const CanvasProvider = ({ children }: { children: React.ReactNode }) => {
  const { userID, backend } = useAppContext();

  // Canvas ID in the format <userID>-<canvas number>, e.g., "shm-1"
  const [canvasId, setCanvasId] = useState<string>(`new-canvas`);//special new-canvas is the default user starter canvas
  const [canvasName, setCanvasName] = useState<string>("Untitled Canvas");

  //  Save Canvas
  const saveCanvas = useCallback(
    async (canvasData: ReactFlowJsonObject) => {
      if (!userID) {
        console.error("No user ID found. Cannot save.");
        return;
      }

      try {
        const { nodes, viewport } = canvasData; // Extract nodes, edges, and viewport data from canvasData
        await axios.post(`${backend}/api/save-canvas`, {
          userID,
          canvasId,
          canvasName,
          nodes, //leave out the edges
          viewport, // Include the viewport data
        });
        console.log("Canvas saved successfully!");
      } catch (error) {
        console.error("Error saving canvas:", error);
      }
    },
    [userID, backend, canvasId, canvasName]
  );

  //load Canvas

  const loadCanvas = useCallback(async (canvasID: string) => {
    try {
      const response = await axios.get(`${backend}/api/get-canvas/${canvasID}`);
      if (response.status === 200 && response.data.success) {
        return response.data.canvas;
      }
    } catch (error) {
      console.error("Error loading canvas:", error);
    }
    return null;
  }, [backend]);



  return (
    <CanvasContext.Provider value={{ canvasId, setCanvasId, canvasName, setCanvasName, saveCanvas, loadCanvas }}>
      {children}
    </CanvasContext.Provider>
  );
};


export const useCanvasContext = () => {
  const context = useContext(CanvasContext);
  if (!context) {
    throw new Error("useCanvasContext must be used within a CanvasProvider");
  }
  return context;
};
