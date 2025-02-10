// handle saving and loading the canvas! 
import { createContext, useContext, useState, useCallback } from "react";
import axios from "axios";
import { useAppContext } from "./AppContext";
import { type ReactFlowJsonObject } from "@xyflow/react";

interface CanvasContextType {
  canvasId: string;
  setCanvasId: (canvasId: string) => void;
  canvasName: string;
  setCanvasName: (canvasName: string) => void;
  saveCanvas: () => void;
  loadCanvas: (canvasID: string) => void;
  quickSaveToBrowser: (canvasData: ReactFlowJsonObject) => void;
  loadCanvasFromBrowser: (canvasId: string) => ReactFlowJsonObject | null;
}

const CanvasContext = createContext<CanvasContextType | undefined>(undefined);

export const CanvasProvider = ({ children }: { children: React.ReactNode }) => {
  const { userID, backend } = useAppContext();

  // Canvas ID in the format <userID>-<canvas number>, e.g., "shm-1"
  const [canvasId, setCanvasId] = useState<string>(`new-canvas`);//special new-canvas is the default user starter canvas
  const [canvasName, setCanvasName] = useState<string>("Untitled Canvas");


  // Load Canvas from backend Database
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

  // Quick Save to Browser
  const quickSaveToBrowser = (canvasData: ReactFlowJsonObject) => {
    try {
      const { nodes, viewport } = canvasData;
      localStorage.setItem(
        `new-canvas`,
        JSON.stringify({ nodes, viewport })
      );
      //console.log("Canvas saved to browser storage!");
    } catch (error) {
      console.error("Error saving canvas to browser:", error);
    }
  };

  // Load Canvas from Browser
  const loadCanvasFromBrowser = (canvasId: string) => {
    try {
      const localStorageKey = `${canvasId}`; // Ensure you're using the same key format
      const storedCanvas = localStorage.getItem(localStorageKey);
      if (storedCanvas) {
        console.log("Found canvas in localStorage:", JSON.parse(storedCanvas));
        return JSON.parse(storedCanvas);
      } else {
        console.warn("No canvas found for:", canvasId);
      }
    } catch (error) {
      console.error("Error loading canvas from browser:", error);
    }
    return null;
  };

    //  Save Canvas to backend database
    const saveCanvas = useCallback(
      async () => {
        if (!userID || userID === "default") {
          console.error("You have to log in before you can save your canvas.");
          return;
        }
        
        const canvasData = loadCanvasFromBrowser(canvasId);
        if (!canvasData) {
          console.error("No canvas data found in browser storage. Cannot save.");
          return;
        }
  
        try {
          const { nodes, viewport } = canvasData; // Extract nodes and viewport data from canvasData
          await axios.post(`${backend}/api/save-canvas`, {
            userID,
            canvasId,
            canvasName,
            nodes, // leave out the edges
            viewport, // Include the viewport data
          });
            console.log(`Canvas saved successfully! UserID: ${userID}, CanvasID: ${canvasId}, CanvasName: ${canvasName}`);
        } catch (error) {
          console.error("Error saving canvas:", error);
        }
      },
      [userID, backend, canvasId, canvasName, loadCanvasFromBrowser]
    );

  return (
    <CanvasContext.Provider value={{ canvasId, setCanvasId, canvasName, setCanvasName, saveCanvas, loadCanvas, quickSaveToBrowser, loadCanvasFromBrowser }}>
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
