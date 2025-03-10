import { createContext, useContext, useState, useCallback } from "react";
import axios from "axios";
import { useAppContext } from "./AppContext";
import { type ReactFlowJsonObject } from "@xyflow/react";


interface CanvasContextType {
  canvasID: string;
  setCanvasId: (canvasID: string) => void;
  canvasName: string;
  setCanvasName: (canvasName: string) => void;
  saveCanvas: (canvasData: ReactFlowJsonObject, canvasIDToSave?: string, canvasNameToSave?: string) => void;
  deleteCanvas: (canvasID: string) => void;
  createNewCanvas: (userID: string) => void;
  pullCanvas: (canvasID: string) => Promise<ReactFlowJsonObject | null>;
  quickSaveToBrowser: (canvasData: ReactFlowJsonObject, targetCanvasID?: string, targetCanvasName?: string) => void;
  pullCanvasFromBrowser: (canvasID: string) => ReactFlowJsonObject | null;
  lastSaved: string;
  setLastSaved: (timestamp: string) => void;
}

const CanvasContext = createContext<CanvasContextType | undefined>(undefined);

export const CanvasProvider = ({ children }: { children: React.ReactNode }) => {
  const { userID, backend, loginStatus } = useAppContext();
  // const [searchParams] = useSearchParams();
  // const { setNodes } = useNodeContext();

  const [canvasID, setCanvasId] = useState<string>(`new-canvas`);
  const [canvasName, setCanvasName] = useState<string>("Untitled");
  const [lastSaved, setLastSaved] = useState<string>("Never");

  

  const quickSaveToBrowser = useCallback((canvasData: ReactFlowJsonObject, targetCanvasID?: string, targetCanvasName?: string) => {
    const idToSave = targetCanvasID || canvasID;
    try {
      const { nodes, edges, viewport } = canvasData;
      //console.log("saving canvas to browser:", idToSave);
      localStorage.setItem(`${idToSave}`, JSON.stringify({ nodes, edges, viewport }));
      if (targetCanvasName){
        localStorage.setItem(`${idToSave}-name`, targetCanvasName);
      }
    } catch (error) {
      console.error("Error saving canvas to browser:", error);
    }
  }, [canvasID]);

  const pullCanvas = useCallback(async (canvasID: string): Promise<ReactFlowJsonObject | null> => {
    try {
      const response = await axios.get(`${backend}/api/get-canvas/${canvasID}`);
      if (response.status === 200 && response.data.success) {
        const canvasData: ReactFlowJsonObject = response.data.canvas;
        canvasData.edges = typeof canvasData.edges === 'string' ? JSON.parse(canvasData.edges) : canvasData.edges; // Parse edges field if it's a string
        return canvasData;
      }
    } catch (error) {
      console.error("Error loading canvas:", error);
    }
    return null;
  }, [backend, canvasID]);

  const pullCanvasFromBrowser = (canvasID: string) => {
    try {
      const storedCanvas = localStorage.getItem(`${canvasID}`);
      if (storedCanvas) {
        console.log("canvas context found canvasID:", canvasID, JSON.parse(storedCanvas));
        return JSON.parse(storedCanvas);
      } else {
        console.warn("No canvas found for:", canvasID);
      }
    } catch (error) {
      console.error("Error loading canvas from browser:", error);
    }
    return null;
  };

  const saveCanvas = useCallback(async (canvasData: ReactFlowJsonObject, canvasIDToSave?: string, canvasNameToSave?: string) => {
    console.log("attempting to save this canvasID: ", canvasIDToSave, " with this data:", canvasData);
    canvasIDToSave = canvasIDToSave || canvasID;
    canvasNameToSave = canvasNameToSave || canvasName;
    if (!userID) {
      console.error("You have to log in before you can save your canvas.");
      return;
    }

    if (!canvasData) {
      console.error("No canvas data provided. Cannot save.");
      return;
    }
    //console.log(`attempting to save this canvasID: ${canvasID} with this data:`, canvasData);

    try {
      const { nodes, edges, viewport } = canvasData;
      //console.log("saving canvas to server:", canvasData.edges);
      const timestamp =(new Date().toISOString());
      await axios.post(`${backend}/api/save-canvas`, {
        userID,
        canvasID : canvasIDToSave,
        canvasName : canvasNameToSave,
        nodes,
        edges,
        viewport,
        timestamp
      });
      //console.log(`Canvas ${canvasIDToSave} (${canvasNameToSave}) saved successfully for user ${userID}.`);
      setLastSaved(timestamp);
    } catch (error) {
      console.error("Error saving canvas:", error);
    }
  }, [backend, userID, canvasID, canvasName]);

  const createNewCanvas = useCallback(async (userID: string) => {
    if (loginStatus != "logged in"){
      console.error("You have to log in before you can create a new canvas. You are currently... ", loginStatus);
      return;
    }
    const response = await fetch(`${backend}/api/next-canvas-id/${userID}`);
    const data = await response.json();
    if (!data.success) {
      console.error("Error fetching next canvas ID.");
      return;
    }
    console.log("received next canvas ID:", data.nextCanvasId, " calling save canvas on the new canvas");
    await saveCanvas({ nodes: [], edges: [], viewport: { x: 0, y: 0, zoom: 1 }}, `${data.nextCanvasId}`, "Untitled");
    console.log("redirecting to new canvas: " + data.nextCanvasId);
    window.location.href = `/?user=${userID}&canvas=${data.nextCanvasId}`;
  },[loginStatus]);

  const deleteCanvas = useCallback(async (canvasIDToDelete: string) => {
    if (!userID || userID === "default") {
      console.error("You aren't logged in.");
      return;
    }
    if (canvasIDToDelete === "new-canvas") {
      console.error("Nothing to delete!");
      return;
    }

    try {
      const response = await axios.delete(`${backend}/api/delete-canvas/${userID}/${canvasIDToDelete}`);
      if (response.status === 200 && response.data.success) {
        console.log(`Canvas ${canvasIDToDelete} deleted successfully for user ${userID}.`);
      } else {
        console.error("Error deleting canvas:", response.data.error);
      }
    } catch (error) {
      console.error("Error deleting canvas:", error);
    }
  }, [userID, backend]);

  return (
    <CanvasContext.Provider value={{ canvasID, setCanvasId, canvasName, setCanvasName, saveCanvas, deleteCanvas, createNewCanvas, pullCanvas, quickSaveToBrowser, pullCanvasFromBrowser, lastSaved, setLastSaved }}>
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
