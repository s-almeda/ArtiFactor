import { createContext, useContext, useState, useCallback, useEffect } from "react";
import axios from "axios";
import { useAppContext } from "./AppContext";
import { type ReactFlowJsonObject } from "@xyflow/react";
import { useSearchParams } from "react-router-dom";

interface CanvasContextType {
  canvasID: string;
  setCanvasId: (canvasID: string) => void;
  canvasName: string;
  setCanvasName: (canvasName: string) => void;
  saveCanvas: (canvasData: ReactFlowJsonObject) => void;
  deleteCanvas: (canvasIDToDelete: string) => void;
  pullCanvas: (canvasID: string) => Promise<ReactFlowJsonObject | null>;
  quickSaveToBrowser: (canvasData: ReactFlowJsonObject, targetCanvasID?: string) => void;
  pullCanvasFromBrowser: (canvasID: string) => ReactFlowJsonObject | null;
  lastSaved: string;
}

const CanvasContext = createContext<CanvasContextType | undefined>(undefined);

export const CanvasProvider = ({ children }: { children: React.ReactNode }) => {
  const { userID, backend } = useAppContext();
  const [searchParams] = useSearchParams();

  const [canvasID, setCanvasId] = useState<string>(`new-canvas`);
  const [canvasName, setCanvasName] = useState<string>("Untitled");
  const [lastSaved, setLastSaved] = useState<string>("Never");

  useEffect(() => {
    const userParam = searchParams.get('user');
    const canvasParam = searchParams.get('canvas');

    console.log('User from url:', userParam);
    console.log('Canvas from url:', canvasParam);

    if (userParam && canvasParam) {
      setCanvasId(`${userParam}-${canvasParam}`);
      console.log("set canvas id to: ", `${userParam}-${canvasParam}`);
    } else {
      setCanvasId('');
    }
  }, [searchParams]);

  const quickSaveToBrowser = useCallback((canvasData: ReactFlowJsonObject, targetCanvasID?: string) => {
    const idToSave = targetCanvasID || canvasID;
    try {
      const { nodes, viewport } = canvasData;
      localStorage.setItem(`${idToSave}`, JSON.stringify({ nodes, viewport }));
    } catch (error) {
      console.error("Error saving canvas to browser:", error);
    }
  }, [canvasID]);

  const pullCanvas = useCallback(async (canvasID: string): Promise<ReactFlowJsonObject | null> => {
    try {
      const response = await axios.get(`${backend}/api/get-canvas/${canvasID}`);
      if (response.status === 200 && response.data.success) {
        const canvasData: ReactFlowJsonObject = response.data.canvas;
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
        console.log("canvas context found canvasID:", canvasID);
        return JSON.parse(storedCanvas);
      } else {
        console.warn("No canvas found for:", canvasID);
      }
    } catch (error) {
      console.error("Error loading canvas from browser:", error);
    }
    return null;
  };

  const saveCanvas = useCallback(async (canvasData: ReactFlowJsonObject) => {
    if (!userID || userID === "default") {
      console.error("You have to log in before you can save your canvas.");
      return;
    }

    if (!canvasData) {
      console.error("No canvas data provided. Cannot save.");
      return;
    }
    console.log(`attempting to save this canvasID: ${canvasID} with this data:`, canvasData);

    try {
      const { nodes, viewport } = canvasData;
      const timestamp =(new Date().toISOString());
      await axios.post(`${backend}/api/save-canvas`, {
        userID,
        canvasID,
        canvasName,
        nodes,
        viewport,
        timestamp
      });
      console.log(`Canvas ${canvasID} saved successfully for user ${userID}.`);
      setLastSaved(timestamp);
    } catch (error) {
      console.error("Error saving canvas:", error);
    }
  }, [backend, userID, canvasID, canvasName]);

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
    <CanvasContext.Provider value={{ canvasID, setCanvasId, canvasName, setCanvasName, saveCanvas, deleteCanvas, pullCanvas, quickSaveToBrowser, pullCanvasFromBrowser, lastSaved }}>
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
