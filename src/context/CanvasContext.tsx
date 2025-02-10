// handle saving and loading the canvas! 
import { createContext, useContext, useEffect, useState, useCallback } from "react";
import axios from "axios";
import { useAppContext } from "./AppContext";
import { type ReactFlowJsonObject } from "@xyflow/react";

interface CanvasContextType {
  canvasID: string;
  setCanvasId: (canvasID: string) => void;
  canvasName: string;
  setCanvasName: (canvasName: string) => void;
  saveCanvas: () => void;
  saveNewCanvas: (newCanvasId: string, newCanvasName: string) => void;
  createCanvas: () => void;
  deleteCanvas: (canvasIDToDelete: string) => void;
  loadCanvas: (canvasID: string) => void;
  quickSaveToBrowser: (canvasData: ReactFlowJsonObject, targetCanvasID?: string) => void;
  loadCanvasFromBrowser: (canvasID: string) => ReactFlowJsonObject | null;
}

const CanvasContext = createContext<CanvasContextType | undefined>(undefined);

export const CanvasProvider = ({ children }: { children: React.ReactNode }) => {
  const { userID, backend } = useAppContext();

  // Canvas ID in the format <userID>-<canvas number>, e.g., "shm-1"
  const [canvasID, setCanvasId] = useState<string>(`new-canvas`);//special new-canvas is the default user starter canvas
  const [canvasName, setCanvasName] = useState<string>("Untitled");


  // Quick Save to Browser
  const quickSaveToBrowser = useCallback((canvasData: ReactFlowJsonObject, targetCanvasID?: string) => {
    const idToSave = targetCanvasID || canvasID;
    console.log("Quick saving to browser storage for canvasID:", idToSave);
    try {
      const { nodes, viewport } = canvasData;
      localStorage.setItem(
        `${idToSave}`,
        JSON.stringify({ nodes, viewport })
      );
      console.log(`Canvas saved to browser storage! CanvasID: ${idToSave}, CanvasName: ${canvasName}`);
    } catch (error) {
      console.error("Error saving canvas to browser:", error);
    }
  }, [canvasID, canvasName]);

  
  // Load Canvas from backend Database
  const loadCanvas = useCallback(async (canvasID: string) => {
    try {
      const response = await axios.get(`${backend}/api/get-canvas/${canvasID}`);
      if (response.status === 200 && response.data.success) {
        quickSaveToBrowser(response.data.canvas, canvasID); // update browser storage with the database canvas
        return response.data.canvas;
      }
    } catch (error) {
      console.error("Error loading canvas:", error);
    }
    return null;
  }, [backend, quickSaveToBrowser, canvasID]);



  // Load Canvas from Browser
  const loadCanvasFromBrowser = (canvasID: string) => {
    try {
      const localStorageKey = `${canvasID}`; 
      const storedCanvas = localStorage.getItem(localStorageKey);
      if (storedCanvas) {
        console.log("Found canvas in localStorage:", JSON.parse(storedCanvas));
        return JSON.parse(storedCanvas);
      } else {
        console.warn("No canvas found for:", canvasID);
      }
    } catch (error) {
      console.error("Error loading canvas from browser:", error);
    }
    return null;
  };

    // Save a New Canvas that didn't previously exist to backend database
    const saveNewCanvas = useCallback(
      async (newCanvasId: string, newCanvasName: string) => {
        if (!userID || userID === "default") {
          console.error("You have to log in before you can save your canvas.");
          return;
        }
        const canvasData = loadCanvasFromBrowser(canvasID); //grab the existing canvas data
        setCanvasId(newCanvasId);
        setCanvasName(newCanvasName);


        if (!canvasData) {
          console.error("No canvas data found in browser storage. Cannot save.");
          return;
        }

        try {
          const { nodes, viewport } = canvasData; // Extract nodes and viewport data from canvasData
          await axios.post(`${backend}/api/save-canvas`, {
            userID,
            canvasID: newCanvasId,
            canvasName: newCanvasName,
            nodes, // leave out the edges
            viewport, // Include the viewport data
          });
          console.log(`Canvas saved successfully! UserID: ${userID}, CanvasID: ${newCanvasId}, CanvasName: ${newCanvasName}`);
          quickSaveToBrowser(canvasData, newCanvasId); // Save the new canvas to browser storage under new name and id
        } catch (error) {
          console.error("Error saving canvas:", error);
        }
      },
      [userID, backend, loadCanvasFromBrowser]
    );

    // Save Canvas to backend database
    const saveCanvas = useCallback(async () => {
      if (!userID || userID === "default") {
        console.error("You have to log in before you can save your canvas.");
        return;
      }

      const canvasData = loadCanvasFromBrowser(canvasID);
      if (!canvasData) {
        console.error("No canvas data found in browser storage. Cannot save.");
        return;
      }

      try {
        const { nodes, viewport } = canvasData; // Extract nodes and viewport data from canvasData
        await axios.post(`${backend}/api/save-canvas`, {
          userID,
          canvasID,
          canvasName,
          nodes, // leave out the edges
          viewport, // Include the viewport data
        });
        console.log(`Canvas saved successfully! UserID: ${userID}, CanvasID: ${canvasID}, CanvasName: ${canvasName}`);
      } catch (error) {
        console.error("Error saving canvas:", error);
      }
    }, [userID, backend, canvasID, canvasName, loadCanvasFromBrowser]);


    const createCanvas = useCallback(async () => {
      try {
        const response = await fetch(`${backend}/api/next-canvas-id/${userID}`);
        const data = await response.json();
    
        if (!data.success) {
          console.error("Error fetching new canvas ID.");
          return;
        }
        const newCanvasId = data.nextCanvasId;
        const newCanvasName = "Untitled";
        const newCanvasData: ReactFlowJsonObject = {
          nodes: [],
          edges: [],
          viewport: { x: 0, y: 0, zoom: 1 },
        };
    
        await setCanvasId(newCanvasId);
        await setCanvasName(newCanvasName);
        quickSaveToBrowser(newCanvasData, newCanvasId);
        console.log(`New canvas created with ID: ${newCanvasId} and Name: ${newCanvasName}`);
      } catch (error) {
        console.error("Error fetching new canvas ID:", error);
      }
    }, [userID, backend, setCanvasId, setCanvasName, quickSaveToBrowser]);
    

    // Delete Canvas from backend database
    const deleteCanvas = useCallback(
      async (canvasIDToDelete: string) => {
        if (!userID || userID === "default") {
          console.error("You aren't logged in.");
          return;
        }
        if (canvasIDToDelete === "new-canvas") {
          console.error("Nothing to delete!."); //can't delete the new canvas
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
      },
      [userID, backend]
    );

  return (
    <CanvasContext.Provider value={{ canvasID, setCanvasId, canvasName, setCanvasName, saveCanvas, saveNewCanvas, createCanvas, deleteCanvas, loadCanvas, quickSaveToBrowser, loadCanvasFromBrowser }}>
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
