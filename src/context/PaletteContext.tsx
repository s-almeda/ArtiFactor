import type React from "react";
import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";

import { useAppContext } from "./AppContext"; //add this back when we start saving palette data to the database
import { useEffect } from "react";

// Define the type of a node
export interface NodeData {
  id: number;
  type: "image" | "text";
  content: string; // is either text, or an imageUrl
  provenance?: "history" | "user" | "ai"; // history = straight from the database, user = user added/edited, ai = generated
  parentNodeId?: string;
  prompt: string; // for storing the prompt, or the title/artist alt text for the artwork
  similarArtworks?: any[]; // for storing similar artworks
  similarTexts?: any[]; // for storing similar texts
  words?: any[]; // for storing words
}

// Define the context type
interface PaletteContextType {
  clippedNodes: NodeData[];
  setClippedNodes: (nodes: NodeData[]) => void;
  addClippedNode: (node: NodeData) => void;
  removeNode: (id: number) => void;
  loadPalette: (userID: string) => Promise<void>;
  activeTab: "images" | "text";
  setActiveTab: (tab: "images" | "text") => void;
  getNextPaletteIndex: () => number;
}

// Create the context with a default value
const PaletteContext = createContext<PaletteContextType | undefined>(undefined);

// Create a provider component
export const PaletteProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const { userID, backend, loginStatus } = useAppContext(); // add this back when we start saving palette data to the database
  const [clippedNodes, setClippedNodes] = useState<NodeData[]>([]);
  const [activeTab, setActiveTab] = useState<"images" | "text">("images");

  const addClippedNode = async (node: NodeData) => {
    // if the node with the same content is already in the list, don't add it again
    if (clippedNodes.some((n) => n.content === node.content)) {
      return;
    }
    console.log("Adding node to palette:", node);
    setClippedNodes((prevNodes) => [...prevNodes, node]);

    if (node.type == 'image') {
      setActiveTab("images");
    } else {
      setActiveTab("text");
    }

    if (loginStatus === "logged in") { // if logged in, also add the clipping to database
      try {
        const response = await fetch(`${backend}/api/add-clipping`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ userID, clipping: node }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Failed to add clipping");
        }
        console.log("Clipping added to database:", data);
      } catch (error) {
        console.error("Error adding clipping to database:", error);
      }
    }
    else if (loginStatus === "logged out") {
      savePaletteToBrowser();
    }
  };

  const removeNode = (id: number) => {
    setClippedNodes((prevNodes) => prevNodes.filter((node) => node.id !== id));
    if (loginStatus === "logged in") { // if logged in, also remove the clipping from database
      try {
      fetch(`${backend}/api/remove-clipping`, {
        method: "POST",
        headers: {
        "Content-Type": "application/json",
        },
        body: JSON.stringify({ userID, clipping: { id } }),
      });
      } catch (error) {
      console.error("Error removing clipping from database:", error);
      }
    }
    else if (loginStatus === "logged out") {
      savePaletteToBrowser();
    }
  };

  const loadPalette = async (userID: string) => {
    try {
      const response = await fetch(`${backend}/api/get-clippings/${userID}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to load clippings");
      }
      if (!data.clippings) {
        console.log("No clippings found in database");
        return;
      }
      setClippedNodes(data.clippings);
      console.log("Clippings loaded from database:", data.clippings);
    } catch (error) {
      console.error("Error loading clippings from database:", error);
    }
  };

  const loadPaletteFromBrowser = () => {
    try {
      const storedPalette = localStorage.getItem("palette");
      if (storedPalette) {
        console.log("Found palette in browser storage:", JSON.parse(storedPalette));
        setClippedNodes(JSON.parse(storedPalette));
      } else {
        console.warn("No palette found in browser storage.");
      }
    } catch (error) {
      console.error("Error loading palette from browser:", error);
    }
  }

  const savePaletteToBrowser = () => {
    try {
      localStorage.setItem("palette", JSON.stringify(clippedNodes));
      console.log("Palette saved to browser storage:", clippedNodes);
    } catch (error) {
      console.error("Error saving palette to browser:", error);
    }
  }

  const getNextPaletteIndex = () => {
    return clippedNodes.length;
  };

  useEffect(() => {
    if (loginStatus === "logged in" && userID) {
      loadPalette(userID);
    }
    else if (loginStatus === "logged out") {
      loadPaletteFromBrowser();
    }
  }, [userID, loginStatus]);

  return (
    <PaletteContext.Provider
      value={{
        clippedNodes,
        setClippedNodes,
        addClippedNode,
        removeNode,
        loadPalette,
        activeTab,
        setActiveTab,
        getNextPaletteIndex,
      }}
    >
      {children}
    </PaletteContext.Provider>
  );
};

// Custom hook to use the PaletteContext
export const usePaletteContext = () => {
  const context = useContext(PaletteContext);
  if (!context) {
    throw new Error("usePaletteContext must be used within a PaletteProvider");
  }
  return context;
};
