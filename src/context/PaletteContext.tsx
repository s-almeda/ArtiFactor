import type React from "react";
import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";
//import { useAppContext } from "./AppContext"; //add this back when we start saving palette data to the database

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
  // const { userID, backend } = useAppContext(); //add this back when we start saving palette data to the database
  const [clippedNodes, setClippedNodes] = useState<NodeData[]>([]);
  const [activeTab, setActiveTab] = useState<"images" | "text">("images");

  const addClippedNode = (node: NodeData) => {
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
  };

  const getNextPaletteIndex = () => {
    return clippedNodes.length;
  };

  return (
    <PaletteContext.Provider
      value={{
        clippedNodes,
        setClippedNodes,
        addClippedNode,
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
