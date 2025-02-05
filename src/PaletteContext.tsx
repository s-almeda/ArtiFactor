import type React from "react";
import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";

// Define the type of a node
interface NodeData {
  id: string;
  type: "image" | "text";
  content: string;
}

// Define the context type
interface PaletteContextType {
  nodes: NodeData[];
  addClippedImage: (content: string) => void;
  clippedImages?: string[];
}

// Create the context with a default value
const PaletteContext = createContext<PaletteContextType | undefined>(undefined);

// Create a provider component
export const NodeProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [clippedImages, setClippedImages] = useState<string[]>([]);
  const [nodes, _] = useState<NodeData[]>([]);

  const addClippedImage = (imageUrl: string) => {
    // if the image is already in the list, don't add it again
    if (clippedImages.includes(imageUrl)) {
      return;
    }
    setClippedImages((prevImages) => [...prevImages, imageUrl]);
  };

  return (
    <PaletteContext.Provider value={{ nodes, clippedImages, addClippedImage }}>
      {children}
    </PaletteContext.Provider>
  );
};

// Custom hook to use the PaletteContext
export const usePaletteContext = () => {
  const context = useContext(PaletteContext);
  if (!context) {
    throw new Error("usePaletteContext must be used within a NodeProvider");
  }
  return context;
};
