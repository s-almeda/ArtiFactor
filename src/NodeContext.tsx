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
interface NodeContextType {
  nodes: NodeData[];
  addGeneratedImage: (content: string) => void;
  generatedImages?: string[];
}

// Create the context with a default value
const NodeContext = createContext<NodeContextType | undefined>(undefined);

// Create a provider component
export const NodeProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [generatedImages, setGeneratedImages] = useState<string[]>([]);
  const [nodes, setNodes] = useState<NodeData[]>([]);

  const addGeneratedImage = (imageUrl: string) => {
    // if the image is already in the list, don't add it again
    if (generatedImages.includes(imageUrl)) {
      return;
    }
    setGeneratedImages((prevImages) => [...prevImages, imageUrl]);
  };

  //   // Function to add a new node
  //   const onAddNode = (type: "image" | "text", content: string) => {
  //     const newNode: NodeData = {
  //       id: `node-${nodes.length + 1}`,
  //       type,
  //       content,
  //     };
  //     setNodes([...nodes, newNode]);
  //     console.log;
  //   };

  return (
    <NodeContext.Provider value={{ nodes, generatedImages, addGeneratedImage }}>
      {children}
    </NodeContext.Provider>
  );
};

// Custom hook to use the NodeContext
export const useNodeContext = () => {
  const context = useContext(NodeContext);
  if (!context) {
    throw new Error("useNodeContext must be used within a NodeProvider");
  }
  return context;
};
