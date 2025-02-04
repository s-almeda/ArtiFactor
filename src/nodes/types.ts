import type { Node, BuiltInNode } from "@xyflow/react";


export interface Artwork {
  title: string;
  date: number;
  artist: string;
  genre: string;
  style: string;
  description: string;
  image: string;
}

// Define a generic node type for default nodes
export type DefaultNode = Node<{ label: string; content: string }, "default">;

// all our types of nodes...
export type PositionLoggerNode = Node<{ content: string }, "position-logger">;
export type TextNode = Node<{ content: string; loading: boolean }, "text">;
export type ImageNode = Node<{ 
  content: string 
  lookUp: (position: {x: number; y:number}, imageUrl: string) => void;
}, "image">;
export type FunctionNode = Node<{ content: string }, "function">;
export type PaletteNode = Node< {content: string }, "palette">;

export type LookupNode = Node< {
  content: string 
  artworks: Artwork[];
}, "lookup">;

export type T2IGeneratorNode = Node<{
  yOffset: number;
  xOffset: any;
  mode: string; // a text to image generator node  
  updateNode: (content: string, mode: "ready" | "generating" | "dragging" | "check") => boolean;  
  content: string; 
  className?: string;
},"t2i-generator">;




// Aggregate node types
export type AppNode =
  | BuiltInNode
  | DefaultNode
  | TextNode
  | PositionLoggerNode
  | ImageNode
  | LookupNode
  | FunctionNode
  | PaletteNode
//  | IntersectionNode
  | T2IGeneratorNode;

