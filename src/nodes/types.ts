import type { Node, BuiltInNode } from "@xyflow/react";

// Define a generic node type for default nodes
export type DefaultNode = Node<{ label: string; content: string }, "default">;

// all our types of nodes...
export type PositionLoggerNode = Node<{ content: string }, "position-logger">;
export type TextNode = Node<{ content: string }, "text">;
export type ImageNode = Node<{ content: string}, "image">;
export type FunctionNode = Node<{ content: string }, "function">;
export type PaletteNode = Node< {content: string }, "palette">
// export type IntersectionNode = Node<{
//   updateNode: (content: string) => void; 
//   content: string; 
//   className?: string 
// },"intersection">;
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
  | FunctionNode
  | PaletteNode
//  | IntersectionNode
  | T2IGeneratorNode;

