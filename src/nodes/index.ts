import type { NodeTypes } from "@xyflow/react";
import { TextNode } from "./TextNode";
import { ImageNode } from "./ImageNode";
import FunctionNode from "./FunctionNode";
import T2IGeneratorNode from "./T2IGeneratorNode";
import LookupNode from "./LookupNode";
import { AppNode, TextNodeData, T2IGeneratorNodeData } from "./types";

export const initialNodes: AppNode[] = [
  {
    id: "example",
    type: "text",
    position: { x: 100, y: 100 },
    data: { content: "bunny on the moon", loading: false, combinable: false } as TextNodeData,
  },
  // {
  //   id: "t2i-generator-1",
  //   type: "t2i-generator",
  //   position: { x: 10, y: 10 },
  //   data: {
  //     content: "",
  //     mode: "ready",
  //     yOffset: 0,
  //     xOffset: 0,
  //     updateNode: (content: string, mode: "dragging" | "ready" | "generating" | "check") => {
  //       console.log(`new node with content: ${content} and mode: ${mode}`);
  //       return true;
  //     },
  //   } as T2IGeneratorNodeData,
  // },
];

export const nodeTypes: NodeTypes = {
  default: TextNode,
  text: TextNode,
  image: ImageNode,
  function: FunctionNode,
  lookup: LookupNode,
  "t2i-generator": T2IGeneratorNode,
};
