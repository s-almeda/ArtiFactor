import type { NodeTypes } from "@xyflow/react";

import { ImageWithLookupNode } from "./ImageWithLookupNode";
import { TextWithKeywordsNode } from "./TextWithKeywordsNode";
import { LoadingNode } from "./LoadingNode";

import { AppNode } from "./types"; //TextNodeData, SynthesizerNodeData


export const initialNodes: AppNode[] = [
  // {
  //   id: "example",
  //   type: "text",
  //   position: { x: 100, y: 100 },
  //   data: { content: "bunny on the moon", loading: false, combinable: false } as TextNodeData,
  // },
  // {
  //   id: "synthesizer-1",
  //   type: "synthesizer",
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
  //   } as SynthesizerNodeData,
  // },
];

export const nodeTypes: NodeTypes = {
  default: LoadingNode,
  image: ImageWithLookupNode,
  text: TextWithKeywordsNode,
  imagewithlookup: ImageWithLookupNode,
  textwithkeywords: TextWithKeywordsNode,
};
