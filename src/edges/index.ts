import type { Edge, EdgeTypes } from "@xyflow/react";
import WireEdge from "./WireEdge";

export const initialEdges: Edge[] = [
  // { id: "a->c", source: "a", target: "c", animated: true },
  // { id: "b->d", source: "b", target: "d" },
  // { id: "c->d", source: "c", target: "d", animated: true },
  // add wire edge
  // { id: "a->f", source: "a", target: "f", type: "wire" },
];

export const edgeTypes = {
  // Add your custom edge types here!
  wire: WireEdge,
} satisfies EdgeTypes;
