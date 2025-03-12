import type { NodeTypes } from "@xyflow/react";

import { ImageWithLookupNode } from "./ImageWithLookupNode";
import { TextWithKeywordsNode } from "./TextWithKeywordsNode";
import { LoadingNode } from "./LoadingNode";

import { TextWithKeywordsNodeData } from "./types"; //TextNodeData, SynthesizerNodeData


export const defaultTextWithKeywordsNodeData: TextWithKeywordsNodeData = {
  words: [
    "your", "text", "here.", "click", "the", "pencil", "icon", "in", "the", "top", "right", "to", "edit.",
    "Use", "Option+Click", "to", "generate", "new", "nodes."
  ].map(value => ({ value })),
  intersections: [],
  similarTexts: [],
  provenance: "user",
  hasNoKeywords: true,
  hasNoSimilarTexts: true
};

export const nodeTypes: NodeTypes = {
  default: LoadingNode,
  image: ImageWithLookupNode,
  text: TextWithKeywordsNode,
  imagewithlookup: ImageWithLookupNode,
  textwithkeywords: TextWithKeywordsNode,
};
