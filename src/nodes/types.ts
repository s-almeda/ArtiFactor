import type { Node, NodeProps } from "@xyflow/react";

/** ✅ Shared properties for all custom nodes */
export type ArtifactorNodeData = {
  content?: string; // Common content field for all nodes
  className?: string;
  xOffset?: number;
  yOffset?: number;
}

/**  Specific Node Data Types */
export type TextNodeData = ArtifactorNodeData & {
  loading?: boolean;
  combinable?: boolean;
};

export type ImageNodeData = ArtifactorNodeData & {
  prompt?: string;
  activateLookUp?: (position: { x: number; y: number }, imageUrl: string) => void;
};
export type ImageWithLookupNodeData = ArtifactorNodeData & {
  prompt?: string;
  artworks?: Artwork[];
  //activateLookUp?: (position: { x: number; y: number }, imageUrl: string) => void;
};

export type LookupNodeData = ArtifactorNodeData & {
  artworks: Artwork[];
};

export type SynthesizerNodeData = ArtifactorNodeData & {
  mode: "ready" | "generating-image" | "dragging" | "generating-text";
  inputNodeContent?: string;
  //updateNode: (content: string, mode: "ready" | "generating" | "dragging" | "check") => boolean;
};

export type TextWithKeywordsNodeData = ArtifactorNodeData & {
  words: Array<Word | Keyword>;
};

/** Generalized AppNode Type */
export type AppNode<T extends ArtifactorNodeData = ArtifactorNodeData> = Node<T>;

/** Specialized Nodes */
export type TextNode = AppNode<TextNodeData>;
export type ImageNode = AppNode<ImageNodeData>;
export type LookupNode = AppNode<LookupNodeData>;
export type SynthesizerNode = AppNode<SynthesizerNodeData>;
export type TextWithKeywordsNode = AppNode<TextWithKeywordsNodeData>;
export type ImageWithLookupNode = AppNode<ImageWithLookupNodeData>;

/** Node Props for Custom Components */
export type AppNodeProps = NodeProps<AppNode>;

/** Artwork & Keyword Types */
export type Word = {
  value: string;
}

export type Keyword = Word & {
  id: string;
  databaseValue: string;
  isArtist?: boolean;
  isArtwork?: boolean;
  aliases?: string[];
  type: string;
  description: string;
  shortDescription?: string;
  relatedKeywordStrings: string[];
  relatedKeywordIds: string[];
}
// id TEXT PRIMARY KEY,
// value TEXT NOT NULL,
// isArtist BOOLEAN,
// isArtwork BOOLEAN,
// type TEXT NOT NULL,
// aliases TEXT,
// description TEXT,
// relatedKeywordIds TEXT

// export type TextWithKeywords = Array<Word | Keyword>;

export interface Artwork {
  title: string;
  date: number;
  artist: string;
  keywords: Keyword[];
  description: string;
  image: string;
}

// export interface Artist {
//   id: string;
//   name: string;
//   aliases?: string[];
//   relatedArtists: string[];
//   relatedKeywords: Keyword[];
// }
