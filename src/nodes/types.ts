import type { Node, NodeProps } from "@xyflow/react";
//all nodes will have these things by default from react: https://reactflow.dev/api-reference/types/node#fields

/** ✅ Shared properties for all custom nodes */
export type ArtifactorNodeData = { // all of our nodes will have these things! 
  content?: string; // Common content field for all nodes
  className?: string;
  xOffset?: number;
  yOffset?: number;
}

/**  Specific Node Data Types */


export type ImageWithLookupNodeData = ArtifactorNodeData & {
  prompt?: string;
  artworks?: Artwork[];
  provenance?: "history" | "user" | "ai"; // history = straight from the database, user = user added/edited, ai = generated
  //activateLookUp?: (position: { x: number; y: number }, imageUrl: string) => void;
};

export type TextWithKeywordsNodeData = ArtifactorNodeData & {
  words: Array<Word | Keyword>;
  wordsAsString: string;
  provenance?: "history" | "user" | "ai"; // history = straight from the database, user = user added/edited, ai = generated
};

/* --------------- DEPRECATED --------------- */


// export type TextNodeData = ArtifactorNodeData & {
//   loading?: boolean;
//   combinable?: boolean;
// };

// export type LookupNodeData = ArtifactorNodeData & {
//   artworks: Artwork[];
// };

// export type SynthesizerNodeData = ArtifactorNodeData & {
//   mode: "ready" | "generating-image" | "dragging" | "generating-text";
//   inputNodeContent?: string;
//   //updateNode: (content: string, mode: "ready" | "generating" | "dragging" | "check") => boolean;
// };
// export type ImageNodeData = ArtifactorNodeData & {
//   prompt?: string;
//   activateLookUp?: (position: { x: number; y: number }, imageUrl: string) => void;
// };


/** Generalized AppNode Type */
export type AppNode<T extends ArtifactorNodeData = ArtifactorNodeData> = Node<T>;

/** Specialized Nodes */
export type LoadingNode = AppNode<ArtifactorNodeData>;
export type TextWithKeywordsNode = AppNode<TextWithKeywordsNodeData>;
export type ImageWithLookupNode = AppNode<ImageWithLookupNodeData>;

// export type TextNode = AppNode<TextNodeData>;
// export type ImageNode = AppNode<ImageNodeData>;
// export type LookupNode = AppNode<LookupNodeData>;
// export type SynthesizerNode = AppNode<SynthesizerNodeData>;

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
