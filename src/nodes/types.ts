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
  prompt: string;
  intersections: Array<{ id: string; position: { x: number; y: number }; content: string; }>; //info on nodes that are overlapping with this node
  similarArtworks?: Artwork[];
  provenance?: "history" | "user" | "ai"; // history = straight from the database, user = user added/edited, ai = generated
  parentNodeId?: string;
  lookUpOn?: boolean;
};

export type TextWithKeywordsNodeData = ArtifactorNodeData & {
  words: Array<Word | Keyword>; //to get the words as a string, use content
  intersections: Array<{ id: string; position: { x: number; y: number }; content: string; }>; //info on nodes that are overlapping with this node
  similarTexts?: Keyword[];
  provenance?: "history" | "user" | "ai"; // history = straight from the database, user = user added/edited, ai = generated
  hasNoKeywords?: boolean;
  hasNoSimilarTexts?: boolean;
  parentNodeId?: string;
};


/** Generalized AppNode Type */
export type AppNode<T extends ArtifactorNodeData = ArtifactorNodeData> = Node<T>;

/** Specialized Nodes */
export type LoadingNode = AppNode<ArtifactorNodeData>;
export type TextWithKeywordsNode = AppNode<TextWithKeywordsNodeData>;
export type ImageWithLookupNode = AppNode<ImageWithLookupNodeData>;


/** Node Props for Custom Components */
export type AppNodeProps = NodeProps<AppNode>;

/** Artwork & Keyword Types */
export type Word = {
  value: string;
}

export type Keyword = Word & { // a keyword is a word (with a value) and possibly a bunch of additional properties
  entryId: string;
  databaseValue: string; // Corresponds to text_entries.value, may be the same as value, is likely different when doing a keyword lookup task
  images: string[]; // JSON string array of image_entries.image_ids relevant to this keyword
  isArtist?: boolean; // Corresponds to text_entries.isArtist (converted from INTEGER to boolean)
  type: string; // Corresponds to text_entries.type
  aliases?: string[]; // JSON string array (only if isArtist == 1, corresponds to text_entries.artist_aliases)
  descriptions: Entry[]; // Can be a series of Entry objects
  relatedKeywordIds: string[]; // JSON string array of text_entries.entry_id
  relatedKeywordStrings: string[]; // JSON string array of related keyword labels
}


export type Entry = {
  source: string; // The source of the description, e.g., "synth", "user", etc.
  description: string; // A description for the entry, can be an empty string
  [key: string]: any; // Allows any additional key-value pairs
};



export interface Artwork {
  image_id: string;
  image_url: string;
  image_urls: Record<string, string>;
  filename: string;
  value: string; // artwork title or label
  artist_names: string[]; // array of artist names (as strings)
  
  descriptions: Record<
    string,
    {
      date?: string;
      description?: string;
      [key: string]: any;
    }
  >;

  relatedKeywordIds: string[];
  relatedKeywordStrings: string[];

  rights?: string;
  distance?: number;
}
