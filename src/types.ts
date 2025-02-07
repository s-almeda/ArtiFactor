
//types.ts
import {
  type Node,
  type BuiltInNode,
} from '@xyflow/react';


/*--- First, let's define all our special custom nodes... --*/

// Define a generic node type for default nodes...
export type DefaultNode = Node<{ label: string; content: string }, "default">;

// all our types of nodes...
export type TextNode = Node<{ 
  content: string; 
  loading: boolean, //if loading is true, its a loading message!
  combinable:boolean}, 
  "text">; 
export type ImageNode = Node<{ 
  content: string 
  prompt: string;
  lookUp: (position: {x: number; y:number}, 
    imageUrl: string) => void;
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

/*- export them all as types of an AppNode -*/

// Aggregate node types
export type AppNode =
  | BuiltInNode
  | DefaultNode
  | TextNode
  | ImageNode
  | LookupNode
  | FunctionNode
  | PaletteNode
  | T2IGeneratorNode;

/*----- Define our AppState ----*/

export interface Clipping { //a generic interface for clippings... 
// currently these are distinct from nodes because you can have Many duplicate nodes in the canvas of the same concept, but only ONE clipping per piece of content. idk if this is the best solution for that tho
  type: "image" | "text"; 
  content: string; // either text, or an imageUrl
  prompt: string; // for storing the prompt used to generate the image OR (if its not ai generated) the title/artist alt text for the artwork
}




/* also define what an artwork is, and the kind of data associated with each artwork--*/
export interface keyword{
  //"gene"
  id: string;
  aliases?: string[]; // optional array of aliases, e.g. different words for this same concept
  type: string; // eg. genre, style, movement, medium -- Artsy's "genes" https://www.artsy.net/categories
  value: string;
  description: string;
}
export interface Artwork {
  title: string;
  date: number;
  artist: string;
  keywords: keyword[]; // an array of keywords
  description: string;
  image: string;
}

export interface Artist{
  id: string; // unique identifier from Artsy
  name: string;
  aliases?: string[]; // optional array of aliases (e.g., last name)
  relatedArtists: string[]; // list of related artists from Artsy, filtered to those in the WikiArt database
  relatedKeywords: keyword[]; // list of related keywords from Artsy - e.g. andy warhol is known for "pop art"
}
