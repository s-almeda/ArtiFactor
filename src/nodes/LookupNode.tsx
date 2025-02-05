import { Position,  NodeResizeControl , NodeToolbar } from "@xyflow/react";
import { ChevronLeft, ChevronRight} from 'lucide-react';
import { useState, useEffect, memo, useRef } from "react";
import type { NodeProps } from "@xyflow/react";
import type { LookupNode, Artwork } from "./types";

import { useDnD } from "../DnDContext";

const controlStyle: React.CSSProperties = {
  background: 'white',
  border: '1px solid grey',
  borderRadius: '0px',
  width: '10px',
  height: '10px',
  position: 'absolute', 
  bottom: 0, 
  right: 0
  
};

interface LookupNodeProps extends NodeProps<LookupNode> {
  data: {
    content: string;
    artworks: Artwork[];
    };
}


  const DraggableText = ({ content, styling }: { content: string, styling?: string }) => {
    const [_, setDraggableType, __, setDraggableContent] = useDnD();

    const onDragStart = (
      event: React.DragEvent<HTMLDivElement>,
      type: string,
      content: string
    ) => {
      event.dataTransfer.effectAllowed = "move";
      setDraggableType(type);
      setDraggableContent(content);
    };

    const getStyledContent = () => {
      switch (styling) {
        case "title":
          return <h2 className="text-sm/5 font-bold text-gray-800">{content}</h2>;
        case "artist":
          return <p className="text-xs text-gray-600">{content}</p>;
        case "description":
          return (
            <div className="bg-gray-50 p-2 rounded-lg max-h-24 overflow-y-auto">
              <p className="text-gray-700 leading-relaxed text-[10px]">{content}</p>
            </div>
          );
        default:
          return <p className="text-[10px] text-gray-500">{content}</p>;
      }
    };

    return (
      <div
        draggable
        onDragStart={(event) => onDragStart(event, "lookupText", content)}
        className="text-left mb-2"
      >
        {getStyledContent()}
      </div>
    );
  };
  const DraggableImage = ({ src, alt, height }: { src: string, alt: string, height: number }) => {
    const [_, setDraggableType, __, setDraggableContent] = useDnD();

    const onDragStart = (
      event: React.DragEvent<HTMLDivElement>,
      content: string
    ) => {
      event.dataTransfer.effectAllowed = "move";
      setDraggableType("image");
      setDraggableContent(content);
    };

    return (
      <div
        draggable
        onDragStart={(event) => onDragStart(event, src)}
        className="relative flex items-center justify-center mb-2 mx-auto"
        style={{ width: `${height * 0.45}px`, height: `${height * 0.4}px` }}
      >
        <img
          src={src}
          alt={alt}
          className="object-cover rounded-md w-full h-full"
          key={src}
        />
      </div>
    );
  };

const LookupNode = ({ data, selected }: LookupNodeProps) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [width, setWidth] = useState(150);
    const [height, setHeight] = useState(250);
    const [currentIndex, setCurrentIndex] = useState(0);
    //const [isFocused, setIsFocused] = useState(true);
    const [artworks, setArtworks] = useState( // Default artworks
    [
        {
      title: "Impression, Sunrise",
      date: 1872,
      artist: "Claude Monet",
      genre: "Landscape",
      style: "Impressionism",
      description: "This painting is a depiction of the port of Le Havre at sunrise, with small rowboats in the foreground and ships and cranes in the background. The orange sun is shown as a distinct circle, reflecting on the water below. This piece gave the Impressionist movement its name when critics seized upon the title of this painting to give the entire movement a derisive moniker.",
      image: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Monet_-_Impression%2C_Sunrise.jpg/600px-Monet_-_Impression%2C_Sunrise.jpg"
        },
        {
      title: "Morning on the Seine",
      date: 1897,
      artist: "Claude Monet",
      genre: "Landscape",
      style: "Impressionism",
      description: "Part of a series of paintings depicting the Seine River, this work showcases Monet's mastery of light and atmosphere. The artist painted the same scene at different times of day to capture varying effects of light and weather conditions.",
      image: "https://www.claude-monet.com/assets/img/paintings/morning-on-the-seine-near-giverny.jpg"
        },
        {
      title: "Water Lilies",
      date: 1919,
      artist: "Claude Monet",
      genre: "Landscape",
      style: "Impressionism",
      description: "This painting is part of Monet's famous Water Lilies series, which he painted in his garden at Giverny. The series captures the beauty and tranquility of the water garden, with its reflections and play of light.",
      image: "https://upload.wikimedia.org/wikipedia/commons/9/9e/WLA_metmuseum_Water_Lilies_by_Claude_Monet.jpg"
        }
      ]
    );

    const handlePrevious = () => {
      console.log("Previous clicked in LookupNode");
      setCurrentIndex((prev) => (prev === 0 ? data.artworks.length - 1 : prev - 1));
    };

    const handleNext = () => {
      console.log("Next clicked in LookupNode");
      setCurrentIndex((prev) => (prev === data.artworks.length - 1 ? 0 : prev + 1));
    };

    useEffect(() => {
      if (data.artworks) {
        setArtworks(data.artworks);
  
        }
    }, [data.artworks]);

    useEffect(() => {
      if (containerRef.current) {
        const {offsetWidth, offsetHeight} = containerRef.current;
        setWidth(offsetWidth);
        setHeight(offsetHeight);  
      }
    }, []);

    const currentArtwork = artworks[currentIndex];

    return (
      <>
      <span className="drag-handle__custom" />

      
          <NodeToolbar isVisible={selected} position={Position.Top}>
            
              <button
                onClick={handlePrevious}
                className="p-1 text-white-600 hover:text-white-900"
                aria-label="Previous artwork"
              >
                <ChevronLeft size={20} />
              </button>
              {/* let's add text here that corresponds to current index / total artworks like "1/3" etc*/}
                <span className="p-2 text-gray-600">{currentIndex + 1}/{artworks.length}</span>

              <button
                onClick={handleNext}
                className="p-1 text-white-600 hover:text-white-900"
                aria-label="Next artwork"
              >
                <ChevronRight size={20} />
              </button>
            </NodeToolbar>


            <NodeResizeControl
              style={controlStyle}
              minWidth={150}
              maxWidth={350}
              keepAspectRatio={true}
              color="black"
              onResize={(_, params) => {
                if (containerRef.current) {
                  containerRef.current.style.width = `${params.width}px`;
                  containerRef.current.style.height = `${params.height}px`;
                  setWidth(params.width);
                  setHeight(params.height);
                }
              }}
            />
          <div ref={containerRef} className={`max-w-sm p-4 bg-white pt-10 rounded-lg shadow-lg overflow-scroll`} style={{width, height}}>

        

        {/* Main Content of the lookup component */}

        <div className="relative flex flex-col h-full overflow-scroll cursor-default">             
              
          < DraggableImage src={currentArtwork.image} alt={currentArtwork.title} height={height} />
          < DraggableText styling="title" content={currentArtwork.title + "(" + currentArtwork.date + ")"} />
          < DraggableText styling="artist" content={currentArtwork.artist} />
          < DraggableText styling="default" content={currentArtwork.genre + " • " + currentArtwork.style} />
          < DraggableText styling="description" content={currentArtwork.description} />
              
              </div>


         </div>
       </>
      );
  };


  
  export default memo(LookupNode);

