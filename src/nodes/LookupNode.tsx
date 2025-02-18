import { Position,  NodeResizeControl , NodeToolbar } from "@xyflow/react";
import { ChevronLeft, ChevronRight} from 'lucide-react';
import { useState, useEffect, memo, useRef } from "react";
import type { NodeProps } from "@xyflow/react";
import type { LookupNode, Artwork, Keyword, Word} from "./types";

import { useDnD } from "../context/DnDContext";

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
    const [_, setDraggableType, __, setDraggableData] = useDnD();

    const onDragStart = (
      event: React.DragEvent<HTMLDivElement>,
      content: string
    ) => {
      event.dataTransfer.effectAllowed = "move";
      setDraggableType("text");
      setDraggableData({content: content});
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
              <p className="text-gray-700 leading-relaxed text-[12px]">{content}</p>
            </div>
          );
        default:
          return <p className="text-[12px] text-gray-500">{content}</p>;
      }
    };

    return (
      <div
        draggable
        onDragStart={(event) => onDragStart(event, content)}
        className="text-left mb-0 mt-0"
      >
        {getStyledContent()}
      </div>
    );
  };
  const DraggableImage = ({ src, alt, height }: { src: string, alt: string, height: number }) => {
    const [_, setDraggableType, __, setDraggableData] = useDnD();

    const onDragStart = (
      event: React.DragEvent<HTMLDivElement>,
      content: string
    ) => {
      event.dataTransfer.effectAllowed = "move";
      setDraggableType("image");
      setDraggableData({content: content, prompt: alt});
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
    // default artworks state, if lookup node doesn't get anything 
    const [artworks, setArtworks] = useState<Artwork[]>([
      {/*---  some example artwork data ---*/
        title: "Impression, Sunrise",
        date: 1872,
        artist: "Claude Monet",
        keywords: [
          { id: "1", type: "genre", value: "Landscape", description: "A genre of art that depicts natural scenery." } as Keyword,
          { id: "2", type: "style", value: "Impressionism", description: "An art movement characterized by small, thin brush strokes and an emphasis on light and its changing qualities." }as Keyword 
        ],
        description: "This painting is a depiction of the port of Le Havre at sunrise, with small rowboats in the foreground and ships and cranes in the background. The orange sun is shown as a distinct circle, reflecting on the water below. This piece gave the Impressionist movement its name when critics seized upon the title of this painting to give the entire movement a derisive moniker.",
        image: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Monet_-_Impression%2C_Sunrise.jpg/600px-Monet_-_Impression%2C_Sunrise.jpg"
      }
    ]);

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
              
          <DraggableImage src={currentArtwork.image} alt={`${currentArtwork.title}${currentArtwork.date.toString() !== "None" ? ` (${currentArtwork.date})` : ''} by ${currentArtwork.artist}`} height={height} />
          <DraggableText styling="title" content={currentArtwork.date.toString() !== "None" ? `${currentArtwork.title} (${currentArtwork.date})` : currentArtwork.title} />
          <DraggableText styling="artist" content={currentArtwork.artist} />
          {currentArtwork.keywords.map((keyword) => (
            <DraggableText key={keyword.id} styling="default" content={`${keyword.type}: ${keyword.value}`} />
          ))}
          <DraggableText styling="description" content={currentArtwork.description} />
              
        </div>


         </div>
       </>
      );
  };


  
  export default memo(LookupNode);

