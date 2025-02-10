import {Position, NodeResizeControl, NodeToolbar } from "@xyflow/react";
import { useState, useEffect } from "react";
import type { NodeProps } from "@xyflow/react";
import type { ImageNode } from "./types";
import { usePaletteContext } from "../context/PaletteContext";
import { useDnD } from "../context/DnDContext";
import { motion } from "framer-motion";

const controlStyle: React.CSSProperties = {
  background: 'white',
  //border: '1px solid grey',
  width: '8px',
  height: '8px',
  position: 'absolute', 
  bottom: 0, 
  right: 0
  
};

  const DraggableText = ({ content }: { content: string }) => {
    const [_, setDraggableType, __, setDraggableData] = useDnD();

    const onDragStart = (
      event: React.DragEvent<HTMLDivElement>,
      content: string
    ) => {
      event.dataTransfer.effectAllowed = "move";
      setDraggableType("text");
      setDraggableData({content: content});
    };

    return (

      <div
        draggable
        onDragStart={(event) => onDragStart(event, content)}
        className="text-left mb-2"
      >
        <h3 className="mt-2 text-xs/3 italic text-gray-800">"{content}"</h3>
      </div>
    );
  };


// interface ImageNodeProps extends NodeProps<ImageNode> {
//   //ons: (imageUrl: string) => void;
// }


export function ImageNode({ data, selected, positionAbsoluteX, positionAbsoluteY }: NodeProps<ImageNode>) {
  const { addClippedNode } = usePaletteContext(); 
  const [imageUrl, setImageUrl] = useState("");
  const [showPrompt, setShowPrompt] = useState(false);
  const [width, setWidth] = useState(80);
  const [height, setHeight] = useState(80);

  useEffect(() => {
    if (data.content) {
      setImageUrl(data.content);
    }
  }, [data.content]);

  return (
    <motion.div
      initial={{ opacity: 0, x: 0, y: 10, scale: 1.1, rotateY: -45, filter: "blur(10px) drop-shadow(10px 10px 10px rgba(0, 0, 0, 0.55))" }}
      animate={{ opacity: 1, x: 0, y: 0, scale: 1, rotateY: 0, scaleX: 1, filter: "drop-shadow(1px 2px 1px rgba(0, 0, 0, 0.15))" }}
      transition={{ duration: 0.3, type: "spring", bounce: 0.1 }}
    >
      <div
        className="react-flow__node-default"
        style={{
          width: `${width}px`,
          height: `${height}px`,
          position: "relative",
          padding: "2px",
        }}
      >
        <NodeResizeControl
          className="controlStyle"
          style={controlStyle}
          minWidth={50}
          minHeight={50}
          keepAspectRatio={true}
          color="black"
          onResize={(_, params) => {
            const aspectRatio = width / height;
            const newHeight = Math.floor(params.width / aspectRatio);
            setWidth(params.width);
            setHeight(newHeight);
          }}
        />
        <img
          className="drag-handle__invisible"
          src={imageUrl}
          alt={data.prompt || "generated image"}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
        {/* show the prompt if it exists and the button has been pressed! */}
        {showPrompt && data.prompt !== "None" && (
          <DraggableText content={data.prompt || ""} />
        )}
        {/* React Flow's built-in NodeToolbar */}
        <NodeToolbar isVisible={selected} position={Position.Top}>
          <button
            type="button"
            className="border-5 bg-white border-gray-800 shadow-lg rounded-full hover:bg-gray-400 dark:hover:bg-gray-400"
            onClick={() => {
              data.activateLookUp && data.activateLookUp({ x: positionAbsoluteX, y: positionAbsoluteY }, imageUrl);
            }} // calls the function passed to it in the data prop
            aria-label="Action 1"
            style={{ marginRight: '4px' }}
          >
            🔍
          </button>
          <button
            className="border-5 bg-white border-gray-800 shadow-lg rounded-full hover:bg-gray-400 dark:hover:bg-gray-400"
            type="button"
            onClick={() => addClippedNode(
              {
                type: 'image',
                content: imageUrl,
                prompt: data.prompt || ""
              }
            )} // Add to Palette
            aria-label="Save to Palette"
            style={{ marginRight: '4px' }}
          >
            📎
          </button>
          {data.prompt !== "None" && (
            <button
              type="button"
              className="border-5 bg-white border-gray-800 shadow-lg rounded-full hover:bg-gray-400 dark:hover:bg-gray-400"
              onClick={() => setShowPrompt(!showPrompt)}
              aria-label="View Prompt"
              style={{ marginRight: '0px' }}
            >
              📝
            </button>
          )}
          </NodeToolbar>
      </div>
    </motion.div>
  );
}
