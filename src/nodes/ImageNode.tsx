import {Position, NodeResizer, NodeToolbar } from "@xyflow/react";
import { useState, useEffect } from "react";
import type { NodeProps } from "@xyflow/react";
import type { ImageNode } from "./types";


interface ImageNodeProps extends NodeProps<ImageNode> {
  //ons: (imageUrl: string) => void;
}


export function ImageNode({ data, selected, positionAbsoluteX, positionAbsoluteY }: ImageNodeProps) {
  const [imageUrl, setImageUrl] = useState("");
  const [width, setWidth] = useState(200);
  const [height, setHeight] = useState(200);

  useEffect(() => {
    if (data.content) {
      setImageUrl(data.content);
    }
  }, [data.content]);

  return (
    <div
      className="react-flow__node-default"
      style={{
        width: `${width}px`,
        height: `${height}px`,
        position: "relative",
        padding: "2px",
      }}
    >
      <NodeResizer
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
        isVisible={true}
      />
      {/* React Flow's built-in NodeToolbar */}
      <NodeToolbar isVisible={selected} position={Position.Top}>
        <button
          type="button"
          onClick={() => data.lookUp({x: positionAbsoluteX, y: positionAbsoluteY}, imageUrl)} // calls the function passed to it in the data prop
          aria-label="Action 1"
        >
          🔍
        </button>
        <button
          type="button"
          onClick={() => alert("Action 2")}
          aria-label="Action 2"
        >
          🖼️
        </button>
      </NodeToolbar>

      <img
        src={imageUrl}
        alt="Generated Output"
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
      {/* <Handle type="source" position={Position.Bottom} />
      <Handle type="source" position={Position.Right} />
      <Handle type="target" position={Position.Left} />
      <Handle type="target" position={Position.Top} /> */}
    </div>
  );
}
