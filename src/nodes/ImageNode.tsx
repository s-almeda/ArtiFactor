import { Handle, Position, type NodeProps, NodeResizer } from "@xyflow/react";
import { useState, useEffect } from "react";
import { type ImageNode } from "./types";

export function ImageNode({ data }: NodeProps<ImageNode>) {
  const [imageUrl, setImageUrl] = useState("");
  const [width, setWidth] = useState(70);
  const [height, setHeight] = useState(70);

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
        padding: "2px"
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
      <img
        src={imageUrl}
        alt="Node Image"
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
      <Handle type="source" position={Position.Bottom} />
      <Handle type="source" position={Position.Right} />
      <Handle type="target" position={Position.Left} />
      <Handle type="target" position={Position.Top} />
    </div>
  );
}
