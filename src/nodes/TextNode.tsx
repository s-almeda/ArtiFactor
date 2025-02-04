import { type NodeProps } from "@xyflow/react";
import { useState, useRef, useEffect } from "react";
import { type TextNode } from "./types";

export function TextNode({ data }: NodeProps<TextNode>) {
  const [content, setContent] = useState(data.content || "");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    data.content = e.target.value; // update this node's data
    setContent(data.content); // re-render with the new content
  };

  useEffect(() => {
    if (textareaRef.current) {
      const textArea = textareaRef.current;
      if (textArea.scrollHeight > textArea.clientHeight) {
        textArea.style.height = `${textArea.scrollHeight}px`;
      }
    }
  }, [content]);

  return (
    <div
      className={`react-flow__node-default`}
      style={{ width: "fit-content" }}
    >
      <textarea
        ref={textareaRef}
        value={content}
        onChange={handleChange}
        style={{
          width: "100%",
          height: "10%",
          resize: "none", // todo, make these text nodes dynamically resizable
          border: "none",
          outline: "none",
          overflow: "y-scroll",
          color: "inherit",
          fontFamily: "inherit",
          fontSize: "12px",
        }}
      />
      {data.loading && <div className="loader"></div>}
      {/* 
      // handles for connecting edges - not needed!
      <Handle type="source" position={Position.Bottom} />
      <Handle type="source" position={Position.Right} />
      <Handle type="target" position={Position.Left} />
      <Handle type="target" position={Position.Top} /> */}
    </div>
  );
}
