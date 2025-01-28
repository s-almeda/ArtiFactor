import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useState, useRef, useEffect } from "react";
import { type TextNode } from "./types";

export function TextNode({ data }: NodeProps<TextNode>) {
  const [content, setContent] = useState(data.content || "");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    data.content = e.target.value; //update this node's data
    setContent(data.content); //re render with the new content
  };

  useEffect(() => {
    if (textareaRef.current) {
      const textArea = textareaRef.current;
      if (textArea.scrollHeight > textArea.clientHeight) {
        textArea.style.height = `${textArea.scrollHeight}px`;
      }
    }
  }, [content]);
  //console.log(`TextNode content is currently: ${content}`);

  return (
    (
      <div
        className="react-flow__node-default"
        style={{ width: "fit-content" }}
      >
        <textarea
          ref={textareaRef}
          value={content}
          onChange={handleChange}
          style={{
            width: "100%",
            height: "auto",
            resize: "none",
            border: "none",
            outline: "none",
            overflow: "hidden",
            color: "inherit",
            fontFamily: "inherit",
            fontSize: "inherit",
          }}
        />
        <Handle type="source" position={Position.Bottom} />
        <Handle type="source" position={Position.Right} />
        <Handle type="target" position={Position.Left} />
        <Handle type="target" position={Position.Top} />
      </div>
    )
  );
}
