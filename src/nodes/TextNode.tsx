/* an editable text node*/
import { type NodeProps } from "@xyflow/react";//, NodeToolbar, Position } from "@xyflow/react";
import { useState, useRef, useEffect } from "react";
import { type TextNode } from "./types";


export function TextNode({ data }: NodeProps<TextNode>) { //selected, positionAbsoluteX, positionAbsoluteY
  const [content, setContent] = useState(data.content || "");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // const updateNode = (content: string) => { 
  //   setContent(content);
  // }

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
  }, [content, data]);

  return (
    <div className={`react-flow__node-default ${data.combinable ? 'bg-yellow-100' : ''}`} style={{ width: "fit-content" }}>
      {/* if this is just a loading message.... */}
      {data.loading ? (
      <div className={`${data.combinable ? 'bg-yellow-50' : ''}`} style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", height: "100%", fontStyle: "italic" }}>
        {content.split('\n').map((line, index) => (
        <div className="text-gray-500" key={index}>{line}</div>
        ))}
        <div className="loader"></div>
      </div>
      ) : 
      (
      <div style={{ border: "1px solid black", padding: "1px" }}>
      {/*this is NOT a loading message! */}
        <textarea className={`${data.combinable ? 'bg-yellow-50' : ''}`}
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
          fontSize: "10px",
          padding: "3px"
        }}
        />
        {/* {selected && (
        <NodeToolbar isVisible={selected} position={Position.Top}>
          <button
          type="button"
          className="border-5 bg-white border-gray-800 shadow-lg rounded-full hover:bg-gray-400 dark:hover:bg-gray-400"
          onClick={() => console.log("clicked")} // calls the function passed to it in the data prop
          aria-label="Action 1"
          style={{ marginRight: '8px' }}
          >
          🔍
          </button>
          <button
          className="border-5 bg-white border-gray-800 shadow-lg rounded-full hover:bg-gray-400 dark:hover:bg-gray-400"
          type="button"
          onClick={() => console.log("clicked")} // Add to Palette
          aria-label="Save to Palette"
          style={{ marginRight: '0px' }}
          >
          🖼️
          </button>
        </NodeToolbar>
        )} */}
      </div>
      )}
    </div>
  );
}
