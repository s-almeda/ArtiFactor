/* an editable text node*/
import { type NodeProps, NodeToolbar, Position } from "@xyflow/react";
import { useState, useRef, useEffect } from "react";
import { type TextNode } from "./types";
import { usePaletteContext } from "../contexts/PaletteContext";

export function TextNode({ data, selected }: NodeProps<TextNode>) {
  const [content, setContent] = useState(data.content || "");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { addClippedNode } = usePaletteContext();

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    data.content = e.target.value; // update this node's data
    setContent(data.content); // re-render with the new content
  };

  const maxWidth = 120; // Maximum width of the text area
  const maxLines = 10;


  useEffect(() => {
    if (textareaRef.current) {
      const textArea = textareaRef.current;
      textArea.style.height = "auto"; // Reset height to get the correct scrollHeight
      textArea.style.height = `${textArea.scrollHeight}px`;

      const context = document.createElement("canvas").getContext("2d");
      if (context) {
        context.font = getComputedStyle(textArea).font;
        const width = Math.max(...content.split('\n').map(line => context.measureText(line).width));
        textArea.style.width = `${Math.min(width + 10, maxWidth)}px`; // Add some padding and limit width to 70px
        textArea.style.whiteSpace = width > maxWidth ? "pre-wrap" : "pre"; // Wrap text if width exceeds 70px
      }

      // Make textarea scrollable if content exceeds 4 lines
      const lineHeight = parseInt(getComputedStyle(textArea).lineHeight, 10);
      const maxHeight = lineHeight * maxLines;
      if (textArea.scrollHeight > maxHeight) {
        textArea.style.height = `${maxHeight}px`;
        textArea.style.overflowY = "scroll";
      } else {
        textArea.style.overflowY = "hidden";
      }
    }
  }, [content, data]);

  return (
    
    <div className={`react-flow__node-default ${data.combinable ? 'bg-yellow-100' : ''}`} style={{ width: "fit-content" }}>

      {data.loading ? (
        <div className={`${data.combinable ? 'bg-yellow-50' : ''}`} style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", height: "100%", fontStyle: "italic" }}>
          {content.split('\n').map((line, index) => (
            <div className="text-gray-500" key={index}>{line}</div>
          ))}
          <div className="loader"></div>
        </div>
      ) : (
        <div style={{ border: "1px solid black", padding: "1px" }}>
          <textarea className={`noDrag, ${data.combinable ? 'bg-yellow-50' : ''}`}
            ref={textareaRef}
            value={content}
            onChange={handleChange}
            style={{
              resize: "none",
              border: "none",
              outline: "none",
              overflow: "hidden",
              color: "inherit",
              fontFamily: "inherit",
              fontSize: "10px",
              padding: "3px"
            }}
          />
          {selected && (
            <NodeToolbar isVisible={selected} position={Position.Top}>
              <button
                className="border-5 bg-white border-gray-800 shadow-lg rounded-full hover:bg-gray-400 dark:hover:bg-gray-400"
                type="button"
                onClick={() => addClippedNode(
                  {
                    type: 'text',
                    content: content,
                    prompt: "none"
                  })}
                aria-label="Save to Palette"
                style={{ marginRight: '0px' }}
              >
                📎
              </button>
            </NodeToolbar>
          )}
        </div>
      )}
    </div>
  );
}
