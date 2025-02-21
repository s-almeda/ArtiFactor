// DEPRECATED!  (we use TextWithKeywordsNode for text, or the LoadingNode to show a loading message now! )
// /* an editable text node*/
// import { type NodeProps, NodeToolbar, Position } from "@xyflow/react";
// import { useState, useRef, useEffect } from "react";
// import { type TextNode } from "./types";
// import { usePaletteContext } from "../context/PaletteContext";
// import { motion } from "framer-motion";

// export function TextNode({ data, selected }: NodeProps<TextNode>) {
//   const [content, setContent] = useState(data.content || "");
//   const textareaRef = useRef<HTMLTextAreaElement>(null);
//   const { addClippedNode, getNextPaletteIndex } = usePaletteContext();
    
//   const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
//     data.content = e.target.value; // update this node's data
//     setContent(data.content); // re-render with the new content
//   };

//   const maxLines = 10;


//   useEffect(() => {
//     if (textareaRef.current) {
//       const textArea = textareaRef.current;
//       textArea.style.height = "auto"; // Reset height to get the correct scrollHeight
//       textArea.style.height = `${textArea.scrollHeight}px`;

//       // Make textarea scrollable if content exceeds 4 lines
//       const lineHeight = parseInt(getComputedStyle(textArea).lineHeight, 10);
//       const maxHeight = lineHeight * maxLines;
//       if (textArea.scrollHeight > maxHeight) {
//         textArea.style.height = `${maxHeight}px`;
//         textArea.style.overflowY = "scroll";
//       } else {
//         textArea.style.overflowY = "hidden";
//       }
//     }
//   }, [content, data]);

//   return (
//       <>
//         {data.loading ? (
//           // initial animation for loading nodes
//         <motion.div
//         initial={{ opacity: 0, x:0, y: 3, scale: 0.6}}
//         animate={ { opacity: 1, x: 0, y: 0, scale: 1,  scaleX:1}}
//         transition={{ duration: 0.4, type: "spring", bounce: 0.1 }}
//         className ="p-3 border border-gray-700 rounded bg-white z-10" //Z INDEX
//         >
        
//         <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", height: "100%", fontStyle: "italic" }}>
//           {content.split('\n').map((line, index) => (
//           <div className="text-gray-500" key={index}>{line}</div>
//           ))}
//           <div className="loader"></div>
//         </div>
//         </motion.div>


//         ) : (

//         //animation for editable text nodes, looks like a paper card
//         <motion.div
//         initial={{ opacity: 0.2, x:0, y: 10, scale: 1.1, rotateY: Math.random()*30-15, rotateX: -75,  filter: "drop-shadow(6px 6px 6px rgba(0, 0, 0, 0.65))"}}
//         animate={{ opacity: 1, x: 0, y: 0, scale: 1, rotateY:0, rotateX: 0, scaleX:1, filter: "drop-shadow(1px 2px 1px rgba(0, 0, 0, 0.25))"}}
//         transition={{ duration: 0.15, type: "spring", bounce: 0.1 }}
//         className={`${data.combinable ? 'bg-yellow-50' : ''} p-3 border border-gray-700 rounded bg-white`}
//         >
//       <div style={{ border: "1px solid black", padding: "1px" }} className="nodrag">
//         <textarea className={`nodrag, ${data.combinable ? 'bg-yellow-50' : ''}`}
//         ref={textareaRef}
//         value={content}
//         onChange={handleChange}
//         style={{
//           resize: "none",
//           border: "none",
//           outline: "none",
//           overflow: "hidden",
//           color: "inherit",
//           fontFamily: "inherit",
//           fontSize: "10px",
//           padding: "3px"
//         }}
//         />
//         {selected && (
//         <NodeToolbar isVisible={selected} position={Position.Top}>
//           <button
//           className="border-5 bg-white border-gray-800 shadow-lg rounded-full hover:bg-gray-400 dark:hover:bg-gray-400"
//           type="button"
//           onClick={() => addClippedNode(
//             {
//               id: getNextPaletteIndex(),
//             type: 'text',
//             content: content,
//             prompt: "none"
//             })}
//           aria-label="Save to Palette"
//           style={{ marginRight: '0px' }}
//           >
//           📎
//           </button>
//         </NodeToolbar>
//         )}
//       </div>
//       </motion.div>
//       )}
//       </>

//   );
// }
