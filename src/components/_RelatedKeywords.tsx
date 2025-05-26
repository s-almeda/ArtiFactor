// import React from "react";
// import { useDnD } from "../context/DnDContext";

// export const RelatedKeywords: React.FC<{
//     relatedKeywords: string[];
//     isAIGenerated: boolean;
// }> = ({ relatedKeywords, isAIGenerated }) => {
//     const { setDraggableType, setDraggableData } = useDnD();

//     const onDragStart = (
//         event: React.DragEvent<HTMLDivElement>,
//         content: string
//     ) => {
//         event.dataTransfer.effectAllowed = "move";
//         setDraggableType("text");
//         setDraggableData({ content: content, provenance: "history" });
//     };

//     return (
//         <div className="nodrag p-2 flex flex-wrap gap-0.5">
//             <strong className="text-gray-900 text-sm italic mr-1">see also: </strong>
//             {relatedKeywords.map((relatedKeyword, index) => (
//                 <div
//                     key={index}
//                     className={`text-xs p-0.5 rounded-sm cursor-grab ${
//                         isAIGenerated
//                             ? "text-blue-50 bg-blue-500 hover:bg-blue-400"
//                             : "text-stone-50 bg-stone-500 hover:bg-stone-400"
//                     }`}
//                     draggable
//                     onDragStart={(event) => onDragStart(event, relatedKeyword)}
//                 >
//                     {relatedKeyword}
//                 </div>
//             ))}
//         </div>
//     );
// };