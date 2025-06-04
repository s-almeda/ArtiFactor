// import React, { useState, useEffect } from 'react';
// import { Entry } from '../nodes/types'; // Adjust the import path as necessary
// import { useDnD } from '../context/DnDContext'; // Adjust the import path as necessary

// export const DynamicDescription: React.FC<{
//     descriptions: Entry[];
//     isAIGenerated: boolean;
//     as?: 'div' | 'span';
// }> = ({ descriptions, isAIGenerated, as = 'div' }) => {
//     const [selectedSource, setSelectedSource] = useState<string | null>(null);
//     const { setDraggableType, setDraggableData } = useDnD();

//     useEffect(() => {
//         setSelectedSource(
//             descriptions.find((entry) => entry.source === "synth")?.source ||
//             descriptions[0]?.source ||
//             null
//         );
//     }, [descriptions]);

//     const getSelectedDescription = (): string =>
//         descriptions.find((entry) => entry.source === selectedSource)?.description || "";

//     if (!descriptions || descriptions.length === 0) {
//         return <span className="text-gray-500 italic">No descriptions available.</span>;
//     }

//     const handleSourceClick = (source: string) => {
//         setSelectedSource(source);
//     };

//     const handleDragStart = (event: React.DragEvent<HTMLSpanElement>) => {
//         const selectedDescription = getSelectedDescription();
//         event.dataTransfer.effectAllowed = "move";
//         setDraggableType("text");
//         setDraggableData({ 
//             content: selectedDescription, 
//             provenance: selectedSource === "synth" ? "ai-description" : "description"
//         });
//     };

//     const selectedDescription = getSelectedDescription();
//     const Wrapper = as;

//     return (
//         <Wrapper className="mb-4">
//             {selectedDescription ? (
//                 <span
//                     className={`cursor-move inline-block p-1 text-xs 
//                         ${ isAIGenerated
//                             ? "hover:bg-blue-200"
//                             : "hover:bg-[#dbcdb4]"
//                     }`}
//                     draggable={true}
//                     onDragStart={handleDragStart}
//                 >
//                     {selectedDescription}
//                 </span>
//             ) : (
//                 <span className="text-gray-500 italic inline-block p-2">No description available.</span>
//             )}
            
//             <span className="italic text-gray-500 text-xs mt-1 block" style={{ userSelect: 'none' }}>
//                 Source:{" "}
//                 {descriptions
//                     .sort((a, b) => (a.source === "synth" ? -1 : b.source === "synth" ? 1 : 0))
//                     .map((entry) => (
//                         <button
//                             key={entry.source}
//                             onClick={() => handleSourceClick(entry.source)}
//                             className={`px-1 py-1 mx-0.5 rounded text-xxs ${
//                                 entry.source === selectedSource
//                                     ? isAIGenerated
//                                         ? "bg-blue-500 text-white"
//                                         : "bg-stone-400 text-white"
//                                     : "bg-gray-200 text-gray-700 hover:bg-gray-300"
//                             }`}
//                         >
//                             {entry.source}
//                         </button>
//                     ))}
//             </span>
//         </Wrapper>
//     );
// };

// export default DynamicDescription;
