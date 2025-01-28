// /* -- note: commented out because it is basically deprecated, reimplemented as T2IGenerator Node -- */
// import { memo, useEffect } from "react";
// import { Handle, Position, type NodeProps } from "@xyflow/react";
// //import { type IntersectionNode } from "./types";

// const IntersectionNode = ({ data }: NodeProps<IntersectionNode>) => {
//   const printContent = (inputNodeContent: string | 'owo! drag content error!') => {
//     console.log(`Overlapping node content: ${inputNodeContent}`);
//   };

//   useEffect(() => {
//     (data as any).updateNode = (content: string) => printContent(content);
//   }, [data]);

    
//   return (
//     <div
//       style={{
//         padding: 10,
//         border: "1px solid #333",
//         borderRadius: "5px",
//       }}
//       className={data.className} // Add highlight class dynamically
//     >
//       <div>{data.content || "Intersection Node"}</div>
//       <Handle type="target" position={Position.Top} />
//       <Handle type="source" position={Position.Bottom} />
//     </div>
//   );
// };

// export default memo(IntersectionNode);
