import React from "react";
import { EdgeProps, getBezierPath, useReactFlow } from "@xyflow/react";

const WireEdge: React.FC<EdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style = {},
}) => {
  const [edgePath] = getBezierPath({ sourceX, sourceY, targetX, targetY });
  const reactFlowInstance = useReactFlow();

  // Calculate the midpoint for the delete button
  const midX = (sourceX + targetX) / 2;
  const midY = (sourceY + targetY) / 2;

  // Function to handle deletion of the edge
  const handleDeleteEdge = () => {
    console.log("deleting");
    reactFlowInstance.setEdges((edges) =>
      edges.filter((edge) => edge.id !== id)
    );
  };

  return (
    <g>
      <path
        id={id}
        d={edgePath}
        style={{
          fill: "none",
          stroke: "blue", // Blue color for the wire
          strokeWidth: 2, // Width of the line
          strokeDasharray: "6,4", // Dash pattern
          animation: "dash 1s linear infinite", // Animation for the dashes
          ...style,
        }}
      />
      {/* Delete Button */}
      <circle
        cx={midX}
        cy={midY}
        r={2} // Radius for the delete button
        fill="red"
        stroke="red"
        strokeWidth={10} // Increases clickable area
        onClick={handleDeleteEdge}
        style={{ cursor: "pointer" }}
      />
      <style>
        {`
          @keyframes dash {
            to {
              stroke-dashoffset: -20;
            }
          }
        `}
      </style>
    </g>
  );
};

export default WireEdge;
