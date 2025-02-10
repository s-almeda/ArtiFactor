import React, { useState } from "react";
import { useDnD } from "../context/DnDContext";
import type { NodeData } from "../context/PaletteContext";

interface PaletteNodeProps {
  data: NodeData;
  charLimit: number;
  type: "text" | "image";
}

const PaletteNode: React.FC<PaletteNodeProps> = ({ data, charLimit, type }) => {
  const [___, setIsHovered] = useState(false);
  const [_, setDraggableType, __, setDraggableData] = useDnD();

  const onDragStart = (event: React.DragEvent<HTMLDivElement>) => {
    event.dataTransfer.effectAllowed = "move";
    setDraggableType(type);
    setDraggableData(data);
  };

  // Download image as a .png file (Fixed)
  const downloadImage = () => {
    const link = document.createElement("a");
    link.href = data.content; // Directly use the image URL
    link.download = "palette-image.png"; // Suggested filename
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div
      className="border border-gray-600 rounded-md p-2 cursor-grab hover:bg-gray-50 hover:shadow-sm transition-all text-sm relative"
      draggable
      onDragStart={onDragStart}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{ width: "21vw" }}
    >
      {type === "text" ? (
        <div>
          {data.content.length > charLimit
            ? `${data.content.substring(0, charLimit)}...`
            : data.content}
        </div>
      ) : (
        <div className="relative max-h-[60px] overflow-hidden">
          <img
            src={data.content}
            alt={data.prompt}
            className="rounded-md object-cover w-full h-full"
          />
          <div
            className="absolute bottom-0 left-0 w-full h-full bg-white bg-opacity-50 
             text-gray-800 text-md uppercase text-center p-1 opacity-0 
             transition-opacity duration-300 hover:opacity-100 font-bold italic 
             leading-tight flex flex-col items-center justify-center"
            style={{ zIndex: 10 }} // Ensure the button is above everything
          >
            <span>
              {data.prompt.length > 60
                ? `${data.prompt.substring(0, 60)}...`
                : data.prompt}
            </span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation(); // Prevent parent div from interfering
                downloadImage();
              }}
              className="bg-gray-800 text-white px-2 py-1 rounded-md text-sm mt-2"
            >
              Save Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PaletteNode;
