import React, { useState } from "react";
import { useDnD } from "./DnDContext";

const charLimit = 25;

interface PaletteProps {
  onAddNode: (type: string, content: string) => void;
}

interface PaletteNodeProps {
  content: string;
  charLimit: number;
  type: string;
  onAddNode: (type: string, content: string) => void;
}

const PaletteNode: React.FC<PaletteNodeProps> = ({
  type = "text",
  content,
  charLimit,
  onAddNode,
}) => {
  const [_, setDraggableType, __, setDraggableContent] = useDnD();

  const onDragStart = (
    event: { dataTransfer: { effectAllowed: string } },
    type: string,
    content: string
  ) => {
    event.dataTransfer.effectAllowed = "move";
    setDraggableType(type);
    setDraggableContent(content);
  };

  const handleNodeClick = (content: string) => {
    onAddNode(type, content);
  };

  return (
    <div
      className="bg-white border border-gray-600 rounded-md p-2 cursor-grab hover:bg-gray-50 hover:shadow-sm transition-all text-sm"
      draggable
      onDragStart={(event) => onDragStart(event, type, content)}
      onClick={() => handleNodeClick(content)}
      onKeyDown={(event) => {
        if (event.key === "Enter") handleNodeClick(content);
      }}
      tabIndex={0}
      style={{ width: "21vw" }}
    >
      {type === "text" ? (
        content.length > charLimit ? (
          `${content.substring(0, charLimit)}...`
        ) : (
          content
        )
      ) : type === "image" ? (
        <img
          src={content}
          alt="Saved Image"
          className="w-full h-auto rounded-md"
        />
      ) : (
        ""
      )}
    </div>
  );
};

const Palette: React.FC<PaletteProps> = ({ onAddNode }) => {
  const [activeTab, setActiveTab] = useState<"text" | "image">("text");

  // Example data for text & images
  const textPrompts = [
    "claude monet was an amazing guy and he was soooo cool",
    "van gogh",
    "picasso",
    "dali",
    "michelangelo",
    "raphael",
    "leonardo",
  ];

  const savedImages = [
    "https://via.placeholder.com/150", // Replace with actual image URLs
    "https://via.placeholder.com/160",
    "https://via.placeholder.com/170",
  ];

  return (
    <div className="bg-white p-4 w-[23vw]">
      {/* Toggle Tabs */}
      <div className="flex space-x-2 mb-4">
        <button
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
            activeTab === "text"
              ? "bg-gray-800 text-white"
              : "bg-gray-200 text-gray-600"
          }`}
          onClick={() => setActiveTab("text")}
          type="button"
        >
          Text
        </button>
        <button
          type="button"
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
            activeTab === "image"
              ? "bg-gray-800 text-white"
              : "bg-gray-200 text-gray-600"
          }`}
          onClick={() => setActiveTab("image")}
        >
          Images
        </button>
      </div>

      {/* Tab Content */}
      <div className="space-y-3">
        {activeTab === "text"
          ? textPrompts.map((item, i) => (
              <PaletteNode
                key={`text_${i}`}
                type="text"
                content={item}
                charLimit={charLimit}
                onAddNode={onAddNode}
              />
            ))
          : savedImages.map((image, i) => (
              <PaletteNode
                key={`image_${i}`}
                type="image"
                content={image}
                charLimit={charLimit}
                onAddNode={onAddNode}
              />
            ))}
      </div>
    </div>
  );
};

export default Palette;
