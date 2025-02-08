import React from "react"; // {useState}
import { useDnD } from "./context/DnDContext";
import { usePaletteContext, NodeData } from "./context/PaletteContext";

const charLimit = 25;


interface PaletteProps {
  // onAddNode: (type: string, content: string) => void;
}

interface PaletteNodeProps {
  data: NodeData;
  charLimit: number;
  type: string;
  //onAddNode: (type: string, content: string) => void;
}


const PaletteNode: React.FC<PaletteNodeProps> = ({
  data,
  charLimit,
  type,
}) => {
  const [_, setDraggableType, __, setDraggableData] = useDnD();

  const onDragStart = (
    event: { dataTransfer: { effectAllowed: string } },
  ) => {
    event.dataTransfer.effectAllowed = "move";
    setDraggableType(type);
    setDraggableData(data);
  };

  return (
    <div
      className="bg-white border border-gray-600 rounded-md p-2 cursor-grab hover:bg-gray-50 hover:shadow-sm transition-all text-sm"
      draggable
      onDragStart={(event) => onDragStart(event)}
      tabIndex={0}
      style={{ width: "21vw" }}
    >
      {type === "text" ? (
        data.content.length > charLimit ? (
          `${data.content.substring(0, charLimit)}...`
        ) : (
          data.content
        )
      ) : type === "image" ? (
        <div style={{ position: "relative", maxHeight: "60px", overflow: "hidden" }}>
          <img
            src={data.content}
            alt={data.prompt}
            className="rounded-md object-cover"
            style={{ width: "auto", height: "100%", transform: "translateY(-30%)", position: "relative" }}
          />
            <div
              className="absolute bottom-0 left-0 w-full h-full bg-white bg-opacity-50 text-gray-800 text-md uppercase text-center p-1 opacity-0 transition-opacity duration-300 hover:opacity-100 font-bold italic leading-tight"
            >
            {data.prompt.length > 60 ? `${data.prompt.substring(0, 60)}...` : data.prompt}
          </div>
        </div>
      ) : (
        ""
      )}
    </div>
  );
};

const Palette: React.FC<PaletteProps> = ({  }) => {//onAddNode
  const { activeTab, setActiveTab } = usePaletteContext();
  const { clippedNodes = [] } = usePaletteContext();

  const clippedTextNodes = clippedNodes.filter(node => node.type === "text");

  const clippedImageNodes = clippedNodes.filter(node => node.type === "image");


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
            activeTab === "images"
              ? "bg-gray-800 text-white"
              : "bg-gray-200 text-gray-600"
          }`}
          onClick={() => setActiveTab("images")}
        >
          Images
        </button>
      </div>

      {/* Tab Content */}
      <div className="space-y-3">
        {activeTab === "text"
          ? clippedTextNodes.map((data, index) => (
              <PaletteNode
          key={index}
          data={data}
          charLimit={charLimit}
          type="text"
              />
            ))
          : clippedImageNodes.map((data, index) => (
              <PaletteNode
          key={index}
          data={data}
          charLimit={charLimit}
          type="image"
              />
            ))}
      </div>
    </div>
  );
};

export default Palette;
