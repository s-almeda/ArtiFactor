import React from "react";
import { usePaletteContext } from "./context/PaletteContext";
import PaletteNode from "./nodes/PaletteNode";
import { useEffect } from "react";

//TODO: palette nodes need to store their parentNodeId, and broadcast it when they are dragged and dropped
const charLimit = 25;

const Palette: React.FC = () => {
  const {
    activeTab,
    setActiveTab,
    clippedNodes = [],
    setClippedNodes, // Add setClippedNodes from context
  } = usePaletteContext() as {
    activeTab: "text" | "image";
    setActiveTab: (tab: "text" | "image") => void;
    clippedNodes: any[];
    setClippedNodes: (nodes: any[]) => void; // Ensure this function is in context
  };

  // Ensure there's always an active tab when new nodes are added
  useEffect(() => {
    if (clippedNodes.length > 0) {
      setActiveTab(clippedNodes[clippedNodes.length - 1].type); // Set tab to the latest clipped node type
    }
  }, [clippedNodes, setActiveTab]);

  const removeNode = (index: number) => {
    setClippedNodes(clippedNodes.filter((_, i) => i !== index)); // Update state
  };

  const filteredNodes = clippedNodes.filter((node) => node.type === activeTab);

  return (
    <div className="p-4 ">
      {/* Toggle Tabs */}
      <div className="flex space-x-2 mb-4">
        {["text", "image"].map((tab) => (
          <button
            key={tab}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === tab
                ? "bg-gray-800 text-white"
                : "bg-gray-200 text-gray-600"
            }`}
            onClick={() => setActiveTab(tab as "image" | "text")}
            type="button"
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="space-y-3">
        {filteredNodes.map((data, index) => (
          <div key={index} className="relative">
            <PaletteNode
              data={data}
              charLimit={charLimit}
              type={activeTab as "text" | "image"}
              removeNode={() => removeNode(index)} // Pass correct function
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default Palette;
