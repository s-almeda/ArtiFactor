import React from "react";
import { usePaletteContext } from "./context/PaletteContext";
import PaletteNode from "./nodes/PaletteNode";
import { useEffect } from "react";


const Palette: React.FC = () => {
  const {
    removeNode,
    activeTab,
    setActiveTab,
    clippedNodes = [],
  } = usePaletteContext() as {
    removeNode: (id: number) => void;
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

  const handleRemoveNode = (id: number) => {
    removeNode(id);
  };

  const filteredNodes = clippedNodes.filter((node) => node.type === activeTab);

  return (
    <div className="p-4 overflow-y-auto rounded-md">
      {/* Toggle Tabs */}
      <div className="flex space-x-2 mb-4"></div>
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

      {/* Tab Content */}
      <div className="space-y-3">
        {filteredNodes.map((data, index) => (
          <div key={index} className="relative">
            <PaletteNode
              data={data}
              type={activeTab as "text" | "image"}
              removeNode={() => handleRemoveNode(data.id)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default Palette;
