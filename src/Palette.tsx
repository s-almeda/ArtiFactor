import type React from "react";
import { usePaletteContext } from "./context/PaletteContext";
import PaletteNode from "./nodes/PaletteNode";

const charLimit = 25;

const Palette: React.FC = () => {
  const { activeTab, setActiveTab, clippedNodes = [] } = usePaletteContext() as { activeTab: "text" | "image"; setActiveTab: (tab: "text" | "image") => void; clippedNodes: any[] };

  const filteredNodes = clippedNodes.filter((node) => node.type === activeTab);

  return (
    <div className="p-4 w-[23vw]">
      {/* Toggle Tabs */}
      <div className="flex space-x-2 mb-4">
        {["text", "image"].map((tab) => (
          <button
            key={tab}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === tab ? "bg-gray-800 text-white" : "bg-gray-200 text-gray-600"
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
          <PaletteNode key={index} data={data} charLimit={charLimit} type={activeTab as "text" | "image"} />
        ))}
      </div>
    </div>
  );
};

export default Palette;
