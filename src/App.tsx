import { ReactFlowProvider } from "@xyflow/react";

import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { useState } from "react";
import Flow from "./Flow";
import About from "./About";
import Sidebar from "./Sidebar";
import Palette from "./Palette";

import { DnDProvider } from "./DnDContext";
import { NodeProvider } from "./PaletteContext";


function AppContent() {
  /*--- SIDEBAR ----*/
  const [showSidebar, setShowSidebar] = useState(false);
  const toggleSidebar = () => {
    //console.log("Toggling sidebar");
    setShowSidebar((prev) => !prev);
  };

  /*--- called by Palette or by the Lookup window to add a new node to the Flow canvas --*/
  const handleNewNode = (type: string, content: string) => {
    console.log(`Adding new node with type: ${type} and content: ${content}`);
    // Add new node to the Flow canvas
  };

  return (
    <div className="relative h-screen w-screen">
      {/* Sidebar */}
      <div
        className={`fixed top-0 left-0 h-full bg-gray-800 text-white w-64 transition-transform duration-300 ${
          showSidebar ? "translate-x-0" : "-translate-x-64"
        }`}
        style={{ zIndex: 50 }}
      >
        <Sidebar onClose={toggleSidebar} />
      </div>

      {/* Sidebar Toggle Button */}
      <button
        type="button"
        onClick={toggleSidebar}
        className="absolute top-4 left-64 z-50 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-3 rounded-r focus:outline-none transition-all duration-300"
        style={{
          transform: showSidebar ? "translateX(0)" : "translateX(-256px)", // Move button along with sidebar
          transition: "transform 0.3s ease",
        }}
      >
        {showSidebar ? "⬅️" : "➡️"}
        </button>
      {/* <div className="absolute top-0 left-0 z-50 p-4">
        <Lookup artworks={[
          {
        title: "Impression, Sunrise",
        year: 1872,
        artist: "Claude Monet",
        genre: "Landscape",
        style: "Impressionism",
        description: "This painting is a depiction of the port of Le Havre at sunrise, with small rowboats in the foreground and ships and cranes in the background. The orange sun is shown as a distinct circle, reflecting on the water below. This piece gave the Impressionist movement its name when critics seized upon the title of this painting to give the entire movement a derisive moniker.",
        image: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Monet_-_Impression%2C_Sunrise.jpg/600px-Monet_-_Impression%2C_Sunrise.jpg"
          },
          {
        title: "Morning on the Seine",
        year: 1897,
        artist: "Claude Monet",
        genre: "Landscape",
        style: "Impressionism",
        description: "Part of a series of paintings depicting the Seine River, this work showcases Monet's mastery of light and atmosphere. The artist painted the same scene at different times of day to capture varying effects of light and weather conditions.",
        image: "https://www.claude-monet.com/assets/img/paintings/morning-on-the-seine-near-giverny.jpg"
          },
          {
        title: "Water Lilies",
        year: 1919,
        artist: "Claude Monet",
        genre: "Landscape",
        style: "Impressionism",
        description: "This painting is part of Monet's famous Water Lilies series, which he painted in his garden at Giverny. The series captures the beauty and tranquility of the water garden, with its reflections and play of light.",
        image: "https://upload.wikimedia.org/wikipedia/commons/9/9e/WLA_metmuseum_Water_Lilies_by_Claude_Monet.jpg"
          }
        ]} />
      </div> */}

      

      {/* Main Content */}
      
      <ReactFlowProvider>
        <NodeProvider>
          <DnDProvider>
            <Routes>
              <Route path="/" element={<About />} />
              <Route
                path="/canvas"
                element={
                  <div className="flex">
                    <div
                      className="border border-gray-500"
                      style={{
                        height: "90vh",
                        width: "70vw",
                        margin: 0,
                        padding: 0,
                      }}
                    >
                      <Flow />
                    </div>
                    <div
                      className="border border-gray-500"
                      style={{
                        height: "90vh",
                        width: "30vw",
                        margin: 0,
                        padding: 0,
                      }}
                    >
                      <Palette
                        onAddNode={(type, content) =>
                          handleNewNode(type, content)
                        }
                      />
                    </div>
                  </div>
                }
              />
            </Routes>
          </DnDProvider>
        </NodeProvider>
      </ReactFlowProvider>
        </div>
      );
    }

export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}
