import { ReactFlowProvider } from "@xyflow/react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { useState } from "react";
import Flow from "./Flow";
import About from "./About";
import Sidebar from "./Sidebar";
import Palette from "./Palette";

import { DnDProvider } from "./context/DnDContext";
import { PaletteProvider } from "./context/PaletteContext";
import { CanvasProvider } from "./context/CanvasContext";
import { AppProvider } from "./context/AppContext"; //useAppContext



//--- ONLY UNCOMMENT ONE OF THESE (depending on which backend server you're running.).... ---//
//USE THIS LOCAL ONE for local development...
//const backend_url = "http://localhost:3000"; // URL of the LOCAL backend server (use this if you're running server.js in a separate window!)

// TURN THIS ONLINE ONE back on before you run "npm build" and deploy to Vercel!
const backend_url = "https://snailbunny.site"; // URL of the backend server hosted online! 
//const backend_url = "http://104.200.25.53/"; //IP address of backend server hosted online, probably don't use this one.

function AppContent() {
  //const { backend, loadCanvasRequest, setLoadCanvasRequest, userID, handleUserLogin } = useAppContext();

  /*--- should the sidebar be showing? ----*/
  const [showSidebar, setShowSidebar] = useState(false);
  const toggleSidebar = () => {
    setShowSidebar((prev) => !prev);
  };



  return (
    <CanvasProvider>
      <PaletteProvider>
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

          {/* Main Content */}
          <ReactFlowProvider>
            <DnDProvider>
              <Routes>
                <Route path="/about" element={<About />} />
                <Route
                  path="/"
                  element={
                    <div className="flex">
                      <div
                        className="border border-gray-500"
                        style={{
                          height: "90vh",
                          width: "80vw",
                          margin: 0,
                          padding: 0,
                        }}
                      >
                        {/*<!-- maybe move the toolbar here? -->*/}

                        <Flow />

                      </div>
                      {/*SET PALETTE WIDTH HERE */}
                        <div 
                        className="border border-gray-500 ml-5 rounded-lg"
                        style={{
                          height: "90vh",
                          width: "20vw",
                          margin: 0,
                          zIndex: 3, 
                        }}
                        >
                        <Palette />
                        </div>
                    </div>
                  }
                />
              </Routes>
            </DnDProvider>
          </ReactFlowProvider>
        </div>
      </PaletteProvider>
    </CanvasProvider>
  );
}


export default function App() {
  return (
    <Router>
      <AppProvider backend={backend_url}>
        <AppContent />
      </AppProvider>
    </Router>
  );
}
