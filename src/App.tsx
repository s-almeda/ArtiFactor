import { ReactFlowProvider } from "@xyflow/react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { useState, useEffect } from "react";
import Flow from "./Flow";
import About from "./About";
import Sidebar from "./Sidebar";
import Palette from "./Palette";

import { DnDProvider } from "./contexts/DnDContext";
import { PaletteProvider } from "./contexts/PaletteContext";
import { CanvasProvider } from "./contexts/CanvasContext";

//--- ONLY UNCOMMENT ONE OF THESE (depending on which backend server you're running.).... ---//
//USE THIS LOCAL ONE for local development...
const backend_url = "http://localhost:3000"; // URL of the LOCAL backend server (use this if you're running server.js in a separate window!)

// TURN THIS ONLINE ONE back on before you run "npm build" and deploy to Vercel!
//const backend_url = "https://snailbunny.site"; // URL of the backend server hosted online! 
//const backend_url = "http://104.200.25.53/"; //IP address of backend server hosted online, probably don't use this one.

function AppContent() {
  /*--- SIDEBAR ----*/
  const [showSidebar, setShowSidebar] = useState(false);
  const toggleSidebar = () => {
    setShowSidebar((prev) => !prev);
  };

  /*--- USER AUTH ----*/
  const [userID, setUserID] = useState<string | null>("default");
  const handleUserLogin = (enteredUserID: string) => {
    setUserID(enteredUserID);
    console.log(`User logged in: ${enteredUserID}`);
  };

  /*--- called by Palette or Lookup window to add a new node to the Flow canvas --*/
  const handleNewNode = (type: string, content: string) => {
    console.log(`Adding new node with type: ${type} and content: ${content}`);
  };

  //allows any of its children to tell the App that it's time to reload the whole canvas! 
  const [loadCanvasRequest, setLoadCanvasRequest] = useState(false); // 

  useEffect(() => {
    console.log(`Initializing App with backend: ${backend_url} for userID: ${userID}`);
  }, [backend_url, userID]);


  return (
    <CanvasProvider userID={userID ?? ''} backend={backend_url}>
      <PaletteProvider userID={userID ?? ''}>
        <div className="relative h-screen w-screen">
          {/* Sidebar */}
            <div
            className={`fixed top-0 left-0 h-full bg-gray-800 text-white w-64 transition-transform duration-300 ${
              showSidebar ? "translate-x-0" : "-translate-x-64"
            }`}
            style={{ zIndex: 50 }}
            >
            <Sidebar 
              onClose={toggleSidebar} 
              onUserLogin={handleUserLogin} 
              backend={backend_url} 
              setLoadCanvasRequest={setLoadCanvasRequest} 
            />
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
                    width: "70vw",
                    margin: 0,
                    padding: 0,
                  }}
                  >
                  <Flow 
                    userID={userID ?? ''}
                    backend={backend_url} 
                    loadCanvasRequest={loadCanvasRequest} 
                    setLoadCanvasRequest={setLoadCanvasRequest} 
                  />
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
                  <Palette onAddNode={handleNewNode} />
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
      <AppContent />
    </Router>
  );
}
