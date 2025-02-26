import { ReactFlowProvider } from "@xyflow/react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { useState, useEffect } from "react";
import Flow from "./Flow";
import About from "./About";
import Sidebar from "./Sidebar";
import Palette from "./Palette";

import { DnDProvider } from "./context/DnDContext";
import { PaletteProvider } from "./context/PaletteContext";
import { CanvasProvider } from "./context/CanvasContext";
import { AppProvider } from "./context/AppContext"; //useAppContext

import TitleBar from "./TitleBar";
import { NodeProvider } from "./context/NodeContext";


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

  const [lastSaved, _] = useState("Never"); //setLastSaved
  const [canvasName, setCanvasName] = useState("Untitled");
  const [canvasID, __] = useState("null");
  // load canvas when the app starts
  useEffect(() => {
    const loadCanvas = async () => {
      console.log("todo")
      // try {
      //   const response = await fetch(`${backend_url}/api/get-latest-canvas`);
      //   const data = await response.json();
        
      //   if (data.success && data.canvas) {
      //     setCanvasID(data.canvas.id); // ensure correct ID
      //     setCanvasName(data.canvas.name);
      //   } else {
      //     console.warn("No existing canvas found. Creating a new one.");
      //     await createNewCanvas();
      //   }
      // } catch (error) {
      //   console.error("Error loading canvas:", error);
      // }
    };
    loadCanvas();
  }, []);
    // create new canvas if none exists
    // const createNewCanvas = async () => {
    //   try {
    //     const response = await fetch(`${backend_url}/api/create-canvas`, {
    //       method: "POST",
    //       headers: { "Content-Type": "application/json" },
    //     });
  
    //     const data = await response.json();
    //     if (data.success) {
    //       setCanvasID(data.canvas.id);
    //       setCanvasName(data.canvas.name);
    //     } else {
    //       console.error("Error creating a new canvas:", data.message);
    //     }
    //   } catch (error) {
    //     console.error("Error creating canvas:", error);
    //   }
    // };
  // update last saved time
  // const updateLastSaved = () => {
  //   const now = new Date();
  //   setLastSaved(now.toLocaleString());
  // };
  // update canvas name
  const updateCanvasName = async (newName: string) => {
    if (!newName.trim() || newName === canvasName) return; // ignore empty names
    setCanvasName(newName);

    try {
      const response = await fetch(`${backend_url}/api/update-canvas-name`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canvasID, newCanvasName: newName }), // ensure canvasID sent
      });
      if (!response.ok) {
        throw new Error("Failed to update canvas name.");
      }
      console.log(`Canvas name updated to "${newName}" in the backend.`);
    } catch (error) {
      console.error("Error updating canvas name:", error);
    }
  };

  return (
    <NodeProvider>
    <CanvasProvider>
      <PaletteProvider>
        <div className="relative h-screen w-screen overflow-hidden">
          {/* Sidebar */}
          <div
            className={`fixed top-0 left-0 h-full bg-gray-800 text-white w-64 transition-transform duration-300 ${
              showSidebar ? "translate-x-0" : "-translate-x-64"
            }`}
            style={{ zIndex: 50, overflow: "hidden" }}
          >
            <Sidebar onClose={toggleSidebar} />
          </div>
          <div
            className={`fixed top-0 left-0 w-full h-12 bg-gray-800 text-white transition-transform duration-300`}
            style={{ overflow: "hidden" }}
          >
            <TitleBar
              toggleSidebar={toggleSidebar}
              canvasName={canvasName}
              onCanvasNameChange={updateCanvasName}
              lastSaved={lastSaved}
            />
          </div>
          {/* Main Content */}
          <ReactFlowProvider>
            <DnDProvider>
              <Routes>
                <Route path="/about" element={<About />} />
                <Route
                  path="/"
                  element={
                    <div className="flex fixed" style={{height: "calc(100vh - 80px)", marginTop: "24px", overflow: "hidden" }}>
                      <div
                        className="border border-gray-500"
                        style={{
                          height: '100%',
                          width: "75vw",
                          margin: 0,
                          padding: 0,
                          overflow: "hidden",
                        }}
                      >
                        {/*<!-- maybe move the toolbar here? -->*/}
                        <Flow />
                      </div>
                      {/*SET PALETTE WIDTH HERE */}
                      <div
                        className="border border-gray-500 ml-5 rounded-lg"
                        style={{
                          height: "100%",
                          width: "21vw",
                          margin: 0,
                          marginRight: 10,
                          zIndex: 3,
                          overflow: "hidden",
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
    </NodeProvider>
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

