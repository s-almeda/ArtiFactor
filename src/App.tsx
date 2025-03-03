import { ReactFlowProvider } from "@xyflow/react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { useState } from "react";
import Flow from "./pages/Flow";
import About from "./pages/About";
import Admin from "./pages/Admin";
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
  //const { backend, pullCanvasRequest, setpullCanvasRequest, userID, handleUserLogin } = useAppContext();

  /*--- should the sidebar be showing? ----*/
  const [showSidebar, setShowSidebar] = useState(false);
  const toggleSidebar = () => {
    setShowSidebar((prev) => !prev);
  };

  return (
    <AppProvider backend={backend_url}>
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
            />
          </div>
          {/* Main Content */}
          <ReactFlowProvider>
            <DnDProvider>
              <Routes>
                <Route path="/about" element={
                  <div className="flex bottom-0 bg-red" style={{height: "calc(100vh - 100px)", marginTop: "50px", padding: "10px", overflow: "scroll", background: "red" }}>
                  <About /> 
                  </div>
                }/>
                <Route path="/admin" element={<Admin />} />
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
    </AppProvider>
  );
}


export default function App() {
  return (
    <Router>
        <AppContent />
    </Router>
  );
}

