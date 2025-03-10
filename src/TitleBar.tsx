import { useEffect, useState } from "react";
import { Pencil, Menu } from "lucide-react";
import { useCanvasContext } from "./context/CanvasContext"; // Adjust the import path as needed
import { useNodeContext } from "./context/NodeContext"; // Adjust the import path as needed
import { useAppContext } from "./context/AppContext";

interface TitleBarProps {
  toggleSidebar: () => void;  // Pass toggleSidebar function from App.tsx
}

const TitleBar = ({ toggleSidebar }: TitleBarProps) => {
  const { canvasID, canvasName, setCanvasName, saveCanvas, lastSaved } = useCanvasContext();
  const { loginStatus } = useAppContext();
  const { canvasToObject } = useNodeContext();
  const [isEditing, setIsEditing] = useState(false);
  const [newName, setNewName] = useState(canvasName);
  const [displayDate, setDisplayDate] = useState("Never");

  const handleBlur = async () => {
    setIsEditing(false);
    if (newName.trim() !== "" && newName !== canvasName) {
      setCanvasName(newName);
    } else {
      setNewName(canvasName);
    }
    const canvasObject = canvasToObject();
    console.log("title bar is saving the canvas.")
    saveCanvas(canvasObject, canvasID, canvasName);
  };

  useEffect(() => { //if the canvas name has changed in the context, update the display name and save the canvas
    if (loginStatus != "logged in"){
      return
    }
    setNewName(canvasName);
  }, [canvasName]);

  useEffect(() => {
    setDisplayDate(
      lastSaved
        ? new Date(lastSaved).toLocaleString("en-US", {
            timeZone: "PST",
            dateStyle: "short",
            timeStyle: "medium",
          })
        : "Never"
    );
  }, [lastSaved]);



  return (
    <div className="w-full flex items-center justify-between bg-gray-900 text-white px-4 py-2 shadow-md">
      {/* Left: Hamburger Menu (Opens Sidebar) */}
      <button onClick={toggleSidebar} className="p-2">
        <Menu size={24} />
      </button>

      {/* Center: App Title */}
      <h1 className="text-lg font-semibold text-blue-400">ArtiFactor</h1>

      {loginStatus === "logged in" && (
        <div className="flex items-center gap-4">
          <div className="flex items-center">
        {isEditing ? (
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onBlur={handleBlur}
            onKeyDown={(e) => e.key === "Enter" && handleBlur()}
            autoFocus
            className="bg-gray-800 text-white px-2 py-1 border border-gray-700 rounded"
          />
        ) : (
          <span className="text-white cursor-pointer" onClick={() => setIsEditing(true)}>
            {canvasName}
          </span>
        )}
        <Pencil size={16} className="ml-2 cursor-pointer" onClick={() => setIsEditing(true)} />
          </div>
          {/* Last Saved Timestamp */}
          <span className="text-sm text-gray-400">
        Last saved: {displayDate}
          </span>
        </div>
      )}
      </div>
  );
};

export default TitleBar;
