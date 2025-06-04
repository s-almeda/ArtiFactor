import { useEffect, useState } from "react";
import { Pencil, Menu, Save } from "lucide-react"; // Import Save icon
import { useCanvasContext } from "./context/CanvasContext";
import { useNodeContext } from "./context/NodeContext";
import { useAppContext } from "./context/AppContext";

interface TitleBarProps {
  toggleSidebar: () => void;
}

const TitleBar = ({ toggleSidebar }: TitleBarProps) => {
  const { canvasID, canvasName, setCanvasName, saveCanvas, lastSaved } = useCanvasContext();
  const { loginStatus, condition } = useAppContext();
  const { canvasToObject } = useNodeContext();
  
  const [isEditing, setIsEditing] = useState(false);
  const [newName, setNewName] = useState(canvasName);
  const [displayDate, setDisplayDate] = useState("Never");
  const [isSaving, setIsSaving] = useState(false);

  const handleBlur = async () => {
    setIsEditing(false);
    if (newName.trim() !== "" && newName !== canvasName) {
      setCanvasName(newName);
    } else {
      setNewName(canvasName);
    }
    handleSaveCanvas();
  };

  const handleSaveCanvas = async () => {
    if (!canvasID) return;
    
    setIsSaving(true);
    console.log("Title bar is saving the canvas.");
    try {
      await saveCanvas(canvasToObject(), canvasID, canvasName);
    } catch (error) {
      console.error("Error saving canvas:", error);
    }
    setIsSaving(false);
  };

  useEffect(() => {
    if (loginStatus !== "logged in") return;
    setNewName(canvasName);
  }, [canvasName]);
  
  useEffect(() => {
    if (lastSaved) {
      const now = new Date();
      const savedTime = new Date(lastSaved);
      const diffInSeconds = Math.floor((now.getTime() - savedTime.getTime()) / 1000);

      if (diffInSeconds <= 3) {
        setDisplayDate("just now");
      } else {
        const minutes = Math.floor(diffInSeconds / 60);
        const seconds = diffInSeconds % 60;

        setDisplayDate(
          minutes > 0
            ? `${minutes} minute${minutes > 1 ? "s" : ""} and ${seconds} second${seconds !== 1 ? "s" : ""} ago`
            : `${seconds} second${seconds !== 1 ? "s" : ""} ago`
        );
      }
    } else {
      setDisplayDate("Never");
    }
  }, [lastSaved]);

  return (
    <div className="w-full flex items-center justify-between bg-stone-900 text-white px-4 py-2 shadow-md">
      {/* Left: Sidebar Toggle */}
      {condition && loginStatus === "logged in" && (
        <button onClick={toggleSidebar} className="p-2">
          <Menu size={16} />
        </button>
      )}

      {/* Center: Canvas Name & Last Saved (Flex Centered) */}
      {loginStatus !== "logged in" ? (
        <h1 className="text-lg font-semibold text-stone-400">ArtiFactor</h1>
      ) : (
        <div className="flex-1 flex items-center justify-center gap-4">
          <div className="flex items-center">
            {isEditing ? (
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onBlur={handleBlur}
                onKeyDown={(e) => e.key === "Enter" && handleBlur()}
                autoFocus
                className="bg-stone-800 text-white px-2 py-1 border border-gray-700 rounded"
              />
            ) : (
              <span className="text-white cursor-pointer" onClick={() => setIsEditing(true)}>
                {canvasName}
              </span>
            )}
            <Pencil size={16} className="ml-2 cursor-pointer" onClick={() => setIsEditing(true)} />
          </div>
          <span className="text-sm text-stone-400">Last saved... {displayDate}</span>
        </div>
      )}

      {loginStatus === "logged in" && (
        <button
          onClick={handleSaveCanvas}
          className={`flex items-center gap-2 px-4 py-2 rounded ${
            isSaving ? "bg-gray-600" : "bg-stone-500"
          } text-white`}
        >

          {isSaving ? "..." : <Save size={16} />}
        </button>
      )}
    </div>
  );
};

export default TitleBar;
