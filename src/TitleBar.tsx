import { useEffect, useState } from "react";
import { Pencil, Menu } from "lucide-react";

interface TitleBarProps {
  canvasName: string;
  onCanvasNameChange: (newName: string) => void;
  lastSaved: string;
  toggleSidebar: () => void;  // Pass toggleSidebar function from App.tsx
}

const TitleBar = ({ canvasName, onCanvasNameChange, lastSaved, toggleSidebar }: TitleBarProps) => {
  const [isEditing, setIsEditing] = useState(false);
  const [newName, setNewName] = useState(canvasName);

  useEffect(() => {
    setNewName(canvasName);
  }, [canvasName]);

  const handleBlur = () => {
    setIsEditing(false);
    if (newName.trim() !== "" && newName !== canvasName) {
      onCanvasNameChange(newName);
    } else {
      setNewName(canvasName);
    }
  };

  return (
    <div className="w-full flex items-center justify-between bg-gray-900 text-white px-4 py-2 shadow-md">
      {/* Left: Hamburger Menu (Opens Sidebar) */}
      <button onClick={toggleSidebar} className="p-2">
        <Menu size={24} />
      </button>

      {/* Center: App Title */}
      <h1 className="text-lg font-semibold text-blue-400">ArtiFactor</h1>

      {/* Right: Canvas Name & Last Saved */}
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
        <span className="text-sm text-gray-400">Last saved: {lastSaved}</span>
      </div>
    </div>
  );
};

export default TitleBar;
