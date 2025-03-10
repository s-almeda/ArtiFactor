import { useState, useEffect } from "react";
import { useCanvasContext } from "./context/CanvasContext";
import { useAppContext } from "./context/AppContext";
import { useNodeContext } from "./context/NodeContext";
import { motion } from "framer-motion";



// TODO: add createCanvas, which takes us to a fresh new canvas by gettingNextCanvas from the backend and switching the url
const Sidebar = ({ onClose }: { 
  onClose: () => void; 
}) => {
  const { canvasID, canvasName, saveCanvas, deleteCanvas, createNewCanvas } = useCanvasContext(); // canvasName
  const { backend, handleUserLogin, userID, admins, loginStatus } = useAppContext();
  const { canvasToObject } = useNodeContext();

  const [enteredUserID, setEnteredUserID] = useState("");
  // const [enteredPassword, setEnteredPassword] = useState("");

  const [error, setError] = useState("");
  const [canvasList, setCanvasList] = useState<{ id: string; name: string }[]>([]); // Initialize as an empty array
  const [editingCanvasList, setEditingCanvasList] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => { // if any of these things change, refresh the list of canvases
      refreshCanvases();
  }, [userID, canvasName, canvasID]);

  const handleSaveCanvas = async () => {
    setIsSaving(true);
    if (!canvasID || !userID) {
      console.error("No canvas to save.");
      return;
    }

    try {
      console.log("sidebar is calling save canvas")
      await saveCanvas(canvasToObject(), canvasID, canvasName);
    } catch (error) {
      console.error("Error saving canvas:", error);
    }
    refreshCanvases();
    setIsSaving(false);
  };

  const handleCreateCanvas = async () => {
    if (!userID) {
      console.error("You have to log in before you can create a new canvas.");
      return;
    }
    else{
      try {
        createNewCanvas(userID);
      } catch (error) {
        console.error("Sidebar failed to create new canvas:", error);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    try {
      await handleUserLogin(enteredUserID, "");

    } catch (error) {
      console.error("Error logging in user:", error);
      setError("Something went wrong.");
    }
  };
  const handleUserLogout = () => {
    const confirmLogout = window.confirm("Are you sure you want to log out? (You may lose unsaved changes)");
    if (confirmLogout) {
      window.location.href = "/";
      localStorage.removeItem("last_userID");
      localStorage.removeItem("last_password");
      //remove from storage to stop from quick-logging-back-in
      handleUserLogin("", "");
    }
  };
  

  const refreshCanvases = async () => {
    if (!userID || userID === "default") return; // user isn't logged in

    try {
      console.log("Refreshing canvases...");
      const response = await fetch(`${backend}/api/list-canvases/${userID}`);
      const data = await response.json();

      if (!data.success) {
        setError("Error fetching canvases.");
        return;
      }

      //console.log("user's canvasList", data.canvases);
      setCanvasList(data.canvases);
    } catch (error) {
      console.error("Error refreshing canvasList:", error);
      setError("Something went wrong.");
    }
  };
  const handleDeleteCanvas = async (canvasIDToDelete: string) => {
    if (canvasIDToDelete === canvasID) {
      alert("You can't delete the canvas you're currently working on!");
      return;
    }

    const confirmDelete = window.confirm("Are you sure you want to delete this canvas?");
    if (confirmDelete) {
      await deleteCanvas(canvasIDToDelete);
      refreshCanvases();
    }
  };

  // useEffect(() => {
  //   if (userID) {
  //     setLoggedInUser(userID);
  //   }
  // }, [userID]);


  return (
    <div className="fixed top-0 left-0 w-64 h-full bg-gray-800 text-white p-4 z-40">
      <button
        onClick={onClose}
        className="absolute top-2 right-2 bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-2 rounded"
      >
        Close
      </button>

      {/* Login Form */}
      {(!userID || userID === "default") && (
        <>
          <h2 className="text-lg font-bold mb-2">Enter UserID</h2>
          <form onSubmit={handleSubmit} className="mb-4">
            <input
              type="text"
              value={enteredUserID}
              onChange={(e) => setEnteredUserID(e.target.value)} //keep track of what the person types in the login form
              placeholder="enter userID:"
              className="w-full p-2 border rounded text-black"
            />
            <button type="submit" className="mt-2 w-full bg-blue-500 text-white p-2 rounded">
              Submit
            </button>
          </form>
          <p className="text-red-500">{error}</p>
        </>
      )}

      {/* Logged in (not default) user view */}
      {userID && loginStatus === "logged in" && (
        <>


          <h2 className="text-lg font-bold mb-2">Hi, {userID}!</h2>
            <motion.button 
              onClick={handleSaveCanvas} 
              className={`mt-2 w-full text-white p-2 rounded ${isSaving ? "bg-stone-600" : "bg-stone-500"}`}
              animate={{ 
              y: isSaving ? [0, -2, 0] : 0
              }}
              transition={{ 
              duration: 0.5, 
              ease: "easeInOut",
              repeat: isSaving ? Infinity : 0
              }}
            >
              {isSaving ? "Saving..." : `Save this canvas (${canvasName})`}
            </motion.button>

            <button 
              onClick={handleCreateCanvas} 
              className="w-full text-center text-left bg-gray-700 hover:bg-gray-600 p-2 rounded"
            >
              + Start New Canvas
            </button> 

            <div className="flex items-center justify-between mt-2">

            <h3 className="text-md font-semibold">Your Canvases:</h3>
            <button 
              onClick={refreshCanvases} 
              className="text-white p-2 rounded"
            >
              🔄
            </button>
            <button 
              onClick={() => setEditingCanvasList(!editingCanvasList)} 
              className="text-white p-2 rounded"
            >
              ✏️
            </button>
            </div>


            <ul className="mt-2">
            {canvasList && canvasList.length > 0 ? ( // Add a check to ensure canvasList is defined
              canvasList.map((canvas) => (
              <li key={canvas.id} className="mt-1 flex items-center">
                <motion.button
                  onClick={() => {
                  window.location.href = `/?user=${userID}&canvas=${canvas.id}`;
                    }}
                    className={`w-full text-left p-2 rounded ${canvas.id === canvasID ? "bg-stone-500" : editingCanvasList ? "bg-gray-700" : "bg-gray-700 hover:bg-gray-600"}`}
                    disabled={editingCanvasList || canvas.id === canvasID}
                    animate={editingCanvasList && canvas.id != canvasID ? { rotate: [0, -1, 1, 0] } : {}}
                    transition={{ duration: 0.3, repeat: editingCanvasList ? Infinity : 0 }}
                  >
                  {canvas.name} {canvas.id === canvasID && "(current canvas)"}
                </motion.button>
                {editingCanvasList && canvas.id!= canvasID &&(
                <button 
                  onClick={() => handleDeleteCanvas(canvas.id)} 
                  className="ml-2 text-red-500 hover:text-red-700">
                  🗑️
                </button>
                )}
              </li>
              ))
            ) : (
              <p className="text-gray-400">No canvases yet.</p>
            )}
            </ul>
          <button 
            onClick={handleUserLogout} 
            className="mt-4 w-full bg-red-500 text-white p-2 rounded"
          >
            Log Out
          </button>
        </>
      )}


      
      {userID && admins.includes(userID) && (
        <div className="mt-4 p-4 bg-black rounded">
          <h2 className="text-lg font-bold mb-2">Admin Controls</h2>

          <button 
            onClick={() => window.location.href = "/admin"} 
            className="w-full bg-blue-500 text-white p-2 rounded mt-2"
          >
            Go to Admin Page
          </button>

        </div>
      )}
    </div>
  );
};

export default Sidebar;