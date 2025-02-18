import { useState, useEffect } from "react";
import { useCanvasContext } from "./context/CanvasContext";
import { useAppContext } from "./context/AppContext";

const Sidebar = ({ onClose }: { 
  onClose: () => void; 
}) => {
  const { canvasID, canvasName, loadCanvas, saveNewCanvas, saveCanvas, deleteCanvas, createCanvas } = useCanvasContext();
  const { backend, handleUserLogin, userID, addUser, admins } = useAppContext();

  const [enteredUserID, setEnteredUserID] = useState("");
  const [enteredPassword, setEnteredPassword] = useState("");

  const [error, setError] = useState("");
  const [canvasList, setCanvasList] = useState<{ id: string; name: string }[]>([]); // Initialize as an empty array
  const [editingCanvasList, setEditingCanvasList] = useState(false);

  useEffect(() => { // check if the user has logged in
    if (userID && userID !== "default") {
      refreshCanvases();
    }
  }, [userID]);

  const handleSaveCanvas = async () => {
    if (canvasID !== "new-canvas") {
      saveCanvas();
      return;
    }
    try {
      const response = await fetch(`${backend}/api/next-canvas-id/${userID}`);
      const data = await response.json();

      if (!data.success) {
        setError("Error fetching new canvas ID.");
        return;
      }
      console.log("will use new canvas ID:", data.nextCanvasId);

      const newCanvasName = canvasName === "Untitled" ? prompt("Give this canvas a name:") || "Untitled" : canvasName;

      if (newCanvasName) {
        await saveNewCanvas(data.nextCanvasId, newCanvasName);
        refreshCanvases();
      } else {
        setError("Canvas save canceled. You've gotta name it something!");
      }
    } catch (error) {
      console.error("Error fetching new canvas ID:", error);
      setError("Something went wrong.");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    try {
      await handleUserLogin(enteredUserID, "");
      //save the login to local storage
      localStorage.setItem("last_userID", enteredUserID);
      localStorage.setItem("last_password", ""); //todo, fix when we make passwords a thing

    } catch (error) {
      console.error("Error logging in user:", error);
      setError("Something went wrong.");
    }
  };
  const handleUserLogout = () => {
    const confirmLogout = window.confirm("Are you sure you want to log out? (You may lose unsaved changes)");
    if (confirmLogout) {
      handleUserLogin("default", "");
    }
  };
  
  const handleCreateCanvas = async () => {
    if (canvasID === "new-canvas") {
      const confirmNewCanvas = window.confirm("Are you sure you want to start a new canvas? (You may lose unsaved changes)");
      confirmNewCanvas && createCanvas()
    } else {
      const confirmSaveCanvas = window.confirm("Would you like to save your current canvas before starting a new one?");
      (confirmSaveCanvas) ? handleSaveCanvas(): //TODO, make this a modal with 3 options, save and start new, start new, cancel
      createCanvas()
    }
  }

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

      console.log("user's canvasList", data.canvases);
      setCanvasList(data.canvases);
    } catch (error) {
      console.error("Error refreshing canvasList:", error);
      setError("Something went wrong.");
    }
  };

  const handleDeleteCanvas = async (canvasIDToDelete: string) => {
    await deleteCanvas(canvasIDToDelete);
    refreshCanvases();
  }

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
      {userID && userID !== "default" && (
        <>
        <button 
          onClick={handleSaveCanvas} 
          className="mt-2 w-full bg-green-500 text-white p-2 rounded">
          Save Canvas
        </button>

          <h2 className="text-lg font-bold mb-2">Hi, {userID}!</h2>

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
                <button 
                onClick={() => loadCanvas(canvas.id)} // 🔹 Clicking on a canvas button loads it! 
                className="w-full text-left bg-gray-700 hover:bg-gray-600 p-2 rounded"
                >
                {canvas.name}
                </button>
                {editingCanvasList && (
                <button 
                  onClick={() => handleDeleteCanvas(canvas.id)} 
                  className="ml-2 text-red-500 hover:text-red-700">
                  ❌
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


      
      {/* Admin Controls  TODO: make this its own component */}
      {userID && admins.includes(userID) && (
        <div className="mt-4 p-4 bg-black rounded">
          <h2 className="text-lg font-bold mb-2">Admin Controls</h2>

          <h3>Add new user</h3>
            <input
            type="text"
            placeholder="Enter new user ID"
            value={enteredUserID}
            onChange={(e) => setEnteredUserID(e.target.value)}
            className="w-full p-2 border rounded text-black mb-2"
            />
            <input
            type="password"
            placeholder="Enter new user password"
            value={enteredPassword}
            onChange={(e) => setEnteredPassword(e.target.value)}
            className="w-full p-2 border rounded text-black mb-2"
            />
            <button 
            onClick={() => addUser(enteredUserID, enteredPassword)} 
            className="w-full bg-green-500 text-white p-2 rounded"
            >
            Add User
            </button>
            <button 
            onClick={() => console.log(localStorage)} 
            className="w-full bg-blue-500 text-white p-2 rounded mt-2"
            >
            Print Local Storage to console
            </button>
        </div>
      )}
    </div>
  );
};

export default Sidebar;