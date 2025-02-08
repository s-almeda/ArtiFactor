import { useState, useEffect } from "react";
import { useCanvasContext } from "./context/CanvasContext";
import { useAppContext } from "./context/AppContext";

const Sidebar = ({ onClose }: { 
  onClose: () => void; 
}) => {
  const { loadCanvas } = useCanvasContext();
  const { backend, handleUserLogin, setLoadCanvasRequest, userID } = useAppContext();
  const [enteredUserID, setEnteredUserID] = useState("");

  const [error, setError] = useState("");
  const [canvasList, setCanvasList] = useState<{ id: string; name: string }[]>([]); // Initialize as an empty array
  const [loggedInUser, setLoggedInUser] = useState<string | null>(userID); // for displaying in the top left! 

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    try {
      await handleUserLogin(enteredUserID);
      setLoadCanvasRequest(true);
      setLoggedInUser(enteredUserID); // Update to use enteredUserID
    } catch (error) {
      console.error("Error logging in user:", error);
      setError("Something went wrong.");
    }
  };

  const refreshCanvases = async () => {
    if (!loggedInUser) return;

    try {
      const response = await fetch(`${backend}/api/list-users`);
      const data = await response.json();

      if (!data.success) {
        setError("Error fetching users.");
        return;
      }

      const user = data.users.find((u: { id: string }) => u.id === loggedInUser);
      if (user) {
        setCanvasList(user.canvasList);
      } else {
        setError("User not found.");
      }
    } catch (error) {
      console.error("Error refreshing canvasList:", error);
      setError("Something went wrong.");
    }
  };

  useEffect(() => {
    if (userID) {
      setLoggedInUser(userID);
    }
  }, [userID]);

  return (
    <div className="fixed top-0 left-0 w-64 h-full bg-gray-800 text-white p-4 z-40">
      <button
        onClick={onClose}
        className="absolute top-2 right-2 bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-2 rounded"
      >
        Close
      </button>

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
      {error && <p className="text-red-500">{error}</p>}

      {loggedInUser && (
        <>
          <h2 className="text-lg font-bold mb-2">Hello, {loggedInUser}!</h2>
          <button 
            onClick={refreshCanvases} 
            className="mt-2 w-full bg-green-500 text-white p-2 rounded"
          >
            Refresh Canvases
          </button>
          <h3 className="text-md font-semibold mt-4">Your Canvases:</h3>
          <ul className="mt-2">
            {canvasList && canvasList.length > 0 ? ( // Add a check to ensure canvasList is defined
              canvasList.map((canvas) => (
                <li key={canvas.id} className="mt-1">
                  <button 
                    onClick={() => loadCanvas(canvas.id)} // 🔹 Clicking on a canvas button loads it! 
                    className="w-full text-left bg-gray-700 hover:bg-gray-600 p-2 rounded"
                  >
                    {canvas.name}
                  </button>
                </li>
              ))
            ) : (
              <p className="text-gray-400">No canvases yet.</p>
            )}
          </ul>
        </>
      )}
    </div>
  );
};

export default Sidebar;