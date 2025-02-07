import { useState } from "react";
import { useCanvasContext } from "./contexts/CanvasContext";

const Sidebar = ({ onClose, onUserLogin, backend }: { 
  onClose: () => void; 
  onUserLogin: (userID: string) => void;
  backend: string; 
}) => {
  const { loadCanvas } = useCanvasContext();
  const [userID, setUserID] = useState("");
  const [error, setError] = useState("");
  const [canvases, setCanvases] = useState<{ id: string; name: string }[]>([]);
  const [loggedInUser, setLoggedInUser] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    try {
      const response = await fetch(`${backend}/api/list-users`);
      const data = await response.json();

      if (!data.success) {
        setError("Error fetching users.");
        return;
      }

      const user = data.users.find((u: { id: string }) => u.id === userID);
      if (user) {
        setLoggedInUser(userID);
        setCanvases(user.canvases);
        onUserLogin(userID);
      } else {
        setError("User not found.");
      }
    } catch (error) {
      console.error("Error checking user:", error);
      setError("Something went wrong.");
    }
  };

  return (
    <div className="fixed top-0 left-0 w-64 h-full bg-gray-800 text-white p-4 z-40">
      <button
        onClick={onClose}
        className="absolute top-2 right-2 bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-2 rounded"
      >
        Close
      </button>

      {!loggedInUser ? (
        <>
          <h2 className="text-lg font-bold mb-2">Enter UserID</h2>
          <form onSubmit={handleSubmit} className="mb-4">
            <input
              type="text"
              value={userID}
              onChange={(e) => setUserID(e.target.value)}
              placeholder="Enter userID"
              className="w-full p-2 border rounded text-black"
            />
            <button type="submit" className="mt-2 w-full bg-blue-500 text-white p-2 rounded">
              Submit
            </button>
          </form>
          {error && <p className="text-red-500">{error}</p>}
        </>
      ) : (
        <>
          <h2 className="text-lg font-bold mb-2">Hello, {loggedInUser}!</h2>
          <h3 className="text-md font-semibold mt-4">Your Canvases:</h3>
          <ul className="mt-2">
            {canvases.length > 0 ? (
              canvases.map((canvas) => (
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
