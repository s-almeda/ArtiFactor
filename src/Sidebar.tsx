import { Link } from "react-router-dom";

const Sidebar = ({ onClose }: { onClose: () => void }) => {
  return (
    <div className="fixed top-0 left-0 w-64 h-full bg-gray-800 text-white p-4 z-40">
      <button
        onClick={onClose}
        className="absolute top-2 right-2 bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-2 rounded"
      >
        Close
      </button>

      <Link to="/">
        <p>Home</p>
      </Link>
      <Link to="/canvas">
        <p>Canvas</p>
      </Link>
    </div>
  );
};

export default Sidebar;
