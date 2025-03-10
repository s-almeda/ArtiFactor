/* what shows up in the Palette! */
import { type FC, type DragEvent } from "react";//useState
import { useDnD } from "../context/DnDContext";
import type { NodeData } from "../context/PaletteContext";
import { Download } from "lucide-react";

interface PaletteNodeProps {
  data: NodeData;
  charLimit: number;
  type: "text" | "image";
  removeNode: (id: number) => void; // Function to remove the node
}

const PaletteNode: FC<PaletteNodeProps> = ({
  data,
  charLimit,
  type,
  removeNode,
}) => {
  //const [___, setIsHovered] = useState(false);
  const { setDraggableType, setDraggableData } = useDnD();

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.effectAllowed = "move";
    setDraggableType(type);
    setDraggableData(data);
  };

  const downloadImage = async () => {
    try {
      const imageUrl = data.content;
      const response = await fetch(imageUrl);

      if (!response.ok) throw new Error("Failed to fetch the image.");

      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = `${data.prompt}.png`;

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error("Error downloading image:", error);
    }
  };

  return (
    <div
      className="border border-gray-600 rounded-md p-2 cursor-grab hover:bg-gray-50 hover:shadow-sm transition-all text-sm relative"
      draggable
      onDragStart={onDragStart}
      // onMouseEnter={() => setIsHovered(true)}
      // onMouseLeave={() => setIsHovered(false)}
      style={{ width: "100%" }}
    >
      {type === "text" ? (
        <div>
          <div>
            {data.content.length > charLimit
              ? `${data.content.substring(0, charLimit)}...`
              : data.content}
          </div>
          <button
            type="button"
            className="absolute top-1 right-1 bg-white text-black rounded-full w-5 h-5 text-xs flex items-center justify-center z-50"
            onClick={(e) => {
              e.stopPropagation(); // Prevent interference with dragging
              removeNode(data.id);
            }}
          >
            ✕
          </button>
        </div>
      ) : (
        <div className="relative max-h-[60px] overflow-hidden">
          <img
            src={data.content}
            alt={data.prompt}
            className="rounded-md object-cover w-full h-full"
          />
          <div
            className="absolute bottom-0 left-0 w-full h-full bg-white bg-opacity-50 
             text-gray-800 text-md uppercase text-center p-1 opacity-0 
             transition-opacity duration-300 hover:opacity-100 font-bold italic 
             leading-tight flex flex-col items-center justify-center"
            style={{ zIndex: 10 }}
          >
            <span>
              {data.prompt.length > 60
                ? `${data.prompt.substring(0, 60)}...`
                : data.prompt}
            </span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                downloadImage();
              }}
              className="bg-gray-800 text-white px-2 py-1 rounded-md text-sm mt-2"
            >
              <Download size={16} />
            </button>
            <button
              type="button"
              className="absolute top-1 right-1 bg-white text-black rounded-full w-5 h-5 text-xs flex items-center justify-center z-50"
              onClick={(e) => {
                e.stopPropagation(); // Prevent interference with dragging
                removeNode(data.id);
              }}
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PaletteNode;
