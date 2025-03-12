import { FC, DragEvent, useState } from "react";
import { useDnD } from "../context/DnDContext";
import { Download} from "lucide-react";
import type { NodeData } from "../context/PaletteContext";
import { useAppContext } from "../context/AppContext";
import { motion } from "framer-motion";

interface PaletteNodeProps {
  data: NodeData;
  type: "text" | "image";
  removeNode: (id: number) => void;
}

const PaletteNode: FC<PaletteNodeProps> = ({
  data,
  type,
  removeNode,
}) => {
  const { setDraggableType, setDraggableData } = useDnD();
  const [expanded, setExpanded] = useState(false); // State for image expansion
  const { userID, admins } = useAppContext();
  const [showTitle, setShowTitle] = useState(false); // State for image title

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
      style={{ width: "100%" }}
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => {
        setExpanded(false)
        setShowTitle(false)
      }}
      onClick={() => setShowTitle(!showTitle)}
        >
      {type === "text" ? (
        <motion.div
          className="relative pr-2.5"
          style={{ textOverflow: "ellipsis", overflow: "hidden" }}
          animate={{ height: expanded ? "auto" : "20px" }}
          transition={{ duration: 0.35 }}
        >

        {data.content}
          <button
        type="button"
        className="absolute top-0 right-0 bg-white text-black rounded-full w-5 h-5 text-xs flex items-center justify-center z-50"
        onClick={(e) => {
          e.stopPropagation();
          removeNode(data.id);
        }}
          >
        ✕
          </button>
        </motion.div>
      ) : (
        <div className="relative">
          <motion.div
            className={`max-h-${
              expanded ? "full" : "60"
            } overflow-hidden relative`}
            style={{ height: expanded ? "auto" : "60px" }}
            animate={{ height: expanded ? "auto" : "60px" }}
            transition={{ duration: 0.35 }}
          >
            <img
              src={data.content}
              alt={data.prompt}
              className="rounded-md object-cover w-full h-full"
            />
            <div
              className={`absolute bottom-0 left-0 w-full h-full bg-white bg-opacity-50 
               text-gray-800 text-md uppercase text-center p-1 
               ${showTitle ? "opacity-100" : "opacity-0"} 
               transition-opacity duration-300 font-bold italic 
               leading-tight flex flex-col items-center justify-center`}
              style={{ zIndex: 10, visibility: showTitle ? "visible" : "hidden" }}
            >
              <span>
              {data.prompt.length > 40
              ? `${data.prompt.substring(0, 40)}...`
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
            </div>
          </motion.div>
          <button
            type="button"
            className="absolute top-1 right-1 bg-white text-black rounded-full w-5 h-5 text-xs flex items-center justify-center z-50"
            onClick={(e) => {
              e.stopPropagation();
              removeNode(data.id);
            }}
          >
            ✕
          </button>
        { (userID && admins.includes(userID)) &&
            <button
                type="button"
                className="absolute top-1 left-1 bg-white text-black rounded-full w-5 h-5 text-xs flex items-center justify-center z-50"
                onClick={(e) => {
                  e.stopPropagation();
                  console.log(data);
                }}
              >
                data
            </button>
          }

        </div>
      )}
    </div>
  );
};

export default PaletteNode;
