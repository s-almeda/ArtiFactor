//import type { PaletteNode as PaletteNodeType } from "./types";

interface PaletteNodeProps {
  content: string;
  type: "text" | "image";
  onAddNode: (type: string, content: string) => void;
}

export function PaletteNode({ type, content, onAddNode }: PaletteNodeProps) {
  return (
    <div
      className="bg-white border border-gray-600 rounded-md p-2 cursor-pointer hover:bg-gray-50 hover:shadow-sm transition-all text-sm"
      onClick={() => onAddNode(type, content)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          onAddNode(type, content);
        }
      }}
      tabIndex={0}
      style={{ width: "21vw" }}
    >
      {type === "text" ? (
        <div>
          {content.length > 25 ? `${content.substring(0, 25)}...` : content}
        </div>
      ) : type === "image" ? (
        <img
          src={content}
          alt="Saved Image"
          className="w-full h-auto rounded-md"
          style={{ maxWidth: "100px", maxHeight: "100px", objectFit: "cover" }}
        />
      ) : (
        ""
      )}
    </div>
  );
}

export default PaletteNode;
