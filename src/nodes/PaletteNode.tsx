// NOTE--- THIS DOESN'T ACTUALLY SEEM TO BE USED...

interface PaletteNodeProps {
  content: string;
  type: "text" | "image";
}


export function PaletteNode({ type, content }: PaletteNodeProps) {
  return (
      <div>
      {type === "text" ? (
        <div>
          {content.length > 25 ? `${content.substring(0, 25)}...` : content}
        </div>
      ) : type === "image" ? (
        <img
          src={content}
          alt="Saved Image"
          className="w-full h-auto rounded-md"
          style={{ maxWidth: "100px", maxHeight: "50px", objectFit: "cover" }}
        />
      ) : (
        ""
      )}
    </div>
  );
}

export default PaletteNode;
