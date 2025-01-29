import React from 'react';

const charLimit = 25;



interface PaletteProps {
  onAddNode: (type: string, content: any) => void;
}

const Palette: React.FC<PaletteProps> = ({ onAddNode }) => {

// -------- Let's first define Palette NODES  (the bubbles that show up in the palette) ------------------- 
    interface PaletteNodeProps {
        content: string;
        charLimit: number;
        type: string;
    }

const handleNodeClick = (content: string) => {
        console.log(`PaletteNode clicked: ${content}`);
};

const PaletteNode: React.FC<PaletteNodeProps> = ({ type = "text", content, charLimit}) => (
        <div
                className="bg-white border border-gray-600 rounded-md p-2 cursor-grab hover:bg-gray-50 hover:shadow-sm transition-all text-sm"
                draggable
                onDragStart={handleDragStart}
                onClick={() => {
                        handleNodeClick(content);
                        onAddNode(type, content);
                }}
                style={{ width: '21vw'}}
        >
                {
                        // --- text palette nodes --- // 
                        type === "text" 
                                ? (content.length > charLimit ? `${content.substring(0, charLimit)}...` : content)
                                : "" //something like this for image nodes: : <img src={content} alt="img" style={{ width: "100%" }} />
                }
        </div>
);

    const handleDragStart = (event: { dataTransfer: { setData: (arg0: string, arg1: string) => void; effectAllowed: string; }; }) => {
        console.log("you are dragging me!!");
        event.dataTransfer.setData('application/reactflow', 'textNode');
        event.dataTransfer.effectAllowed = 'move';
    };
    const testArray = ["claude monet was an amazing guy and he was soooo cool", "van gogh", "picasso", "dali", "michelangelo", "raphael", "leonardo"];
    


    return (
        <div className="bg-white p-4">
                <div className="space-y-3">
                {
                testArray.map((item,i) => (

                        <PaletteNode key={item + "_" + i}  type={'text'} content={item} charLimit={charLimit} />
                ))}
                
                </div>
            </div>
    );
};

export default Palette;