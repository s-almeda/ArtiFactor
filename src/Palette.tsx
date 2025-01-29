import React from 'react';
import { useDnD } from './DnDContext';

const charLimit = 25;

interface PaletteProps {
    onAddNode: (type: string, content: string) => void;
}

interface PaletteNodeProps {
    content: string;
    charLimit: number;
    type: string;
    onAddNode: (type: string, content: string) => void;
}

//--- The Palette Node (the draggable nodes that you can drag into the Flow canvas)
const PaletteNode: React.FC<PaletteNodeProps> = ({ type = "text", content, charLimit, onAddNode }) => {
    const [_, setDraggableType, __, setDraggableContent] = useDnD();
    


    const onDragStart = (event: { dataTransfer: { effectAllowed: string; }; }, type: string, content: string) => {
        console.log(`you're dragging me! sending over ${type} and ${content}`);
        event.dataTransfer.effectAllowed = 'move';
        setDraggableType(type);
        setDraggableContent(content);
    };

    const handleNodeClick = (content: string) => {
        console.log(`PaletteNode clicked: ${content}`);
        onAddNode(type, content);
    };

    return (
        <div
            className="bg-white border border-gray-600 rounded-md p-2 cursor-grab hover:bg-gray-50 hover:shadow-sm transition-all text-sm"
            draggable
            onDragStart={(event) => onDragStart(event, type, content)}
            onClick={() => handleNodeClick(content)}
            style={{ width: '21vw' }}
        >
            {type === "text"
                ? (content.length > charLimit ? `${content.substring(0, charLimit)}...` : content)
                : ""}
        </div>
    );
};

// -- The Palette itself (the right side box that holds all the palette nodes!)

const Palette: React.FC<PaletteProps> = ({ onAddNode }) => {
    const testArray = ["claude monet was an amazing guy and he was soooo cool", "van gogh", "picasso", "dali", "michelangelo", "raphael", "leonardo"];

    return (
        <div className="bg-white p-4">
            <div className="space-y-3">
                {testArray.map((item, i) => (
                    <PaletteNode key={item + "_" + i} type={'text'} content={item} charLimit={charLimit} onAddNode={onAddNode} />
                ))}
            </div>
        </div>
    );
};

export default Palette;
