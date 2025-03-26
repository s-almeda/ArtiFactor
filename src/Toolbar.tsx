import React, { useState, MouseEvent } from 'react';
import { Plus } from 'lucide-react';

interface ToolbarProps {
    addTextWithKeywordsNode: () => void;
    addImageNode: () => void;
    addSynthesizer: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({ addTextWithKeywordsNode }) => { //addImageNode, addSynthesizer
    const [toolbarPosition, setToolbarPosition] = useState({ x: 20, y: 100 });
    const [dragging, setDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

    const handleDragStart = (event: MouseEvent<HTMLDivElement>) => {
        setDragging(true);
        setDragStart({ x: event.clientX - toolbarPosition.x, y: event.clientY - toolbarPosition.y });
    };

    const handleDragMove = (event: MouseEvent<HTMLDivElement>) => {
        if (dragging) {
            setToolbarPosition({ x: event.clientX - dragStart.x, y: event.clientY - dragStart.y });
        }
    };

    const handleDragEnd = () => {
        setDragging(false);
    };

    return (
        <div
            style={{
                position: 'absolute',
                top: `${toolbarPosition.y}px`,
                left: `${toolbarPosition.x}px`,
                display: 'flex',
                flexDirection: 'column',
                gap: '10px',
                background: 'rgba(0,0,0,0.8)',
                padding: '10px',
                borderRadius: '10px',
                zIndex: 10,
                cursor: dragging ? 'grabbing' : 'grab',
                userSelect: 'none'
            }}
            onMouseMove={handleDragMove}
            onMouseUp={handleDragEnd}
            onMouseLeave={handleDragEnd}
        >
            <div
                onMouseDown={handleDragStart}
                style={{
                    cursor: 'grab',
                    padding: '-10px',
                    margin: '-5px',
                    textAlign: 'center',
                    color: '#AAA',
                    fontSize: '14px',
                    fontWeight: 'bold'
                }}
            >
                ☰
            </div>
            <button
                onClick={() => addTextWithKeywordsNode()}
                className={'text-white, bg-stone-800'}
                style={{
                    width: '30px',
                    height: '30px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: 'gray 1.5px solid',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    transition: 'background 0.3s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = '#777')}
                onMouseLeave={(e) => (e.currentTarget.style.background = '#555')}
            >
                
                <Plus size={28} />
            </button>
            {/* <button onClick={() => addImageNode()}>🌄</button>
            <button onClick={() => addSynthesizer()}>✨</button> */}
            
        </div>
    );
};

export default Toolbar;