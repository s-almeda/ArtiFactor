import React, { useState, MouseEvent } from 'react';

interface ToolbarProps {
    addTextWithKeywordsNode: () => void;
    addImageNode: () => void;
    addSynthesizer: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({ addTextWithKeywordsNode, addImageNode, addSynthesizer }) => {
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
                    padding: '5px',
                    textAlign: 'center',
                    background: '#444',
                    color: 'white',
                    borderRadius: '5px',
                    fontSize: '16px',
                    fontWeight: 'bold'
                }}
            >
                ☰
            </div>
            <button onClick={() => addTextWithKeywordsNode()}>T</button>
            <button onClick={() => addImageNode()}>🌄</button>
            <button onClick={() => addSynthesizer()}>✨</button>
            
        </div>
    );
};

export default Toolbar;