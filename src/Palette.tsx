import React from 'react';

interface PaletteProps {
  onAddNode: () => void;
}

const Palette: React.FC<PaletteProps> = ({ onAddNode }) => {
  const handleDragStart = (event: { dataTransfer: { setData: (arg0: string, arg1: string) => void; effectAllowed: string; }; }) => {
    event.dataTransfer.setData('application/reactflow', 'textNode');
    event.dataTransfer.effectAllowed = 'move';
  };
  const testArray = ["monet", "van gogh", "picasso", "dali", "michelangelo", "raphael", "leonardo"];

  return (
    <div className="bg-white p-4">
        <div className="space-y-3">
        {
        testArray.map((item) => (
          <div
          className="bg-white border border-gray-600 rounded-md p-2 cursor-grab hover:bg-gray-50 hover:shadow-sm transition-all text-sm"
          onDragStart={handleDragStart}
          onClick={onAddNode}
          draggable
          style={{ width: '21vw'}}
          >
            {item}
            </div>
        ))}
        </div>
      </div>
  );
};

export default Palette;