import { createContext, useContext, useState } from 'react';

interface DnDContextType {
  draggableType: string;
  setDraggableType: React.Dispatch<React.SetStateAction<string>>;
  draggableData: object;
  setDraggableData: React.Dispatch<React.SetStateAction<object>>;
  dragStartPosition: { x: number, y: number };
  setDragStartPosition: React.Dispatch<React.SetStateAction<{ x: number, y: number }>>;
  // parentNodeId: string;
  // setParentNodeId: React.Dispatch<React.SetStateAction<string>>;
}

const DnDContext = createContext<DnDContextType | undefined>(undefined);

export const DnDProvider = ({ children }: { children: any }) => {
  const [draggableType, setDraggableType] = useState("default");
  const [draggableData, setDraggableData] = useState({});
  const [dragStartPosition, setDragStartPosition] = useState({ x: 0, y: 0 });

  return (
    <DnDContext.Provider value={{ draggableType, setDraggableType, draggableData, setDraggableData, dragStartPosition, setDragStartPosition }}>
      {children}
    </DnDContext.Provider>
  );
}

export default DnDContext;

export const useDnD = () => {
  const context = useContext(DnDContext);
  if (context === undefined) {
    throw new Error('useDnD must be used within a DnDProvider');
  }
  return context;
}
