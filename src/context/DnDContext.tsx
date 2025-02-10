import { createContext, useContext, useState } from 'react';

// A Drag and Drop context keeps track of the type of node and content of node currently being dragged around the app! 
// it needs to be separate so people can drag between different components (the palette and the Flow canvas)
const DnDContext = createContext<
  [
    string, 
    React.Dispatch<React.SetStateAction<string>>, 
    object, 
    React.Dispatch<React.SetStateAction<object>>, 
    { x: number, y: number }, 
    React.Dispatch<React.SetStateAction<{ x: number, y: number }>>
  ]
>(["default", () => {}, {}, () => {}, { x: 0, y: 0 }, () => {}]);

export const DnDProvider = ({ children }: { children: any }) => {
  const [draggableType, setDraggableType] = useState("default");
  const [draggableData, setDraggableData] = useState({});
  const [dragStartPosition, setDragStartPosition] = useState({x: 0, y: 0});

  // useEffect(() => {
  //   //console.log(`DnDProvider receiving: ${draggableType} and ${JSON.stringify(draggableData)}`);
  // }, [draggableType, draggableData]);
  
  return (
    <DnDContext.Provider value={[draggableType, setDraggableType, draggableData, setDraggableData, dragStartPosition, setDragStartPosition]}>
      {children}
      {/* {console.log(`you are dragging a <${draggableType}> node with this data: ${JSON.stringify(draggableData)}`)} */}
    </DnDContext.Provider>
  );
}

export default DnDContext;

export const useDnD = () => {
  return useContext(DnDContext);
}