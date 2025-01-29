import { createContext, useContext, useState, useEffect } from 'react';
 
// A Drag and Drop context keeps track of the type of node and content of node currently being dragged around the app! 
// it needs to be separate so people can drag between different components (the palette and the Flow canvas)
const DnDContext = createContext<[string, React.Dispatch<React.SetStateAction<string>>, string, React.Dispatch<React.SetStateAction<string>>]>(["default", () => {}, "null", () => {}]);
 
export const DnDProvider = ({ children }: { children: any }) => {
  const [draggableType, setDraggableType] = useState("default");
  const [draggableContent, setDraggableContent] = useState("null");
 
  useEffect(() => {
    console.log(`DnDProvider receiving: ${draggableType} and ${draggableContent}`);
  }, [draggableType, draggableContent]);
  
  return (
    <DnDContext.Provider value={[draggableType, setDraggableType, draggableContent, setDraggableContent]}>
      {children}
      {console.log(`${draggableType} and ${draggableContent}`)}
    </DnDContext.Provider>
  );
}
 
export default DnDContext;
 
export const useDnD = () => {
  return useContext(DnDContext);
}