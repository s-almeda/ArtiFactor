

import { useCallback, useEffect, useState, type MouseEvent } from "react";
import axios from "axios";
import {
  ReactFlow,
  Background,
  Controls,
  //MiniMap,
  //ReactFlowJsonObject,
  Node,
  // useNodesState,
  useReactFlow,
  applyNodeChanges,
} from "@xyflow/react";



import "@xyflow/react/dist/style.css";
import type { AppNode, LoadingNode, ImageWithLookupNodeData, TextWithKeywordsNodeData} from "./nodes/types";
import {wordsToString } from './utils/utilityFunctions';
import { nodeTypes } from "./nodes";
import useClipboard from "./hooks/useClipboard";
import { useDnD } from './context/DnDContext';
import { calcNearbyPosition } from './utils/utilityFunctions';
import { useNodeContext } from "./context/NodeContext";
import { useAppContext } from './context/AppContext';
import { useCanvasContext } from './context/CanvasContext';
import Toolbar from "./Toolbar";
// import { data } from "react-router-dom";
//FYI: we now set the backend in App.tsx!

const Flow = () => {
  const { userID, backend } = useAppContext();
  const { canvasName, canvasID, loadCanvas, quickSaveToBrowser, loadCanvasFromBrowser } = useCanvasContext();  //setCanvasName//the nodes as saved to the context and database
  const { nodes, setNodes } = useNodeContext(); //useNodesState(initialNodes);   //the nodes as being rendered in the Flow Canvas
  const { toObject, getIntersectingNodes, screenToFlowPosition, setViewport, getNodesBounds } = useReactFlow();
  const [draggableType, setDraggableType, draggableData, setDraggableData] = useDnD(); //dragStartPosition, setDragStartPosition

  const [attemptedQuickLoad, setattemptedQuickLoad] = useState(false);

  const [synthesisMode, setSynthesisMode] = useState(false);



  // for TitleBar
  const [_, setLastSaved] = useState("");
  useEffect(() => {

    const updateLastSaved = () => {
      const now = new Date();
      setLastSaved(now.toLocaleString());
    };
    const saveInterval = setInterval(updateLastSaved, 30000); // Update every 30 seconds
    return () => clearInterval(saveInterval);
  }, []);

  //attempt (just once) to load the canvas from the browser storage
  useEffect(() => {
    if (!attemptedQuickLoad) {
      //console.log ("ATTEMPTING A CANVAS QUICK LOAD");
      //TODO: check if they have saved user data / specific canvas they are working on.       
      const savedCanvas = loadCanvasFromBrowser("new-canvas");
      if (savedCanvas) {
        console.log("Canvas loaded from browser storage!");
        const { nodes = [], viewport = { x: 0, y: 0, zoom: 1 } } = savedCanvas;
        setNodes(nodes);
        setViewport(viewport);
      }
      else{
        console.log("No saved canvas found in browser storage.");
      }
      setattemptedQuickLoad(true); 
    }
  }, [attemptedQuickLoad, canvasID, setNodes, setViewport]);


  //our custom nodes change function called everytime a node changes, even a little
  const handleOnNodesChange = useCallback(
    (changes: any) => {
      setNodes((nds) => applyNodeChanges(changes, nds));
      quickSaveToBrowser(toObject()); //everytime a node is changed, save it to the browser storage
    },
      [setNodes, quickSaveToBrowser, canvasID]
  );
    const handleNodeClick = useCallback(
    (event: MouseEvent, node: Node) => {
      if (event.altKey) { 
        if (node.data.content) {
          generateNode(node.data.content as string, calcNearbyPosition(getNodesBounds([node])));
        } else if (Array.isArray(node.data.words)) {
          const content = wordsToString(node.data.words);
          generateNode(content, calcNearbyPosition(getNodesBounds([node])));
        }  console.log("you option clicked this node:", node.data);
      }
    },
  []
);


  /* ---------------------------------------------------- */
  // TODO - move this to a KeyboardShortcut Provider Context situation so we cna also track Undos/Redos
  const { handleCopy, handleCut, handlePaste } = useClipboard(nodes, setNodes); // Use the custom hook

  // Keyboard Event Listener for Copy, Cut, Paste
  useEffect(() => {
    const handleKeydown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey) {
        if (event.key === "c") handleCopy();
        if (event.key === "x") handleCut();
        if (event.key === "v") handlePaste();
      }
    };
    document.addEventListener("keydown", handleKeydown);
    return () => document.removeEventListener("keydown", handleKeydown);
  }, [handleCopy, handleCut, handlePaste]);

  /* ----------------------------------------------------*/
  
  
  //*** -- Node Adders  (functions that add nodes to the canvas) -- ***/

  const addTextWithKeywordsNode = (content: string = "your text here", provenance: "user" | "history" | "ai" = "user", position?: { x: number; y: number }) => {
    const words = content.split(' ').map((word) => ({
      value: word,
    }));

    const data: TextWithKeywordsNodeData = {
      words,
      provenance,
      content: wordsToString(words),
      intersections: [],
    };  
    
    const newTextWithKeywordsNode: AppNode = {
      id: `text-${Date.now()}`,
      type: "text",
      zIndex: 1000,
      position: position ?? {//if you've passed a position, put it there. otherwise, place it randomly.
        x: Math.random() * 250,
        y: Math.random() * 250,
      },
      data: data,
    };

    setNodes((prevNodes) => [...prevNodes, newTextWithKeywordsNode]);
  };




  const addImageWithLookupNode = (content?: string, position?: { x: number; y: number }, prompt?:string, provenance?: string) => {
    content = content ?? "https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png";
    prompt = prompt ?? "default placeholder image. try creating something of your own!";
    provenance = provenance ?? "user";
    console.log("adding an image to the canvas: ", content, prompt);
    position = position ?? { 
      x: Math.random() * 250,
      y: Math.random() * 250,
    };
    
    const newNode: AppNode = {
      id: `image-${Date.now()}`,
      type: "image",
      position: position,
      zIndex: 1000,
      data: {
        content: content,
        prompt: prompt,
        provenance: provenance,
      } as ImageWithLookupNodeData,
      dragHandle: '.drag-handle__invisible',
    };

    setNodes((prevNodes) => [...prevNodes, newNode]);
  };


  /*------------ functions to handle changes to the Flow canvas ---------------*/


  /* -- when something else is dragged over the canvas -- */
  const onDragOver = useCallback((event: { preventDefault: () => void; dataTransfer: { dropEffect: string; }; }) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  
 
  const onDrop = useCallback(
    (event: { preventDefault: () => void; clientX: any; clientY: any; }) => {
      event.preventDefault();
      console.log(`you just dropped and: ${JSON.stringify(draggableType)} with this content: ${JSON.stringify(draggableData)}`);  // check if the dropped element is valid
      if (!draggableType) {
        return;
      }
      const position = screenToFlowPosition({
        x: event.clientX - 60,
        y: event.clientY - 60,
      });

      if (draggableType === "image") {
        const content = "content" in draggableData ? draggableData["content"] as string : "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/660px-No-Image-Placeholder.svg.png?20200912122019";
        const prompt = "prompt" in draggableData ? draggableData["prompt"] as string : "";
        const provenance = "provenance" in draggableData ? draggableData["provenance"] as "user" | "history" | "ai" : "user";
        addImageWithLookupNode(content, position, prompt, provenance);

      } else if ("content" in draggableData) {
       const provenance = "provenance" in draggableData ? draggableData["provenance"] as "user" | "history" | "ai" : "user";
        addTextWithKeywordsNode(draggableData["content"] as string, provenance, position);
      }
      
    },
    [draggableType, draggableData,screenToFlowPosition],
  );


  /* -------------------------------- NODE DRAGGING -------------------------*/


  //when someone starts dragging, send the starting position to the DnD Context
  // const onNodeDragStart = useCallback(
  //   (_: MouseEvent, node: Node) => {
  //     setDragStartPosition({ x: node.position.x, y: node.position.y });
  //   },
  //   []
  // );

  //keep track o fhte node dragging 
  const onNodeDrag = useCallback(
    (_: MouseEvent, draggedNode: Node) => {
      setDraggableType(draggedNode.type as string);
      setDraggableData(draggedNode.data);
      updateIntersections(draggedNode, nodes);
    },
      [setNodes, getIntersectingNodes]
    );
    

  const onNodeDragStop = useCallback(
    (_: MouseEvent, draggedNode: Node) => {
      setNodes((currentNodes: AppNode[]) => updateIntersections(draggedNode, currentNodes));
    },
    [setNodes]
  );



  // ------------------- HELPER FUNCTIONS ------------------- //

  const updateIntersections = (draggedNode: Node, currentNodes: AppNode[]) => {
    const intersections = getIntersectingNodes(draggedNode).map((n) => n.id);
    return currentNodes.map((node: AppNode) => {
      if (node.id === draggedNode.id) {
        const updatedIntersections = [
          {
            id: node.id,
            position: node.position,
            content: node.data.content,
          },
          ...intersections.map((id) => {
            const intersectingNode = currentNodes.find((n) => n.id === id);
            if (intersectingNode && intersectingNode.type === "text") {
              //console.log(`${node.data.content} is overlapping with: ${intersectingNode.data.content}`);
              return {
                id: intersectingNode.id,
                position: intersectingNode.position,
                content: intersectingNode.data.content,
              };
            }
            return null;
          }).filter(Boolean),
        ];
        //console.log("updated intersections for node", node.data.content, ": ", updatedIntersections);
        return {
          ...node,
          data: {
            ...node.data,
            intersections: updatedIntersections,
          },
        };
      }
      return node;
    });
  };


  const deleteNodeById = (nodeId: string) => {  
    setNodes((currentNodes) => currentNodes.filter((node) => node.id !== nodeId));
    quickSaveToBrowser(toObject(), canvasID); //everytime a node is changed, save it to the browser storage
  };



// ------------------ GENERATE NODE FUNCTION ------------------ //
  const generateNode = useCallback(
    // Requests through reagent, generates an image node with a prompt and optional position
    async (prompt: string = "bunny on the moon", position: { x: number; y: number } = { x: 250, y: 250 }) => {

      // True if the "prompt" is actually an image
      const isValidImage = /\.(jpeg|jpg|gif|png|webp)$/.test(prompt);

      const loadingNodeId = `loading-${Date.now()}`;
      const loadingNode: LoadingNode = {
        id: loadingNodeId,
        type: "default",
        position,
        zIndex: 1000,
        data: { content: "loading "},
      };
      console.log("LOADING NODE:"+ loadingNode.zIndex);
      setNodes((nodes) => [...nodes, loadingNode]);

      // let's generate a node... 

      try {
        if (isValidImage){// the user has sent an image for tfext generation
          //console.log(`Describing this image: ${prompt}`);
          const response = await axios.post(`${backend}/api/generate-text`, {
            imageUrl: prompt, // Send the imageUrl as part of the request body
          });

          if (response.status === 200) {
            addTextWithKeywordsNode(response.data.text, "ai", position);
            deleteNodeById(loadingNodeId);
          } // response error
          else {
            console.error(`Text generation Error: ${response.status}`);
          }
        }

        else{ // the user has sent a text prompt for generation
          console.log(`Generating with prompt: ${prompt}`);
          const formData = new FormData();
          formData.append("prompt", prompt); // make the prompt into form data

          // Make a POST request to the backend server 
          const response = await axios.post(`${backend}/api/generate-image`, {
              prompt, // Send the prompt as part of the request body
            });
        
            if (response.status === 200) {

              addImageWithLookupNode(response.data.imageUrl, position, prompt, "ai");
              deleteNodeById(loadingNodeId);

            } // response error
            else {
              console.error(`Image generation Error: ${response.status}`);
            }
          }
        } 

        catch (error) {
          if (error instanceof Error) {
            console.error("Failed to generate image:", error.message);
          } else {
            console.error("Unknown error occurred:", error);
          }
        }
    },
    []
  );

  const [showDebugInfo, setShowDebugInfo] = useState(false);


return(
      <>



            
      <div style={{ width: '100%', height: '100%' }}>
        <ReactFlow
          nodes={nodes}
          nodeTypes={nodeTypes}
          minZoom={0.009}
          onNodesChange={handleOnNodesChange}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
          //onNodeDragStart={onNodeDragStart}
          onNodeClick={(event, node) => handleNodeClick(event, node)}
          onDrop={onDrop}
          onDragOver={onDragOver}
          zoomOnDoubleClick={false}
          fitView
          selectionOnDrag={true}
        >
          <Background />
          {/*<MiniMap />*/}
          <Controls />
        </ReactFlow>
      </div>  

      < Toolbar addTextWithKeywordsNode={addTextWithKeywordsNode} addImageNode={addImageWithLookupNode} addSynthesizer={() => setSynthesisMode(true)} />




      Debug Info
          <div className="fixed bottom-0 left-0 bg-gray-900 text-white p-4 z-50 text-sm rounded-md shadow-md">
          <button
            onClick={() => setShowDebugInfo((prev) => !prev)}
            className="bg-blue-500 text-white p-2 rounded mb-2"
          >
          {showDebugInfo ? "Hide Debug Info" : "Show Debug Info"}
          </button>
          {showDebugInfo && (
           <div> //debug info for debugging user state
            {synthesisMode}
          <p><strong>User ID:</strong> {userID}</p>
          <p><strong>Canvas Name:</strong> {canvasName}</p>
          <p><strong>Canvas ID:</strong> {canvasID}</p>
          <p><strong>Flow Nodes:</strong> {JSON.stringify(nodes, null, 2)}</p>
          <p><strong>CanvasData Currently Stored in Context:</strong> {JSON.stringify(loadCanvas, null, 2)}</p>
          </div>

          // <div>
          //   <p><strong>Draggable Type:</strong> {draggableType}</p>
          //   <p><strong>Draggable Data:</strong> {JSON.stringify(draggableData, null, 2)}</p>
          //   <p><strong>Drag Start Position:</strong> {JSON.stringify(dragStartPosition, null, 2)}</p>
          // </div>
          )
          }
        </div>


      </>

  );
};

export default Flow;



    // /* 
    // =============================================================================== 
    // ||      Saving and Loading... (quick save functions are in CanvasContext)    || 
    // =============================================================================== 
    //  */
    
    // /* ----- call the CanvasContext save to the backend database  ----*/
    // const handleLoadCanvas = useCallback(async () => {
    //   console.log("loading canvas data for: ", canvasID);
    //   const canvasData = await loadCanvas(canvasID);
    //   console.log("Loaded Canvas Data:", canvasData); // Print to console for debugging
    //   if (canvasData !== undefined) {
    //     const { nodes = [], viewport = { x: 0, y: 0, zoom: 1 } } = canvasData;
    //     setNodes(nodes);
    //     setViewport(viewport);
    //   }
    // }, [canvasID, loadCanvas, setNodes, setViewport]);
