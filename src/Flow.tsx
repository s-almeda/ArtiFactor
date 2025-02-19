

import { useCallback, useEffect, useState, type MouseEvent } from "react";
import axios from "axios";
import {
  ReactFlow,
  Background,
  Controls,
  //MiniMap,
  //ReactFlowJsonObject,
  Node,
  useNodesState,
  useReactFlow,
  applyNodeChanges,
} from "@xyflow/react";



import "@xyflow/react/dist/style.css";
import type { AppNode, TextNodeData, SynthesizerNodeData, ImageWithLookupNodeData, TextWithKeywordsNodeData} from "./nodes/types";
import { initialNodes, nodeTypes } from "./nodes";
import useClipboard from "./hooks/useClipboard";
import { useDnD } from './context/DnDContext';
import { calcNearbyPosition } from './utils/calcNearbyPosition';
//import { addTextNode, addImageNode, addSynthesizer } from './utils/nodeAdders';
import { useAppContext } from './context/AppContext';
import { useCanvasContext } from './context/CanvasContext';
import Toolbar from "./Toolbar";
// import { data } from "react-router-dom";
//FYI: we now set the backend in App.tsx!

const Flow = () => {
  const { userID, backend } = useAppContext();
  //const { canvasName, canvasID, loadCanvas, quickSaveToBrowser, loadCanvasFromBrowser } = useCanvasContext();  //setCanvasName//the nodes as saved to the context and database
  const { canvasName, canvasID, loadCanvas, quickSaveToBrowser, loadCanvasFromBrowser } = useCanvasContext();  //setCanvasName//the nodes as saved to the context and database
  const [ nodes, setNodes] = useNodesState(initialNodes);   //the nodes as being rendered in the Flow Canvas
  const { toObject, getIntersectingNodes, screenToFlowPosition, setViewport, getNodesBounds } = useReactFlow();
  const [draggableType, setDraggableType, draggableData, setDraggableData, dragStartPosition, setDragStartPosition] = useDnD();

  const [attemptedQuickLoad, setattemptedQuickLoad] = useState(false);


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
      console.log("you clicked me while pressing the 'option' key!");
      console.log("Node Data:", node.data);
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

  const addTextWithKeywordsNode = (content: string = "your text here", position?: { x: number; y: number }) => {
    const words = content.split(' ').map((word) => ({
      value: word,
    }));

    const data: TextWithKeywordsNodeData = {
      words,
    };  const newTextWithKeywordsNode: AppNode = {
      id: `text-${Date.now()}`,
      type: "textWithKeywords",
      position: position ?? {//if you've passed a position, put it there. otherwise, place it randomly.
        x: Math.random() * 250,
        y: Math.random() * 250,
      },
      data: data,
    };

    setNodes((prevNodes) => [...prevNodes, newTextWithKeywordsNode]);
  };


  const addTextNode = (content: string = "your text here", position?: { x: number; y: number }) => {
    const newTextNode: AppNode = {
      id: `text-${Date.now()}`,
      type: "text",
      position: position ?? {//if you've passed a position, put it there. otherwise, place it randomly.
      x: Math.random() * 250,
      y: Math.random() * 250,
      },
      data: {
      //label: `Text Node ${nodes.length + 1}`,
      content: content,
      loading: false,
      combinable: false
      } as TextNodeData,
    };

    setNodes((prevNodes) => [...prevNodes, newTextNode]);
  };

  const addImageWithLookupNode = (content?: string, position?: { x: number; y: number }, prompt?:string) => {
    content = content ?? "https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png";
    prompt = prompt ?? "default alligator image";
    console.log("adding an image to the canvas: ", content, prompt);
    position = position ?? { 
      x: Math.random() * 250,
      y: Math.random() * 250,
    };
    
    const newNode: AppNode = {
      id: `image-${nodes.length + 1}`,
      type: "imagewithlookup",
      position: position,
      data: {
        content: content,
        prompt: prompt,
      } as ImageWithLookupNodeData,
      dragHandle: '.drag-handle__invisible',
    };

    setNodes((prevNodes) => [...prevNodes, newNode]);
  };



  const addSynthesizer = (position ?:{ x:number, y:number}) => {
    const newSynthesizerNode: AppNode = {
      id: `synthesizer-${nodes.length + 1}`,
      type: "synthesizer",
      position: position ?? {
      x: Math.random() * 250,
      y: Math.random() * 250,
      },
      data: {
      content: "",
      mode: "ready",
      yOffset: 0,
      xOffset: 0,
      updateNode: (content: string, mode: "dragging" | "ready" | "generating" | "check") => {
        console.log(`new node passed with content: ${content} and mode: ${mode}`);
        return true;
      }
      } as SynthesizerNodeData,
    };
    

    setNodes((prevNodes) => [...prevNodes, newSynthesizerNode]);
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
        addImageWithLookupNode(content, position, prompt);

      } else if ("content" in draggableData) {
        addTextWithKeywordsNode(draggableData["content"] as string, position);
      }
      
    },
    [draggableType, draggableData,screenToFlowPosition],
  );


  /* -------------------------------- NODE DRAGGING -------------------------*/


  //when someone starts dragging, send the starting position to the DnD Context
  const onNodeDragStart = useCallback(
    (_: MouseEvent, node: Node) => {
      setDragStartPosition({ x: node.position.x, y: node.position.y });
    },
    []
  );

  //keep track o fhte node dragging 
  const onNodeDrag = useCallback(
    (_: MouseEvent, draggedNode: Node) => {
      const intersections = getIntersectingNodes(draggedNode).map((n) => n.id);
      setDraggableType(draggedNode.type as string);
      setDraggableData(draggedNode.data);
      setNodes((currentNodes: AppNode[]) =>
        currentNodes.map((node: AppNode) => {
          if (node.id === draggedNode.id) {
            return {
              ...node,
              position: draggedNode.position,
              combinable: intersections.includes(node.id),
            };
          }
          if (node.type === "text" && draggedNode.type === "text") {
            return {
              ...node,
              data: {
                ...node.data,
                combinable: intersections.includes(node.id),
              },
            };
          }
          if (node.type === "synthesizer") {
            return {
              ...node,
              data: {
                ...node.data,
                mode: intersections.includes(node.id) ? "dragging" : "ready",
              },
            };
          }
          return node;
        })
      );
    },
    [setNodes, getIntersectingNodes]
  );


  const onNodeDragStop = useCallback(
  (_: MouseEvent, draggedNode: Node) => {
    const intersections = getIntersectingNodes(draggedNode).map((n) => n.id);

    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        /**** 2 text Nodes have been dragged together! ****/
        if (node.type === "text" && draggedNode.type === "text" && intersections.includes(node.id)) {
          const draggedTextNode = draggedNode as Node<TextNodeData>;
          const textNode = node as Node<TextNodeData>;

          // Combine text and create a new text node
          addTextNode(textNode.data.content + ", " + draggedTextNode.data.content, { 
                x: textNode.position.x + (Math.random() * 10 - 5), 
                y: textNode.position.y - 10,
                });

          deleteNodeById(draggedNode.id);
          deleteNodeById(node.id);
          
          return { ...node, data: { ...node.data, combinable: true } };
        } 
        
        if (node.type === "text") {
          return { ...node, data: { ...node.data, combinable: false } };
        }

        /* --- If a node has been dragged on top of a synthesizer --- */
        if (node.type === "synthesizer" && intersections.includes(node.id)) {
              console.log(`Node ${draggedNode.id} dragged on top of synthesizer, with data: ${JSON.stringify(draggedNode.data)}`);
              const inputNodeContent = ("words" in draggedNode.data
                ? Array.isArray(draggedNode.data.words) ? draggedNode.data.words.map((word: { value: string }) => word.value).join(' ') : ""
                : "content" in draggedNode.data
                ? draggedNode.data.content
                : "label" in draggedNode.data
                ? draggedNode.data.label
                : "No content") as string;  // Determine the mode based on inputNodeContent
              const isValidImage = /\.(jpeg|jpg|gif|png|webp)$/.test(inputNodeContent);
              const mode = isValidImage ? "generating-image" : "generating-text";
              // Generate a new node with the inputNodeContent
              generateNode(inputNodeContent, calcNearbyPosition(getNodesBounds([node, draggedNode])));  // Set synthesizer node to the appropriate mode
              // Move the draggedNode out of the way, back to where it was before the drag 
              updatePosition(draggedNode.id, calcNearbyPosition(getNodesBounds([draggedNode]), dragStartPosition));
              return { 
              ...node, 
              data: { ...node.data, mode, inputNodeContent }, 
              className: "" // Reset highlight
              };  
            }
        return { ...node, className: "" };
      })
    );
  },
  [setNodes, getIntersectingNodes, addTextNode]
);



  // ------------------- HELPER FUNCTIONS ------------------- //
  const deleteNodeById = (nodeId: string) => {  
    setNodes((currentNodes) => currentNodes.filter((node) => node.id !== nodeId));
    quickSaveToBrowser(toObject(), canvasID); //everytime a node is changed, save it to the browser storage
  };

  const updatePosition = (nodeId: string, newPosition: { x: number; y: number }) => {
    /* takes a node and a new position as input, moves the node there */
    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        if (node.id === nodeId) {
          if (node.position.x !== newPosition.x || node.position.y !== newPosition.y) {
            return {
              ...node,
              position: newPosition,
              className: `${node.className} node-transition`, // Add transition class
            };
          }
        }
        return node;
      })
    );
    // Remove the transition class after the animation completes
    setTimeout(() => {
      setNodes((currentNodes) =>
        currentNodes.map((node) => {
          if (node.id === nodeId) {
            return {
              ...node,
              className: node.className ? node.className.replace(" node-transition", "") : "",
            };
          }
          return node;
        })
      );
    }, 500); // Match the duration of the CSS transition
  };





// ------------------ GENERATE NODE FUNCTION ------------------ //
  const generateNode = useCallback(
    // Requests through reagent, generates an image node with a prompt and optional position
    async (prompt: string = "bunny on the moon", position: { x: number; y: number } = { x: 250, y: 250 }) => {

      // True if the "prompt" is actually an image
      const isValidImage = /\.(jpeg|jpg|gif|png|webp)$/.test(prompt);

      const loadingNodeId = `loading-${Date.now()}`;
      const loadingNode: AppNode = {
        id: loadingNodeId,
        type: "text",
        position,
        data: { content: "loading ", loading: true, combinable: false } as TextNodeData,
      };
      setNodes((nodes) => [...nodes, loadingNode]);

      // let's generate a node... 

      try {
        if (isValidImage){// the user has sent an image for text generation
          //console.log(`Describing this image: ${prompt}`);
          const response = await axios.post(`${backend}/api/generate-text`, {
            imageUrl: prompt, // Send the imageUrl as part of the request body
          });

          if (response.status === 200) {
            addTextWithKeywordsNode(response.data.text, position);
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

              addImageWithLookupNode(response.data.imageUrl, position, prompt);
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
          onNodeDragStart={onNodeDragStart}
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

      < Toolbar addTextWithKeywordsNode={addTextWithKeywordsNode} addImageNode={addImageWithLookupNode} addSynthesizer={addSynthesizer} />




      {/* Debug Info */}
          <div className="fixed bottom-0 left-0 bg-gray-900 text-white p-4 z-50 text-sm rounded-md shadow-md">
          <button
            onClick={() => setShowDebugInfo((prev) => !prev)}
            className="bg-blue-500 text-white p-2 rounded mb-2"
          >
          {showDebugInfo ? "Hide Debug Info" : "Show Debug Info"}
          </button>
          {showDebugInfo && (
           <div> //debug info for debugging user state
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
