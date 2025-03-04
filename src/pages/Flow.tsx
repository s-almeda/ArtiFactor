

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
  useOnViewportChange,
  useReactFlow,
  applyNodeChanges,
} from "@xyflow/react";
import {useSearchParams} from "react-router-dom";
import { defaultTextWithKeywordsNodeData } from "../nodes";



import "@xyflow/react/dist/style.css";
import type { AppNode, LoadingNode, ImageWithLookupNodeData, TextWithKeywordsNodeData} from "../nodes/types";
import {wordsToString } from '../utils/utilityFunctions';
import { nodeTypes } from "../nodes";
import useClipboard from "../hooks/useClipboard";
import { useDnD } from '../context/DnDContext';
import { calcNearbyPosition } from '../utils/utilityFunctions';
import { useNodeContext } from "../context/NodeContext";
import { useAppContext } from '../context/AppContext';
import { useCanvasContext } from '../context/CanvasContext';
import Toolbar from "../Toolbar";
// import { data } from "react-router-dom";
//FYI: we now set the backend in App.tsx!

const Flow = () => {
  const { userID, backend, loginStatus } = useAppContext();
  const { canvasName, canvasID, setCanvasId, pullCanvas, saveCanvas, quickSaveToBrowser, pullCanvasFromBrowser, setCanvasName, setLastSaved, createNewCanvas } = useCanvasContext();  //setCanvasName//the nodes as saved to the context and database
  const { nodes, setNodes, saveCurrentViewport } = useNodeContext(); //useNodesState(initialNodes);   //the nodes as being rendered in the Flow Canvas
  const { toObject, getIntersectingNodes, screenToFlowPosition, setViewport, getViewport, getNodesBounds } = useReactFlow();
  const [draggableType, setDraggableType, draggableData, setDraggableData] = useDnD(); //dragStartPosition, setDragStartPosition

  const [attemptedQuickLoad, setattemptedQuickLoad] = useState(false);

  const [___, setSynthesisMode] = useState(false);


  //const location = useLocation();
  const [searchParams] = useSearchParams();
  const userParam = searchParams.get('user');
  const canvasParam = searchParams.get('canvas');

  const checkCanvasParam = async (userParam: string | null, canvasParam: string | null) => {
    if (!canvasParam){ //does the url requested canvas exist
        return false;
    }
    else if (userParam && canvasParam && !canvasParam.includes(userParam)) { //is the url requested canvas formatted correctly
      window.location.href = `/?userId=${userParam}&canvasId=${userParam}-${canvasParam}`;
      return false;
    }
    else// check if it exists in the database
    {
        const response = await fetch(`${backend}/api/list-canvases/${userParam}`);
        const data = await response.json();
        if (!data.success || !data.canvases.includes(canvasParam)) { 
            console.error(`${canvasParam} not found for the user ` + userParam + "in" + data.canvases);
            return false;
        }
    }
    //ok we're good
    return true;
  };

  //our custom nodes change function called everytime a node changes, even a little
  const handleOnNodesChange = useCallback(
    (changes: any) => {
      setNodes((nds) => applyNodeChanges(changes, nds));
      quickSaveToBrowser(toObject()); //everytime a node is changed, save it to the browser storage
      if (userParam && canvasParam && loginStatus  === "logged in") { //if we have a user and canvas id set in the url,
        console.log("flow is saving canvas to the database");
        saveCanvas(toObject(), canvasID, canvasName); //everytime a node is changed, save to the database
      }
    },
      [setNodes, quickSaveToBrowser, saveCanvas]
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

useOnViewportChange({
  onEnd: () => saveCurrentViewport(getViewport()),
});


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

  const addTextWithKeywordsNode = (
    content: string = "your text here",
    provenance: "user" | "history" | "ai" = "user",
    position?: { x: number; y: number },
    hasNoKeywords: boolean = false
  ) => {
    const data: TextWithKeywordsNodeData = content === "your text here" && provenance === "user" && !position && !hasNoKeywords
      ? defaultTextWithKeywordsNodeData
      : {
          words: content.split(' ').map((word) => ({ value: word })),
          provenance,
          content,
          intersections: [],
          hasNoKeywords,
        };

    const newTextWithKeywordsNode: AppNode = {
      id: `text-${Date.now()}`,
      type: "text",
      zIndex: 1000,
      position: position ?? {
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
      ///console.log("LOADING NODE:"+ loadingNode.zIndex);
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


  // ------------------------ LOAD UP THE PROPER NODES UPON LOGIN EFFECT ------------------------ //


  useEffect(() => {
    if (attemptedQuickLoad) return;

    if (loginStatus === "logging in") return;

    const fetchData = async () => {
      if (loginStatus === "logged out") {
        console.log("USER IS LOGGED OUT")
        setCanvasId("browser");
        const browserCanvas = pullCanvasFromBrowser("browser");
        if (browserCanvas) {
          console.log("Canvas loaded from browser storage!: ",  browserCanvas);
          const { nodes = [], viewport = { x: 0, y: 0, zoom: 1 } } = browserCanvas;
          setNodes(nodes);
          setViewport(viewport);
        } else {
          console.log("No canvas found in browser storage. Creating a new one for the logged out user...");
          setNodes([]);
          addTextWithKeywordsNode("your text here", "user", { x: 0, y: 0 });
          setViewport({ x: 0, y: 0, zoom: 1 });
          quickSaveToBrowser(toObject(), "browser");
        }
        setattemptedQuickLoad(true);
        return;
      }


      if (loginStatus === "logged in" && userID) {
        //log in the canvasParam in the url
        if (canvasParam) {
          setCanvasId(canvasParam);
          
          // const browserCanvas = pullCanvasFromBrowser(canvasParam);
          // if (browserCanvas) {
          //   console.log("Canvas loaded from browser storage!: ", browserCanvas);
          //   const { nodes = [], viewport = { x: 0, y: 0, zoom: 1 } } = browserCanvas;
          //   setNodes(nodes);
          //   setViewport(viewport);
          //   const storedCanvasName = localStorage.getItem(`${canvasParam}-name`);
          //   if (storedCanvasName) {
          //     setCanvasName(storedCanvasName);
          //   }
          //   saveCanvas(browserCanvas, canvasParam);
          //   setattemptedQuickLoad(true);
          //   return;
          // }

          pullCanvas(`${canvasParam}`).then((savedCanvas: any) => {
            if (savedCanvas) {
              console.log("URL requested canvas found in the database!");
              const { nodes = [], viewport = { x: 0, y: 0, zoom: 1 }, name, timestamp } = savedCanvas as { nodes: Node[], viewport: { x: number, y: number, zoom: number }, name: string, timestamp: string };
              setNodes(nodes);
              setViewport(viewport);
              setCanvasName(name);
              setLastSaved(timestamp);
              setattemptedQuickLoad(true);
              return
            }
          });
        }
        // no canvas param, or canvas param didn't work. let's find a valid canvas param.
        const response = await fetch(`${backend}/api/list-canvases/${userID}`);
        const data = await response.json();
        if (!canvasParam && data.success && data.canvases.length > 0) {
          const lastCanvas = data.canvases[data.canvases.length - 1].id;
          console.log("redirecting to the last canvas: ", lastCanvas);
          //window.location.href = `/?user=${userID}&canvas=${lastCanvas}`;
          setattemptedQuickLoad(true);

          window.location.href = `/?user=${userID}&canvas=${lastCanvas}`;
        } 
        else {
          createNewCanvas(userID);
        }
        setattemptedQuickLoad(true);
        
      }
    };

    fetchData();
  }, [userID, loginStatus, attemptedQuickLoad, canvasParam, backend, checkCanvasParam, pullCanvas, setCanvasId, setNodes, setViewport, setCanvasName, setLastSaved, quickSaveToBrowser, toObject, addTextWithKeywordsNode, createNewCanvas, pullCanvasFromBrowser]);



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

      <Toolbar addTextWithKeywordsNode={() => addTextWithKeywordsNode()} addImageNode={addImageWithLookupNode} addSynthesizer={() => setSynthesisMode(true)} />

      <div className="fixed bottom-0 left-0 bg-gray-900 text-white p-4 z-50 text-sm rounded-md shadow-md overflow-scroll max-h-64">
        <button
          onClick={() => setShowDebugInfo((prev) => !prev)}
          className="bg-blue-500 text-white p-2 rounded mb-2"
        >
          {showDebugInfo ? "Hide Debug Info" : "Show Debug Info"}
        </button>
        {showDebugInfo && (
          <div className="overflow-scroll max-h-48" onClick={() => setShowDebugInfo(false)}>
            <p><strong>User ID:</strong> {userID}</p>
            <p><strong>Canvas Name:</strong> {canvasName}</p>
            <p><strong>Canvas ID:</strong> {canvasID}</p>
            <p><strong>Flow Nodes:</strong> {JSON.stringify(nodes, null, 2)}</p>
            <p><strong>CanvasData Currently Stored in Context:</strong> {JSON.stringify(pullCanvas, null, 2)}</p>
          </div>
        )}
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
    // const handlepullCanvas = useCallback(async () => {
    //   console.log("loading canvas data for: ", canvasID);
    //   const canvasData = await pullCanvas(canvasID);
    //   console.log("Loaded Canvas Data:", canvasData); // Print to console for debugging
    //   if (canvasData !== undefined) {
    //     const { nodes = [], viewport = { x: 0, y: 0, zoom: 1 } } = canvasData;
    //     setNodes(nodes);
    //     setViewport(viewport);
    //   }
    // }, [canvasID, pullCanvas, setNodes, setViewport]);
