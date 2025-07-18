

import { useCallback, useEffect, useState, type MouseEvent } from "react";
import axios from "axios";
import {
  ReactFlow,
  Background,
  Controls,
  //MiniMap,
  ReactFlowJsonObject,
  Node,
  // // useNodesState,
  // Edge,
  // addEdge,
  useOnViewportChange,
  useReactFlow,

} from "@xyflow/react";
import {useSearchParams} from "react-router-dom";
import { defaultTextWithKeywordsNodeData } from "../nodes";
import { v4 as uuidv4 } from "uuid";



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
  const { saveCanvas, canvasName, canvasID, setCanvasId, pullCanvas, quickSaveToBrowser, pullCanvasFromBrowser, setCanvasName, setLastSaved, createNewCanvas } = useCanvasContext();  //setCanvasName//the nodes as saved to the context and database
  const { nodes, setNodes, edges, setEdges, drawEdge, saveCurrentViewport, canvasToObject, handleOnEdgesChange, handleOnNodesChange, onNodesDelete, deleteNodeById } = useNodeContext(); //useNodesState(initialNodes);   //the nodes as being rendered in the Flow Canvas
  const { screenToFlowPosition, setViewport, getViewport, getNodesBounds, getIntersectingNodes, getNode } = useReactFlow();
  const { draggableType, draggableData, setDraggableData, setDraggableType} = useDnD(); //dragStartPosition, setDragStartPosition

  const [attemptedQuickLoad, setattemptedQuickLoad] = useState(false);

  const [___, setSynthesisMode] = useState(false);

  const [ repeatedCreationOffset, setRepeatedCreationOffset] = useState(0);


  //const location = useLocation();
  const [searchParams] = useSearchParams();
  //const userParam = searchParams.get('user');
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


  const onNodeDrag = useCallback(
    (_: MouseEvent, draggedNode: Node) => {
      setDraggableType(draggedNode.type as string);
      setDraggableData(draggedNode.data);
      if (draggedNode.type != "default") {
        updateIntersections(draggedNode, nodes);
      }
    },
      [setNodes, getIntersectingNodes]
    );
    

  const onNodeDragStop = useCallback(
    (_: MouseEvent, draggedNode: Node) => {
      setNodes((currentNodes: AppNode[]) => updateIntersections(draggedNode, currentNodes));
    },
    [setNodes, userID]
  );



  // ------------------- HELPER FUNCTIONS ------------------- //

  const updateIntersections = (draggedNode: Node, currentNodes: AppNode[]) => {
    const intersections = getIntersectingNodes(draggedNode)
      .filter((n) => n.type === draggedNode.type) // Only consider nodes of the same type
      .map((n) => n.id);

    return currentNodes.map((node: AppNode) => {
      if (node.id === draggedNode.id) {
        const updatedIntersections = [
          {
            id: node.id,
            position: node.position,
            content: node.type === "image" ? (node.data as ImageWithLookupNodeData).prompt : node.data.content,
          },
          ...intersections.map((id) => {
            const intersectingNode = currentNodes.find((n) => n.id === id);
            if (intersectingNode) {
              return {
                id: intersectingNode.id,
                position: intersectingNode.position,
                content: intersectingNode.type === "image" ? (intersectingNode.data as ImageWithLookupNodeData).prompt : intersectingNode.data.content,
              };
            }
            return null;
          }).filter(Boolean),
        ];
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


  const onNodeContextMenu = useCallback(
    (event: MouseEvent, node: Node) => {
      event.preventDefault();
      if (node.data.type === "default"){ //don't generate the loading node
        return
      }
      else if (node.data.content && node.data.content!= "loading ") {
        generateNode(node.id, node.data.content as string, calcNearbyPosition(getNodesBounds([node])));
      } 
      else if (Array.isArray(node.data.words)) {
        const content = wordsToString(node.data.words);
        generateNode(node.id, content, calcNearbyPosition(getNodesBounds([node])));
      }  
      console.log("you right clicked this node:", node.data);
    
    },[]);
  



  const handleNodeClick = useCallback(
  (event: MouseEvent, node: Node) => {
    //console.log(event.button)
    
    if (event.altKey) { 
      if (node.data.type === "default"){ //don't generate the loading node
        return
      }
      else if (node.data.content && node.data.content!= "loading ") {
        generateNode(node.id, node.data.content as string, calcNearbyPosition(getNodesBounds([node])));
      } 
      else if (Array.isArray(node.data.words)) {
        const content = wordsToString(node.data.words);
        generateNode(node.id, content, calcNearbyPosition(getNodesBounds([node])));
      }  
      console.log("you option clicked this node:", node.data);
    }
  },[]);

useOnViewportChange({
  onStart: () => setRepeatedCreationOffset(0),
  onEnd: () => {
    const viewport = getViewport();
    saveCurrentViewport(viewport);
  },
});


  /* ---------------------------------------------------- */
  // TODO - move this to a KeyboardShortcut Provider Context situation so we cna also track Undos/Redos
  const { handleCopy, handleCut, handlePaste } = useClipboard(nodes); // Use the custom hook

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
    content: string = "your text here. click the pencil icon in the top right to edit. Use Alt+Click to generate new nodes.",
    provenance: "user" | "history" | "ai" = "user",
    position?: { x: number; y: number },
    hasNoKeywords: boolean = false,
    parentNodeId?: string,
    similarTexts?: any[],
  ) => {
    let finalPosition = position;

    // Check if this is a default text node and no position is provided
    const isDefaultNode =
      // content === "your text here" &&
      provenance === "user" &&
      !position &&
      !hasNoKeywords;

    if (isDefaultNode) {
      // Get the center of the top-left quadrant of the screen
      const boundingRect = {
        x: window.innerWidth / 4,
        y: window.innerHeight / 4,
      };

      const center = screenToFlowPosition({
        x: boundingRect.x,
        y: boundingRect.y,
      });

      finalPosition = {
        x: center.x + repeatedCreationOffset * 15 + Math.random() * 50 - 25,
        y: center.y + repeatedCreationOffset * 20,
      };

      setRepeatedCreationOffset(repeatedCreationOffset + 1);
      console.log("making a default text node");
    }

    const data: TextWithKeywordsNodeData = isDefaultNode
      ? defaultTextWithKeywordsNodeData
      : {
          words: content.split(" ").map((word) => ({ value: word })),
          provenance,
          content,
          intersections: [],
          similarTexts: similarTexts ?? [],
          hasNoKeywords,
        };

    const newTextWithKeywordsNode: AppNode = {
      id: `text-${uuidv4()}`,
      type: "text",
      zIndex: 1000,
      position: finalPosition ?? { x: 250, y: 250 },
      data: data,
    };

    console.log("put the new text node at: ", finalPosition);

    setNodes((prevNodes) => {
      const updatedNodes = [...prevNodes, newTextWithKeywordsNode];
      if (parentNodeId) {
        drawEdge(parentNodeId, newTextWithKeywordsNode.id, updatedNodes);
      }
      return updatedNodes;
    });
  };



  // // Placeholder function to create an edge between two specific nodes
  // const createPlaceholderEdge = () => {
  //   const placeholderNodes = [
  //     { id: 'image-1740983392180', type: 'image', position: { x: 100, y: 100 }, zIndex: 1000, data: { content: 'Image 1' } },
  //     { id: 'image-1740983412123', type: 'image', position: { x: 200, y: 200 }, zIndex: 1000, data: { content: 'Image 2' } }
  //   ];

  //   drawEdge('image-1740983392180', 'image-1740983412123', placeholderNodes);
  // };






  const addImageWithLookupNode = (content?: string, position?: { x: number; y: number }, prompt?:string, provenance?: string, parentNodeId?: string, similarArtworks?: any[]) => {
    content = content ?? "https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png";
    prompt = prompt ?? "default placeholder image. try creating something of your own!";
    provenance = provenance ?? "user";
    similarArtworks = similarArtworks && similarArtworks.length > 0 ? similarArtworks : undefined;  
    //console.log("addImageWithLookupNode is adding an image to the canvas: ", content, prompt, provenance, parentNodeId);
    position = position ?? { 
      x: Math.random() * 250,
      y: Math.random() * 250,
    };
    const newImageNodeId = `image-${uuidv4()}`
    
    const newImageNode: AppNode = {
      id: newImageNodeId,
      type: "image",
      position: position,
      zIndex: 1000,
      data: {
      content: content,
      prompt: prompt,
      provenance: provenance,
      parentNodeId: parentNodeId,
      similarArtworks: similarArtworks,
      intersections: [],
      } as ImageWithLookupNodeData,
      dragHandle: '.drag-handle__invisible',

    };


    setNodes((prevNodes) => {
      const updatedNodes = [...prevNodes, newImageNode];
      if (parentNodeId){
      drawEdge(parentNodeId, newImageNodeId, updatedNodes);
      }
      return updatedNodes;
    });
  };



  /*------------ functions to handle changes to the Flow canvas ---------------*/


  /* -- when something else is dragged over the canvas -- */
  const onDragOver = useCallback((event: { preventDefault: () => void; clientX: any; clientY: any; dataTransfer: { dropEffect: string; }; }) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
    //console.log("dragging over the canvas at: ", position);

  }, []);

  
 
  const onDrop = useCallback(
    (event: { preventDefault: () => void; clientX: any; clientY: any; }) => {
      event.preventDefault();
            //console.log("the parent for the node you just dropped has this id: ", parentNodeId);
      if (!draggableType) {
        return;
      }
      const position = screenToFlowPosition({
        x: event.clientX - 60,
        y: event.clientY - 60,
      });

      console.log(`you just dropped a: ${JSON.stringify(draggableType)} in this location: ${JSON.stringify(position)} with this content: ${JSON.stringify(draggableData)}`);  // check if the dropped element is valid


      if (draggableType === "image") {
        const content = "content" in draggableData ? draggableData["content"] as string : "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/660px-No-Image-Placeholder.svg.png?20200912122019";
        const prompt = "prompt" in draggableData ? draggableData["prompt"] as string : "";
        const provenance = "provenance" in draggableData ? draggableData["provenance"] as "user" | "history" | "ai" : "user";
        const parentNodeId = "parentNodeId" in draggableData ? draggableData["parentNodeId"] as string : undefined;
        const similarArtworks = "similarArtworks" in draggableData ? draggableData["similarArtworks"] as any[] : [];
        addImageWithLookupNode(content, position, prompt, provenance, parentNodeId, similarArtworks);

      } else if ("content" in draggableData) {
       const provenance = "provenance" in draggableData ? draggableData["provenance"] as "user" | "history" | "ai" : "user";
       const parentNodeId = "parentNodeId" in draggableData ? draggableData["parentNodeId"] as string : undefined; 
       const similarTexts = "similarTexts" in draggableData ? draggableData["similarTexts"] as any[] : [];
       addTextWithKeywordsNode(draggableData["content"] as string, provenance, position, false, parentNodeId, similarTexts);
      }
    
      saveCanvas(canvasToObject(), canvasID, canvasName);
    },
    [draggableType, draggableData,screenToFlowPosition, loginStatus],
  );


  // const handleNodeDelete = useCallback(
  //   (node: Node) => {
  //     console.log("deleting node: ", node);
  //     setEdges((currentEdges) => currentEdges.filter((edge) => edge.source !== node.id && edge.target !== node.id));
  //     setNodes((currentNodes) => currentNodes.filter((n) => n.id !== node.id));
  //     if (userID){

  //       saveCanvas(canvasToObject(), canvasID, canvasName);
  //     }
  //   }
  //   ,[setNodes, setEdges, saveCanvas, canvasToObject, userID, canvasID, canvasName]
  // );

  /* -------------------------------- NODE DRAGGING -------------------------*/


  //when someone starts dragging, send the starting position to the DnD Context
  // const onNodeDragStart = useCallback(
  //   (_: MouseEvent, node: Node) => {
  //     setDragStartPosition({ x: node.position.x, y: node.position.y });
  //   },
  //   []
  // );

  //keep track o fhte node dragging 





  // ------------------- HELPER FUNCTIONS ------------------- //



// ------------------ GENERATE NODE FUNCTION ------------------ //
  const generateNode = useCallback(
    //todo, add parentnodeid for edges
    // Requests through reagent, generates an image node with a prompt and optional position
    async (parentNodeId: string, prompt: string = "bunny on the moon", position: { x: number; y: number } = { x: 250, y: 250 }) => {
      
      // True if the "prompt" is actually an image
      const isValidImage = /\.(jpeg|jpg|gif|png|webp)$/.test(prompt);

      const loadingNodeId = `loading-${uuidv4()}`;
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
            const updatedPosition = getNode(loadingNodeId)?.position;
            addTextWithKeywordsNode(response.data.text, "ai", updatedPosition, false, parentNodeId, []);  
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
              const updatedPosition = getNode(loadingNodeId)?.position;
              addImageWithLookupNode(response.data.imageUrl, updatedPosition||loadingNode.position, prompt, "ai", parentNodeId, []);
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

  //const [showDebugInfo, setShowDebugInfo] = useState(false);


  // ------------------------ LOAD UP THE PROPER NODES UPON LOGIN EFFECT ------------------------ //


  useEffect(() => {

    if (attemptedQuickLoad) return; //we've already tried loading

    if (loginStatus === "logging in") return; //we are still logging in

    const fetchData = async () => {
      if (loginStatus === "logged out") { //we are logged out; use the browser
        console.log("USER IS LOGGED OUT")
        setCanvasId("browser");
        const browserCanvas = pullCanvasFromBrowser("browser");
        if (browserCanvas) {
          console.log("Canvas loaded from browser storage!: ",  browserCanvas);
          const { nodes = [], edges = [], viewport = { x: 0, y: 0, zoom: 1 } } = browserCanvas;
          setNodes(nodes);
          setEdges(edges);
          setViewport(viewport);
        } else {
          console.log("No canvas found in browser storage. Loading default canvas data...");
          // Fetch the default canvas data from the public folder
          fetch("/default_canvas_data.json")
            .then((res) => res.json())
            .then((defaultCanvas) => {
              const { nodes = [], edges = [], viewport = { x: 0, y: 0, zoom: 1 } } = defaultCanvas;
              setNodes(nodes);
              setEdges(edges);
              setViewport(viewport);
              quickSaveToBrowser(canvasToObject(), "browser", "browserCanvas");
            })
            .catch((err) => {
              console.error("Failed to load default canvas data:", err);
              // fallback: create a single default node
              setNodes([]);
              setEdges([]);
              addTextWithKeywordsNode("your text here. click the pencil icon in the top right to edit. Right-Click, or use Alt+Click to generate new nodes.", "user", { x: 0, y: 0 });
              setViewport({ x: 0, y: 0, zoom: 1 });
              quickSaveToBrowser(canvasToObject(), "browser", "browserCanvas");
            });
        }  setattemptedQuickLoad(true);
        return;
      }
      if (loginStatus === "logged in" && userID) { //were logged in!
        // If a canvasParam exists in the URL

        if (canvasParam) {
          setCanvasId(canvasParam);
          // Pull the canvas data from the API
          pullCanvas(`${canvasParam}`).then((savedCanvas: { canvasData: ReactFlowJsonObject, canvasName: string, timestamp: string } | null) => {
            if (savedCanvas !== null) {
              console.log("URL requested canvas found in the database!");
              console.log(savedCanvas);
          
              // Destructure and set canvas data
              const { nodes = [], edges = [], viewport = { x: 0, y: 0, zoom: 1 } } = savedCanvas.canvasData;
              
              console.log ("flow has received: ", savedCanvas);
              // Update state with canvas data
              setNodes(nodes);
              setEdges(edges);
              setViewport(viewport);
              setCanvasName(savedCanvas.canvasName);  // Make sure to use canvasName here
              setLastSaved(savedCanvas.timestamp);
              setattemptedQuickLoad(true);
              return;
            } else {
              console.log("No canvas found for the given ID.");
            }
          });
        }  // If no canvasParam or canvasParam didn't work, find a valid canvas param
        else {
          const response = await fetch(`${backend}/api/list-canvases/${userID}`);
          const data = await response.json();
      
          if (data.success && data.canvases.length > 0) {
            const lastCanvas = data.canvases[data.canvases.length - 1].canvasId;
            console.log("Redirecting to the last canvas:", lastCanvas);
      
            // Redirect to the last canvas
            setattemptedQuickLoad(true);
            window.location.href = `/?user=${userID}&canvas=${lastCanvas}`;
          } else if (data.success && data.canvases.length === 0) {
            console.log("No canvases found, creating a new one.");
            createNewCanvas(userID); // Trigger the function to create a new canvas
          }
      
          setattemptedQuickLoad(true);
        }
      }      
    };

    fetchData();

  }, [userID, loginStatus, attemptedQuickLoad, canvasParam, backend, checkCanvasParam, pullCanvas, setCanvasId, setNodes, setViewport, setCanvasName, setLastSaved, quickSaveToBrowser, canvasToObject, addTextWithKeywordsNode, createNewCanvas, pullCanvasFromBrowser]);

useEffect(() => {
  const interval = setInterval(() => {
    if (userID) {
      saveCanvas(canvasToObject());
    }
    else{
      quickSaveToBrowser(canvasToObject(), "browser", "browserCanvas");
    }
  }, 60000); // 60 seconds

  return () => clearInterval(interval); // Cleanup on unmount
}, [userID, canvasID, canvasName, quickSaveToBrowser, saveCanvas, canvasToObject]);

return(
      <>



            
      <div style={{ width: '100%', height: '100%' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          minZoom={0.009}
          onNodesChange={handleOnNodesChange}
          onEdgesChange={handleOnEdgesChange}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
          onNodesDelete={onNodesDelete}
          onNodeContextMenu={onNodeContextMenu}
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

      {/* <div className="fixed bottom-0 left-0 bg-gray-900 text-white p-4 z-50 text-sm rounded-md shadow-md overflow-scroll max-h-64">
        <button
          onClick={() => setShowDebugInfo((prev) => !prev)}
          className="bg-blue-500 text-white p-2 rounded mb-2"
        >
          {showDebugInfo ? "Hide Debug Info" : "Show Debug Info"}
        </button>
        {showDebugInfo && (

          <div className="overflow-scroll max-h-48" onClick={() => setShowDebugInfo(false)}>
              <button
            onClick={() => console.log(canvasToObject())}
            className="bg-green-500 text-white p-2 rounded mb-2 ml-2"
          >
            Print Canvas to Console
          </button>
            <p><strong>User ID:</strong> {userID}</p>
            <p><strong>Canvas Name:</strong> {canvasName}</p>
            <p><strong>Canvas ID:</strong> {canvasID}</p>
            <p><strong>Login Status: </strong> {loginStatus}</p>
            </div>
        )}
      </div> */}


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
