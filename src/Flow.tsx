

import { useCallback, useEffect, useState, type MouseEvent } from "react";
import axios from "axios";
import {
  ReactFlow,
  Background,
  Controls,
  //MiniMap,
  Node,
  applyNodeChanges,
  useNodesState,
  useReactFlow,
} from "@xyflow/react";

import "@xyflow/react/dist/style.css";
import type { AppNode, Artwork, ImageNodeData, TextNodeData, LookupNode, T2IGeneratorNodeData} from "./nodes/types";
import { initialNodes, nodeTypes } from "./nodes";
import useClipboard from "./hooks/useClipboard";
import { useDnD } from './context/DnDContext';
import { useCanvasContext } from './context/CanvasContext';
import { calcPositionAfterDrag } from './utils/calcPositionAfterDrag';
//import { addTextNode, addImageNode, addT2IGenerator } from './utils/nodeAdders';
import { useAppContext } from './context/AppContext';

//we now set the backend in App.tsx and grab it here!
const Flow = () => {
  const { userID, backend, loadCanvasRequest, setLoadCanvasRequest} = useAppContext();
  const { canvasName, canvasId, savedNodes, saveNodes, loadCanvas, saveCanvas, setCanvasName } = useCanvasContext();  //the nodes as saved to the context and database
  const [flowNodes, setFlowNodes] = useNodesState(initialNodes);   //the nodes as being rendered in the Flow Canvas
  const { getIntersectingNodes, screenToFlowPosition } = useReactFlow();
  const [draggableType, __, draggableData, _] = useDnD();
  const [saveCanvasRequest, setSaveCanvasRequest] = useState(false);

  // sync and reload the flow canvas when "Load" is triggered
  useEffect(() => {

    if (loadCanvasRequest) { //the App has requested the Flow to reload entirely- the context should already hold the new set of nodes
      setFlowNodes(savedNodes); //overwrite the Flow Canvas to match what's in the context
      setLoadCanvasRequest(false); //turn the App flag off!
    }
    else if(saveCanvasRequest){
      saveNodes(flowNodes); //save the current state of the Flow canvas to the context
    }
  }, [loadCanvasRequest, savedNodes]);

    /* 
    ================================================== 
    ||            Saving and Loading...             || 
    =======================================-========== 
     */
  // manually trigger the load and save functions
  const handleLoadCanvas = async () => {
    const canvasID = prompt("Enter the Canvas ID to load:");
    if (canvasID) {
      await loadCanvas(canvasID);
      setLoadCanvasRequest(true); 
    }
  };

  const handleSaveCanvas = async () => {
    const newCanvasName = prompt("Enter the Canvas Name to save as:", canvasName);
    if (newCanvasName) {
      await saveNodes(flowNodes); // Update the context with the current flow nodes
      saveCanvas(newCanvasName); // Trigger the database save with the new canvas name
    }
    else{
      saveCanvas(canvasName);
    }
  };

  const onNodesChange = useCallback(
    (changes: any) => {
      setFlowNodes((prevNodes) => {
        const updatedNodes = applyNodeChanges(changes, prevNodes);
        setSaveCanvasRequest(true); // Trigger the save request
        return updatedNodes;
      });

    },[]
  );

  /* ---------------------------------------------------- */
  // TODO - move this to a KeyboardShortcut Provider Context situation so we cna also track Undos/Redos
  const { handleCopy, handleCut, handlePaste } = useClipboard(flowNodes, setFlowNodes); // Use the custom hook

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


  const onNodeDrag = useCallback(
    (_: MouseEvent, draggedNode: Node) => {
      setFlowNodes((currentNodes: AppNode[]) =>
        currentNodes.map((node: AppNode) => {
          if (node.id === draggedNode.id) {
            return {
              ...node,
              position: draggedNode.position,
            };
          }
          return node;
        })
      );
      const intersections = getIntersectingNodes(draggedNode).map((n) => n.id);
      handleIntersectionsOnDrag(draggedNode, intersections);
    },
    [setFlowNodes, getIntersectingNodes]
  );

  /* -- when something else is dragged over the canvas -- */
  const onDragOver = useCallback((event: { preventDefault: () => void; dataTransfer: { dropEffect: string; }; }) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  
 
  const onDrop = useCallback(
    (event: { preventDefault: () => void; clientX: any; clientY: any; }) => {
      event.preventDefault();
      console.log(`you just dropped: ${draggableType} and ${draggableData}`);
      // check if the dropped element is valid
      if (!draggableType) {
        return;
      }
 
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      if (draggableType === "image" && "content" in draggableData && "prompt" in draggableData) {
        addImageNode(draggableData["content"] as string, position, draggableData["prompt"] as string);
      } else if ("content" in draggableData) {
        addTextNode(draggableData["content"] as string, position);
      }
      
    },
    [draggableType, draggableData,screenToFlowPosition],
    
  );

  const onNodeDragStop = useCallback(
    (_: MouseEvent, draggedNode: Node) => {
      //console.log("user stopped dragging a node ", draggedNode)
      const intersections = getIntersectingNodes(draggedNode).map((n) => n.id);
      handleIntersectionsOnDrop(draggedNode, intersections);
    },
    [getIntersectingNodes, setFlowNodes]
  );


  // --- HELPER FUNCTIONS --- //

   /*** ---- this is the code that makes stuff fly away lol --- */


  const updatePosition = (nodeId: string, newPosition: { x: number; y: number }) => {
    /* takes a node and a new position as input, moves the node there */
    setFlowNodes((currentNodes) =>
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
      setFlowNodes((currentNodes) =>
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
    }, 1000); // Match the duration of the CSS transition
  };

  
const handleIntersectionsOnDrag = (draggedNode: Node, intersections: string[]) => {
  setFlowNodes((currentNodes) => {
    const updatedNodes = currentNodes.map((node) => {
      /*--- If the draggedNode and the node it overlaps with are BOTH text nodes ---*/
      if (node.type === "text" && draggedNode.type === "text" && intersections.includes(node.id)) {
        const draggedTextNode = draggedNode as Node<TextNodeData>;
        const textNode = node as Node<TextNodeData>;
        draggedTextNode.data.combinable = true;
        textNode.data.combinable = true;
      } 
      else if (node.type === "text") {
        const textNode = node as Node<TextNodeData>;
        textNode.data.combinable = false;
      }
      /*this node is a generator node and something is hovering over it */
      else if ((node.type === "t2i-generator") && intersections.includes(node.id)) {
        const generatorNode = node as Node<T2IGeneratorNodeData>;
        generatorNode.data.updateNode?.("", "dragging"); // Trigger dragging mode
        return {
          ...generatorNode,
          className: "highlight-green",
        };
      } else if (node.type === "t2i-generator") {
        const generatorNode = node as Node<T2IGeneratorNodeData>;
        generatorNode.data.updateNode?.("", "ready");
      }

      return {
        ...node,
        className: "",
      };
    });
    return updatedNodes;
  });
};

  const handleIntersectionsOnDrop = (draggedNode: Node, intersections: string[]) => {
    setFlowNodes((currentNodes) => {
      const updatedNodes = currentNodes.map((node) => {
        if (node.type === "text" && draggedNode.type === "text" && intersections.includes(node.id)) {
          const draggedTextNode = draggedNode as Node<TextNodeData>;
          const textNode = node as Node<TextNodeData>;
          draggedTextNode.data.combinable = true;
          textNode.data.combinable = true;
          addTextNode(textNode.data.content + ", " + draggedTextNode.data.content, { x: textNode.position.x + 20, y: textNode.position.y + 20 });
          console.log(draggedTextNode.data.content + " " + textNode.data.content);

          // deleteNodeById(draggedTextNode.id);
          // deleteNodeById(textNode.id);
        } else if (node.type === "text") {
          const textNode = node as Node<TextNodeData>;
          textNode.data.combinable = false;
        }
        /*--- If a node has been dragged on top of a t2i generator ---*/
        else if (node.type === "t2i-generator" && intersections.includes(node.id)) {
          const generatorNode = node as Node<T2IGeneratorNodeData>;
          const overlappingNode = currentNodes.find((n) => n.id === draggedNode.id);

          const inputNodeContent =
            //grab the content of the node that was dragged on top, and use it as input for our generator
            overlappingNode && "content" in overlappingNode.data
              ? overlappingNode.data.content
              : overlappingNode && "label" in overlappingNode.data
              ? overlappingNode.data.label
              : "No content";

          // Check if the node is ready for generation
          const isReady = generatorNode.data.updateNode?.("", "check") === true;

          if (
            isReady &&
            typeof inputNodeContent === "string" &&
            inputNodeContent.trim() !== ""
          ) {
            // Rerender the node in "generating" mode with the prompt
            generatorNode.data.updateNode?.(inputNodeContent, "generating");

            // Generate a new image node, use helper function to decide where it should appear
            generateNode(inputNodeContent, calcPositionAfterDrag(node.position, node, "below"))
              .then(() => {
                console.log("Generation complete!");

                generatorNode.data.updateNode?.("", "ready");
              })
              .catch((error) => {
                console.error("Image generation failed:", error);
                generatorNode.data.updateNode?.("", "ready"); // Reset even on failure
              });

            // For the original overlapping node we just used as input,
            // let's move it out of the way so it's not overlapping anymore.
            if (overlappingNode) {
              const overlappingNodePrevPosition = { x: overlappingNode.position.x, y: overlappingNode.position.y };
              const newPosition = calcPositionAfterDrag(overlappingNodePrevPosition, node, "above");
              updatePosition(overlappingNode.id, newPosition); //move it
            }

            return {
              ...node, className: "", // Reset the highlight
            };
          }
        } else if (node.type === "t2i-generator") { // the node is a generator, but nothing is intersecting with it
          const generatorNode = node as Node<T2IGeneratorNodeData>;
          generatorNode.data.updateNode?.("", "ready");
        }
        return {
          ...node,
          className: "",
        };
      });

      return updatedNodes;
    });
  };



  
  //*** -- Node Adders  (functions that add nodes to the canvas) -- ***/

  

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

    setFlowNodes((prevNodes) => [...prevNodes, newTextNode]);
  };

  const addImageNode = (content?: string, position?: { x: number; y: number }, prompt?: string) => {
    console.log(content);
    position = position ?? { 
      x: Math.random() * 250,
      y: Math.random() * 250,
    };
    content = content ?? "https://noggin-run-outputs.rgdata.net/b88eb8b8-b2b9-47b2-9796-47fcd15b7289.webp";
    prompt = prompt ?? "None";
    const newNode: AppNode = {
      id: `image-${flowNodes.length + 1}`,
      type: "image",
      position: position,
      data: {
        content: content,
        prompt: prompt,
        activateLookUp: () => handleImageLookUp(position, content),
      } as ImageNodeData,
      dragHandle: '.drag-handle__invisible',
    };

    setFlowNodes((prevNodes) => [...prevNodes, newNode]);
  };

  const addT2IGenerator = (position ?:{ x:number, y:number}) => {
    const newT2IGeneratorNode: AppNode = {
      id: `t2i-generator-${flowNodes.length + 1}`,
      type: "t2i-generator",
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
      } as T2IGeneratorNodeData,
    };

    setFlowNodes((prevNodes) => [...prevNodes, newT2IGeneratorNode]);
  };

  /*-- adds a lookup window ---*/
  const handleImageLookUp = useCallback(async (position: {x: number; y: number;}, imageUrl: string) => {
    //takes an image and its position as input, looks up the image in the backend, adds the results as a LookupNode to the canvas
    console.log(`Looking up image with url: ${imageUrl}`);
    console.log(`!!!received position: ${position.x, position.y}`);
    
    // Add a blank text node to indicate loading
    const loadingNodeId = `loading-${Date.now()}`;
    const loadingNode: AppNode = {
      id: loadingNodeId,
      type: "text",
      position: { x: position.x - 20, y: position.y - 20 },
      data: { content: "...that reminds me of something...", loading: true, combinable: false } as TextNodeData,
    };
    setFlowNodes((nodes) => [...nodes, loadingNode]);

    try {
      const response = await axios.post(`${backend}/api/get-similar-images`, {
        image: imageUrl
      }, {
        headers: {
          "Content-Type": "application/json",
        },
      });
      console.log(`Image lookup response: ${response.data}`);
      if (response.status === 200) {
        const responseData = JSON.parse(response.data);
        const artworks: Artwork[] = responseData.map((item: any) => ({
          title: item.title || "Unknown",
          date: item.date || "Unknown",
          artist: item.artist || "Unknown",
          keywords: [
        {
          id: `genre-${Date.now()}`, // todo, this should be replaced with the actual gene ids from Artsy!
          type: "genre",
          value: item.genre || "Unknown",
        },
        {
          id: `style-${Date.now()}`,
          type: "style",
          value: item.style || "Unknown",
        },
          ],
          description: item.description || "Unknown",
          image: item.image || "Unknown",
        }));
        //Replace the loading node with the new lookup node
        const newLookupNode: LookupNode = {
          id: `lookup-${Date.now()}`,
          type: "lookup",
          position,
          data: {
            content: "Similar Images",
            artworks,
          },
          dragHandle: '.drag-handle__custom',
        };

        setFlowNodes((nodes) =>
          nodes.map((node) =>
            node.id === loadingNodeId ? newLookupNode : node
          )
        );
      }
    } catch (error) {
      console.error("Failed to lookup image:", error);
    }
  }, [setFlowNodes]);




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
      setFlowNodes((nodes) => [...nodes, loadingNode]);

      // let's generate a node... 

      try {
        if (isValidImage){// the user has sent an image for text generation
          console.log(`Describing this image: ${prompt}`);
          const response = await axios.post(`${backend}/api/generate-text`, {
            imageUrl: prompt, // Send the imageUrl as part of the request body
          });

          if (response.status === 200) {
            const newTextNode: AppNode = {
              id: `text-${Date.now()}`,
              type: "text",
              position,
              data: { 
                content: response.data.text,
                loading: false,
                combinable: false,
              } as TextNodeData,
            };
            setFlowNodes((nodes) =>
              nodes.map((node) =>
                node.id === loadingNodeId ? newTextNode : node
              )
            );
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
              //console.log(`Image url ${response.data.imageUrl} received from backend!`);
              const newImageNode: AppNode = {
                id: `image-${Date.now()}`,
                type: "image",
                position,
                data: { 
                  content: response.data.imageUrl,
                  prompt: prompt,
                  activateLookUp: () => handleImageLookUp(position, response.data.imageUrl),
                } as ImageNodeData,
                dragHandle: '.drag-handle__invisible',
              };

              setFlowNodes((nodes) =>
                nodes.map((node) =>
                  node.id === loadingNodeId ? newImageNode : node
                )
              );
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
    [setFlowNodes]
  );

  const [showDebugInfo, setShowDebugInfo] = useState(false);

  return (
    <>
      <div style={{ position: 'absolute', top: '10px', left: '45%', transform: 'translateX(-50%)', display: 'flex', justifyContent: 'center', gap: '10px', zIndex: 10 }}>
        <button onClick={() => addTextNode()}>Text</button>
        <button onClick={() => addImageNode()}>Image</button>
        <button onClick={() => addT2IGenerator()}>New Text to Image Generator</button>
      </div>

      {/* Load & Save Buttons */}
      <div className="absolute top-10 left-10 z-10 space-x-2">
        <button onClick={handleLoadCanvas} className="bg-blue-500 text-white p-2 rounded">
          Load Canvas
        </button>
        <button onClick={handleSaveCanvas} className="bg-green-500 text-white p-2 rounded">
          Save Canvas
        </button>
      </div>

            
      <div style={{ width: '100%', height: '100vh' }}>
        <ReactFlow
          nodes={flowNodes}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
          onDrop={onDrop}
          onDragOver={onDragOver}
          zoomOnDoubleClick={false}
          fitView
          selectionOnDrag
        >
          <Background />
          {/*<MiniMap />*/}
          <Controls />
        </ReactFlow>
      </div>  




      {/* Debug Info */}
          <div className="fixed bottom-0 left-0 bg-gray-900 text-white p-4 z-50 text-sm rounded-md shadow-md">
          <button
            onClick={() => setShowDebugInfo((prev) => !prev)}
            className="bg-blue-500 text-white p-2 rounded mb-2"
          >
          {showDebugInfo ? "Hide Debug Info" : "Show Debug Info"}
          </button>
          {showDebugInfo && (
            <div>
          <p><strong>User ID:</strong> {userID}</p>
          <p><strong>Canvas Name:</strong> {canvasName}</p>
          <p><strong>Canvas ID:</strong> {canvasId}</p>
          <p><strong>Flow Nodes:</strong> {JSON.stringify(flowNodes, null, 2)}</p>
          <p><strong>Saved Nodes:</strong> {JSON.stringify(savedNodes, null, 2)}</p>
            </div>
          )
          }
        </div>


      </>

  );
};

export default Flow;
