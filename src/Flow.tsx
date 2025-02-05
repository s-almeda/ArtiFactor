import { useCallback, useEffect, type MouseEvent } from "react";
import axios from "axios";
import {
  ReactFlow,
  Background,
  Controls,
  //MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
} from "@xyflow/react";
import type { Node, OnConnect } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { AppNode, Artwork } from "./nodes/types";
import { initialNodes, nodeTypes } from "./nodes";
import { initialEdges, edgeTypes } from "./edges";
import useClipboard from "./hooks/useClipboard";
import { useDnD } from './DnDContext';

// import { ImageNode } from "./nodes/ImageNode";

//--- ONLY UNCOMMENT ONE OF THESE (depending on which backend server you're running.).... ---//
//USE THIS LOCAL ONE for local development...
//const backend_url = "http://localhost:3000"; // URL of the LOCAL backend server (use this if you're running server.js in a separate window!)
//const backend_url = "http://104.200.25.53/"; //IP address of backend server hosted online, probably don't use this one.

// TURN THIS ONLINE ONE back on before you run "npm build" and deploy to Vercel!
const backend_url = "https://snailbunny.site"; // URL of the backend server hosted online! 


const Flow = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const { getIntersectingNodes, getNodesBounds, screenToFlowPosition } = useReactFlow();
  const [draggableType,__, draggableContent,_ ] = useDnD();

  const { handleCopy, handleCut, handlePaste } = useClipboard(nodes, setNodes); // Use the custom hook

  

  const deleteNodeById = (id: string) => {
    setNodes((currentNodes) => currentNodes.filter((node) => node.id !== id))
  }

  
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
  
   // --- FUNCTIONS TRIGGERED BY CANVAS INTERACTIONS --- //
  const onConnect: OnConnect = useCallback(
    async (connection) => {
      const sourceNode = nodes.find((node) => node.id === connection.source);
      const targetNode = nodes.find((node) => node.id === connection.target);

      if (targetNode && targetNode.type === "function") {
        const animatedConnection = { ...connection, type: "wire" };
        setEdges((edges) => addEdge(animatedConnection, edges));

        if (sourceNode && targetNode) {
          const content =
            "content" in sourceNode.data
              ? sourceNode.data.content
              : sourceNode.data.label;

          console.log(`functionNode content is currently: ${content}`);

          // Update the target node's content after the source node content is ready
          setNodes(
            (nodes) =>
              nodes.map((node) =>
                node.id === targetNode.id
                  ? {
                      ...node,
                      data: {
                        ...node.data,
                        content: content,
                      },
                    }
                  : node
              ) as AppNode[]
          );

          const prompt =
            "content" in sourceNode.data
              ? sourceNode.data.content
              : sourceNode.data.label;
          console.log(`Generating image with prompt: ${prompt}`);
          // await generateImageNode(prompt);
        }
      } else {
        setEdges((edges) => addEdge(connection, edges));
      }
    },
    [nodes, setEdges, setNodes]
  );

  const onNodeDrag = useCallback(
    (_: MouseEvent, draggedNode: Node) => {
      setNodes((currentNodes) =>
        currentNodes.map((node) => {
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
    [setNodes, getIntersectingNodes]
  );

  /* -- when something else is dragged over the canvas -- */
  const onDragOver = useCallback((event: { preventDefault: () => void; dataTransfer: { dropEffect: string; }; }) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  
 
  const onDrop = useCallback(
    (event: { preventDefault: () => void; clientX: any; clientY: any; }) => {
      event.preventDefault();
      console.log(`you dropped: ${draggableType} and ${draggableContent}`);
      // check if the dropped element is valid
      if (!draggableType) {
        return;
      }
 
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      switch (draggableType) {
        case "image":
          addImageNode(draggableContent as unknown as string, position);
          break;
        case "t2i-generator":
          addT2IGenerator(position);
          break;
        default:
          addTextNode(draggableContent as unknown as string, position);
          break;
          //console.warn(`Unknown draggable type: ${draggableType}`);
      }


      
    },
    [draggableType, draggableContent,screenToFlowPosition],
    
  );

  const onNodeDragStop = useCallback(
    (_: MouseEvent, draggedNode: Node) => {
      //console.log("user stopped dragging a node ", draggedNode)
      const intersections = getIntersectingNodes(draggedNode).map((n) => n.id);
      handleIntersectionsOnDrop(draggedNode, intersections);
    },
    [getIntersectingNodes, setNodes]
  );


  // --- HELPER FUNCTIONS --- //

  // const isIntersector = (nodeType: string | undefined) => {
  //   if (!nodeType) return false;
  //   //returns true if the type of node is an "intersector"
  //   return nodeType === "intersection" || nodeType === "t2i-generator";
  // }

   /*** ---- this is the code that makes stuff fly away lol --- */

  const calcPositionAfterDrag = (
    previousPosition: { x: number; y: number },
    intersectionNode: Node,
    direction: "above" | "below" = "below"
  ) => {
    const bounds = getNodesBounds([intersectionNode]);
    let newPosition = { ...previousPosition };
  
    const xOffset = typeof intersectionNode.data.xOffset === 'number' ? intersectionNode.data.xOffset : 10;
    const yOffset = typeof intersectionNode.data.yOffset === 'number' ? intersectionNode.data.yOffset : 10;
  
    newPosition.x = bounds.x + xOffset; // place to the right
    newPosition.y = bounds.y + bounds.height + yOffset; // place below

    if (direction === "below") {
      newPosition.x = bounds.x + xOffset; // place to the right
      newPosition.y = bounds.y + bounds.height + yOffset; // place below
    } else if (direction === "above") {
      newPosition.x = bounds.x - bounds.width/2 - xOffset; // place to the left
      newPosition.y = bounds.y - bounds.height - yOffset; // place above
    }
    return newPosition;
  };
  
  const moveNode = (nodeId: string, newPosition: { x: number; y: number }) => {
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
    }, 1000); // Match the duration of the CSS transition
  };

const handleIntersectionsOnDrag = (draggedNode: Node, intersections: string[]) => {
  setNodes((currentNodes) => {
    const updatedNodes = currentNodes.map((node) => {
      /*--- If the draggedNode and the node it overlaps with are BOTH text nodes ---*/
      if (node.type === "text" && draggedNode.type === "text" && intersections.includes(node.id)) {
        draggedNode.data.combinable = true;
        node.data.combinable = true;
      } 
      else if (node.type === "text") {
        node.data.combinable = false;
      }
      /*this node is a generator node and something is hovering over it */
      else if ((node.type === "t2i-generator") && intersections.includes(node.id)) {
        node.data.updateNode?.("", "dragging"); // Trigger dragging mode
        return {
          ...node,
          className: "highlight-green",
        };
      } else if (node.type === "t2i-generator") {
        node.data.updateNode?.("", "ready");
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
    setNodes((currentNodes) => {
      const updatedNodes = currentNodes.map((node) => {
        if (node.type === "text" && draggedNode.type === "text" && intersections.includes(node.id)) {
          draggedNode.data.combinable = true;
          node.data.combinable = true;
          addTextNode(node.data.content + ", " + draggedNode.data.content, { x: node.position.x + 20, y: node.position.y + 20 });
          console.log(draggedNode.data.content + " " + node.data.content);

          deleteNodeById(draggedNode.id);
          deleteNodeById(node.id);
        } 
        else if (node.type === "text") {
          node.data.combinable = false;
        }
        /*--- If a node has been dragged on top of a t2i generator ---*/
        else if (node.type === "t2i-generator" && intersections.includes(node.id)) {
          const overlappingNode = currentNodes.find((n) => n.id === draggedNode.id);
          const inputNodeContent = 
          //grab the content of the node that was dragged on top, and use it as input for our generator
            overlappingNode && "content" in overlappingNode.data
              ? overlappingNode.data.content
              : overlappingNode && "label" in overlappingNode.data
              ? overlappingNode.data.label
              : "No content";
  
          // Check if the node is ready for generation
          const isReady = node.data.updateNode?.("", "check") === true;

          if (
            isReady &&
            typeof inputNodeContent === "string" &&
            inputNodeContent.trim() !== ""
          ) 
          {

            // Rerender the node in "generating" mode with the prompt
            node.data.updateNode?.(inputNodeContent, "generating");
  
            // Generate a new image node, use helper function to decide where it should appear
            generateImageNode(inputNodeContent, calcPositionAfterDrag(node.position, node, "below"))
              .then(() => {
                console.log("Image generation complete!");
                
                node.data.updateNode?.("", "ready");  
              })
              .catch((error) => {
                console.error("Image generation failed:", error);
                node.data.updateNode?.("", "ready"); // Reset even on failure
              });

            // For the original overlapping node we just used as input,
            // let's move it out of the way so it's not overlapping anymore.
            if (overlappingNode) {
              const overlappingNodePrevPosition = { x: overlappingNode.position.x, y: overlappingNode.position.y };
              const newPosition = calcPositionAfterDrag(overlappingNodePrevPosition, node, "above");
              moveNode(overlappingNode.id, newPosition); //move it
            }
  
            return {
              ...node, className:"", // Reset the highlight
            };
          }
        }
        else if (node.type === "t2i-generator") { // the node is a generator, but nothing is intersecting with it
          node.data.updateNode?.("", "ready"); 
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
      },
    };

    setNodes((prevNodes) => [...prevNodes, newTextNode]);
  };

  const addImageNode = (content?: string, position?: { x: number; y: number }) => {
    console.log(content);
    position = position ?? { 
      x: Math.random() * 250,
      y: Math.random() * 250,
    };
    content = content ?? "https://noggin-run-outputs.rgdata.net/b88eb8b8-b2b9-47b2-9796-47fcd15b7289.webp";
    const newNode: AppNode = {
      id: `image-${nodes.length + 1}`,
      type: "image",
      position: position,
      data: {
        content: content,
        lookUp: () => handleImageLookUp(position, content),
      },
    };

    setNodes((prevNodes) => [...prevNodes, newNode]);
  };

  const addT2IGenerator = (position ?:{ x:number, y:number}) => {
    const newT2IGeneratorNode: AppNode = {
      id: `t2i-generator-${nodes.length + 1}`,
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
      },
    };

    setNodes((prevNodes) => [...prevNodes, newT2IGeneratorNode]);
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
      data: { content: "...that reminds me of something...", loading: true, combinable: false },
    };
    setNodes((nodes) => [...nodes, loadingNode]);

    try {
      const response = await axios.post(`${backend_url}/api/get-similar-images`, {
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
          genre: item.genre || "Unknown",
          style: item.style || "Unknown",
          description: item.description || "Unknown",
          image: item.image || "Unknown",
        }));
        
        // Replace the loading node with the new lookup node
        const newLookupNode: AppNode = {
          id: `lookup-${Date.now()}`,
          type: "lookup",
          position,
          data: {
            content: "Similar Images",
            artworks,
          },
          dragHandle: '.drag-handle__custom',
        };

        setNodes((nodes) =>
          nodes.map((node) =>
            node.id === loadingNodeId ? newLookupNode : node
          )
        );
      }
    } catch (error) {
      console.error("Failed to lookup image:", error);
    }
  }, [setNodes]);




  const generateImageNode = useCallback(
    // Requests through reagent, generates an image node with a prompt and optional position
    async (prompt: string = "bunny on the moon", position: { x: number; y: number } = { x: 250, y: 250 }, testing: boolean = false) => {
      console.log(`Generating image with prompt: ${prompt}`);
      const loadingNodeId = `loading-${Date.now()}`;
      const loadingNode: AppNode = {
        id: loadingNodeId,
        type: "text",
        position,
        data: { content: "loading ", loading: true, combinable: false },
      };
      setNodes((nodes) => [...nodes, loadingNode]);

      // if the testing paramter is true, let's just use a default image to avoid generation costs
      if (testing) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const newImageNode: AppNode = {
          id: `image-${Date.now()}`,
          type: "image",
          position,
          data: { 
            content: "https://collectionapi.metmuseum.org/api/collection/v1/iiif/459123/913555/main-image", 
            lookUp: () => handleImageLookUp(position, "https://collectionapi.metmuseum.org/api/collection/v1/iiif/459123/913555/main-image"),
            },
        };

        setNodes((nodes) =>
          nodes.map((node) =>
            node.id === loadingNodeId ? newImageNode : node
          )
        );
        return;
      }

      // let's generate the image for real for real

      try {
        const formData = new FormData();
        formData.append("prompt", prompt); // make the prompt into form data

        // Make a POST request to the backend server 
        const response = await axios.post(`${backend_url}/api/generate-image`, {
            prompt, // Send the prompt as part of the request body
          });
      
          if (response.status === 200) {
            console.log(`Image url ${response.data.imageUrl} received from backend!`);

            const newImageNode: AppNode = {
              id: `image-${Date.now()}`,
              type: "image",
              position,
              data: { 
                content: response.data.imageUrl,
                lookUp: () => handleImageLookUp(position, response.data.imageUrl),
              },
            };

            setNodes((nodes) =>
              nodes.map((node) =>
                node.id === loadingNodeId ? newImageNode : node
              )
            );

          } else {
            console.error(`Generation Error: ${response.status}`);
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
    [setNodes]
  );


  return (
    <>
      <div style={{ position: 'absolute', top: '10px', left: '25%', transform: 'translateX(-50%)', display: 'flex', justifyContent: 'center', gap: '10px', zIndex: 10 }}>
        <button onClick={() => addTextNode()}>Text</button>
        <button onClick={() => addImageNode()}>Image</button>
        <button onClick={() => addT2IGenerator()}>New Text to Image Generator</button>
      </div>

      <ReactFlow
        nodes={nodes}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onNodeDrag={onNodeDrag}
        onNodeDragStop={onNodeDragStop}
        onDrop={onDrop}
        onDragOver={onDragOver}
        edges={edges}
        edgeTypes={edgeTypes}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        zoomOnDoubleClick={false}
        fitView
//        fitViewOptions={{minZoom: 0.001}}
        selectionOnDrag
      >
        <Background />
        {/*<MiniMap />*/}
        <Controls />
      </ReactFlow>
    </>
  );
};

export default Flow;
