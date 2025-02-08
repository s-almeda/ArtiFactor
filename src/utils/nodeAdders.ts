import { AppNode, TextNodeData, ImageNodeData, LookupNode, T2IGeneratorNodeData } from '../nodes/types'; // Adjust the import path as needed
import axios from 'axios';
import { useCallback } from 'react';
import { Artwork } from '../nodes/types'; // TODO- move Artwork definition somewhere else maybe?
import { useAppContext } from '../context/AppContext'; // Adjust the import path as needed

export const addTextNode = (setFlowNodes: Function, content: string = "your text here", position?: { x: number; y: number }) => {
    const newTextNode: AppNode = {
        id: `text-${Date.now()}`,
        type: "text",
        position: position ?? {
            x: Math.random() * 250,
            y: Math.random() * 250,
        },
        data: {
            content: content,
            loading: false,
            combinable: false
        } as TextNodeData,
    };

    setFlowNodes((prevNodes: AppNode[]) => [...prevNodes, newTextNode]);
};

export const addImageNode = (setFlowNodes: Function, content?: string, position?: { x: number; y: number }, prompt?: string) => {
    console.log("creating and image node with this url: ", content);
    position = position ?? { 
        x: Math.random() * 250,
        y: Math.random() * 250,
    };
    content = content ?? "https://noggin-run-outputs.rgdata.net/b88eb8b8-b2b9-47b2-9796-47fcd15b7289.webp";
    prompt = prompt ?? "None";
    const newNode: AppNode = {
        id: `image-${Date.now()}`,
        type: "image",
        position: position,
        data: {
            content: content,
            prompt: prompt,
            activateLookUp: handleImageLookUp(setFlowNodes),
        } as ImageNodeData,
        dragHandle: '.drag-handle__invisible',
    };

    setFlowNodes((prevNodes: AppNode[]) => [...prevNodes, newNode]);
};

export const addT2IGenerator = (setFlowNodes: Function, position?: { x: number; y: number }) => {
    const newT2IGeneratorNode: AppNode = {
        id: `t2i-generator-${Date.now()}`,
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

    setFlowNodes((prevNodes: AppNode[]) => [...prevNodes, newT2IGeneratorNode]);
};

export const handleImageLookUp = (setFlowNodes: Function) => {
    const {backend} = useAppContext();
    return useCallback(async (position: { x: number; y: number }, imageUrl: string) => {
        console.log(`Looking up image with url: ${imageUrl}`);
        console.log(`!!!received position: ${position.x}, ${position.y}`);

        const loadingNodeId = `loading-${Date.now()}`;
        const loadingNode: AppNode = {
            id: loadingNodeId,
            type: "text",
            position: { x: position.x - 20, y: position.y - 20 },
            data: { content: "...that reminds me of something...", loading: true, combinable: false } as TextNodeData,
        };
        setFlowNodes((nodes: AppNode[]) => [...nodes, loadingNode]);

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
                            id: `genre-${Date.now()}`,
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

                setFlowNodes((nodes: AppNode[]) =>
                    nodes.map((node) =>
                        node.id === loadingNodeId ? newLookupNode : node
                    )
                );
            }
        } catch (error) {
            console.error("Failed to lookup image:", error);
        }
    }, [setFlowNodes, backend]);
};
