import { useCallback, useState } from "react";
import { AppNode } from "../nodes/types"; // adjust the import as needed
import { v4 as uuidv4 } from "uuid";
import {useNodeContext } from "../context/NodeContext";

const useClipboard = (
  clipboardNodes: AppNode[],
) => {
  const [clipboard, setClipboard] = useState<Array<AppNode> | null>(null);
  const { setNodes, setEdges } = useNodeContext();

  const handleCopy = useCallback(() => {
    const selectedNodes = clipboardNodes.filter((node) => node.selected);
    setClipboard(selectedNodes);
  }, [clipboardNodes]);

  const handleCut = useCallback(() => {
    const selectedNodes = clipboardNodes.filter((node) => node.selected);
    setClipboard(selectedNodes);
    setNodes((nodes) => nodes.filter((node) => !node.selected));
  }, [clipboardNodes, setNodes]);

  const handlePaste = useCallback(() => {
    if (clipboard) {
      const offset = 10 * (clipboardNodes.length + 1);
      const newNodes = clipboard.map((node) => {
        const nodeType = node.type; // Assuming `type` is a property of AppNode that indicates the node type (e.g., "text" or "image")
        const newNode = {
          ...node,
          id: `${nodeType}-${uuidv4()}`,
          position: {
            x: node.position.x + offset,
            y: node.position.y + offset,
          },
          selected: true,
        };
        return newNode;
      });

      const newEdges = clipboard.map((node, index) => ({
        id: `edge-${node.id}-${newNodes[index].id}`,
        source: node.id,
        target: newNodes[index].id,
        type: 'default',
      }));

      setEdges((prevEdges) => [...prevEdges, ...newEdges]);

      setNodes((nodes) =>
        nodes.map((node) => ({ ...node, selected: false })).concat(newNodes)
      );
    }
  }, [clipboard, clipboardNodes, setNodes, setEdges]);

  return {
    handleCopy,
    handleCut,
    handlePaste,
  };
};

export default useClipboard;
