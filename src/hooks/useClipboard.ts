import { useCallback, useState } from "react";
import { AppNode } from "../nodes/types"; // adjust the import as needed

const useClipboard = (
  nodes: AppNode[],
  setNodes: React.Dispatch<React.SetStateAction<AppNode[]>>
) => {
  const [clipboard, setClipboard] = useState<Array<AppNode> | null>(null);

  const handleCopy = useCallback(() => {
    const selectedNodes = nodes.filter((node) => node.selected);
    setClipboard(selectedNodes);
  }, [nodes]);

  const handleCut = useCallback(() => {
    const selectedNodes = nodes.filter((node) => node.selected);
    setClipboard(selectedNodes);
    setNodes((nodes) => nodes.filter((node) => !node.selected));
  }, [nodes, setNodes]);

  const handlePaste = useCallback(() => {
    if (clipboard) {
      const offset = 10 * (nodes.length + 1);
      const newNodes = clipboard.map((node) => ({
        ...node,
        id: `${nodes.length + 1}`,
        position: {
          x: node.position.x + offset,
          y: node.position.y + offset,
        },
      }));
      setNodes((nodes) => [...nodes, ...newNodes]);
    }
  }, [clipboard, nodes, setNodes]);

  return {
    handleCopy,
    handleCut,
    handlePaste,
  };
};

export default useClipboard;
