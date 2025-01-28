import { memo, useEffect, useState } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { type T2IGeneratorNode } from "./types";

interface T2IGeneratorNodeProps extends NodeProps<T2IGeneratorNode> {}

const T2IGeneratorNode = ({ data }: T2IGeneratorNodeProps) => {
    const [mode, setMode] = useState<"ready" | "generating" | "dragging">("ready");
    const [content, setContent] = useState("ready to generate! \n(drag and drop here)");

    const updateNode = ( //todo, refactor as "updateNode"
        inputNodeContent: string, //todo, refactor as "input"
        newMode?: "ready" | "generating" | "dragging" | "check"
    ) => {

        if (newMode === "generating") {
            setMode("generating");
            setContent(`now generating "${inputNodeContent}"`);

            // Increment offsets
            data.xOffset = (data.xOffset || 0) + 15;
            data.yOffset = (data.yOffset || 0) + 15;

            if (data.yOffset >= 100) {
                data.xOffset = 10;
                data.yOffset = 0;
            }
        } else if (newMode === "dragging") {
            setMode("dragging");
            setContent("drop to generate...");
        } else {
            setMode("ready");
            setContent("ready to generate! \n(drag and drop here)");
        }

        return mode === "ready" || mode === "dragging"; // Return whether the node is ready

    };


    useEffect(() => {
        // Expose updateNode to Flow.tsx for external control
        data.updateNode = updateNode;
    }, [data, mode]);

    // Dynamically determine the classes
    const backgroundClass =
        mode === "dragging"
            ? "highlight-green" // Custom CSS class for dragging
            : mode === "generating"
            ? "bg-sky-500" // Tailwind for sky blue
            : "bg-white"; // Tailwind for white

    return (

        <div
            className={`p-3 border border-gray-700 rounded ${backgroundClass} transition-all duration-300`}
        >
            <div style={{ fontWeight: "bold" }}>{"Generator"}</div>
            <div style={{ fontSize: "smaller", color: "#555" }}>{content}</div>
           {/*<div>Current Mode: {mode}</div>*/}

            <Handle type="target" position={Position.Top} />
            <Handle type="source" position={Position.Bottom} />
        </div>
    );
};

export default memo(T2IGeneratorNode);