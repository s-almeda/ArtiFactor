import { useCallback, useState, useEffect } from "react";
import { type NodeProps } from "@xyflow/react";
import { type SynthesizerNode } from "./types";
import { motion } from "framer-motion";

function SynthesizerNode({ data }: NodeProps<SynthesizerNode>) {
    
    const [content, setContent] = useState("");
    const mode = data.mode;

    const updateNode = useCallback(() => {
        if (mode === "generating-image") {
            setContent("describing image...");
        } 
        else if(mode === "generating-text") {
            setContent(`generating image...`);// with prompt: "${data.inputNodeContent}"`);
        }
        else if (mode === "dragging") {
            setContent("drop here!");
        } 
        else{
            setContent("ready!");
        }
    }, [data]);

    useEffect(() => {
        updateNode();
    }, [mode, updateNode]);

    const backgroundClass =
        mode === "dragging"
            ? "highlight-green"
            : mode === "generating-image"
            ? "bg-sky-500"
            : mode === "generating-text"
            ? "bg-tan-500"
            : "bg-white";

    return (
        <motion.div
        initial={{ opacity: 0, x:0, y: 3, scale: 0.6, filter: "blur(10px)"}}
        animate={ { opacity: 1, x: 0, y: 0, scale: 1,  scaleX:1, filter: ""}}
        transition={{ duration: 0.4, type: "spring", bounce: 0.1 }}
        >
        <div
            className={`p-3 border border-gray-700 rounded ${backgroundClass} transition-all duration-300`}
        >
            <div style={{ fontWeight: "bold" }}>{"Synthesizer"}</div>
            <div className="text-xs text-gray-600 break-words whitespace-pre-wrap">{content}</div>
        </div>
        </motion.div>
    );
}

export default SynthesizerNode;
