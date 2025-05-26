import { useDnD } from "../context/DnDContext";
import { Search } from "lucide-react";
import { motion } from "framer-motion";
import React, { useState } from "react";

import NavigationButtons from "./NavigationButtons";
import DynamicDescription from "./DynamicDescription";
import { RelatedKeywords } from "./RelatedKeywords";

import { Keyword } from "../nodes/types"; // Adjust the import path as necessary


// -- FOLDER PANEL COMPONENT (the panel that opens on the left side) --- //
const FolderPanel: React.FC<{
    parentNodeId: string;
    width: number;
    height: number;
    showFolder: boolean;
    toggleFolder: () => void;
    similarTexts: Keyword[];
    isAIGenerated: boolean;
}> = ({
    parentNodeId,
    width,
    height,
    showFolder,
    toggleFolder,
    similarTexts,
    isAIGenerated = false,
}) => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const { setDraggableType, setDraggableData } = useDnD();

    const onDragStart = (
        event: React.DragEvent<HTMLDivElement>,
        type: string,
        content: string,
        prompt?: string
    ) => {
        event.dataTransfer.effectAllowed = "move";
        setDraggableType(type);
        setDraggableData({
            content: content,
            prompt: prompt,
            provenance: "history",
            parentNodeId: parentNodeId,
        });
    };

    const handlePrev = () => {
        setCurrentIndex((prevIndex: number) =>
            prevIndex > 0 ? prevIndex - 1 : similarTexts.length - 1
        );
    };

    const handleNext = () => {
        setCurrentIndex((prevIndex: number) =>
            prevIndex < similarTexts.length - 1 ? prevIndex + 1 : 0
        );
    };

    const currentText = similarTexts[currentIndex];

    return (
        <>
            {/* SEARCH ICON BUTTON */}
            <div
                className={`absolute left-0 top-2 transform -translate-x-7 -translate-y-2 
                                        w-12 h-20 p-1
                                        ${
                                            isAIGenerated
                                                ? "bg-blue-200 hover:bg-blue-300"
                                                : "bg-[#dbcdb4] hover:bg-[#b39e79]"
                                        }
                                        flex items-center justify-left rounded-l-md
                                        cursor-pointer transition-colors duration-200 
                                        ${
                                            showFolder
                                                ? isAIGenerated
                                                    ? "bg-blue-200"
                                                    : "bg-[#dbcdb4]"
                                                : ""
                                        }`}
                onClick={toggleFolder}
            >
                <Search size={20} className="text-gray-600" />
            </div>

            <motion.div
                initial={{ transform: `rotateX(-45deg)` }}
                animate={{
                    transform: showFolder ? `rotateX(0deg)` : `rotateX(-60deg)`,
                }}
                transition={{ duration: 0.3, type: "spring", bounce: 0.2 }}
                className="absolute"
            >
                <div
                    className={`absolute left-0 top-0 transform -translate-x-[${
                        width + 6
                    }px] ${
                        isAIGenerated
                            ? "bg-blue-100"
                            : showFolder
                            ? "bg-[#f2e7ce]"
                            : "bg-[#dbcdb4]"
                    } rounded-md shadow-md`}
                    style={{ height: `${height * 2}px`, width: `${width}px` }}
                >
                    {similarTexts.length > 0 ? (
                        <>
                            <div className="p-3 pt-0 ml-0 h-full overflow-y-auto">
                                <NavigationButtons
                                    currentIndex={currentIndex}
                                    totalItems={similarTexts.length}
                                    handlePrev={handlePrev}
                                    handleNext={handleNext}
                                />

                                {currentText && (
                                    <div className="nodrag nowheel text-gray-600 overflow-y-auto">
                                        <p
                                            draggable
                                            onDragStart={(event) =>
                                                onDragStart(event, "text", currentText.value)
                                            }
                                            className={`text-md font-bold ${
                                                isAIGenerated
                                                    ? "hover:bg-blue-200"
                                                    : "hover:bg-[#dbcdb4]"
                                            }`}
                                        >
                                            {currentText.value}
                                        </p>

                                        <p
                                            draggable
                                            onDragStart={(event) =>
                                                onDragStart(event, "text", currentText.type)
                                            }
                                            className={`italic text-xs ${
                                                isAIGenerated
                                                    ? "hover:bg-blue-200"
                                                    : "hover:bg-[#dbcdb4]"
                                            }`}
                                        >
                                            {currentText.type}
                                        </p>

                                        <div className={`text-xs/4 mt-2 p-0.5 rounded-sm`}>
                                            {currentText.descriptions && (
                                                <DynamicDescription
                                                    descriptions={currentText.descriptions}
                                                    isAIGenerated={isAIGenerated}
                                                />
                                            )}
                                        </div>

                                        {currentText.relatedKeywordStrings &&
                                            currentText.relatedKeywordStrings.length > 0 && (
                                                <RelatedKeywords
                                                    relatedKeywords={currentText.relatedKeywordStrings}
                                                    isAIGenerated={isAIGenerated}
                                                />
                                            )}
                                    </div>
                                )}
                            </div>
                        </>
                    ) : (
                        <div className="p-3 ml-0 h-full overflow-y-auto flex flex-col items-center justify-center">
                            <h2 className="text-xs font-medium text-gray-900 italic font-bold text-center mb-5">
                                ...We haven't found anything relevant to this in our
                                database yet...
                            </h2>
                            <div className="loader"></div>
                        </div>
                    )}
                </div>
            </motion.div>
        </>
    );
};

export default FolderPanel;
