import React, { useState, useEffect } from 'react';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Entry, Artwork} from '../nodes/types'; // Adjust the import path as necessary
import { useDnD } from '../context/DnDContext'; // Adjust the import path as necessary
import { useAppContext } from '../context/AppContext'; // Adjust the import path as necessary

export const NavigationButtons: React.FC<{
    currentIndex: number;
    totalItems: number;
    handlePrev: () => void;
    handleNext: () => void;
    itemLabel?: string;
}> = ({ currentIndex, totalItems, handlePrev, handleNext, itemLabel }) => {
    return (
        <div className="flex items-center w-full">
            <button
                onClick={handlePrev}
                className="flex-1 flex justify-end items-center p-2 rounded-full text-gray-500 hover:text-amber-800 cursor-pointer bg-transparent"
                aria-label="Previous"
                type="button"
                style={{ background: "none" }}
            >
                <ArrowLeft size="18" />
            </button>
            <span className="flex-shrink-0 mx-4 text-gray-600 text-xs">
                {currentIndex + 1}/{totalItems}
                {typeof itemLabel === "string" && ` ${itemLabel}${totalItems > 1 ? "s" : ""}`}
            </span>
            <button
                onClick={handleNext}
                className="flex-1 flex justify-start items-center p-2 rounded-full text-gray-500 hover:text-amber-800 cursor-pointer bg-transparent"
                aria-label="Next"
                type="button"
                style={{ background: "none" }}
            >
                <ArrowRight size="18" />
            </button>
        </div>
    );
};


//TODO: create the component LiveImageDisplay that takes in an imageIds[] array and displays the images in a gallery, with buttons to navigate through them 
export const LiveImageDisplay: React.FC<{ imageIds: string[], parentNodeId?: string }> = ({ imageIds, parentNodeId }) => {
    
    const [artworks, setArtworks] = useState<Artwork[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [loading, setLoading] = useState(false);
    const {  backend } = useAppContext();
    const { setDraggableType, setDraggableData } = useDnD();

    const onDragStart = (
    event: React.DragEvent<HTMLElement>,
    type: string,
    content: string,
    prompt?: string
    ) => {
        event.dataTransfer.effectAllowed = "move";
        setDraggableType(type);
        setDraggableData({
            content: content,
            prompt: prompt,
            parentNodeId: parentNodeId,
            provenance: "history"
        });
    };



    useEffect(() => {
        if (!imageIds || imageIds.length === 0) {
            setArtworks([]);
            return;
        }
        setLoading(true);
        fetch(`${backend}/api/get-artworks-by-ids`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ entryIds: imageIds }),
        })
            .then(res => res.json())
            .then(data => {
                if (data && data.success && Array.isArray(data.artworks)) {
                    setArtworks(data.artworks);
                } else {
                    setArtworks([]);
                }
                setCurrentIndex(0);
            })
            .finally(() => setLoading(false));
    }, [imageIds, backend]);

    const handlePrev = () => {
        setCurrentIndex(idx => (idx > 0 ? idx - 1 : artworks.length - 1));
    };

    const handleNext = () => {
        setCurrentIndex(idx => (idx < artworks.length - 1 ? idx + 1 : 0));
    };

    if (loading) {
        return <div className="text-center text-gray-400 text-xs py-4">Loading images...</div>;
    }

    if (!artworks.length) {
        return <div className="text-center text-gray-400 text-xs py-4">No images to display.</div>;
    }

    const artwork = artworks[currentIndex];
    const date =
        Object.values(artwork.descriptions || {})
            .find(desc => desc.date)?.date || "";

    return (
        <div draggable className="nodrag flex flex-col items-center w-full">
            <div
            draggable
            onDragStart={e => {
                const artists = artwork.artist_names && artwork.artist_names.length > 0
                    ? artwork.artist_names.join(", ")
                    : "Unknown artist";
                const prompt = date
                    ? `"${artwork.value}" (${date}) by ${artists}`
                    : `"${artwork.value}" by ${artists}`;
                onDragStart(e, "image", artwork.image_url, prompt);
            }}
            className="w-full flex justify-center">
                <img
                    src={artwork.image_url}
                    alt={artwork.value}
                    className="max-h-32 max-w-full rounded shadow"
                    style={{ objectFit: "contain" }}
                />
            </div>

            <div className="w-full mt-2 text-left text-xs italic text-gray-600">
                <div>
                    {artwork.value}
                    {date && ` (${date})`}
                    {/* {artwork.artist_names && artwork.artist_names.length > 0 && (
                        `, ${artwork.artist_names.join(", ")}`
                    )} */}
                </div>
            </div>
                        <NavigationButtons
                currentIndex={currentIndex}
                totalItems={artworks.length}
                handlePrev={handlePrev}
                handleNext={handleNext}
                itemLabel='artwork'
            />
        </div>
    );
};

export const DynamicDescription: React.FC<{
    descriptions: Entry[];
    isAIGenerated: boolean;
    as?: 'div' | 'span';
}> = ({ descriptions, isAIGenerated, as = 'div' }) => {
    const [selectedSource, setSelectedSource] = useState<string | null>(null);
    const { setDraggableType, setDraggableData } = useDnD();

    useEffect(() => {
        setSelectedSource(
            descriptions.find((entry) => entry.source === "synth")?.source ||
            descriptions[0]?.source ||
            null
        );
    }, [descriptions]);

    const getSelectedDescription = (): string =>
        descriptions.find((entry) => entry.source === selectedSource)?.description || "";

    if (!descriptions || descriptions.length === 0) {
        return 
        //<span className="text-gray-500 italic">No descriptions available.</span>;
    }

    const handleSourceClick = (source: string) => {
        setSelectedSource(source);
    };

    const handleDragStart = (event: React.DragEvent<HTMLSpanElement>) => {
        const selectedDescription = getSelectedDescription();
        event.dataTransfer.effectAllowed = "move";
        setDraggableType("text");
        setDraggableData({ 
            content: selectedDescription, 
            provenance: selectedSource === "synth" ? "ai-description" : "description"
        });
    };

    const selectedDescription = getSelectedDescription();
    const Wrapper = as;

    return (
        <Wrapper className="mb-4">
            {selectedDescription ? (
                <span
                    className={`cursor-move inline-block p-1 text-xs 
                        ${ isAIGenerated
                            ? "hover:bg-blue-200"
                            : "hover:bg-[#dbcdb4]"
                    }`}
                    draggable={true}
                    onDragStart={handleDragStart}
                >
                    {selectedDescription}
                </span>
            ) : (
                <span className="text-gray-500 italic inline-block p-2"></span>
            )}
            
            <span className="italic text-gray-500 text-xs mt-1 block" style={{ userSelect: 'none' }}>
                Source:{" "}
                {descriptions
                    .sort((a, b) => (a.source === "synth" ? -1 : b.source === "synth" ? 1 : 0))
                    .map((entry) => (
                        <button
                            key={entry.source}
                            onClick={() => handleSourceClick(entry.source)}
                            className={`px-1 py-1 mx-0.5 rounded text-xxs ${
                                entry.source === selectedSource
                                    ? isAIGenerated
                                        ? "bg-blue-500 text-white"
                                        : "bg-stone-400 text-white"
                                    : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                            }`}
                        >
                            {entry.source}
                        </button>
                    ))}
            </span>
        </Wrapper>
    );
};

export default {
    NavigationButtons,
    DynamicDescription
}