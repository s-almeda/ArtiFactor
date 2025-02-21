import { Position, NodeToolbar } from "@xyflow/react";
import { useState, useEffect } from "react";
import type { NodeProps } from "@xyflow/react";
import type { ImageWithLookupNode, Artwork } from "./types";
import { usePaletteContext } from "../context/PaletteContext";
import { useDnD } from "../context/DnDContext";
import { motion } from "framer-motion";
import { Search, Bookmark, Paperclip} from 'lucide-react'; // Eye, EyeClosed
import { useAppContext } from "../context/AppContext";
import NavigationButtons from '../utils/commonComponents';
import axios from "axios";

const FolderPanel: React.FC<{ similarArtworks: Artwork[]; width: number; height: number; showFolder: boolean; toggleFolder: () => void; imageUrl?: string; isAIGenerated: boolean }> = ({ similarArtworks, width, height, showFolder, toggleFolder, isAIGenerated }) => {
    const [currentIndex, setCurrentIndex] = useState(0);

    const [_, setDraggableType, __, setDraggableData] = useDnD();

    const onDragStart = (
        event: React.DragEvent<HTMLDivElement>,
        type: string,
        content: string,
        prompt?: string
    ) => {
        event.dataTransfer.effectAllowed = "move";
        setDraggableType(type);
        setDraggableData({ content: content, prompt: prompt });
    };

    const handlePrev = () => {
        setCurrentIndex((prevIndex) => (prevIndex > 0 ? prevIndex - 1 : similarArtworks.length - 1));
    };

    const handleNext = () => {
        setCurrentIndex((prevIndex) => (prevIndex < similarArtworks.length - 1 ? prevIndex + 1 : 0));
    };

    const currentArtwork = similarArtworks[currentIndex];

    return (
        <>
            {/* FOLDER BUTTON */}
            <div
                className={`absolute left-0 top-2 transform -translate-x-7 -translate-y-2 
                                        w-12 h-20 p-1
                                        ${isAIGenerated ? 'bg-blue-200 hover:bg-blue-300' : 'bg-[#dbcdb4] hover:bg-[#b39e79]'}
                                        flex items-center justify-left rounded-l-md
                                        cursor-pointer transition-colors duration-200 
                                        ${showFolder ? (isAIGenerated ? 'bg-blue-200' : 'bg-[#dbcdb4]') : ''}`}
                onClick={toggleFolder}
            >
                <Search size={20} className="text-gray-600" /> {/* FOLDER ICON */}
            </div>

            {/* FOLDER BODY PANEL */}
            <motion.div
                initial={{ transform: `rotateX(-45deg) scaleX(0.5)` }}
                animate={{ 
                    transform: showFolder ? `rotateX(0deg) scaleX(1)` : `rotateX(-60deg) scaleX(0.5)`
                }}
                transition={{ 
                    duration: showFolder ? 0.5 : 0.1, 
                    type: "spring", 
                    bounce: 0.3 }}
                className="absolute z-0 "
            >
                
                <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: showFolder ? 1 : 0 }}
                transition={{ duration: showFolder ? 0.5 : 0.1, type: "spring", bounce: 0.3 }}
                    className={`nowheel absolute left-0 top-0 transform -translate-x-[${width + 6}px] ${isAIGenerated ? 'bg-blue-100' : 'bg-[#f2e7ce]'} border shadow-md z-3`}
                    style={{ height: `${height}px`, width: `${width}px` }}
                >
                    {/* Display a loader until the initial check for keywords is done */}
                    {currentArtwork.title === "" ? (
                        <div style={{ display: 'flex', width:'100%', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                            <div className="loader"></div>
                        </div>
                    ) : (
                        <>
                            <div className="sticky top-0 z-10 ${isAIGenerated ? 'bg-blue-100' : 'bg-[#f2e7ce]'}">
                                <NavigationButtons handlePrev={handlePrev} handleNext={handleNext} currentIndex={currentIndex} totalItems={similarArtworks.length} />
                            </div>
                            
                            {currentArtwork ? (
                                <div className="p-3 pr-5 ml-0 h-[85%] overflow-y-scroll flex flex-col items-center">
                                    <h2 
                                    draggable
                                    onDragStart={(event) => onDragStart(event, "text", currentArtwork.title)}
                                    className={`artwork-title nodrag rounded-md p-1 ${isAIGenerated ? 'hover:bg-blue-300' : 'hover:bg-[#dbcdb4]'} text-md font-medium text-gray-900 italic font-bold text-center`}>
                                        {currentArtwork.title} ({currentArtwork.date})
                                    </h2>
                                    {/* artwork's representative image */}
                                    <img
                                        draggable
                                        onDragStart={(event) => onDragStart(event, "image", currentArtwork.image, `"${currentArtwork.title}"(${currentArtwork.date}) by ${currentArtwork.artist}` )}
                                        className={`artwork-image nodrag rounded-md m-2 border-2 ${isAIGenerated ? 'border-blue-200' : 'border-[#dbcdb4]'}`}
                                        src={currentArtwork.image}
                                        alt={`${currentArtwork.title} by ${currentArtwork.artist}`}
                                        style={{ maxHeight: '90%', width: 'auto', objectFit: 'cover' }}
                                    />

                                    <p 
                                    draggable
                                    onDragStart={(event) => onDragStart(event, "text", currentArtwork.artist)}
                                    className={`nodrag ${isAIGenerated ? 'hover:bg-blue-300' : 'hover:bg-[#dbcdb4]'} text-xs mt-2 text-center`}>{currentArtwork.artist}</p>
                                    {/* {currentArtwork.keywords.map((keyword) => (
                                        <p key={keyword.id} className="text-xs">{`${keyword.type}: ${keyword.value}`}</p>
                                    ))} */}
                                    <p 
                                    draggable
                                    onDragStart={(event) => onDragStart(event, "text", currentArtwork.description)}
                                    className={`nodrag ${isAIGenerated ? 'hover:bg-blue-300' : 'hover:bg-[#dbcdb4]'} text-xs mt-2 text-center`}>{currentArtwork.description}</p>
                                </div>
                            ) : (
                                <div className="p-3 ml-0 h-full overflow-y-auto">
                                    <h2 className="text-xs font-medium text-gray-900 italic font-bold">
                                        No content
                                    </h2>
                                </div>
                            )}
                        </>
                    )}
                </motion.div>
            </motion.div>
        </>
    );
};
//TODO: if AIGenerated, use ai gen styles, and switch between saying "Prompt" or "Alt text" in the description panel
const DescriptionPanel: React.FC<{ 
    content?: string 
    containerHeight: number;
    containerWidth: number;
    showDescription: boolean; 
    toggleDescription: () => void; 
    isAIGenerated: boolean;
}> = ({content, showDescription = false, containerHeight = 100, containerWidth=100, toggleDescription, isAIGenerated=false }) => {
    
    const [_, setDraggableType, __, setDraggableData] = useDnD()
    
    const onDragStart = (
        event: React.DragEvent<HTMLDivElement>, 
        content: string
    ) => {
      event.dataTransfer.effectAllowed = "move";
      setDraggableType("text");
      setDraggableData({ content: content });
    };   

    if (!content) return null;
    
    return (
        <div className="relative" 
        style={{
          height: `${containerHeight}px`,
          overflow: "visible"
        }}
        >{/* div to contain everything including the button */}
    
        <div className="overflow-hidden" 
        style={{ 
            height: `calc(${containerHeight - 24}px)`,
            width: `${containerWidth}px` 
            }}> 
                <motion.div
                    initial={{}}
                    animate={{ 
                        scaleY: showDescription ? 1 : 0.5, 
                        rotateX: showDescription ? 0 : 90 
                    }}
                    transition={{ 
                        duration: showDescription ? 0.5 : 0.1, 
                        opacity: showDescription ? 1 : 0,
                        type: "spring", 
                        bounce: 0.2 
                    }}
                    className={`nowheel p-3 overflow-scroll border rounded-md shadow-md p-0 h-full ${isAIGenerated ? 'bg-blue-50' : 'bg-[#f4efe3] border-[#998056]'}`}
                >
                    <div className="flex flex-col justify-between">
                    {isAIGenerated ? 'Prompt:' : ''}
                        <div className={`nodrag ${isAIGenerated ? 'hover:bg-blue-200' : 'hover:bg-[#dbcdb4]'} rounded-md p-1 flex flex-col gap-3 overflow-y-auto text-xs flex-grow`}>
                           <p 
                            draggable
                            onDragStart = {(event) => onDragStart(event, content)}
                           >
                            {content}
                           </p>

                        </div>
                    </div>
                </motion.div>
            </div>
            <div
            className={`
                    w-12 h-10 p-1 mt-5
                    ${isAIGenerated ? 'bg-blue-200 hover:bg-blue-300' : 'bg-[#dbcdb4] hover:bg-[#b39e79]'}
                    flex items-center justify-center rounded-br-md rounded-bl-md
                    cursor-pointer transition-colors duration-200 
                    absolute bottom-0 right-2
                    ${showDescription ? (isAIGenerated ? 'bg-blue-200' : 'bg-[#dbcdb4]') : ''}`}
            onClick={toggleDescription}
            >
            <Bookmark size={20} className="text-gray-600" /> {/* BOOKMARK ICON */}
            </div>
    </div>
        
    );
};

export function ImageWithLookupNode({ data, selected }: NodeProps<ImageWithLookupNode>) {
    const { addClippedNode, getNextPaletteIndex } = usePaletteContext(); 
    const [imageUrl, setImageUrl] = useState("");
    const [width, _] = useState(150);
    const [height, __] = useState(150);
    
    const [showDescription, setShowDescription] = useState(false);
    const [showFolder, setShowFolder] = useState(false);

    const [isAIGenerated, setIsAIGenerated] = useState(false);



    useEffect(() => {
        if (data.content) {
            setImageUrl(data.content);
        }
    }, [data.content]);

    const toggleDescription = () => {
        setShowDescription(!showDescription);
    };

    const toggleFolder = () => {
        setShowFolder(!showFolder);
    };

    const [showControls, setShowControls] = useState(true); 
    // const hideControls = () => {  //just set show controls using selected/not selected
    //     setShowControls(!showControls);
    // };

    //--- similar artworks data --- //

    const defaultArtworks: Artwork[] = [        {
        title: "",
        date: 404,
        artist: "",
        keywords: [ ],
        description: "This painting is a depiction of the port of Le Havre at sunrise, with small rowboats in the foreground and ships and cranes in the background. The orange sun is shown as a distinct circle, reflecting on the water below. This piece gave the Impressionist movement its name when critics seized upon the title of this painting to give the entire movement a derisive moniker.",
        image: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/660px-No-Image-Placeholder.svg.png?20200912122019"
    }];
    

    const [similarArtworks, setSimilarArtworks] = useState<Artwork[]>(data.artworks || defaultArtworks);
    const { backend } = useAppContext();

    const fetchSimilarArtworks = async () => {
        try {
            console.log("sending this image to the backend for lookup:", imageUrl);
            const response = await axios.post(`${backend}/api/get-similar-images`, {
              image: imageUrl
            }, {
              headers: {
                "Content-Type": "application/json",
              },
            });
            const responseData = await JSON.parse(response.data);
            const artworks: Artwork[] = responseData.map((item: any) => ({
                title: item.title || "Unknown",
                date: item.date || "Unknown",
                artist: item.artist || "Unknown",
                keywords: [
                    {
                        id: `genre-${Date.now()}`, // todo, this should be replaced with the actual genre ids from Artsy!
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
            console.log('Similar artworks:', artworks);
            data.artworks=artworks;
            setSimilarArtworks(artworks);
            setInitialCheck(false);
        } catch (error) {
            console.error('Error fetching similar artworks:', error);
            setSimilarArtworks(defaultArtworks);
        }
    };

    const [initialCheck, setInitialCheck] = useState(true);

    useEffect(() => {   
        setImageUrl(data.content || 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/660px-No-Image-Placeholder.svg.png?20200912122019');

        if (!selected) {
            setShowDescription(false);
            setShowFolder(false);
            setShowControls(false);
          }
          else {
            setShowControls(true);
          }
    }, [selected, imageUrl, data.content]);

    
    useEffect(() => {
        // console.log("initial check", initialCheck);
        if (data.artworks && initialCheck){
            setSimilarArtworks(data.artworks);
            setInitialCheck(false);
        }
        else if (initialCheck && imageUrl!=='') {
            fetchSimilarArtworks();
        }
    }, [initialCheck, imageUrl]);

    //update isAIGenerated flag if data.provenance changes
    useEffect(() => {
        setIsAIGenerated(data.provenance === 'ai');
      },[data.provenance]);


        // Node styling based on isAIGenerated flag
    const nodeBaseClasses = `relative nowheel p-0.5`;
    const nodeStyles = isAIGenerated 
        ? `${nodeBaseClasses} bg-blue-400 rounded-xl   ${selected ? 'ring-2 ring-blue-300' : ''}`
        : `${nodeBaseClasses} bg-stone-500  ${selected ? 'ring-2 ring-[#dbcdb4] border-[#b39e79]' : ''}`;


    return (
        <motion.div
            initial={{ opacity: 0, x: 0, y: 10, scale: 1.1, rotateY: -45, filter: "drop-shadow(10px 10px 10px rgba(0, 0, 0, 0.55))" }}
            animate={{ opacity: 1, x: 0, y: 0, scale: 1, rotateY: 0, scaleX: 1, filter: "drop-shadow(1px 2px 1px rgba(0, 0, 0, 0.15))" }}
            transition={{ duration: 0.3, type: "spring", bounce: 0.1 }}
            className="drag-handle__invisible bg-transparent"
        >
            {/*----folder panel----*/}

            <motion.div
            initial={{ left: '-6px', transform: `scaleY(0.5)` }}
            animate={{ 
                left: showFolder ? `-${width*2 - 10}px` : '-6px',
                transform: `scaleY(1)`,
                opacity: showControls ? 1 : 0
            }}
            transition={{ duration: 0.2 }}
            className="absolute"
            >
            
            <FolderPanel similarArtworks={similarArtworks} width={width*2} height={height*2} showFolder={showFolder} toggleFolder={toggleFolder} isAIGenerated={isAIGenerated} />
            </motion.div>
            {/* end folder panel */}


            <NodeToolbar isVisible={selected} position={Position.Top}>
                <div className="flex items-center justify-center space-x-2">
                <button
                    className="border-5 text-gray-800  bg-white border-gray-800 shadow-lg rounded-full hover:bg-[#dbcdb4]"
                    type="button"
                    onClick={() => addClippedNode(
                    {
                        id: getNextPaletteIndex(),
                        type: 'image',
                        content: imageUrl,
                        prompt: data.prompt || ""
                    }
                    )}
                    aria-label="Save to Palette"
                    style={{  }}
                >
                 < Paperclip size={16}/>
                </button>
                {/* <button
                    className="border-5 bg-white"
                    type="button"
                    onClick={() => hideControls()}
                    aria-label="Hide Controls"
                    style={{height:'30px' }}
                >
                    {showControls ? <Eye size={16} className="text-gray-600" /> : <EyeClosed size={16} className="text-gray-600" />}
                </button> */}
                </div>
            </NodeToolbar>

            {/* CONTAINER FOR MAIN NODE BODY IMAGE */}
            <div
            className={`bg-transparent rounded-full ${isAIGenerated ? 'rounded-xl' : ''}`}
            style={{
                width: `${width}px`,
                height: `${height}px`,
                position: "relative",
            }}
            >

             {/* the main node image */}
            <img
                className={`${nodeStyles}`}
                src={imageUrl}
                alt={data.prompt || "generated image"}
                style={{ width: "100%", height: "100%", objectFit: "cover", objectPosition: "top" }}
            />
            </div>


            <motion.div
                initial={{ top: 0 }}
                animate={{ 
                top: showDescription ? height : 40,
                opacity: showControls ? 1 : 0
                
                }}
                transition={{ type: "spring", bounce: 0.1, duration: 0.3 }}
                className="absolute mt-0"
            >
                <DescriptionPanel 
                containerHeight={height}
                containerWidth={width}
                showDescription={showDescription} 
                toggleDescription={toggleDescription} 
                content={data.prompt}
                isAIGenerated={isAIGenerated} 
                />

            </motion.div>
        </motion.div>
    );
}
