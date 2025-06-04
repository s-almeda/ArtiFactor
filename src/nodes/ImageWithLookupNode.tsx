import { Position, NodeToolbar, Handle } from "@xyflow/react";
import { useState, useEffect } from "react";
import type { NodeProps } from "@xyflow/react";
import type { ImageWithLookupNode, Artwork, Entry } from "./types";
import { usePaletteContext } from "../context/PaletteContext";
import { useDnD } from "../context/DnDContext";
import { motion } from "framer-motion";
import { Search, Bookmark, Paperclip, Expand, BookCopy, Minimize2 } from 'lucide-react'; // Eye, EyeClosed
import { useAppContext } from "../context/AppContext";
import { NavigationButtons, DynamicDescription } from '../components/NavigationButtons';
import axios from "axios";
import { useNodeContext } from "../context/NodeContext";

const FolderPanel: React.FC<{
    parentNodeId: string;
    similarArtworks: Artwork[];
    width: number;
    height: number;
    showFolder: boolean;
    toggleFolder: () => void;
    imageUrl?: string;
    isAIGenerated: boolean;
    selected?: boolean;
}> = ({
    parentNodeId,
    similarArtworks,
    width,
    height,
    showFolder,
    toggleFolder,
    isAIGenerated,
    selected
}) => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const { setDraggableType, setDraggableData } = useDnD();
    const [isExpanded, setIsExpanded] = useState(false);

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

    const handlePrev = () => {
        setCurrentIndex((prevIndex) =>
            prevIndex > 0 ? prevIndex - 1 : similarArtworks.length - 1
        );
    };

    const handleNext = () => {
        setCurrentIndex((prevIndex) =>
            prevIndex < similarArtworks.length - 1 ? prevIndex + 1 : 0
        );
    };

    const currentArtwork = similarArtworks[currentIndex];

    useEffect(() => {
        setCurrentIndex(0);
    }, [similarArtworks]);

    useEffect(() => {
        if (!selected){
            setIsExpanded(false); // Collapse when not selected     
        }
    });

    // Helper function to get the best available date
    const getArtworkDate = (artwork: Artwork): string => {
        if (!artwork.descriptions) return '';

        // Check different sources for date
        for (const source of Object.values(artwork.descriptions)) {
            if (source.date) return source.date;
        }
        return '';
    };

    // Helper function to transform descriptions for DynamicDescription component
    const getDescriptionEntries = (artwork: Artwork): Entry[] => {
        if (!artwork.descriptions) return [];

        const entries: Entry[] = [];

        for (const [source, data] of Object.entries(artwork.descriptions)) {
            if (data.description) {
                entries.push({
                    source,
                    description: data.description
                });
            }
        }

        return entries;
    };

    // Helper function to extract metadata fields from descriptions
    const getMetadataFields = (artwork: Artwork): Array<{ key: string; value: string }> => {
        if (!artwork.descriptions) return [];

        const metadataFields: Array<{ key: string; value: string }> = [];
        const excludeKeys = ['description', 'date', 'additional_information']; // Keys that shouldn't become bubbles

        for (const data of Object.values(artwork.descriptions)) {
            for (const [key, value] of Object.entries(data)) {
                if (!excludeKeys.includes(key) && value && typeof value === 'string') {
                    // Capitalize the key for display
                    const displayKey = (key.charAt(0).toUpperCase() + key.slice(1)).replace(/_/g, ' ');
                    metadataFields.push({ key: displayKey, value });
                }
            }
        }

        // Remove duplicates
        const uniqueFields = metadataFields.filter((field, index, self) =>
            index === self.findIndex((f) => f.key === field.key && f.value === field.value)
        );

        return uniqueFields;
    };

    // Helper function to format artist names
    const formatArtists = (artists: string[]): string => {
        if (!artists || artists.length === 0) return 'Unknown Artist';
        if (artists.length === 1) return artists[0];
        if (artists.length === 2) return artists.join(' and ');
        return artists.slice(0, -1).join(', ') + ', and ' + artists[artists.length - 1];
    };

    // Helper function to get the best image URL
    const getBestImageUrl = (artwork: Artwork): string => {
        // Priority: medium > large > original url > any available
        if (artwork.image_urls?.medium) return artwork.image_urls.medium;
        if (artwork.image_urls?.large) return artwork.image_urls.large;
        if (artwork.image_url) return artwork.image_url;

        // Fallback to any available URL
        const urls = Object.values(artwork.image_urls || {});
        return urls.length > 0 ? urls[0] : '';
    };

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
                    bounce: 0.3
                }}
                className="absolute z-0"
            >
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: showFolder ? 1 : 0 }}
                    transition={{ duration: showFolder ? 0.5 : 0.1, type: "spring", bounce: 0.3 }}
                    className={`nowheel absolute left-0 top-0 transform -translate-x-[${width + 6}px] ${isAIGenerated ? 'bg-blue-100' : 'bg-[#f2e7ce]'} border shadow-md z-3`}
                    style={{ height: `${height}px`, width: `${width}px` }}
                >
                    {/* Display a loader until the initial check for artworks is done */}
                    {!currentArtwork || !currentArtwork.value ? (
                        <div style={{ display: 'flex', width: '100%', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                            <div className="loader"></div>
                        </div>
                    ) : (
                        <>

                            <div className={`sticky top-0 z-10 ${isAIGenerated ? 'bg-blue-100' : 'bg-[#f2e7ce]'}`}>
                       {/* EXPAND BUTTON FOR FOLDER PANEL */}
                       <button
                            onClick={() => setIsExpanded(true)}
                            className="absolute top-2 right-5 p-1 rounded bg-gray-500 hover:bg-gray-300"
                            title="Expand reader view"
                        >
                            <Expand size={16} />
                        </button>
                                <NavigationButtons handlePrev={handlePrev} handleNext={handleNext} currentIndex={currentIndex} totalItems={similarArtworks.length} />

                            </div>

                            {currentArtwork ? (
                                <div className="p-3 pr-5 ml-0 h-[85%] overflow-y-scroll flex flex-col items-center">
                                    {/* Artwork title with date and artist */}
                                    <div
                                        draggable
                                        onDragStart={(event) => {
                                            const date = getArtworkDate(currentArtwork);
                                            const artists = formatArtists(currentArtwork.artist_names);
                                            const titleText = date
                                                ? `${currentArtwork.value} (${date}) by ${artists}`
                                                : `${currentArtwork.value} by ${artists}`;
                                            onDragStart(event, "text", titleText);
                                        }}
                                        className={`nodrag rounded-md p-1 ${isAIGenerated ? 'hover:bg-blue-300' : 'hover:bg-[#dbcdb4]'} text-center`}
                                    >
                                        <span className="artwork-title text-md font-medium text-gray-900 italic font-bold block">
                                            {currentArtwork.value}
                                        </span>
                                        {getArtworkDate(currentArtwork) && (
                                            <span className="text-gray-700 block">
                                                {getArtworkDate(currentArtwork)}
                                            </span>
                                        )}
                                        <span className="text-md text-black-700 block">
                                            by {formatArtists(currentArtwork.artist_names)}
                                        </span>
                                    </div>

                                    {/* Artwork image */}
                                    <img
                                        draggable
                                        onDragStart={(event) => {
                                            const date = getArtworkDate(currentArtwork);
                                            const artists = formatArtists(currentArtwork.artist_names);
                                            const prompt = date
                                                ? `"${currentArtwork.value}" (${date}) by ${artists}`
                                                : `"${currentArtwork.value}" by ${artists}`;
                                            onDragStart(event, "image", getBestImageUrl(currentArtwork), prompt);
                                        }}
                                        className={`artwork-image nodrag rounded-md m-2 border-2 ${isAIGenerated ? 'border-blue-200' : 'border-[#dbcdb4]'}`}
                                        src={getBestImageUrl(currentArtwork)}
                                        alt={`${currentArtwork.value} by ${formatArtists(currentArtwork.artist_names)}`}
                                        style={{ maxHeight: '90%', width: 'auto', objectFit: 'cover' }}
                                    />

                                    {/* Metadata fields as keyword bubbles */}
                                    {getMetadataFields(currentArtwork).length > 0 && (
                                        <div className="mt-2 text-center">
                                            <div className="flex flex-wrap justify-center gap-1">
                                                {getMetadataFields(currentArtwork).map((field, index) => (
                                                    <span
                                                        key={`${field.key}-${index}`}
                                                        draggable
                                                        onDragStart={(event) => onDragStart(event, "text", `${field.key}: ${field.value}`)}
                                                        className={`nodrag text-xs px-2 py-1 rounded ${isAIGenerated ? 'bg-blue-200 hover:bg-blue-300' : 'bg-[#dbcdb4] hover:bg-[#b39e79]'} cursor-move`}
                                                    >
                                                        {field.key}: {field.value}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Keywords/Tags */}
                                    {currentArtwork.relatedKeywordStrings && currentArtwork.relatedKeywordStrings.length > 0 && (
                                        <div className="mt-2 text-center">
                                            <div className="flex flex-wrap justify-center gap-1">
                                                {currentArtwork.relatedKeywordStrings.slice(0, 5).map((keyword, index) => (
                                                    <span
                                                        key={index}
                                                        draggable
                                                        onDragStart={(event) => onDragStart(event, "text", keyword)}
                                                        className={`nodrag text-xs px-2 py-1 rounded ${isAIGenerated ? 'bg-blue-200 hover:bg-blue-300' : 'bg-[#dbcdb4] hover:bg-[#b39e79]'} cursor-move`}
                                                    >
                                                        {keyword}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Description using DynamicDescription component */}
                                    <div className="mt-2 text-center">
                                        <DynamicDescription
                                            descriptions={getDescriptionEntries(currentArtwork)}
                                            isAIGenerated={isAIGenerated}
                                            as="div"
                                        />
                                    </div>

                                    {/* Rights information */}
                                    {currentArtwork.rights && (
                                        <p className="text-xs text-gray-500 mt-2 text-center">
                                            {currentArtwork.rights}
                                        </p>
                                    )}
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

            {/* EXPANDED READER VIEW OVERLAY */}
      {isExpanded && (
        <>
            
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-50"
            onClick={() => setIsExpanded(false)}
          />
          
          {/* Expanded content */}
          
          <div
            className="fixed z-50 bg-white border border-gray-700 rounded shadow-lg p-5 w-[600px] h-[700px] overflow-y-auto"
            style={{
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)'
            }}
          >
            {/* Close button */}
            <button
              onClick={() => setIsExpanded(false)}
              className="absolute top-2 right-2 p-1 rounded hover:bg-gray-400 bg-gray-700"
            >
              <Minimize2 size={20} className="text-white" />
            </button>
         <NavigationButtons handlePrev={handlePrev} handleNext={handleNext} currentIndex={currentIndex} totalItems={similarArtworks.length} />

            
            {/* Content in reader view */}
            <div className="text-black">
              <h2 className="text-2xl font-bold text-center mb-4">
                {currentArtwork.value}
                {getArtworkDate(currentArtwork) && ` (${getArtworkDate(currentArtwork)})`}
              </h2>
              
              <p className="text-lg text-center mb-4">
                by {formatArtists(currentArtwork.artist_names)}
              </p>
              
              <img
                className="mx-auto mb-4 max-h-[300px] rounded"
                src={getBestImageUrl(currentArtwork)}
                alt={`${currentArtwork.value} by ${formatArtists(currentArtwork.artist_names)}`}
              />
              
              {/* Metadata */}
              {getMetadataFields(currentArtwork).length > 0 && (
                <div className="mb-4">
                  <div className="space-y-1">
                    {getMetadataFields(currentArtwork).map((field, index) => (
                      <p key={`${field.key}-${index}`} className="text-sm">
                        <span className="font-medium">{field.key}:</span> {field.value}
                      </p>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Keywords */}
              {currentArtwork.relatedKeywordStrings && currentArtwork.relatedKeywordStrings.length > 0 && (
                <div className="mb-4">
                  <p className="text-sm">{currentArtwork.relatedKeywordStrings.join(', ')}</p>
                </div>
              )}
              
              {/* Description */}
              {getDescriptionEntries(currentArtwork).length > 0 && (
                <div className="mb-4">
                  <h3 className="font-semibold mb-2">Description:</h3>
                  <div className="text-sm space-y-2">
                    {getDescriptionEntries(currentArtwork).map((entry) => (
                      <div key={entry.source}>
                        <p className="font-medium text-gray-600">{entry.source}:</p>
                        <p>{entry.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Rights */}
              {currentArtwork.rights && (
                <p className="text-xs text-gray-500 mt-4">
                  {currentArtwork.rights}
                </p>
              )}
            </div>
          </div>
        </>
      )}

        </>
    );
};

const DescriptionPanel: React.FC<{ 
    content?: string 
    containerHeight: number;
    containerWidth: number;
    showDescription: boolean; 
    parentNodeId: string;
    toggleDescription: () => void; 
    isAIGenerated: boolean;
}> = ({content, showDescription = false, containerHeight = 100, containerWidth=100, parentNodeId, toggleDescription, isAIGenerated=false }) => {
    
    const { setDraggableType, setDraggableData } = useDnD();
    
    const onDragStart = (
        event: React.DragEvent<HTMLDivElement>, 
        content: string,
        type: string = "text",
        prompt?: string
    ) => {
      event.dataTransfer.effectAllowed = "move";
      setDraggableType(type);
      setDraggableData({ content: content, prompt: prompt, provenance: isAIGenerated? "ai" : "history", parentNodeId: parentNodeId });
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
                    className={`nowheel p-3 overflow-scroll border rounded-b-md shadow-md p-0 h-full ${isAIGenerated ? 'bg-blue-50' : 'bg-[#f4efe3] border-[#998056]'}`}
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

export function ImageWithLookupNode({ id, data, selected, dragging }: NodeProps<ImageWithLookupNode>) {
    const { mergeNodes } = useNodeContext();
    const { addClippedNode, getNextPaletteIndex } = usePaletteContext(); 
    const [imageUrl, setImageUrl] = useState("");
    const [width, _] = useState(150);
    const [height, __] = useState(150);
    
    const [showDescription, setShowDescription] = useState(false);
    const [showFolder, setShowFolder] = useState(false);

    const [isAIGenerated, setIsAIGenerated] = useState(false);


    const [isImageEnlarged, setIsImageEnlarged] = useState(false);

    const toggleEnlargedImage = () => {
        setIsImageEnlarged(!isImageEnlarged);
    };



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

    const defaultArtworks: Artwork[] = [
      {
        image_id: "",
        image_url: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/660px-No-Image-Placeholder.svg.png?20200912122019",
        image_urls: {
          original: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/660px-No-Image-Placeholder.svg.png?20200912122019"
        },
        filename: "",
        value: "",
        artist_names: [],
        descriptions: {
          default: {
            date: "",
            description: "No description available."
           }
        },
        relatedKeywordIds: [],
        relatedKeywordStrings: [],
        rights: undefined,
        distance: undefined
      }
    ];
    

    const [similarArtworks, setSimilarArtworks] = useState<Artwork[]>(data.similarArtworks || defaultArtworks);
    const { backend, userID, admins, condition} = useAppContext();

    const fetchSimilarArtworks = async (imageUrl: string): Promise<Artwork[]> => {
        try {
            console.log("sending this image to the backend for lookup:", imageUrl);
            const response = await axios.post(`${backend}/api/get-similar-images`, {
              image: imageUrl
            }, {
              headers: {
                "Content-Type": "application/json",
              },
            });
            
            // response.data is already parsed JSON (array of objects)
            const responseData = response.data as any[];

            const artworks: Artwork[] = responseData.map((item: any) => ({
                image_id: item.image_id,
                image_url: item.image_url,
                image_urls: item.image_urls || {},
                filename: item.filename,
                value: item.value,
                artist_names: Array.isArray(item.artist_names)
                    ? item.artist_names.flatMap((name: string) => {
                        // Some artist_names are JSON strings, e.g. '["Name"]'
                        try {
                            const parsed = JSON.parse(name);
                            if (Array.isArray(parsed)) return parsed;
                            if (typeof parsed === "string") return [parsed];
                            return [];
                        } catch {
                            // fallback: just use as string
                            return [name];
                        }
                    })
                    : [],
                descriptions: item.descriptions || {},
                relatedKeywordIds: item.relatedKeywordIds || [],
                relatedKeywordStrings: item.relatedKeywordStrings || [],
                rights: item.rights,
                distance: item.distance,
            }));
            console.log('fetched similar artworks from backend:', artworks);
            return artworks;
        } catch (error) {
            console.error('Error fetching similar artworks:', error);
            return defaultArtworks;
        }
    };

    const [initialCheck, setInitialCheck] = useState(true);

    useEffect(() => {   
        setImageUrl(data.content || 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/660px-No-Image-Placeholder.svg.png?20200912122019');
        if (!selected) {
            setShowDescription(false);
            setShowFolder(false);
            setShowControls(false);
            setIsImageEnlarged(false); // close pop up
          }
          else {
            setShowControls(true);
            if (data.similarArtworks && data.similarArtworks.length > 0 && !dragging){
                setShowFolder(true);
            }
            if (data.prompt && !dragging){
                setShowDescription(true
                );
            }

          }
    }, [selected, imageUrl, data.content, data.similarArtworks]);

    
    useEffect(() => {
        if (data.similarArtworks && data.similarArtworks.length > 0 && condition === "experimental") {
            //console.log('setting new similar artworks data:', data.similarArtworks);
            setSimilarArtworks(data.similarArtworks);
        }
    }, [JSON.stringify(data.similarArtworks), condition],);


    //update isAIGenerated flag if data.provenance changes
    useEffect(() => {
        setIsAIGenerated(data.provenance === 'ai');
      },[data.provenance]);

    
    useEffect(() => {
        const fetchArtworks = async () => {
                const artworks = await fetchSimilarArtworks(imageUrl);
                data.similarArtworks = artworks;
                setSimilarArtworks(artworks);
                setInitialCheck(false);
        };
        if (condition === "experimental" && initialCheck && (!data.similarArtworks || data.similarArtworks.length <= 0) && imageUrl !== "") {
            fetchArtworks();
        }
    }, [initialCheck, data.similarArtworks, imageUrl, condition]);

        


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
            {condition === "experimental" && (
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
                <FolderPanel parentNodeId={id} 
                similarArtworks={similarArtworks} 
                width={width*2} 
                height={height*2} 
                showFolder={showFolder} 
                toggleFolder={toggleFolder} 
                isAIGenerated={isAIGenerated} 
                selected={selected}
                />
            </motion.div>
            )}
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
                prompt: data.prompt || "",
                provenance: data.provenance || "user",
                parentNodeId: data.parentNodeId || id,
                similarArtworks: similarArtworks
                }
                )}
                aria-label="Save to Palette"
                style={{  }}
            >
             < Paperclip size={16}/>
            </button>

            {data.intersections && data.intersections.length > 1 && (
                <button
                className="border-5 text-gray-800 bg-white border-gray-800 shadow-lg rounded-full hover:bg-gray-400 dark:hover:bg-gray-400"
                type="button"
                onClick={() => mergeNodes(data.intersections)}
                aria-label="Merge"
                style={{ marginRight: "0px" }}
                >
                <BookCopy size={16} />
                </button>
            )}

            <button
                className="border-5 text-gray-800  bg-white border-gray-800 shadow-lg rounded-full hover:bg-[#dbcdb4]"
                type="button"
                onClick={toggleEnlargedImage}
                aria-label="Enlarge Image"
            >
                <Expand size={16} />
            </button>
            </div>

            { (userID && admins.includes(userID)) && //only show if we're in admin mode
                <button
                className="border-5 text-gray-800 bg-white border-gray-800 shadow-lg rounded-full hover:bg-[#dbcdb4]"
                type="button"
                onClick={() => console.log(id, data)}
                aria-label="Print Node Data"
                >
                <Bookmark size={16} />
                </button>
            }

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

            
            <Handle
        type="source"
        position={Position.Bottom}
        id="a"
        isConnectable={false}
        className="invisible"
        onConnect={(params) => console.log('handle onConnect', params)}
          />
          <Handle
        type="target"
        position={Position.Top}
        id="b"
        isConnectable={false}
        className="invisible"
        onConnect={(params) => console.log('handle onConnect', params)}
          />

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
            parentNodeId={id}
            content={data.prompt}
            isAIGenerated={isAIGenerated} 
            />

            </motion.div>


            {/* Enlarged Image Overlay */}
            {isImageEnlarged && (
            <motion.div
                className="fixed top-0 left-0 w-screen-[50vw] h-screen-[50vw] bg-black/70 flex justify-center items-center z-50"
                onClick={() => setIsImageEnlarged(false)}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
            >
                <img src={imageUrl} className="max-w-[45vw] max-h-[45vh] rounded-lg shadow-xl" />
            </motion.div>
            )}
        </motion.div>
    );
}