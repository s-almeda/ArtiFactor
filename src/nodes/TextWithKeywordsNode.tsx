import { useAppContext } from '../context/AppContext';
import type { NodeProps} from "@xyflow/react";
import {NodeToolbar, Position} from "@xyflow/react";
import { type Word, type Keyword, type TextWithKeywordsNode } from './types';
import { stringToWords, wordsToString } from '../utils/utilityFunctions';
import React, { useRef, useState, useEffect } from 'react';
import { useDnD } from '../context/DnDContext';
import { Bookmark, Search, Edit2, Paperclip, BookCopy } from 'lucide-react';// Eye, EyeClosed
import {motion} from 'framer-motion';
import { usePaletteContext } from '../context/PaletteContext';
import NavigationButtons from '../utils/commonComponents';

import {useNodeContext} from '../context/NodeContext';

export const WordComponent: React.FC<{ word: Word }> = ({ word }) => {
  return (
  <span style={{ position: 'relative', display: 'inline-block' }}>
    <span className='cursor-pointer rounded-sm m-0.5 p-0.5'>
      {word.value}
    </span>
  </span>
  );
};
export const KeywordComponent: React.FC<{ keyword: Keyword; handleKeywordClick: () => void }> = ({ keyword, handleKeywordClick }) => {
  const [isSelected, setIsSelected] = useState(false);

  const handleClick = () => {
    handleKeywordClick();
    setIsSelected(!isSelected);
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
      style={{ position: 'relative', display: 'inline-block' }}>
      <span
        className={`cursor-pointer 
                    rounded-sm mx-0.5 p-0
                    transition-all duration-200 
                    bg-amber-100 hover:bg-amber-200`}
        onClick={handleClick}
      >
        {keyword.value}
      </span>
    </motion.div>
  );
};
export const RelatedKeywords: React.FC<{ relatedKeywords: string[], isAIGenerated: boolean }> = ({ relatedKeywords, isAIGenerated }) => {
  const [_, setDraggableType, __, setDraggableData] = useDnD();

  const onDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    content: string
  ) => {
    event.dataTransfer.effectAllowed = "move";
    setDraggableType("text");
    setDraggableData({ content: content, provenance: "history" });
  };

  return (
    <div className="nodrag p-2 flex flex-wrap gap-0.5">
      <strong className="text-gray-900 text-sm italic mr-1">see also: </strong>
      {relatedKeywords.map((relatedKeyword, index) => (
        <div
          key={index}
          className={`text-xs p-0.5 rounded-sm cursor-grab ${isAIGenerated ? 'text-blue-50 bg-blue-500 hover:bg-blue-400' : 'text-stone-50 bg-stone-500 hover:bg-stone-400'}`}
          draggable
          onDragStart={(event) => onDragStart(event, relatedKeyword)}
        >
          {relatedKeyword}
        </div>
      ))}
    </div>
  );
};

export const KeywordDescription: React.FC<{ 
  keyword: Keyword | null;
  containerHeight?: number;
  containerWidth?: number;
  showDescription: boolean; 
  toggleDescription: () => void;
  isAIGenerated: boolean;
}> = ({ keyword, containerHeight = 100, showDescription = false, toggleDescription, isAIGenerated=false }) => { //containerWidth = 100,

  const [_, setDraggableType, __, setDraggableData] = useDnD();

  const onDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    type: string,
    content: string,
    prompt?: string
  ) => {
    event.dataTransfer.effectAllowed = "move";
    setDraggableType(type);
    setDraggableData({ content: content, prompt: prompt, provenance: "history" });
  };

  if (!keyword) return null;

  return ( 
    <div className="relative" 
    style={{
      height: `${containerHeight}px`,
      overflow: "visible"
    }}
    >{/* div to contain everything including the button */}

    {/* DIV TO CONTAIN THE DESCRIPTION ONLY. animate getting shrunk to fit behind the main node, as this section is bigger than the height.*/}
    <div id="descriptionpanel-body" className="overflow-hidden" style={{ height: `calc(${containerHeight-24}px`, zIndex: 2}}> 
      <motion.div 
        initial={{}}
        animate={{ 
          scaleY: showDescription ? 1 : 0.5, 
          rotateX: showDescription ? 0 : 90 
          
        }}
        transition={{ 
          duration: showDescription ? 0.5 : 0.1, 

          type: "spring", 
          bounce: 0.2 
        }}
        // className={`nowheel overflow-scroll nodrag border rounded-md shadow-md p-0 h-full ${isAIGenerated ? 'bg-blue-50' : 'bg-[#f2e7ce]'}`}
        className={`nodrag nowheel overflow-scroll border rounded-b-md shadow-md p-0 h-full ${isAIGenerated ? 'bg-blue-50' : 'bg-[#f4efe3] border-[#d9cdb2]'}`}
        
      >
        {/* ----- TITLE of keyword ----- */}
        <div className="flex flex-col justify-between">
          {keyword.databaseValue && (
            <div 
              draggable
              onDragStart={(event) => onDragStart(event, "text", keyword.databaseValue)}
              className={`nodrag text-sm font-bold text-gray-800 mb-2 cursor-pointer ${isAIGenerated ? ' bg-blue-100 hover:bg-blue-300' : 'hover:bg-[#D1BC97]'}`}
              style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '8px 12px',
              backgroundColor: isAIGenerated ? '#f5f5dc' : '#dbcdb4',
              borderBottom: '1px solidrgba(0, 0, 0, 0.34)'
              }}
            >
              {keyword.databaseValue}
            </div>
          )}
          {keyword.description && (
            <div
              draggable
              onDragStart={(event) => onDragStart(event, "text", keyword.description)}
              className={`p-1 m-2 flex flex-col gap-3 overflow-y-auto text-xs flex-grow ${isAIGenerated ? 'hover:bg-blue-300' : 'hover:bg-[#dbcdb4]'}`}
            >
              {keyword.description}
            </div>
          )}

          {/* RELATED KEYWORDS */}
          {keyword.relatedKeywordStrings && keyword.relatedKeywordStrings.length > 0 && (
            <RelatedKeywords relatedKeywords={keyword.relatedKeywordStrings} isAIGenerated={isAIGenerated} />
          )}
        </div>
      </motion.div>

    </div>

    {/* Bookmark button */}
    <div
      className={`
        w-12 h-10 p-1
        ${isAIGenerated ? 'bg-blue-200 hover:bg-blue-300' : 'bg-[#dbcdb4] hover:bg-yellow-300'}
        flex items-center justify-center rounded-br-md rounded-bl-md
        cursor-pointer transition-colors duration-200 
        absolute bottom-0 right-2
        ${showDescription ? (isAIGenerated ? 'bg-blue-200' : 'bg-[#dbcdb4]') : ''}`}
      style={{zIndex: 1}}
      onClick={toggleDescription}
    >
      <Bookmark size={20} className="text-gray-600" /> {/* BOOKMARK ICON */}
    </div>
  </div>
  );
};


// -- FOLDER PANEL COMPONENT (the panel that opens on the left side) --- //
const FolderPanel: React.FC<{ width: number; height: number; showFolder: boolean; toggleFolder: () => void; similarTexts: Keyword[]; isAIGenerated: boolean; }> = ({ width, height, showFolder, toggleFolder, similarTexts, isAIGenerated=false }) => {

  const [currentIndex, setCurrentIndex] = useState(0);
  // const [contentState, setContentState] = useState<string | null>(content || '');
  const [_, setDraggableType, __, setDraggableData] = useDnD();

  const onDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    type: string,
    content: string,
    prompt?: string
  ) => {
    event.dataTransfer.effectAllowed = "move";
    setDraggableType(type);
    setDraggableData({ content: content, prompt: prompt, provenance: "history" }); //everything in the folder wil  be human made
  };


  const handlePrev = () => {
    setCurrentIndex((prevIndex) => (prevIndex > 0 ? prevIndex - 1 : similarTexts.length - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex < similarTexts.length - 1 ? prevIndex + 1 : 0));
  };

  const currentText = similarTexts[currentIndex];

  return (
    <>
      {/* SEARCH ICON BUTTON -- move it all the way to the left -6 (relative to the panel) */}
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

      <motion.div
        initial={{ transform: `rotateX(-45deg)` }}
        animate={{ 
          transform: showFolder ? `rotateX(0deg)` : `rotateX(-60deg)`
        }}
        transition={{ duration: 0.3, type: "spring", bounce: 0.2 }}
        className="absolute"
      >
        <div
        className={`absolute left-0 top-0 transform -translate-x-[${width + 6}px] ${isAIGenerated ? 'bg-blue-100' : (showFolder ? 'bg-[#f2e7ce]' : 'bg-[#dbcdb4]')} rounded-md shadow-md`}
        style={{ height: `${height * 2}px`, width: `${width}px` }}
        >
            {similarTexts.length > 0 ? (
              <>
                {/* LEFT AND RIGHT BUTTONS */}
                <div className="p-3 ml-0 h-full overflow-y-auto">
                  <h2 className="text-xs font-medium text-gray-900 italic font-bold">This could be related...</h2>

                  <NavigationButtons currentIndex={currentIndex} totalItems={similarTexts.length} handlePrev={handlePrev} handleNext={handleNext} />

                    {currentText && (
                    <div className="nodrag nowheel text-gray-600 overflow-y-auto">

                      {/* TITLE */}
                      <p draggable onDragStart={(event) => onDragStart(event, "text", currentText.value)} className={`text-md font-bold ${isAIGenerated ? 'hover:bg-blue-200' : 'hover:bg-[#dbcdb4]'}`}>{currentText.value}</p>
                      <p draggable onDragStart={(event) => onDragStart(event, "text", currentText.type)} className={`italic text-xs ${isAIGenerated ? 'hover:bg-blue-200' : 'hover:bg-[#dbcdb4]'}`}>{currentText.type}</p>
                     
                     {/* DESCRIPTION */}
                      <p 
                      draggable 
                      onDragStart={(event) => onDragStart(event, "text", currentText.description, "human")}
                      className={`text-sm/5 mt-2 p-0.5 rounded-sm ${isAIGenerated ? 'hover:bg-blue-200' : 'hover:bg-[#dbcdb4]'}`}
                      >
                      {currentText.description}
                      </p>

                      {/* RELATED KEYWORDS */}
                      {currentText.relatedKeywordStrings && currentText.relatedKeywordStrings.length > 0 && (
                      <RelatedKeywords relatedKeywords={currentText.relatedKeywordStrings} isAIGenerated={isAIGenerated} />
                      )}
                    </div>
                    )}
                  </div>
                  </>
                ) : (
                <div className="p-3 ml-0 h-full overflow-y-auto flex flex-col items-center justify-center">
                <h2 className="text-xs font-medium text-gray-900 italic font-bold text-center mb-5">...We haven't found anything relevant to this prompt in our database yet...</h2>
                <div className="loader"></div>
                </div>
            )}
            
        </div>
      </motion.div>
    </>
  );
};


//------------------- TEXT WITH KEYWORDS NODE ------------------//

export function TextWithKeywordsNode({ data, selected }: NodeProps<TextWithKeywordsNode>) {
  // -- handling state on text area edit  ---//
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [content, setContent] = useState(data.content || '');
  const [isEditing, setIsEditing] = useState(false);

  // --  the array of words and keywords -- // 
  const [words, setWords] = useState<(Word | Keyword)[]>(data.words || []);
  const [similarTexts, setSimilarTexts] = useState<Keyword[]>(data.similarTexts || []);
  const [selectedKeyword, setSelectedKeyword] = useState<Keyword | null>(null);

  const {mergeNodes} = useNodeContext();


  const [width, _] = useState(200);
  const [height, __] = useState(150);

  const [showDescription, setShowDescription] = useState(false);
  const [showFolder, setShowFolder] = useState(false);

  const { backend } = useAppContext();
  const { addClippedNode, getNextPaletteIndex } = usePaletteContext();

  const [initialCheck, setInitialCheck] = useState(true);
  const [isAIGenerated, setIsAIGenerated] = useState(data.provenance === 'ai');





  const fetchSimilarTexts = async (query: string) => {
    try {
      const response = await fetch(`${backend}/api/get-similar-texts`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "query": query }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch similar texts');
      }

      const data = await response.json();
      if (data.length === 0) {
        return [];
      } else {
        const keywords = data.map((item: any) => ({
          id: item.id,
          value: item.database_value,
          description: item.description || item.full_description,
          relatedKeywordStrings: item.relatedKeywordStrings,
          type: item.type,
        }));
        return keywords;
      }
    } catch (error) {
      console.error('Error fetching similar texts:', error);
    }
  };

  const checkForKeywords = async (queryWords: Word[]): Promise<(Word | Keyword)[]> => {
    console.log('Checking for keywords in:', words);

    if (queryWords.length === 0) {
      return [];
    }

    try {
      const response = await fetch(`${backend}/api/check-for-keywords`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: queryWords.map(word => word.value).join(' ') }),
      });

      if (!response.ok) {
        throw new Error('Failed to check for keywords');
      }

      const data = await response.json();
      const result = data.words.map((item: any) => {
        if (item.id) {
          return {
            id: item.id,
            value: item.value,
            databaseValue: item.database_value,
            description: item.description,
            relatedKeywordIds: item.relatedKeywordIds,
            relatedKeywordStrings: item.relatedKeywordStrings,
            type: item.type,
          } as Keyword;
        } else {
          return { value: item.value } as Word;
        }
      });
      return result;
    } catch (error) {
      console.error('Error checking for keywords:', error);
      console.log('Fallback words array:', words);
      return words;
    }
  };

  // --- HANDLER functions for the text with keywords node --- //

  const handleChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(event.target.value);
  };

  const handleEditClick = () => {
    setContent(words.map((word: Word | Keyword) => word.value).join(' '));
    setIsEditing(true);
  };

  const handleKeyDown = async (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      const updatedWords = stringToWords(content);
      setWords(updatedWords); // set it to the updated words quickly, then add the highlights for keywords
      
      //reset all of these guys on a change
      setIsEditing(false);
      setShowDescription(false);
      setShowFolder(false);
      setSelectedKeyword(null);
      //check for keywords in the new content
      const checkedWords = await checkForKeywords(updatedWords);
      setWords(checkedWords);
      data.words = checkedWords; // Update the data object with the new words

      //check for similar texts in the new content
      const result = await fetchSimilarTexts(wordsToString(checkedWords));
      setSimilarTexts(result);
      data.similarTexts = result;
    }
  };

  const handleKeywordClick = (keyword: Keyword) => {
    setSelectedKeyword(keyword);
    setShowDescription(!showDescription);
  }

  const toggleDescription= () => {
    setShowDescription(!showDescription);
    console.log(selectedKeyword);
  };

  const toggleFolder = () => {
    setShowFolder(!showFolder);
  };

  const [showControls, setShowControls] = useState(true);
  // const hideControls = () => {
  //     setShowControls(!showControls);
  // };


  // ---------- EFFECTS ---------- //

  useEffect(() => {
    setWords(data.words || []);
    data.content = wordsToString(words);

    if (selectedKeyword === null) {
      const firstKeyword = words.find(word => 'id' in word) as Keyword;
      if (firstKeyword) {
        setSelectedKeyword(firstKeyword);
      }
    }
    if (similarTexts.length > 0) {
      setSimilarTexts(data.similarTexts || []);
    }
    //console.log('current data in this node:', data);
  }, [data.words, selectedKeyword]);

  useEffect(() => {
    if (data.words && data.words.some((word: Word | Keyword) => 'id' in word)) {
      setInitialCheck(false);
    } 
    else if (initialCheck) {
      const onCreate = async () => {
        console.log('A new TextWithKeywordsNode has been created');
        const updatedWords = await checkForKeywords(words);
        setWords(updatedWords);
        data.words = updatedWords;
        data.content = wordsToString(words);
          if(data.similarTexts && data.similarTexts.length > 0){
            setSimilarTexts(data.similarTexts);
          }
          else{
            const result = await fetchSimilarTexts(wordsToString(words));
            setSimilarTexts(result);
            data.similarTexts = result;
          }

        setInitialCheck(false);
      }
      onCreate();

    }
  }, []);

  // useEffect(() => { //WHEN CONTENT CHANGES, FIND THE SIMILAR TEXTS
  //   if (words.length > 0 && initialCheck) {
  //     fetchSimilarTexts(wordsToString(words)).then((result) => {
  //       setSimilarTexts(result);
  //       data.similarTexts = result;
  //     });
  //   }

  // },[data.words]);

  // --- ADJUST SIZE OF TEXT AREA ON EDIT --- //
  useEffect(() => {
    const maxLines = 10;
    if (textareaRef.current) {
      const textArea = textareaRef.current;
      textArea.style.height = "auto"; // Reset height to get the correct scrollHeight
      textArea.style.height = `${textArea.scrollHeight}px`;

      // Make textarea scrollable if content exceeds 4 lines
      const lineHeight = parseInt(getComputedStyle(textArea).lineHeight, 10);
      const maxHeight = lineHeight * maxLines;
      if (textArea.scrollHeight > maxHeight) {
        textArea.style.height = `${maxHeight}px`;
        textArea.style.overflowY = "scroll";
      } else {
        textArea.style.overflowY = "hidden";
      }
    }

    if (!selected) {
      setShowDescription(false);
      setShowFolder(false);
      setSelectedKeyword(null);
      setIsEditing(false);
      setShowControls(false);
    }
    else{
      setShowControls(true);
    }
    if (isEditing){
      setSelectedKeyword(null);
    }

  }, [content, data, words, isEditing, selected]);

  // Node styling based on isAIGenerated flag
  const nodeBaseClasses = `relative nowheel p-4 border ${selected ? 'ring-2 ring-blue-400' : ''}`;
  const nodeStyles = isAIGenerated 
    ? `${nodeBaseClasses} border-blue-200 bg-blue-50 rounded-lg shadow-sm`
    : `${nodeBaseClasses} border-amber-300 bg-amber-50 shadow-sm`;

  useEffect(() => {
    setIsAIGenerated(data.provenance === 'ai');
  },[data.provenance]);


  return (
    <motion.div
      initial={{ opacity: 0.2, x:0, y: 10, scale: 1.1, rotateY: Math.random()*30-15, rotateX: -75}}
      animate={{ opacity: 1, x: 0, y: 0, scale: 1, rotateY:0, rotateX: 0, scaleX:1}}
      transition={{ duration: 0.15, type: "spring", bounce: 0.1 }}
      className={``} 
    > {/*TODO: implement combinability `${data.combinable ? 'bg-yellow-50' : ''} */}
    <div className="relative" style={{ width: `${width}px` }}>

      {isEditing ? (
        <>
            <div style={{ fontSize: '0.75rem', color: 'gray', filter: 'none' }}>ENTER TO CONFIRM</div>
          <div className="text-xs p-2 border border-gray-700 rounded bg-white nowheel" style={{ width: `${width}px`, height: `${height}px` }}>
            <textarea
              className="nowheel nodrag resize-none border  overflow-auto text-inherit font-inherit  p-2 w-full h-full"
              ref={textareaRef}
              value={content}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              style={{ fontSize: 'inherit', width: '100%', height: '100%', fontStyle: 'italic', color: '#333' }}
            />
          </div>
        </>
      ) : (
        <>

        {/* --- FOLDER PANEL --- */}
        <motion.div
          initial={{ left: '-6px', transform: `scaleY(0.5)` }}
          animate={{ 
            left: showFolder ? `-${width}px` : '-6px',
            transform: `scaleY(1)`,
            opacity: showControls ? 1 : 0
          }}
          transition={{ duration: 0.2 }}
          className="absolute"
        >
          <FolderPanel
            width={width} 
            height={height} 
            showFolder={showFolder} 
            toggleFolder={toggleFolder} 
            similarTexts={similarTexts} 
            isAIGenerated={isAIGenerated}
          />
        </motion.div>

        {/* ---- NODE TOOLBAR ---- */}

        <NodeToolbar isVisible={selected} position={Position.Top}>
          <div className="flex items-center justify-center space-x-2">
                      
            <button
            className="border-5 text-gray-800 bg-white border-gray-800 shadow-lg rounded-full hover:bg-gray-400 dark:hover:bg-gray-400"
            type="button"
            onClick={() => addClippedNode(
              {
              id: getNextPaletteIndex(),
              type: 'text',
              content: wordsToString(words),
              prompt: "none"
              })}
            aria-label="Save to Palette"
            style={{ marginRight: '0px' }}
            >
            < Paperclip size={16}/>
            </button>
            {data.intersections && data.intersections.length > 1 && (
              <button
                className="border-5 text-gray-800 bg-white border-gray-800 shadow-lg rounded-full hover:bg-gray-400 dark:hover:bg-gray-400"
                type="button"
                onClick={() => mergeNodes(data.intersections)}
                aria-label="Merge"
                style={{ marginRight: '0px' }}
              >
                <BookCopy size={16} />
              </button>
            )}
          </div>
        </NodeToolbar>


        {/* ---- MAIN BODY OF NODE CONTENT -----*/}
        
        <div className={`${nodeStyles}`} style={{ zIndex: 5, height: `${height}px`, filter: "drop-shadow(3px 3px 3px rgba(0, 0, 0, 0.25))"}}>{/* Edit button */}
            
            {initialCheck ? (
                <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", height: "100%", fontStyle: "italic" }}>
                  <div className="text-xs nowheel" style={{ position: 'relative', opacity: 0.7, color: 'gray', overflow: 'auto', height: '100%', width: '100%' }}>
                    {content}
                  </div>
                  <div className="loader" style={{opacity: 0.7, position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}></div>
                </div>
            ) : (
              <>
              {/* EDIT BUTTON */}
                <button 
                  onClick={handleEditClick}
                  style={{ display: showControls ? 'block' : 'none' }}
                  className="absolute top-2 -right-3 p-1 text-gray-400 hover:text-gray-600 transition-colors rounded-full hover:bg-gray-100"
                >
                  <Edit2 size={16} />
                </button>

                <div className="nowheel p-0 overflow-y-auto overflow-x-visible h-full text-xs/4 text-gray-800 relative inline-block">
                  {words.map((word, index) => (
                    <React.Fragment key={index}>
                      {'id' in word ? ( // if its a keyword, render the keyword component
                        <KeywordComponent keyword={word} handleKeywordClick={() => handleKeywordClick(word)} />
                      ) : ( //otherwise, render a normal word
                        <WordComponent word={word} />
                      )}
                    </React.Fragment>
                  ))}
                </div>
              </>
            )}

        </div>

         {/* Description panel that appears below. motion.div animates it moving up and down.  */}
        <motion.div
          initial={{ top: 0 }}
          animate={{ 
            top: showDescription ? height : -(height)+40,
            opacity: showControls ? 1 : 0
          }}
          transition={{ type: "spring", bounce: 0.1, duration: 0.3 }}
          className="absolute mt-0"
        >
          <KeywordDescription 
            keyword={selectedKeyword} 
            containerHeight={height*2}
            containerWidth={width}
            showDescription={showDescription}
            toggleDescription={toggleDescription} 
            isAIGenerated={isAIGenerated}
          />
        </motion.div>


    </>
  )}
</div>
</motion.div>
);
};
