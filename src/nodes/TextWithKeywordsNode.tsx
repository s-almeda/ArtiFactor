import { useAppContext } from '../context/AppContext';
import type { NodeProps } from "@xyflow/react";
import { type Word, type Keyword, type TextWithKeywordsNode } from './types';
import React, { useRef, useState, useEffect } from 'react';
import { useDnD } from '../context/DnDContext';
import { Bookmark, Folder, Edit2 } from 'lucide-react';
import {motion} from 'framer-motion';

export const WordComponent: React.FC<{ word: Word }> = ({ word }) => {
  return <span className='text-xs px-0.5 py-1'>{word.value}</span>;
};

export const KeywordComponent: React.FC<{ keyword: Keyword; handleKeywordClick: () => void }> = ({ keyword, handleKeywordClick }) => {
  const [isSelected, setIsSelected] = useState(false);

  const handleClick = () => {
    handleKeywordClick();
    setIsSelected(!isSelected);
  };

  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <span
        className={`cursor-pointer px-0.5 py-0 text-xs rounded-sm m-0 transition-all duration-200 
                    bg-amber-100 hover:bg-amber-200
                    ${isSelected ? 'ring-2 ring-blue-500' : ''}`}
        onClick={handleClick}
      >
        {keyword.value}
      </span>
    </span>
  );
};

export const KeywordDescription: React.FC<{ 
  keyword: Keyword | null;
  containerHeight?: number;
  containerWidth?: number;
  onClose?: () => void;
}> = ({ keyword, containerHeight = 100, containerWidth = 100, onClose }) => {

  const [_, setDraggableType, __, setDraggableData] = useDnD();
  if (!keyword) return null;

  const onDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    content: string
  ) => {
    event.dataTransfer.effectAllowed = "move";
    setDraggableType("text");
    setDraggableData({ content: content });
  };

  return (
    <div
      className="nowheel nodrag bg-white border border-gray-300 rounded-md shadow-md overflow-auto z-10"
      style={{
        maxHeight: `${containerHeight * 1.5}px`,
        width: `${containerWidth}px`,
      }}
    >

      {keyword.description && (
        <div
          draggable
          onDragStart={(event) => onDragStart(event, keyword.description)}
          className="p-3 flex flex-col gap-3 overflow-y-auto text-sm"
          style={{ maxHeight: `calc(${containerHeight * 1.5}px - 3rem)` }}
        >
          {keyword.description}
        </div>
      )}

      {/* RELATED KEYWORDS */}
      {keyword.relatedKeywordStrings && keyword.relatedKeywordStrings.length > 0 && (
        <div className="p-2">
          <div className="flex flex-wrap gap-0.5">
          <strong className="text-gray-900 text-md">Related:</strong>
            {keyword.relatedKeywordStrings.map((relatedKeyword, index) => (
              <div
                key={index}
                className="text-xs text-gray-700 p-1 bg-gray-100 hover:bg-gray-300 rounded-sm cursor-grab"
                draggable
                onDragStart={(event) => onDragStart(event, relatedKeyword)}
              >
                {relatedKeyword}
              </div>
            ))} 
          </div>
        </div>
      )}

      <button 
          className={`absolute -bottom-6
                right-4 
                w-20 h-6 z-0 bg-gray-200 
                flex items-center justify-center 
                rounded-md
                cursor-pointer 
                hover:bg-gray-300 
                transition-colors duration-200
                text-gray-600
                `}
          onClick={onClose}
          >
        <Bookmark size={16} />
      </button>
    </div>
  );
};

// -- ONLY EDIT THE FOLDER PANEL COMPONENT BELOW -- //
const FolderPanel: React.FC<{ width: number; height: number; showFolder: boolean; toggleFolder: () => void; content?: string }> = ({ width, height, showFolder, toggleFolder, content }) => {
  const [similarTexts, setSimilarTexts] = useState<Keyword[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [contentState, setContentState] = useState<string | null>(content || '');
  const { backend } = useAppContext();

  useEffect(() => {
    const fetchSimilarTexts = async () => {
      try {
        const response = await fetch(`${backend}/api/get-similar-texts`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ "query": contentState }),
        });

        if (!response.ok) {
          throw new Error('Failed to fetch similar texts');
        }

        const data = await response.json();
        if (data.length === 0) {
          setContentState(null);
        } else {
          const keywords = data.map((item: any) => ({
            id: item.id,
            value: item.database_value,
            description: item.description || item.full_description,
            relatedKeywordStrings: item.relatedKeywordStrings,
            type: item.type,
          }));
          setSimilarTexts(keywords);
        }
      } catch (error) {
        console.error('Error fetching similar texts:', error);
        setContentState(null);
      }
    };

    if (contentState) {
      fetchSimilarTexts();
    }
  }, [contentState]);

  const handlePrev = () => {
    setCurrentIndex((prevIndex) => (prevIndex > 0 ? prevIndex - 1 : similarTexts.length - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex < similarTexts.length - 1 ? prevIndex + 1 : 0));
  };

  const currentText = similarTexts[currentIndex];

  return (
    <>
      <div
        className={`absolute left-0 top-1/2 transform -translate-x-6 -translate-y-1/2 
                    w-6 h-20 bg-gray-200 flex items-center justify-center rounded-l-md
                    cursor-pointer hover:bg-gray-300 transition-colors duration-200
                    ${showFolder ? 'bg-gray-400' : ''}`}
        onClick={toggleFolder}
      >
        <Folder size={20} className="text-gray-600" /> {/* FOLDER ICON */}
      </div>

      {showFolder && (
        <div
          className={`absolute left-0 top-0 transform -translate-x-[${width}px] bg-white border border-gray-300 rounded-md shadow-md z-3`}
          style={{ height: `${height * 2}px`, width: `${width}px` }}
        >
          <div className="p-3 ml-0 h-full overflow-y-auto">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-sm font-medium text-gray-900">This is reminiscent of...</h3>
              <div
                className={`absolute left-0 transform translate-y-1/2 -translate-x-6 
                      w-6 h-20 z-0 bg-gray-200 flex items-center justify-center rounded-l-md
                      cursor-pointer hover:bg-gray-300 transition-colors duration-200
                      ${showFolder ? 'bg-gray-400' : ''}`}
                onClick={toggleFolder}
              >
                <Folder size={20} className="text-gray-600" /> {/* FOLDER ICON */}
              </div>
            </div>
            {contentState === null ? (
              <div className="nodrag nowheel text-xs text-gray-600 flex flex-col items-center">
                <p>Look-up similar concepts</p>
                <button className="mt-2 px-3 py-1 bg-gray-200 rounded-md text-gray-700 hover:bg-gray-300">Placeholder Button</button>
              </div>
            ) : (
              currentText && (
                <div className="text-xs text-gray-600">
                  <p><strong>Database Value:</strong> {currentText.value}</p>
                  <p><strong>Type:</strong> {currentText.type}</p>
                  <p><strong>Description:</strong> {currentText.description}</p>
                  <p><strong>Related Keywords:</strong> {currentText.relatedKeywordStrings.join(', ')}</p>
                  <div className="flex justify-between mt-2">
                    <button onClick={handlePrev} className="text-gray-600 hover:text-gray-800">&larr; Prev</button>
                    <button onClick={handleNext} className="text-gray-600 hover:text-gray-800">Next &rarr;</button>
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      )}
    </>
  );
};


// DO NOT TOUCH ANYTHING BELOW

export function TextWithKeywordsNode({ data, selected }: NodeProps<TextWithKeywordsNode>) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [content, setContent] = useState(data.content || '');
  const [isEditing, setIsEditing] = useState(false);
  const [words, setWords] = useState<(Word | Keyword)[]>(data.words || []);
  const [width, setWidth] = useState(200);
  const [height, setHeight] = useState(150);
  const [showDescription, setShowDescription] = useState(false);
  const [showFolder, setShowFolder] = useState(false);
  const [selectedKeyword, setSelectedKeyword] = useState<Keyword | null>(null);
  const { backend } = useAppContext();
  const isAIGenerated = false; //TODO implement: data.isAIGenerated || false;

  // --- HELPER functions for the text with keywords node --- //
  const stringToWords = (str: string): Word[] => {
    return str.split(' ').map((word) => ({ value: word } as Word));
  };
  const wordsToString = (words: Word[]): string => {
    return words.map((word) => word.value).join(' ');
  }

  const checkForKeywords = async (words: Word[]): Promise<(Word | Keyword)[]> => {
    console.log('Checking for keywords in:', words);

    try {
      const response = await fetch(`${backend}/api/check-for-keywords`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: words.map(word => word.value).join(' ') }),
      });

      if (!response.ok) {
        throw new Error('Failed to check for keywords');
      }

      const data = await response.json();
      console.log('response from server found:', data.words);
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
      setIsEditing(false);
      const checkedWords = await checkForKeywords(updatedWords);
      setWords(checkedWords);
      data.words = checkedWords; // Update the data object with the new words
    }
  };

  const handleKeywordClick = (keyword: Keyword) => {
    setSelectedKeyword(keyword);
    setShowDescription(!showDescription);
  }

  const toggleBookmark = () => {
    setShowDescription(!showDescription);
    if (showDescription === false && selectedKeyword === null) {
      // If we're opening the bookmark panel but no keyword is selected
      // Pick the first keyword from the words array if it exists
      const firstKeyword = words.find(word => 'id' in word) as Keyword;
      if (firstKeyword) {
        setSelectedKeyword(firstKeyword);
      }
    }
  };

  const toggleFolder = () => {
    setShowFolder(!showFolder);
  };

  // ---------- EFFECTS ---------- //

  useEffect(() => {
    setWords(data.words || []);
    //console.log('current data in this node:', data);
  }, [data.words]);

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
      setSelectedKeyword(null);
      setIsEditing(false);
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


  return (
    <motion.div
    initial={{ opacity: 0.2, x:0, y: 10, scale: 1.1, rotateY: Math.random()*30-15, rotateX: -75,  filter: "drop-shadow(6px 6px 6px rgba(0, 0, 0, 0.65))"}}
    animate={{ opacity: 1, x: 0, y: 0, scale: 1, rotateY:0, rotateX: 0, scaleX:1, filter: "drop-shadow(1px 2px 1px rgba(0, 0, 0, 0.25))"}}
    transition={{ duration: 0.15, type: "spring", bounce: 0.1 }}
    className={``} 
    > {/*TODO: implement combinability `${data.combinable ? 'bg-yellow-50' : ''} */}
    <div className="relative" style={{ width: `${width}px` }}>

      {isEditing ? (
        <>
            <div style={{ fontSize: '0.75rem', color: 'gray', filter: 'none' }}>ENTER TO CONFIRM</div>
          <div className="text-xs p-3 border border-gray-700 rounded bg-white nowheel" style={{ width: `${width}px`, height: `${height}px` }}>
            <textarea
              className="nowheel nodrag resize-none border  overflow-auto text-inherit font-inherit p-1 w-full h-full"
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
        <FolderPanel width={width} height={height} showFolder={showFolder} toggleFolder={toggleFolder} content={wordsToString(words)} />

        {/* MAIN BODY OF NODE CONTENT */}
        <div className={nodeStyles} style={{ height: `${height}px`, whiteSpace: 'normal', wordWrap: 'break-word'}}>
          
            {/* Edit button */}
            <button 
              onClick={handleEditClick}
              className="absolute top-2 right-2 p-1 text-gray-400 hover:text-gray-600 transition-colors rounded-full hover:bg-gray-100 z-10"
            >
              <Edit2 size={16} />
            </button>

          <div className="nowheel p-3 border border-gray-700 rounded bg-white overflow-y-auto h-full text-sm text-gray-800">
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
        </div>

       { /* Bookmark button */}
        <button 
          className={`absolute -bottom-6
          right-4 
          w-20 h-6 z-0 bg-gray-200 
          flex items-center justify-center 
          rounded-md
          cursor-pointer 
          hover:bg-gray-300 
          transition-colors duration-200
          text-gray-600
          ${showDescription || selectedKeyword == null ? 'display-none' : 'bg-gray-300'}`}
          onClick={toggleBookmark}
          >
        <Bookmark size={16} />
            </button>
      
      {/* Description panel that appears below */}
      {showDescription && selectedKeyword && (
        <div 
          className="absolute mt-0 right-0 z-0" //move it down under the node
          style={{ top: `${height}px`}}
        >
          <KeywordDescription 
            keyword={selectedKeyword} 
            containerHeight={height}
            containerWidth={width}
            onClose={() => setShowDescription(false)}
          />
        </div>
      )}
    </>
  )}
</div>
</motion.div>
);
};
