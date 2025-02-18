import { useAppContext } from '../context/AppContext';
import type { NodeProps } from "@xyflow/react";
import { type Word, type Keyword, type TextWithKeywordsNode } from './types';
import React, { useRef, useState, useEffect } from 'react';
import { useDnD } from '../context/DnDContext';

export const WordComponent: React.FC<{ word: Word }> = ({ word }) => {
  return <span>{word.value}</span>;
};
export const KeywordComponent: React.FC<{ keyword: Keyword; handleKeywordClick: () => void }> = ({ keyword, handleKeywordClick }) => {
  const [isSelected, setIsSelected] = useState(false);

  useEffect(() => {
    if (!keyword.relatedKeywordStrings || keyword.relatedKeywordStrings.length === 0) {
      keyword.relatedKeywordStrings = ["insert", "related", "words", "here"];
    }
  }, [keyword.relatedKeywordStrings]);

  const handleClick = () => {
    handleKeywordClick();
    setIsSelected(!isSelected);
  };

  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <span
        onClick={handleClick}
        style={{
          backgroundColor: 'yellow',
          cursor: 'pointer',
          padding: '1px 1px',
          borderRadius: '2px',
          margin: '0 1px',
          outline: isSelected ? '2px solid blue' : 'none',
        }}
      >
        {keyword.value}
      </span>
    </span>
  );
};

export const KeywordDescription: React.FC<{ keyword: Keyword | null }> = ({ keyword }) => {
  const [_, setDraggableType, __, setDraggableData] = useDnD();
  if (!keyword) return null;

  const onDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    content: string
  ) => {
    event.dataTransfer.effectAllowed = "move";
    setDraggableType("text");
    setDraggableData({content: content});
  };

  return (
    <div className="nowheel nodrag"
      style={{
        position: 'absolute',
        top: '100%',
        left: '0',
        backgroundColor: 'white',
        border: '1px solid black',
        padding: '10px',
        borderRadius: '4px',
        zIndex: 1,
        width: '200px',
      }}
    >
      {keyword.databaseValue && <div><strong>{keyword.databaseValue}</strong></div>}

      {keyword.description && (
        <div draggable 
        onDragStart={(event) => onDragStart(event, keyword.description)} style={{ maxHeight: '200px', overflowY: 'scroll' }}>
          <strong>Description:</strong> {keyword.description}
        </div>
      )}

      {keyword.relatedKeywordStrings && keyword.relatedKeywordStrings.length > 0 && (
        <div>
          <strong>Related Keywords:</strong>
         
          <ul>
            {keyword.relatedKeywordStrings.map((relatedKeyword, index) => (
               <div key={index} draggable onDragStart={(event) => onDragStart(event, relatedKeyword)} >
              <li>{relatedKeyword}</li>
              </div>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
  



export function TextWithKeywordsNode({ data, selected}: NodeProps<TextWithKeywordsNode>) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [content, setContent] = useState(data.content || '');
  const [isEditing, setIsEditing] = useState(false);
  const [words, setWords] = useState<(Word | Keyword)[]>(data.words || []);
  const [width, setWidth] = useState(200);
  const [height, setHeight] = useState(100);
  const [showDescription, setShowDescription] = useState(false);
  const [selectedKeyword, setSelectedKeyword] = useState<Keyword | null>(null);
  const { backend } = useAppContext();

  useEffect(() => {
    setWords(data.words || []);
    console.log('current data in this node:', data);
  }, [data.words]);

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

  const stringToWords = (str: string): Word[] => {
    return str.split(' ').map((word) => ({ value: word } as Word));
  };

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

  const handleKeywordClick = (keyword: Keyword) => {
    setSelectedKeyword(keyword);
    setShowDescription(!showDescription);
  }

  //ADJUST SIZE OF TEXT AREA ON EDIT
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


  return (
    <>
      {showDescription &&  <KeywordDescription keyword={selectedKeyword}/>}
      
      {isEditing ? (
        <>
          <span style={{ fontSize: '0.75rem', color: 'gray' }}>ENTER TO CONFIRM</span>
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
          <button onClick={handleEditClick}>✎</button>
          <div className="nowheel p-3 border border-gray-700 rounded bg-white overflow-scroll" style={{ width: `${width}px`, height: `${height}px` }}>
            {words.map((word, index) => (
                <React.Fragment key={index}>
                  {'id' in word ? (
                    <KeywordComponent keyword={word} handleKeywordClick={() => handleKeywordClick(word)} />
                  ) : (
                    <>
                      <span> </span>
                      <WordComponent word={word} />
                      <span> </span>
                    </>
                  )}
                </React.Fragment>
              ))}
          </div>
        </>
      )}
    </>
  );
};

