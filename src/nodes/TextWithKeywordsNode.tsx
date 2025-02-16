import { useAppContext } from '../context/AppContext';
import type { NodeProps } from "@xyflow/react";
import { type Word, type Keyword, type TextWithKeywordsNode } from './types';
import React, { useRef, useState, useEffect } from 'react';

export const WordComponent: React.FC<{ word: Word }> = ({ word }) => {
  return <span>{word.value}</span>;
};
export const KeywordComponent: React.FC<{ keyword: Keyword }> = ({ keyword }) => {
  const [showDescription, setShowDescription] = useState(false);
  useEffect(() => {
    if (!keyword.relatedKeywordStrings || keyword.relatedKeywordStrings.length === 0) {
      keyword.relatedKeywordStrings = ["insert", "related", "words", "here"];
    }
  }, [keyword.relatedKeywordStrings]);
  const handleKeywordClick = () => {
    setShowDescription(!showDescription);
  };

  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <span
      onClick={handleKeywordClick}
      style={{ backgroundColor: 'yellow', cursor: 'pointer', padding: '2px 4px', borderRadius: '4px', margin: '0 2px' }}
      >
      {keyword.value}
      </span>
      {showDescription && (
      <div
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
            <div style={{ maxHeight: '200px', overflowY: 'scroll' }}>
              <strong>Description:</strong> {keyword.description}
            </div>
          )}
          {keyword.relatedKeywordStrings && keyword.relatedKeywordStrings.length > 0 && (
        <div>
          <strong>Related Keywords:</strong>
          <ul>
          {keyword.relatedKeywordStrings.map((relatedKeyword, index) => (
            <li key={index}>{relatedKeyword}</li>
          ))}
          </ul>
        </div>
        )}
      </div>
      )}
    </span>
  );
};



export function TextWithKeywordsNode({ data, selected}: NodeProps<TextWithKeywordsNode>) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [content, setContent] = useState(data.content || '');
  const [isEditing, setIsEditing] = useState(false);
  const [words, setWords] = useState<(Word | Keyword)[]>(data.words || []);
  const [width, setWidth] = useState(100);
  const [height, setHeight] = useState(100);
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
      setIsEditing(false);
      const updatedWords = await stringToWords(content);
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
      const { words: newWords, keywords } = data;

      const newWordsArray = newWords.map((word: any) => {
        if (word.id) {
          const keywordInfo = keywords.find((keyword: any) => keyword.id === word.id);
          if (keywordInfo) {
            console.log("found keyword:", keywordInfo);
            return {
              id: keywordInfo.id,
              value: word.value,
              databaseValue: keywordInfo.databaseValue,
              description: keywordInfo.description,
              relatedKeywordIds: keywordInfo.relatedKeywordIds,
              relatedKeywordStrings: keywordInfo.relatedKeywordStrings,
              type: keywordInfo.type,
            } as Keyword;
          }
        }
        return { value: word.value } as Word;
      });

      console.log('New words and keywords array:', newWordsArray);
      return newWordsArray;
    } catch (error) {
      console.error('Error checking for keywords:', error);
      console.log('Fallback words array:', words);
      return words;
    }
  };
  const adjustNodeSize = () => {
    const totalTextLength = words.reduce((acc, word) => acc + word.value.length, 0);
    const maxLineWidth = 200;
    const newWidth = Math.min(Math.max(totalTextLength * 8, 100), maxLineWidth);
    const lines = Math.ceil(totalTextLength * 8 / maxLineWidth);
    setWidth(newWidth);
  };

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
  }, [content, data]);

  useEffect(() => {
    if (!selected && isEditing) {
      setIsEditing(false);
    }
    adjustNodeSize();
  }, [words]);


  return (
    <>
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
          <div className="nowheel p-3 border border-gray-700 rounded bg-white overflow-visible" style={{ width: `${width}px`, height: `${height}px` }}>
            {words.map((word, index) => (
                <React.Fragment key={index}>
                  {'id' in word ? (
                    <KeywordComponent keyword={word} />
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

