import type { Word, Keyword } from "../nodes/types";

export const stringToWords = (str: string): Word[] => {
  return str.split(' ').map((word) => ({ value: word } as Word));
};
export const wordsToString = (words: Word[]): string => {
  return words.map((word) => word.value).join(' ');
}

export const keywordJSONtoKeyword = (json: any): Keyword => {
  const details = json.details || json;

  return {
    value: json.value,
    entryId: details.entry_id,
    databaseValue: details.databaseValue,
    images: details.images || [],
    isArtist: details.isArtist === 1,
    type: details.type,
    aliases: details.isArtist === 1 ? (details.artist_aliases || []).map((alias: any) => alias.value) : undefined,
    descriptions: details.descriptions
      ? Object.entries(details.descriptions).map(([source, desc]: [string, any]) => ({
          source,
          description: desc.description,
          ...desc,
        }))
      : [],
    relatedKeywordIds: details.relatedKeywordIds || [],
    relatedKeywordStrings: details.relatedKeywordStrings || [],
  };
};

export const calcNearbyPosition = (
    inputBounds: { x: number; y: number; width: number; height: number }
) => {
    //console.log("getting nearby position for... inputBounds", inputBounds);
    const padding = 10;
    const randomness = () => Math.random() * 40 - 10; // Random value between -10 and 10

    // Calculate the center of the inputBounds
    const boundsCenterX = inputBounds.x + inputBounds.width / 2;
    const boundsCenterY = inputBounds.y + inputBounds.height / 2;

    // Move it outside the bounds in the direction of the center with added randomness
    const newX = boundsCenterX - padding + randomness();
    // >= inputBounds.x + inputBounds.width / 2 
    //     ? inputBounds.x + inputBounds.width + padding + randomness()
    //     : inputBounds.x - padding + randomness();

    const newY = boundsCenterY >= inputBounds.y + inputBounds.height / 2 
        ? inputBounds.y + inputBounds.height  + padding + randomness()
        : inputBounds.y + padding + randomness();

    return { x: newX, y: newY };
};
