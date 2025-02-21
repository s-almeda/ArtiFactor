import type { Word } from "../nodes/types";

export const stringToWords = (str: string): Word[] => {
  return str.split(' ').map((word) => ({ value: word } as Word));
};
export const wordsToString = (words: Word[]): string => {
  return words.map((word) => word.value).join(' ');
}

export const calcNearbyPosition = (
    inputBounds: { x: number; y: number; width: number; height: number }
) => {
    const padding = 10;
    const randomness = () => Math.random() * 30 - 10; // Random value between -10 and 10

    // Calculate the center of the inputBounds
    const boundsCenterX = inputBounds.x + inputBounds.width / 2;
    const boundsCenterY = inputBounds.y + inputBounds.height / 2;

    // Move it outside the bounds in the direction of the center with added randomness
    const newX = boundsCenterX >= inputBounds.x + inputBounds.width / 2 
        ? inputBounds.x + inputBounds.width + padding + randomness()
        : inputBounds.x - padding + randomness();

    const newY = boundsCenterY >= inputBounds.y + inputBounds.height / 2 
        ? inputBounds.y + inputBounds.height  - randomness()
        : inputBounds.y - padding - randomness();

    return { x: newX, y: newY };
};
