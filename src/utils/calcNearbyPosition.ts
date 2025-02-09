
/**
 * Calculates a nearby position OUSTIDE the input bounding box.
 * decides which direction to move the new position based on the "origin" provided
 * @param inputBounds - The bounding box with properties x, y, width, and height.
 * @param origin - The origin point with properties x and y. Defaults to { x: 0, y: 0 }.
 * @returns An object containing the new x and y coordinates outside the input bounds.
 */

export const calcNearbyPosition = (
    inputBounds: { x: number; y: number; width: number; height: number },
    origin: { x: number; y: number } = { x: 0, y: 0 }
) => {
    const padding = 5;

    // Determine the direction to move based on the origin
    const directionX = origin.x >= inputBounds.x ? 1 : -1;
    const directionY = origin.y >= inputBounds.y ? 1 : -1;

    // Calculate the closest position that is outside the inputBounds
    const newX = directionX === 1 
        ? inputBounds.x + inputBounds.width + padding 
        : inputBounds.x - padding;

    const newY = directionY === 1 
        ? inputBounds.y + inputBounds.height + padding 
        : inputBounds.y - padding;

    return { x: newX, y: newY };
};
