export const calcNearbyPosition = (
    inputBounds: { x: number; y: number; width: number; height: number },
    origin: { x: number; y: number } = { x: 0, y: 0 },
    
) => {
    const padding = 10;
    const maxReturnDistance = 100 // Define a threshold for what is considered "very far away"

    // Calculate the center of the inputBounds
    const boundsCenterX = inputBounds.x + inputBounds.width / 2;
    const boundsCenterY = inputBounds.y + inputBounds.height / 2;

    // Calculate the distance from origin to the bounds center
    const distanceX = Math.abs(origin.x - boundsCenterX);
    const distanceY = Math.abs(origin.y - boundsCenterY);

    // If the original position is not too far, move the node back to its original position
    if (distanceX <= maxReturnDistance && distanceY <= maxReturnDistance) {
        return origin;
    }

    // Otherwise, move it outside the bounds in the direction of its original position
    const directionX = origin.x >= boundsCenterX ? 1 : -1;
    const directionY = origin.y >= boundsCenterY ? 1 : -1;

    const newX = directionX === 1 
        ? inputBounds.x + inputBounds.width + padding 
        : inputBounds.x - padding;

    const newY = directionY === 1 
        ? inputBounds.y + inputBounds.height + padding 
        : inputBounds.y - padding;

    return { x: newX, y: newY };
};
