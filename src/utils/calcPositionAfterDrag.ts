import { Node } from "@xyflow/react";
import { useReactFlow } from "@xyflow/react";

export const calcPositionAfterDrag = (
    previousPosition: { x: number; y: number },
    intersectionNode: Node,
    direction: "above" | "below" = "below"
) => {
    // const { getNodesBounds } = useReactFlow();

    // const bounds = getNodesBounds([intersectionNode]);
    // let newPosition = { ...previousPosition };

    // const xOffset = typeof intersectionNode.data.xOffset === 'number' ? intersectionNode.data.xOffset : 10;
    // const yOffset = typeof intersectionNode.data.yOffset === 'number' ? intersectionNode.data.yOffset : 10;

    // newPosition.x = bounds.x + xOffset; // place to the right
    // newPosition.y = bounds.y + bounds.height + yOffset; // place below

    // if (direction === "below") {
    //     newPosition.x = bounds.x + xOffset; // place to the right
    //     newPosition.y = bounds.y + bounds.height + yOffset; // place below
    // } else if (direction === "above") {
    //     newPosition.x = bounds.x - bounds.width / 2 - xOffset; // place to the left
    //     newPosition.y = bounds.y - bounds.height - yOffset; // place above
    // }
    return { x: 100, y: 100 };
};