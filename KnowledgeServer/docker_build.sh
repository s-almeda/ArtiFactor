#!/bin/bash

set -euo pipefail

# Define variables
PROJECT_ID="artifactor-449507"
IMAGE_NAME="resnet50wikiart"
IMAGE_TAG="${1:-amd64-$(date +%Y%m%d-%H%M%S)}"
FULL_IMAGE_PATH="gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG"
LAST_TAG_FILE=".last_image_tag"
export DOCKER_BUILDKIT=1

echo "Using image: $FULL_IMAGE_PATH"
echo "Building linux/amd64 image with buildx..."
docker buildx create --name xbuilder --use 2>/dev/null || docker buildx use xbuilder
docker buildx inspect --bootstrap >/dev/null
docker buildx build --platform linux/amd64 --pull --progress=plain -t "$FULL_IMAGE_PATH" --load .

ARCH=$(docker image inspect "$FULL_IMAGE_PATH" --format '{{.Architecture}}')
if [[ "$ARCH" != "amd64" ]]; then
	echo "❌ ERROR: Built image architecture is '$ARCH' (expected amd64)."
	exit 1
fi
echo "✅ Built image architecture: $ARCH"
echo "$IMAGE_TAG" > "$LAST_TAG_FILE"
echo "Saved last built tag to $LAST_TAG_FILE"

echo "Push image to Google Container Registry now?"
read -p "Press 'y' to push $FULL_IMAGE_PATH: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
	docker push "$FULL_IMAGE_PATH"
	echo "✅ Docker image pushed: $FULL_IMAGE_PATH"
else
	echo "Skipped push."
fi


# echo "Pushing image to Google Container Registry..."
# docker push $FULL_IMAGE_PATH

# echo "Docker image pushed: $FULL_IMAGE_PATH"
