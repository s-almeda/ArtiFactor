#!/bin/bash

# Define variables
PROJECT_ID="artifactor-449507"
IMAGE_NAME="resnet50wikiart"
FULL_IMAGE_PATH="gcr.io/$PROJECT_ID/$IMAGE_NAME:latest"
export DOCKER_BUILDKIT=1

echo "Building Docker image..."
docker build --progress=plain -t $FULL_IMAGE_PATH . 


# echo "Pushing image to Google Container Registry..."
# docker push $FULL_IMAGE_PATH

# echo "Docker image pushed: $FULL_IMAGE_PATH"
