#!/bin/bash

set -euo pipefail

# Set the image name
PROJECT_ID="artifactor-449507"
IMAGE_NAME="resnet50wikiart"
IMAGE_TAG="${IMAGE_TAG:-}"
if [ -z "$IMAGE_TAG" ]; then
    echo "❌ ERROR: IMAGE_TAG is required."
    echo "Usage: IMAGE_TAG=amd64-YYYYMMDD-HHMMSS ./docker_run.sh"
    exit 1
fi
FULL_IMAGE_PATH="gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG"


# Detect if running on Google Cloud VM using the metadata service
if curl -fsSL -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/ >/dev/null; then
    echo "✅ Running on Google Cloud VM..."
    # Canonical server dataset path
    LOCALDB_PATH="/home/admin/LOCALDB"

    if [ ! -d "$LOCALDB_PATH" ]; then
        echo "❌ ERROR: Expected server LOCALDB path is missing: $LOCALDB_PATH"
        exit 1
    fi
else
    echo "Running on local machine..."
    
    # Local: Use the correct LOCALDB path relative to the script’s directory
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    LOCALDB_PATH="$SCRIPT_DIR/app/LOCALDB"
fi

echo "Using LOCALDB_PATH: $LOCALDB_PATH"
echo "Using image: $FULL_IMAGE_PATH"

if ! docker image inspect "$FULL_IMAGE_PATH" >/dev/null 2>&1; then
    echo "❌ ERROR: Image not found locally: $FULL_IMAGE_PATH"
    echo "Pull or build this image first."
    exit 1
fi

IMAGE_ARCH=$(docker image inspect "$FULL_IMAGE_PATH" --format '{{.Architecture}}')
if [ "$IMAGE_ARCH" != "amd64" ]; then
    echo "❌ ERROR: Refusing to run non-amd64 image: $FULL_IMAGE_PATH (arch=$IMAGE_ARCH)"
    echo "Build/pull an amd64 image tag and set IMAGE_TAG accordingly."
    exit 1
fi
echo "✅ Image architecture check passed: $IMAGE_ARCH"

# Fail fast if the selected LOCALDB path is not the expected dataset.
if [ ! -f "$LOCALDB_PATH/knowledgebase.db" ]; then
    echo "❌ ERROR: knowledgebase.db not found at $LOCALDB_PATH/knowledgebase.db"
    echo "Refusing to start container with an empty/wrong LOCALDB mount."
    exit 1
fi

if [ ! -d "$LOCALDB_PATH/images" ]; then
    echo "❌ ERROR: images directory not found at $LOCALDB_PATH/images"
    echo "Refusing to start container with an empty/wrong LOCALDB mount."
    exit 1
fi

# Stop and remove any existing container
echo "Stopping and removing existing container (if any)..."
docker stop $IMAGE_NAME 2>/dev/null
docker rm $IMAGE_NAME 2>/dev/null

# Run the container with appropriate volume mounts 
echo "Running Docker container..."
docker run -d -p 8080:8080 \
    -v "$LOCALDB_PATH:/app/LOCALDB" \
    -v "$HOME/model_cache:/root/.cache/torch/hub" \
    -v "$HOME/transformers_cache:/root/.cache/transformers" \
    -e RUNNING_IN_DOCKER=true \
    -e FINAL_SQL_ADMIN_PASSWORD="${FINAL_SQL_ADMIN_PASSWORD:-Girimehkala}" \
    --name $IMAGE_NAME $FULL_IMAGE_PATH

echo "Container is running! Use 'docker logs $IMAGE_NAME' to check logs."
docker ps
