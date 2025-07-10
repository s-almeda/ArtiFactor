#!/bin/bash

# Set the image name
PROJECT_ID="artifactor-449507"
IMAGE_NAME="resnet50wikiart"
FULL_IMAGE_PATH="gcr.io/$PROJECT_ID/$IMAGE_NAME:latest"


# Detect if running on Google Cloud VM using the metadata service
if curl -fsSL -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/ >/dev/null; then
    echo "✅ Running on Google Cloud VM..."
    LOCALDB_PATH="$HOME/LOCALDB"  # Use ~/LOCALDB on Google Cloud VM    echo "Running on Google Cloud VM..."
    
    # Google Cloud: Mount ~/LOCALDB into the container, putting it at /app/LOCALDB
    LOCALDB_PATH="$HOME/LOCALDB"
else
    echo "Running on local machine..."
    
    # Local: Use the correct LOCALDB path relative to the script’s directory
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    LOCALDB_PATH="$SCRIPT_DIR/app/LOCALDB"
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
