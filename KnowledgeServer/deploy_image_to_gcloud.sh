#!/bin/bash
# see sync_to_gcloud.sh in LOCALDB for the script for updating the remote copy of the database.

set -euo pipefail

INSTANCE_NAME="resnet50wikiart"
ZONE="us-west1-b"
PROJECT_ID="artifactor-449507"
IMAGE_NAME="resnet50wikiart"
REMOTE_USER="admin"
REMOTE_TARGET="$REMOTE_USER@$INSTANCE_NAME"
REMOTE_LOCALDB_PATH="/home/admin/LOCALDB"
REMOTE_STAGE_PATH="/home/admin/deploy_staging_localdb"
LAST_TAG_FILE=".last_image_tag"

copy_to_remote_localdb() {
    local local_file="$1"
    local filename
    filename=$(basename "$local_file")

    gcloud compute ssh "$REMOTE_TARGET" --zone="$ZONE" --command "mkdir -p $REMOTE_STAGE_PATH"
    gcloud compute scp --zone="$ZONE" "$local_file" "$REMOTE_TARGET:$REMOTE_STAGE_PATH/"
    gcloud compute ssh "$REMOTE_TARGET" --zone="$ZONE" --command "sudo mkdir -p $REMOTE_LOCALDB_PATH; sudo mv \"$REMOTE_STAGE_PATH/$filename\" $REMOTE_LOCALDB_PATH/$filename; sudo chown admin:admin $REMOTE_LOCALDB_PATH/$filename"
}

# Always sync docker_run.sh so deploy uses latest restart logic
echo "Copying docker_run.sh to remote instance..."
gcloud compute scp --zone="$ZONE" docker_run.sh "$REMOTE_TARGET:/home/admin/docker_run.sh"

# Prompt user about scrape_to_staging.py changes
echo "Have you made changes to LOCALDB/scrape_to_staging.py?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying scrape_to_staging.py to remote instance..."
    copy_to_remote_localdb ./app/LOCALDB/scrape_to_staging.py
fi

# Prompt user about update_embeddings.py changes
echo "Have you made changes to LOCALDB/update_embeddings.py?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying update_embeddings.py to remote instance..."
    copy_to_remote_localdb ./app/LOCALDB/update_embeddings.py
fi

# Prompt user about artist_names.txt changes
echo "Have you made changes to LOCALDB/artist_names.txt?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying artist_names.txt to remote instance..."
    copy_to_remote_localdb ./app/LOCALDB/artist_names.txt
fi

# Prompt user about build_comicsbase.py changes
echo "Have you made changes to LOCALDB/build_comicsbase.py?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying build_comicsbase.py to remote instance..."
    copy_to_remote_localdb ./app/LOCALDB/build_comicsbase.py
fi

# Prompt user about comics_collections.txt changes
echo "Have you made changes to LOCALDB/comics_collections.txt?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying comics_collections.txt to remote instance..."
    copy_to_remote_localdb ./app/LOCALDB/comics_collections.txt
fi

# Prompt user about comic_images directory updates
echo "Have you added/updated files in LOCALDB/comic_images/?"
read -p "If yes, press 'y' to copy comic_images directory to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying comic_images to remote instance..."
    gcloud compute ssh "$REMOTE_TARGET" --zone="$ZONE" --command "mkdir -p $REMOTE_STAGE_PATH"
    gcloud compute scp --zone="$ZONE" --recurse ./app/LOCALDB/comic_images "$REMOTE_TARGET:$REMOTE_STAGE_PATH/"
    gcloud compute ssh "$REMOTE_TARGET" --zone="$ZONE" --command "sudo mkdir -p $REMOTE_LOCALDB_PATH/comic_images; sudo cp -r \"$REMOTE_STAGE_PATH/comic_images/.\" $REMOTE_LOCALDB_PATH/comic_images/; sudo chown -R admin:admin $REMOTE_LOCALDB_PATH/comic_images"
fi

# # Prompt user about rebuilding comics.db on the server
# echo "Do you want to rebuild LOCALDB/comics.db on the remote instance (instead of copying .db)?"
# read -p "If yes, press 'y' to open a remote shell and run build_comicsbase.py manually: " -n 1 -r
# echo
# if [[ $REPLY =~ ^[Yy]$ ]]; then
#     echo "Opening remote shell. Then run:"
#     echo "  cd ~/LOCALDB"
#     echo "  python3 build_comicsbase.py"
#     echo "Use ~/LOCALDB/comics_collections.txt when prompted."
#     gcloud compute ssh resnet50wikiart
# fi


# Prompt user about copying cached maps from the server

# Prompt user about copying cached maps from the server
echo "Do you want to copy cached map JSONs from the server? do this to save the version of the map before resetting the docker image"
read -p "If yes, press 'y' to proceed: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Connecting to remote instance to check for running container..."
    
    # Check if container is running - SUPPRESS STDERR to avoid zone messages
    CONTAINER_CHECK=$(gcloud compute ssh resnet50wikiart --zone=us-west1-b --command="docker ps --format '{{.Names}}' | grep resnet50wikiart" 2>/dev/null)
    
    # Debug: Show what we actually got
    echo "Container check result: '$CONTAINER_CHECK'"
    
    if [[ -z "$CONTAINER_CHECK" ]]; then
        echo "No running container named resnet50wikiart found."
        echo "Skipping map JSON copying..."
    else
        echo "Found running container: $CONTAINER_CHECK"
        
        # Create local generated_maps directory if it doesn't exist
        mkdir -p ./app/generated_maps
        
        # List JSON files inside /app/generated_maps in the container
        echo "Looking for JSON files in container..."
        JSON_FILES=$(gcloud compute ssh resnet50wikiart --zone=us-west1-b --command="docker exec resnet50wikiart find /app/generated_maps -name '*.json' -type f" 2>/dev/null)
        
        if [[ -z "$JSON_FILES" ]]; then
            echo "No JSON files found in /app/generated_maps/"
        else
            echo "Found JSON files:"
            echo "$JSON_FILES"
            COUNT=0
            
            # Process each JSON file
            for json_file in $JSON_FILES; do
                filename=$(basename "$json_file")
                echo "Copying $filename..."
                
                # Copy JSON file from container to remote instance home
                gcloud compute ssh resnet50wikiart --zone=us-west1-b --command="docker cp resnet50wikiart:$json_file ~/$filename"
                
                if [[ $? -eq 0 ]]; then
                    # Copy JSON file from remote instance to local app/generated_maps
                    gcloud compute scp resnet50wikiart:~/$filename ./app/generated_maps/$filename --zone=us-west1-b
                    
                    if [[ $? -eq 0 ]]; then
                        echo "✅ Successfully copied $filename"
                        ((COUNT++))
                        
                        # Clean up the temporary file on remote instance
                        gcloud compute ssh resnet50wikiart --zone=us-west1-b --command="rm ~/$filename"
                    else
                        echo "❌ Failed to copy $filename from remote to local"
                    fi
                else
                    echo "❌ Failed to copy $filename from container to remote"
                fi
            done
            
            echo "Successfully copied $COUNT map JSON file(s) to ./app/generated_maps/"
        fi
    fi
fi
# Prompt user about Docker image/app changes
echo "Have you made changes to the Docker image / Flask app (e.g., that runs via ./bootstrap.sh)?"
read -p "If yes, press 'y' to locally push/ remotely pull the Docker image (otherwise, script will exit): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "No changes to Docker image/app. Exiting."
    exit 0
fi

IMAGE_TAG="${IMAGE_TAG:-}"
if [[ -z "$IMAGE_TAG" ]]; then
    if [[ -f "$LAST_TAG_FILE" ]]; then
        CANDIDATE_TAG=$(tr -d '[:space:]' < "$LAST_TAG_FILE")
        if [[ -n "$CANDIDATE_TAG" ]] && docker image inspect "gcr.io/$PROJECT_ID/$IMAGE_NAME:$CANDIDATE_TAG" >/dev/null 2>&1; then
            IMAGE_TAG="$CANDIDATE_TAG"
            echo "Using last built tag from $LAST_TAG_FILE: $IMAGE_TAG"
        fi
    fi
fi

if [[ -z "$IMAGE_TAG" ]]; then
    IMAGE_TAG=$(docker images "gcr.io/$PROJECT_ID/$IMAGE_NAME" --format '{{.Tag}}' | grep -v '^latest$' | grep -v '^<none>$' | head -n 1 || true)
    if [[ -n "$IMAGE_TAG" ]]; then
        echo "Using newest local tag: $IMAGE_TAG"
    fi
fi

if [[ -z "$IMAGE_TAG" ]]; then
    echo "❌ ERROR: No deployable image tag found."
    echo "Build an image first, e.g. ./docker_build.sh"
    exit 1
fi

echo "Auto-selected image tag: $IMAGE_TAG"

FULL_IMAGE_PATH="gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG"

if [[ "$IMAGE_TAG" == "latest" ]]; then
    echo "❌ ERROR: refusing to deploy mutable tag 'latest'. Use an explicit tag."
    exit 1
fi

LOCAL_ARCH=$(docker image inspect "$FULL_IMAGE_PATH" --format '{{.Architecture}}' 2>/dev/null || true)
if [[ -n "$LOCAL_ARCH" && "$LOCAL_ARCH" != "amd64" ]]; then
    echo "❌ ERROR: Local image arch for $FULL_IMAGE_PATH is '$LOCAL_ARCH' (expected amd64)."
    exit 1
fi

# Push Docker image to Google Container Registry
docker push "$FULL_IMAGE_PATH"


# SSH into the compute instance and run commands
echo "Connecting to remote instance..."
gcloud compute ssh "$REMOTE_TARGET" --zone="$ZONE" --command="
    set -e
    cd /home/admin
    chmod +x /home/admin/docker_run.sh
    BEFORE_ID=\$(docker ps -q -f name=^resnet50wikiart$ || true)
    echo 'Container before deploy:' \"\${BEFORE_ID:-<none>}\"
    echo 'Configuring Docker auth for gcr.io...'
    gcloud auth configure-docker gcr.io --quiet || true
    echo 'Pulling Docker image: $FULL_IMAGE_PATH'
    if ! docker pull $FULL_IMAGE_PATH; then
        echo 'ERROR: Failed to pull image from gcr.io.'
        echo 'Ensure this VM/user has Artifact Registry read access (roles/artifactregistry.reader).'
        exit 1
    fi
    ARCH=\$(docker image inspect $FULL_IMAGE_PATH --format '{{.Architecture}}')
    if [ \"\$ARCH\" != \"amd64\" ]; then
        echo 'ERROR: Pulled image architecture is' \"\$ARCH\" '(expected amd64)'
        exit 1
    fi
    echo 'Running docker_run.sh...'
    IMAGE_TAG=$IMAGE_TAG /home/admin/docker_run.sh
    AFTER_ID=\$(docker ps -q -f name=^resnet50wikiart$ || true)
    echo 'Container after deploy:' \"\${AFTER_ID:-<none>}\"
    if [ -z \"\$AFTER_ID\" ]; then
        echo 'ERROR: resnet50wikiart container is not running after deploy.'
        exit 1
    fi
    if [ -n \"\$BEFORE_ID\" ] && [ \"\$BEFORE_ID\" = \"\$AFTER_ID\" ]; then
        echo 'WARNING: container ID unchanged; restart may not have been applied.'
    fi
    echo 'Verifying container mount + image...'
    docker inspect resnet50wikiart --format '{{.Config.Image}}'
    docker inspect resnet50wikiart --format '{{json .Mounts}}'
"