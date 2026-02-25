#!/bin/bash
# see sync_to_gcloud.sh in LOCALDB for the script for updating the remote copy of the database.

set -euo pipefail

INSTANCE_NAME="resnet50wikiart"
ZONE="us-west1-b"
PROJECT_ID="artifactor-449507"
IMAGE_NAME="resnet50wikiart"
REMOTE_LOCALDB_PATH="/home/admin/LOCALDB"
REMOTE_STAGE_PATH="~/deploy_staging_localdb"

copy_to_remote_localdb() {
    local local_file="$1"
    local filename
    filename=$(basename "$local_file")

    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "mkdir -p $REMOTE_STAGE_PATH"
    gcloud compute scp --zone="$ZONE" "$local_file" "$INSTANCE_NAME:$REMOTE_STAGE_PATH/"
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "STAGE_DIR=\$HOME/deploy_staging_localdb; sudo mkdir -p $REMOTE_LOCALDB_PATH; sudo mv \"\$STAGE_DIR/$filename\" $REMOTE_LOCALDB_PATH/$filename; sudo chown admin:admin $REMOTE_LOCALDB_PATH/$filename"
}

# Prompt user about docker_run.sh changes
echo "Have you made changes to docker_run.sh?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying docker_run.sh to remote instance..."
    gcloud compute scp --zone="$ZONE" docker_run.sh "$INSTANCE_NAME:~"
fi

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
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "mkdir -p $REMOTE_STAGE_PATH"
    gcloud compute scp --zone="$ZONE" --recurse ./app/LOCALDB/comic_images "$INSTANCE_NAME:$REMOTE_STAGE_PATH/"
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "STAGE_DIR=\$HOME/deploy_staging_localdb; sudo mkdir -p $REMOTE_LOCALDB_PATH/comic_images; sudo cp -r \"\$STAGE_DIR/comic_images/.\" $REMOTE_LOCALDB_PATH/comic_images/; sudo chown -R admin:admin $REMOTE_LOCALDB_PATH/comic_images"
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


# Prompt user about Docker image/app changes
echo "Have you made changes to the Docker image / Flask app (e.g., that runs via ./bootstrap.sh)?"
read -p "If yes, press 'y' to locally push/ remotely pull the Docker image (otherwise, script will exit): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "No changes to Docker image/app. Exiting."
    exit 0
fi

read -p "Enter image tag to deploy (example: amd64-20260225-130500): " IMAGE_TAG
if [[ -z "${IMAGE_TAG:-}" ]]; then
    echo "❌ ERROR: image tag is required."
    exit 1
fi

FULL_IMAGE_PATH="gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG"

if [[ "$IMAGE_TAG" == "latest" ]]; then
    echo "❌ ERROR: refusing to deploy mutable tag 'latest'. Use an explicit tag."
    exit 1
fi

# Push Docker image to Google Container Registry
docker push "$FULL_IMAGE_PATH"


# SSH into the compute instance and run commands
echo "Connecting to remote instance..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    set -e
    echo 'Pulling Docker image: $FULL_IMAGE_PATH'
    docker pull $FULL_IMAGE_PATH
    ARCH=\$(docker image inspect $FULL_IMAGE_PATH --format '{{.Architecture}}')
    if [ \"\$ARCH\" != \"amd64\" ]; then
        echo 'ERROR: Pulled image architecture is' \"\$ARCH\" '(expected amd64)'
        exit 1
    fi
    echo 'Running docker_run.sh...'
    IMAGE_TAG=$IMAGE_TAG ./docker_run.sh
    echo 'Verifying container mount + image...'
    docker inspect resnet50wikiart --format '{{.Config.Image}}'
    docker inspect resnet50wikiart --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
"