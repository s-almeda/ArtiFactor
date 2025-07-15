#!/bin/bash
# see sync_to_gcloud.sh in LOCALDB for the script for updating the remote copy of the database.

# Prompt user about docker_run.sh changes
echo "Have you made changes to docker_run.sh?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying docker_run.sh to remote instance..."
    gcloud compute scp docker_run.sh resnet50wikiart:~
fi

# Prompt user about scrape_to_staging.py changes
echo "Have you made changes to LOCALDB/scrape_to_staging.py?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying scrape_to_staging.py to remote instance..."
    gcloud compute scp ./app/LOCALDB/scrape_to_staging.py resnet50wikiart:~/LOCALDB/
fi

# Prompt user about update_embeddings.py changes
echo "Have you made changes to LOCALDB/update_embeddings.py?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying update_embeddings.py to remote instance..."
    gcloud compute scp ./app/LOCALDB/update_embeddings.py resnet50wikiart:~/LOCALDB/
fi

# Prompt user about artist_names.txt changes
echo "Have you made changes to LOCALDB/artist_names.txt?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying artist_names.txt to remote instance..."
    gcloud compute scp ./app/LOCALDB/artist_names.txt resnet50wikiart:~/LOCALDB/
fi


# Prompt user about Docker image/app changes
echo "Have you made changes to the Docker image / Flask app (e.g., that runs via ./bootstrap.sh)?"
read -p "If yes, press 'y' to locally push/ remotely pull the Docker image (otherwise, script will exit): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "No changes to Docker image/app. Exiting."
    exit 0
fi
# Push Docker image to Google Container Registry
docker push gcr.io/artifactor-449507/resnet50wikiart:latest


# SSH into the compute instance and run commands
echo "Connecting to remote instance..."
gcloud compute ssh resnet50wikiart --command="
    echo 'Pulling latest Docker image...'
    docker pull gcr.io/artifactor-449507/resnet50wikiart:latest
    echo 'Running docker_run.sh...'
    ./docker_run.sh
"