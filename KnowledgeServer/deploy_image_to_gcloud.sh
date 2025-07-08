#!/bin/bash
# see sync_to_gcloud.sh in LOCALDB for the script for updating the remote copy of the database.

# Push Docker image to Google Container Registry
docker push gcr.io/artifactor-449507/resnet50wikiart:latest

# Prompt user about docker_run.sh changes
echo "Have you made changes to docker_run.sh?"
read -p "If yes, press 'y' to copy it to the remote instance: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying docker_run.sh to remote instance..."
    gcloud compute scp docker_run.sh resnet50wikiart:~
fi

# SSH into the compute instance and run commands
echo "Connecting to remote instance..."
gcloud compute ssh resnet50wikiart --command="
    echo 'Pulling latest Docker image...'
    docker pull gcr.io/artifactor-449507/resnet50wikiart:latest
    echo 'Running docker_run.sh...'
    ./docker_run.sh
"