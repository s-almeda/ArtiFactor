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

# FINAL WORKING VERSION:

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