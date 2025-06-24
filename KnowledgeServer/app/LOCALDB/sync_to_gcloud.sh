#!/bin/bash

# Configuration
INSTANCE_NAME="resnet50wikiart"
ZONE="us-west1-b"
LOCAL_IMAGES_DIR="images"
REMOTE_DB_PATH="~/LOCALDB/knowledgebase.db"
REMOTE_IMAGES_DIR="~/LOCALDB/images"

# Script paths
AUDIT_SCRIPT="./audit_images.sh"
DOWNLOAD_SCRIPT="./download_missing_images.sh"

echo "========================================"
echo "Complete Database Audit and GCloud Sync"
echo "========================================"
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo ""

# Function to ask for user confirmation
ask_confirmation() {
    local message="$1"
    echo "$message"
    read -p "Do you want to proceed? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping this step."
        return 1
    fi
    return 0
}

# Function to check if script exists and is executable
check_script() {
    local script="$1"
    if [ ! -f "$script" ]; then
        echo "Error: $script not found!"
        return 1
    fi
    if [ ! -x "$script" ]; then
        echo "Making $script executable..."
        chmod +x "$script"
    fi
    return 0
}

# Function to run a script with proper error handling
run_script() {
    local script="$1"
    local description="$2"
    
    echo "Running: $description"
    echo "Script: $script"
    echo "Working directory: $(pwd)"
    echo ""
    
    # Run the script and capture its exit code
    bash "$script"
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ $description failed with exit code: $exit_code"
        return $exit_code
    else
        echo "✅ $description completed successfully"
        return 0
    fi
}

# ============================================================================
# STEP 1: AUDIT DATABASE AND LOCAL IMAGES
# ============================================================================

echo "STEP 1: Database and Image Audit"
echo "================================="

if ask_confirmation "This will audit your local images against the database and optionally delete unnecessary files."; then
    if check_script "$AUDIT_SCRIPT"; then
        run_script "$AUDIT_SCRIPT" "Image audit"
        if [ $? -ne 0 ]; then
            echo "Audit failed. Exiting."
            exit 1
        fi
    else
        exit 1
    fi
else
    echo "Audit step skipped."
fi

echo ""
sleep 2  # Give a moment to let any file operations complete

# ============================================================================
# STEP 2: DOWNLOAD MISSING IMAGES
# ============================================================================

echo "STEP 2: Download Missing Images"
echo "==============================="

if [ -f "missing_image_ids.txt" ]; then
    # Count missing images (exclude header line if it exists)
    if [ -s "missing_image_ids.txt" ]; then
        # Check if file has header
        FIRST_LINE=$(head -n 1 missing_image_ids.txt)
        if [[ "$FIRST_LINE" == *"image_id"* ]]; then
            MISSING_COUNT=$(($(wc -l < missing_image_ids.txt) - 1))
        else
            MISSING_COUNT=$(wc -l < missing_image_ids.txt)
        fi
        
        if [ $MISSING_COUNT -gt 0 ]; then
            echo "Found $MISSING_COUNT missing images."
            
            if ask_confirmation "Download them now?"; then
                if check_script "$DOWNLOAD_SCRIPT"; then
                    run_script "$DOWNLOAD_SCRIPT" "Missing image download"
                    if [ $? -ne 0 ]; then
                        echo "Download had some failures. Check download_log.txt for details."
                        if ! ask_confirmation "Continue with sync anyway?"; then
                            exit 1
                        fi
                    fi
                else
                    exit 1
                fi
            else
                echo "Download step skipped."
            fi
        else
            echo "No missing images found. Skipping download step."
        fi
    else
        echo "missing_image_ids.txt is empty. Skipping download step."
    fi
else
    echo "No missing_image_ids.txt file found. Skipping download step."
fi

echo ""
sleep 2  # Give a moment to let any file operations complete

# ============================================================================
# STEP 3: SYNC TO GOOGLE CLOUD
# ============================================================================

echo "STEP 3: Sync to Google Cloud"
echo "============================="

# First, verify local files exist
if [ ! -f "knowledgebase.db" ]; then
    echo "Error: knowledgebase.db not found!"
    exit 1
fi

if [ ! -d "$LOCAL_IMAGES_DIR" ]; then
    echo "Error: Local images directory '$LOCAL_IMAGES_DIR' not found!"
    exit 1
fi

# Count local images
LOCAL_IMAGE_COUNT=$(find "$LOCAL_IMAGES_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.gif" -o -name "*.webp" \) | wc -l)
echo "Found $LOCAL_IMAGE_COUNT local images"

if ask_confirmation "This will sync your database and images to Google Cloud instance '$INSTANCE_NAME'."; then
    
    echo "Starting sync to Google Cloud instance: $INSTANCE_NAME"
    
    # Test connection first
    echo "Testing connection to instance..."
    if ! gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "echo 'Connection successful'" 2>/dev/null; then
        echo "❌ Failed to connect to instance. Please check:"
        echo "  - Instance name: $INSTANCE_NAME"
        echo "  - Zone: $ZONE"
        echo "  - Your gcloud authentication"
        exit 1
    fi
    
    # Step 3a: Create remote directories if they don't exist
    echo "3a. Setting up remote directories..."
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "mkdir -p $(dirname $REMOTE_DB_PATH) $REMOTE_IMAGES_DIR"
    
    # Step 3b: Copy the knowledgebase.db file
    echo "3b. Copying knowledgebase.db..."
    if gcloud compute scp knowledgebase.db $INSTANCE_NAME:$REMOTE_DB_PATH --zone=$ZONE; then
        echo "✅ Database file copied successfully"
    else
        echo "❌ Failed to copy database file"
        exit 1
    fi
    
    # Step 3c: Get list of existing images on remote server
    echo "3c. Checking existing images on remote server..."
    REMOTE_FILES_LIST=$(mktemp)
    
    if gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "ls -1 $REMOTE_IMAGES_DIR 2>/dev/null" > "$REMOTE_FILES_LIST"; then
        REMOTE_FILE_COUNT=$(wc -l < "$REMOTE_FILES_LIST")
        echo "Found $REMOTE_FILE_COUNT existing files on remote server"
    else
        echo "Remote images directory is empty or doesn't exist"
        > "$REMOTE_FILES_LIST"  # Create empty file
        REMOTE_FILE_COUNT=0
    fi
    
    # Step 3d: Copy only new images
    echo "3d. Copying new images..."
    NEW_FILES_COUNT=0
    SKIPPED_FILES_COUNT=0
    FAILED_FILES_COUNT=0
    
    # Process each local image file
    for file in "$LOCAL_IMAGES_DIR"/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            
            # Skip non-image files
            if [[ ! "$filename" =~ \.(jpg|jpeg|png|gif|webp|bmp)$ ]]; then
                continue
            fi
            
            # Check if file already exists on remote server
            if grep -q "^$filename$" "$REMOTE_FILES_LIST"; then
                ((SKIPPED_FILES_COUNT++))
                # Show progress every 100 skipped files
                if [ $((SKIPPED_FILES_COUNT % 100)) -eq 0 ]; then
                    echo "  Skipped $SKIPPED_FILES_COUNT files so far..."
                fi
            else
                echo "  Copying $filename..."
                if gcloud compute scp "$file" "$INSTANCE_NAME:$REMOTE_IMAGES_DIR/" --zone="$ZONE" 2>/dev/null; then
                    ((NEW_FILES_COUNT++))
                else
                    echo "  ❌ Failed to copy $filename"
                    ((FAILED_FILES_COUNT++))
                fi
            fi
        fi
    done
    
    # Clean up temp file
    rm -f "$REMOTE_FILES_LIST"
    
    echo ""
    echo "========================================"
    echo "Sync Summary:"
    echo "========================================"
    echo "✅ Database file updated"
    echo "✅ New files copied: $NEW_FILES_COUNT"
    echo "⏭️  Files skipped (already exist): $SKIPPED_FILES_COUNT"
    if [ $FAILED_FILES_COUNT -gt 0 ]; then
        echo "❌ Failed copies: $FAILED_FILES_COUNT"
    fi
    echo "📊 Total files on remote: $((REMOTE_FILE_COUNT + NEW_FILES_COUNT))"
    
else
    echo "Google Cloud sync skipped."
fi

echo ""
echo "========================================"
echo "All operations completed!"
echo "========================================"

# Show final status
echo ""
echo "Final Status:"
echo "-------------"

# Check for any remaining missing images
if [ -f "missing_image_ids.txt" ] && [ -s "missing_image_ids.txt" ]; then
    REMAINING_MISSING=$(grep -v "^image_id" missing_image_ids.txt | wc -l)
    if [ $REMAINING_MISSING -gt 0 ]; then
        echo "⚠️  Warning: $REMAINING_MISSING images still missing (check download_log.txt)"
    fi
fi

# Show download log summary if it exists
if [ -f "download_log.txt" ]; then
    FAILED_DOWNLOADS=$(grep -c "^FAILED:" download_log.txt || echo "0")
    if [ $FAILED_DOWNLOADS -gt 0 ]; then
        echo "⚠️  Warning: $FAILED_DOWNLOADS download failures recorded"
        echo "   Run 'grep ^FAILED: download_log.txt' to see details"
    fi
fi

echo ""
echo "Done!"