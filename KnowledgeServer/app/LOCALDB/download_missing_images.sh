#!/bin/bash

# Configuration
DB_FILE="knowledgebase.db"
IMAGES_DIR="images"
MISSING_IDS_FILE="missing_image_ids.txt"
LOG_FILE="download_log.txt"

echo "Starting missing image download process..."
echo "Database: $DB_FILE"
echo "Images directory: $IMAGES_DIR"
echo "Missing IDs file: $MISSING_IDS_FILE"
echo ""

# Check if required files exist
if [ ! -f "$DB_FILE" ]; then
    echo "Error: Database file '$DB_FILE' not found!"
    exit 1
fi

if [ ! -f "$MISSING_IDS_FILE" ]; then
    echo "Error: Missing IDs file '$MISSING_IDS_FILE' not found!"
    echo "Run the audit script first to generate this file."
    exit 1
fi

# Create images directory if it doesn't exist
mkdir -p "$IMAGES_DIR"

# Initialize log file
echo "Download log started at $(date)" > "$LOG_FILE"

# Counters
TOTAL_COUNT=0
SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0

echo "Processing missing images..."
echo ""

# Read each line from missing_image_ids.txt
while IFS='|' read -r image_id current_filename; do
    if [ -z "$image_id" ] || [ "$image_id" = "image_id" ]; then
        # Skip header line or empty lines
        continue
    fi
    
    ((TOTAL_COUNT++))
    echo "[$TOTAL_COUNT] Processing image_id: $image_id"
    
    # Get image_urls from database
    IMAGE_URLS=$(sqlite3 "$DB_FILE" "SELECT image_urls FROM image_entries WHERE image_id = '$image_id';")
    
    if [ -z "$IMAGE_URLS" ]; then
        echo "  ✗ No image_urls found for image_id: $image_id"
        echo "FAILED: $image_id - No image_urls found" >> "$LOG_FILE"
        ((FAILED_COUNT++))
        continue
    fi
    
    echo "  Found image_urls: ${IMAGE_URLS:0:100}..." # Show first 100 chars for debugging
    
    # Parse JSON to extract URLs (assuming common formats)
    DOWNLOAD_URL=""
    FOUND_KEY=""
    
    # Try different JSON keys in order of preference
    for key in "small" "medium" "medium_rectangle" "normalized" "square" "large"; do
        # Extract URL for the specific key
        URL=$(echo "$IMAGE_URLS" | grep -o "\"$key\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | sed 's/.*":[[:space:]]*"\([^"]*\)".*/\1/')
        if [ -n "$URL" ] && [[ "$URL" == http* ]]; then
            DOWNLOAD_URL="$URL"
            FOUND_KEY="$key"
            echo "  Found URL using key '$key': $DOWNLOAD_URL"
            break
        fi
    done
    
    # If no structured key found, try to extract any URL from the JSON
    if [ -z "$DOWNLOAD_URL" ]; then
        DOWNLOAD_URL=$(echo "$IMAGE_URLS" | grep -o 'https\?://[^"]*' | head -1)
        if [ -n "$DOWNLOAD_URL" ]; then
            echo "  Found URL (generic): $DOWNLOAD_URL"
        fi
    fi
    
    if [ -z "$DOWNLOAD_URL" ]; then
        echo "  ✗ No valid URL found in image_urls JSON"
        echo "FAILED: $image_id - No valid URL in JSON: $IMAGE_URLS" >> "$LOG_FILE"
        ((FAILED_COUNT++))
        continue
    fi
    
    # Determine file extension from URL or use jpg as default
    EXTENSION=$(echo "$DOWNLOAD_URL" | sed 's/.*\.//' | grep -E '^(jpg|jpeg|png|gif|webp|bmp)' || echo "jpg")
    
    # Generate the new filename using the actual size variant that was found
    if [ -n "$FOUND_KEY" ]; then
        NEW_FILENAME="${image_id}_${FOUND_KEY}.${EXTENSION}"
    else
        # If no specific key was found (generic URL), use a default
        NEW_FILENAME="${image_id}_image.${EXTENSION}"
    fi
    FULL_PATH="$IMAGES_DIR/$NEW_FILENAME"
    
    # Check if file already exists
    if [ -f "$FULL_PATH" ]; then
        echo "  ⚠ File already exists: $NEW_FILENAME"
        ((SKIPPED_COUNT++))
        continue
    fi
    
    echo "  Downloading to: $NEW_FILENAME"
    
    # Download the image with retry logic
    DOWNLOAD_SUCCESS=false
    for attempt in 1 2 3; do
        echo "  Attempt $attempt..."
        if curl -L --fail --connect-timeout 10 --max-time 30 -o "$FULL_PATH" "$DOWNLOAD_URL"; then
            DOWNLOAD_SUCCESS=true
            break
        else
            echo "  Attempt $attempt failed (curl exit code: $?)"
            if [ $attempt -lt 3 ]; then
                echo "  Retrying in 2 seconds..."
                sleep 2
            fi
        fi
    done
    
    if [ "$DOWNLOAD_SUCCESS" = true ]; then
        # Verify the file was actually downloaded and has content
        if [ -s "$FULL_PATH" ]; then
            FILE_SIZE=$(stat -f%z "$FULL_PATH" 2>/dev/null || stat -c%s "$FULL_PATH" 2>/dev/null || echo "0")
            echo "  ✓ Downloaded successfully (${FILE_SIZE} bytes)"
            
            # Update database with new filename
            sqlite3 "$DB_FILE" "UPDATE image_entries SET filename = '$NEW_FILENAME' WHERE image_id = '$image_id';"
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Database updated"
                echo "SUCCESS: $image_id -> $NEW_FILENAME" >> "$LOG_FILE"
                ((SUCCESS_COUNT++))
            else
                echo "  ✗ Failed to update database"
                echo "PARTIAL: $image_id - Downloaded but DB update failed" >> "$LOG_FILE"
                ((FAILED_COUNT++))
            fi
        else
            echo "  ✗ Download failed - empty file"
            rm -f "$FULL_PATH"
            echo "FAILED: $image_id - Empty download" >> "$LOG_FILE"
            ((FAILED_COUNT++))
        fi
    else
        echo "  ✗ Download failed after 3 attempts"
        rm -f "$FULL_PATH" 2>/dev/null
        echo "FAILED: $image_id - Download failed: $DOWNLOAD_URL" >> "$LOG_FILE"
        ((FAILED_COUNT++))
    fi
    
    echo ""
    
done < "$MISSING_IDS_FILE"

echo "Download process completed!"
echo ""
echo "Summary:"
echo "========"
echo "Total processed: $TOTAL_COUNT"
echo "Successfully downloaded: $SUCCESS_COUNT"
echo "Failed downloads: $FAILED_COUNT"
echo "Skipped (already exist): $SKIPPED_COUNT"
echo ""
echo "Detailed log saved to: $LOG_FILE"

# Show any failures for review
if [ $FAILED_COUNT -gt 0 ]; then
    echo ""
    echo "Failed downloads:"
    grep "FAILED:" "$LOG_FILE"
fi