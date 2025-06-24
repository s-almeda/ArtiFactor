#!/bin/bash

# Configuration
DB_FILE="knowledgebase.db"
IMAGES_DIR="images"

echo "Starting image directory audit..."
echo "Database: $DB_FILE"
echo "Images directory: $IMAGES_DIR"
echo ""

# Check if database exists
if [ ! -f "$DB_FILE" ]; then
    echo "Error: Database file '$DB_FILE' not found!"
    exit 1
fi

# Check if images directory exists
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory '$IMAGES_DIR' not found!"
    exit 1
fi

# Create temporary files for comparison
TEMP_DB_FILES=$(mktemp)
TEMP_LOCAL_FILES=$(mktemp)
TEMP_MISSING_IDS=$(mktemp)

# Cleanup function
cleanup() {
    rm -f "$TEMP_DB_FILES" "$TEMP_LOCAL_FILES" "$TEMP_MISSING_IDS"
}
trap cleanup EXIT

echo "1. Extracting filenames from database..."
# Get all non-null, non-empty filenames from database
sqlite3 "$DB_FILE" "SELECT DISTINCT filename FROM image_entries WHERE filename IS NOT NULL AND filename != '';" > "$TEMP_DB_FILES"

# Remove any empty lines
sed -i '/^$/d' "$TEMP_DB_FILES"

DB_FILE_COUNT=$(wc -l < "$TEMP_DB_FILES")
echo "Found $DB_FILE_COUNT files referenced in database"

echo "2. Scanning local images directory..."
# Get all files in the images directory (just filenames, not full paths)
# Use ls instead of find -printf for macOS compatibility
ls "$IMAGES_DIR" | sort > "$TEMP_LOCAL_FILES"

LOCAL_FILE_COUNT=$(wc -l < "$TEMP_LOCAL_FILES")
echo "Found $LOCAL_FILE_COUNT files in local directory"

echo ""
echo "3. Finding files to delete (exist locally but not in database)..."

# Files that exist locally but not in database
TO_DELETE=$(comm -23 "$TEMP_LOCAL_FILES" <(sort "$TEMP_DB_FILES"))
DELETE_COUNT=$(echo "$TO_DELETE" | grep -c .)

if [ "$DELETE_COUNT" -gt 0 ] && [ -n "$TO_DELETE" ]; then
    echo "Files to delete ($DELETE_COUNT):"
    echo "$TO_DELETE"
    echo ""
    
    read -p "Do you want to delete these files? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting unnecessary files..."
        while IFS= read -r filename; do
            if [ -n "$filename" ]; then
                if rm "$IMAGES_DIR/$filename" 2>/dev/null; then
                    echo "Deleted: $filename"
                else
                    echo "Failed to delete: $filename"
                fi
            fi
        done <<< "$TO_DELETE"
    else
        echo "Skipped deletion."
    fi
else
    echo "No files need to be deleted."
fi

echo ""
echo "4. Finding missing files (in database but not locally)..."

# Files that are in database but not local
MISSING_FILES=$(comm -13 "$TEMP_LOCAL_FILES" <(sort "$TEMP_DB_FILES"))
MISSING_COUNT=$(echo "$MISSING_FILES" | grep -c .)

if [ "$MISSING_COUNT" -gt 0 ] && [ -n "$MISSING_FILES" ]; then
    echo "Missing files ($MISSING_COUNT):"
    echo "$MISSING_FILES"
    echo ""
    
    echo "Getting image_ids for missing files..."
    
    # Create SQL query to get image_ids for missing files
    {
        echo "SELECT image_id, filename FROM image_entries WHERE filename IN ("
        first=true
        while IFS= read -r filename; do
            if [ -n "$filename" ]; then
                if [ "$first" = true ]; then
                    echo -n "'$filename'"
                    first=false
                else
                    echo -n ",'$filename'"
                fi
            fi
        done <<< "$MISSING_FILES"
        echo ");"
    } | sqlite3 "$DB_FILE" > "$TEMP_MISSING_IDS"
    
    echo "Missing image_ids and their filenames:"
    echo "image_id|filename"
    echo "----------------"
    cat "$TEMP_MISSING_IDS"
    
    # Also save to a file for later use
    echo "$MISSING_FILES" > missing_files.txt
    cat "$TEMP_MISSING_IDS" > missing_image_ids.txt

    echo ""
    echo "Missing files saved to: missing_files.txt"
    echo "Missing image_ids saved to: missing_image_ids.txt"
else
    # Overwrite files with blank content if no missing files
    > missing_files.txt
    > missing_image_ids.txt
    echo "No missing files found - all database entries have corresponding local files!"
fi

echo ""
echo "Audit Summary:"
echo "=============="
echo "Database entries: $DB_FILE_COUNT"
echo "Local files: $LOCAL_FILE_COUNT"
echo "Files to delete: $DELETE_COUNT"
echo "Missing files: $MISSING_COUNT"
echo ""
echo "Audit completed!"