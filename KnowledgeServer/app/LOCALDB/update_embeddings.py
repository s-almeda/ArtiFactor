"""
update_embeddings.py
=====================
This script combines the functionality of extracting text, image, and CLIP multimodal features into a single script.

It updates the database with features for any new rows in the `text_entries` and `image_entries` tables
that don't already have embeddings.

1. For text entries:
    - Processes `text_entries` table and updates `vec_description_features` and `vec_value_features` tables.
2. For image entries:
    - Processes `image_entries` table and updates `vec_image_features` table.
3. For CLIP multimodal features:
    - Processes `image_entries` table and updates `vec_clip_features` table with combined image and text embeddings.
4. Skips rows that already have embeddings in the respective tables (unless CLIP embeddings are remade).

Usage:
    python3 update_embeddings.py                    # Normal update mode
    python3 update_embeddings.py --clip remake      # Remake all CLIP embeddings from scratch
"""

import sqlite3
import logging
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import sqlite_vec
import sqlean as sqlite3
from transformers import CLIPModel, CLIPProcessor
import json
import requests
import argparse
import sys

# LOCALDB = "LOCALDB"

def update_embeddings(remake_clip=False):
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Show confirmation prompt
    operation_description = []
    if remake_clip:
        operation_description.append("remaking CLIP from scratch")
        operation_description.append("remaking artwork text features from scratch")
    else:
        operation_description.append("updating CLIP")
        operation_description.append("updating artwork text features")

    operation_description.append("updating ResNet50 and MiniLM")

    confirmation_msg = f"Will be {', '.join(operation_description)}. Continue? (y/n): "
    user_input = input(confirmation_msg).strip().lower()

    if user_input not in ['y', 'yes']:
        print("Operation cancelled.")
        return

    # Connect to SQLite database
    # db_path = os.path.join(LOCALDB, "knowledgebase.db")
    db_path = ("knowledgebase.db")
    conn = sqlite3.connect(db_path)

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    vec_version, = conn.execute("SELECT vec_version()").fetchone()
    logging.info(f"vec_version={vec_version}")

    cursor = conn.cursor()

    # Track updated entries
    updated_text_entries = []
    updated_image_entries = []

    # --- TEXT FEATURES ---
    logging.info("Updating text features...")

    # Create virtual tables if they don't exist
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_description_features USING vec0(
        id TEXT PRIMARY KEY,
        embedding float[384])
    ''')
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_value_features USING vec0(
        id TEXT PRIMARY KEY,
        embedding float[384])
    ''')

    # Load the SentenceTransformer model
    text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logging.info("Loaded SentenceTransformer model.")

    # Process `vec_description_features`
    cursor.execute('SELECT entry_id, type, value, artist_aliases, descriptions FROM text_entries')
    text_entries = cursor.fetchall()
    logging.info(f"Fetched {len(text_entries)} text entries.")

    for entry_id, type_, value, artist_aliases, descriptions in text_entries:
        # Skip if already in `vec_description_features`
        cursor.execute('SELECT 1 FROM vec_description_features WHERE id = ?', (entry_id,))
        if cursor.fetchone():
            logging.info(f"Skipping entry_id {entry_id} in vec_description_features (already exists).")
            continue

        if descriptions:
            full_description = f"{type_}, {value}, {artist_aliases}, {descriptions}"
            features_array = text_model.encode(full_description)
            cursor.execute('''
            INSERT INTO vec_description_features (id, embedding)
            VALUES (?, ?)
            ''', (entry_id, features_array.tobytes()))
            updated_text_entries.append(value)  # Track updated text entry
            logging.info(f"✅ - Inserted description features for entry_id {entry_id}:{value}.")

            # Commit every 250 entries to save progress
            if len(updated_text_entries) % 250 == 0:
                conn.commit()
                logging.info(f"💾 Committed progress: {len(updated_text_entries)} text description entries processed.")


    # commit after finishing description features
    conn.commit()
    logging.info("Committed description changes to the database.")

    # Process `vec_value_features`
    for entry_id, _, value, _, _ in text_entries:
        # Skip if already in `vec_value_features`
        cursor.execute('SELECT 1 FROM vec_value_features WHERE id = ?', (entry_id,))
        if cursor.fetchone():
            logging.info(f"Skipping entry_id {entry_id} in vec_value_features (already exists).")
            continue

        if value:
            features_array = text_model.encode(value)
            cursor.execute('''
            INSERT INTO vec_value_features (id, embedding)
            VALUES (?, ?)
            ''', (entry_id, features_array.tobytes()))
            logging.info(f"✅ - Inserted value features for entry_id {entry_id}:{value}.")

            # Commit every 250 entries to save progress
            if len(updated_text_entries) % 250 == 0:
                conn.commit()
                logging.info(f"💾 Committed progress: {len(updated_text_entries)} text value entries processed.")

    # commit after finishing value features
    conn.commit()
    logging.info("Committed value changes to the database.")
    # --- IMAGE FEATURES ---
    logging.info("Updating image features...")

    # Create virtual table if it doesn't exist
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_image_features USING vec0(
        image_id TEXT PRIMARY KEY,
        embedding float[2048])
    ''')

    # Retrieve image entries
    cursor.execute('SELECT image_id, value, image_urls, filename FROM image_entries')
    image_entries = cursor.fetchall()
    logging.info(f"Fetched {len(image_entries)} image entries.")

    # Check device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet50 model and remove the classification layer
    image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    image_model = torch.nn.Sequential(*list(image_model.children())[:-1])  # Remove final classification layer
    image_model.to(device)
    image_model.eval()
    logging.info("Loaded ResNet50 model.")

    # Define image preprocessing (Resize, Normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images_folder = os.path.join(os.getcwd(), "images")

    for image_id, value, image_urls, filename in image_entries:
        # Skip if already in `vec_image_features`
        cursor.execute('SELECT 1 FROM vec_image_features WHERE image_id = ?', (image_id,))
        if cursor.fetchone():
            logging.info(f"Skipping image_id {image_id}: {value} (already indexed).")
            continue

        image_path = None

        # Try to use filename if present and file exists
        if filename:
            image_path_candidate = os.path.join(images_folder, filename)
            if os.path.exists(image_path_candidate):
                image_path = image_path_candidate

        # If image_path is still None, try to use image_urls to download
        if not image_path:
            image_url = None
            if image_urls:
                try:
                    urls = json.loads(image_urls)
                    # Priority order for image sizes
                    priority_keys = ["small", "medium", "medium_rectangle", "normalized", "large", "larger"]
                    for key in priority_keys:
                        if key in urls and urls[key]:
                            image_url = urls[key]
                            break
                except Exception as e:
                    logging.warning(f"Could not parse image_urls for image_id {image_id}: {e}")

            if image_url:
                # Generate a filename from image_id and url extension
                ext = os.path.splitext(image_url)[1] or ".jpg"
                filename = f"{image_id}{ext}"
                image_path = os.path.join(images_folder, filename)
                # Download the image if not already present
                if not os.path.exists(image_path):
                    try:
                        response = requests.get(image_url, timeout=10)
                        response.raise_for_status()
                        with open(image_path, "wb") as f:
                            f.write(response.content)
                        logging.info(f"Downloaded image for image_id {image_id} from {image_url}")
                    except Exception as e:
                        logging.warning(f"❌ - Failed to download image for image_id {image_id} from {image_url}: {e}")
                        continue
            else:
                logging.warning(f"Skipping image_id {image_id} due to missing filename and valid image_urls.")
                continue

        if not image_path or not os.path.exists(image_path):
            logging.warning(f"❌ - Image file {image_path} not found. Skipping.")
            continue

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            features = image_model(image).squeeze().cpu().numpy()
        logging.info(f"Extracted features for image_id {image_id}.")

        # Insert features into the vec_image_features table
        cursor.execute('''
        INSERT INTO vec_image_features (image_id, embedding)
        VALUES (?, ?)
        ''', (image_id, features.tobytes()))
        updated_image_entries.append(value)  # Track updated image entry

        # Commit every 250 entries to save progress
        if len(updated_image_entries) % 10 == 0:
            conn.commit()
            logging.info(f"💾 Committed progress: {len(updated_image_entries)} image entries processed.")

        
        logging.info(f"✅ Inserted features for image_id {image_id}: {value}.")

    # Commit any remaining changes
    conn.commit()
    logging.info(f"💾 Committed progress: {len(updated_image_entries)} image entries processed.")

    
    # --- CLIP MULTIMODAL FEATURES ---
    if remake_clip:
        logging.info("Remaking ALL CLIP multimodal features from scratch...")
    else:
        logging.info("Updating CLIP multimodal features...")

    # Create virtual table if it doesn't exist
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_clip_features USING vec0(
        image_id TEXT PRIMARY KEY,
        embedding float[1024])
    ''')

    # Load CLIP model
    logging.info("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    clip_model.eval()
    logging.info("Loaded CLIP model.")

    # Track updated CLIP entries
    updated_clip_entries = []

    # Process each image entry
    cursor.execute('''
        SELECT i.image_id, i.value, i.filename, i.artist_names, i.descriptions, i.relatedKeywordStrings
        FROM image_entries i
    ''')
    clip_candidates = cursor.fetchall()
    if remake_clip:
        logging.info(f"Found {len(clip_candidates)} images to REMAKE for CLIP (will process ALL).")
    else:
        logging.info(f"Found {len(clip_candidates)} images to process for CLIP (will skip existing).")

    for image_id, title, filename, artist_names_json, descriptions_json, keywords_json in clip_candidates:
        # Check if already processed
        cursor.execute('SELECT 1 FROM vec_clip_features WHERE image_id = ?', (image_id,))
        existing_record = cursor.fetchone()
        
        if existing_record and not remake_clip:
            logging.info(f"Skipping image_id {image_id} (CLIP features already exist).")
            continue
        elif existing_record and remake_clip:
            # Delete existing entry if remaking
            cursor.execute('DELETE FROM vec_clip_features WHERE image_id = ?', (image_id,))
            logging.info(f"Deleted existing CLIP features for image_id {image_id} (remaking).")

        if not filename:
            logging.warning(f"Skipping image_id {image_id} due to missing filename.")
            continue

        image_path = os.path.join(images_folder, filename)
        if not os.path.exists(image_path):
            logging.warning(f"❌ - Image file {image_path} not found. Skipping.")
            continue

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Build text representation
            text_parts = []
            
            # Add title
            if title:
                text_parts.append(title)
            
            # Add artist names (no additional artist info lookup)
            if artist_names_json:
                try:
                    artist_names = json.loads(artist_names_json)
                    if artist_names:
                        text_parts.append(f"by {', '.join(artist_names[:3])}")
                except json.JSONDecodeError:
                    pass
            
            # Add artwork descriptions (values only, no keys)
            if descriptions_json:
                try:
                    desc = json.loads(descriptions_json)
                    for source, content in desc.items():
                        if isinstance(content, dict):
                            for key, value in content.items():
                                if isinstance(value, str) and value.strip():
                                    # Only add the value, not the key
                                    text_parts.append(value)
                        elif isinstance(content, str) and content.strip():
                            text_parts.append(content)
                except json.JSONDecodeError:
                    pass
            
            # Add keywords
            if keywords_json:
                try:
                    keywords = json.loads(keywords_json)
                    if keywords:
                        text_parts.extend(keywords)  # Add all keywords individually
                except json.JSONDecodeError:
                    pass
            
            # Combine text (limit length for CLIP)
            combined_text = ', '.join(text_parts)  # Use comma separation for better readability
            if len(combined_text) > 300:  # Increased limit to accommodate more descriptive text
                combined_text = combined_text[:297] + "..."
            
            # Process with CLIP
            inputs = clip_processor(text=combined_text, images=image, return_tensors="pt", 
                                  padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                
                # Get normalized embeddings
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Concatenate features
                combined_features = torch.cat([image_features, text_features], dim=-1)
                combined_features = combined_features.cpu().numpy().squeeze()
            
            # Insert into database
            cursor.execute('''
                INSERT INTO vec_clip_features (image_id, embedding)
                VALUES (?, ?)
            ''', (image_id, combined_features.tobytes()))
            
            updated_clip_entries.append(title or image_id)
            logging.info(f"✅ Inserted CLIP features for {image_id}: {title}")

            # Commit every 250 entries to save progress
            if len(updated_clip_entries) % 250 == 0:
                conn.commit()
                logging.info(f"💾 Committed progress: {len(updated_clip_entries)} CLIP entries processed.")
            
        except Exception as e:
            logging.error(f"Error processing CLIP features for {image_id}: {e}")
            continue


    # --- ARTWORK TEXT FEATURES (NEW) ---
    if remake_clip:
        logging.info("Remaking ALL artwork text features from scratch...")
    else:
        logging.info("Updating artwork text features...")

    # Create virtual table if it doesn't exist
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_artworktext_features USING vec0(
        image_id TEXT PRIMARY KEY,
        embedding float[384])
    ''')

    # Track updated artwork text entries
    updated_artworktext_entries = []

    # Process each image entry (same candidates as CLIP)
    cursor.execute('''
        SELECT i.image_id, i.value, i.filename, i.artist_names, i.descriptions, i.relatedKeywordStrings
        FROM image_entries i
    ''')
    artworktext_candidates = cursor.fetchall()
    if remake_clip:
        logging.info(f"Found {len(artworktext_candidates)} artworks to REMAKE text features for (will process ALL).")
    else:
        logging.info(f"Found {len(artworktext_candidates)} artworks to process for text features (will skip existing).")

    for image_id, title, filename, artist_names_json, descriptions_json, keywords_json in artworktext_candidates:
        # Check if already processed
        cursor.execute('SELECT 1 FROM vec_artworktext_features WHERE image_id = ?', (image_id,))
        existing_record = cursor.fetchone()

        if existing_record and not remake_clip:
            logging.info(f"Skipping image_id {image_id} (artwork text features already exist).")
            continue
        elif existing_record and remake_clip:
            # Delete existing entry if remaking
            cursor.execute('DELETE FROM vec_artworktext_features WHERE image_id = ?', (image_id,))
            logging.info(f"Deleted existing artwork text features for image_id {image_id} (remaking).")

        try:
            # Build text representation (SAME as CLIP but without image processing)
            text_parts = []

            # Add title
            if title:
                text_parts.append(title)

            # Add artist names
            if artist_names_json:
                try:
                    artist_names = json.loads(artist_names_json)
                    if artist_names:
                        text_parts.append(f"by {', '.join(artist_names[:3])}")
                except json.JSONDecodeError:
                    pass

            # Add artwork descriptions (values only, no keys)
            if descriptions_json:
                try:
                    desc = json.loads(descriptions_json)
                    for source, content in desc.items():
                        if isinstance(content, dict):
                            for key, value in content.items():
                                if isinstance(value, str) and value.strip():
                                    # Only add the value, not the key
                                    text_parts.append(value)
                        elif isinstance(content, str) and content.strip():
                            text_parts.append(content)
                except json.JSONDecodeError:
                    pass

            # Add keywords
            if keywords_json:
                try:
                    keywords = json.loads(keywords_json)
                    if keywords:
                        text_parts.extend(keywords)  # Add all keywords individually
                except json.JSONDecodeError:
                    pass

            # Combine text (no length limit for MiniLM since it's more flexible)
            combined_text = ', '.join(text_parts)  # Use comma separation for better readability

            if not combined_text.strip():
                # Skip if no meaningful text content
                logging.warning(f"Skipping image_id {image_id} due to empty text content.")
                continue

            # Process with MiniLM (reuse the text_model from earlier)
            text_features = text_model.encode(combined_text)

            # Insert into database
            cursor.execute('''
                INSERT INTO vec_artworktext_features (image_id, embedding)
                VALUES (?, ?)
            ''', (image_id, text_features.tobytes()))

            updated_artworktext_entries.append(title or image_id)
            logging.info(f"✅ Inserted artwork text features for {image_id}: {title}")

            # Commit every 250 entries to save progress
            if len(updated_artworktext_entries) % 250 == 0:
                conn.commit()
                logging.info(f"💾 Committed progress: {len(updated_artworktext_entries)} artwork text entries processed.")

        except Exception as e:
            logging.error(f"Error processing artwork text features for {image_id}: {e}")
            continue

    # Commit artwork text changes
    conn.commit()
    logging.info(f"💾 Committed progress: {len(updated_artworktext_entries)} artwork text entries processed.")

    # Commit and close
    conn.commit()
    logging.info("Committed changes to the database.")
    conn.close()
    logging.info("Closed the database connection.")


    # Log updated entries
    if updated_text_entries:
        logging.info(f"Updated {len(updated_text_entries)} text entries: {', '.join(updated_text_entries)}.")
    else:
        logging.info("No new text entries.")

    if updated_image_entries:
        logging.info(f"Updated {len(updated_image_entries)} image entries: {', '.join(updated_image_entries)}.")
    else:
        logging.info("No new image entries.")
    
    # At the end, log the CLIP updates:
    if updated_clip_entries:
        logging.info(f"Updated {len(updated_clip_entries)} CLIP entries.")
    else:
        logging.info("No new CLIP entries.")
    
    # And log the artwork text updates:
    if updated_artworktext_entries:
        logging.info(f"Updated {len(updated_artworktext_entries)} artwork text entries.")
    else:
        logging.info("No new artwork text entries.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update embeddings in the database')
    parser.add_argument('--clip', choices=['remake'], 
                       help='Specify "remake" to regenerate all CLIP embeddings from scratch')
    
    args = parser.parse_args()
    
    remake_clip = args.clip == 'remake'
    
    update_embeddings(remake_clip=remake_clip)