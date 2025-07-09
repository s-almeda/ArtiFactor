"""
update_embeddings.py
=====================
This script combines the functionality of extracting text and image features into a single script.

It updates the database with features for any new rows in the `text_entries` and `image_entries` tables
that don't already have embeddings.

1. For text entries:
    - Processes `text_entries` table and updates `vec_description_features` and `vec_value_features` tables.
2. For image entries:
    - Processes `image_entries` table and updates `vec_image_features` table.
3. Skips rows that already have embeddings in the respective tables.
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

# LOCALDB = "LOCALDB"

def update_embeddings():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

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

    # --- IMAGE FEATURES ---
    logging.info("Updating image features...")

    # Create virtual table if it doesn't exist
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_image_features USING vec0(
        image_id TEXT PRIMARY KEY,
        embedding float[2048])
    ''')

    # Retrieve image entries
    cursor.execute('SELECT image_id, value, filename FROM image_entries')
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

        
        logging.info(f"✅ Inserted features for image_id {image_id}: {value}.")

    
    
    # --- CLIP MULTIMODAL FEATURES ---
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
    logging.info(f"Found {len(clip_candidates)} images to process for CLIP.")

    for image_id, title, filename, artist_names_json, descriptions_json, keywords_json in clip_candidates:
        # Skip if already processed
        cursor.execute('SELECT 1 FROM vec_clip_features WHERE image_id = ?', (image_id,))
        if cursor.fetchone():
            logging.info(f"Skipping image_id {image_id} (CLIP features already exist).")
            continue

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
            
            # Add artist names and fetch artist info
            if artist_names_json:
                try:
                    artist_names = json.loads(artist_names_json)
                    if artist_names:
                        text_parts.append(f"by {', '.join(artist_names[:3])}")
                        
                        # Look up each artist in text_entries
                        for artist_name in artist_names[:2]:  # Limit to first 2
                            cursor.execute('''
                                SELECT descriptions FROM text_entries 
                                WHERE LOWER(value) = LOWER(?) AND isArtist = 1
                            ''', (artist_name,))
                            artist_row = cursor.fetchone()
                            if artist_row and artist_row[0]:
                                try:
                                    artist_desc = json.loads(artist_row[0])
                                    # Extract all values from artist description
                                    for source, content in artist_desc.items():
                                        if isinstance(content, dict):
                                            for key, value in content.items():
                                                if isinstance(value, str) and value.strip():
                                                    text_parts.append(f"{key}: {value}")
                                except json.JSONDecodeError:
                                    pass
                except json.JSONDecodeError:
                    pass
            
            # Add artwork descriptions
            if descriptions_json:
                try:
                    desc = json.loads(descriptions_json)
                    for source, content in desc.items():
                        if isinstance(content, dict):
                            for key, value in content.items():
                                if isinstance(value, str) and value.strip():
                                    text_parts.append(f"{key}: {value}")
                        elif isinstance(content, str) and content.strip():
                            text_parts.append(content)
                except json.JSONDecodeError:
                    pass
            
            # Add keywords
            if keywords_json:
                try:
                    keywords = json.loads(keywords_json)
                    if keywords[:5]:
                        text_parts.append(' '.join(keywords[:5]))
                except json.JSONDecodeError:
                    pass
            
            # Combine text (limit length for CLIP)
            combined_text = ' '.join(text_parts)
            if len(combined_text) > 200:
                combined_text = combined_text[:197] + "..."
            
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
            
        except Exception as e:
            logging.error(f"Error processing CLIP features for {image_id}: {e}")
            continue


    
    
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
    

if __name__ == "__main__":
    update_embeddings()