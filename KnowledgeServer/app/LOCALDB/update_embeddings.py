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

    for image_id, value, filename in image_entries:
        # Skip if already in `vec_image_features`
        cursor.execute('SELECT 1 FROM vec_image_features WHERE image_id = ?', (image_id,))
        if cursor.fetchone():
            logging.info(f"Skipping image_id {image_id}: {value} (already indexed).")
            continue

        if not filename:
            logging.warning(f"Skipping image_id {image_id} due to missing filename.")
            continue

        image_path = os.path.join(images_folder, filename)
        if not os.path.exists(image_path):
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

if __name__ == "__main__":
    update_embeddings()