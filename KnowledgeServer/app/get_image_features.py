"""
get_image_features.py
==================
This script extracts image features from images stored in a SQLite database and saves them in a vectorized format.

This script processes images listed in the `image_entries` table of `knowledgebase.db` 
and creates a `vec_image_features` table to store 2048-dimensional feature embeddings. 
If `remake` is True, the table is recreated.

1. Connects to the SQLite database and enables the `sqlite-vec` extension.
2. Creates a virtual table `vec_image_features` for image IDs and embeddings.
3. Loads a pre-trained ResNet50 model (without the classification layer) for feature extraction.
4. Preprocesses images (resize, normalize) and computes embeddings.
5. Inserts embeddings into the table, skipping existing entries if `remake` is False.
6. Logs progress, handles missing files, commits changes, and closes the database connection.
"""



import torch

import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image
import sqlean as sqlite3
import sqlite_vec
import os
import logging

LOCALDB = "LOCALDB"

def create_image_features_table(db_path, images_folder, remake=False):
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    vec_version, = conn.execute("SELECT vec_version()").fetchone()
    logging.info(f"vec_version={vec_version}")

    cursor = conn.cursor()

    if remake:
        # Drop table if it exists
        cursor.execute('DROP TABLE IF EXISTS vec_image_features')
        logging.info("Dropped existing vec_image_features table if it existed.")

    # Create virtual table with vector columns using sqlite-vec
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_image_features USING vec0(
        image_id TEXT PRIMARY KEY,
        embedding float[2048])
    ''')
    logging.info("Created virtual table vec_image_features.")

    # Retrieve image_id and filename from the image_entries table
    cursor.execute('SELECT image_id, filename FROM image_entries')
    entries = cursor.fetchall()
    logging.info(f"Fetched {len(entries)} entries from the image_entries table.")

    # Check device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet50 model and remove the classification layer
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
    model.to(device)
    model.eval()  # Set model to evaluation mode
    logging.info("Loaded ResNet50 model.")

    # Define image preprocessing (Resize, Normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for image_id, filename in entries:
        if not remake:
            # Check if the image_id is already present in the vec_image_features table
            cursor.execute('SELECT 1 FROM vec_image_features WHERE image_id = ?', (image_id,))
            if cursor.fetchone():
                logging.info(f"Skipping image_id {image_id} as it already exists in vec_image_features table.")
                continue

        if not filename:
            logging.warning(f"Skipping entry with image_id {image_id} due to missing filename.")
            continue

        image_path = os.path.join(images_folder, filename)
        if not os.path.exists(image_path):
            logging.warning(f"Image file {image_path} not found. Skipping.")
            continue

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            features = model(image).squeeze().cpu().numpy()
        logging.info(f"Extracted features for image_id {image_id}.")

        # Insert features into the vec_image_features table
        cursor.execute('''
        INSERT INTO vec_image_features (image_id, embedding)
        VALUES (?, ?)
        ''', (image_id, features.tobytes()))
        logging.info(f"Inserted features for image_id {image_id} into vec_image_features table.")

    # Commit and close
    conn.commit()
    logging.info("Committed changes to the database.")
    conn.close()
    logging.info("Closed the database connection.")




if __name__ == "__main__":
    db_path = os.path.join(LOCALDB, "knowledgebase.db")
    images_folder = "LOCALDB/images"
    # Create the image features table
    
    create_image_features_table(db_path, images_folder, remake=True)

