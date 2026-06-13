"""
update_comics_embeddings.py
===========================
Adds vector embeddings to the comics database (comics.db).

Mirrors the structure of update_embeddings.py for the art history database,
but adapted for the comics schema:

  text_entries  (series, creators, publishers)
    -> vec_description_features  float[384]  MiniLM on type+value+descriptions
    -> vec_value_features         float[384]  MiniLM on value alone

  image_entries  (individual comic pages)
    -> vec_image_features         float[2048] ResNet50 visual features
    -> vec_clip_features          float[1024] CLIP image+text multimodal
    -> vec_artworktext_features   float[384]  MiniLM on page text

Creates the virtual tables if they don't exist yet (safe to run on a fresh db).

Usage:
    python3 update_comics_embeddings.py                    # normal update mode
    python3 update_comics_embeddings.py --clip remake      # remake all CLIP + artworktext embeddings

Note: activate the project venv first with `venv_pls`
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


DB_PATH = "comics.db"
IMAGES_DIR = "comic_images"


def update_comics_embeddings(remake_clip=False):
    logging.basicConfig(level=logging.INFO)

    operation_description = []
    if remake_clip:
        operation_description.append("remaking CLIP from scratch")
        operation_description.append("remaking page text features from scratch")
    else:
        operation_description.append("updating CLIP")
        operation_description.append("updating page text features")
    operation_description.append("updating ResNet50 and MiniLM")

    confirmation_msg = f"Will be {', '.join(operation_description)}. Continue? (y/n): "
    user_input = input(confirmation_msg).strip().lower()
    if user_input not in ['y', 'yes']:
        print("Operation cancelled.")
        return

    if not os.path.exists(DB_PATH):
        print(f"Error: '{DB_PATH}' not found. Build the comics database first with build_comicsbase.py.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    vec_version, = conn.execute("SELECT vec_version()").fetchone()
    logging.info(f"vec_version={vec_version}")

    cursor = conn.cursor()

    # -------------------------------------------------------------------------
    # TEXT FEATURES  (text_entries: series, creators, publishers)
    # -------------------------------------------------------------------------
    logging.info("--- TEXT FEATURES (text_entries) ---")

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

    text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logging.info("Loaded SentenceTransformer model.")

    # Comics text_entries: entry_id, value, images, isArtist, type,
    #                      artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings
    cursor.execute(
        'SELECT entry_id, type, value, artist_aliases, descriptions FROM text_entries'
    )
    text_entries = cursor.fetchall()
    logging.info(f"Fetched {len(text_entries)} text entries.")

    updated_text_entries = []

    # vec_description_features
    for entry_id, type_, value, artist_aliases, descriptions in text_entries:
        cursor.execute('SELECT 1 FROM vec_description_features WHERE id = ?', (entry_id,))
        if cursor.fetchone():
            logging.info(f"Skipping {entry_id} in vec_description_features (already exists).")
            continue

        if descriptions:
            full_description = f"{type_}, {value}, {artist_aliases}, {descriptions}"
            features_array = text_model.encode(full_description)
            cursor.execute(
                'INSERT INTO vec_description_features (id, embedding) VALUES (?, ?)',
                (entry_id, features_array.tobytes())
            )
            updated_text_entries.append(value)
            logging.info(f"Inserted description features for {entry_id}: {value}")

            if len(updated_text_entries) % 250 == 0:
                conn.commit()
                logging.info(f"Committed {len(updated_text_entries)} description entries.")

    conn.commit()
    logging.info("Committed description features.")

    # vec_value_features
    for entry_id, _, value, _, _ in text_entries:
        cursor.execute('SELECT 1 FROM vec_value_features WHERE id = ?', (entry_id,))
        if cursor.fetchone():
            logging.info(f"Skipping {entry_id} in vec_value_features (already exists).")
            continue

        if value:
            features_array = text_model.encode(value)
            cursor.execute(
                'INSERT INTO vec_value_features (id, embedding) VALUES (?, ?)',
                (entry_id, features_array.tobytes())
            )
            logging.info(f"Inserted value features for {entry_id}: {value}")

            if len(updated_text_entries) % 250 == 0:
                conn.commit()

    conn.commit()
    logging.info("Committed value features.")

    # -------------------------------------------------------------------------
    # IMAGE FEATURES  (image_entries: individual comic pages)
    # -------------------------------------------------------------------------
    logging.info("--- IMAGE FEATURES (ResNet50) ---")

    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_image_features USING vec0(
        image_id TEXT PRIMARY KEY,
        embedding float[2048])
    ''')

    cursor.execute(
        'SELECT image_id, value, image_urls, filename FROM image_entries'
    )
    image_entries = cursor.fetchall()
    logging.info(f"Fetched {len(image_entries)} image entries.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    image_model = torch.nn.Sequential(*list(image_model.children())[:-1])
    image_model.to(device)
    image_model.eval()
    logging.info("Loaded ResNet50 model.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images_folder = os.path.join(os.getcwd(), IMAGES_DIR)

    updated_image_entries = []

    for image_id, value, image_urls, filename in image_entries:
        cursor.execute('SELECT 1 FROM vec_image_features WHERE image_id = ?', (image_id,))
        if cursor.fetchone():
            logging.info(f"Skipping {image_id}: {value} (already indexed).")
            continue

        image_path = None

        # Try local file first
        if filename:
            candidate = os.path.join(images_folder, filename)
            if os.path.exists(candidate):
                image_path = candidate

        # Fall back to downloading from remote URL
        if not image_path:
            image_url = None
            if image_urls:
                try:
                    urls = json.loads(image_urls)
                    # Comics image_urls keys: full, full_local, thumb, thumb_remote
                    for key in ["full", "thumb_remote"]:
                        if key in urls and urls[key]:
                            image_url = urls[key]
                            break
                except Exception as e:
                    logging.warning(f"Could not parse image_urls for {image_id}: {e}")

            if image_url:
                ext = os.path.splitext(image_url)[1] or ".jpg"
                dl_filename = f"{image_id}{ext}"
                image_path = os.path.join(images_folder, dl_filename)
                if not os.path.exists(image_path):
                    try:
                        response = requests.get(image_url, timeout=15)
                        response.raise_for_status()
                        os.makedirs(images_folder, exist_ok=True)
                        with open(image_path, "wb") as f:
                            f.write(response.content)
                        logging.info(f"Downloaded {image_id} from {image_url}")
                    except Exception as e:
                        logging.warning(f"Failed to download {image_id}: {e}")
                        continue
            else:
                logging.warning(f"Skipping {image_id}: no local file and no valid image_url.")
                continue

        if not image_path or not os.path.exists(image_path):
            logging.warning(f"Image file not found for {image_id}. Skipping.")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = image_model(img).squeeze().cpu().numpy()

            cursor.execute(
                'INSERT INTO vec_image_features (image_id, embedding) VALUES (?, ?)',
                (image_id, features.tobytes())
            )
            updated_image_entries.append(value)

            if len(updated_image_entries) % 10 == 0:
                conn.commit()
                logging.info(f"Committed {len(updated_image_entries)} image entries.")

            logging.info(f"Inserted ResNet50 features for {image_id}: {value}")
        except Exception as e:
            logging.error(f"Error processing ResNet50 for {image_id}: {e}")
            continue

    conn.commit()
    logging.info(f"Committed all ResNet50 image features ({len(updated_image_entries)} entries).")

    # -------------------------------------------------------------------------
    # CLIP MULTIMODAL FEATURES
    # -------------------------------------------------------------------------
    if remake_clip:
        logging.info("--- CLIP FEATURES (remaking all from scratch) ---")
    else:
        logging.info("--- CLIP FEATURES (updating new entries) ---")

    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_clip_features USING vec0(
        image_id TEXT PRIMARY KEY,
        embedding float[1024])
    ''')

    logging.info("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    clip_model.eval()
    logging.info("Loaded CLIP model.")

    cursor.execute('''
        SELECT image_id, value, filename, artist_names, descriptions, relatedKeywordStrings
        FROM image_entries
    ''')
    clip_candidates = cursor.fetchall()
    logging.info(f"Found {len(clip_candidates)} image entries for CLIP.")

    updated_clip_entries = []

    for image_id, title, filename, artist_names_json, descriptions_json, keywords_json in clip_candidates:
        cursor.execute('SELECT 1 FROM vec_clip_features WHERE image_id = ?', (image_id,))
        existing = cursor.fetchone()

        if existing and not remake_clip:
            logging.info(f"Skipping {image_id} (CLIP already exists).")
            continue
        elif existing and remake_clip:
            cursor.execute('DELETE FROM vec_clip_features WHERE image_id = ?', (image_id,))
            logging.info(f"Deleted existing CLIP for {image_id} (remaking).")

        if not filename:
            logging.warning(f"Skipping {image_id}: no filename.")
            continue

        image_path = os.path.join(images_folder, filename)
        if not os.path.exists(image_path):
            logging.warning(f"Image not found for {image_id}: {image_path}. Skipping.")
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            text_parts = []

            if title:
                text_parts.append(title)

            if artist_names_json:
                try:
                    artist_names = json.loads(artist_names_json)
                    if artist_names:
                        text_parts.append(f"by {', '.join(artist_names[:3])}")
                except json.JSONDecodeError:
                    pass

            # Pull book_title and cover_date from descriptions
            if descriptions_json:
                try:
                    desc = json.loads(descriptions_json)
                    for source, content in desc.items():
                        if isinstance(content, dict):
                            for key, val in content.items():
                                if key in ("book_title", "cover_date", "description") and isinstance(val, str) and val.strip():
                                    text_parts.append(val)
                        elif isinstance(content, str) and content.strip():
                            text_parts.append(content)
                except json.JSONDecodeError:
                    pass

            # Add keyword strings (series name, creator names, etc.)
            if keywords_json:
                try:
                    keywords = json.loads(keywords_json)
                    if keywords:
                        text_parts.extend(keywords)
                except json.JSONDecodeError:
                    pass

            combined_text = ', '.join(text_parts)
            if len(combined_text) > 300:
                combined_text = combined_text[:297] + "..."

            inputs = clip_processor(
                text=combined_text, images=image,
                return_tensors="pt", padding=True, truncation=True, max_length=77
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = clip_model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                combined_features = torch.cat([image_features, text_features], dim=-1)
                combined_features = combined_features.cpu().numpy().squeeze()

            cursor.execute(
                'INSERT INTO vec_clip_features (image_id, embedding) VALUES (?, ?)',
                (image_id, combined_features.tobytes())
            )
            updated_clip_entries.append(title or image_id)
            logging.info(f"Inserted CLIP features for {image_id}: {title}")

            if len(updated_clip_entries) % 250 == 0:
                conn.commit()
                logging.info(f"Committed {len(updated_clip_entries)} CLIP entries.")

        except Exception as e:
            logging.error(f"Error processing CLIP for {image_id}: {e}")
            continue

    conn.commit()
    logging.info(f"Committed all CLIP features ({len(updated_clip_entries)} entries).")

    # -------------------------------------------------------------------------
    # ARTWORK TEXT FEATURES  (MiniLM on page text, same candidates as CLIP)
    # -------------------------------------------------------------------------
    if remake_clip:
        logging.info("--- PAGE TEXT FEATURES (remaking all from scratch) ---")
    else:
        logging.info("--- PAGE TEXT FEATURES (updating new entries) ---")

    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_artworktext_features USING vec0(
        image_id TEXT PRIMARY KEY,
        embedding float[384])
    ''')

    cursor.execute('''
        SELECT image_id, value, filename, artist_names, descriptions, relatedKeywordStrings, ocr_text
        FROM image_entries
    ''')
    artworktext_candidates = cursor.fetchall()

    updated_artworktext_entries = []

    for image_id, title, filename, artist_names_json, descriptions_json, keywords_json, ocr_text in artworktext_candidates:
        cursor.execute('SELECT 1 FROM vec_artworktext_features WHERE image_id = ?', (image_id,))
        existing = cursor.fetchone()

        if existing and not remake_clip:
            logging.info(f"Skipping {image_id} (page text features already exist).")
            continue
        elif existing and remake_clip:
            cursor.execute('DELETE FROM vec_artworktext_features WHERE image_id = ?', (image_id,))
            logging.info(f"Deleted existing page text features for {image_id} (remaking).")

        try:
            text_parts = []

            if title:
                text_parts.append(title)

            if artist_names_json:
                try:
                    artist_names = json.loads(artist_names_json)
                    if artist_names:
                        text_parts.append(f"by {', '.join(artist_names[:3])}")
                except json.JSONDecodeError:
                    pass

            if descriptions_json:
                try:
                    desc = json.loads(descriptions_json)
                    for source, content in desc.items():
                        if isinstance(content, dict):
                            for key, val in content.items():
                                if key in ("book_title", "cover_date", "description") and isinstance(val, str) and val.strip():
                                    text_parts.append(val)
                        elif isinstance(content, str) and content.strip():
                            text_parts.append(content)
                except json.JSONDecodeError:
                    pass

            if keywords_json:
                try:
                    keywords = json.loads(keywords_json)
                    if keywords:
                        text_parts.extend(keywords)
                except json.JSONDecodeError:
                    pass

            # OCR text from the page image — most content-rich signal for text search
            if ocr_text and ocr_text.strip():
                text_parts.append(ocr_text.strip())

            combined_text = ' | '.join(text_parts)
            if not combined_text.strip():
                logging.warning(f"Skipping {image_id}: empty text content.")
                continue

            text_features = text_model.encode(combined_text)
            cursor.execute(
                'INSERT INTO vec_artworktext_features (image_id, embedding) VALUES (?, ?)',
                (image_id, text_features.tobytes())
            )
            updated_artworktext_entries.append(title or image_id)
            logging.info(f"Inserted page text features for {image_id}: {title}")

            if len(updated_artworktext_entries) % 250 == 0:
                conn.commit()
                logging.info(f"Committed {len(updated_artworktext_entries)} page text entries.")

        except Exception as e:
            logging.error(f"Error processing page text features for {image_id}: {e}")
            continue

    conn.commit()
    logging.info(f"Committed all page text features ({len(updated_artworktext_entries)} entries).")

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    conn.close()
    logging.info("Closed database connection.")

    logging.info(f"Text entries updated: {len(updated_text_entries)}" if updated_text_entries else "No new text entries.")
    logging.info(f"Image entries (ResNet50) updated: {len(updated_image_entries)}" if updated_image_entries else "No new image entries.")
    logging.info(f"CLIP entries updated: {len(updated_clip_entries)}" if updated_clip_entries else "No new CLIP entries.")
    logging.info(f"Page text entries updated: {len(updated_artworktext_entries)}" if updated_artworktext_entries else "No new page text entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add vector embeddings to comics.db')
    parser.add_argument('--clip', choices=['remake'],
                        help='Specify "remake" to regenerate all CLIP + page text embeddings from scratch')
    args = parser.parse_args()

    update_comics_embeddings(remake_clip=(args.clip == 'remake'))
