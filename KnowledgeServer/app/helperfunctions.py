
# from itertools import chain
import numpy as np
import sqlean as sqlite3
import struct
import pandas as pd
import os
import json
import requests
import umap
# -- image conversion -- #
import base64
from io import BytesIO
from PIL import (Image, UnidentifiedImageError)




# --- imports for using ResNet50  --- #
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# -- imports for using CLIP -- #
from transformers import CLIPProcessor, CLIPModel

# -- for using KMeans clustering -- #
from sklearn.cluster import KMeans


# The database paths inside the container will always be:
from config import (
    IMAGES_PATH, 
    MODEL_CACHE_DIR, 
    TRANSFORMERS_CACHE_DIR,
)

from sentence_transformers import SentenceTransformer
print("Loading MiniLM text encoding model...")
text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# print("loaded MiniLM!")

# # Will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading ResNet50 from {MODEL_CACHE_DIR}...")
# Load ResNet50 weights and remove the last classification layer
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
# Move model to correct device
model.to(device)
model.eval()  # Set model to evaluation mode

#-- CLIP model loading --#
# Then use TRANSFORMERS_CACHE_DIR for CLIP:
print(f"Loading CLIP model...")
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=TRANSFORMERS_CACHE_DIR)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=TRANSFORMERS_CACHE_DIR)
# clip_model.to(device)
# clip_model.eval()

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir=TRANSFORMERS_CACHE_DIR,
    use_safetensors=True
)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir=TRANSFORMERS_CACHE_DIR
)

clip_model.to(device)
clip_model.eval()
print("Loaded CLIP!")



# --------------- Function Definitions ------------------- #

# ========== Functions that extract features ===========

# Extracts CLIP features for an image + text pair
def extract_clip_multimodal_features(image, text):
    """
    Extract CLIP embeddings for image-text pair.
    Returns concatenated features to preserve both visual and semantic information.
    
    Args:
        image: PIL Image object
        text: String text from retrieve_artwork_text()
    
    Returns:
        numpy array of concatenated embeddings (1024D)
    """
    # Truncate text to avoid token length issues
    # CLIP has a max of 77 tokens, so we truncate conservatively
    if len(text) > 200:  # Rough character limit
        text = text[:200] + "..."
    
    # Process image and text separately to ensure consistent tokenization
    image_inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    text_inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    
    # Move to device
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        # Get embeddings separately
        image_outputs = clip_model.get_image_features(**image_inputs)
        text_outputs = clip_model.get_text_features(**text_inputs)
        
        # Normalize
        image_features = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
        text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        
        # Concatenate
        features = torch.cat([image_features, text_features], dim=-1)
        features = features.cpu().numpy().squeeze()
    
    return features

def extract_img_features(img): # USING RESNET50!
    """
    Extract features from a PIL image using ResNet50.
    Returns a 2048D feature vector.
    """
    # Preprocess images sent from the client
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = preprocess(img).unsqueeze(0).to(device)  # Preprocess & add batch dim

    with torch.no_grad():
        features = model(img_tensor)  # Extract features

    result = features.view(-1).cpu().numpy()  # Flatten as NumPy array
    print("Extracted feature vector shape:", result.shape)  
    return result

def extract_text_features(text): 
    """
    Extract features from text using MiniLM
    Returns a feature vector.
    """
    features_array = text_model.encode(text) #364 float array
    return features_array

# ==== Functions that act on embeddings ====

def reduce_to_2d_umap(embeddings, n_neighbors=5, min_dist=0.5, random_state=42):
    """
    Reduce high-dimensional embeddings to 2D coordinates using UMAP.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
                   Can be CLIP (512D) or ResNet50 (2048D) embeddings
        n_neighbors: int, number of neighbors for UMAP (default 8)
        min_dist: float, minimum distance between points (default 0.1)
        random_state: int, for reproducibility (default 42)
    
    Returns:
        numpy array of shape (n_samples, 2) with x,y coordinates
    """
    # Adjust n_neighbors if we have too few samples
    n_samples = embeddings.shape[0]
    n_neighbors = min(n_neighbors, n_samples - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    coordinates_2d = reducer.fit_transform(embeddings)
    
    # Normalize to [0, 1] range for easier visualization
    min_coords = coordinates_2d.min(axis=0)
    max_coords = coordinates_2d.max(axis=0)
    coordinates_2d_normalized = (coordinates_2d - min_coords) / (max_coords - min_coords)
    
    return coordinates_2d_normalized


def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def apply_kmeans_clustering(coordinates_2d, k):
    """
    Run k-means clustering on 2D coordinates.
    
    Args:
        coordinates_2d: numpy array of shape (n_samples, 2)
        k: number of clusters
    
    Returns:
        numpy array of cluster labels
    """
    # Adjust k if we have fewer points than clusters
    n_samples = coordinates_2d.shape[0]
    k = min(k, n_samples)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coordinates_2d)
    
    return cluster_labels


# ====== Functions for retrieving stuff from the database ======

def retrieve_by_id(entry_id, conn, entry_type="image"):
    """
    Retrieve a single entry by its ID from either image_entries or text_entries.

    Args:
        entry_id: ID of the entry (image_id or entry_id)
        conn: database connection
        entry_type: "image" (default) or "text"

    Returns:
        dict: Entry details or None if not found
    """
    if entry_type == "text":
        table = "text_entries"
        id_column = "entry_id"
    else:
        table = "image_entries"
        id_column = "image_id"

    cursor = conn.execute(f"SELECT * FROM {table} WHERE {id_column} = ?", (entry_id,))
    row = cursor.fetchone()
    if row:
        return dict(row)
    else:
        return None
    
def convert_row_to_text(row):
    """
    Convert image_entries row into text for CLIP.
    
    Args:
        row: sqlite3.Row object from image_entries table
    
    Returns:
        str: Combined text description (limited length for CLIP)
    """
    text_parts = []
    
    # Title
    if row['value']:
        text_parts.append(row['value'])
    
    # Artists
    if row['artist_names']:
        try:
            artists = json.loads(row['artist_names'])
            if artists:
                text_parts.append(f"by {', '.join(artists[:3])}")
        except json.JSONDecodeError:
            pass
    
    # Artwork description - grab all key-value pairs
    if row['descriptions']:
        try:
            desc = json.loads(row['descriptions'])
            for source, content in desc.items():
                if isinstance(content, dict):
                    for key, value in content.items():
                        if isinstance(value, str) and value.strip():
                            text_parts.append(f"{key}: {value}")
                elif isinstance(content, str) and content.strip():
                    text_parts.append(content)
        except json.JSONDecodeError:
            pass
    
    # Related keywords
    if row['relatedKeywordStrings']:
        try:
            keywords = json.loads(row['relatedKeywordStrings'])
            if keywords[:5]:  # Just first 5
                text_parts.append(' '.join(keywords[:5]))
        except json.JSONDecodeError:
            pass
    
    result = ' '.join(text_parts) if text_parts else 'Untitled artwork'
    
    # Final length check
    if len(result) > 200:
        result = result[:197] + "..."
    
    return result

def find_semantic_keyword_matches(ngrams, text_db, threshold=0.3, top_k=3):
    """
    Given a series of phrases, which are n-grams of the input text,
    Finds the most semantically similar keywords using SQLite's `vec0` extension.
    Returns:
    list: A list of dictionaries containing:
        - "phrase" (str): The phrase (an n-gram of the input text).
        - "id" (int): The ID of the matched entry in the database.
        - "similarity" (float): The similarity score of the match.
    """
    matches = []

    for phrase, _, _ in ngrams:  # Extract only the phrase
        phrase_embedding = extract_text_features(phrase)  # Convert phrase to embedding
        serialized_embedding = serialize_f32(phrase_embedding)  # Convert embedding to binary format
        query = """
            SELECT id, distance
            FROM vec_value_features
            WHERE embedding MATCH ?
            ORDER BY distance ASC
            LIMIT ?
        """
        cursor = text_db.execute(query, [serialized_embedding, top_k])  # Correct parameterized query

        rows = cursor.fetchall()
        for row in rows:
            similarity = 1 - row["distance"]  # Convert distance to similarity
            if similarity >= threshold:
                matches.append({
                    "phrase": phrase,
                    "id": row["id"],
                    "similarity": similarity
                })

    return matches




def find_most_similar_texts(text_features, conn, top_k=3, search_in="description"):
    """
    Find the top-k most similar texts by cosine similarity.
    Args:
        text_features: embedding vector (serialized)
        conn: database connection
        top_k: number of results to return
        search_in: "description", "value", or "both"
    Returns:
        pandas.DataFrame with entry_id and distance columns
    """
    print("Finding similar texts...")

    queries = []
    params = []

    if search_in == "description":
        queries.append("""
            SELECT id AS entry_id, distance
            FROM vec_description_features
            WHERE embedding MATCH ?
        """)
        params.append([text_features])
    elif search_in == "value":
        queries.append("""
            SELECT id AS entry_id, distance
            FROM vec_value_features
            WHERE embedding MATCH ?
        """)
        params.append([text_features])
    elif search_in == "both":
        queries.append("""
            SELECT id AS entry_id, distance
            FROM vec_description_features
            WHERE embedding MATCH ?
        """)
        queries.append("""
            SELECT id AS entry_id, distance
            FROM vec_value_features
            WHERE embedding MATCH ?
        """)
        params = [[text_features], [text_features]]
    else:
        raise ValueError("search_in must be 'description', 'value', or 'both'")

    # Collect results from all queries
    all_rows = []
    for q, p in zip(queries, params):
        rows = conn.execute(q + " ORDER BY distance LIMIT ?", p + [top_k]).fetchall()
        all_rows.extend(rows)

    # Combine, deduplicate by entry_id, and sort by distance
    if all_rows:
        df = pd.DataFrame(all_rows, columns=["entry_id", "distance"])
        df = df.sort_values("distance").drop_duplicates("entry_id").head(top_k).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["entry_id", "distance"])

    return df


def retrieve_text_details(similar_texts, conn):
    """
    Given a DataFrame of similar texts, retrieve detailed information from the database.
    """
    result = []

    for row in similar_texts.itertuples():  # similar_texts comes from the previous function, is a pd DataFrame
        query = "SELECT * FROM text_entries WHERE entry_id = ?"
        cursor = conn.execute(query, (row.entry_id,))
        matched_entry = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        if not matched_entry.empty:
            entry = matched_entry.iloc[0].to_dict()
            print(entry)

            # Parse descriptions
            descriptions = entry.get("descriptions")
            parsed_descriptions = None
            if descriptions:
                try:
                    parsed_descriptions = json.loads(descriptions)
                except json.JSONDecodeError:
                    print("Error parsing descriptions JSON", descriptions)

            # Retrieve images for the entry
            try:
                image_ids = json.loads(entry.get("images", "[]"))
            except json.JSONDecodeError:
                image_ids = entry.get("images", [])
            images = get_images_from_image_ids(image_ids, conn)

            result.append({
                "entry_id": entry["entry_id"],
                "database_value": entry["value"],
                "type": entry["type"],
                "isArtist": bool(entry.get("isArtist", 0)),
                "description": entry.get("short_description"),
                "full_description": parsed_descriptions,
                "relatedKeywordIds": entry.get("relatedKeywordIds", []),
                "relatedKeywordStrings": entry.get("relatedKeywordStrings", "").split(", ") if entry.get("relatedKeywordStrings") else [],
                "images": images
            })

    return json.dumps(result)


def get_images_from_image_ids(image_ids, conn, max=3):
    """
    Given a list of image IDs, retrieve their validated URLs or base64 representations.
    Returns a list of up to 'max' images.
    """
    result = []
    conn.row_factory = sqlite3.Row  # Process rows as dictionary-like objects

    for image_id in image_ids[:max]:  # Limit to 'max' images
        query = "SELECT * FROM image_entries WHERE image_id = ?"
        cursor = conn.execute(query, (image_id,))
        matched_entry = cursor.fetchone()

        if matched_entry:
            entry = dict(matched_entry)  # Convert row to dictionary
            image_urls = entry.get('image_urls', {})
            if isinstance(image_urls, str):
                try:
                    image_urls = json.loads(image_urls)
                except json.JSONDecodeError:
                    image_urls = {}

            image_data = None

            # Try to get the first valid image URL from 'large', 'medium', or 'small'
            for size in ['large', 'medium', 'larger', 'small', 'square', 'tall']:
                image_url = image_urls.get(size)
                if image_url and check_image_url(image_url):
                    print("✅")
                    image_data = image_url
                    break

            # If no valid URL, fallback to base64 from filename
            if not image_data:
                try:
                    print("Trying to load image from file...")
                    image_path = os.path.join(IMAGES_PATH, entry['filename'])
                    with open(image_path, "rb") as image_file:
                        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    image_data = f"data:image/jpeg;base64,{image_base64}"
                except FileNotFoundError:
                    print("Failed to load image from file, skipping...")
                    continue

            result.append(image_data)

        if len(result) >= max:  # Stop if we reach the max number of images
            break

    return result

#-- get precomputed embeddings from the vector tables --#

def get_clip_embeddings(db, image_ids):
    """Get CLIP embeddings for given image IDs."""
    if not image_ids:
        return {}
    
    placeholders = ','.join(['?' for _ in image_ids])
    query = f"""
        SELECT image_id, embedding 
        FROM vec_clip_features 
        WHERE image_id IN ({placeholders})
    """
    
    cursor = db.execute(query, image_ids)
    results = cursor.fetchall()
    
    embeddings = {}
    for row in results:
        embedding_data = row['embedding']
        if isinstance(embedding_data, bytes):
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
        else:
            try:
                embedding = np.array(embedding_data, dtype=np.float32)
            except (ValueError, TypeError):
                continue
        embeddings[row['image_id']] = embedding
    
    return embeddings


# Get ResNet50 embeddings
def get_resnet_embeddings(db, image_ids):
    if not image_ids:
        return {}
    placeholders = ','.join(['?' for _ in image_ids])
    query = f"""
        SELECT image_id, embedding 
        FROM vec_image_features 
        WHERE image_id IN ({placeholders})
    """
    cursor = db.execute(query, image_ids)
    results = cursor.fetchall()
    embeddings = {}
    for row in results:
        embedding_data = row['embedding']
        if isinstance(embedding_data, bytes):
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
        else:
            try:
                embedding = np.array(embedding_data, dtype=np.float32)
            except (ValueError, TypeError):
                continue
        embeddings[row['image_id']] = embedding
    return embeddings


def get_value_embeddings(db, entry_ids):
    """Get value embeddings for given text_entries entry_ids (artists and keywords)."""
    if not entry_ids:
        return {}
    
    placeholders = ','.join(['?' for _ in entry_ids])
    query = f"""
        SELECT id, embedding 
        FROM vec_value_features 
        WHERE id IN ({placeholders})
    """
    
    cursor = db.execute(query, entry_ids)
    results = cursor.fetchall()
    
    embeddings = {}
    for row in results:
        embedding_data = row['embedding']
        if isinstance(embedding_data, bytes):
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
        else:
            try:
                embedding = np.array(embedding_data, dtype=np.float32)
            except (ValueError, TypeError):
                continue
        embeddings[row['id']] = embedding
    
    return embeddings
def get_description_embeddings(db, entry_ids):
    """Get description embeddings for given text_entries entry_ids (artists and keywords)."""
    if not entry_ids:
        return {}
    
    placeholders = ','.join(['?' for _ in entry_ids])
    query = f"""
        SELECT id, embedding 
        FROM vec_description_features 
        WHERE id IN ({placeholders})
    """
    
    cursor = db.execute(query, entry_ids)
    results = cursor.fetchall()
    
    embeddings = {}
    for row in results:
        embedding_data = row['embedding']
        if isinstance(embedding_data, bytes):
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
        else:
            try:
                embedding = np.array(embedding_data, dtype=np.float32)
            except (ValueError, TypeError):
                continue
        embeddings[row['id']] = embedding
    
    return embeddings



def find_exact_matches(query, conn, artists_only=False):
    """
    Find exact matches for a query in the text database.
    Looks for matches in the 'value' column, ignoring case sensitivity.
    
    Args:
        query: search string
        conn: database connection
        artists_only: if True, only return results where isArtist = 1
    
    Returns:
        list of matching rows as dictionaries
    """
    #print(f"Finding matches for '{query}'...")
    
    query_lower = query.lower()
    
    if artists_only:
        sql_query = """
            SELECT * FROM text_entries
            WHERE LOWER(value) = ? AND isArtist = 1
        """
    else:
        sql_query = """
            SELECT * FROM text_entries
            WHERE LOWER(value) = ?
        """
    
    cursor = conn.execute(sql_query, (query_lower,))
    rows = cursor.fetchall()
    
    # Convert the results to a list of dictionaries
    matches = [{key: row[key] for key in row.keys()} for row in rows]
    
    match_values = [match['value'] for match in matches]
    #print(f"Found {match_values} for the query '{query}'")
    
    return matches
# === Formatting / Preprocessing / data processing functions ===

def preprocess_text(text, max_length=3):
    """
    Preprocesses text and extracts candidate phrases along with their positions.
    - Splits text into words
    - Generates n-grams (1-word to max_length words, up to 3-grams)
    
    Returns:
    - List of tuples [(phrase, start_index, end_index)]
    """
    words = text.lower().split()  # Split text into words
    candidate_phrases = []

    # Generate n-grams (1-word to max_length words, up to 3-grams)
    for n in range(1, max_length + 1):
        for i in range(len(words) - n + 1):
            ngram_words = words[i:i + n]
            ngram = ' '.join(ngram_words)
            start_index = i  # Start index of first word
            end_index = i + n - 1  # End index of last word
            candidate_phrases.append((ngram, start_index, end_index))

    return candidate_phrases


def serialize_f32(vector):
    """Serializes a list of floats into a compact 'raw bytes' format for SQLite vector search."""
    return struct.pack("%sf" % len(vector), *vector)


def check_image_url(url):
    """
    Check if an image URL is valid (not 404).
    Returns True if the image exists, False if it returns 404.
    """
    try:
        print("checking image url...", url, end="")
        response = requests.head(url, allow_redirects=True, timeout=5)  # Use HEAD to save bandwidth
        #print(response.status_code == 200)
        return response.status_code == 200  # Returns True if the URL is valid
    except requests.RequestException:
        print("❌ Error checking image URL:", url)
        return False  # Return False if there's a connection error
    

def safe_json_loads(json_string, default=None):
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}

def base64_to_image(base64_string):
    # what we will use for now before we have user database solution
    # Convert base64 string to PIL Image
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        return img
    except (base64.binascii.Error, UnidentifiedImageError) as e:
        print(f"Error decoding base64 image: {e}")
        return None

def url_to_image(url):
    # what we will use once we have stable image urls
    # Convert image URL to PIL Image
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img


def match_input_text_to_keywords(original_words, candidate_keywords, all_matches, keyword_details):
    final_results = []
    seen_indices = set()

    i = 0
    while i < len(original_words):
        matched_keyword = None

        # Check if this position starts a matched phrase
        for phrase, start_idx, end_idx in candidate_keywords:
            if start_idx == i:
                match = next((m for m in all_matches if m["phrase"] == phrase), None)
                if match:
                    details = keyword_details.get(match["id"])
                    if details:
                        original_phrase = ' '.join(original_words[start_idx:end_idx + 1])  # Restore original words
                        matched_keyword = {
                            "id": details["entry_id"],
                            "value": original_phrase,  # Use original words
                            "database_value": details["value"],
                            "type": details["type"],
                            "isArtist": bool(details.get("isArtist", 0)),
                            "artist_aliases": details.get("artist_aliases", []),
                            "description": details.get("short_description"),
                            "full_description": details.get("descriptions"),
                            "relatedKeywordIds": details.get("relatedKeywordIds", []),
                            "relatedKeywordStrings": details.get("relatedKeywordStrings", "").split(", ") if details.get("relatedKeywordStrings") else [],
                            "images": details.get("images", [])
                        }
                        seen_indices.update(range(start_idx, end_idx + 1))
                        i = end_idx + 1  # Skip past the full phrase
                        break  # Stop checking other matches

        # If a matched keyword was found, add it as a single entry
        if matched_keyword:
            final_results.append(matched_keyword)
        else:
            if i not in seen_indices:
                final_results.append({"value": original_words[i]})
            i += 1

    return final_results