
# from itertools import chain
import numpy as np
import sqlean as sqlite3
import struct
import pandas as pd
import os
import base64
import json
import requests

# --- imports for using ResNet50  --- #
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np

# The database paths inside the container will always be:
MODEL_CACHE_DIR = "/root/.cache/torch/hub"

# Get the absolute path to the /app directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This gets "/app"
TEXT_DB_PATH = os.path.join(BASE_DIR, "LOCALDB", "text.db")
IMAGE_DB_PATH = os.path.join(BASE_DIR, "LOCALDB", "image.db")
IMAGES_PATH = os.path.join(BASE_DIR, "LOCALDB", "images")

from sentence_transformers import SentenceTransformer
print("Loading MiniLM text encoding model...")
text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("loaded MiniLM!")

# # Will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading ResNet50 from {MODEL_CACHE_DIR}...")
# Load ResNet50 weights and remove the last classification layer
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
# Move model to correct device
model.to(device)
model.eval()  # Set model to evaluation mode


def extract_img_features(img):
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

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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


def find_most_similar_texts(text_features, conn, top_k=3):
    """
    given a vector of text embeddings
    Find the top-k most similar texts by cosine similarity.
    Takes in the text features and a database connection.
    Returns a pandas.DataFrame containing the entry_ids and distances of the top-k most similar texts.
    """
    print("Finding similar texts...")

    # Query the vector database (on the text descriptions for each entry) for the most similar texts
    query = """
        SELECT
            id,
            distance
        FROM vec_description_features
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """
    rows = conn.execute(query, [text_features, top_k]).fetchall()

    # Convert the results to a DataFrame
    similar_texts = pd.DataFrame(rows, columns=["entry_id", "distance"])
    return similar_texts


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
    
def find_exact_matches(query, conn):
    """
    Find exact matches for a query in the text database.
    Looks for matches in the 'value' or 'aliases' columns, ignoring case sensitivity.
    """
    print(f"Finding matches for '{query}'...")

    #conn.row_factory = sqlite3.Row

    # Prepare the query to find exact matches in 'value' or 'aliases'
    query_lower = query.lower()
    sql_query = """
        SELECT * FROM text_entries
        WHERE LOWER(value) = ?
    """
    cursor = conn.execute(sql_query, (query_lower, f"%{query_lower}%"))
    rows = cursor.fetchall()

    # Convert the results to a list of dictionaries
    matches = [{key: row[key] for key in row.keys()} for row in rows]

    match_values = [match['value'] for match in matches]
    print(f"Found {match_values} for the query '{query}'")

    return matches

def safe_json_loads(json_string, default=None):
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}

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