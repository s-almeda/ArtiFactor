# index.py
# run me with './bootstrap.sh' in terminal
import json
from flask import Flask, jsonify, request, g
from difflib import SequenceMatcher

# --- imports for using ResNet50  --- #
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
# -- image conversion -- #
import base64
from io import BytesIO
from PIL import (Image, UnidentifiedImageError)

#from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

#using sqlean to make using extensions easy,,
import sqlite_vec
import sqlean as sqlite3
# then using the sqlite vector extension... https://alexgarcia.xyz/sqlite-vec/python.html

import helperfunctions as hf# helper functions including preprocess_text

import requests, re, os, ast

#todo, incorporate exact matches into keyword checking too


print("Mabuhay! Loading...")


# # Check if we're inside Docker (set by Docker when running)
# if os.getenv("RUNNING_IN_DOCKER"):
#     TEXT_DB_PATH = "/app/LOCALDB/text.db"  
#     IMAGE_DB_PATH = "/app/LOCALDB/wikiart.db"  
#     IMAGES_PATH = "/app/LOCALDB/images/"  
#     MODEL_CACHE_DIR = "/root/.cache/torch/hub"
# else:
#     TEXT_DB_PATH = "../../LOCALDB/text.db"
#     IMAGE_DB_PATH = "../../LOCALDB/wikiart.db"
#     IMAGES_PATH = "../../images/"
#     MODEL_CACHE_DIR = os.path.expanduser("~/model_cache/")



# # ----- Load the databases, so we can use the info in them to respond to user requests ----- #
# print(f"Using Text DB: {TEXT_DB_PATH}")
# print(f"Using Image DB: {IMAGE_DB_PATH}")
# print(f"Using Images Path: {IMAGES_PATH}")
# os.environ["TORCH_HOME"] = MODEL_CACHE_DIR

# The database paths inside the container will always be:
MODEL_CACHE_DIR = "/root/.cache/torch/hub"

# Get the absolute path to the /app directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This gets "/app"

# Define paths relative to the correct base directory
TEXT_DB_PATH = os.path.join(BASE_DIR, "LOCALDB", "text.db")
IMAGE_DB_PATH = os.path.join(BASE_DIR, "LOCALDB", "image.db")
IMAGES_PATH = os.path.join(BASE_DIR, "LOCALDB", "images")

# Debugging: Print paths
print(f"✅ Using Text DB: {TEXT_DB_PATH}")
print(f"✅ Using Image DB: {IMAGE_DB_PATH}")
print(f"✅ Using Images Path: {IMAGES_PATH}")
# Check if files exist
if not os.path.exists(TEXT_DB_PATH):
    print(f"🚨 ERROR: Text DB not found at {TEXT_DB_PATH}")

if not os.path.exists(IMAGE_DB_PATH):
    print(f"🚨 ERROR: Image DB not found at {IMAGE_DB_PATH}")

if not os.path.exists(IMAGES_PATH):
    print(f"🚨 ERROR: Images directory not found at {IMAGES_PATH}")

# Test call to the text database
try:
    with sqlite3.connect(TEXT_DB_PATH) as text_db:
        cursor = text_db.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if tables:
            print(f"✅ Text DB is valid. Tables: {tables}")
        else:
            print(f"🚨 ERROR: Text DB at {TEXT_DB_PATH} is empty or invalid.")
except sqlite3.Error as e:
    print(f"🚨 ERROR: Failed to connect to Text DB at {TEXT_DB_PATH}. Error: {e}")


# # Will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading ResNet50 from {MODEL_CACHE_DIR}...")
# Load ResNet50 weights and remove the last classification layer
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
# Move model to correct device
model.to(device)
model.eval()  # Set model to evaluation mode


def load_sqlite_vec(db):
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    print("loaded sqlite vector extension...")
    return

# --- to load up straight from huggingface (if you haven't run load_dataset_to_database.py yet ---#)
# print("loading up portion of the dataset...")
# # load up small portion of the dataset...
# imageset = load_dataset("Artificio/WikiArt_Full", split="train[:1000]")#[:1%]")
# imageset = imageset.remove_columns(['embeddings_pca512', 'resnet50_non_robust_feats', 'resnet50_robust_feats'])
# features_csv = "../image_features.csv"


print("Done! Time to run the app...")

app = Flask(__name__)



@app.route("/")
def hello_world():
    print("User connected...")

        # Use "dogs" as the query text
    query_text = "dogs"
    top_k = 5
    print(f"Query text: {query_text}")
    print(f"Top K: {top_k}")

    # Find similar texts
    query_features = hf.extract_text_features(query_text)
    print(f"Query features shape: {query_features.shape}")

    with sqlite3.connect(IMAGE_DB_PATH) as image_db, sqlite3.connect(TEXT_DB_PATH) as text_db:
        image_db.row_factory = sqlite3.Row
        text_db.row_factory = sqlite3.Row

        # Query the image database to get a random row
        image_query = "SELECT value, filename FROM image_entries ORDER BY RANDOM() LIMIT 1"
        image_cursor = image_db.execute(image_query)
        image_row = image_cursor.fetchone()

        if image_row:
            image_row_dict = {key: image_row[key] for key in image_row.keys()}
        else:
            image_row_dict = {"error": "No data found in image database."}

        print("Testing entry search; looking for match in database...")
        # Find matching entry in the image database
        random_image_vec_row = find_matching_entry(image_row_dict.get("filename", ""), image_db) if image_row else None

        # Query the text database for additional information
        text_query = "SELECT * FROM text_entries ORDER BY RANDOM() LIMIT 1"
        text_cursor = text_db.execute(text_query)
        text_row = text_cursor.fetchone()

        if text_row:
            text_row_dict = {key: text_row[key] for key in text_row.keys()}
        else:
            text_row_dict = {"error": "No data found in text database."}

        # Find matching entry in the text database
        text_db.enable_load_extension(True)
        sqlite_vec.load(text_db)
        text_db.enable_load_extension(False)

        similar_texts_df = hf.find_most_similar_texts(query_features, text_db, top_k=top_k)
        print(f"Received similar texts: {similar_texts_df}")

        # Convert DataFrame to list of dictionaries
        similar_texts = similar_texts_df.to_dict(orient='records')

        # Ensure similar_texts is a DataFrame with the necessary columns
        similar_texts_df = pd.DataFrame(similar_texts)

        # Get detailed information of similar texts
        result = hf.retrieve_text_details(similar_texts_df, text_db)
        detailed_result = json.loads(result)
        for i, row in enumerate(similar_texts_df.itertuples()):
            detailed_result[i]['distance'] = row.distance

        # Print just the result.database_value
        print("Found matches for query: " + query_text)
        for item in detailed_result:
            print(item.get('database_value', 'N/A'))

        #load extension for image_db
        image_db.enable_load_extension(True)
        sqlite_vec.load(image_db)
        image_db.enable_load_extension(False)
        
        # Use a predefined image URL
        image_url = "https://d32dm0rphc51dk.cloudfront.net/gTPexURCjkBek6MrG7g1bg/small.jpg"
        print(f"Using image URL: {image_url}")

        # Load the image from the URL
        img = url_to_image(image_url)

        # Extract features from the image
        query_features = extract_features(img)
        print("Query features shape:", query_features.shape)

        # Find the 3 most similar images
        similar_images = find_most_similar_images(query_features, image_db, top_k=3)
        print(similar_images)

        # Get detailed information of similar images
        image_result = retrieve_from_imageset(similar_images, image_db)
        print("Result in imageset:", result)


        # add code for checking that check_keywords works
        # Tokenize original input text while keeping stopwords
        input_text = "dog eating a sandwich abstract expressionism portraiture cezanne"

        # Look for semantically best matches

        # Step 2: Find semantically similar matches
        final_keyword_results = keyword_check(input_text, threshold=0.3)

       
    # Format the output with <br> for readability
    result_str = "Hello! If you can read this, the ML/data server is running.<br><br>"
    result_str += "Random Image DB Row:<br>" + "<br>".join([f"{k}: {v}" for k, v in image_row_dict.items()]) + "<br><br>"
    result_str += "Random Dataset Row:<br>" + json.dumps(random_image_vec_row, indent=2, ensure_ascii=False).replace("\n", "<br>") + "<br><br>"
    result_str += "Random Text DB Row:<br>" + "<br>".join([f"{k}: {v}" for k, v in text_row_dict.items()]) + "<br>"


        # Add the similar texts to the result_str
    result_str += "Similar Text Matches for 'dogs':<br>"
    for item in detailed_result:
        result_str += f"- {item.get('database_value', 'N/A')} (Distance: {item.get('distance', 'N/A')})<br>"

        # Add the results to the result_str
    keywords_result = final_keyword_results
    result_str += "Keywords Result for 'dog eating a sandwich':<br>"
    for keyword in keywords_result:
        result_str += f"- Value: {keyword.get('value', 'N/A')}<br>"
        result_str += f"  Database Value: {keyword.get('database_value', 'N/A')}<br>"
        result_str += f"  Type: {keyword.get('type', 'N/A')}<br>"
        result_str += f"  Description: {keyword.get('description', 'N/A')}<br><br>"


     # Add the image similarity results to the result_str
    result_str += f"<img src='{image_url}' alt='Query Image' style='max-width:200px;'><br><br>"
    result_str += "Similar Image Matches:<br>"
    for item in json.loads(image_result):
        result_str += f"- Title: {item.get('title', 'N/A')}<br>"
        result_str += f"  Artists: {', '.join(item.get('artists', []))}<br>"
        result_str += f"  Description: {item.get('short_description', 'N/A')}<br>"
        result_str += f"  Image: <img src='{item.get('image', '')}' alt='Image' style='max-width:200px;'><br><br>"


    print("DONE! Sending home page... ")
    return f"<pre>{result_str}</pre>"  # Preserve formatting in the browser



#--------------------------- TEXT HANDLING --------------------------#
@app.route('/keyword_check', methods=['POST'])
def handle_keyword_check():
    input_text = request.json['text']
    threshold = request.json.get('threshold', 0.3)
    print(f"Received input text: {input_text}")
    print(f"Using threshold: {threshold}")

    # Call the helper function to process the keyword check
    final_results = keyword_check(input_text, threshold)

    # Return the results as a JSON response
    return jsonify({"words": final_results})


def keyword_check(input_text, threshold):
    """
    Helper function to process keyword checking.
    """
    # Tokenize original input text while keeping stopwords
    original_words = input_text.split()  # Preserves all words

    # Step 1: Preprocess text to get candidate keywords with positions
    candidate_keywords = hf.preprocess_text(input_text)  # Returns (phrase, start_idx, end_idx)

    # Look for semantically best matches
    with sqlite3.connect(TEXT_DB_PATH) as text_db:
        load_sqlite_vec(text_db)
        text_db.row_factory = sqlite3.Row

        # Step 2: Find semantically similar matches
        matches = hf.find_semantic_keyword_matches(candidate_keywords, text_db, threshold)
        print(f"Semantic Matches: {matches}")

        # Step 3: Retrieve keyword details from `keywords` table
        matched_ids = [match["id"] for match in matches]
        keyword_details = {}

        if matched_ids:
            query = f"SELECT * FROM text_entries WHERE entry_id IN ({','.join(['?'] * len(matched_ids))})"
            cursor = text_db.execute(query, matched_ids)
            keyword_details = {row["entry_id"]: dict(row) for row in cursor.fetchall()}

        # Step 4: Build Final Ordered List (Preserving Stopwords & Multi-Word Matches)
        final_results = []
        seen_indices = set()

        i = 0
        while i < len(original_words):
            matched_keyword = None

            # Check if this position starts a matched phrase
            for phrase, start_idx, end_idx in candidate_keywords:
                if start_idx == i:
                    match = next((m for m in matches if m["phrase"] == phrase), None)
                    if match:
                        details = keyword_details.get(match["id"])
                        if details:
                            print(details)
                            original_phrase = ' '.join(original_words[start_idx:end_idx + 1])  # Restore original words
                            matched_keyword = {
                                "id": details["entry_id"],
                                "value": original_phrase,  # Use original words
                                "database_value": details["value"],
                                "type": details["type"],
                                "description": details["short_description"] or details["descriptions"].get("artsy", "No description available"),
                                "relatedKeywordIds": details.get("relatedKeywordIds", []),
                                "relatedKeywordStrings": details.get("relatedKeywordStrings", "").split(",") if details.get("relatedKeywordStrings") else [],
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

        for result in final_results:
            print(f"Value: {result.get('value', 'N/A')}, Database Value: {result.get('database_value', 'N/A')}, {result.get('description', 'n/a')}")

    return final_results




@app.route('/lookup_text', methods=['POST'])
def handle_text(): 
    """
    given a text query, find and return the most similar text entries in the database
    """
    print("Received request for text handling...")
    with sqlite3.connect(TEXT_DB_PATH) as text_db:
        text_db.enable_load_extension(True)
        sqlite_vec.load(text_db)
        text_db.enable_load_extension(False)
        
        # Get text from request
        query_text = request.json['query']
        top_k = request.json.get('top_k', 5)
        print(f"Query text: {query_text}")
        print(f"Top K: {top_k}")
        
        # Find similar texts
        query_features = hf.extract_text_features(query_text)
        print(f"Query features shape: {query_features.shape}")

        # Find the most similar text entries
        similar_texts_df = hf.find_most_similar_texts(query_features, text_db, top_k=top_k)
        print(f"received similar texts: {similar_texts_df}")

        # Convert DataFrame to list of dictionaries
        similar_texts = similar_texts_df.to_dict(orient='records')

        # Ensure similar_texts is a DataFrame with the necessary columns
        similar_texts_df = pd.DataFrame(similar_texts)
        print("DATAFRAME:", similar_texts_df)

        # Get detailed information of similar texts

        result = hf.retrieve_text_details(similar_texts_df, text_db)
        detailed_result = json.loads(result)
        for item in detailed_result:
            print(f"Value: {item.get('database_value', 'N/A')}, Description: {item.get('description', 'N/A')}")

        #print("DETAILED RESULTs:", detailed_result)
        for i, row in enumerate(similar_texts_df.itertuples()):
            if i < len(detailed_result):
                detailed_result[i]['distance'] = row.distance
            else:
                detailed_result.append({'distance': 1.5})

        # Print just the result.database_value
        print("found matches for query: " + query_text)
        # for item in detailed_result:
        #     print(item.get('database_value', 'N/A'))

        return jsonify(detailed_result)





def get_db_connection():
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = sqlite3.connect(IMAGE_DB_PATH)
        g.sqlite_db.row_factory = sqlite3.Row  # Ensure results can be accessed as dictionaries
    return g.sqlite_db

@app.teardown_appcontext
def close_db_connection(exception):
    db = getattr(g, 'sqlite_db', None)
    if db is not None:
        db.close()


#--------------------------- IMAGE HANDLING --------------------------#
@app.route('/image', methods=['POST'])
def handle_image():
    print("received request for image handling...")
    image_db = get_db_connection()
    load_sqlite_vec(image_db)

    # Get image from request
    if 'image' in request.json and hf.check_image_url(request.json['image']):
        img = url_to_image(request.json['image'])
    else:
        img = base64_to_image(request.json['image'])
    
    # Extract features from the posted image
    query_features = extract_features(img)
    print("Query features shape:", query_features.shape)

    # Find the 3 most similar images
    similar_images = find_most_similar_images(query_features, image_db, top_k=3)
    print(similar_images)
    # Get detailed information of similar images
    result = retrieve_from_imageset(similar_images, image_db)
    print("Result in imageset:", result)
    return jsonify(result)

def find_most_similar_images(image_features, conn, top_k=3):
    """
    Find the top-k most similar images by cosine similarity.
    Takes in the image features and a database connection.
    Returns a list of dictionaries containing the ids and distances of the top-k most similar images.
    """
    print("Finding similar images...")

    # Query the database for the most similar images
    query = """
        SELECT
            image_id,
            distance
        FROM vec_image_features
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """
    rows = conn.execute(query, [image_features, top_k]).fetchall()

    # Convert the results to a list of dictionaries
    similar_images = [{"image_id": row[0], "distance": row[1]} for row in rows]
    print("Found similar images:", similar_images)

    return similar_images

def retrieve_from_imageset(similar_images, conn):
    """
    Given a list of similar images, retrieve detailed information from the database.
    """
    result = []
    conn.row_factory = sqlite3.Row  # Process rows as dictionary-like objects

    for image in similar_images:
        query = "SELECT * FROM image_entries WHERE image_id = ?"
        cursor = conn.execute(query, (image["image_id"],))
        print(f"Looking for image with ID: {image['image_id']}")

        matched_entry = cursor.fetchone()
        if matched_entry:
            entry = dict(matched_entry)  # Convert row to dictionary
            image_urls = entry.get('image_urls', {})
            print("Type of image_urls:", type(image_urls))
            print("image_urls1: ", image_urls)

           # Ensure image_urls is a dictionary
            if isinstance(image_urls, str):
                try:
                    # Use ast.literal_eval instead of json.loads
                    image_urls = ast.literal_eval(image_urls)
                    print("image_urls2:", image_urls)
                except (ValueError, SyntaxError):
                    print("Failed to parse image_urls. Using empty dictionary.")
                    image_urls = {}
            print("Final image_urls:", image_urls)
            image_data = None

            # Try to get the first valid image URL from 'large', 'medium', or 'small'
            for size in [  'large', 'medium', 'larger', 'small', 'square','tall']:
                print("image_urls: ", image_urls)
                image_url = image_urls.get(size)
                print(f"Trying image URL for size '{size}': {image_url}")
                if image_url and hf.check_image_url(image_url):
                    image_data = image_url
                    break
                
            if not image_data:
                try:
                    print("Trying to load image from file...")
                    image_path = os.path.join(IMAGES_PATH, entry['filename'])
                    with open(image_path, "rb") as image_file:
                        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    image_data = f"data:image/jpeg;base64,{image_base64}"
                except FileNotFoundError:
                    print("Failed, using default image...")
                    image_data = "https://upload.wikimedia.org/wikipedia/commons/a/a3/Image-not-found.png"

            # Process artist names as a list
            artist_names = entry.get('artist_names', 'Unknown').split(", ")

            # Process descriptions as a JSON object
            descriptions = entry.get('descriptions', '[]')
            if isinstance(descriptions, str):
                try:
                    # Attempt to parse descriptions as JSON
                    descriptions = json.loads(descriptions)
                except json.JSONDecodeError:
                    # If parsing fails, treat it as a single description
                    descriptions = [descriptions]

            # If descriptions is a dictionary, extract the values
            if isinstance(descriptions, dict):
                descriptions = list(descriptions.values())

            # Use the first description if multiple exist
            descriptions = descriptions if isinstance(descriptions, list) else [descriptions]

            if not entry.get('short_description') and descriptions:
                first_description = descriptions[0] if isinstance(descriptions, list) and descriptions else None
                if first_description and isinstance(first_description, str):
                    entry['short_description'] = first_description

            result.append({
                "image": image_data,
                "title": entry.get('value', 'Unknown'),
                "artists": artist_names,
                "rights": entry.get('rights', 'Unknown'),
                "descriptions": descriptions,
                "image_urls": image_urls,
                "short_description": entry.get('short_description', 'No short description available'),
                "related_keywords": entry.get('relatedKeywordStrings', '').split(",") if entry.get('relatedKeywordStrings') else []
            })

    return json.dumps(result)

# -- helper functions -- #

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

def extract_features(img):
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


# function for getting the matched enty matched_entry = next((entry for entry in dataset if entry["filename"] == row.filename), None)
def find_matching_entry(filename, conn):
    """
    Given a filename and the database connection, find the corresponding entry in the database.
    """
    #print("looking for an entry with filename: ", filename)
    #matched_entry = next((entry for entry in dataset if str(entry["filename"]) == str(filename)), None)
    #return matched_entry

    query = "SELECT * FROM image_entries WHERE filename = ?"
    cursor = conn.execute(query, (filename,))
    matched_entry = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    if not matched_entry.empty:
        return matched_entry.iloc[0].to_dict()
    return None




# --- helper functions for grabbing image urls --- #


def clean(s):
    """
    Cleans a string by:
    - Replacing accented characters with hyphens
    - Removing punctuation (except hyphens)
    - Converting spaces and non-breaking spaces to hyphens
    - Lowercasing everything
    """
    s = s.strip().replace("\xa0", "-").lower() # Replace non-breaking spaces

    s = re.sub(r"[éèêëíîìáàâäãåūüùúóòôöõøñçśźżąęłńščřðþāēīōū&/`\'.:]", "-", s) #replace weird things with -
    s = re.sub(r"[^\w\s-]", "", s)  # Remove punctuation except hyphens
    s = s.replace(" ", "-").lower()  # Convert spaces to hyphens
    s = re.sub(r"-{2,}", "-", s) #remove double hyphens
    return s
    
#   deprecated! 
#  def try_wikiart_url(title, artist, date):
#     """
#     Given an artwork's title, artist, and date, generate the WikiArt image URL.
#     """
#     try:
#         formatted_artist = clean(artist)
#         formatted_title = clean(title)
        
#         if date == 'None':
#             formatted_date = ''
#         else:
#             formatted_date = str(int(re.sub(r'\D', '', date)))[:4]  # Ensure date is an integer

#         base_url = f"https://uploads3.wikiart.org/images/{formatted_artist}/{formatted_title}"
#         possible_urls = [
#             f"{base_url}-{formatted_date}.jpg",
#             f"{base_url}.jpg",
#             f"{base_url}-{formatted_date}(1).jpg"
#         ]

#         for j in range(1, 4):
#             possible_urls.append(f"{base_url}-{formatted_date}({j}).jpg")
#             possible_urls.append(f"{base_url}({j}).jpg")
#             possible_urls.append(f"{base_url}-0{j}.jpg")
#             possible_urls.append(f"{base_url}-0{j}-{formatted_date}.jpg")

#         for url in possible_urls:
#             if check_image_url(url):
#                 return url

#         print(f"broken: {base_url}")
#         return None
#     except Exception as e:
#         print(f"Error generating URL: {e}")
#         return None  # Return None if there's an issue
