# index.py
# run me with './bootstrap.sh' in terminal
import json
from flask import Flask, jsonify, request, g
from difflib import SequenceMatcher
from admin import admin_bp




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

import helperfunctions as helpers # helper functions including preprocess_text

import requests, re, os, ast
import pandas as pd

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

# The database paths inside the container will always be:
MODEL_CACHE_DIR = "/root/.cache/torch/hub"

# Get the absolute path to the /app directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This gets "/app"

# Define paths relative to the correct base directory
DB_PATH = os.path.join(BASE_DIR, "LOCALDB", "knowledgebase.db")
IMAGES_PATH = os.path.join(BASE_DIR, "LOCALDB", "images")

# Debugging: Print paths
print(f"✅ Using text and image database file: {DB_PATH}")
print(f"✅ Using Images Path: {IMAGES_PATH}")
# Check if files exist
if not os.path.exists(DB_PATH):
    print(f"🚨 ERROR: DB not found at {DB_PATH}")

if not os.path.exists(IMAGES_PATH):
    print(f"🚨 ERROR: Images directory not found at {IMAGES_PATH}")

# Test call to the text database
try:
    with sqlite3.connect(DB_PATH) as text_db:
        cursor = text_db.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if tables:
            print(f"✅ DB is valid. Tables: {tables}")
        else:
            print(f"🚨 ERROR: Text DB at {DB_PATH} is empty or invalid.")
except sqlite3.Error as e:
    print(f"🚨 ERROR: Failed to connect to Text DB at {DB_PATH}. Error: {e}")


# # Will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading ResNet50 from {MODEL_CACHE_DIR}...")
# Load ResNet50 weights and remove the last classification layer
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
# Move model to correct device
model.to(device)
model.eval()  # Set model to evaluation mode


# def load_sqlite_vec(db):
#     db.enable_load_extension(True)
#     sqlite_vec.load(db)
#     db.enable_load_extension(False)
#     print("loaded sqlite vector extension...")
#     return

def get_db():
    """Get a database connection for the current request.
    
    This creates a new connection if one doesn't exist for the current request,
    and reuses it for all subsequent calls during the same request.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        
        # Load sqlite-vec extension once per connection
        g.db.enable_load_extension(True)
        sqlite_vec.load(g.db)
        g.db.enable_load_extension(False)
        
    return g.db

print("Done! Time to run the app...")

app = Flask(__name__)


# Register the admin blueprint
app.register_blueprint(admin_bp)

@app.route("/")
def hello_world():
    print("User connected...")
    
    results = {}
    errors = {}
    
    # Test 1: Keyword Check
    try:
        test_text = "dog and cats eating a sandwich abstract-expressionism art nouveau abstract expressionistic portraiture michelangelo"
        keyword_results = keyword_check(test_text, threshold=0.3)
        results['keyword_check'] = {
            'input': test_text,
            'words': keyword_results
        }
    except Exception as e:
        errors['keyword_check'] = str(e)
    
    # Test 2: Text Lookup
    try:
        # Call the text lookup handler directly
        with app.test_request_context(json={'query': 'dogs', 'top_k': 5}):
            text_response = handle_text()
            results['text_lookup'] = {
                'query': 'dogs',
                'results': text_response.get_json()
            }
    except Exception as e:
        errors['text_lookup'] = str(e)
    
    # Test 3: Image Lookup
    try:
        test_image_url = "https://d32dm0rphc51dk.cloudfront.net/gTPexURCjkBek6MrG7g1bg/small.jpg"
        with app.test_request_context(json={'image': test_image_url}):
            image_response = handle_image()
            results['image_lookup'] = {
                'query_image': test_image_url,
                'results': image_response.get_json()
            }
    except Exception as e:
        errors['image_lookup'] = str(e)
    
    # Test 4: Database connectivity check
    try:
        db = get_db()
        
        # Check text entries
        text_cursor = db.execute("SELECT COUNT(*) as count FROM text_entries")
        text_count = text_cursor.fetchone()['count']
        
        # Check image entries
        image_cursor = db.execute("SELECT COUNT(*) as count FROM image_entries")  # or image_entries
        image_count = image_cursor.fetchone()['count']
        
        results['database_stats'] = {
            'text_entries': text_count,
            'image_entries': image_count
        }
    except Exception as e:
        errors['database'] = str(e)
    
    # Format the output
    output = "ML/Data Server Status Check<br><br>"
    
    if results.get('database_stats'):
        output += f"<b>Database Status:</b><br>"
        output += f"- Text Entries: {results['database_stats']['text_entries']}<br>"
        output += f"- Image Entries: {results['database_stats']['image_entries']}<br><br>"
    
    if results.get('keyword_check'):
        output += f"<b>Keyword Check Test:</b><br>"
        output += f"<br><b>Full JSON Response:</b><br><pre>{json.dumps(results['keyword_check']['words'], indent=2)}</pre><br>"
        output += f"Input: '{results['keyword_check']['input']}'<br>"
        output += "Results:<br>"
        for word in results['keyword_check']['words']:  
            if 'details' in word:
                output += f"- {word['value']} → {word['details']['databaseValue']}<br>"
            else:
                output += f"- {word['value']}<br>"
        output += "<br>"
    
    if results.get('text_lookup'):
        output += f"<b>Text Lookup Test:</b><br>"
        output += f"<br><b>Full JSON Response:</b><br><pre>{json.dumps(results['text_lookup']['results'], indent=2)}</pre><br>"
        output += f"Query: 'dogs'<br>"
        output += "Top matches:<br>"
        for match in results['text_lookup']['results']: 
            output += f"- {match.get('value', 'N/A')} (distance: {match.get('distance', 'N/A'):.3f})<br>"
        output += "<br>"
    
    if results.get('image_lookup'):
        # Print the entire JSON response for image lookup
        output += f"<br><b>Full JSON Response:</b><br><pre>{json.dumps(results['image_lookup']['results'], indent=2)}</pre><br>"
        output += f"<b>Image Lookup Test:</b><br>"
        output += f"<img src='{results['image_lookup']['query_image']}' style='max-width:200px;'><br>"
        output += "Similar images:<br>"
        
        for img in results['image_lookup']['results'][:3]:  # Show first 3
            img_url = img.get('image_url', 'image url failed')
            output += f"<img src='{img_url}' style='max-width:200px;'><br>"
            title = img.get('value', 'N/A')
            artists = ', '.join(img.get('artist_names', []))
            output += f"- {title} by {artists}<br>"
        output += "<br>"
    
    if errors:
        output += "<b>Errors:</b><br>"
        for component, error in errors.items():
            output += f"- {component}: {error}<br>"
    
    output += "<br>All systems operational!" if not errors else "<br>Some systems need attention."
    
    return f"<pre>{output}</pre>"


#--------------------------- TEXT HANDLING --------------------------#
@app.route('/keyword_check', methods=['POST'])
def handle_keyword_check():
    """
    Handles a keyword check request by processing input text and threshold from the JSON payload.
    JSON Request Structure:
    - 'text' (str): The input text to analyze.
    - 'threshold' (float, optional): The similarity threshold for keyword matching (default is 0.3).
    JSON Response Structure:
    {
        "words": [
            {
                "value": "<word>" # for normal words
            },
            {
                "value": "<phrase>", # for words/phrases with matched keywords
                "details": {
                    "entry_id": <int>,  # Unique ID of the keyword in the database
                    "value": "<str>",  # The keyword or phrase
                    "images": [<str>, ...],  # List of associated image URLs
                    "isArtist": <int>,  # 1 if the keyword is an artist, 0 otherwise
                    "type": "<str>",  # Type of the keyword (e.g., "artist", "movement", etc.)
                    "artist_aliases": [<str>, ...],  # List of aliases iff the keyword is an artist
                    "descriptions": {<str>: {<str>: <str>, ...}},  # each description comes from a source and contains its own dictionary, like {source: {description: "", otherinfo: "",...}, source2: {...}}
                    "relatedKeywordIds": [<int>, ...],  # List of related keyword IDs
                    "relatedKeywordStrings": [<str>, ...]  # List of related keyword strings
                }
            },
            ...
        ] # where the value of each word reconstructs the original input text.
    }
    EXAMPLE:
    [
        {
            "value": "cats"
        },
        {
            "value": "eating",
            "details": {
            "entry_id": "4de7d8bf91b76c000100b370",
            "value": "Food",
            "images": [],
            "isArtist": 0,
            "type": "Subject Matter",
            "artist_aliases": [],
            "descriptions": {
                "artsy": "_\"Tell me what you eat and I will tell you what you are.\" \u2014Jean Anthelme Brillat-Savarin_\n\nFood has long been a favored subject of artists, and at times even a medium for making art. In Western Art, depictions of food date back to funerary paintings of food offerings in ancient Egypt. The Classical historian Pliny claimed that Greek painter Zeuxis once painted grapes so realistic that birds came to pick at them. Depictions continued in [Roman art](/gene/roman-art), where the putto (male infant) depicted with grape vines was a common motif. In [Baroque](/gene/baroque) painting, food appeared regularly as a still-life element, as exemplified by [Carravaggio](/artist/michelangelo-merisi-da-caravaggio)'s _Bacchus_ or the _bodegones_ (meaning 'pantry still-lives') of Spanish painters [Diego Vel\u00e1zquez](/artist/diego-velazquez) and [Francisco de Zurbar\u00e1n](/artist/francisco-de-zurbaran). [Paul C\u00e9zanne](/artist/paul-cezanne)'s fruit still lifes presented new forms of representing three-dimensional space. In the 20th century, food was central to [Pop Art](/gene/pop-art)'s explorations of consumerism, as in [Andy Warhol](/artist/andy-warhol)'s Campell's soup cans, [Claes Oldenburg](/artist/claes-oldenburg)'s monumental hamburger and ice cream cone sculptures, and [Wayne Thiebaud](/artist/wayne-thiebaud)'s paintings of cakes and pastries. At its most logical extreme, food has been used as an actual medium for creating artworks, as in [Dieter Roth](/artist/dieter-roth)'s chocolate self-portraits and [Vik Muniz](/artist/vik-muniz)'s reproductions of iconic works of art using materials such as lunch meat and peanut butter."
            },
            "relatedKeywordIds": [
                "4d8b93b04eb68a1b2c001b9d"
            ],
            "relatedKeywordStrings": []
            }
        },
        {
            "value": "a"
        },
        {
            "value": "sandwich"
        },
        {
            "value": "abstract-expressionism",
            "details": {
            "entry_id": "52277c7debad644d2800051f",
            "value": "Abstract Expressionism",
            "images": [],
            "isArtist": 0,
            "type": "Styles and Movements",
            "artist_aliases": [],
            "descriptions": [
                {"artsy": {"date": "1800", "description": "_\u201cIt seems to me that the modern painter cannot express this age, the airplane, the atom bomb, the radio, in the old forms of the [Renaissance](/gene/renaissance) or of any other past culture."}
                },
                ...
            ],
            "relatedKeywordIds": [],
            "relatedKeywordStrings": []
            }
        },
    ]
    """

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
    Identifies semantically similar keywords from a database based on the input text and a similarity threshold.
    """
    print("Received request to check an input text for keywords...")
    # Tokenize original input text while keeping stopwords
    original_words = input_text.split()  # Preserves all words

    # Step 1: Preprocess text to get candidate phrases (unigrams, bigrams, and trigrams) with positions
    candidate_phrases = helpers.preprocess_text(input_text)  # Returns (phrase, start_idx, end_idx)

    # Step 2: Find semantically similar matches
    db = get_db()
    matches = helpers.find_semantic_keyword_matches(candidate_phrases, db, threshold)
    matches_df = pd.DataFrame(matches)
    print("Semantic Matches:\n", matches_df)

    # Step 3: Retrieve keyword details from `keywords` table
    matched_ids = [match["id"] for match in matches]
    keyword_details = {}

    if matched_ids:
        query = f"SELECT * FROM text_entries WHERE entry_id IN ({','.join(['?'] * len(matched_ids))})"
        cursor = db.execute(query, matched_ids)
        keyword_details = {row["entry_id"]: dict(row) for row in cursor.fetchall()}

    # Step 4: Find optimal matches using dynamic programming
    n = len(original_words)
    
    # Create a list of all valid matches with their scores
    valid_matches = []
    for phrase, start_idx, end_idx in candidate_phrases:
        match = next((m for m in matches if m["phrase"] == phrase), None)
        if match and match["id"] in keyword_details:
            valid_matches.append({
                'phrase': phrase,
                'start': start_idx,
                'end': end_idx,
                'score': match.get('similarity', 0),  # Use 'similarity' from your matches
                'id': match['id'],
                'details': keyword_details[match['id']]
            })
    
    # Sort matches by start position for easier processing
    valid_matches.sort(key=lambda x: x['start'])
    
    # Dynamic programming: dp[i] = (best_score, matches_used) for words 0 to i-1
    dp = [(0, [])] * (n + 1)
    
    for i in range(1, n + 1):
        # Option 1: Don't use any match ending at position i-1
        dp[i] = dp[i-1]
        
        # Option 2: Use a match ending at position i-1
        for match in valid_matches:
            if match['end'] == i - 1:  # Match ends at position i-1
                score = dp[match['start']][0] + match['score']
                if score > dp[i][0]:
                    dp[i] = (score, dp[match['start']][1] + [match])
    
    # Get the optimal set of matches
    _, optimal_matches = dp[n]
    
    # Build final results using the optimal matches
    final_results = []
    position = 0
    
    while position < len(original_words):
        # Check if this position starts an optimal match
        matching = None
        for match in optimal_matches:
            if match['start'] == position:
                matching = match
                break
        
        if matching:
            # Use the matched phrase
            original_phrase = ' '.join(original_words[matching['start']:matching['end'] + 1])
            db_row = matching['details']
            
            result_entry = {
                "value": original_phrase,
                "details": {
                    "entry_id": db_row["entry_id"],
                    "databaseValue": db_row["value"],
                    "images": helpers.safe_json_loads(db_row.get("images", "[]"), default=[]),
                    "isArtist": db_row.get("isArtist", 0),
                    "type": db_row.get("type"),
                    "artist_aliases": helpers.safe_json_loads(db_row.get("artist_aliases", "[]"), default=[]) 
                        if db_row.get("isArtist") == 1 else [],
                    "descriptions": helpers.safe_json_loads(db_row.get("descriptions", "{}"), default={}),
                    "relatedKeywordIds": helpers.safe_json_loads(db_row.get("relatedKeywordIds", "[]"), default=[]),
                    "relatedKeywordStrings": helpers.safe_json_loads(db_row.get("relatedKeywordStrings", "[]"), default=[])
                }
            }
            
            try:
                # Validate JSON serialization
                json.dumps(result_entry)
                final_results.append(result_entry)
                position = matching['end'] + 1
            except (TypeError, ValueError) as e:
                # If parsing fails, append the word as is
                final_results.append({"value": original_words[position]})
                position += 1
        else:
            # No match at this position
            final_results.append({"value": original_words[position]})
            position += 1
    
    print(f"Final results: {len(final_results)} entries")
    for result in final_results:
        if "details" in result:
            print(f"Matched: '{result['value']}' -> '{result['details']['databaseValue']}' (type: {result['details']['type']})")
        else:
            print(f"Word: '{result['value']}'")

    return final_results



@app.route('/lookup_text', methods=['POST'])
def handle_text(): 
    """
    Given a text query, find and return the most similar text entries in the database.
    
    Expected request JSON:
    {
        "query": "search text",
        "top_k": 5  (optional, defaults to 5)
    }
    
    Returns JSON array of matches with distance scores and full database details.
    Example:
    [
        {
            "artist_aliases": [],
            "descriptions": {
            "artsy": "_\u201cAnimals are such agreeable friends\u2014they ask no questions, they pass no criticisms.\u201d \u2014George Eliot_\n\nWhether pets, mythological beasts, or wild creatures, animals have always been a major subject of art and literature. Dating back to Paleolithic cave paintings in France and ancient Egyptian reliefs and artifacts, animals have been depicted by artists as friends, allegories, muses, and reflections on human nature."
            },
            "distance": 1.1766963005065918,
            "entry_id": "4d937b5517cb1325370000ee",
            "images": [],
            "isArtist": 0,
            "relatedKeywordIds": [],
            "relatedKeywordStrings": [],
            "type": "Subject Matter",
            "value": "Animals"
        },
        ...
    ]
    """
    print("Received request for text handling...")
    
    # ---- PROCESS THE REQUEST ---- #
    query_text = request.json.get('query')
    if not query_text:
        return jsonify({"error": "No query text provided"}), 400
        
    top_k = request.json.get('top_k', 5)
    print(f"Query text: {query_text}")
    print(f"Top K: {top_k}")

    # Extract features from query text
    query_features = helpers.extract_text_features(query_text)
    print(f"Query features shape: {query_features.shape}")
    
    # ---- LOOK UP SIMILAR TEXTS ---- #
    db = get_db()

    # Find the most similar text entries
    similar_texts_df = helpers.find_most_similar_texts(query_features, db, top_k=top_k)
    print(f"Found {len(similar_texts_df)} similar texts")

    # Get detailed information for each match
    results = []
    
    for idx, row in similar_texts_df.iterrows():
        # Fetch the full record from the database
        query = "SELECT * FROM text_entries WHERE entry_id = ?"
        cursor = db.execute(query, [row['entry_id']])
        db_row = cursor.fetchone()
        
        if db_row:
            db_row_dict = dict(db_row)
            
            # Build the result with parsed JSON fields
            result_entry = {
                "entry_id": db_row_dict["entry_id"],
                "value": db_row_dict["value"],
                "distance": row['distance'],  # Add the similarity distance
                "images": helpers.safe_json_loads(db_row_dict.get("images", "[]"), default=[]),
                "isArtist": db_row_dict.get("isArtist", 0),
                "type": db_row_dict.get("type"),
                "artist_aliases": helpers.safe_json_loads(db_row_dict.get("artist_aliases", "[]"), default=[]) 
                    if db_row_dict.get("isArtist") == 1 else [],
                "descriptions": helpers.safe_json_loads(db_row_dict.get("descriptions", "{}"), default={}),
                "relatedKeywordIds": helpers.safe_json_loads(db_row_dict.get("relatedKeywordIds", "[]"), default=[]),
                "relatedKeywordStrings": helpers.safe_json_loads(db_row_dict.get("relatedKeywordStrings", "[]"), default=[])
            }
            
            results.append(result_entry)
            print(f"Match: '{result_entry['value']}' (distance: {result_entry['distance']:.4f})")
    
    print(f"Returning {len(results)} matches for query: '{query_text}'")
    return jsonify(results)




#--------------------------- IMAGE HANDLING --------------------------#
@app.route('/image', methods=['POST'])
def handle_image():
    """
    Find similar images based on a query image.
    
    Expected request JSON:
    {
        "image": "url or base64 string",
        "top_k": 3  (optional, defaults to 3)
    }
    
    Returns JSON array of matches with distance scores and full database details.
    [
        {
            "image_id": <int>,  # Unique identifier for the image in the database
            "value": <str>,  # Title of the image
            "distance": <float>,  # Cosine similarity distance between query and database image
            "image_url": <str>,  # URL or base64-encoded string of the matched image
            "artist_names": [<str>, ...],  # List of artist names associated with the image
            "image_urls": {  # Dictionary of image URLs by size
                "large": <str>,
                "medium": <str>,
                "small": <str>,
                ...
            },
            "filename": <str>,  # Filename of the image in the database
            "rights": <str>,  # Rights or copyright information for the image
            "descriptions": {  # Dictionary of descriptions for the image
                <str>: <str>,  # Key-value pairs of description types and their content
                ...
            },
            "relatedKeywordIds": [<int>, ...],  # List of related keyword IDs
            "relatedKeywordStrings": [<str>, ...]  # List of related keyword strings
        },
        ...
    ]
    """
    print("Received request for image handling...")
    
    # ---- PROCESS THE REQUEST ---- #

    
    # Get image from request
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400
        
    # Load image from URL or base64
    if helpers.check_image_url(request.json['image']):
        print("✅")
        img = url_to_image(request.json['image'])
    else:
        img = base64_to_image(request.json['image'])
    
    if img is None:
        return jsonify({"error": "Failed to load image"}), 400
    
    # Get top_k parameter (how many matches to return)
    top_k = request.json.get('top_k', 3)
    
    # Extract features from the posted image
    query_features = extract_features(img)
    print(f"Query features shape: {query_features.shape}")

    # ---- LOOK UP SIMILAR IMAGES ---- #

    # Get database connection
    db = get_db()

    # Find the most similar images
    similar_images = find_most_similar_images(query_features, db, top_k=top_k)
    print(f"Found {len(similar_images)} similar images")
    
    # Get detailed information for each match
    results = []
    
    for match in similar_images:
        # Fetch the full record from the database
        query = "SELECT * FROM image_entries WHERE image_id = ?"
        cursor = db.execute(query, [match["image_id"]])
        db_row = cursor.fetchone()
        
        if db_row:
            db_row_dict = dict(db_row)
            
            # Parse image_urls to find a valid image URL
            image_urls = helpers.safe_json_loads(db_row_dict.get('image_urls', '{}'), default={})
            image_url = None
            
            # Try to get the first valid image URL
            for size in ['large', 'medium', 'larger', 'small', 'square', 'tall']:
                url = image_urls.get(size)
                if url and helpers.check_image_url(url):
                    print("✅")
                    image_url = url
                    break
            
            # Fallback to local file if no valid URL
            if not image_url and db_row_dict.get('filename'):
                try:
                    image_path = os.path.join(IMAGES_PATH, db_row_dict['filename'])
                    with open(image_path, "rb") as image_file:
                        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{image_base64}"
                except FileNotFoundError:
                    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a3/Image-not-found.png"
            
            # Build the result with parsed JSON fields
            result_entry = {
                "image_id": db_row_dict["image_id"],
                "value": db_row_dict.get("value", "Unknown"),
                "distance": match["distance"],
                "image_url": image_url,  # The resolved image URL
                "artist_names": db_row_dict.get("artist_names", "").split(", ") if db_row_dict.get("artist_names") else [],
                "image_urls": image_urls,  # The full dictionary of URLs
                "filename": db_row_dict.get("filename"),
                "rights": db_row_dict.get("rights", "Unknown"),
                "descriptions": helpers.safe_json_loads(db_row_dict.get("descriptions", "{}"), default={}),
                "relatedKeywordIds": helpers.safe_json_loads(db_row_dict.get("relatedKeywordIds", "[]"), default=[]),
                "relatedKeywordStrings": helpers.safe_json_loads(db_row_dict.get("relatedKeywordStrings", "[]"), default=[])
            }
            
            results.append(result_entry)
            print(f"Match: '{result_entry['value']}' (distance: {result_entry['distance']:.4f})")
    
    print(f"Returning {len(results)} matches")
    return jsonify(results)


def find_most_similar_images(image_features, conn, top_k=3):
    """
    Find the top-k most similar images by cosine similarity.
    
    Args:
        image_features: Feature vector from the query image
        conn: Database connection
        top_k: Number of results to return
        
    Returns:
        List of dicts with image_id and distance
    """
    print("Finding similar images...", end=' ')

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

    return similar_images

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



@app.teardown_appcontext
def close_db(error):
    """Close the database connection at the end of the request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


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
