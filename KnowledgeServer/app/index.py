# index.py
# run me with './bootstrap.sh' in terminal
import json
from flask import Flask, jsonify, request, g, render_template
# 
import requests

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


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

import re, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd

# Replace the path definitions with:
from config import BASE_DIR, DB_PATH, IMAGES_PATH, MODEL_CACHE_DIR, TRANSFORMERS_CACHE_DIR

# Remove the old definitions and use the imported ones

#todo, incorporate exact matches into keyword checking too


print("Mabuhay! Loading...")


# Debugging: Print paths, which are imported from config.py

print(f"✅ Running in Docker: {os.environ.get('RUNNING_IN_DOCKER', 'false').lower() == 'true'}")
print(f"✅ Base directory: {BASE_DIR}")
print(f"✅ Model cache: {MODEL_CACHE_DIR}")
print(f"✅ Transformers cache: {TRANSFORMERS_CACHE_DIR}")
print(f"✅ Using database: {DB_PATH}")
print(f"✅ Using images: {IMAGES_PATH}")

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



# Initialize jobs database on startup
from jobs import init_jobs_db
init_jobs_db()
print("Initialized jobs database.")

print("Done! Time to run the app...")


app = Flask(__name__, static_folder='static')


# Add API routes for artwork similarity space mapping functionalities
from templates.map_api import map_api_bp
app.register_blueprint(map_api_bp)
from templates.hierarchical_map_api import hierarchical_map_api_bp
app.register_blueprint(hierarchical_map_api_bp)
# Register blueprints for other pages
from templates.health_check import health_check_bp
app.register_blueprint(health_check_bp)

from templates.staging_review import staging_review_bp
app.register_blueprint(staging_review_bp)

from templates.data_cleaner import data_cleaner_bp
app.register_blueprint(data_cleaner_bp)

from templates.database_requests import database_requests_bp
app.register_blueprint(database_requests_bp)

from templates.map_api_v3 import map_api_v3_bp
app.register_blueprint(map_api_v3_bp)

@app.route("/")
def browse_database():
    return render_template('database_browser.html')

@app.route("/api/browse_database")
def api_browse_database():
    """API endpoint for database browsing with pagination and sorting"""
    try:
        # Get query parameters
        table = request.args.get('table', 'text_entries')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 25))
        sort_by = request.args.get('sort_by', None)
        sort_dir = request.args.get('sort_dir', 'asc')
        
        # Validate table name
        if table not in ['text_entries', 'image_entries']:
            return jsonify({
                'success': False,
                'error': 'Invalid table name'
            })
        
        # Validate sort direction
        if sort_dir not in ['asc', 'desc']:
            sort_dir = 'asc'
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get database connection
        db = get_db()
        
        # Get total count
        count_cursor = db.execute(f"SELECT COUNT(*) as count FROM {table}")
        total_rows = count_cursor.fetchone()['count']
        
        # Build ORDER BY clause
        if sort_by:
            # Validate sort column to prevent SQL injection
            valid_columns = {
                'text_entries': ['entry_id', 'value', 'type', 'isArtist'],
                'image_entries': ['image_id', 'value', 'filename']
            }
            
            if sort_by in valid_columns.get(table, []):
                order_clause = f"ORDER BY {sort_by} {sort_dir.upper()}"
            else:
                order_clause = f"ORDER BY {'entry_id' if table == 'text_entries' else 'image_id'} ASC"
        else:
            order_clause = f"ORDER BY {'entry_id' if table == 'text_entries' else 'image_id'} ASC"
        
        # Get paginated data
        if table == 'text_entries':
            query = f"""
                SELECT entry_id, value, images, isArtist, type, 
                       artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings
                FROM text_entries
                {order_clause}
                LIMIT ? OFFSET ?
            """
        else:
            query = f"""
                SELECT image_id, value, artist_names, image_urls, filename,
                       rights, descriptions, relatedKeywordIds, relatedKeywordStrings
                FROM image_entries
                {order_clause}
                LIMIT ? OFFSET ?
            """
        
        cursor = db.execute(query, (page_size, offset))
        
        # Convert rows to list of dicts
        rows = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            rows.append(row_dict)
        
        return jsonify({
            'success': True,
            'table': table,
            'page': page,
            'page_size': page_size,
            'total_rows': total_rows,
            'rows': rows
        })
        
    except Exception as e:
        print(f"ERROR in api_browse_database: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route("/status")
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
            text_response = handle_lookup_text()
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
            image_response = handle_lookup_image()
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
def handle_lookup_text():
    """
    Handles a text lookup request by processing input JSON and calling the lookup_text function.
    
    Expected request JSON:
    {
        "query": "search text",
        "top_k": 5  (optional, defaults to 5),
        "search_in": "description" | "value" | "both" (optional, defaults to "description")
    }
    
    Returns JSON array of matches with distance scores and full database details.
    """
    print("Received request for text handling...")
    
    # ---- PROCESS THE REQUEST ---- #
    query_text = request.json.get('query')
    if not query_text:
        return jsonify({"error": "No query text provided"}), 400
        
    top_k = request.json.get('top_k', 5)
    search_in = request.json.get('search_in', 'description')
    print(f"Query text: {query_text}")
    print(f"Top K: {top_k}")
    print(f"Search in: {search_in}")

    # Call the general lookup_text function
    try:
        results = lookup_text(query_text, top_k, search_in)
        return jsonify(results)
    except Exception as e:
        print(f"Error during text lookup: {e}")
        return jsonify({"error": str(e)}), 500


def lookup_text(query_text, top_k=5, search_in='description'):
    """
    Given a text query, find and return the most similar text entries in the database.
    
    Args:
        query_text (str): The text to search for.
        top_k (int): Number of top matches to return.
        search_in (str): Field(s) to search in: "description", "value", or "both".
    
    Returns:
        List of dictionaries containing matches with distance scores and full database details.
    """
    print(f"Performing text lookup for query: '{query_text}' with top_k={top_k}, search_in={search_in}")
    
    # Extract features from query text
    query_features = helpers.extract_text_features(query_text)
    print(f"Query features shape: {query_features.shape}")
    
    # ---- LOOK UP SIMILAR TEXTS ---- #
    db = get_db()

    # Find the most similar text entries
    similar_texts_df = helpers.find_most_similar_texts(query_features, db, top_k=top_k, search_in=search_in)
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
    return results
    



#--------------------------- IMAGE HANDLING --------------------------#
@app.route('/image', methods=['POST'])
def handle_lookup_image():
    """
    Handles a request to find similar images based on a query image.

    Expected request JSON:
    {
        "image": "url or base64 string",
        "top_k": 3  (optional, defaults to 3)
    }

    Returns JSON array of matches with distance scores and full database details.
    """
    print("Received request for image handling...")

    # SOMETHING IS WRONG STARTING HERE....
    # Validate the request
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400

    # Load image from URL or base64
    if helpers.check_image_url(request.json['image']):
        img = helpers.url_to_image(request.json['image'])
    else:
        img = helpers.base64_to_image(request.json['image'])

    if img is None:
        return jsonify({"error": "Failed to load image"}), 400

    # Get top_k parameter (how many matches to return)
    top_k = request.json.get('top_k', 3)


    # .. TO HERE (basd on the comments tht actually get printed)
    print("sending to lookup_image function: " + str(img) + " with top_k=" + str(top_k))

    # Call the general lookup_image function
    try:
        results = lookup_image(img, top_k)
        return jsonify(results)
    except Exception as e:
        print(f"Error during image lookup: {e}")
        return jsonify({"error": str(e)}), 500


def lookup_image(img, top_k=3):
    """
    Finds similar images based on the provided image.

    Args:
        img: Preprocessed image object (e.g., PIL Image).
        top_k: Number of top matches to return.

    Returns:
        List of dictionaries containing matches with distance scores and full database details.
    """
    print("Performing image lookup...")

    # Extract features from the image
    query_features = helpers.extract_img_features(img)
    print(f"Query features shape: {query_features.shape}")

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
    return results

@app.route('/get_image_features', methods=['POST'])
def get_image_features():
    """
    Extract features from an image provided in the request.
    
    Expected request JSON:
    {
        "image": "url or base64 string"
    }
    
    Returns JSON with extracted features.
    Example:
    {
        "features": [0.123, 0.456, ...]
    }
    """
    print("Received request to extract image features...")
    
    # Validate the request
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400
    
    # Load image from URL or base64
    image_data = request.json['image']
    if helpers.check_image_url(image_data):
        img = helpers.url_to_image(image_data)
    else:
        img = helpers.base64_to_image(image_data)
    
    if img is None:
        return jsonify({"error": "Failed to load image"}), 400
    
    # Extract features using the helper function
    try:
        features = helpers.extract_img_features(img)
        print(f"Extracted features: {features.shape}")
        return jsonify({"features": features.tolist()})
    except Exception as e:
        print(f"Error extracting features: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_text_features', methods=['POST'])
def get_text_features():
    """
    Extract features from text provided in the request.

    Expected request JSON:
    {
        "text": "sample text",
        "dimensions": 2  # Optional, if provided, returns features in 2D using t-SNE
    }

    Returns JSON with extracted features.
    
    """
    print("Received request to extract text features...")

    # Validate the request
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400

    # Extract text from the request
    input_text = request.json['text']
    dimensions = request.json.get('dimensions', None)

    # Extract features using the helper function
    try:
        features = helpers.extract_text_features(input_text)
        print(f"Extracted features: {features.shape}")

        # TODO: If dimensions=2 is requested, reduce to 2D
        # if dimensions == 2:
        #     continue
            # features_2d = helpers.tsne_similarity_flatten(features, num_dims=2)
            # print(f"Reduced features to 2D: {features_2d.shape}")
            # return jsonify({"features": features_2d.tolist()})

        return jsonify({"features": features.tolist()})
    except Exception as e:
        print(f"Error extracting features: {e}")
        return jsonify({"error": str(e)}), 500




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

@app.route('/lookup_entry', methods=['POST'])
def lookup_entry():
    """
    General lookup API.
    Expects JSON:
    {
        "entryId": "<id>",
        "type": "text" | "image"
    }
    Returns the matching row from text_entries or image_entries, or 404 if not found.
    """
    data = request.get_json()
    entry_id = data.get('entryId')
    entry_type = data.get('type')

    if not entry_id or entry_type not in ('text', 'image'):
        return jsonify({"error": "Missing or invalid entryId/type"}), 400

    db = get_db()
    row_dict = helpers.retrieve_by_id(entry_id, db, entry_type=entry_type)
    if not row_dict:
        return jsonify({"error": "Entry not found"}), 404

    return jsonify(row_dict)


@app.route('/validate_admin_password', methods=['POST'])
def validate_admin_password():
    """
    Validates the admin password sent in the request JSON.
    Expects JSON: { "password": "<password>" }
    Returns: { "success": true } if correct, else { "success": false }
    """
    data = request.get_json()
    password = data.get('password') if data else None
    admin_password = os.environ.get('FINAL_SQL_ADMIN_PASSWORD')

    if password and admin_password and password == admin_password:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False})
    

# --- ADMIN CLEANUP ROUTE --- #
import glob
from jobs import cleanup_old_jobs

@app.route('/cleanup', methods=['POST'])
def cleanup_jobs_and_cache():
    """
    Admin endpoint to clean up old jobs and cache files.
    Deletes jobs with status completed/failed older than N days (default 7), or all if days_old=0.
    Removes all files in the generated_maps cache.
    Returns a summary of what was deleted.
    """
    # Only allow if admin password is provided (optional, for safety)
    data = request.get_json(silent=True) or {}
    admin_password = os.environ.get('FINAL_SQL_ADMIN_PASSWORD')
    if admin_password and data.get('password') != admin_password:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    # Get days_old parameter (default 7, 0 means all)
    days_old = data.get('days_old', 7)
    try:
        days_old = int(days_old)
    except Exception:
        days_old = 7

    # Clean up jobs
    try:
        cleanup_old_jobs(days_old=days_old)
        jobs_cleaned = True
        jobs_error = None
    except Exception as e:
        jobs_cleaned = False
        jobs_error = str(e)

    # Clean up cache files
    cache_dir = os.path.join(BASE_DIR, 'generated_maps')
    cache_files = glob.glob(os.path.join(cache_dir, '*.json'))
    deleted_files = []
    cache_error = None
    for f in cache_files:
        try:
            os.remove(f)
            deleted_files.append(os.path.basename(f))
        except Exception as e:
            cache_error = str(e)

    return jsonify({
        'success': jobs_cleaned and cache_error is None,
        'jobs_cleaned': jobs_cleaned,
        'jobs_error': jobs_error,
        'cache_files_deleted': deleted_files,
        'cache_error': cache_error
    })

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
    


# def get_coordinates_dict(method='umap'):
#     """
#     Retrieve coordinates as a dictionary for use in other scripts.
    
#     Args:
#         method (str): The method used to generate coordinates ('umap', 'pca', 'tsne')
        
#     Returns:
#         dict: {entry_id: {"x": 1.23, "y": 4.56}, ...}
#     """
#     import json
    
#     conn = sqlite3.connect('LOCALDB/knowledgebase.db')
#     cursor = conn.cursor()
    
#     try:
#         column_name = f"{method}_coords"
#         cursor.execute(f'''
#         SELECT entry_id, {column_name}
#         FROM text_coordinates 
#         WHERE {column_name} IS NOT NULL
#         ''')
        
#         results = cursor.fetchall()
#         coords_dict = {}
        
#         for entry_id, coord_json in results:
#             coords_dict[entry_id] = json.loads(coord_json)
        
#         logging.info(f"Retrieved {len(coords_dict)} coordinates for method '{method}'")
#         return coords_dict
        
#     finally:
#         conn.close()


# def get_all_coordinates_dict():
#     """
#     Retrieve all coordinates for all methods.
    
#     Returns:
#         dict: {entry_id: {"umap": {"x": 1.23, "y": 4.56}, "pca": {...}, "tsne": {...}}, ...}
#     """
#     import json
    
#     conn = sqlite3.connect('LOCALDB/knowledgebase.db')
#     cursor = conn.cursor()
    
#     try:
#         cursor.execute('''
#         SELECT entry_id, umap_coords, pca_coords, tsne_coords
#         FROM text_coordinates
#         ''')
        
#         results = cursor.fetchall()
#         coords_dict = {}
        
#         for entry_id, umap_json, pca_json, tsne_json in results:
#             coords_dict[entry_id] = {}
            
#             if umap_json:
#                 coords_dict[entry_id]['umap'] = json.loads(umap_json)
#             if pca_json:
#                 coords_dict[entry_id]['pca'] = json.loads(pca_json)
#             if tsne_json:
#                 coords_dict[entry_id]['tsne'] = json.loads(tsne_json)
        
#         logging.info(f"Retrieved coordinates for {len(coords_dict)} entries")
#         return coords_dict
        
#     finally:
#         conn.close()